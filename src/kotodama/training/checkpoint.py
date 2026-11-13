"""
Checkpoint management with SIGTERM handling for Scheduler compatibility.

Scheduler sends SIGTERM with a 2-second grace period before SIGKILL on cancel.
Our strategy: save periodically during training, and on SIGTERM set a flag
to exit cleanly at the next step boundary. We accept losing at most
``save_every_steps`` steps of work.

For timeout (estimated_minutes exceeded), Scheduler sends SIGTERM only
(no follow-up SIGKILL), so we have unlimited time to save.

Async mode (AsyncCheckpointManager):
  Synchronous CPU clone (~0.5s) + background torch.save to /dev/shm.
  Optional zstd compression in background. Backpressure: if queue > 2,
  skip compression and keep raw in SHM.
"""

from __future__ import annotations

import logging
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Optional

import torch
import torch.distributed as dist

from kotodama.training.fsdp import full_model_state_dict, is_fsdp2

logger = logging.getLogger(__name__)


_TOKEN_SUFFIX = {"K": 10**3, "M": 10**6, "B": 10**9, "T": 10**12}


def parse_token_count(text: str) -> int:
    """``"250M"`` -> 250_000_000; plain integers pass through."""
    s = text.strip().upper()
    if not s:
        raise ValueError("empty token count")
    mult = _TOKEN_SUFFIX.get(s[-1])
    num = s[:-1] if mult else s
    try:
        value = float(num) * (mult or 1)
    except ValueError as e:
        raise ValueError(f"bad token count {text!r}") from e
    if value <= 0:
        raise ValueError(f"token count must be positive: {text!r}")
    return int(round(value))


def parse_save_schedule(spec: str, total_tokens: int) -> list[int]:
    """Piecewise token cadence -> sorted save thresholds (tokens).

    ``"250M:4B,1B:10B,5B:end"`` = every 250M up to 4B, every 1B up to 10B,
    every 5B to the end of training. Each segment starts where the previous
    stopped; ``end`` = ``total_tokens``. Bounds are included when they land
    on the cadence.
    """
    thresholds: list[int] = []
    start = 0
    for seg in (s.strip() for s in spec.split(",") if s.strip()):
        try:
            every_s, until_s = seg.split(":")
        except ValueError as e:
            raise ValueError(f"save_schedule segment {seg!r} is not EVERY:UNTIL") from e
        every = parse_token_count(every_s)
        until = total_tokens if until_s.strip().lower() == "end" else parse_token_count(until_s)
        if until <= start:
            raise ValueError(f"save_schedule segment {seg!r} ends before it starts ({start})")
        t = start + every
        while t <= min(until, total_tokens):
            thresholds.append(t)
            t += every
        start = until
    return sorted(set(thresholds))


class TokenThresholds:
    """Fires once per crossed threshold set; resume-safe via ``skip_through``."""

    def __init__(self, thresholds: list[int]) -> None:
        self._t = sorted(thresholds)
        self._i = 0

    def skip_through(self, tokens: int) -> None:
        while self._i < len(self._t) and self._t[self._i] <= tokens:
            self._i += 1

    def due(self, tokens: int) -> bool:
        """True if ``tokens`` crossed >=1 pending threshold (all crossed ones consumed)."""
        if self._i < len(self._t) and self._t[self._i] <= tokens:
            self.skip_through(tokens)
            return True
        return False

    def __len__(self) -> int:
        return len(self._t)


def save_weights_only(
    directory: str | Path,
    step: int,
    model: torch.nn.Module,
    tokens_consumed: int,
    rank: int = 0,
) -> Optional[Path]:
    """Model-only checkpoint in the ``kotodama.ckpt`` format (``{"model", "step",
    "tokens_consumed"}``) — the formation-ladder cadence save. Never pruned.
    All ranks must call (barrier); rank 0 writes. A sharded (FSDP2) model is gathered first (collective) and
    written as bf16, the same file the unsharded path writes."""
    path: Optional[Path] = None
    raw = model.module if hasattr(model, "module") else model
    raw = getattr(raw, "_orig_mod", raw)
    full_sd: Optional[dict[str, torch.Tensor]] = None
    if is_fsdp2(raw):
        full_sd = full_model_state_dict(raw, dtype=torch.bfloat16)
    if rank == 0:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"step_{step:08d}.pt"
        t0 = time.time()
        tmp = path.with_suffix(".tmp")
        torch.save({"model": full_sd if full_sd is not None else raw.state_dict(), "step": step,
                    "tokens_consumed": tokens_consumed}, tmp)
        tmp.rename(path)
        logger.info("Weights saved: %s (%.1fs, %.2fB tokens)",
                    path, time.time() - t0, tokens_consumed / 1e9)
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()
    return path


class SIGTERMHandler:
    """
    Cooperative SIGTERM handler for Scheduler integration.

    Install once at the start of training.  Check ``handler.received``
    in the training loop to know when to checkpoint and exit.
    """

    def __init__(self) -> None:
        self.received: bool = False
        self._original_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum: int, frame: Any) -> None:
        self.received = True
        logger.warning(
            "SIGTERM received — will checkpoint and exit at next step boundary"
        )

    def restore(self) -> None:
        """Restore the original SIGTERM handler."""
        signal.signal(signal.SIGTERM, self._original_handler)


class CheckpointManager:
    """
    Save and load training checkpoints.

    Only rank 0 writes checkpoint files.  All ranks call :meth:`save`
    and synchronize via a barrier so that no rank races ahead while
    rank 0 is writing.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        rank: int = 0,
        keep_last_n: int = 5,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.rank = rank
        self.keep_last_n = keep_last_n

        if rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # -- Save / Load -----------------------------------------------------------

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        muon_opt: torch.optim.Optimizer,
        adamw_opt: torch.optim.Optimizer,
        scheduler: Any,
        tokens_consumed: int = 0,
        data_state: Optional[dict[str, Any]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> Optional[Path]:
        """
        Save a checkpoint.  Returns the path on rank 0, None on other ranks.
        """
        path: Optional[Path] = None

        if self.rank == 0:
            path = self.checkpoint_dir / f"step_{step:08d}.pt"

            # Unwrap DDP and torch.compile to get clean state dict
            raw_model = model.module if hasattr(model, "module") else model
            if hasattr(raw_model, "_orig_mod"):
                raw_model = raw_model._orig_mod

            state: dict[str, Any] = {
                "step": step,
                "tokens_consumed": tokens_consumed,
                "model": raw_model.state_dict(),
                "muon_optimizer": muon_opt.state_dict(),
                "adamw_optimizer": adamw_opt.state_dict(),
                "scheduler": _scheduler_state(scheduler),
            }

            if data_state is not None:
                state["data"] = data_state
            if extra is not None:
                state.update(extra)

            t0 = time.time()
            # Save to temp file first, then rename for atomicity
            tmp_path = path.with_suffix(".tmp")
            torch.save(state, tmp_path)
            tmp_path.rename(path)
            elapsed = time.time() - t0

            logger.info("Checkpoint saved: %s (%.1fs)", path, elapsed)
            self._cleanup_old(keep=self.keep_last_n)

        # Synchronize so no rank races ahead during save
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()

        return path

    def load_latest(
        self,
        model: torch.nn.Module,
        muon_opt: torch.optim.Optimizer,
        adamw_opt: torch.optim.Optimizer,
        device: torch.device,
    ) -> Optional[dict[str, Any]]:
        """
        Load the most recent checkpoint.  Returns the full state dict
        (including ``step``, ``tokens_consumed``, ``data``) or None if
        no checkpoint exists.
        """
        checkpoints = sorted(self.checkpoint_dir.glob("step_*.pt"))
        if not checkpoints:
            return None

        latest = checkpoints[-1]
        logger.info("Loading checkpoint: %s", latest)

        state = torch.load(latest, map_location=device, weights_only=False)

        # Unwrap DDP and torch.compile to get the base model
        raw_model = model.module if hasattr(model, "module") else model
        # torch.compile wraps in OptimizedModule with _orig_mod attribute
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod

        # Strip _orig_mod. prefix from checkpoint keys if present
        model_state = state["model"]
        if any(k.startswith("_orig_mod.") for k in model_state):
            model_state = {
                k.replace("_orig_mod.", ""): v for k, v in model_state.items()
            }
        raw_model.load_state_dict(model_state)

        muon_opt.load_state_dict(state["muon_optimizer"])
        adamw_opt.load_state_dict(state["adamw_optimizer"])

        step = state["step"]
        tokens = state.get("tokens_consumed", 0)
        logger.info("Resumed from step %d (%.2fB tokens)", step, tokens / 1e9)

        return state

    # -- Helpers ---------------------------------------------------------------

    def _cleanup_old(self, keep: int = 5) -> None:
        """Remove old checkpoints, keeping the most recent ``keep`` (<=0 keeps all)."""
        if keep <= 0:
            return
        checkpoints = sorted(self.checkpoint_dir.glob("step_*.pt"))
        if len(checkpoints) > keep:
            for old_ckpt in checkpoints[:-keep]:
                try:
                    old_ckpt.unlink()
                    logger.debug("Removed old checkpoint: %s", old_ckpt)
                except OSError as e:
                    logger.warning("Failed to remove %s: %s", old_ckpt, e)


def _scheduler_state(scheduler: Any) -> dict[str, Any]:
    """Extract serializable state from HybridScheduler."""
    return {
        "muon_base_lr": scheduler.muon_base_lr,
        "adamw_base_lr": scheduler.adamw_base_lr,
        "warmup_steps": scheduler.warmup_steps,
        "total_steps": scheduler.total_steps,
        "decay_start_step": scheduler.decay_start_step,
        "decay_type": scheduler.decay_type,
    }


# ── Sentinel for async worker shutdown ────────────────────────────────────────
_SHUTDOWN = object()


def _deep_cpu(obj: Any) -> Any:
    """Recursively clone tensors to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone().cpu()
    if isinstance(obj, dict):
        return {k: _deep_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_deep_cpu(v) for v in obj)
    return obj


class AsyncCheckpointManager(CheckpointManager):
    """Async checkpoint saving: CPU clone + background SHM write + zstd compression.

    Architecture:
      save() (training thread):
        1. Unwrap model, clone all state dicts to CPU (synchronous, ~0.5s)
        2. Barrier so all ranks sync after CPU clone
        3. Enqueue snapshot for background serialization

      _bg_worker (daemon thread):
        1. torch.save to /dev/shm (RAM bandwidth, ~300 GB/s)
        2. Optionally compress with zstd
        3. Rotate old checkpoints

      save_blocking():
        SIGTERM path — synchronous save, no background thread.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        rank: int = 0,
        keep_last_n: int = 5,
        shm_dir: str = "/dev/shm/kotodama-ckpts",
        compress: bool = True,
    ) -> None:
        super().__init__(checkpoint_dir, rank, keep_last_n)
        # Scope SHM dir per run using the full resolved checkpoint_dir path
        # to prevent cross-contamination. Two runs with different absolute paths
        # but the same basename (e.g. /a/checkpoints vs /b/checkpoints) get
        # different SHM directories.
        import hashlib
        resolved = str(Path(checkpoint_dir).resolve())
        run_hash = hashlib.sha256(resolved.encode()).hexdigest()[:12]
        run_label = f"{Path(checkpoint_dir).resolve().name}_{run_hash}"
        self.shm_dir = Path(shm_dir) / run_label
        self.compress = compress
        self._queue: Queue[Any] = Queue(maxsize=5)
        self._worker: Optional[threading.Thread] = None
        self._saved_checkpoints: list[tuple[int, Path]] = []  # (step, path) for rotation
        self._has_zstd: Optional[bool] = None

        if rank == 0:
            self.shm_dir.mkdir(parents=True, exist_ok=True)
            self._scan_existing()
            self._worker = threading.Thread(
                target=self._bg_worker, name="async-ckpt", daemon=True
            )
            self._worker.start()
            logger.info(
                "Async checkpoint: SHM=%s, compress=%s, keep=%d",
                self.shm_dir, compress, keep_last_n,
            )

    def _check_zstd(self) -> bool:
        """Check if zstd CLI is available."""
        if self._has_zstd is None:
            try:
                subprocess.run(
                    ["zstd", "--version"], capture_output=True, timeout=5,
                )
                self._has_zstd = True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                self._has_zstd = False
        return self._has_zstd

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        muon_opt: torch.optim.Optimizer,
        adamw_opt: torch.optim.Optimizer,
        scheduler: Any,
        tokens_consumed: int = 0,
        data_state: Optional[dict[str, Any]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Synchronous CPU clone + async SHM write."""
        if self.rank != 0:
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.barrier()
            return None

        t0 = time.time()

        # ── Synchronous: clone state to CPU ──
        raw_model = model.module if hasattr(model, "module") else model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod

        state: dict[str, Any] = {
            "step": step,
            "tokens_consumed": tokens_consumed,
            "model": _deep_cpu(raw_model.state_dict()),
            "muon_optimizer": _deep_cpu(muon_opt.state_dict()),
            "adamw_optimizer": _deep_cpu(adamw_opt.state_dict()),
            "scheduler": _scheduler_state(scheduler),
        }
        if data_state is not None:
            state["data"] = data_state
        if extra is not None:
            state.update(extra)

        clone_time = time.time() - t0

        # ── Barrier: all ranks sync after CPU clone ──
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()

        # ── Warn on backpressure ──
        queue_depth = self._queue.qsize()
        if queue_depth > 1:
            logger.warning(
                "Async checkpoint backpressure: queue=%d at step %d",
                queue_depth, step,
            )

        # ── Enqueue for background thread ──
        self._queue.put((step, state))
        logger.info(
            "Checkpoint step %d: CPU clone %.2fs, queued for background save",
            step, clone_time,
        )
        return self.checkpoint_dir / f"step_{step:08d}.pt"

    def save_blocking(
        self,
        step: int,
        model: torch.nn.Module,
        muon_opt: torch.optim.Optimizer,
        adamw_opt: torch.optim.Optimizer,
        scheduler: Any,
        tokens_consumed: int = 0,
        data_state: Optional[dict[str, Any]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> Optional[Path]:
        """SIGTERM path: synchronous save, no background thread."""
        path = super().save(
            step, model, muon_opt, adamw_opt, scheduler,
            tokens_consumed, data_state, extra,
        )
        # Register in bookkeeping so rotation tracks it
        if path is not None:
            self._saved_checkpoints.append((step, path))
            self._rotate_old()
        return path

    def load_latest(
        self,
        model: torch.nn.Module,
        muon_opt: torch.optim.Optimizer,
        adamw_opt: torch.optim.Optimizer,
        device: torch.device,
    ) -> Optional[dict[str, Any]]:
        """Load the most recent checkpoint, handling .pt.zst compressed files.

        Only searches checkpoint_dir (durable storage), NOT shm_dir.
        SHM is ephemeral scratch — checkpoints there may belong to other
        runs or be partially written.
        """
        candidates: list[tuple[int, Path]] = []
        if self.checkpoint_dir.exists():
            for pattern in ["step_*.pt", "step_*.pt.zst"]:
                for ckpt in self.checkpoint_dir.glob(pattern):
                    name = ckpt.name.replace(".pt.zst", "").replace(".pt", "")
                    try:
                        step = int(name.replace("step_", ""))
                        candidates.append((step, ckpt))
                    except ValueError:
                        continue

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        _, latest = candidates[-1]
        logger.info("Loading checkpoint: %s", latest)

        # Decompress .zst if needed
        if latest.name.endswith(".pt.zst"):
            if not self._check_zstd():
                raise RuntimeError(
                    f"Checkpoint {latest} is zstd-compressed but zstd CLI not found"
                )
            # Rank 0 alone decompresses (tmp + rename); every rank loads the
            # same file between barriers; rank 0 removes it. (If every rank ran
            # zstd -f into one shared path and unlinked it, a multi-rank resume
            # from a compressed checkpoint would die on the first unlink.)
            decompressed = latest.with_name(latest.name.replace(".pt.zst", ".pt.decomp"))
            distributed = dist.is_initialized() and dist.get_world_size() > 1
            if self.rank == 0:
                tmp = decompressed.with_name(decompressed.name + ".tmp")
                subprocess.run(
                    ["zstd", "-d", str(latest), "-o", str(tmp), "-f"],
                    check=True, capture_output=True,
                )
                tmp.rename(decompressed)
            if distributed:
                dist.barrier()
            try:
                state = torch.load(decompressed, map_location=device, weights_only=False)
            finally:
                if distributed:
                    dist.barrier()
                if self.rank == 0:
                    decompressed.unlink(missing_ok=True)
        else:
            state = torch.load(latest, map_location=device, weights_only=False)

        # Same unwrap/load logic as parent
        raw_model = model.module if hasattr(model, "module") else model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod

        model_state = state["model"]
        if any(k.startswith("_orig_mod.") for k in model_state):
            model_state = {
                k.replace("_orig_mod.", ""): v for k, v in model_state.items()
            }
        raw_model.load_state_dict(model_state)

        muon_opt.load_state_dict(state["muon_optimizer"])
        adamw_opt.load_state_dict(state["adamw_optimizer"])

        step = state["step"]
        tokens = state.get("tokens_consumed", 0)
        logger.info("Resumed from step %d (%.2fB tokens)", step, tokens / 1e9)
        return state

    # ── Background worker ────────────────────────────────────────────────

    def _bg_worker(self) -> None:
        """Background: serialize to SHM, optionally compress, rotate."""
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except Empty:
                continue

            if item is _SHUTDOWN:
                # Drain remaining items
                while not self._queue.empty():
                    try:
                        remaining = self._queue.get_nowait()
                        if remaining is not _SHUTDOWN:
                            self._process_item(remaining)
                    except Empty:
                        break
                break

            self._process_item(item)

    def _persist_to_disk(self, src: Path, step: int) -> None:
        """Atomically copy a checkpoint from SHM to checkpoint_dir for durability.

        Writes to a .tmp file first, then renames. If the process dies during
        the copy, the partial .tmp file won't be picked up by load_latest().
        """
        dest = self.checkpoint_dir / src.name
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        try:
            shutil.copy2(str(src), str(tmp))
            tmp.rename(dest)
            logger.info("Checkpoint step %d: persisted to %s", step, dest)
        except OSError as e:
            logger.warning(
                "Failed to persist step %d to disk: %s. "
                "Checkpoint remains in SHM only.", step, e,
            )
            # Clean up partial temp file
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

    def _process_item(self, item: tuple[int, dict[str, Any]]) -> None:
        """Write checkpoint to SHM, optionally compress, then persist to disk."""
        step, state = item
        t0 = time.time()

        shm_path = self.shm_dir / f"step_{step:08d}.pt"
        shm_tmp = shm_path.with_suffix(".tmp")

        try:
            torch.save(state, shm_tmp)
            shm_tmp.rename(shm_path)
        except Exception as e:
            logger.error("Background save failed for step %d: %s", step, e)
            if shm_tmp.exists():
                try:
                    shm_tmp.unlink()
                except OSError:
                    pass
            return

        save_time = time.time() - t0
        file_size = shm_path.stat().st_size

        # Backpressure: if queue is backing up, skip compression
        skip_compress = self._queue.qsize() > 2

        if self.compress and not skip_compress and self._check_zstd():
            t1 = time.time()
            zst_path = self.shm_dir / f"step_{step:08d}.pt.zst"
            try:
                subprocess.run(
                    ["zstd", "-3", "-T0", "-f", str(shm_path), "-o", str(zst_path)],
                    check=True, capture_output=True,
                )
                shm_path.unlink()
                zst_size = zst_path.stat().st_size
                compress_time = time.time() - t1
                logger.info(
                    "Checkpoint step %d: %.0f MB → %.0f MB (%.0f%%) in SHM, "
                    "save=%.1fs compress=%.1fs",
                    step, file_size / 1e6, zst_size / 1e6,
                    100 * zst_size / file_size, save_time, compress_time,
                )
                self._saved_checkpoints.append((step, zst_path))
                self._persist_to_disk(zst_path, step)
            except (subprocess.CalledProcessError, OSError) as e:
                logger.warning(
                    "Compression failed for step %d: %s. Keeping uncompressed.", step, e,
                )
                self._saved_checkpoints.append((step, shm_path))
                self._persist_to_disk(shm_path, step)
        else:
            if skip_compress and self.compress:
                logger.info(
                    "Checkpoint step %d: %.0f MB in SHM (backpressure, skipped compress), "
                    "save=%.1fs",
                    step, file_size / 1e6, save_time,
                )
            else:
                logger.info(
                    "Checkpoint step %d: %.0f MB in SHM, save=%.1fs",
                    step, file_size / 1e6, save_time,
                )
            self._saved_checkpoints.append((step, shm_path))
            self._persist_to_disk(shm_path, step)

        self._rotate_old()

    def _rotate_old(self) -> None:
        """Delete oldest checkpoints beyond keep_last_n (both SHM and disk)."""
        if self.keep_last_n <= 0:
            return
        self._saved_checkpoints.sort(key=lambda x: x[0])
        while len(self._saved_checkpoints) > self.keep_last_n:
            old_step, old_path = self._saved_checkpoints.pop(0)
            # Remove from SHM
            try:
                if old_path.is_file():
                    old_path.unlink()
                elif old_path.is_dir():
                    shutil.rmtree(old_path)
            except OSError as e:
                logger.warning("Failed to delete SHM checkpoint step %d: %s", old_step, e)
            # Remove corresponding disk copy
            disk_path = self.checkpoint_dir / old_path.name
            try:
                if disk_path.exists():
                    disk_path.unlink()
            except OSError as e:
                logger.warning("Failed to delete disk checkpoint step %d: %s", old_step, e)
            logger.info("Rotated out checkpoint step %d", old_step)

    def _scan_existing(self) -> None:
        """Scan SHM and checkpoint_dir for existing checkpoints.

        Tracks both so rotation works correctly even after resume with
        empty SHM (e.g. node restart clears /dev/shm).
        """
        seen_steps: set[int] = set()
        for search_dir in [self.shm_dir, self.checkpoint_dir]:
            if not search_dir.exists():
                continue
            for pattern in ["step_*.pt", "step_*.pt.zst"]:
                for ckpt in search_dir.glob(pattern):
                    name = ckpt.name.replace(".pt.zst", "").replace(".pt", "")
                    try:
                        step = int(name.replace("step_", ""))
                    except ValueError:
                        continue
                    # Prefer SHM copy if both exist (avoids double-counting)
                    if step not in seen_steps:
                        self._saved_checkpoints.append((step, ckpt))
                        seen_steps.add(step)
        self._saved_checkpoints.sort(key=lambda x: x[0])
        if self._saved_checkpoints:
            logger.info(
                "Found %d existing checkpoints (SHM + disk)",
                len(self._saved_checkpoints),
            )

    def shutdown(self) -> None:
        """Drain queue and stop background worker."""
        if self._worker is None or not self._worker.is_alive():
            return
        logger.info("Shutting down async checkpoint worker...")
        self._queue.put(_SHUTDOWN)
        self._worker.join(timeout=120)
        if self._worker.is_alive():
            logger.warning("Async checkpoint worker did not finish within timeout")


_DCP_NAME = re.compile(r"^step_(\d{8})\.dcp$")


class ShardedCheckpointManager(CheckpointManager):
    """Checkpoints of an FSDP2-sharded run, as torch.distributed.checkpoint (DCP) directories.

    ``step_XXXXXXXX.dcp/`` holds the sharded model and both optimizers (every rank writes its own shards, so the
    checkpoint directory must be one filesystem all ranks see) plus ``meta.pt`` (step, tokens, scheduler, data
    position and the run's extras, written by rank 0). A save goes to ``<name>.tmp`` and is renamed only once every
    rank has finished, so an interrupted save is never mistaken for a checkpoint. Resume reshards on load, so the
    world size may change between save and resume."""

    META = "meta.pt"

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        muon_opt: torch.optim.Optimizer,
        adamw_opt: torch.optim.Optimizer,
        scheduler: Any,
        tokens_consumed: int = 0,
        data_state: Optional[dict[str, Any]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> Optional[Path]:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict

        path = self.checkpoint_dir / f"step_{step:08d}.dcp"
        tmp = path.with_name(path.name + ".tmp")
        if self.rank == 0 and tmp.exists():
            shutil.rmtree(tmp)  # an earlier interrupted save of this very step
        _barrier()
        t0 = time.time()
        raw = _unwrap(model)
        msd, osd = get_state_dict(raw, [muon_opt, adamw_opt])
        dcp.save({"model": msd, "optim": osd}, checkpoint_id=str(tmp))
        if self.rank == 0:
            meta: dict[str, Any] = {
                "step": step,
                "tokens_consumed": tokens_consumed,
                "scheduler": _scheduler_state(scheduler),
                "ns_coefficients": getattr(muon_opt, "_ns_coefficients", None),
                "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            }
            if data_state is not None:
                meta["data"] = data_state
            if extra is not None:
                meta.update(extra)
            torch.save(meta, tmp / self.META)
        _barrier()
        if self.rank == 0:
            tmp.rename(path)
            logger.info("Sharded checkpoint saved: %s (%.1fs)", path, time.time() - t0)
            self._cleanup_old(keep=self.keep_last_n)
        _barrier()
        return path if self.rank == 0 else None

    def save_blocking(self, **kwargs: Any) -> Optional[Path]:
        return self.save(**kwargs)

    def latest(self) -> Optional[Path]:
        """The newest complete ``step_*.dcp`` directory, or None."""
        found = sorted(d for d in self.checkpoint_dir.glob("step_*.dcp")
                       if _DCP_NAME.match(d.name) and (d / self.META).exists())
        return found[-1] if found else None

    def load_latest(
        self,
        model: torch.nn.Module,
        muon_opt: torch.optim.Optimizer,
        adamw_opt: torch.optim.Optimizer,
        device: torch.device,
    ) -> Optional[dict[str, Any]]:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

        latest = self.latest()
        if latest is None:
            unsharded = sorted(self.checkpoint_dir.glob("step_*.pt")) + sorted(self.checkpoint_dir.glob("step_*.pt.zst"))
            if unsharded:
                raise RuntimeError(
                    f"{self.checkpoint_dir} holds unsharded checkpoints ({unsharded[-1].name}) but no sharded ones; "
                    "an FSDP2 run resumes only from step_*.dcp (refusing to start over at step 0)")
            return None
        logger.info("Loading sharded checkpoint: %s", latest)
        raw = _unwrap(model)
        msd, osd = get_state_dict(raw, [muon_opt, adamw_opt])
        state = {"model": msd, "optim": osd}
        dcp.load(state, checkpoint_id=str(latest))
        set_state_dict(raw, [muon_opt, adamw_opt], model_state_dict=state["model"], optim_state_dict=state["optim"])
        meta = torch.load(latest / self.META, map_location="cpu", weights_only=False)
        ns = meta.get("ns_coefficients")
        if ns is not None and getattr(muon_opt, "_ns_coefficients", ns) != ns:
            logger.warning("Muon ns_coefficients mismatch: checkpoint has %r, CLI has %r. Using the checkpoint's.",
                           ns, muon_opt._ns_coefficients)
            muon_opt._ns_coefficients = ns
        logger.info("Resumed from step %d (%.2fB tokens; saved at world size %s)", meta["step"],
                    meta.get("tokens_consumed", 0) / 1e9, meta.get("world_size"))
        return meta

    def _cleanup_old(self, keep: int = 5) -> None:
        """Remove old sharded checkpoints, keeping the newest ``keep`` (<=0 keeps all). Only directories named
        ``step_XXXXXXXX.dcp`` directly inside the checkpoint dir, holding a DCP ``.metadata``, are removed."""
        if keep <= 0:
            return
        found = sorted(d for d in self.checkpoint_dir.glob("step_*.dcp") if _DCP_NAME.match(d.name))
        for old in found[:-keep]:
            if old.parent != self.checkpoint_dir or not old.is_dir() or not (old / ".metadata").exists():
                logger.warning("Not removing %s: not a complete sharded checkpoint", old)
                continue
            try:
                shutil.rmtree(old)
            except OSError as e:
                logger.warning("Failed to remove %s: %s", old, e)


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    raw = model.module if hasattr(model, "module") else model
    return getattr(raw, "_orig_mod", raw)


def _barrier() -> None:
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()
