"""
Distributed training script for kotodama.

Launch with torchrun::

    torchrun --nproc_per_node=8 -m kotodama.training.train \\
        --data_path data/fineweb_sample.bin \\
        --checkpoint_dir checkpoints/proxy_p1 \\
        --model_size proxy \\
        --total_tokens 6_000_000_000

For testing without data::

    torchrun --nproc_per_node=1 -m kotodama.training.train \\
        --random_data --model_size smoke --total_steps 100

Uses DDP for gradient synchronization.  Muon's Newton-Schulz
orthogonalization operates on the full (replicated) weight matrices
after DDP all-reduces the gradients.  This is correct and simple — no
FSDP sharding complications.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import math
import os
import queue
import re
import sys
import threading
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Suppress Inductor's verbose "Online softmax is disabled" warnings — fires per
# subgraph and floods logs.  Must be AFTER `import torch` (torch resets filters).
warnings.filterwarnings("ignore", message="Online softmax", module=r"torch\._inductor")

# Project imports — put src/ on sys.path so a direct script launch resolves `kotodama`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from kotodama.model.init_schemes import SCHEMES as _INIT_SCHEMES
from kotodama.data.dataset import (
    MixedTokenDataset,
    RandomTokenDataset,
    TokenizedDataset,
    collate_packed,
)
from kotodama import presets
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
from kotodama.monitoring.geometric import GeometricMonitor, MonitorConfig, MonitorSchedule
from kotodama.monitoring.wandb_callback import WandbLogger
from kotodama.training.checkpoint import (
    AsyncCheckpointManager,
    CheckpointManager,
    SIGTERMHandler,
    TokenThresholds,
    parse_save_schedule,
    parse_token_count,
    save_weights_only,
)
from kotodama.training.inloop import InloopConfig, InloopEvaluator
from kotodama.training.muon import HybridScheduler, build_hybrid_optimizer, decay_start_step

logger = logging.getLogger(__name__)


def mix_window_seqs(spec: str | None, tokens_per_seq_step: int) -> tuple[int, int] | None:
    """'START:END' in global tokens -> [start, end) in per-rank sequence index (each rank emits one
    sequence per ``tokens_per_seq_step`` = seq_len * dp_size global tokens). None/'' = no window."""
    if not spec:
        return None
    try:
        a, b = spec.split(":")
    except ValueError as e:
        raise ValueError(f"--mix_window {spec!r} is not START:END") from e
    start = 0 if a.strip() in ("0", "") else parse_token_count(a)
    end = parse_token_count(b)
    if not 0 <= start < end:
        raise ValueError(f"--mix_window {spec!r}: need 0 <= START < END")
    return start // tokens_per_seq_step, end // tokens_per_seq_step


# ── Scheduler metrics (lightweight, no dependency) ────────────────────────────

_hm_job_id = os.environ.get("KOTODAMA_SCHEDULER_JOB_ID")
# Metrics post from a background thread: a synchronous POST on rank 0 stalled every rank at the next all-reduce for as
# long as the coordinator took to answer. Bounded queue; when full, new points are dropped (telemetry never stalls
# training). One worker, so points arrive in order. Flushed (bounded wait) at exit.
_HM_QUEUE_MAX = 4096
_hm_queue: Optional["queue.Queue[Optional[tuple[int, dict[str, float]]]]"] = None


def _hm_worker(q: "queue.Queue[Optional[tuple[int, dict[str, float]]]]") -> None:
    client = None
    while True:
        item = q.get()
        try:
            if item is None:
                return
            if client is None:
                import httpx
                client = httpx.Client(
                    base_url=os.environ.get("KOTODAMA_SCHEDULER_URL", "http://127.0.0.1:7000"), timeout=5)
            step, clean = item
            client.post(f"/api/v1/jobs/{_hm_job_id}/metrics", json={"step": step, "metrics": clean})
        except Exception:
            pass  # Never crash the training loop
        finally:
            q.task_done()


def hm_flush(timeout: float = 30.0) -> None:
    """Wait (at most ``timeout`` s) for queued metrics to be posted."""
    if _hm_queue is None:
        return
    deadline = time.time() + timeout
    while _hm_queue.unfinished_tasks and time.time() < deadline:
        time.sleep(0.05)


def hm_log(step: int, **metrics: float) -> None:
    """Queue metrics for the Scheduler (posted by a background thread). No-op if not running under Scheduler."""
    global _hm_queue
    if not _hm_job_id:
        return
    clean = {k: v for k, v in metrics.items() if v is not None}
    if not clean:
        return
    if _hm_queue is None:
        import atexit

        _hm_queue = queue.Queue(maxsize=_HM_QUEUE_MAX)
        threading.Thread(target=_hm_worker, args=(_hm_queue,), daemon=True, name="hm_log").start()
        atexit.register(hm_flush)
    try:
        _hm_queue.put_nowait((step, clean))
    except queue.Full:
        pass


_SECRETISH = re.compile(r"(key|secret|passw|auth|credential|_token$|^token$)", re.I)


def data_parallel_rank(rank: int, dp_group: Any) -> int:
    """This process's data-partition index.

    Under TP, TP peers must share data, so the rank inside the DP group is used. Without TP
    (``dp_group is None``) every DDP rank is its own data-parallel replica: its global rank."""
    if dp_group is not None:
        return dist.get_rank(dp_group)
    return rank


def hm_config(config: dict[str, Any]) -> None:
    """Record run hyperparameters with Scheduler (runs-table columns, group-by, scatter
    axes). Scalars and strings only; merged server-side. No-op outside Scheduler."""
    if not _hm_job_id:
        return
    clean = {k: v for k, v in config.items()
             if v is not None and isinstance(v, (bool, int, float, str))
             and not _SECRETISH.search(k)}
    try:
        import httpx
        with httpx.Client(base_url=os.environ.get("KOTODAMA_SCHEDULER_URL", "http://127.0.0.1:7000"),
                          timeout=5) as c:
            c.post(f"/api/v1/jobs/{_hm_job_id}/config", json={"config": clean})
    except Exception:
        pass  # Never crash the training loop


def _upload_checkpoint_to_hf(
    repo_id: str,
    checkpoint_dir: Path,
    step: int,
    tokens_consumed: int,
    config_file: Optional[str] = None,
) -> None:
    """Upload final checkpoint to HuggingFace Hub. Best-effort — never crashes training."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True)

        ckpt_name = f"step_{step:08d}"
        candidates = [
            checkpoint_dir / f"{ckpt_name}.pt.zst",
            checkpoint_dir / f"{ckpt_name}.pt",
        ]
        ckpt_path = next((p for p in candidates if p.exists()), None)
        if ckpt_path is None:
            logger.warning("HF upload: final checkpoint not found at %s", checkpoint_dir)
            return

        logger.info("Uploading %s to %s ...", ckpt_path.name, repo_id)
        api.upload_file(
            path_or_fileobj=str(ckpt_path),
            path_in_repo=ckpt_path.name,
            repo_id=repo_id,
            commit_message=f"Final checkpoint: step {step}, {tokens_consumed / 1e9:.1f}B tokens",
        )

        if config_file and Path(config_file).exists():
            api.upload_file(
                path_or_fileobj=config_file,
                path_in_repo=Path(config_file).name,
                repo_id=repo_id,
                commit_message="Training config",
            )

        logger.info("HF upload complete: %s", repo_id)
    except Exception:
        logger.exception("HF upload failed (training completed successfully)")


# =============================================================================
# Model configs — one per kotodama.presets shape
# =============================================================================

MODEL_CONFIGS: dict[str, dict[str, Any]] = {name: presets.shape(name) for name in presets.names()}


def _model_flops_per_token(
    param_count: int,
    config: KotodamaModelConfig,
    seq_len: int,
) -> float:
    """Analytic model FLOPs per token, forward + backward (PaLM-style 6N).

    6 × matmul params (embedding lookup is free; the tied lm_head matmul IS
    the embedding matrix, so tied models count param_count as-is) plus the
    attention score/value term at causal average context seq_len/2.
    Doc masking shortens real attention context, so the attention term is a
    mild overestimate (~10% of the total) — reported MFU is comparable across
    runs, which is what matters.
    """
    embed_params = config.vocab_size * config.hidden_size
    if config.tie_word_embeddings:
        n_matmul = param_count  # embed excluded, lm_head included — same matrix
    else:
        n_matmul = param_count - embed_params
    attn_flops = (
        6.0
        * seq_len
        * config.num_layers
        * config.num_attention_heads
        * config.head_dim
    )
    return 6.0 * n_matmul + attn_flops


class _PrefetchIterator:
    """Background-thread batch prefetcher with exact dataset-state tracking.

    Pulls (batch, dataset_state) pairs ahead of the training loop so memmap
    page faults, EOS scanning, and collation overlap GPU compute instead of
    running in the gaps between kernel launches. ``consumed_state`` always
    reflects only batches actually handed to the training loop, so a
    checkpoint saved mid-run resumes with zero skipped / replayed sequences
    — identical semantics to the unprefetched loader.
    """

    def __init__(
        self,
        make_iter: Callable[[], Iterator[Any]],
        dataset: Any,
        depth: int = 2,
    ) -> None:
        self._make_iter = make_iter
        self._dataset = dataset
        self._queue: queue.Queue[Optional[tuple[Any, Optional[dict[str, int]]]]] = (
            queue.Queue(maxsize=max(1, depth))
        )
        self._stop = threading.Event()
        self._exc: Optional[BaseException] = None
        self.consumed_state: Optional[dict[str, int]] = (
            dataset.state_dict() if hasattr(dataset, "state_dict") else None
        )
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="data-prefetch"
        )
        self._thread.start()

    def _worker(self) -> None:
        try:
            it = self._make_iter()
            while not self._stop.is_set():
                try:
                    batch = next(it)
                except StopIteration:
                    it = self._make_iter()
                    continue
                state = (
                    self._dataset.state_dict()
                    if hasattr(self._dataset, "state_dict")
                    else None
                )
                while not self._stop.is_set():
                    try:
                        self._queue.put((batch, state), timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except BaseException as e:  # surface to consumer, never die silently
            self._exc = e
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass

    def __iter__(self) -> "_PrefetchIterator":
        return self

    def __next__(self) -> Any:
        while True:
            try:
                item = self._queue.get(timeout=5.0)
            except queue.Empty:
                if self._exc is not None:
                    raise RuntimeError("data prefetch worker failed") from self._exc
                if not self._thread.is_alive():
                    raise RuntimeError("data prefetch worker died without an exception")
                continue
            if item is None:
                raise RuntimeError("data prefetch worker failed") from self._exc
            batch, state = item
            if state is not None:
                self.consumed_state = state
            return batch

    def shutdown(self) -> None:
        self._stop.set()
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=5.0)


# =============================================================================
# Distributed setup
# =============================================================================


class _FlexBlockMaskBuilder:
    """Builds the per-step FlexAttention BlockMask (causal-within-document).

    Built OUTSIDE the compiled model (BlockMask is a graph input, not traced
    construction). Doc ids come from position_ids resets (position 0 = doc
    start — the same signal cu_seqlens encodes). The doc-id tensor is a
    persistent buffer updated with copy_ so the mask_mod closure always sees
    the same tensor object: a fresh closure/tensor identity per step would
    force dynamo to recompile create_block_mask every step.
    """

    def __init__(self, device: torch.device) -> None:
        from torch.nn.attention.flex_attention import create_block_mask

        self._create = create_block_mask
        self._device = device
        self._doc_ids: Optional[torch.Tensor] = None

        def _doc_causal(b, h, q_idx, kv_idx):  # noqa: ANN001
            return (
                self._doc_ids[b, q_idx] == self._doc_ids[b, kv_idx]
            ) & (q_idx >= kv_idx)

        self._mask_mod = _doc_causal

    def build(self, position_ids: torch.Tensor):
        bsz, seq_len = position_ids.shape
        doc_ids = (position_ids == 0).cumsum(dim=1, dtype=torch.int32)
        if self._doc_ids is None or self._doc_ids.shape != doc_ids.shape:
            self._doc_ids = doc_ids.contiguous()
        else:
            self._doc_ids.copy_(doc_ids)
        return self._create(
            self._mask_mod, bsz, None, seq_len, seq_len,
            device=self._device, _compile=True,
        )


def setup_distributed() -> tuple[int, int, int]:
    """Initialize NCCL process group.  Returns (rank, world_size, local_rank)."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# Logging
# =============================================================================


def setup_logging(rank: int) -> None:
    """Configure logging — only rank 0 logs to stdout."""
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format=f"[rank {rank}] %(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# =============================================================================
# Training loop
# =============================================================================


def train(args: argparse.Namespace) -> None:
    # -- Distributed setup -----------------------------------------------------
    rank, world_size, local_rank = setup_distributed()
    setup_logging(rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    # -- Tensor parallelism groups ---------------------------------------------
    tp_size = getattr(args, "tp_size", 1)
    tp_group = None
    dp_group = None
    if tp_size > 1:
        from kotodama.training.tensor_parallel import create_process_groups
        tp_group, dp_group = create_process_groups(world_size, tp_size)
    dp_size = world_size // tp_size

    if is_main:
        logger.info("World size: %d, device: %s", world_size, device)
        if tp_size > 1:
            logger.info("TP=%d, DP=%d", tp_size, dp_size)
        job_id = os.environ.get("KOTODAMA_SCHEDULER_JOB_ID", "local")
        logger.info("Scheduler job ID: %s", job_id)

    # -- Model -----------------------------------------------------------------
    if args.model_size not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model size '{args.model_size}'. "
            f"Choose from: {list(MODEL_CONFIGS.keys())}"
        )

    model_kwargs = MODEL_CONFIGS[args.model_size].copy()
    model_kwargs["qk_norm"] = True
    model_kwargs["z_loss_weight"] = args.z_loss_weight
    model_kwargs["activation_checkpointing"] = args.activation_checkpointing
    model_kwargs["use_liger"] = getattr(args, "use_liger", False)
    model_kwargs["attn_res_gain"] = getattr(args, "attnres_res_gain", "learned")
    model_kwargs["attn_res_query_norm"] = getattr(args, "attnres_query_norm", False)
    model_kwargs["attn_res_query_temp_gain"] = getattr(args, "attnres_query_temp_gain", 1.0)
    model_kwargs["liger_rmsnorm_tuned"] = getattr(args, "liger_rmsnorm_tuned", False)
    model_kwargs["liger_rmsnorm_tuned_scope"] = getattr(args, "liger_rmsnorm_tuned_scope", "all")
    model_kwargs["rope_fp32"] = getattr(args, "rope_fp32", False)
    model_kwargs["attn_impl"] = getattr(args, "attn_impl", "auto")
    model_kwargs["attn_res"] = args.attn_res
    model_kwargs["attn_res_n_blocks"] = args.attn_res_n_blocks
    model_kwargs["attn_res_freeze_unused"] = getattr(args, "attn_res_freeze_unused", True)
    if args.attn_res_boundaries:
        model_kwargs["attn_res_boundaries"] = [int(x) for x in args.attn_res_boundaries.split(",")]
    config = KotodamaModelConfig(**model_kwargs)

    # Seed the init: without this every run of the same config draws different
    # initial weights (data order is already seeded in the dataset). DDP then
    # broadcasts rank 0's params, so ranks agree either way.
    torch.manual_seed(args.seed)
    model = KotodamaModel(config).to(device=device, dtype=torch.bfloat16)
    init_scheme = getattr(args, "init_scheme", "standard")
    if init_scheme != "standard":
        from kotodama.model.init_schemes import apply_init_scheme
        apply_init_scheme(model, init_scheme)
        if is_main:
            logger.info("Init scheme: %s (applied over the standard init)", init_scheme)
    param_count = sum(p.numel() for p in model.parameters())

    if is_main:
        logger.info(
            "Model: %s (%.1fM params, config estimate: %s)",
            args.model_size,
            param_count / 1e6,
            f"{config.param_count() / 1e6:.1f}M",
        )

    # -- CPT / weight-only resume ------------------------------------------------
    if args.resume_weights:
        resume_path = Path(args.resume_weights)
        if is_main:
            logger.info("Loading weights for CPT: %s", resume_path)
        load_path = resume_path
        if resume_path.name.endswith(".pt.zst"):
            import subprocess as _sp
            load_path = resume_path.with_name(resume_path.name.replace(".pt.zst", ".pt"))
            if is_main:
                _sp.run(["zstd", "-d", str(resume_path), "-o", str(load_path), "-f"],
                        check=True, capture_output=True)
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.barrier()

        weight_state = torch.load(load_path, map_location=device, weights_only=False)

        if resume_path.name.endswith(".pt.zst") and is_main:
            load_path.unlink(missing_ok=True)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()

        model_state = weight_state.get("model", weight_state)
        cleaned_state = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}
        model.load_state_dict(cleaned_state, strict=True)
        if is_main:
            logger.info("CPT weights loaded (step %d, %.2fB tokens from prior run)",
                        weight_state.get("step", -1),
                        weight_state.get("tokens_consumed", 0) / 1e9)

    # -- Warm resume (model + optimizer, reset step/data) -------------------------
    if getattr(args, "resume_warm", None):
        resume_path = Path(args.resume_warm)
        if is_main:
            logger.info("Warm resume (model + optimizer): %s", resume_path)
        load_path = resume_path
        if resume_path.name.endswith(".pt.zst"):
            import subprocess as _sp
            load_path = resume_path.with_name(resume_path.name.replace(".pt.zst", ".pt"))
            if is_main:
                _sp.run(["zstd", "-d", str(resume_path), "-o", str(load_path), "-f"],
                        check=True, capture_output=True)
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.barrier()

        ckpt = torch.load(load_path, map_location=device, weights_only=False)

        if resume_path.name.endswith(".pt.zst") and is_main:
            load_path.unlink(missing_ok=True)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()

        # Model weights
        model_state = ckpt.get("model", ckpt)
        cleaned_state = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}
        model.load_state_dict(cleaned_state, strict=True)

        # Weights only: optimizer state is NOT restored on a warm resume (fresh
        # Muon/AdamW moments; the schedule restarts from step 0).

        if is_main:
            logger.info(
                "Warm resume loaded (prior step %d, %.2fB tokens). "
                "Step counter reset to 0, data starts from beginning.",
                ckpt.get("step", -1),
                ckpt.get("tokens_consumed", 0) / 1e9,
            )

    # -- NCA → language transition ---------------------------------------------
    # Skip if checkpoint dir already has saves (restart after preemption).
    if args.reset_routing_layers and not args.resume_nca:
        raise ValueError("--reset_routing_layers only applies at the NCA -> language handoff (needs --resume_nca)")
    _has_existing_ckpts = (any(Path(args.checkpoint_dir).glob("step_*.pt")) or any(Path(args.checkpoint_dir).glob("step_*.pt.zst"))) if args.checkpoint_dir else False
    if args.resume_nca and not _has_existing_ckpts:
        if is_main:
            logger.info("Loading NCA checkpoint: %s", args.resume_nca)
        nca_path = Path(args.resume_nca)
        load_path = nca_path
        if nca_path.name.endswith(".pt.zst"):
            import subprocess as _sp
            load_path = nca_path.with_name(nca_path.name.replace(".pt.zst", ".pt"))
            if is_main:
                _sp.run(["zstd", "-d", str(nca_path), "-o", str(load_path), "-f"],
                        check=True, capture_output=True)
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.barrier()

        nca_state = torch.load(load_path, map_location=device, weights_only=False)

        # Every rank must finish loading before rank 0 removes the temp file.
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()
        if nca_path.name.endswith(".pt.zst") and is_main:
            load_path.unlink(missing_ok=True)

        nca_model_state = nca_state.get("model", nca_state)

        # Strip _orig_mod. prefix from torch.compile'd checkpoints
        cleaned_state: dict[str, torch.Tensor] = {}
        for k, v in nca_model_state.items():
            clean_key = k.replace("_orig_mod.", "")
            cleaned_state[clean_key] = v
        # strict=False: AttnRes params (queries, norms) won't be in NCA checkpoint
        # and stay at their defaults (zero queries = uniform weighting, ones for norms)
        missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
        if is_main:
            if missing:
                logger.info("NCA load — missing keys (expected for AttnRes): %d keys", len(missing))
            if unexpected:
                logger.warning("NCA load — unexpected keys: %s", unexpected[:5])
            logger.info("NCA weights loaded")

        # Reinitialize embeddings to language vocab
        if is_main:
            logger.info(
                "Reinitializing embeddings: NCA vocab → language vocab (%d)",
                config.vocab_size,
            )
        model.reinit_embeddings(new_vocab_size=config.vocab_size)

        # Optionally return named routing points to fresh init (--reset_routing_layers)
        if args.reset_routing_layers:
            _rr_layers, _rr_final = parse_routing_spec(args.reset_routing_layers, config.num_layers)
            _rr_names = model.reset_routing(_rr_layers, final=_rr_final)
            if is_main:
                logger.info("Reset AttnRes routing to fresh init (%s): %d params",
                            args.reset_routing_layers, len(_rr_names))

        # Optionally reinitialize MLPs
        if args.reinit_mlps:
            if is_main:
                logger.info("Reinitializing MLPs (--reinit_mlps set)")
            model.reinit_mlps()
        else:
            if is_main:
                logger.info("Keeping NCA-trained MLPs (embed-only reinit)")

        if is_main:
            logger.info("NCA → language transition complete")
    elif args.resume_nca and _has_existing_ckpts:
        if is_main:
            logger.info("Skipping NCA init — existing checkpoints found (resuming from preemption)")

    # --attnres_res_gain folded: move every routing gain into its query now, after any weights were loaded and before
    # the optimizers take their fp32 masters (fold_attn_res_gains is idempotent; the full-resume load re-applies it).
    if config.attn_res and config.attn_res_gain == "folded":
        _folded = model.fold_attn_res_gains()
        if is_main:
            logger.info("AttnRes res_norm gains folded into the queries (%d sites had a gain != 1); gains frozen",
                        _folded)
    elif config.attn_res and config.attn_res_gain == "frozen" and is_main:
        logger.info("AttnRes res_norm gains frozen at their current values")

    # Enable TF32 for fp32 matmuls outside autocast (grad norm, loss compute)
    torch.set_float32_matmul_precision("high")

    # Tensor parallelism: shard projection weights across TP group
    if tp_group is not None:
        from kotodama.training.tensor_parallel import apply_tensor_parallelism
        apply_tensor_parallelism(model, tp_group)
        if is_main:
            tp_param_count = sum(p.numel() for p in model.parameters())
            logger.info("After TP sharding: %.1fM params per rank", tp_param_count / 1e6)

    # FP8 training: convert Linear layers to Float8Linear (before compile)
    if getattr(args, "fp8", False):
        fp8_recipe = getattr(args, "fp8_recipe", "tensorwise")
        if fp8_recipe == "mxfp8":
            # Per-32-element block scales, native B200 format. Gated like any
            # number-format change (see kotodama.model.mxfp8).
            from ..model.mxfp8 import convert_to_mxfp8_training

            convert_to_mxfp8_training(model)
            if is_main:
                logger.info("FP8 training enabled via torchao (MXFP8Linear)")
        else:
            try:
                from torchao.float8 import convert_to_float8_training, Float8LinearConfig
                fp8_config = Float8LinearConfig()
                convert_to_float8_training(model, config=fp8_config)
                if is_main:
                    logger.info("FP8 training enabled via torchao (Float8Linear)")
            except ImportError:
                raise ImportError(
                    "--fp8 requires torchao. Install with: uv pip install torchao"
                )

    # torch.compile for throughput (compile before DDP)
    if args.compile:
        # --attn_impl fa4 is compile-safe with or without --doc_masking: both
        # the dense and varlen paths run through kotodama::fa4_* custom ops
        # (model/fa4_varlen.py), opaque to dynamo.
        compile_mode = getattr(args, "compile_mode", None)
        if getattr(args, "capture_scalar_outputs", False):
            # Lets dynamo trace through the Liger FLCE .item() call — the one
            # remaining graph-break site.
            torch._dynamo.config.capture_scalar_outputs = True
            if is_main:
                logger.info("dynamo capture_scalar_outputs=True (FLCE tail-break fix)")
        ac_budget = getattr(args, "ac_budget", 0.0)
        if ac_budget and ac_budget > 0.0:
            # Inductor min-cut partitioner with a memory budget: recompute is
            # chosen optimally per-graph instead of the manual every-sublayer
            # AC policy. Math-identical (recompute-schedule-only).
            # Use with --no_activation_checkpointing — combining both would
            # nest recompute regions.
            if config.activation_checkpointing:
                raise ValueError(
                    "--ac_budget requires manual AC off "
                    "(--no_activation_checkpointing): nesting both recompute "
                    "policies double-recomputes."
                )
            import torch._functorch.config as functorch_config

            functorch_config.activation_memory_budget = ac_budget
            if is_main:
                logger.info(
                    "Inductor activation_memory_budget=%.2f (partitioner-driven "
                    "selective recompute; manual AC off)", ac_budget)
        if getattr(args, "compile_regional", False):
            if is_main:
                logger.info("Regional compile: per-%s regions (mode=%s)...",
                            getattr(args, "compile_regional_scope", "sublayer"), compile_mode)
            model.compile_regions(scope=getattr(args, "compile_regional_scope", "sublayer"), mode=compile_mode)
        else:
            if is_main:
                logger.info("Compiling model with torch.compile (mode=%s)...", compile_mode)
            model = torch.compile(model, mode=compile_mode)
        if is_main:
            logger.info("Compilation registered (will compile on first forward)")

    # Wrap in DDP for gradient synchronization across data-parallel ranks.
    # Pure TP (tp_size == world_size): no DDP — all ranks are in the same
    # TP group with identical replicated-param gradients.
    if dp_size > 1:
        # With attn_res_freeze_unused, the only never-used params (first
        # boundary layer's pre-attn routing) have requires_grad=False, so DDP
        # can skip the per-iteration unused-param graph traversal entirely.
        ddp_find_unused = config.attn_res and not config.attn_res_freeze_unused
        ddp_kwargs = ddp_options(args, find_unused=ddp_find_unused)
        model = DDP(model, device_ids=[local_rank], process_group=dp_group, **ddp_kwargs)
        if is_main:
            logger.info("DDP %s", ddp_kwargs)
    else:
        if is_main:
            logger.info("Pure TP mode (TP=%d = world_size) — skipping DDP", tp_size)

    # -- Optimizer -------------------------------------------------------------
    raw_model = model.module if hasattr(model, "module") else model

    # Per-microbatch grad stats + spike capture (opt-in; kotodama.training.grad_stats)
    micro_stats = None
    spike_det = None
    if getattr(args, "micro_grad_stats", False) or getattr(args, "spike_capture_factor", 0.0):
        from kotodama.training.grad_stats import MicroGradStats, SpikeDetector

        _is_ddp = isinstance(model, DDP)
        micro_stats = MicroGradStats(
            [p for p in raw_model.parameters() if p.requires_grad], device, group=dp_group, ddp=_is_ddp,
            out_path=(Path(args.checkpoint_dir) / "micro_grad_stats.jsonl"
                      if is_main and getattr(args, "micro_grad_stats", False) else None))
        if _is_ddp:
            model.register_comm_hook(dp_group, micro_stats.comm_hook)
        if getattr(args, "spike_capture_factor", 0.0):
            spike_det = SpikeDetector(args.spike_capture_factor, window=args.spike_capture_window,
                                      max_captures=args.spike_capture_max)
        if is_main:
            logger.info("micro grad stats on (file=%s); spike capture factor=%s",
                        micro_stats.out_path, getattr(args, "spike_capture_factor", 0.0) or "off")
    _micro_rec = None

    if args.adamw_only:
        # Pure AdamW baseline (no Muon) — all params in one optimizer.
        # MasterAdamW keeps fp32 masters: stock AdamW on a bf16 model freezes
        # every RMSNorm gain at 1.0.
        from kotodama.training.muon import MasterAdamW as TorchAdamW

        all_params = [p for p in raw_model.parameters() if p.requires_grad]
        adamw_opt = TorchAdamW(
            all_params,
            lr=args.adamw_lr,
            betas=(args.adamw_beta1, args.adamw_beta2),
            weight_decay=args.adamw_weight_decay,
        )
        # Create a no-op optimizer stand-in for Muon
        muon_opt = _NoOpOptimizer()
        if is_main:
            total_p = sum(p.numel() for p in all_params)
            logger.info("AdamW-only mode: %d params (%.1fM)", total_p, total_p / 1e6)
    else:
        muon_opt, adamw_opt = build_hybrid_optimizer(
            raw_model,
            muon_lr=args.muon_lr,
            muon_momentum=args.muon_momentum,
            muon_weight_decay=args.muon_weight_decay,
            muon_ns_iterations=args.muon_ns_iterations,
            muon_ns_coefficients=args.muon_ns_coefficients,
            muon_distributed=getattr(args, "muon_distributed", False),
            adamw_lr=args.adamw_lr,
            adamw_betas=(args.adamw_beta1, args.adamw_beta2),
            adamw_weight_decay=args.adamw_weight_decay,
            adamw_no_decay_norms=getattr(args, "adamw_no_decay_norms", False),
            muon_fp32_master=getattr(args, "muon_fp32_master", False),
            attn_group=_attn_lr_scaled(args),
            attn_layers=parse_layer_range(getattr(args, "attn_lr_scale_layers", "")),
            res_query_lr_scale=getattr(args, "attnres_query_lr_scale", 1.0),
        )

    # -- Compute training geometry ---------------------------------------------
    seq_len = args.sequence_length
    micro_batch = args.micro_batch_size
    tokens_per_micro = micro_batch * seq_len  # per GPU
    # Under TP, all ranks in a TP group process the same data.
    # Only data-parallel replicas contribute unique tokens.
    tokens_per_global_micro = tokens_per_micro * dp_size

    if args.total_steps is not None:
        total_steps = args.total_steps
        # Infer total_tokens from steps (useful for smoke tests)
        grad_accum = max(1, args.global_batch_tokens // tokens_per_global_micro)
        total_tokens = total_steps * grad_accum * tokens_per_global_micro
    else:
        total_tokens = args.total_tokens
        grad_accum = max(1, args.global_batch_tokens // tokens_per_global_micro)
        total_steps = total_tokens // (grad_accum * tokens_per_global_micro)

    tokens_per_step = grad_accum * tokens_per_global_micro
    loop_end = _loop_end_step(total_steps, args.stop_at_step)
    if is_main and loop_end < total_steps:
        logger.info("--stop_at_step: exiting after step %d of a %d-step schedule",
                    loop_end - 1, total_steps)

    if is_main:
        logger.info(
            "Training: %d steps, %d grad_accum, %d tokens/step (%.2fM), "
            "%.1fB total tokens",
            total_steps,
            grad_accum,
            tokens_per_step,
            tokens_per_step / 1e6,
            total_tokens / 1e9,
        )

    # -- Scheduler -------------------------------------------------------------
    scheduler = HybridScheduler(
        muon_opt,
        adamw_opt,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        decay_start_pct=args.decay_start_pct,
        decay_type=args.decay_type,
        attn_lr_scale=getattr(args, "attn_lr_scale", 1.0),
        attn_lr_scale_steps=getattr(args, "attn_lr_scale_steps", 0),
    )
    if is_main and _attn_lr_scaled(args):
        logger.info("Attention-projection LR x%.3g for steps < %d", args.attn_lr_scale, args.attn_lr_scale_steps)

    # -- Data ------------------------------------------------------------------
    # Under TP, all ranks in a TP group must see the same data.
    # Partition by dp_rank so TP peers share batches.
    dp_rank = data_parallel_rank(rank, dp_group)
    if apply_attnres_eps_compat(args) and is_main:
        logger.info("AttnRes eps fp64 compat ON: forward routing runs in fp64 under torch.compile (pre-constexpr numerics)")
    if apply_attnres_stack_grad_chain(args) and is_main:
        logger.info("AttnRes stack-gradient chain ON: routes sharing a committed stack sum its gradient in-kernel")
    doc_masking = getattr(args, "doc_masking", False)
    varlen_dynamic = bool(getattr(args, "varlen_dynamic", False) and args.compile and doc_masking)
    if is_main and getattr(args, "varlen_dynamic", False) and not varlen_dynamic:
        logger.warning("--varlen_dynamic ignored: it needs --compile and --doc_masking")
    data_seed = training_data_seed(args)
    if is_main and data_seed != args.seed:
        logger.info("Data order seed %d (init seed %d)", data_seed, args.seed)

    if args.random_data:
        dataset = RandomTokenDataset(
            vocab_size=config.vocab_size,
            seq_len=seq_len,
            seed=data_seed,
            rank=dp_rank,
            doc_masking=doc_masking,
        )
        if is_main:
            logger.info("Using random data (vocab=%d, doc_masking=%s)", config.vocab_size, doc_masking)
    else:
        if not args.data_path:
            raise ValueError("--data_path required when not using --random_data")
        dataset = TokenizedDataset(
            path=args.data_path,
            seq_len=seq_len,
            rank=dp_rank,
            world_size=dp_size,
            seed=data_seed,
            doc_masking=doc_masking,
        )
        mix_fraction = float(getattr(args, "mix_fraction", 0.0) or 0.0)
        if getattr(args, "mix_data_path", None) and mix_fraction > 0.0:
            window = mix_window_seqs(getattr(args, "mix_window", None), seq_len * dp_size)
            dataset = MixedTokenDataset(
                dataset,
                TokenizedDataset(path=args.mix_data_path, seq_len=seq_len, rank=dp_rank,
                                 world_size=dp_size, seed=data_seed, doc_masking=doc_masking),
                mix_fraction,
                window=window,
            )
            if is_main:
                logger.info("Mixing %s at %.2f%% of tokens (token-debt interleave)%s",
                            args.mix_data_path, 100 * mix_fraction,
                            f" inside tokens [{args.mix_window}) = per-rank sequences {window}" if window else "")
        elif getattr(args, "mix_window", None):
            raise ValueError("--mix_window needs --mix_data_path and --mix_fraction > 0")

    def _make_data_iter():
        collate_fn = collate_packed if doc_masking else None
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=micro_batch,
            num_workers=0,  # IterableDataset handles its own partitioning
            pin_memory=True,
            collate_fn=collate_fn,
        )
        return iter(loader)

    data_iter = _make_data_iter()

    # -- Eval data (optional held-out set) -------------------------------------
    eval_dataset = None
    eval_iter_fn = None
    eval_every = getattr(args, "eval_every", 0)
    eval_batches = getattr(args, "eval_batches", 20)

    if getattr(args, "eval_data_path", None):
        eval_dataset = TokenizedDataset(
            path=args.eval_data_path,
            seq_len=seq_len,
            rank=dp_rank,
            world_size=dp_size,
            seed=args.seed + 7919,
            doc_masking=doc_masking,
        )
        def _make_eval_iter():
            collate_fn = collate_packed if doc_masking else None
            loader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=micro_batch,
                num_workers=0,
                pin_memory=True,
                collate_fn=collate_fn,
            )
            return iter(loader)
        eval_iter_fn = _make_eval_iter
        if is_main:
            logger.info(
                "Eval data: %s (%.1fM tokens, eval every %d steps, %d batches)",
                args.eval_data_path,
                eval_dataset.total_tokens / 1e6,
                eval_every,
                eval_batches,
            )

    # -- Checkpoint manager ----------------------------------------------------
    if getattr(args, "async_checkpoint", False):
        ckpt_mgr = AsyncCheckpointManager(
            checkpoint_dir=args.checkpoint_dir,
            rank=rank,
            keep_last_n=args.keep_checkpoints,
            shm_dir=getattr(args, "checkpoint_shm_dir", "/dev/shm/kotodama-ckpts"),
            compress=getattr(args, "checkpoint_compress", True),
        )
    else:
        ckpt_mgr = CheckpointManager(
            checkpoint_dir=args.checkpoint_dir,
            rank=rank,
            keep_last_n=args.keep_checkpoints,
        )

    # Resume from checkpoint if one exists
    start_step = 0
    tokens_consumed = 0
    ckpt_state = ckpt_mgr.load_latest(model, muon_opt, adamw_opt, device)
    if ckpt_state is not None and config.attn_res and config.attn_res_gain == "folded":
        raw_model.fold_attn_res_gains()  # a folded run's own checkpoint: a no-op (its gains are already 1)
    if ckpt_state is not None:
        start_step = ckpt_state["step"] + 1
        tokens_consumed = ckpt_state.get("tokens_consumed", 0)
        # Restore data loader position
        if "data" in ckpt_state and hasattr(dataset, "load_state_dict"):
            dataset.load_state_dict(ckpt_state["data"])
            data_iter = _make_data_iter()
        # Restore scheduler position
        scheduler.step(start_step)

    # -- Formation-ladder cadence: weights-only saves + protected full anchors --
    weight_saves: Optional[TokenThresholds] = None
    anchor_saves: Optional[TokenThresholds] = None
    anchor_mgr: Optional[CheckpointManager] = None
    if getattr(args, "save_schedule", None):
        weight_saves = TokenThresholds(parse_save_schedule(args.save_schedule, total_tokens))
        weight_saves.skip_through(tokens_consumed)
        if is_main:
            logger.info("Weights-only save schedule %r: %d thresholds -> %s/weights",
                        args.save_schedule, len(weight_saves), args.checkpoint_dir)
    if getattr(args, "anchor_tokens", None):
        anchor_saves = TokenThresholds(
            [parse_token_count(t) for t in args.anchor_tokens.split(",") if t.strip()])
        anchor_saves.skip_through(tokens_consumed)
        anchor_mgr = CheckpointManager(
            checkpoint_dir=Path(args.checkpoint_dir) / "anchors", rank=rank, keep_last_n=0)

    # -- SIGTERM handler -------------------------------------------------------
    sigterm = SIGTERMHandler()

    # -- Geometric monitoring (rank 0; every rank with --geo_monitor_sharded) ---
    monitor: Optional[GeometricMonitor] = None
    geo_sharded = bool(getattr(args, "geo_monitor_sharded", False) and dist.is_initialized()
                       and dist.get_world_size() > 1)
    if args.geo_monitor and (is_main or geo_sharded):
        geo_schedule_str = getattr(args, "geo_monitor_schedule", None)
        if geo_schedule_str:
            schedule = MonitorSchedule.from_string(geo_schedule_str)
            logger.info("Geo monitor schedule: %s", schedule.phases)
        else:
            schedule = MonitorSchedule.fixed(args.geo_monitor_tier1_every)
            logger.info("Geo monitor fixed cadence: every %d steps", args.geo_monitor_tier1_every)
        monitor_config = MonitorConfig(
            schedule=schedule,
            tier1_every=args.geo_monitor_tier1_every,
            tier2_every=args.geo_monitor_tier2_every,
            sharded=geo_sharded,
        )
        monitor = GeometricMonitor(model, monitor_config)

        # Build or restore fixed probe batch for longitudinal monitoring.
        # Restoring from checkpoint ensures metrics are comparable across
        # stop/resume cycles (TwoNN, RankMe are sensitive to input sequences).
        _restored_probe = False
        if ckpt_state is not None and "probe_batch" in ckpt_state:
            monitor.set_probe_batch(ckpt_state["probe_batch"])
            logger.info("Probe batch restored from checkpoint")
            _restored_probe = True

        if not _restored_probe and is_main:
            probe_ids: list[torch.Tensor] = []
            # The probe must not advance the training stream: rank 0 would skip
            # ahead of the other ranks, and the saved (rank-0) data position
            # would then shift every rank on resume.
            _pre_probe_state = dataset.state_dict() if hasattr(dataset, "state_dict") else None
            probe_iter = _make_data_iter()
            probe_tokens_needed = min(
                monitor_config.tier1_probe_size, 64
            )  # sequences, not tokens
            for _ in range(probe_tokens_needed):
                try:
                    sample = next(probe_iter)
                    if isinstance(sample, dict):
                        probe_ids.append(sample["input_ids"])
                    else:
                        probe_ids.append(sample)
                except StopIteration:
                    break
            del probe_iter
            if _pre_probe_state is not None:
                dataset.load_state_dict(_pre_probe_state)
                data_iter = _make_data_iter()
            if probe_ids:
                probe_batch = torch.cat(probe_ids, dim=0)[:probe_tokens_needed]
                monitor.set_probe_batch(probe_batch)
            logger.info("Geometric monitor ready (probe batch: %s)", tuple(probe_batch.shape))
        if geo_sharded:
            # Every rank probes its share of rank 0's probe batch (built above, or restored from the checkpoint).
            _probe_obj = [monitor._probe_batch.cpu() if is_main and monitor._probe_batch is not None else None]
            dist.broadcast_object_list(_probe_obj, src=0)
            if _probe_obj[0] is None:
                raise RuntimeError("--geo_monitor_sharded: rank 0 has no probe batch")
            monitor.set_probe_batch(_probe_obj[0])
            if is_main:
                logger.info("Geo monitor sharded over %d ranks", dist.get_world_size())

    # -- Wandb logger (rank 0 only) -------------------------------------------
    wb: Optional[WandbLogger] = None
    if is_main and args.wandb:
        wb = WandbLogger(
            project=args.wandb_project,
            run_name=args.wandb_run_name,
            config={
                "model_size": args.model_size,
                "total_tokens": total_tokens,
                "total_steps": total_steps,
                "grad_accum": grad_accum,
                "tokens_per_step": tokens_per_step,
                "micro_batch_size": micro_batch,
                "sequence_length": seq_len,
                "world_size": world_size,
                "muon_lr": args.muon_lr,
                "muon_ns_coefficients": args.muon_ns_coefficients or "original",
                "adamw_lr": args.adamw_lr,
                "warmup_steps": args.warmup_steps,
                "precision": "bf16",
                "parallelism": f"tp{tp_size}" if tp_size > 1 else "ddp",
                "tp_size": tp_size,
                "dp_size": dp_size,
                "activation_checkpointing": args.activation_checkpointing,
                "use_liger": getattr(args, "use_liger", False),
                "attn_impl": getattr(args, "attn_impl", "auto"),
            },
            log_dir=args.checkpoint_dir,
            tags=["pretraining", args.model_size],
        )

    # -- Profiler setup --------------------------------------------------------
    _profiler = None
    _profile_end_step = -1
    if getattr(args, "profile", False) and is_main:
        _profile_dir = Path(args.profile_dir or args.checkpoint_dir) / "profile"
        _profile_dir.mkdir(parents=True, exist_ok=True)
        _profile_start = args.profile_start_step
        _profile_end_step = _profile_start + args.profile_steps
        logger.info(
            "Profiler enabled: steps %d-%d, output: %s",
            _profile_start, _profile_end_step, _profile_dir,
        )

    # -- Eval helper -----------------------------------------------------------
    @torch.no_grad()
    def run_eval(step: int) -> Optional[float]:
        if eval_iter_fn is None or eval_every <= 0:
            return None
        model.eval()
        eval_it = eval_iter_fn()
        total_loss = 0.0
        count = 0
        for _ in range(eval_batches):
            try:
                batch = next(eval_it)
            except StopIteration:
                break
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
                cu_sq = batch.get("cu_seqlens", None)
                if cu_sq is not None:
                    cu_sq = cu_sq.to(device)
                ms = batch.get("max_seqlen", None)
                pos = batch.get("position_ids", None)
                if pos is not None:
                    pos = pos.to(device)
            else:
                input_ids = batch.to(device)
                cu_sq = ms = pos = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids, labels=input_ids,
                    cu_seqlens=cu_sq, max_seqlen=ms, position_ids=pos,
                )
            total_loss += output["loss"].item()
            count += 1
        model.train()
        if count == 0:
            return None
        avg = total_loss / count
        if is_main:
            logger.info("  eval: loss=%.4f ppl=%.2f (%d batches)", avg, math.exp(min(avg, 20)), count)
            if wb is not None:
                wb.log_custom(step, {"eval/loss": avg, "eval/perplexity": math.exp(min(avg, 20))})
            hm_log(step, **{"eval/loss": avg, "eval/perplexity": math.exp(min(avg, 20))})
        return avg

    # -- In-loop evals (live weights; every rank, deterministic schedule) --------
    inloop: Optional[InloopEvaluator] = None
    _inloop_cfg = InloopConfig.from_args(args)
    if _inloop_cfg.suites:
        if tp_size > 1:
            raise ValueError("--inloop_evals is not supported with --tp_size > 1 "
                             "(prompts are sharded across ranks; TP shards the model)")
        inloop = InloopEvaluator(
            _inloop_cfg, out_dir=args.checkpoint_dir, rank=rank, world_size=world_size,
            device=device, max_seq_len=config.max_position_embeddings,
            vocab_size=config.vocab_size, log_fn=hm_log)
        _cum_tokens = None
        if args.batch_ramp_pct > 0 and args.batch_ramp_initial < 1.0:
            _cum_tokens = list(itertools.accumulate(
                _compute_grad_accum_for_step(s, total_steps, grad_accum,
                                             args.batch_ramp_initial, args.batch_ramp_pct)
                * tokens_per_global_micro for s in range(total_steps)))
        inloop.plan(total_steps, decay_start_step(total_steps, args.decay_start_pct),
                    final_step=loop_end - 1, start_step=start_step, cum_tokens=_cum_tokens)

    # -- Data prefetcher ---------------------------------------------------------
    # Started AFTER resume + probe-batch setup: both iterate the same dataset
    # object, and the prefetch thread must not race their position advances.
    prefetcher: Optional[_PrefetchIterator] = None
    prefetch_depth = getattr(args, "prefetch_batches", 2)
    if prefetch_depth > 0:
        prefetcher = _PrefetchIterator(_make_data_iter, dataset, depth=prefetch_depth)
        data_iter = prefetcher
        if is_main:
            logger.info("Data prefetcher enabled (depth=%d)", prefetch_depth)

    def _current_data_state() -> Optional[dict[str, int]]:
        """Dataset state for checkpointing — from the prefetcher when active,
        so prefetched-but-unconsumed batches are not skipped on resume."""
        if prefetcher is not None:
            return prefetcher.consumed_state
        return dataset.state_dict() if hasattr(dataset, "state_dict") else None

    # -- FlexAttention doc-mask builder -------------------------------------------
    flex_mask_builder: Optional[_FlexBlockMaskBuilder] = None
    if getattr(args, "attn_impl", "auto") == "flex" and doc_masking:
        flex_mask_builder = _FlexBlockMaskBuilder(device)
        if is_main:
            logger.info("FlexAttention BlockMask builder enabled (doc masking)")

    # -- MFU accounting ----------------------------------------------------------
    flops_per_token = _model_flops_per_token(param_count, config, seq_len)
    peak_flops_total = world_size * getattr(args, "peak_tflops_per_gpu", 2250.0) * 1e12
    if is_main:
        logger.info(
            "MFU accounting: %.2e model FLOPs/token, peak %.1f TFLOPs/GPU x %d GPUs",
            flops_per_token, getattr(args, "peak_tflops_per_gpu", 2250.0), world_size,
        )

    # -- Training loop ---------------------------------------------------------
    if is_main:
        logger.info("Starting training from step %d", start_step)
        hm_config({
            **{k: (",".join(map(str, v)) if isinstance(v, (list, tuple)) else v)
               for k, v in vars(args).items()},
            "total_steps": total_steps, "grad_accum": grad_accum,
            "tokens_per_step": tokens_per_step, "micro_batch_size": micro_batch,
            "sequence_length": seq_len, "world_size": world_size,
            "dp_size": dp_size, "tp_size": tp_size, "param_count": param_count,
            "start_step": start_step,
        })

    model.train()
    step_t0 = time.time()
    log_loss_accum = 0.0
    log_z_loss_accum = 0.0
    log_steps = 0
    log_data_wait = 0.0

    # Pre-allocate loss accumulators on GPU to avoid .item() sync per micro-step
    _loss_accum = torch.zeros(1, device=device)
    _z_loss_accum = torch.zeros(1, device=device)
    # Log-window extremes: loss/grad_norm are logged every log_every steps, so a
    # one-step spike between log steps would otherwise never reach telemetry.
    # Kept on device (no extra sync); read once per log step.
    _gn_win_max = torch.zeros((), device=device)
    _gn_win_clipped = torch.zeros((), device=device)
    _loss_win_max = -math.inf

    # Opt-in data-path diagnostic: token-ID hashes of each rank-0 micro-batch.
    _debug_hash_steps = int(os.environ.get("KOTODAMA_DEBUG_BATCH_HASH", "0") or 0)

    for step in range(start_step, loop_end):
        # Start profiler at the configured step
        if _profiler is None and is_main and step == getattr(args, "profile_start_step", -1) and getattr(args, "profile", False):
            _profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_flops=True,
                profile_memory=True,
            )
            _profiler.__enter__()
            logger.info("Profiler started at step %d", step)
        _loss_accum.zero_()
        _z_loss_accum.zero_()

        # Batch size ramp: dynamically adjust grad_accum
        current_grad_accum = _compute_grad_accum_for_step(
            step, total_steps, grad_accum,
            initial_batch_fraction=args.batch_ramp_initial,
            ramp_pct=args.batch_ramp_pct,
        )

        if micro_stats is not None:
            micro_stats.begin(current_grad_accum)
        for micro_step in range(current_grad_accum):
            _t_data = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = _make_data_iter()
                batch = next(data_iter)
            log_data_wait += time.perf_counter() - _t_data

            if _debug_hash_steps and step < _debug_hash_steps and is_main:
                _ids = batch["input_ids"] if isinstance(batch, dict) else batch
                _h = hashlib.md5(_ids.cpu().numpy().tobytes()).hexdigest()[:12]
                logger.info("batch-hash step=%d micro=%d shape=%s md5=%s",
                            step, micro_step, tuple(_ids.shape), _h)

            # Unpack: doc_masking yields dict, legacy yields tensor
            if doc_masking:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                cu_seqlens = batch["cu_seqlens"].to(device, non_blocking=True)
                position_ids = batch["position_ids"].to(device, non_blocking=True)
                max_seqlen_batch = batch["max_seqlen"]
                if varlen_dynamic:
                    # One dynamic compile instead of a static one plus a recompile per varying input: the doc
                    # count is dynamic from the first trace, and max_seqlen is the static upper bound (exact for
                    # the varlen kernels: scripts/utils/fa4_maxseqlen_parity.py) instead of a per-batch int whose
                    # first new value recompiles each rank at a different step.
                    torch._dynamo.maybe_mark_dynamic(cu_seqlens, 0)
                    max_seqlen_batch = seq_len
            else:
                input_ids = batch.to(device, non_blocking=True)
                cu_seqlens = None
                position_ids = None
                max_seqlen_batch = None

            block_mask = (
                flex_mask_builder.build(position_ids)
                if flex_mask_builder is not None and position_ids is not None
                else None
            )

            # Skip gradient sync on all but the last micro-step.
            # no_sync() only exists on DDP-wrapped models.
            sync_ctx = (
                model.no_sync()
                if hasattr(model, "no_sync") and micro_step < current_grad_accum - 1
                else nullcontext()
            )

            with sync_ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(
                        input_ids,
                        labels=input_ids,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen_batch,
                        position_ids=position_ids,
                        block_mask=block_mask,
                    )
                    loss = output["loss"] / current_grad_accum

                loss.backward()

            if micro_stats is not None:
                micro_stats.after_micro(micro_step, output["loss"])
            _loss_accum += output["loss"].detach()
            if "z_loss" in output:
                _z_loss_accum += output["z_loss"].detach()

        # Single GPU→CPU sync after all micro-steps
        step_loss = _loss_accum.item() / current_grad_accum
        step_z_loss = _z_loss_accum.item() / current_grad_accum

        # TP: sync QK-norm gradients (they see sharded heads, need SUM)
        if tp_group is not None and dp_size == 1:
            from kotodama.training.tensor_parallel import sync_tp_replicated_grads
            sync_tp_replicated_grads(raw_model, tp_group)

        # Gradient clipping — TP-aware norm when TP is active
        if args.gradient_clip > 0:
            if tp_group is not None:
                from kotodama.training.tensor_parallel import tp_clip_grad_norm
                grad_norm = tp_clip_grad_norm(model, args.gradient_clip, tp_group)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clip
                )
        else:
            grad_norm = _compute_grad_norm(model.parameters())

        _gn_t = grad_norm if isinstance(grad_norm, torch.Tensor) else torch.tensor(float(grad_norm), device=device)
        _gn_win_max = torch.maximum(_gn_win_max, _gn_t.detach().float())
        if args.gradient_clip > 0:
            _gn_win_clipped = _gn_win_clipped + (_gn_t.detach().float() > args.gradient_clip).float()

        # Per-micro grad stats; on a spike, the PRE-update weights + the step's record (before the optimizer step)
        if micro_stats is not None:
            _micro_rec = micro_stats.finish(step, float(_gn_t))
            if is_main:
                micro_stats.write(_micro_rec)
            if spike_det is not None:
                _spike = torch.tensor([float(is_main and spike_det.update(float(_gn_t)))], device=device)
                if dist.is_available() and dist.is_initialized():
                    dist.broadcast(_spike, src=0)  # one decision for every rank (the save below barriers)
                if _spike.item() > 0:
                    _sdir = Path(args.checkpoint_dir) / "spikes"
                    if is_main:
                        logger.warning("grad-norm spike at step %d: %.3f > %.1f x median %.3f — saving pre-step "
                                       "weights to %s", step, float(_gn_t), spike_det.factor,
                                       spike_det.last_median, _sdir)
                        _sdir.mkdir(parents=True, exist_ok=True)
                        (_sdir / f"step_{step:08d}.json").write_text(json.dumps(
                            {**_micro_rec, "median": spike_det.last_median, "factor": spike_det.factor,
                             "tokens_consumed": tokens_consumed}))
                    save_weights_only(_sdir, step, model, tokens_consumed, rank=rank)

        # Per-layer gradient analysis (before optimizer step clears grads)
        if wb is not None and step % 50 == 0:
            grad_metrics = _per_layer_grad_norms(raw_model)
            wb.log_custom(step, grad_metrics)
        _ng_every_h = getattr(args, "norm_gain_every", 0)
        if is_main and _ng_every_h > 0 and (step % _ng_every_h == 0 or step == loop_end - 1):
            try:
                hm_log(step, **_layer_health_metrics(raw_model))
            except Exception as e:  # telemetry must never stop a run (e.g. sharded params)
                if not getattr(_layer_health_metrics, "_warned", False):
                    logger.warning("layer health metrics disabled: %s: %s", type(e).__name__, e)
                    _layer_health_metrics._warned = True  # type: ignore[attr-defined]

        # EoS sharpness (needs live gradients, must be before zero_grad)
        if monitor is not None and monitor.should_monitor(step):
            eos_metrics = monitor.compute_sharpness(step)
            if eos_metrics and is_main:
                if wb is not None:
                    wb.log_geo(step, eos_metrics)
                hm_log(step, **{k: v for k, v in eos_metrics.items() if isinstance(v, (int, float))})

        # LR for THIS step, set before the optimizer uses it: stepping the
        # scheduler afterwards ran step 0 at the full base LR (an unwarmed step
        # on random init) and every later step one multiplier behind.
        scheduler.step(step)

        # Optimizer step
        muon_opt.step()
        adamw_opt.step()
        muon_opt.zero_grad(set_to_none=True)
        adamw_opt.zero_grad(set_to_none=True)

        # Track tokens (use current_grad_accum for accurate counting)
        current_tokens_per_step = current_grad_accum * tokens_per_global_micro
        tokens_consumed += current_tokens_per_step

        # -- Logging -----------------------------------------------------------
        log_loss_accum += step_loss
        log_z_loss_accum += step_z_loss
        _loss_win_max = max(_loss_win_max, step_loss)
        log_steps += 1

        if is_main and (step % args.log_every == 0 or step == loop_end - 1):
            elapsed = time.time() - step_t0
            avg_loss = log_loss_accum / max(log_steps, 1)
            avg_z_loss = log_z_loss_accum / max(log_steps, 1)
            tokens_per_sec = (log_steps * current_tokens_per_step) / max(elapsed, 1e-6)
            iters_per_sec = log_steps / max(elapsed, 1e-6)
            lrs = scheduler.get_last_lr()
            gpu_mem = torch.cuda.max_memory_allocated(device) / 1e9
            _grad_norm_scalar = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            mfu = flops_per_token * tokens_per_sec / peak_flops_total
            data_wait_per_step = log_data_wait / max(log_steps, 1)

            logger.info(
                "step=%d/%d | loss=%.4f | z_loss=%.6f | grad_norm=%.3f | "
                "muon_lr=%.6f | adamw_lr=%.6f | tok/s=%.0f | it/s=%.2f | "
                "mfu=%.1f%% | data_wait=%.3fs | tokens=%.3fB | gpu_mem=%.1fGB",
                step,
                total_steps,
                avg_loss,
                avg_z_loss,
                _grad_norm_scalar,
                lrs["muon_lr"],
                lrs["adamw_lr"],
                tokens_per_sec,
                iters_per_sec,
                mfu * 100.0,
                data_wait_per_step,
                tokens_consumed / 1e9,
                gpu_mem,
            )

            if wb is not None:
                wb.log_step(
                    step=step,
                    loss=avg_loss,
                    z_loss=avg_z_loss,
                    grad_norm=_grad_norm_scalar,
                    muon_lr=lrs["muon_lr"],
                    adamw_lr=lrs["adamw_lr"],
                    tokens_per_sec=tokens_per_sec,
                    iters_per_sec=iters_per_sec,
                    tokens_consumed=tokens_consumed,
                    gpu_mem_gb=gpu_mem,
                    step_time_s=elapsed / max(log_steps, 1),
                )
                wb.log_custom(step, {
                    "perf/mfu": mfu,
                    "perf/data_wait_s_per_step": data_wait_per_step,
                })

            if is_main:
                _gn_max_s, _gn_clip_s = torch.stack([_gn_win_max, _gn_win_clipped]).tolist()
                hm_log(
                    step,
                    **{"train/loss": avg_loss, "train/grad_norm": _grad_norm_scalar,
                       "train/perplexity": math.exp(min(avg_loss, 20)),
                       "train/loss_max": _loss_win_max,
                       "train/grad_norm_max": _gn_max_s,
                       "train/grad_clip_frac": _gn_clip_s / max(log_steps, 1),
                       "train/z_loss": avg_z_loss,
                       **(_micro_summary(_micro_rec) if _micro_rec is not None else {}),
                       "train/muon_lr": lrs["muon_lr"],
                       "train/adamw_lr": lrs["adamw_lr"],
                       "tokens_seen": float(tokens_consumed),
                       "perf/tokens_per_sec": tokens_per_sec,
                       "perf/iters_per_sec": iters_per_sec,
                       "perf/step_time_s": elapsed / max(log_steps, 1),
                       "perf/data_wait_s_per_step": data_wait_per_step,
                       "perf/gpu_mem_gb": gpu_mem,
                       "perf/mfu": mfu},
                )

            _ng_every = getattr(args, "norm_gain_every", 0)
            if _ng_every > 0 and (step % _ng_every == 0 or step == loop_end - 1):
                hm_log(step, **_norm_gain_metrics(raw_model))

            # Reset accumulators
            log_loss_accum = 0.0
            log_z_loss_accum = 0.0
            _gn_win_max = torch.zeros((), device=device)
            _gn_win_clipped = torch.zeros((), device=device)
            _loss_win_max = -math.inf
            log_steps = 0
            log_data_wait = 0.0
            step_t0 = time.time()

        # -- Geometric monitoring (rank 0; every rank when sharded) ------------
        if monitor is not None and monitor.should_monitor(step):
            geo_metrics = monitor.compute_all(step)
            if geo_metrics and is_main:
                if wb is not None:
                    wb.log_geo(step, geo_metrics)
                hm_log(step, **{k: v for k, v in geo_metrics.items() if isinstance(v, (int, float))})
                if step % args.log_every == 0:
                    logger.info(
                        "  geo: RankMe=%.1f, σ_max=%.4f, time=%.1fs",
                        geo_metrics.get("geo/rankme_last", 0),
                        geo_metrics.get("eoc/sigma_max", 0),
                        geo_metrics.get("geo/compute_all_time_s", 0),
                    )

        # -- Profiler step/stop ------------------------------------------------
        if _profiler is not None:
            _profiler.step()
            if step >= _profile_end_step:
                _profiler.__exit__(None, None, None)
                # Write chrome trace
                trace_path = _profile_dir / "chrome_trace.json"
                _profiler.export_chrome_trace(str(trace_path))
                # Write summary table
                summary_path = _profile_dir / "profile_summary.txt"
                summary = _profiler.key_averages().table(
                    sort_by="cuda_time_total", row_limit=60,
                )
                flops = _profiler.key_averages().table(
                    sort_by="flops", row_limit=20,
                )
                with open(summary_path, "w") as f:
                    f.write(summary)
                    f.write("\n\n--- FLOPS ---\n\n")
                    f.write(flops)
                logger.info(
                    "Profiler stopped at step %d. Trace: %s, Summary: %s",
                    step, trace_path, summary_path,
                )
                _profiler = None

        # -- Checkpoint --------------------------------------------------------
        should_save = (
            step > 0
            and args.save_every > 0
            and step % args.save_every == 0
        )

        if should_save:
            _spread = param_checksum_spread(raw_model)
            if is_main:
                hm_log(step, **{"dist/param_checksum_spread": _spread})
                if _spread != 0.0:
                    logger.error("RANK DESYNC at step %d: weight checksum spread %.6e "
                                 "across DDP ranks (replicas must be bitwise identical)",
                                 step, _spread)
            data_state = _current_data_state()
            _extra: dict[str, Any] = {}
            if monitor is not None and monitor._probe_batch is not None:
                _extra["probe_batch"] = monitor._probe_batch.cpu()
            ckpt_mgr.save(
                step=step,
                model=model,
                muon_opt=muon_opt,
                adamw_opt=adamw_opt,
                scheduler=scheduler,
                tokens_consumed=tokens_consumed,
                data_state=data_state,
                extra=_extra or None,
            )

        if weight_saves is not None and weight_saves.due(tokens_consumed):
            save_weights_only(Path(args.checkpoint_dir) / "weights", step, model,
                              tokens_consumed, rank=rank)
        if anchor_saves is not None and anchor_mgr is not None and anchor_saves.due(tokens_consumed):
            _extra_anchor: dict[str, Any] = {}
            if monitor is not None and monitor._probe_batch is not None:
                _extra_anchor["probe_batch"] = monitor._probe_batch.cpu()
            anchor_mgr.save(
                step=step, model=model, muon_opt=muon_opt, adamw_opt=adamw_opt,
                scheduler=scheduler, tokens_consumed=tokens_consumed,
                data_state=_current_data_state(), extra=_extra_anchor or None,
            )

        # -- In-loop evals (weights == checkpoint step_<step>) -------------------
        if inloop is not None and inloop.due(step):
            _il = inloop.run(step, model, tokens_consumed, stop_requested=sigterm.received)
            if _il and wb is not None:
                wb.log_custom(step, _il)

        # -- Eval on held-out data ---------------------------------------------
        if eval_every > 0 and step > 0 and step % eval_every == 0:
            run_eval(step)

        # Commit wandb step (after all log calls including eval)
        if wb is not None:
            wb.commit(step)

        # -- SIGTERM check -----------------------------------------------------
        if sigterm.received:
            if args.save_every <= 0:
                if is_main:
                    logger.info("SIGTERM exit — checkpoint skipped (save_every<=0)")
                break
            if is_main:
                logger.info("SIGTERM exit — saving checkpoint at step %d", step)
            data_state = _current_data_state()
            # Use blocking save for SIGTERM (async may not finish before SIGKILL)
            save_fn = (
                ckpt_mgr.save_blocking
                if hasattr(ckpt_mgr, "save_blocking")
                else ckpt_mgr.save
            )
            _extra_sigterm: dict[str, Any] = {}
            if monitor is not None and monitor._probe_batch is not None:
                _extra_sigterm["probe_batch"] = monitor._probe_batch.cpu()
            save_fn(
                step=step,
                model=model,
                muon_opt=muon_opt,
                adamw_opt=adamw_opt,
                scheduler=scheduler,
                tokens_consumed=tokens_consumed,
                data_state=data_state,
                extra=_extra_sigterm or None,
            )
            break

    # -- Final -----------------------------------------------------------------
    # All ranks must participate in save (it contains a barrier).
    # save_every <= 0 disables ALL saves (benchmark/canary runs): a 7B final
    # checkpoint is ~30GB+, more than a small root filesystem holds, and an
    # ENOSPC on that final save crashes a run that already completed.
    if not sigterm.received and args.save_every <= 0:
        if is_main:
            logger.info(
                "Training complete: %d steps, %.2fB tokens (no checkpoint — save_every<=0)",
                total_steps, tokens_consumed / 1e9,
            )
    elif not sigterm.received:
        if is_main:
            logger.info(
                "Training complete: %d steps%s, %.2fB tokens",
                loop_end,
                f" (stopped early; schedule {total_steps})" if loop_end < total_steps else "",
                tokens_consumed / 1e9,
            )
        data_state = _current_data_state()
        _extra_final: dict[str, Any] = {}
        if monitor is not None and monitor._probe_batch is not None:
            _extra_final["probe_batch"] = monitor._probe_batch.cpu()
        ckpt_mgr.save(
            step=loop_end - 1,
            model=model,
            muon_opt=muon_opt,
            adamw_opt=adamw_opt,
            scheduler=scheduler,
            tokens_consumed=tokens_consumed,
            data_state=data_state,
            extra=_extra_final or None,
        )

    # Stop the prefetch thread before teardown
    if prefetcher is not None:
        prefetcher.shutdown()

    # Flush async checkpoint queue before shutting down
    if hasattr(ckpt_mgr, "shutdown"):
        ckpt_mgr.shutdown()

    # Upload final checkpoint to HuggingFace
    if is_main and not sigterm.received and getattr(args, "hf_upload_repo", None):
        _upload_checkpoint_to_hf(
            repo_id=args.hf_upload_repo,
            checkpoint_dir=Path(args.checkpoint_dir),
            step=loop_end - 1,
            tokens_consumed=tokens_consumed,
            config_file=getattr(args, "_config_file", None),
        )

    if wb is not None:
        wb.finish()
    sigterm.restore()
    cleanup_distributed()


def parse_routing_spec(spec: str, num_layers: int) -> tuple[list[int], bool]:
    """Parse a ``--reset_routing_layers`` spec: comma-separated layer indices,
    inclusive ranges ``a-b``, and ``final`` (the final readout). E.g.
    ``"24-27,final"`` -> ([24, 25, 26, 27], True)."""
    layers: set[int] = set()
    final = False
    for raw in spec.split(","):
        tok = raw.strip()
        if not tok:
            continue
        if tok == "final":
            final = True
            continue
        try:
            if "-" in tok:
                lo_s, hi_s = tok.split("-", 1)
                lo, hi = int(lo_s), int(hi_s)
                if lo > hi:
                    raise ValueError
                layers.update(range(lo, hi + 1))
            else:
                layers.add(int(tok))
        except ValueError:
            raise ValueError(f"--reset_routing_layers: bad token {tok!r} in {spec!r}") from None
    bad = sorted(i for i in layers if not 0 <= i < num_layers)
    if bad:
        raise ValueError(f"--reset_routing_layers: layers out of range 0..{num_layers - 1}: {bad}")
    if not layers and not final:
        raise ValueError(f"--reset_routing_layers: empty spec {spec!r}")
    return sorted(layers), final


def _loop_end_step(total_steps: int, stop_at_step: Optional[int]) -> int:
    """Exclusive end of the step loop. ``--stop_at_step N`` runs steps < N (final
    checkpoint saved as step N-1) while the LR/batch schedule still spans
    ``total_steps`` — unlike ``--total_steps``, which redefines the schedule."""
    if stop_at_step is None:
        return total_steps
    if stop_at_step <= 0:
        raise ValueError(f"--stop_at_step must be > 0, got {stop_at_step}")
    return min(total_steps, stop_at_step)


def _compute_grad_accum_for_step(
    step: int,
    total_steps: int,
    base_grad_accum: int,
    initial_batch_fraction: float,
    ramp_pct: float,
) -> int:
    """
    Compute gradient accumulation steps with batch size ramp.

    Starts at ``base_grad_accum * initial_batch_fraction`` and linearly
    ramps to ``base_grad_accum`` over the first ``ramp_pct`` of training.

    Spec: 512K → 2M tokens over first 5% (initial_batch_fraction=0.25, ramp_pct=0.05).
    """
    if ramp_pct <= 0 or initial_batch_fraction >= 1.0:
        return base_grad_accum

    ramp_steps = int(total_steps * ramp_pct)
    if step >= ramp_steps:
        return base_grad_accum

    # Linear ramp from initial to full
    progress = step / max(ramp_steps, 1)
    fraction = initial_batch_fraction + (1.0 - initial_batch_fraction) * progress
    return max(1, int(base_grad_accum * fraction))


class _NoOpOptimizer:
    """Dummy optimizer that does nothing. Used when --adamw_only is set."""

    param_groups: list[dict[str, Any]] = []
    defaults: dict[str, Any] = {"lr": 0.0}

    def step(self, closure: Any = None) -> None:
        pass

    def zero_grad(self, set_to_none: bool = False) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        pass


def _compute_grad_norm(params: Any) -> torch.Tensor:
    """Compute total gradient norm on GPU (single sync when .item() is called)."""
    norms = [p.grad.data.float().norm() for p in params if p.grad is not None]
    if not norms:
        return torch.tensor(0.0)
    return torch.stack(norms).square().sum().sqrt()


def param_checksum_spread(model: torch.nn.Module) -> float:
    """max - min over ranks of a float64 weight checksum. DDP replicas must be
    bitwise identical, so anything but 0.0 is a silent rank desync (e.g. an
    in-place weight perturbation that does not restore exactly). 0.0 when not distributed. Collective: every
    rank must call it."""
    total = torch.zeros((), dtype=torch.float64)
    for p in model.parameters():
        total = total + p.detach().double().sum().cpu()
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() < 2:
        return 0.0
    dev = next(model.parameters()).device
    hi = total.to(dev).clone()
    lo = total.to(dev).clone()
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    return (hi - lo).item()


_LAYER_RE = re.compile(r"layers\.(\d+)\.")
# An AttnRes routing query: plain (``attn_res_query``) or UnitRoutingQuery's direction / temperature.
_ROUTER_RE = re.compile(r"(attn|mlp|final)_res_query(\.original|\.0\.temperature)?$")


def _norm_gain_metrics(model: torch.nn.Module) -> dict[str, float]:
    """Mean/std of every RMSNorm/QK-norm gain, as telemetry layer families:
    ``norm_gain/layer_{i}/{kind}`` (mean) and ``norm_gain_std/layer_{i}/{kind}``.
    Non-layer norms land under ``norm_gain/{kind}``. One host sync."""
    keys: list[str] = []
    stats: list[torch.Tensor] = []
    for name, p in model.named_parameters():
        if p.ndim != 1 or not name.endswith("norm.weight"):
            continue
        kind = name.split(".")[-2]
        m = _LAYER_RE.search(name)
        where = f"layer_{m.group(1)}/{kind}" if m else ("final_norm" if kind == "norm" else kind)
        g = p.detach().float()
        keys += [f"norm_gain/{where}", f"norm_gain_std/{where}"]
        stats += [g.mean(), g.std() if g.numel() > 1 else g.new_zeros(())]
    if not stats:
        return {}
    return dict(zip(keys, torch.stack(stats).tolist()))


def _layer_health_metrics(model: torch.nn.Module) -> dict[str, float]:
    """Per-layer weight scale, gradient norms and AttnRes router health; one host sync.

    Call after backward (and clipping) and before ``zero_grad`` — it reads ``.grad``.

    - ``weight_rms/layer_{i}/{q,k,v,o,gate,up,down}_proj`` (+ ``weight_rms/embed``): the
      Muon weight-decay equilibrium read (without live WD, weights grow and act as an
      implicit LR anneal).
    - ``grad/layer_{i}/{attn,mlp}``: grad norm of the sublayer's projection matrices
      (post-clip; the routing queries are counted under ``router/``, not here).
    - ``router/layer_{i}/{attn,mlp}/{query_norm,grad_norm}`` and ``router/final/…``:
      the AttnRes routing queries. ``grad_norm == 0`` exactly is the dead-router signature;
      ``router/dead_sites`` counts them (queries with no ``.grad`` — frozen or
      structurally unused, like layer 0's pre-attn — are not sites).
    """
    keys: list[str] = []
    vals: list[torch.Tensor] = []
    grad_sq: dict[str, list[torch.Tensor]] = {}
    router_grad_sq: dict[str, list[torch.Tensor]] = {}
    for name, p in model.named_parameters():
        m = _LAYER_RE.search(name)
        leaf = name.split(".")[-2] if name.endswith(".weight") else name.split(".")[-1]
        rq = _ROUTER_RE.search(name)
        if rq:
            where = f"layer_{m.group(1)}/{rq.group(1)}" if m and rq.group(1) != "final" else "final"
            # A unit-norm query's magnitude is its temperature (|q| of the direction is not a routing quantity).
            if rq.group(2) != ".original":
                keys.append(f"router/{where}/query_norm"); vals.append(p.detach().float().norm())
            if p.grad is not None:
                router_grad_sq.setdefault(where, []).append(p.grad.detach().float().pow(2).sum())
            continue
        if p.ndim != 2:
            continue
        if m and leaf.endswith("_proj"):
            keys.append(f"weight_rms/layer_{m.group(1)}/{leaf}")
            vals.append(p.detach().float().pow(2).mean().sqrt())
            if p.grad is not None:
                sub = "attn" if ".attn." in name else "mlp"
                grad_sq.setdefault(f"grad/layer_{m.group(1)}/{sub}", []).append(
                    p.grad.detach().float().pow(2).sum())
        elif name.endswith("embed_tokens.weight"):
            keys.append("weight_rms/embed"); vals.append(p.detach().float().pow(2).mean().sqrt())
    for k, parts in grad_sq.items():
        keys.append(k); vals.append(torch.stack(parts).sum().sqrt())
    router_grads: list[torch.Tensor] = []
    for where, parts in router_grad_sq.items():
        g = torch.stack(parts).sum().sqrt()
        keys.append(f"router/{where}/grad_norm"); vals.append(g)
        router_grads.append(g)
    if router_grads:
        keys.append("router/dead_sites")
        vals.append((torch.stack(router_grads) == 0).sum().float())
    if not vals:
        return {}
    return dict(zip(keys, torch.stack(vals).tolist()))


def _per_layer_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    """Compute gradient norms per transformer layer (attn vs MLP split)."""
    metrics: dict[str, float] = {}

    if not hasattr(model, "layers"):
        return metrics

    for i, layer in enumerate(model.layers):
        if i % 4 != 0 and i != len(model.layers) - 1:
            continue

        attn_norms = []
        mlp_norms = []

        for name, param in layer.named_parameters():
            if param.grad is None:
                continue
            norm = param.grad.data.float().norm()
            if "attn" in name:
                attn_norms.append(norm)
            elif "ffn" in name:
                mlp_norms.append(norm)

        attn_total = torch.stack(attn_norms).square().sum().sqrt().item() if attn_norms else 0.0
        mlp_total = torch.stack(mlp_norms).square().sum().sqrt().item() if mlp_norms else 0.0
        metrics[f"grad/layer_{i}/attn"] = attn_total
        metrics[f"grad/layer_{i}/mlp"] = mlp_total

        if mlp_total > 1e-10:
            metrics[f"grad/layer_{i}/attn_mlp_ratio"] = attn_total / (mlp_total + 1e-10)

    return metrics


# =============================================================================
# CLI
# =============================================================================


from kotodama.yaml_config import ConfigError  # noqa: E402,F401  (re-exported for config callers)
from kotodama.yaml_config import load_yaml_config as _load_yaml_config  # noqa: E402


def parse_layer_range(spec: Optional[str]) -> Optional[set[int]]:
    """``"0-6,20"`` -> {0..6, 20}; empty/None -> None (all layers)."""
    if not spec:
        return None
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            if b < a:
                raise ValueError(f"bad layer range {part!r}")
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    return out


def _attn_lr_scaled(args: argparse.Namespace) -> bool:
    """True iff the run scales the attention projections' LR for a window (needs their own Muon group)."""
    return (getattr(args, "attn_lr_scale", 1.0) != 1.0 and getattr(args, "attn_lr_scale_steps", 0) > 0)


def training_data_seed(args: argparse.Namespace) -> int:
    """The training-data order seed: ``--data_seed`` if given, else ``--seed`` (tied by default)."""
    ds = getattr(args, "data_seed", None)
    return int(args.seed if ds is None else ds)


def apply_attnres_eps_compat(args: argparse.Namespace) -> bool:
    """``--attnres_eps_fp64_compat``: the AttnRes forward kernels add the runtime eps, which inductor types as fp64
    inside a compiled graph (rms / logits / softmax in fp64), instead of the constexpr one (fp32 under any launcher).
    For continuing a lineage trained before the constexpr existed. Also enabled by
    KOTODAMA_ATTNRES_EPS_FP64_COMPAT=1. Returns whether compat is on."""
    from kotodama.model.flash_attn_res.ops import phase_1 as p1ops
    from kotodama.model.flash_attn_res.ops import phase_2 as p2ops

    if getattr(args, "attnres_eps_fp64_compat", False):
        p1ops.EPS_FP64_COMPAT = True
        p2ops.EPS_FP64_COMPAT = True
    return bool(p1ops.EPS_FP64_COMPAT and p2ops.EPS_FP64_COMPAT)


def apply_attnres_stack_grad_chain(args: argparse.Namespace) -> bool:
    """``--attnres_stack_grad_chain``: the sublayer routes sharing a committed AttnRes stack pass its gradient along
    a chain (each route's backward adds its contribution in-kernel) instead of each returning a full stack gradient
    that autograd sums. Same math; the sum's order and rounding differ, so a lineage keeps the choice it started
    with. Also enabled by KOTODAMA_ATTNRES_STACK_GRAD_CHAIN=1. Returns whether the chain is on."""
    from kotodama.model import llama

    if getattr(args, "attnres_stack_grad_chain", False):
        llama._STACK_GRAD_CHAIN = True
    return bool(llama._STACK_GRAD_CHAIN)


def _micro_summary(rec: dict[str, Any]) -> dict[str, float]:
    """grad_stats.summarize, never raising (telemetry must not stop a run)."""
    from kotodama.training.grad_stats import summarize

    try:
        return summarize(rec)
    except (ValueError, ZeroDivisionError, KeyError) as e:
        logger.debug("micro grad summary skipped: %s", e)
        return {}


def ddp_options(args: argparse.Namespace, find_unused: bool) -> dict[str, Any]:
    """DistributedDataParallel keyword options from the run flags.

    ``broadcast_buffers`` is off under ``--ddp_no_broadcast_buffers``: the model's buffers (RoPE tables, AttnRes
    masks) are constants built identically on every rank, so the per-forward rank-0 broadcast is redundant.
    ``bucket_cap_mb`` is set only when ``--ddp_bucket_cap_mb`` is given (else DDP's own default)."""
    opts: dict[str, Any] = dict(
        gradient_as_bucket_view=True,
        find_unused_parameters=find_unused,
        broadcast_buffers=not getattr(args, "ddp_no_broadcast_buffers", False),
    )
    if getattr(args, "ddp_bucket_cap_mb", None):
        opts["bucket_cap_mb"] = float(args.ddp_bucket_cap_mb)
    return opts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="kotodama distributed training")

    # Config file (YAML defaults, CLI overrides)
    p.add_argument(
        "--config", type=str, default=None,
        help="YAML config file. CLI flags override config values.",
    )

    # Model
    p.add_argument(
        "--model_size",
        type=str,
        default="proxy",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model size preset",
    )
    p.add_argument("--z_loss_weight", type=float, default=1e-5)
    p.add_argument(
        "--activation_checkpointing",
        action="store_true",
        help="Enable activation checkpointing (saves memory, required for full 3B)",
    )
    p.add_argument(
        "--no_activation_checkpointing", dest="activation_checkpointing",
        action="store_false",
        help="Disable activation checkpointing even when the YAML enables it "
             "(needed for --ac_budget arms)",
    )

    # Block Attention Residuals (Moonshot, 2026)
    p.add_argument(
        "--attn_res",
        action="store_true",
        help="Enable Block Attention Residuals (learned depth-wise routing)",
    )
    p.add_argument(
        "--no_attn_res", dest="attn_res", action="store_false",
        help="Turn off AttnRes set by a config: a plain residual stack of the same shape (controls).",
    )
    p.add_argument(
        "--attn_res_n_blocks",
        type=int,
        default=7,
        help="Number of AttnRes blocks (default 7 for 28 layers = blocks of 4)",
    )
    p.add_argument(
        "--attn_res_boundaries",
        type=str,
        default=None,
        help="Explicit block boundary layers as comma-separated ints (e.g. '0,3,7,12,21,25'). Overrides --attn_res_n_blocks.",
    )
    p.add_argument(
        "--attn_res_freeze_unused", action="store_true", default=True,
        help="Freeze the first boundary layer's never-used pre-attn routing params "
             "so DDP can run with find_unused_parameters=False (default: on)",
    )
    p.add_argument(
        "--no_attn_res_freeze_unused", dest="attn_res_freeze_unused", action="store_false",
        help="Disable routing-param freeze — required when resuming a checkpoint whose "
             "optimizer state was saved without it (AdamW param-count mismatch otherwise)",
    )

    # Training scale
    p.add_argument("--total_tokens", type=int, default=6_000_000_000)
    p.add_argument(
        "--total_steps",
        type=int,
        default=None,
        help="Override total_tokens with explicit step count",
    )
    p.add_argument(
        "--stop_at_step",
        type=int,
        default=None,
        help="Exit (final checkpoint) after step N-1 WITHOUT changing the LR/batch "
             "schedule (--total_steps redefines it) — short arms of a long recipe",
    )
    p.add_argument("--sequence_length", type=int, default=2048)
    p.add_argument("--micro_batch_size", type=int, default=8)
    p.add_argument(
        "--global_batch_tokens",
        type=int,
        default=2_097_152,
        help="Global batch size in tokens (default 2M)",
    )
    p.add_argument("--gradient_clip", type=float, default=1.0)

    # Batch size ramp (e.g. 512K → 2M over the first 5%)
    p.add_argument(
        "--batch_ramp_initial",
        type=float,
        default=1.0,
        help="Initial batch size as fraction of full (0.25 = 1/4, 1.0 = no ramp)",
    )
    p.add_argument(
        "--batch_ramp_pct",
        type=float,
        default=0.0,
        help="Fraction of training to ramp over (0.05 = first 5%%, 0.0 = no ramp)",
    )

    # Optimizer mode
    p.add_argument(
        "--adamw_only",
        action="store_true",
        help="Use pure AdamW (no Muon) — for baseline comparison",
    )

    # Muon optimizer
    p.add_argument("--muon_lr", type=float, default=0.03)
    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument("--muon_weight_decay", type=float, default=0.01)
    p.add_argument("--muon_ns_iterations", type=int, default=5)
    p.add_argument(
        "--muon_ns_coefficients", type=str, default=None,
        choices=["original", "gram_ns", "polar_express"],
        help="NS coefficient preset. 'original': fixed MoonshotAI coefficients. "
             "'gram_ns': per-iteration optimized (Dao-AILab Gram-Newton-Schulz). "
             "'polar_express': conservative per-iteration (Dao-AILab). "
             "Default: original.",
    )
    p.add_argument(
        "--muon_distributed", action="store_true",
        help="Partition Newton-Schulz across DDP ranks and broadcast results "
             "(exact same math, ~1/world_size the NS compute). No effect on "
             "single GPU or with TP.",
    )

    # AdamW optimizer
    p.add_argument("--adamw_lr", type=float, default=6e-4)
    p.add_argument("--adamw_beta1", type=float, default=0.9)
    p.add_argument("--adamw_beta2", type=float, default=0.95)
    p.add_argument("--adamw_weight_decay", type=float, default=0.1)
    p.add_argument(
        "--norm_gain_every", type=int, default=0,
        help="Log RMSNorm/QK-norm gain mean/std per layer to the scheduler telemetry every N "
             "steps (checked on log steps; 0 = off).",
    )
    p.add_argument(
        "--rope_fp32", action="store_true",
        help="Keep RoPE tables fp32 through the bf16 cast and rotate in fp32.",
    )
    p.add_argument(
        "--muon_fp32_master", action="store_true",
        help="Muon keeps fp32 master weights + fp32 momentum (weight decay and updates "
             "applied in fp32). Without it, bf16 Muon weight decay is a no-op. Cannot "
             "resume a checkpoint made without it losslessly (masters rebuilt from params).",
    )
    p.add_argument(
        "--adamw_no_decay_norms", action="store_true",
        help="Exclude RMSNorm/QK-norm gains from AdamW weight decay (second "
             "param group, wd=0). Cannot resume a checkpoint made without it.",
    )

    # Schedule
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--decay_start_pct", type=float, default=0.90)
    p.add_argument("--decay_type", type=str, default="sqrt")

    # Data
    p.add_argument("--data_path", type=str, default="")
    p.add_argument("--mix_data_path", type=str, default=None,
                   help="Secondary token bin interleaved into --data_path "
                        "(e.g. the dyadic bin of configs/ladder-a4.yaml).")
    p.add_argument("--mix_fraction", type=float, default=0.0,
                   help="Exact token fraction of --mix_data_path (token-debt interleave).")
    p.add_argument("--mix_window", type=str, default=None,
                   help="Confine the mix to global tokens START:END (e.g. '0:2.5B', '2.5B:5B'); the "
                        "fraction applies inside the window, the primary runs alone outside it.")
    p.add_argument(
        "--random_data",
        action="store_true",
        help="Use random tokens (for testing)",
    )
    p.add_argument(
        "--doc_masking",
        action="store_true",
        help="Enable document-boundary attention masking via EOS detection. "
             "Uses flash_attn_varlen_func to prevent cross-document attention.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init_scheme", type=str, default="standard",
                   choices=list(_INIT_SCHEMES),
                   help="Weight init applied over the standard init (kotodama.model.init_schemes). "
                        "'standard' = unchanged.")
    p.add_argument(
        "--attn_lr_scale", type=float, default=1.0,
        help="Multiply the attention projections' (q/k/v/o) Muon LR by this for the first "
             "--attn_lr_scale_steps steps (e.g. 0.1 or 0 to protect NCA-formed attention through the "
             "handoff while embeddings/MLPs learn the interface). Puts them in their own Muon group.",
    )
    p.add_argument("--attn_lr_scale_steps", type=int, default=0,
                   help="Steps (from 0) during which --attn_lr_scale applies.")
    p.add_argument("--attn_lr_scale_layers", type=str, default="",
                   help="Restrict --attn_lr_scale to these layers' attention, e.g. '0-6' (default: all layers).")
    p.add_argument(
        "--data_seed", type=int, default=None,
        help="Seed for the training-data order (shuffle) only; default = --seed. Lets init and data order vary "
             "independently (init x data-order studies); --seed alone keeps both tied.",
    )

    # Eval (held-out data for overfit detection)
    p.add_argument(
        "--eval_data_path", type=str, default=None,
        help="Path to held-out eval binary (same format as data_path)",
    )
    p.add_argument(
        "--eval_every", type=int, default=0,
        help="Run eval every N steps (0 = disabled)",
    )
    p.add_argument(
        "--eval_batches", type=int, default=20,
        help="Number of micro-batches per eval pass",
    )

    # In-loop evals (kotodama.training.inloop): small suites on the live weights
    p.add_argument(
        "--inloop_evals", type=str, default=None,
        help="Comma list of in-loop eval suites scored on the live weights: riders "
             "(riders-lite), setbind (needs --inloop_setbind_items), induction, "
             "uncheatable (needs --inloop_uncheatable_path), "
             "nca_gates (needs --inloop_nca_gates_dir). Unset/empty = off.",
    )
    p.add_argument(
        "--inloop_schedule", type=str, default="0:0.025,decay:0.005",
        help="In-loop cadence as START:INTERVAL token fractions; START = fraction or "
             "'decay' (LR decay start). Default: every 2.5%%, every 0.5%% in the decay; "
             "first and final steps always.",
    )
    p.add_argument("--inloop_max_batch", type=int, default=64,
                   help="In-loop: prompts per forward")
    p.add_argument("--inloop_max_batch_tokens", type=int, default=8192,
                   help="In-loop: summed prompt tokens per forward (bounds logits memory)")
    p.add_argument("--inloop_riders_items", type=str, default=None,
                   help="Rider items.jsonl (default: generate the standard battery in-process)")
    p.add_argument("--inloop_riders_shots", type=str, default=None,
                   help="Rider shots.json (with --inloop_riders_items)")
    p.add_argument("--inloop_riders_n", type=int, default=200,
                   help="Rider items per register (balanced over the std (n,m) cells)")
    p.add_argument("--inloop_riders_seed", type=int, default=0,
                   help="Seed of the rider item selection")
    p.add_argument("--inloop_riders_relet", action="store_true", default=True,
                   help="Also score 3-shot with re-lettered exemplar shots (default on)")
    p.add_argument("--no_inloop_riders_relet", dest="inloop_riders_relet",
                   action="store_false", help="Skip the 3shot_relet condition")
    p.add_argument("--inloop_riders_top_n", type=int, default=5,
                   help="Full-vocab top-n tokens kept per rider row")
    p.add_argument("--inloop_setbind_items", type=str, default=None,
                   help="Bindprobes items.jsonl for the setbind suite (its static setbind items are scored "
                        "3-shot; the bp-v3.1 battery gives the benchmark's set-binding k1 read)")
    p.add_argument("--inloop_induction_n_seqs", type=int, default=32)
    p.add_argument("--inloop_induction_len", type=int, default=128,
                   help="Random prefix length (sequence = 2x + EOS)")
    p.add_argument("--inloop_uncheatable_path", type=str, default=None,
                   help="Uncheatable data (domain files / parquet dir) for the uncheatable suite")
    p.add_argument("--inloop_uncheatable_docs", type=int, default=8,
                   help="Leading documents scored per Uncheatable domain")
    p.add_argument("--inloop_nca_gates_dir", type=str, default=None,
                   help="NCA eval dir (eval_ledger*.jsonl + meta.json) for the nca_gates suite")
    p.add_argument("--inloop_nca_gates_max_items", type=int, default=256,
                   help="Leading items scored per ledger file")

    # Checkpointing
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--keep_checkpoints", type=int, default=5)
    p.add_argument(
        "--save_schedule", type=str, default=None,
        help="Weights-only saves on a piecewise token cadence, e.g. "
             "'250M:4B,1B:10B,5B:end' -> <checkpoint_dir>/weights/ (never pruned).",
    )
    p.add_argument(
        "--anchor_tokens", type=str, default=None,
        help="Comma list of token counts (e.g. '10B') for protected FULL "
             "checkpoints in <checkpoint_dir>/anchors/ (branch-cooldown fork points).",
    )
    p.add_argument(
        "--async_checkpoint", action="store_true",
        help="Async checkpointing: CPU clone + background SHM write + zstd",
    )
    p.add_argument(
        "--checkpoint_shm_dir", type=str, default="/dev/shm/kotodama-ckpts",
        help="SHM directory for async checkpoint temp storage",
    )
    p.add_argument(
        "--checkpoint_compress", action="store_true", default=True,
        help="Compress checkpoints with zstd (default: True)",
    )
    p.add_argument(
        "--no_checkpoint_compress", dest="checkpoint_compress", action="store_false",
        help="Disable zstd compression for async checkpoints",
    )

    # Data prefetch
    p.add_argument(
        "--prefetch_batches", type=int, default=2,
        help="Background-thread batch prefetch depth (0 = disabled). "
             "Overlaps memmap reads / EOS scans / collation with GPU compute; "
             "checkpoint data-state tracks consumed batches exactly.",
    )

    # Logging
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument(
        "--peak_tflops_per_gpu", type=float, default=2250.0,
        help="Per-GPU peak TFLOPs for MFU reporting (default: 2250 = B200 dense BF16). "
             "Reported MFU is on the BF16 basis regardless of FP8 use — comparable across runs.",
    )

    # Profiling
    p.add_argument(
        "--profile", action="store_true",
        help="Enable torch.profiler for kernel-level tracing",
    )
    p.add_argument(
        "--profile_start_step", type=int, default=50,
        help="Step to start profiling (skip JIT warmup)",
    )
    p.add_argument(
        "--profile_steps", type=int, default=20,
        help="Number of steps to profile",
    )
    p.add_argument(
        "--profile_dir", type=str, default=None,
        help="Directory for profiler trace output (default: checkpoint_dir/profile)",
    )

    # Geometric monitoring
    p.add_argument(
        "--geo_monitor",
        action="store_true",
        help="Enable geometric health monitoring (rank 0 only)",
    )
    p.add_argument(
        "--geo_monitor_sharded",
        action="store_true",
        default=True,
        help="(default on) Split each geometric probe over the DDP ranks instead of rank 0 computing it while the "
             "others wait (same metric definitions: token means and Gram matrices summed over ranks; random-sample "
             "metrics use rank 0's draw).",
    )
    p.add_argument(
        "--no_geo_monitor_sharded",
        dest="geo_monitor_sharded",
        action="store_false",
        help="Compute the geometric probes on rank 0 only (the pre-sharding behaviour).",
    )
    p.add_argument("--geo_monitor_schedule", type=str, default=None,
                   help="Decaying cadence as 'step:interval,...' e.g. "
                        "'0:25,2000:50,10000:200,50000:500'. "
                        "Overrides tier1/tier2/eoc cadence args.")
    p.add_argument("--geo_monitor_tier1_every", type=int, default=500,
                   help="Fixed cadence fallback when --geo_monitor_schedule is not set")
    p.add_argument("--geo_monitor_tier2_every", type=int, default=5000,
                   help="(Legacy) tier2 cadence, ignored when schedule is set")
    p.add_argument("--geo_monitor_eoc_every", type=int, default=0,
                   help="(Legacy) EoC cadence, ignored when schedule is set")

    # NCA → language transition
    p.add_argument(
        "--resume_nca",
        type=str,
        default=None,
        help="Path to NCA checkpoint — loads weights, reinitializes embeddings",
    )
    p.add_argument(
        "--reset_routing_layers",
        type=str,
        default=None,
        help="With --resume_nca: return these layers' AttnRes routing (query + key-norm "
             "gain, pre-attn and pre-MLP) to fresh init at the handoff; spec like "
             "'24-27,final' ('final' = final readout)",
    )
    # CPT / weight-only resume (no optimizer, no step, no embedding reinit)
    p.add_argument(
        "--resume_weights",
        type=str,
        default=None,
        help="Path to checkpoint — loads model weights only for continued pretraining",
    )
    p.add_argument(
        "--resume_warm",
        type=str,
        default=None,
        help="Path to checkpoint — loads model weights and resets step counter, tokens "
             "consumed, data position, and scheduler. Optimizer moments start FRESH (weights "
             "only; no optimizer state is restored).",
    )
    p.add_argument(
        "--reinit_mlps",
        action="store_true",
        help="Also reinitialize MLPs after NCA (default: keep NCA-trained MLPs)",
    )

    # FP8 training
    p.add_argument(
        "--fp8",
        action="store_true",
        help="Enable FP8 training via torchao (Linear layers only, Muon NS stays fp16)",
    )
    p.add_argument(
        "--fp8_recipe",
        type=str,
        default="tensorwise",
        choices=["tensorwise", "mxfp8"],
        help="FP8 scaling recipe: tensorwise (Float8Linear, production) or "
        "mxfp8 (per-32-block scales, B200-native; validate before production use)",
    )

    # Liger fused kernels
    p.add_argument(
        "--use_liger",
        action="store_true",
        help="Enable Liger fused kernels (RMSNorm, SwiGLU, CrossEntropy)",
    )
    p.add_argument(
        "--liger_rmsnorm_tuned",
        action="store_true",
        help="With --use_liger: launch Liger's RMSNorm kernels with B200-tuned grids (same kernels and math; the "
             "weight gradient's partial sums are grouped differently, so new runs only). +4.3%% tok/s on the 1B "
             "tsumugi shape; to 2.5B tokens the loss tracks the default launch (mean delta +0.0003) and the "
             "induction / rider gauges converge (an induction-formation timing shift around step 476 closes).",
    )
    p.add_argument(
        "--liger_rmsnorm_tuned_scope", choices=["all", "qk"], default="all",
        help="With --liger_rmsnorm_tuned: 'all' norms (default), or 'qk' = the QK-norms only (dX bitwise Liger's; "
             "only their weight gradient's partial-sum grouping differs) - a minimal numerics perturbation for "
             "sensitivity controls, not a throughput setting.",
    )

    # Attention implementation
    p.add_argument(
        "--attn_impl", type=str, default="auto",
        choices=["auto", "fa2", "fa4", "flex", "sdpa"],
        help="Attention backend. 'auto': FA2 if available, else SDPA. "
             "'fa2': Flash Attention 2. 'fa4': CuTeDSL SM100 (lazy import; compile-safe custom ops). "
             "'flex': FlexAttention + BlockMask doc masking (Inductor-fused). "
             "'sdpa': PyTorch SDPA.",
    )


    # Tensor parallelism
    p.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor parallelism degree (1 = DDP only, 8 = full TP on 8 GPUs)",
    )

    # torch.compile
    p.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for throughput",
    )
    p.add_argument(
        "--compile_mode",
        type=str,
        default=None,
        choices=[None, "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        help="torch.compile mode (default: None = PyTorch default)",
    )
    p.add_argument(
        "--capture_scalar_outputs",
        action="store_true",
        help="Set torch._dynamo.config.capture_scalar_outputs=True (kills the "
             "Liger FLCE .item() tail graph-break; bench arm)",
    )
    p.add_argument(
        "--compile_regional",
        action="store_true",
        help="With --compile: compile each layer's attention / MLP sublayer (one compiled function shared by "
             "every layer, KotodamaModel.compile_regions) instead of the whole model — a few small graphs "
             "instead of the unrolled depth; AttnRes models only.",
    )
    p.add_argument(
        "--compile_regional_scope", choices=["sublayer", "layer"], default="sublayer",
        help="With --compile_regional: 'sublayer' compiles attention / MLP sublayers with the AttnRes routes eager "
             "between them; 'layer' compiles each whole layer with both routes (one graph per committed-block count "
             "and boundary).",
    )
    p.add_argument(
        "--attnres_eps_fp64_compat", action="store_true",
        help="AttnRes forward kernels use the runtime eps (fp64 inside a compiled graph) instead of the constexpr "
             "(fp32): reproduces the routing numerics of runs trained before the constexpr existed.",
    )
    p.add_argument(
        "--no_attnres_eps_fp64_compat", dest="attnres_eps_fp64_compat", action="store_false",
        help="Turn off --attnres_eps_fp64_compat set by a config (fp32 AttnRes routing).",
    )
    p.add_argument(
        "--attnres_res_gain", choices=["learned", "frozen", "folded"], default="learned",
        help="AttnRes routing gain (res_norm.weight, which only ever multiplies the routing query): 'learned' trains "
             "both (default; the lineage before this flag); 'frozen' keeps the gain fixed (1 for a fresh run); "
             "'folded' moves it into the query (q <- q*g, g <- 1; outputs unchanged) and keeps it fixed. A run "
             "switching mode must start from --resume_weights (the optimizer state no longer matches).",
    )
    p.add_argument(
        "--attnres_query_norm", action="store_true",
        help="AttnRes routing queries as temperature * q / |q| (a learned per-site sharpness and a unit direction; "
             "the temperature starts at 0, so routing starts uniform as with zero-init queries). New runs only; needs "
             "--attnres_res_gain folded or frozen.",
    )
    p.add_argument(
        "--attnres_query_temp_gain", type=float, default=1.0,
        help="With --attnres_query_norm: constant gain on each routing temperature (effective sharpness = gain x "
             "temperature). AdamW moves the scalar temperature by ~lr per step, a plain query's norm by ~lr*sqrt(D); "
             "sqrt(hidden_size) (39.19 at 1536) matches the plain query's rate.",
    )
    p.add_argument(
        "--attnres_query_lr_scale", type=float, default=1.0,
        help="AdamW LR multiplier for the AttnRes routing queries (their own group; the temperatures of "
             "--attnres_query_norm included). A run with a scale != 1 cannot resume a checkpoint built without it "
             "(the AdamW group count differs).",
    )
    p.add_argument(
        "--attnres_stack_grad_chain", action="store_true",
        help="AttnRes routes sharing a committed stack chain its gradient through a single-query backward kernel "
             "(summed in-kernel, one round per partial sum) instead of autograd summing a full stack gradient per "
             "route. +3%% tok/s on the 1B tsumugi shape; the sum's order and rounding differ (loss +0.002 after "
             "an early +0.01 bump that closes by ~300 steps), so use it for new runs only.",
    )
    p.add_argument(
        "--ddp_bucket_cap_mb", type=float, default=None,
        help="DDP gradient bucket size in MB (default: DDP's own, 25). Comm granularity only.",
    )
    p.add_argument(
        "--ddp_no_broadcast_buffers", action="store_true",
        help="DDP broadcast_buffers=False: skip the per-forward rank-0 broadcast of the model's constant buffers.",
    )
    p.add_argument(
        "--micro_grad_stats", action="store_true",
        help="Per step, per DP rank and micro-step: the micro-batch loss and the accumulated local squared grad norm "
             "(last micro read by a DDP comm hook before the all-reduce) -> <checkpoint_dir>/micro_grad_stats.jsonl, "
             "plus gradstats/* telemetry. Localizes a spike to (rank, micro). One small all-gather per step.",
    )
    p.add_argument(
        "--spike_capture_factor", type=float, default=0.0,
        help="When > 1: a step whose grad norm exceeds this x the running median saves its PRE-update weights and "
             "its micro record to <checkpoint_dir>/spikes/ (re-score the batch offline against the exact state).",
    )
    p.add_argument("--spike_capture_window", type=int, default=200,
                   help="Steps in the running-median window for --spike_capture_factor.")
    p.add_argument("--spike_capture_max", type=int, default=5,
                   help="At most this many spike captures per run (each is a full weights-only save).")
    p.add_argument(
        "--varlen_dynamic",
        action="store_true",
        help="With --compile --doc_masking: mark cu_seqlens' length dynamic from the first trace and pass "
             "max_seqlen = --sequence_length (an exact upper bound) — one compile at startup instead of "
             "per-rank recompiles on the first new doc count / max doc length.",
    )
    p.add_argument(
        "--ac_budget",
        type=float,
        default=0.0,
        help=(
            "Inductor activation_memory_budget in (0,1]: partitioner-driven "
            "selective recompute. Requires --compile and manual AC "
            "off. 0 = disabled (default). 1.0 = save everything eligible; "
            "lower = recompute more."
        ),
    )

    # Wandb
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging (rank 0 only)",
    )
    p.add_argument("--wandb_project", type=str, default="kotodama")
    p.add_argument("--wandb_run_name", type=str, default=None)

    # HuggingFace upload
    p.add_argument(
        "--hf_upload_repo", type=str, default=None,
        help="HF repo (e.g. '<org>/model-name') — upload final checkpoint on completion",
    )

    # First parse to check for --config
    args = p.parse_args()

    # If config file provided, load YAML as defaults and re-parse
    # so CLI flags take precedence over YAML values
    if args.config is not None:
        config = _load_yaml_config(args.config)
        # Warn about unknown keys (likely typos)
        known_args = {a.dest for a in p._actions}
        unknown = set(config.keys()) - known_args
        if unknown:
            logger.warning(
                "Unknown keys in %s (ignored): %s",
                args.config, ", ".join(sorted(unknown)),
            )
            for k in unknown:
                del config[k]
        p.set_defaults(**config)
        args = p.parse_args()
        args._config_file = args.config

    if args.stop_at_step is not None and args.stop_at_step <= 0:
        p.error(f"--stop_at_step must be > 0, got {args.stop_at_step}")
    return args


def main() -> None:
    args = parse_args()
    try:
        train(args)
    except Exception:
        logger.exception("Training failed")
        cleanup_distributed()
        raise


if __name__ == "__main__":
    main()
