"""
Geometric health monitoring for kotodama pretraining.

Metrics computed per monitoring step:
  - RankMe (effective rank) on probe batch hidden states
  - Stable rank per layer (weight-space)
  - Anisotropy (average pairwise cosine similarity)
  - Dead unit fraction per layer
  - Attention entropy distribution
  - TwoNN intrinsic dimensionality at sampled layers
  - EoC Jacobian spectral radius per layer

EoS sharpness is computed separately (requires live gradients).

Usage in training loop::

    monitor = GeometricMonitor(model, config)
    # In training loop:
    if monitor.should_monitor(step):
        sharpness = monitor.compute_sharpness(step)  # before optimizer.step()
        ...
        metrics = monitor.compute_all(step)           # after optimizer.step()
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class MonitorSchedule:
    """Step-function schedule for decaying monitoring cadence.

    Parsed from "start:interval,..." e.g. "0:25,2000:50,10000:200,50000:500".
    At each step, the last phase whose start_step <= step determines the interval.
    """

    phases: list[tuple[int, int]] = field(default_factory=lambda: [(0, 500)])

    @classmethod
    def from_string(cls, s: str) -> MonitorSchedule:
        phases: list[tuple[int, int]] = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            start_s, interval_s = part.split(":")
            phases.append((int(start_s), int(interval_s)))
        phases.sort(key=lambda x: x[0])
        if not phases:
            raise ValueError(f"Empty schedule: {s!r}")
        return cls(phases=phases)

    @classmethod
    def fixed(cls, every: int) -> MonitorSchedule:
        return cls(phases=[(0, every)])

    def should_fire(self, step: int) -> bool:
        interval = self.interval_at(step)
        return interval > 0 and step % interval == 0

    def interval_at(self, step: int) -> int:
        result = self.phases[0][1]
        for start, interval in self.phases:
            if step >= start:
                result = interval
            else:
                break
        return result


@dataclass
class MonitorConfig:
    """Configuration for geometric monitoring."""

    schedule: Optional[MonitorSchedule] = None

    # Probe / layer sampling
    tier1_probe_size: int = 1024  # number of samples in probe batch
    tier1_sample_layers: list[int] = field(
        default_factory=lambda: []
    )  # empty = auto-select

    tier2_twonn_samples: int = 3000  # samples for TwoNN ID estimation
    tier2_twonn_layers: list[int] = field(
        default_factory=lambda: []
    )  # empty = auto-select (5 evenly spaced)

    # Legacy cadence fields (used when schedule is None)
    tier1_every: int = 500
    tier2_every: int = 5000

    # Memory hygiene around probes. torch.cuda.empty_cache() hands the caching allocator's blocks back to the driver,
    # and the next training steps then re-cudaMalloc them (on the 1B tsumugi the steps after each probe point ran
    # ~5-15% slower). So a probe releases the cache only when the device is short of free memory.
    release_cache_below_gb: float = 8.0
    # The sharpness probe snapshots the weights it perturbs: on the GPU when that much memory is free (fast), else on
    # the CPU (the 7B-era default, which avoids 2x weights on rank 0).
    sharpness_gpu_snapshot: bool = True
    # Split each probe over the data-parallel ranks instead of computing it on rank 0 while the others wait. Every
    # rank must then construct the monitor with the same probe batch and call it at the same steps. Metrics keep
    # their definitions: token means and Gram matrices are summed over ranks, random-sample metrics draw their
    # sample from every rank's rows, and the EoC power iteration normalizes its vector over all ranks.
    sharded: bool = False

    # General
    device: str = "cuda"


class _ProbeShard:
    """Which probe rows this rank holds, and the reductions that put a sharded probe back together.

    Rows are interleaved (rank r holds rows r, r + world, ...), so the leading-row subsets some probes use (the
    first 8 sequences for sharpness, the first 4 for EoC and attention entropy) are spread over the ranks too."""

    def __init__(self, enabled: bool) -> None:
        self.on = bool(enabled and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1)
        self.rank = dist.get_rank() if self.on else 0
        self.world = dist.get_world_size() if self.on else 1

    def rows(self, n: int) -> list[int]:
        """Global indices of the rows (of an n-row batch) held here."""
        return list(range(self.rank, n, self.world))

    def take(self, batch: torch.Tensor) -> torch.Tensor:
        return batch[self.rank::self.world] if self.on else batch

    def sum_(self, t: torch.Tensor) -> torch.Tensor:
        """In-place sum over ranks (no-op unsharded)."""
        if self.on:
            dist.all_reduce(t)
        return t

    def norm(self, t: torch.Tensor) -> torch.Tensor:
        """L2 norm of the concatenation of every rank's t."""
        if not self.on:
            return t.norm()
        sq = t.float().pow(2).sum().reshape(1)
        return self.sum_(sq).sqrt()[0]

    def broadcast_(self, t: torch.Tensor) -> torch.Tensor:
        """In place: rank 0's t on every rank (no-op unsharded)."""
        if self.on:
            dist.broadcast(t, src=0)
        return t

    def sample_rows(self, h_local: torch.Tensor, n_seq_global: int, k: int) -> torch.Tensor:
        """The random token sample the unsharded probe would draw. Rank 0 draws ``torch.randperm(n_seq_global * T)[:k]``
        over the flattened (sequence, position) rows of the WHOLE probe (the CPU-RNG call the unsharded code makes) and
        broadcasts it; each rank contributes the drawn rows it holds (h_local: (this rank's sequences, T, D)), and every
        rank gets all k rows (in rank order; the consumers, anisotropy and TwoNN, are order-invariant)."""
        T, D = h_local.shape[1], h_local.shape[2]
        n = n_seq_global * T
        k = min(k, n)
        idx = torch.randperm(n)[:k] if self.rank == 0 else torch.empty(k, dtype=torch.long)
        idx = self.broadcast_(idx.to(h_local.device))
        seq, pos = idx // T, idx % T
        mine = (seq % self.world) == self.rank
        rows = h_local[seq[mine] // self.world, pos[mine]].float()
        buf = torch.full((k, D), float("nan"), device=h_local.device, dtype=torch.float32)
        buf[: rows.shape[0]] = rows
        out = torch.empty((self.world * k, D), device=h_local.device, dtype=torch.float32)
        dist.all_gather_into_tensor(out, buf)
        return out[~torch.isnan(out[:, 0])]


class _PowerIterResult(NamedTuple):
    """One EoC power iteration: σ (spectral-radius estimate), the scale-free gain σ·‖h‖/‖f(h)‖, the vector-Jacobian
    products it took, and its drift |σ_K − σ_⌊K/2⌋| / σ_K (how far σ still moved over the second half of the budget;
    0 when J annihilated the vector)."""
    sigma: float
    gain: float
    iters: int
    drift: float


class GeometricMonitor:
    """
    Geometric health monitor for transformer pretraining.

    Computes metrics at configurable intervals during training.
    All metrics are returned as flat dictionaries suitable for
    wandb/logging.

    The monitor holds a fixed probe batch for longitudinal
    comparability — the same inputs are used at every measurement
    point so changes reflect model development, not data variance.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[MonitorConfig] = None,
    ) -> None:
        self.config = config or MonitorConfig()

        # Unwrap DDP and torch.compile wrappers
        m = model.module if hasattr(model, "module") else model
        # torch.compile wraps in OptimizedModule — unwrap to get the real model
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        self.model = m
        self._shard = _ProbeShard(self.config.sharded)
        self._local_rows: list[int] = []  # global probe-row indices this rank holds in the current probe

        # Determine layer sampling positions
        num_layers = len(self.model.layers)
        if not self.config.tier1_sample_layers:
            # Sample 5 evenly spaced layers + first and last
            step = max(1, num_layers // 4)
            self.config.tier1_sample_layers = sorted(
                set([0, step, 2 * step, 3 * step, num_layers - 1])
            )
        if not self.config.tier2_twonn_layers:
            # 5 evenly spaced positions (input, 25%, 50%, 75%, output)
            self.config.tier2_twonn_layers = [
                0,
                num_layers // 4,
                num_layers // 2,
                3 * num_layers // 4,
                num_layers - 1,
            ]

        # Storage for probe batch (set via set_probe_batch)
        self._probe_batch: Optional[torch.Tensor] = None

        # Hook storage for capturing intermediate activations
        self._hooks: list[Any] = []
        self._captured_hidden: dict[int, torch.Tensor] = {}
        self._captured_attn: dict[int, torch.Tensor] = {}

        # AttnRes diagnostics (populated during _probe_forward_attn_res)
        self._attn_res_diagnostics: dict[str, float] = {}

        # EoC power-iteration start vectors, one per global probe shape (see _eoc_start), and their device copies
        # (this rank's rows) for the sub-layers of one probe point, which share a shape
        self._eoc_starts: dict[tuple[int, ...], torch.Tensor] = {}
        self._eoc_dev_starts: dict[tuple[Any, ...], torch.Tensor] = {}

        logger.info(
            "GeometricMonitor: %d layers, tier1 layers=%s, tier2 ID layers=%s",
            num_layers,
            self.config.tier1_sample_layers,
            self.config.tier2_twonn_layers,
        )

    def _free_gb(self) -> float:
        """Free device memory: the driver's free bytes plus what the caching allocator holds unused (0 without CUDA)."""
        if not torch.cuda.is_available():
            return 0.0
        free, _total = torch.cuda.mem_get_info()
        return (free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1e9

    def _stable_ranks(self, device: torch.device) -> dict[str, float]:
        """geo/layer_{i}/stable_rank_{proj} for the sampled layers. Sharded, the matrices are dealt round-robin over
        the ranks and the values summed back (each rank computes about 1/world of the eigvalsh calls)."""
        items: list[tuple[str, torch.Tensor]] = []
        for layer_idx in self.config.tier1_sample_layers:
            layer = self.model.layers[layer_idx]
            for name, param in [
                ("q_proj", layer.attn.q_proj.weight),
                ("k_proj", layer.attn.k_proj.weight),
                ("o_proj", layer.attn.o_proj.weight),
                ("gate_proj", layer.ffn.gate_proj.weight),
                ("down_proj", layer.ffn.down_proj.weight),
            ]:
                items.append((f"geo/layer_{layer_idx}/stable_rank_{name}", param))
        if not self._shard.on:
            return {k: _stable_rank(p) for k, p in items}
        vals = torch.zeros(len(items), device=device, dtype=torch.float64)
        for i, (_k, p) in enumerate(items):
            if i % self._shard.world == self._shard.rank:
                vals[i] = _stable_rank(p)
        self._shard.sum_(vals)
        return dict(zip((k for k, _p in items), vals.tolist()))

    def _reduce_probe_stats(self, n_local_tokens: int, device: torch.device) -> None:
        """Sharded probe: turn this rank's token means (MLP neuron activity, AttnRes diagnostics) into means over
        every rank's tokens, in place. routing_entropy (the entropy OF a site's token-mean routing weights, so not
        itself a token mean) is recomputed from the reduced per-site means."""
        shard = self._shard
        total = shard.sum_(torch.tensor([float(n_local_tokens)], device=device, dtype=torch.float64))[0]
        for k in sorted(self._mlp_mean_abs):
            m = self._mlp_mean_abs[k].double() * n_local_tokens
            self._mlp_mean_abs[k] = (shard.sum_(m) / total).float()
        diag = self._attn_res_diagnostics
        if not diag:
            return
        keys = sorted(diag)
        vals = torch.tensor([diag[k] for k in keys], device=device, dtype=torch.float64) * n_local_tokens
        reduced = dict(zip(keys, (shard.sum_(vals) / total).tolist()))
        sites: dict[str, list[tuple[int, float]]] = {}
        for k in keys:
            if k.startswith("attnres/_alpha_mean/"):
                site, src = k[len("attnres/_alpha_mean/"):].rsplit("/", 1)
                sites.setdefault(site, []).append((int(src), reduced.pop(k)))
        for site, pairs in sites.items():
            alphas = torch.tensor([a for _s, a in sorted(pairs)], dtype=torch.float64)
            n_src = alphas.shape[0]
            entropy = -(alphas * (alphas + 1e-10).log()).sum().item()
            reduced[f"attnres/routing_entropy/{site}"] = entropy / (math.log(n_src) if n_src > 1 else 1.0)
        diag.clear()
        diag.update(reduced)

    def _maybe_release_cache(self) -> None:
        if torch.cuda.is_available() and self._free_gb() < self.config.release_cache_below_gb:
            torch.cuda.empty_cache()

    def set_probe_batch(self, input_ids: torch.Tensor) -> None:
        """
        Set the fixed probe batch for longitudinal monitoring.

        This should be called once with a diverse, fixed set of token
        sequences that will be used for all forward-pass-based metrics.
        """
        self._probe_batch = input_ids.clone()
        logger.info("Probe batch set: shape %s", tuple(input_ids.shape))

    def should_monitor(self, step: int) -> bool:
        """Check if monitoring should fire at this step."""
        if self.config.schedule is not None:
            return self.config.schedule.should_fire(step)
        return self.config.tier1_every > 0 and step % self.config.tier1_every == 0

    # =========================================================================
    # Edge of Stability: finite-difference sharpness along gradient direction
    # =========================================================================

    def compute_sharpness(
        self, step: int, epsilon: float = 0.1, n_sequences: int = 8,
    ) -> dict[str, float]:
        """
        Compute directional sharpness along the current gradient direction.

        Must be called while gradients are still on model parameters (before
        optimizer.zero_grad). Uses finite differences on a probe batch subset:
            sharpness = (L(θ+εĝ) - 2L(θ) + L(θ-εĝ)) / ε²
        where ĝ = g/‖g‖ is the unit gradient direction.

        At edge of stability: sharpness ≈ 2/lr. Values exceeding this indicate
        the optimizer is actively reducing sharpness.

        Note: ε=0.1 (not 0.01) because at 3B+ scale, unit-normalized gradient
        components are O(1/√D) ≈ 6e-6, and smaller ε causes per-parameter
        perturbations to fall below bf16 precision.
        """
        if self._probe_batch is None:
            return {}

        params_with_grad = [
            p for p in self.model.parameters() if p.grad is not None
        ]
        if not params_with_grad:
            logger.warning("No gradients available for sharpness computation")
            return {}

        t0 = time.time()
        device = next(self.model.parameters()).device
        self._maybe_release_cache()
        batch = self._shard.take(self._probe_batch[:n_sequences]).to(device)

        def _probe_loss() -> float:
            # Token-mean probe loss. Sharded, each rank holds some of the sequences and the (loss x tokens, tokens)
            # pairs are summed over ranks: the same mean (unsharded, loss * n / n returns the loss unchanged).
            acc = torch.zeros(2, device=device, dtype=torch.float64)
            if batch.shape[0] > 0:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = self.model(batch, labels=batch)["loss"].float()
                n_tok = batch.shape[0] * (batch.shape[1] - 1)
                acc[0] = loss.double() * n_tok
                acc[1] = n_tok
            self._shard.sum_(acc)
            return (acc[0] / acc[1]).item()

        grad_norm = torch.sqrt(
            sum(p.grad.float().pow(2).sum() for p in params_with_grad)
        )
        if grad_norm < 1e-10:
            return {}

        embed_params = {id(self.model.embed_tokens.weight)}
        embed_grad_sq = sum(
            p.grad.float().pow(2).sum()
            for p in params_with_grad if id(p) in embed_params
        )
        embed_grad_frac = (embed_grad_sq / grad_norm.pow(2)).item()

        was_training = self.model.training
        self.model.eval()

        metrics: dict[str, float] = {}
        # Exact snapshot — on the GPU when it fits beside the release margin, else on the CPU (so a 7B rank 0 does not
        # need 2x weight memory). In-place +e / -2e / +e adds on bf16 weights do not restore exactly and would leave a
        # residual on rank 0 only -> DDP ranks silently diverge.
        snapshot_gb = sum(p.numel() * p.element_size() for p in params_with_grad) / 1e9
        on_gpu = (self.config.sharpness_gpu_snapshot and params_with_grad[0].is_cuda
                  and self._free_gb() > snapshot_gb + self.config.release_cache_below_gb)
        snapshot = [p.detach().clone() if on_gpu else p.detach().to("cpu", copy=True) for p in params_with_grad]
        inv_norm = 1.0 / grad_norm

        def _set_perturbed(sign: float) -> float:
            """Weights := original + sign*eps*g_hat (computed in fp32, one rounding).
            Returns the norm of the displacement that actually landed."""
            landed_sq = torch.zeros((), device=device, dtype=torch.float32)
            for p, orig in zip(params_with_grad, snapshot):
                o = orig.to(device, non_blocking=True).float()
                new = (o + p.grad.float() * (sign * epsilon * inv_norm)).to(p.dtype)
                landed_sq += (new.float() - o).pow(2).sum()
                p.data.copy_(new)
            return landed_sq.sqrt().item()

        def _restore() -> None:
            for p, orig in zip(params_with_grad, snapshot):
                p.data.copy_(orig.to(device, non_blocking=True))

        try:
            with torch.no_grad():
                L0 = _probe_loss()
                landed_plus = _set_perturbed(+1.0)
                L_plus = _probe_loss()
                _set_perturbed(-1.0)
                L_minus = _probe_loss()

            sharpness = (L_plus - 2.0 * L0 + L_minus) / (epsilon ** 2)
            metrics["eos/sharpness"] = sharpness
            metrics["eos/L0_probe"] = L0
            metrics["eos/L_plus"] = L_plus
            metrics["eos/L_minus"] = L_minus
            metrics["eos/grad_norm"] = grad_norm.item()
            metrics["eos/embed_grad_frac"] = embed_grad_frac
            # < 1 means bf16 rounding swallowed part of the probe step: sharpness
            # is then measured along a truncated direction (read it with that in mind).
            metrics["eos/eps_landed_frac"] = landed_plus / epsilon
            metrics["eos/time_s"] = time.time() - t0

            logger.info(
                "EoS [step %d]: sharpness=%.4f, L0=%.4f, embed_grad=%.1f%%, landed=%.2f, time=%.2fs",
                step, sharpness, L0, embed_grad_frac * 100, landed_plus / epsilon,
                time.time() - t0,
            )

        except Exception as e:
            logger.warning("Sharpness computation failed: %s", e)
            if self._shard.on:
                raise  # the other ranks would be left waiting in this probe's collectives
        finally:
            try:
                with torch.no_grad():
                    _restore()
            except Exception:
                logger.error("CRITICAL: failed to restore weights after sharpness probe")
                raise

        if was_training:
            self.model.train()

        return metrics

    # =========================================================================
    # Edge of Chaos: per-layer Jacobian spectral radius
    # =========================================================================

    def compute_jacobian_spectral(
        self, step: int, n_iters: Optional[int] = None, n_sequences: int = 4,
    ) -> dict[str, float]:
        """
        Compute per-layer Jacobian spectral radius by power iteration (see _power_iterate).

        For each sampled layer, estimates σ_max(J_ℓ) where J_ℓ = ∂output/∂input.
        σ_max > 1 → layer amplifies perturbations (chaotic regime).
        σ_max < 1 → layer damps perturbations (ordered regime).
        σ_max ≈ 1 → edge of chaos (critical regime).

        For AttnRes models, computes separate σ for attn and MLP sub-layers
        (since they receive independently routed inputs in the actual forward).

        n_iters is the power-iteration budget (default EOC_ITERS). Every call starts
        from the same seeded vector and runs the same number of steps, so the readout
        depends only on the model and the probe batch, never on earlier calls.
        """
        if self._probe_batch is None:
            return {}

        t0 = time.time()
        device = next(self.model.parameters()).device
        self._maybe_release_cache()
        if self._shard.on:
            raise ValueError("compute_jacobian_spectral is unsharded only; compute_all computes EoC sharded")
        batch = self._probe_batch[:n_sequences].to(device)

        was_training = self.model.training
        self.model.eval()

        metrics: dict[str, float] = {}

        try:
            # Get hidden states at each sampled layer via probe forward
            with torch.no_grad():
                hidden_states, _ = self._probe_forward(batch)

            metrics.update(self._eoc_metrics(hidden_states, device, n_iters))
            metrics["eoc/time_s"] = time.time() - t0

            logger.info(
                "EoC [step %d]: σ_max=%.4f, σ_mean=%.4f, layer0=%.1f, time=%.2fs",
                step,
                metrics.get("eoc/sigma_max", 0),
                metrics.get("eoc/sigma_mean", 0),
                metrics.get("eoc/layer0_sigma", 0),
                time.time() - t0,
            )

        except Exception as e:
            logger.warning("Jacobian spectral computation failed: %s", e)

        if was_training:
            self.model.train()

        return metrics

    EOC_SEED = 1234        # seed of the dedicated generator that draws the power-iteration start
    EOC_ITERS = 20         # power-iteration steps (vector-Jacobian products) per probed sub-layer, every call
    EOC_DRIFT_TOL = 1e-2   # a sub-layer whose σ drifts more than this over the budget's second half is unconverged

    def _eoc_start(self, shape: tuple[int, ...]) -> torch.Tensor:
        """The power-iteration start for a probe of this GLOBAL shape (EoC sequences, T, D): float32 on the CPU.

        Drawn once per shape from a dedicated CPU generator seeded with EOC_SEED, so it is the same vector at every
        call, on every rank and at every world size, and drawing it leaves the global CPU and CUDA RNG streams (the
        training run's randomness) untouched."""
        v = self._eoc_starts.get(shape)
        if v is None:
            gen = torch.Generator(device="cpu").manual_seed(self.EOC_SEED)
            v = torch.randn(shape, generator=gen, dtype=torch.float32)
            self._eoc_starts[shape] = v
        return v

    def _power_iterate(
        self,
        h_detached: torch.Tensor,
        layer_fn: Any,
        device: torch.device,
        n_iters: int,
    ) -> _PowerIterResult:
        """Power iteration on the Jacobian J = ∂f/∂h of layer_fn at h_detached.

        Iterates vector-Jacobian products only (v ← Jᵀv), so σ is the dominant
        |eigenvalue| of J — the spectral RADIUS, the edge-of-chaos quantity — not
        the spectral norm (that needs Jᵀ J, i.e. forward-mode AD, which the FA4
        attention path does not support). The relative gain σ·‖h‖/‖f(h)‖ is
        scale-free (log-log amplification), so layers whose input sits at a very
        different scale (layer 0 reads the raw embedding) compare with the rest —
        raw σ there is ~‖W‖/rms(embedding), not chaos.

        Every call starts from the same seeded vector (_eoc_start) and takes exactly
        n_iters steps, so σ is a fixed, continuous function of the weights and the
        probe batch. A start vector carried over from earlier calls, or re-drawn on
        a call-count cadence, makes σ depend on the probe's call history (each redraw
        steps every series), and a data-dependent early stop makes it jump where the
        stop moves: σ_k can sit on a transient plateau a few steps in and leave it
        later, so two nearby weight states that stop on either side of the plateau
        read σ values apart by the plateau's height.

        A fixed budget does not make σ the true radius: with clustered top
        eigenvalues (|λ2/λ1| near 1) or a complex dominant pair, power iteration
        approaches |λ1| slowly or circles it. ``drift`` (|σ_K − σ_⌊K/2⌋| / σ_K) is
        reported beside σ, so an under-converged reading is visible as one.
        """
        # Sharded: h_detached holds this rank's rows (possibly none). The Jacobian is block-diagonal across sequences,
        # so iterating on every rank's rows with the vector normalized over all of them is the same power iteration;
        # sigma is global, so every rank takes the same number of steps (and collectives).
        shard = self._shard
        has_rows = h_detached.shape[0] > 0
        tail = tuple(h_detached.shape[1:])
        # The unsharded start over all the EoC sequences; sharded, each rank keeps its rows. The device copy is keyed
        # by everything that determines it, so a cached copy is always the right one.
        shape = ((self._eoc_n_global,) if shard.on else (h_detached.shape[0],)) + tail
        rows = tuple(self._eoc_local_global) if shard.on else None
        key = (shape, rows, str(device))
        v = self._eoc_dev_starts.get(key)
        if v is None:
            v = self._eoc_start(shape)
            v = (v[list(rows)] if rows is not None else v).to(device)
            self._eoc_dev_starts[key] = v
        v = v / shard.norm(v)

        sigmas: list[float] = []
        sigma = 0.0
        for _ in range(n_iters):
            if has_rows:
                h_input = h_detached.clone().requires_grad_(True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    layer_out = layer_fn(h_input)
                Jt_v = torch.autograd.grad(
                    (layer_out * v).sum(), h_input,
                )[0].float()
            else:
                Jt_v = torch.zeros_like(h_detached, dtype=torch.float32)
            sigma = shard.norm(Jt_v).item()
            sigmas.append(sigma)
            if sigma < 1e-10:  # J annihilated v: σ is 0 and another step would divide by it
                break
            v = (Jt_v / sigma).detach()
        if sigma < 1e-10:
            drift = 0.0
        elif len(sigmas) >= 2:
            drift = abs(sigma - sigmas[len(sigmas) // 2 - 1]) / sigma
        else:
            drift = float("nan")

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = layer_fn(h_detached).float() if has_rows else torch.zeros(0, device=device)
        out_norm = shard.norm(out).item()
        in_norm = shard.norm(h_detached.float()).item()
        rel = sigma * in_norm / out_norm if out_norm > 0 else float("nan")
        return _PowerIterResult(sigma, rel, len(sigmas), drift)

    def _eoc_metrics(
        self,
        hidden_states: dict[int, torch.Tensor],
        device: torch.device,
        n_iters: Optional[int] = None,
        n_seq: Optional[int] = None,
    ) -> dict[str, float]:
        """Per-layer Jacobian σ (raw) + relative gain + power-iteration drift, and interior aggregates.

        n_iters is each power iteration's budget (default EOC_ITERS). AttnRes: attn and
        MLP sub-layers are probed at their own routed inputs (pre-attn from attn_norm,
        pre-MLP from ffn_norm).
        """
        metrics: dict[str, float] = {}
        if n_seq and self._shard.on:
            # The first n_seq probe sequences: the ones of them this rank holds.
            local = [i for i, g in enumerate(self._local_rows) if g < n_seq]
            self._eoc_n_global = sum(1 for g in range(n_seq) if g < self._n_probe_rows)
            self._eoc_local_global = [g for g in self._local_rows if g < n_seq]
            cut = lambda t: t[local]  # noqa: E731
        else:
            cut = (lambda t: t[:n_seq]) if n_seq else (lambda t: t)

        budget = self.EOC_ITERS if n_iters is None else n_iters
        drifts: list[float] = []
        try:
            self._eoc_sites(hidden_states, device, budget, cut, metrics, drifts)
        finally:
            self._eoc_dev_starts.clear()  # device copies live for one probe point, not between them

        # Layer 0 raw σ is an embedding-scale artefact ("Spike No More", Takase et al.
        # COLM 2025): report it apart and keep it out of the raw aggregates. The
        # scale-free gain aggregates include every layer.
        layer0_key = "eoc/jacobian_sigma/layer_0"
        if layer0_key in metrics:
            metrics["eoc/layer0_sigma"] = metrics[layer0_key]
        for fam, agg, skip in [("eoc/jacobian_sigma/layer_", "eoc/sigma", layer0_key),
                               ("eoc/jacobian_gain/layer_", "eoc/gain", None)]:
            vals = [v for k, v in metrics.items()
                    if k.startswith(fam) and k[-1].isdigit() and k != skip]
            if vals:
                metrics[f"{agg}_max"] = max(vals)
                metrics[f"{agg}_min"] = min(vals)
                metrics[f"{agg}_mean"] = sum(vals) / len(vals)
        if drifts:
            # Under-convergence is a reading of its own: the budget, the worst drift, and how many probed
            # sub-layers drifted past EOC_DRIFT_TOL (a NaN drift, from a 1-step budget, counts as unconverged).
            metrics["eoc/iters"] = float(budget)
            metrics["eoc/drift_max"] = max(drifts, key=lambda d: math.inf if math.isnan(d) else d)
            metrics["eoc/unconverged"] = float(sum(not d <= self.EOC_DRIFT_TOL for d in drifts))
        return metrics

    def _eoc_sites(
        self,
        hidden_states: dict[int, torch.Tensor],
        device: torch.device,
        budget: int,
        cut: Any,
        metrics: dict[str, float],
        drifts: list[float],
    ) -> None:
        """Power-iterate every sampled layer (AttnRes: its attn and MLP sub-layers) into metrics; drifts collects
        each site's drift for the aggregates."""
        use_attn_res = getattr(self.model.config, "attn_res", False)
        rope_cos, rope_sin = self.model.rope_cos, self.model.rope_sin
        mlp_inputs = getattr(self, "_mlp_inputs", {})

        def _record(site: str, r: _PowerIterResult) -> None:
            metrics[f"eoc/jacobian_sigma/{site}"] = r.sigma
            metrics[f"eoc/jacobian_gain/{site}"] = r.gain
            metrics[f"eoc/jacobian_drift/{site}"] = r.drift
            drifts.append(r.drift)

        for layer_idx in self.config.tier1_sample_layers:
            if layer_idx not in hidden_states:
                continue
            layer = self.model.layers[layer_idx]
            h_attn = cut(hidden_states[layer_idx]).detach()
            base = f"layer_{layer_idx}"
            if use_attn_res:
                h_mlp = cut(mlp_inputs.get(layer_idx, hidden_states[layer_idx])).detach()
                for name, h, fn in [
                    ("attn", h_attn, lambda h, _l=layer: h + _l.attn(_l.attn_norm(h), rope_cos, rope_sin)),
                    ("mlp", h_mlp, lambda h, _l=layer: h + _l.ffn(_l.ffn_norm(h))),
                ]:
                    _record(f"{base}/{name}", self._power_iterate(h, fn, device, budget))
                metrics[f"eoc/jacobian_sigma/{base}"] = max(
                    metrics[f"eoc/jacobian_sigma/{base}/attn"], metrics[f"eoc/jacobian_sigma/{base}/mlp"])
                metrics[f"eoc/jacobian_gain/{base}"] = max(
                    metrics[f"eoc/jacobian_gain/{base}/attn"], metrics[f"eoc/jacobian_gain/{base}/mlp"])
            else:
                _record(base, self._power_iterate(
                    h_attn, lambda h, _l=layer: _l(h, rope_cos, rope_sin), device, budget))

    # =========================================================================
    # Unified monitoring: shared probe forward, all metrics
    # =========================================================================

    def compute_all(self, step: int, n_eoc_iters: Optional[int] = None) -> dict[str, float]:
        """
        Unified geometric monitoring with a single shared probe forward.

        Computes tier1 metrics (RankMe, stable rank, anisotropy, dead units,
        attention entropy), TwoNN intrinsic dimensionality, and EoC Jacobian
        spectral radius from one forward pass.

        NOT decorated @torch.no_grad — EoC power iteration needs autograd.
        The probe forward and no-grad metrics use explicit context blocks.
        """
        if self._probe_batch is None:
            logger.warning("No probe batch set — skipping compute_all")
            return {}

        t0 = time.time()
        device = next(self.model.parameters()).device
        self._maybe_release_cache()
        batch = self._probe_batch.to(device)
        # The AttnRes probe forward caps the batch at 16 sequences (see _probe_forward); cap before sharding so the
        # sharded probe covers the same 16.
        if getattr(self.model.config, "attn_res", False) and batch.shape[0] > 16:
            batch = batch[:16]
        shard = self._shard
        if shard.on and shard.world > batch.shape[0]:
            raise ValueError(f"sharded geo monitor: {shard.world} ranks > {batch.shape[0]} probe sequences")
        n_probe_rows = batch.shape[0]
        self._n_probe_rows = n_probe_rows
        self._local_rows = shard.rows(n_probe_rows)
        n_local_tokens = len(self._local_rows) * batch.shape[1]
        batch = shard.take(batch)

        was_training = self.model.training
        self.model.eval()

        metrics: dict[str, float] = {}
        t_phase = time.time()

        def _phase(name: str) -> None:
            # geo/time/<phase>: where a probe point's seconds go (one sync per phase).
            nonlocal t_phase
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            now = time.time()
            metrics[f"geo/time/{name}"] = now - t_phase
            t_phase = now

        try:
            # == Phase 1: probe forward + activation-based metrics (no grad) ==
            with torch.no_grad():
                hidden_states, attn_weights = self._probe_forward(batch)
                if shard.on:
                    self._reduce_probe_stats(n_local_tokens, device)
                _phase("probe_forward")

                # -- RankMe (effective rank) on last-layer hidden states --
                last_layer_idx = len(self.model.layers) - 1
                if last_layer_idx in hidden_states:
                    last_hidden = hidden_states[last_layer_idx]
                    H = last_hidden.reshape(-1, last_hidden.shape[-1]).float()
                    metrics["geo/rankme_last"] = _rankme_gram(H, shard) if shard.on else _rankme(H)
                _phase("rankme")

                # -- Per-layer metrics --
                metrics.update(self._stable_ranks(device))
                for layer_idx in self.config.tier1_sample_layers:
                    prefix = f"geo/layer_{layer_idx}"
                    layer = self.model.layers[layer_idx]

                    metrics.update(self._mlp_unit_metrics(layer_idx, prefix))
                    if layer_idx in hidden_states:
                        h = hidden_states[layer_idx]
                        h_flat = h.reshape(-1, h.shape[-1])
                        if shard.on:
                            metrics[f"{prefix}/anisotropy"] = _anisotropy(
                                shard.sample_rows(h, n_probe_rows, 512), max_samples=512)
                        else:
                            metrics[f"{prefix}/anisotropy"] = _anisotropy(h_flat, max_samples=512)

                    if shard.on:
                        ent = _attention_entropy_stats_sharded(
                            attn_weights.get(layer_idx), layer.attn.num_heads, shard, device)
                        if ent is not None:
                            metrics[f"{prefix}/attn_entropy_mean"], metrics[f"{prefix}/attn_entropy_std"] = ent
                    elif layer_idx in attn_weights:
                        ent_mean, ent_std = _attention_entropy_stats(attn_weights[layer_idx])
                        metrics[f"{prefix}/attn_entropy_mean"] = ent_mean
                        metrics[f"{prefix}/attn_entropy_std"] = ent_std

                # -- AttnRes diagnostics --
                if self._attn_res_diagnostics:
                    metrics.update(self._attn_res_diagnostics)
                _phase("layer_metrics")

                # -- TwoNN intrinsic dimensionality --
                for layer_idx in self.config.tier2_twonn_layers:
                    if layer_idx in hidden_states:
                        h = hidden_states[layer_idx].reshape(
                            -1, hidden_states[layer_idx].shape[-1]
                        )
                        if shard.on:
                            h_sub = shard.sample_rows(hidden_states[layer_idx], n_probe_rows,
                                                      self.config.tier2_twonn_samples)
                        else:
                            n = min(self.config.tier2_twonn_samples, h.shape[0])
                            idx = torch.randperm(h.shape[0])[:n]
                            h_sub = h[idx].float()
                        id_est = _twonn_id(h_sub)
                        if id_est is not None:
                            metrics[f"geo/twonn_id/layer_{layer_idx}"] = id_est
                _phase("twonn")

            # == Phase 2: EoC Jacobian spectral radius (needs autograd) ==
            metrics.update(self._eoc_metrics(hidden_states, device, n_eoc_iters, n_seq=4))
            _phase("eoc")

        except Exception as e:
            logger.warning("compute_all failed: %s", e, exc_info=True)
            if shard.on:
                raise  # carrying on would leave the other ranks waiting in a collective this rank never reaches

        if was_training:
            self.model.train()

        elapsed = time.time() - t0
        metrics["geo/compute_all_time_s"] = elapsed
        metrics["geo/step"] = float(step)

        logger.info(
            "Geo [step %d]: RankMe=%.1f, σ_max=%.4f, time=%.2fs (%s)",
            step,
            metrics.get("geo/rankme_last", 0),
            metrics.get("eoc/sigma_max", 0),
            elapsed,
            " ".join(f"{k[9:]} {v:.2f}" for k, v in metrics.items() if k.startswith("geo/time/")),
        )

        return metrics

    # =========================================================================
    # Tier 1: Lightweight streaming metrics (< 1% overhead)
    # =========================================================================

    @torch.no_grad()
    def tier1(self, step: int, probe_batch: Optional[torch.Tensor] = None) -> dict[str, float]:
        """
        Compute Tier 1 geometric health metrics.

        Requires a forward pass on the probe batch. Returns a flat dict
        of metric name → value.
        """
        t0 = time.time()
        batch = probe_batch if probe_batch is not None else self._probe_batch
        if batch is None:
            logger.warning("No probe batch set — skipping Tier 1 metrics")
            return {}

        device = next(self.model.parameters()).device
        batch = batch.to(device)

        metrics: dict[str, float] = {}

        # Run forward pass with hooks to capture hidden states and attention
        hidden_states, attn_weights = self._probe_forward(batch)

        # -- RankMe (effective rank) on last-layer hidden states --
        last_hidden = hidden_states[len(self.model.layers) - 1]  # [batch, seq, hidden]
        # Flatten to [n_samples, hidden_dim]
        H = last_hidden.reshape(-1, last_hidden.shape[-1]).float()
        metrics["geo/rankme_last"] = _rankme(H)

        # -- Per-layer metrics --
        for layer_idx in self.config.tier1_sample_layers:
            prefix = f"geo/layer_{layer_idx}"

            # Stable rank of weight matrices
            layer = self.model.layers[layer_idx]
            for name, param in [
                ("q_proj", layer.attn.q_proj.weight),
                ("k_proj", layer.attn.k_proj.weight),
                ("o_proj", layer.attn.o_proj.weight),
                ("gate_proj", layer.ffn.gate_proj.weight),
                ("down_proj", layer.ffn.down_proj.weight),
            ]:
                sr = _stable_rank(param)
                metrics[f"{prefix}/stable_rank_{name}"] = sr

            # Dead unit fraction (fraction of neurons with near-zero activation)
            metrics.update(self._mlp_unit_metrics(layer_idx, prefix))

            # Anisotropy (average pairwise cosine similarity)
            if layer_idx in hidden_states:
                h = hidden_states[layer_idx].reshape(
                    -1, hidden_states[layer_idx].shape[-1]
                )
                aniso = _anisotropy(h, max_samples=512)
                metrics[f"{prefix}/anisotropy"] = aniso

            # Attention entropy
            if layer_idx in attn_weights:
                attn = attn_weights[layer_idx]  # [batch, n_heads, seq, seq]
                ent_mean, ent_std = _attention_entropy_stats(attn)
                metrics[f"{prefix}/attn_entropy_mean"] = ent_mean
                metrics[f"{prefix}/attn_entropy_std"] = ent_std

        # Merge AttnRes diagnostics if present
        if self._attn_res_diagnostics:
            metrics.update(self._attn_res_diagnostics)

        elapsed = time.time() - t0
        metrics["geo/tier1_time_s"] = elapsed
        metrics["geo/step"] = float(step)

        logger.info(
            "Tier 1 [step %d]: RankMe=%.1f, time=%.2fs",
            step,
            metrics.get("geo/rankme_last", 0),
            elapsed,
        )

        return metrics

    # =========================================================================
    # Tier 2: Checkpoint-level metrics (minutes)
    # =========================================================================

    @torch.no_grad()
    def tier2(self, step: int, probe_batch: Optional[torch.Tensor] = None) -> dict[str, float]:
        """
        Compute Tier 2 geometric health metrics.

        TwoNN intrinsic dimensionality at sampled layers.
        """
        t0 = time.time()
        metrics: dict[str, float] = {}

        # -- TwoNN intrinsic dimensionality at sampled layers --
        batch = probe_batch if probe_batch is not None else self._probe_batch
        if batch is not None:
            device = next(self.model.parameters()).device
            batch = batch.to(device)
            hidden_states, _ = self._probe_forward(batch)

            for layer_idx in self.config.tier2_twonn_layers:
                if layer_idx in hidden_states:
                    h = hidden_states[layer_idx].reshape(
                        -1, hidden_states[layer_idx].shape[-1]
                    )
                    # Subsample for speed
                    n = min(self.config.tier2_twonn_samples, h.shape[0])
                    idx = torch.randperm(h.shape[0])[:n]
                    h_sub = h[idx].float()
                    id_est = _twonn_id(h_sub)
                    if id_est is not None:
                        metrics[f"geo/twonn_id/layer_{layer_idx}"] = id_est

        elapsed = time.time() - t0
        metrics["geo/tier2_time_s"] = elapsed

        logger.info(
            "Tier 2 [step %d]: TwoNN layers=%d, time=%.1fs",
            step,
            sum(1 for k in metrics if k.startswith("geo/twonn_id")),
            elapsed,
        )

        return metrics

    # =========================================================================
    # Forward pass with hooks
    # =========================================================================

    def _probe_forward(
        self, input_ids: torch.Tensor
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """
        Run a forward pass on the probe batch, capturing hidden states
        and attention weights at sampled layers.

        When AttnRes is enabled, uses the AttnRes forward path to capture
        per-layer hidden states that match what training actually computes.
        The "last layer" hidden state is the final AttnRes aggregation output.

        Returns (hidden_states, attn_weights) dicts keyed by layer index.
        """
        hidden_states: dict[int, torch.Tensor] = {}
        attn_weights: dict[int, torch.Tensor] = {}

        # Determine which layers we need activations from
        needed_layers = set(self.config.tier1_sample_layers) | set(
            self.config.tier2_twonn_layers
        )
        needed_layers.add(len(self.model.layers) - 1)  # always need last layer

        was_training = self.model.training
        self.model.eval()

        use_attn_res = getattr(self.model.config, "attn_res", False)

        # Cap probe batch: the uncompiled model forward under DDP crashes with
        # large batches (64 seq) but works fine with <=16.  16 × seq_len tokens
        # is more than enough for geometric statistics.
        if use_attn_res and input_ids.shape[0] > 16:
            input_ids = input_ids[:16]

        # MLP neuron activity: the SwiGLU output feeding down_proj, reduced per
        # neuron inside the hook (the residual stream has no "dead" dims to find).
        self._mlp_mean_abs = {}
        self._mlp_inputs: dict[int, torch.Tensor] = {}  # AttnRes pre-MLP routed inputs
        mlp_hooks: list[Any] = []

        def _make_mlp_hook(layer_idx: int):
            def hook(_module: nn.Module, args: tuple) -> None:
                act = args[0].detach().float().abs()
                self._mlp_mean_abs[layer_idx] = act.reshape(-1, act.shape[-1]).mean(dim=0)
            return hook

        for i in self.config.tier1_sample_layers:
            ffn = getattr(self.model.layers[i], "ffn", None)
            if ffn is not None and hasattr(ffn, "down_proj"):
                mlp_hooks.append(ffn.down_proj.register_forward_pre_hook(_make_mlp_hook(i)))

        try:
            self._probe_forward_inner(input_ids, needed_layers, use_attn_res,
                                      hidden_states, attn_weights)
        finally:
            for h in mlp_hooks:
                h.remove()

        if was_training:
            self.model.train()

        return hidden_states, attn_weights

    def _probe_forward_inner(
        self,
        input_ids: torch.Tensor,
        needed_layers: set[int],
        use_attn_res: bool,
        hidden_states: dict[int, torch.Tensor],
        attn_weights: dict[int, torch.Tensor],
    ) -> None:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if use_attn_res:
                hs, aw = self._probe_forward_attn_res(input_ids, needed_layers)
                hidden_states.update(hs)
                attn_weights.update(aw)
            else:
                x = self.model.embed_tokens(input_ids)
                rope_cos = self.model.rope_cos
                rope_sin = self.model.rope_sin

                for i, layer in enumerate(self.model.layers):
                    x = layer(x, rope_cos, rope_sin)
                    if i in needed_layers:
                        hidden_states[i] = x.detach()

                        if i in self.config.tier1_sample_layers:
                            attn_w = self._get_attention_weights(
                                layer, self.model.layers[i].attn_norm(x),
                                rope_cos, rope_sin, input_ids.shape[1]
                            )
                            if attn_w is not None:
                                attn_weights[i] = attn_w

    def _mlp_unit_metrics(self, layer_idx: int, prefix: str) -> dict[str, float]:
        """dead_mlp_units: neurons with mean |act| < 1e-6 over the probe;
        weak_mlp_units: below 1% of the layer's median neuron activity."""
        m = getattr(self, "_mlp_mean_abs", {}).get(layer_idx)
        if m is None:
            return {}
        med = m.median()
        return {
            f"{prefix}/dead_mlp_units": (m < 1e-6).float().mean().item(),
            f"{prefix}/weak_mlp_units": (m < 0.01 * med).float().mean().item(),
        }

    def _probe_forward_attn_res(
        self, input_ids: torch.Tensor, needed_layers: set[int]
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """
        AttnRes-aware probe forward pass using hooks on the model's own forward.

        Instead of manually re-implementing the AttnRes routing (which uses
        different Triton kernel call patterns than training and can crash under
        DDP), we hook into the model's own forward path — the same one that
        compute_sharpness uses successfully.

        Forward pre-hooks on attn_norm capture the routed hidden state h (input
        to normalization) at each layer. A pre-hook on the final norm captures
        the last-layer AttnRes aggregation output.
        """
        hidden_states: dict[int, torch.Tensor] = {}
        attn_weights: dict[int, torch.Tensor] = {}
        captured_normed: dict[int, torch.Tensor] = {}

        model = self.model
        last_layer_idx = len(model.layers) - 1
        hooks: list[Any] = []

        def _make_norm_hook(layer_idx: int):
            def hook(_module: nn.Module, args: tuple, output: torch.Tensor) -> None:
                hidden_states[layer_idx] = args[0].detach()
                if layer_idx in self.config.tier1_sample_layers:
                    captured_normed[layer_idx] = output.detach()
            return hook

        def _final_norm_hook(_module: nn.Module, args: tuple) -> None:
            hidden_states[last_layer_idx] = args[0].detach()

        # The MLP sub-layer reads its OWN routed input (pre-MLP AttnRes), not the
        # pre-attn one; EoC must probe its Jacobian there (probing the MLP at the
        # pre-attn input puts layer 0's MLP on the raw embedding it never sees).
        def _make_ffn_norm_hook(layer_idx: int):
            def hook(_module: nn.Module, args: tuple) -> None:
                self._mlp_inputs[layer_idx] = args[0].detach()
            return hook

        for i in needed_layers:
            if i < len(model.layers):
                hooks.append(
                    model.layers[i].attn_norm.register_forward_hook(
                        _make_norm_hook(i)
                    )
                )
                if i in self.config.tier1_sample_layers:
                    hooks.append(
                        model.layers[i].ffn_norm.register_forward_pre_hook(
                            _make_ffn_norm_hook(i)
                        )
                    )
        hooks.append(model.norm.register_forward_pre_hook(_final_norm_hook))

        # Sharded: the model also reports each site's token-mean routing weights, so _reduce_probe_stats can
        # recompute routing_entropy exactly from the means over every rank.
        model._attn_res_diag_keep_alphas = self._shard.on
        try:
            model(input_ids)
        finally:
            model._attn_res_diag_keep_alphas = False
            for h in hooks:
                h.remove()

        rope_cos = model.rope_cos
        rope_sin = model.rope_sin
        for layer_idx, normed_h in captured_normed.items():
            attn_w = self._get_attention_weights(
                model.layers[layer_idx], normed_h,
                rope_cos, rope_sin, input_ids.shape[1],
            )
            if attn_w is not None:
                attn_weights[layer_idx] = attn_w

        # --- AttnRes diagnostics (computed by _forward_attn_res during eval) ---
        self._attn_res_diagnostics.clear()
        if hasattr(model, "_last_attn_res_diagnostics"):
            self._attn_res_diagnostics.update(model._last_attn_res_diagnostics)

        return hidden_states, attn_weights

    def _get_attention_weights(
        self,
        layer: nn.Module,
        x_normed: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        seq_len: int,
    ) -> Optional[torch.Tensor]:
        """(The first 4 probe sequences; sharded, the ones of those this rank holds — None if it holds none.)"""
        """
        Compute raw attention weights for a layer.

        We compute Q @ K^T / sqrt(d) and softmax manually to get the
        weights without relying on SDPA (which doesn't return them).
        Uses a subset of the batch to keep memory bounded.
        """
        try:
            attn = layer.attn
            # Limit to first few sequences to save memory
            if self._shard.on:
                local = [i for i, g in enumerate(self._local_rows) if g < 4]
                if not local:
                    return None
                x_sub = x_normed[local]
            else:
                x_sub = x_normed[:4]
            bsz = x_sub.shape[0]

            q = attn.q_proj(x_sub).view(bsz, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = attn.k_proj(x_sub).view(bsz, seq_len, attn.num_kv_heads, attn.head_dim).transpose(1, 2)

            if attn.qk_norm:
                q = attn.q_norm(q)
                k = attn.k_norm(k)

            from kotodama.model.llama import apply_rope

            q = apply_rope(q, rope_cos[:seq_len], rope_sin[:seq_len])
            k = apply_rope(k, rope_cos[:seq_len], rope_sin[:seq_len])

            # Expand KV for GQA
            if attn.num_kv_groups > 1:
                k = k.repeat_interleave(attn.num_kv_groups, dim=1)

            # Compute attention weights
            scale = attn.head_dim ** -0.5
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

            # Causal mask
            causal = torch.triu(
                torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
                diagonal=1,
            )
            scores.masked_fill_(causal, float("-inf"))

            weights = F.softmax(scores, dim=-1)
            return weights.detach()
        except Exception as e:
            logger.debug("Failed to compute attention weights: %s", e)
            return None


# =============================================================================
# Metric computation functions (stateless, pure)
# =============================================================================


def _rankme(H: torch.Tensor, eps: float = 1e-7) -> float:
    """
    Compute RankMe (effective rank) of a representation matrix.

    RankMe = exp(entropy of normalized singular values)
    Higher = more diverse/spread representation.
    """
    # H: [n_samples, hidden_dim]
    # SVD of the (potentially large) matrix — use only singular values
    try:
        S = torch.linalg.svdvals(H)
        S = S / (S.sum() + eps)
        S = S[S > eps]
        entropy = -(S * torch.log(S)).sum()
        return torch.exp(entropy).item()
    except Exception:
        return 0.0


def _rankme_gram(H: torch.Tensor, shard: _ProbeShard, eps: float = 1e-7) -> float:
    """_rankme over the rows of every rank's H: the singular values of the stacked matrix are the square roots of the
    eigenvalues of its Gram matrix, which is the sum of the ranks' H^T H (fp64)."""
    try:
        Hd = H.double()
        gram = shard.sum_(Hd.t() @ Hd)
        S = torch.linalg.eigvalsh(gram).clamp_min(0).sqrt().flip(0).float()
        S = S / (S.sum() + eps)
        S = S[S > eps]
        entropy = -(S * torch.log(S)).sum()
        return torch.exp(entropy).item()
    except Exception:
        return 0.0


def _stable_rank(W: torch.Tensor) -> float:
    """
    Compute stable rank: ||W||_F^2 / ||W||_2^2.

    Measures effective dimensionality of a weight matrix.
    Stable rank 1 = rank-1 matrix. Higher = more distributed.

    sigma_max^2 is the top eigenvalue of the Gram matrix on the smaller side (fp64 W^T W or W W^T + eigvalsh):
    ~6x faster than svdvals on the model's projection shapes and equal to it within ~1e-4 relative (svdvals' own
    fp32 error; the fp64 Gram path agrees with an fp32 one).
    """
    try:
        W_d = W.detach().double()
        frob_sq = W_d.pow(2).sum()
        gram = W_d.t() @ W_d if W_d.shape[0] >= W_d.shape[1] else W_d @ W_d.t()
        spectral_sq = torch.linalg.eigvalsh(gram)[-1]
        return (frob_sq / (spectral_sq + 1e-10)).item()
    except Exception:
        return 0.0


def _anisotropy(H: torch.Tensor, max_samples: int = 512) -> float:
    """
    Average pairwise cosine similarity (anisotropy measure).

    High anisotropy (close to 1) = representations are clustered.
    Low anisotropy (close to 0) = representations are spread.
    """
    if H.shape[0] > max_samples:
        idx = torch.randperm(H.shape[0])[:max_samples]
        H = H[idx]

    H = H.float()
    # Normalize rows
    H_norm = F.normalize(H, dim=-1)
    # Compute mean cosine similarity (excluding self-similarity)
    sim = H_norm @ H_norm.T
    n = sim.shape[0]
    # Exclude diagonal
    mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    mean_sim = sim[mask].mean()
    return mean_sim.item()


def _attention_entropy_stats(
    attn: torch.Tensor, eps: float = 1e-10
) -> tuple[float, float]:
    """
    Compute mean and std of per-head attention entropy.

    attn: [batch, n_heads, seq, seq]
    Returns (mean_entropy, std_entropy) across heads.
    """
    # Entropy per head per position: -sum(p * log(p))
    attn_clamped = attn.float().clamp(min=eps)
    entropy = -(attn_clamped * attn_clamped.log()).sum(dim=-1)  # [batch, heads, seq]
    # Average over batch and sequence positions
    per_head = entropy.mean(dim=(0, 2))  # [heads]
    return per_head.mean().item(), per_head.std().item()




def _attention_entropy_stats_sharded(
    attn: Optional[torch.Tensor], n_heads: int, shard: _ProbeShard, device: torch.device, eps: float = 1e-10,
) -> Optional[tuple[float, float]]:
    """_attention_entropy_stats over every rank's sequences: per-head entropy sums and position counts are summed
    over ranks (a rank holding none of the sequences contributes zeros), then mean / std over heads."""
    buf = torch.zeros(n_heads + 1, device=device, dtype=torch.float64)
    if attn is not None:
        a = attn.float().clamp(min=eps)
        entropy = -(a * a.log()).sum(dim=-1)  # [batch, heads, seq]
        buf[:n_heads] = entropy.sum(dim=(0, 2)).double()
        buf[n_heads] = entropy.shape[0] * entropy.shape[2]
    shard.sum_(buf)
    if buf[n_heads] == 0:
        return None
    per_head = (buf[:n_heads] / buf[n_heads]).float()
    return per_head.mean().item(), per_head.std().item()


def _twonn_id(X: torch.Tensor) -> Optional[float]:
    """
    Estimate intrinsic dimensionality using the TwoNN method.

    Facco et al. (2017): "Estimating the intrinsic dimension of
    datasets by a minimal neighborhood information."

    Uses the ratio of second-nearest to nearest neighbor distances.
    """
    try:
        n = X.shape[0]
        if n < 10:
            return None

        # Compute pairwise distances
        # For memory efficiency, compute in chunks if large
        if n > 5000:
            X = X[:5000]
            n = 5000

        dists = torch.cdist(X, X)
        # Set self-distance to infinity
        dists.fill_diagonal_(float("inf"))

        # Get two nearest neighbors
        topk = dists.topk(2, dim=1, largest=False)
        r1 = topk.values[:, 0]  # nearest neighbor distance
        r2 = topk.values[:, 1]  # second nearest

        # Filter out zero distances
        valid = r1 > 1e-10
        if valid.sum() < 10:
            return None

        mu = r2[valid] / r1[valid]

        # Sort mu values
        mu_sorted = mu.sort().values
        n_valid = len(mu_sorted)

        # Empirical CDF
        i = torch.arange(1, n_valid + 1, dtype=torch.float32, device=X.device)
        # log(1 - F(mu)) = -d * log(mu), where F(mu) = i/n
        # So d = -log(1 - i/n) / log(mu)
        # Use linear regression: log(1 - i/(n+1)) vs log(mu)
        log_survival = torch.log(1.0 - i / (n_valid + 1))
        log_mu = torch.log(mu_sorted)

        # Linear regression (slope = -d)
        x = log_mu
        y = log_survival
        x_mean = x.mean()
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean).pow(2).sum() + 1e-10)

        d = -slope.item()

        # Sanity check
        if d < 0.5 or d > 10000:
            return None

        return d
    except Exception:
        return None


