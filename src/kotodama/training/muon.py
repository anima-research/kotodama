"""
Muon optimizer for kotodama.

Muon (Momentum Orthogonalized by Newton-Schulz) performs steepest descent
under the spectral norm by orthogonalizing the momentum matrix. This treats
all singular directions of the weight space proportionally, resisting the
low-rank collapse that Adam-family optimizers promote.

Here this is a deliberate choice, on the hypothesis that
Muon's balanced spectral updates produce richer representational geometry
(more diverse dynamical motifs, higher shattering dimensionality).

Reference: MoonshotAI/Muon (github.com/MoonshotAI/Muon)
           Moonlight paper: Muon is Scalable for LLM Training
           Gram-NS coefficients: Dao-AILab/Gram-Newton-Schulz

Usage:
    optimizer = build_hybrid_optimizer(model, muon_config, adamw_config)
"""

from __future__ import annotations

import re

import logging
from typing import Any, Optional

import torch
from torch.optim import AdamW, Optimizer

logger = logging.getLogger(__name__)


# ── NS coefficient presets ──────────────────────────────────────────────────
#
# Per-iteration (a, b, c) coefficients for the Newton-Schulz polar iteration:
#   X_{k+1} = (a_k I + b_k X_k X_k^T + c_k (X_k X_k^T)^2) X_k
#
# "original": Fixed coefficients from MoonshotAI/Muon, same polynomial every
#   iteration. Standard in all prior proxy ablations.
# "gram_ns": Per-iteration optimized from Gram-Newton-Schulz (Dao-AILab).
#   Earlier iterations more aggressive, later ones polish. Better convergence
#   in the same 5 iterations — free quality improvement.
# "polar_express": From the same repo, with safety factor adjustment (÷1.05
#   on a, ÷1.05^3 on b, ÷1.05^5 on c). More conservative than "gram_ns".

NS_COEFFICIENT_PRESETS: dict[str, list[tuple[float, float, float]]] = {
    "original": [
        (3.4445, -4.7750, 2.0315),
    ] * 5,
    "gram_ns": [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ],
    "polar_express": [
        (
            8.28721201814563 / 1.05,
            -23.595886519098837 / 1.05**3,
            17.300387312530933 / 1.05**5,
        ),
        (
            4.107059111542203 / 1.05,
            -2.9478499167379106 / 1.05**3,
            0.5448431082926601 / 1.05**5,
        ),
        (
            3.9486908534822946 / 1.05,
            -2.908902115962949 / 1.05**3,
            0.5518191394370137 / 1.05**5,
        ),
        (
            3.3184196573706015 / 1.05,
            -2.488488024314874 / 1.05**3,
            0.51004894012372 / 1.05**5,
        ),
        (
            2.300652019954817 / 1.05,
            -1.6689039845747493 / 1.05**3,
            0.4188073119525673 / 1.05**5,
        ),
    ],
}


def _resolve_ns_coefficients(
    preset: Optional[str],
    num_iterations: int,
) -> list[tuple[float, float, float]]:
    """Resolve NS coefficient preset name to per-iteration coefficient list."""
    if preset is None:
        preset = "original"
    if preset not in NS_COEFFICIENT_PRESETS:
        raise ValueError(
            f"Unknown NS coefficient preset '{preset}'. "
            f"Available: {list(NS_COEFFICIENT_PRESETS.keys())}"
        )
    coeffs = NS_COEFFICIENT_PRESETS[preset]
    if len(coeffs) < num_iterations:
        coeffs = coeffs + [coeffs[-1]] * (num_iterations - len(coeffs))
    return coeffs[:num_iterations]


def newton_schulz_orthogonalize(
    M: torch.Tensor,
    num_iterations: int = 5,
    ns_coefficients: Optional[str] = None,
) -> torch.Tensor:
    """
    Approximate orthogonalization via Newton-Schulz iteration.

    Computes the nearest orthogonal matrix to M (in Frobenius norm).
    Converges cubically — 5 iterations is typically sufficient.

    This is the core of Muon: by orthogonalizing the momentum,
    the update treats all singular directions equally.

    Args:
        M: 2D matrix to orthogonalize
        num_iterations: Number of NS iterations
        ns_coefficients: Coefficient preset name ("original", "gram_ns",
            "polar_express"). None defaults to "original".

    Optimizations from reference implementation (KellerJordan/Muon):
    - fp16 iterations (after fp32 normalization) for ~2x speedup
    - Tall-matrix transpose for faster convergence (NS converges
      faster when ncols >= nrows)
    """
    assert M.ndim == 2, f"Newton-Schulz requires 2D matrix, got {M.ndim}D"

    coeffs = _resolve_ns_coefficients(ns_coefficients, num_iterations)
    orig_dtype = M.dtype
    # Normalize in fp32 BEFORE the fp16 cast: momentum entries (~1e-7..1e-5) are
    # fp16 subnormals, and up to ~12% flush to zero if cast first.
    X = M.float()
    X = (X / X.norm().clamp_min(1e-30)).half()

    # Transpose tall matrices: NS converges faster when ncols >= nrows
    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True

    for a, b, c in coeffs:
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.to(orig_dtype)


def _batched_newton_schulz(
    M: torch.Tensor,
    num_iterations: int = 5,
    ns_coefficients: Optional[str] = None,
) -> torch.Tensor:
    """
    Batched Newton-Schulz orthogonalization.

    M: (batch, rows, cols) — all matrices must have the same shape.
    Returns: (batch, rows, cols) orthogonalized matrices.
    """
    coeffs = _resolve_ns_coefficients(ns_coefficients, num_iterations)
    orig_dtype = M.dtype
    # Per-matrix normalization in fp32 before the fp16 cast (see
    # newton_schulz_orthogonalize).
    X = M.float()
    norms = X.flatten(start_dim=-2).norm(dim=-1, keepdim=True).unsqueeze(-1)
    X = (X / norms.clamp_min(1e-30)).half()

    # Transpose if tall
    transposed = False
    if X.shape[-2] > X.shape[-1]:
        X = X.transpose(-2, -1)
        transposed = True

    for a, b, c in coeffs:
        A = X @ X.transpose(-2, -1)  # batched matmul
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.transpose(-2, -1)

    return X.to(orig_dtype)


class Muon(Optimizer):
    """
    Muon: Momentum Orthogonalized by Newton-Schulz.

    Applies to 2D weight matrices only. Uses Newton-Schulz iteration
    to orthogonalize the momentum matrix before stepping.

    Args:
        params: Parameters to optimize (must all be 2D)
        lr: Learning rate (typically 0.01-0.04 for LLM pretraining)
        momentum: Momentum coefficient (typically 0.95)
        nesterov: Whether to use Nesterov momentum
        weight_decay: Decoupled weight decay coefficient
        ns_iterations: Number of Newton-Schulz iterations (5 is standard)
        ns_coefficients: Coefficient preset for NS iteration. Options:
            "original" (MoonshotAI fixed), "gram_ns" (Dao-AILab per-iteration),
            "polar_express" (Dao-AILab conservative). None defaults to "original".
        fp32_master: keep an fp32 master copy + fp32 momentum per parameter and
            apply weight decay and the update there; the (bf16) parameter is
            refreshed from the master every step. Without it, on a bf16 model the
            decay factor (1 - lr*wd ≈ 0.9998) rounds back to 1.0 for every element
            (weight decay is a no-op) and late-training updates fall below the
            bf16 spacing of the grown weights. Off = the legacy bf16 path, kept
            bit-identical so existing lineages resume unchanged.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 0.03,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        ns_iterations: int = 5,
        ns_coefficients: Optional[str] = None,
        distributed: bool = False,
        fp32_master: bool = False,
    ) -> None:
        self._fp32_master = fp32_master
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_iterations=ns_iterations,
        )
        super().__init__(params, defaults)
        self._ns_coefficients = ns_coefficients
        # Distributed NS: under DDP every rank holds identical grads, so
        # orthogonalizing every matrix on every rank is world_size-times
        # redundant work. When enabled, matrices are partitioned across ranks
        # (greedy by numel), each rank runs NS on its subset, and results are
        # broadcast — mathematically identical to local compute (same input
        # bytes, same kernel, one owner). Incompatible with TP-sharded params
        # (falls back to local compute with a one-time warning).
        self._distributed = distributed
        self._warned_tp_fallback = False
        _resolve_ns_coefficients(ns_coefficients, ns_iterations)  # validate early

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["ns_coefficients"] = self._ns_coefficients
        return state

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state_dict = dict(state_dict)
        ns_coeff = state_dict.pop("ns_coefficients", None)
        if self._fp32_master:
            self._load_fp32_state(state_dict)
        else:
            super().load_state_dict(state_dict)
        if ns_coeff is not None:
            if self._ns_coefficients is not None and self._ns_coefficients != ns_coeff:
                logger.warning(
                    "Muon ns_coefficients mismatch: checkpoint has '%s', "
                    "CLI has '%s'. Using checkpoint value for consistency.",
                    ns_coeff, self._ns_coefficients,
                )
            self._ns_coefficients = ns_coeff

    def _load_fp32_state(self, state_dict: dict[str, Any]) -> None:
        """Dtype-preserving load: torch's default casts float state to the param
        dtype (bf16), which would silently re-break the master path on resume."""
        groups, saved_groups = self.param_groups, state_dict["param_groups"]
        if len(groups) != len(saved_groups) or any(
            len(g["params"]) != len(sg["params"]) for g, sg in zip(groups, saved_groups)
        ):
            raise ValueError("loaded Muon state dict has a different parameter layout")
        saved_state = state_dict["state"]
        legacy = False
        for group, saved_group in zip(groups, saved_groups):
            for k, v in saved_group.items():
                if k != "params":
                    group[k] = v
            for p, idx in zip(group["params"], saved_group["params"]):
                s = saved_state.get(idx, saved_state.get(str(idx)))
                if not s:
                    continue
                if "master" in s:
                    master = s["master"].detach().to(device=p.device, dtype=torch.float32)
                else:
                    legacy = True
                    master = p.detach().clone().to(torch.float32)
                buf = s["momentum_buffer"].detach().to(device=p.device, dtype=torch.float32)
                self.state[p] = {"momentum_buffer": buf, "master": master}
                p.copy_(master.to(p.dtype))
        if legacy:
            logger.warning(
                "Muon fp32_master: checkpoint had no fp32 masters (legacy bf16 state) — "
                "masters initialized from the current params, momentum upcast."
            )

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]
            ns_iterations = group["ns_iterations"]

            # Collect params with gradients and update momentum buffers
            params_with_grads: list[torch.Tensor] = []
            updates_to_orthogonalize: list[torch.Tensor] = []
            scales: list[float] = []
            tp_infos: list[Optional[dict[str, Any]]] = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    if self._fp32_master:
                        state["momentum_buffer"] = torch.zeros_like(grad, dtype=torch.float32)
                        state["master"] = p.detach().clone().to(torch.float32)
                    else:
                        state["momentum_buffer"] = torch.zeros_like(grad)

                if self._fp32_master:
                    grad = grad.float()
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                if nesterov:
                    update = momentum * buf + grad
                else:
                    update = buf.clone()

                # TP: all-gather to full matrix for NS, compute scale from full shape
                tp_info: Optional[dict[str, Any]] = None
                if hasattr(p, "_tp_shard_dim"):
                    from kotodama.training.tensor_parallel import all_gather_along_dim
                    tp_info = {
                        "shard_dim": p._tp_shard_dim,
                        "group": p._tp_group,
                        "tp_rank": p._tp_rank,
                        "local_size": update.shape[p._tp_shard_dim],
                    }
                    update = all_gather_along_dim(update, p._tp_shard_dim, p._tp_group)

                updates_to_orthogonalize.append(update)
                params_with_grads.append(p)
                tp_infos.append(tp_info)
                scales.append(max(1.0, (update.shape[0] / update.shape[1]) ** 0.5))

            if not params_with_grads:
                continue

            # Decide whether this step runs distributed NS (see __init__ note).
            use_dist_ns = False
            if self._distributed:
                import torch.distributed as dist
                if any(info is not None for info in tp_infos):
                    if not self._warned_tp_fallback:
                        logger.warning(
                            "Muon distributed NS disabled: TP-sharded params present"
                        )
                        self._warned_tp_fallback = True
                elif dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                    use_dist_ns = True

            owned: Optional[set[int]] = None
            if use_dist_ns:
                # Greedy numel-balanced ownership — deterministic (same param
                # order and sizes on every rank), so all ranks agree without
                # communication.
                world_size = dist.get_world_size()
                my_rank = dist.get_rank()
                loads = [0] * world_size
                owner_of: list[int] = [0] * len(updates_to_orthogonalize)
                order = sorted(
                    range(len(updates_to_orthogonalize)),
                    key=lambda i: (-updates_to_orthogonalize[i].numel(), i),
                )
                for i in order:
                    r = loads.index(min(loads))
                    owner_of[i] = r
                    loads[r] += updates_to_orthogonalize[i].numel()
                owned = {i for i in range(len(owner_of)) if owner_of[i] == my_rank}

            # Batch NS by shape: group matrices with same shape
            # and orthogonalize them as a single batched operation.
            # Under TP, updates are already gathered to full shape,
            # so batching groups match the non-TP case.
            shape_groups: dict[tuple[int, int], list[int]] = {}
            for i, M in enumerate(updates_to_orthogonalize):
                if owned is not None and i not in owned:
                    continue
                shape = (M.shape[0], M.shape[1])
                if shape not in shape_groups:
                    shape_groups[shape] = []
                shape_groups[shape].append(i)

            orthogonalized: list[Optional[torch.Tensor]] = [None] * len(updates_to_orthogonalize)

            for shape, indices in shape_groups.items():
                if len(indices) == 1:
                    idx = indices[0]
                    orthogonalized[idx] = newton_schulz_orthogonalize(
                        updates_to_orthogonalize[idx],
                        num_iterations=ns_iterations,
                        ns_coefficients=self._ns_coefficients,
                    )
                else:
                    batch = torch.stack(
                        [updates_to_orthogonalize[i] for i in indices]
                    )
                    result = _batched_newton_schulz(
                        batch, ns_iterations,
                        ns_coefficients=self._ns_coefficients,
                    )
                    for j, idx in enumerate(indices):
                        orthogonalized[idx] = result[j]

            if use_dist_ns:
                # Non-owners receive; owners send. contiguous() — NS may
                # return transposed views, and broadcast requires dense.
                handles = []
                for i in range(len(updates_to_orthogonalize)):
                    if orthogonalized[i] is None:
                        orthogonalized[i] = torch.empty_like(updates_to_orthogonalize[i])
                    else:
                        orthogonalized[i] = orthogonalized[i].contiguous()
                    handles.append(
                        dist.broadcast(orthogonalized[i], src=owner_of[i], async_op=True)
                    )
                for h in handles:
                    h.wait()

            # Apply updates — slice TP params back to local shard
            for i, p in enumerate(params_with_grads):
                orth = orthogonalized[i]

                if tp_infos[i] is not None:
                    info = tp_infos[i]
                    start = info["tp_rank"] * info["local_size"]
                    orth = orth.narrow(
                        info["shard_dim"], start, info["local_size"]
                    ).contiguous()

                if self._fp32_master:
                    master = self.state[p]["master"]
                    if weight_decay > 0:
                        master.mul_(1 - lr * weight_decay)
                    master.add_(orth.to(torch.float32), alpha=-lr * scales[i])
                    p.copy_(master.to(p.dtype))
                    continue
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)
                p.add_(orth, alpha=-lr * scales[i])

        return loss


# AttnRes routing parameters by name: a plain query or UnitRoutingQuery's direction, and its temperature.
_RES_QUERY_RE = re.compile(r"_res_query(\.original)?$")
_RES_TEMP_RE = re.compile(r"_res_query\.0\.temperature$")


def build_hybrid_optimizer(
    model: torch.nn.Module,
    muon_lr: float = 0.03,
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 0.01,
    muon_ns_iterations: int = 5,
    muon_ns_coefficients: Optional[str] = None,
    muon_distributed: bool = False,
    adamw_lr: float = 6e-4,
    adamw_betas: tuple[float, float] = (0.9, 0.95),
    adamw_weight_decay: float = 0.1,
    adamw_no_decay_norms: bool = False,
    muon_fp32_master: bool = False,
    attn_group: bool = False,
    attn_layers: Optional[set[int]] = None,
    res_query_lr_scale: float = 1.0,
) -> tuple[Muon, "MasterAdamW"]:
    """
    Build hybrid Muon + AdamW optimizer for a transformer model.

    Muon handles all 2D weight matrices (Q/K/V/O projections, FFN weights).
    AdamW handles everything else (embeddings, norms, LM head if untied).

    ``adamw_no_decay_norms`` puts the RMSNorm/QK-norm gains in a second AdamW
    group with weight_decay=0: a gain's neutral value is 1, so decay toward 0
    dims channels (and softens attention via the QK gains) rather than
    regularizing. Off by default — a two-group state dict cannot resume a
    one-group checkpoint, and existing lineages must keep resuming.

    ``attn_group`` puts the attention projections (q/k/v/o) in their own Muon group tagged
    ``"attn"`` so :class:`HybridScheduler` can scale their LR separately (``attn_lr_scale``,
    e.g. protecting NCA-formed attention through a handoff). ``attn_layers`` restricts the group
    to those layer indices (e.g. shallow layers frozen during an NCA phase so its circuit forms
    deeper). Off by default — like the norm group, a two-group Muon state dict cannot resume a
    one-group checkpoint.

    ``res_query_lr_scale`` != 1 puts the AttnRes routing queries in their own AdamW group whose LR is that multiple
    of the AdamW LR (``lr_scale``, applied by :class:`HybridScheduler`). UnitRoutingQuery temperatures (the
    ``attn_res_query_norm`` model option) always get their own group, without weight decay (decay would pull every
    site's sharpness toward uniform routing), at the same ``lr_scale``.

    Returns separate optimizers that should both be stepped each training step.
    """
    if res_query_lr_scale <= 0:
        raise ValueError(f"res_query_lr_scale must be > 0, got {res_query_lr_scale}")
    muon_params = []
    attn_params = []
    adamw_params = []
    norm_params = []
    query_params = []
    temp_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2 and "embed" not in name and "lm_head" not in name:
            m_layer = re.search(r"layers\.(\d+)\.attn\.", name)
            if (attn_group and m_layer and name.endswith("_proj.weight")
                    and (attn_layers is None or int(m_layer.group(1)) in attn_layers)):
                attn_params.append(param)
            else:
                muon_params.append(param)
        elif _RES_TEMP_RE.search(name):
            temp_params.append(param)
        elif res_query_lr_scale != 1.0 and _RES_QUERY_RE.search(name):
            query_params.append(param)
        elif adamw_no_decay_norms and param.ndim == 1 and name.endswith("norm.weight"):
            norm_params.append(param)
        else:
            adamw_params.append(param)
    if res_query_lr_scale != 1.0 and not query_params:
        raise ValueError("res_query_lr_scale != 1 but the model has no AttnRes routing queries")

    if attn_group and not attn_params:
        raise ValueError("attn_group=True but no attention projection weights (*.attn.*_proj.weight) found")
    muon_count = sum(p.numel() for p in muon_params) + sum(p.numel() for p in attn_params)
    adamw_count = sum(p.numel() for p in adamw_params + norm_params + query_params + temp_params)
    total = muon_count + adamw_count

    # Only print on rank 0 (or when not distributed)
    try:
        import torch.distributed as dist
        should_print = not dist.is_initialized() or dist.get_rank() == 0
    except Exception:
        should_print = True

    if should_print:
        print(f"Muon params:  {muon_count:>12,} ({muon_count/total*100:.1f}%)")
        print(f"AdamW params: {adamw_count:>12,} ({adamw_count/total*100:.1f}%)")
        if norm_params:
            print(f"  of which no-decay norm gains: {len(norm_params)} tensors")
        if query_params:
            print(f"  of which AttnRes routing queries (LR x{res_query_lr_scale:g}): {len(query_params)} tensors")
        if temp_params:
            print(f"  of which AttnRes routing temperatures (no decay, LR x{res_query_lr_scale:g}): "
                  f"{len(temp_params)} tensors")

    muon_groups: Any = muon_params
    if attn_group:
        muon_groups = [{"params": muon_params, "tag": "main"}, {"params": attn_params, "tag": "attn"}]
        if should_print:
            print(f"  of which attention projections (own group 'attn'): {len(attn_params)} tensors")
    muon_opt = Muon(
        muon_groups,
        lr=muon_lr,
        momentum=muon_momentum,
        nesterov=True,
        weight_decay=muon_weight_decay,
        ns_iterations=muon_ns_iterations,
        ns_coefficients=muon_ns_coefficients,
        distributed=muon_distributed,
        fp32_master=muon_fp32_master,
    )

    adamw_groups: list[dict[str, Any]] = [{"params": adamw_params}]
    if norm_params:
        adamw_groups.append({"params": norm_params, "weight_decay": 0.0})
    if query_params:
        adamw_groups.append({"params": query_params, "lr_scale": res_query_lr_scale})
    if temp_params:
        adamw_groups.append({"params": temp_params, "weight_decay": 0.0, "lr_scale": res_query_lr_scale})
    adamw_opt = MasterAdamW(
        adamw_groups,
        lr=adamw_lr,
        betas=adamw_betas,
        weight_decay=adamw_weight_decay,
    )

    return muon_opt, adamw_opt


class MasterAdamW(Optimizer):
    """AdamW with fp32 master weights + fp32 moments for low-precision params.

    THE BUG THIS EXISTS FOR: under pure-bf16 training, AdamW's ~6e-4 steps on
    RMSNorm/QK-norm gains sitting at 1.0 round away entirely (bf16 spacing at
    1.0 is 0.0039), so every norm gain stays BIT-FROZEN at its init for the
    whole run (the shipped 3B's did). Here updates accumulate in an fp32
    master; the (possibly bf16) module param is refreshed from the master each
    step, so quantization cannot eat sub-spacing progress. The Muon side is unaffected by design:
    its orthogonalized updates are orders of magnitude above bf16 spacing
    (verified empirically — the 3B's matrices drifted on schedule).

    Semantics match ``torch.optim.AdamW`` (decoupled weight decay, bias
    correction) so fp32 trajectories are comparable; the master IS the
    trajectory, the param is its low-precision view.

    ``load_state_dict`` is custom: torch's default policy casts floating
    state to the PARAM dtype on load, which would silently downcast the
    masters to bf16 on every resume — reintroducing the bug through
    checkpointing. Legacy states from stock AdamW load gracefully: masters
    initialize from the current (quantized) params and moments upcast, with
    a warning.
    """

    def __init__(self, params: Any, lr: float = 6e-4,
                 betas: tuple[float, float] = (0.9, 0.95),
                 eps: float = 1e-8, weight_decay: float = 0.1) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["master"] = p.detach().clone().to(torch.float32)
                    state["exp_avg"] = torch.zeros_like(state["master"])
                    state["exp_avg_sq"] = torch.zeros_like(state["master"])
                master = state["master"]
                m, v = state["exp_avg"], state["exp_avg_sq"]
                grad = p.grad.detach().to(torch.float32)
                state["step"] += 1
                t = state["step"]
                if wd != 0:
                    master.mul_(1.0 - lr * wd)
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                denom = (v.sqrt() / (bc2 ** 0.5)).add_(eps)
                master.addcdiv_(m, denom, value=-lr / bc1)
                p.copy_(master.to(p.dtype))
        return loss

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Dtype-preserving load (masters/moments stay fp32; legacy upcasts)."""
        groups = self.param_groups
        saved_groups = state_dict["param_groups"]
        if len(groups) != len(saved_groups) or any(
            len(g["params"]) != len(sg["params"])
            for g, sg in zip(groups, saved_groups)
        ):
            raise ValueError("loaded state dict has a different parameter layout")
        saved_state = state_dict["state"]
        legacy = False
        for group, saved_group in zip(groups, saved_groups):
            for k, v in saved_group.items():
                if k != "params":
                    group[k] = v
            for p, idx in zip(group["params"], saved_group["params"]):
                s = saved_state.get(idx, saved_state.get(str(idx)))
                if not s:
                    continue
                step = s.get("step", 0)
                new: dict[str, Any] = {
                    "step": int(step.item() if torch.is_tensor(step) else step)
                }
                if "master" in s:
                    new["master"] = s["master"].detach().to(
                        device=p.device, dtype=torch.float32)
                else:  # stock-AdamW checkpoint
                    legacy = True
                    new["master"] = p.detach().clone().to(torch.float32)
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in s:
                        new[key] = s[key].detach().to(
                            device=p.device, dtype=torch.float32)
                    else:
                        new[key] = torch.zeros_like(new["master"])
                self.state[p] = new
                p.copy_(new["master"].to(p.dtype))
        if legacy:
            logger.warning(
                "MasterAdamW: legacy optimizer state had no fp32 masters — "
                "initialized them from the current (quantized) params; "
                "sub-spacing progress before this resume is not recoverable."
            )


def decay_start_step(total_steps: int, decay_start_pct: float) -> int:
    """First step of the WSD decay phase: the LR multiplier is 1.0 (after warmup)
    for steps below it and decays to zero from it to ``total_steps``."""
    return int(total_steps * decay_start_pct)


class HybridScheduler:
    """
    WSD (Warmup-Stable-Decay) schedule applied to both Muon and AdamW.

    Both optimizers follow the same schedule shape but with different base LRs.
    """

    def __init__(
        self,
        muon_opt: Muon,
        adamw_opt: AdamW,
        warmup_steps: int = 2000,
        total_steps: int = 100000,
        decay_start_pct: float = 0.90,
        decay_type: str = "sqrt",
        attn_lr_scale: float = 1.0,
        attn_lr_scale_steps: int = 0,
    ) -> None:
        if attn_lr_scale < 0:
            raise ValueError(f"attn_lr_scale must be >= 0, got {attn_lr_scale}")
        if attn_lr_scale != 1.0 and attn_lr_scale_steps > 0 and not any(
                g.get("tag") == "attn" for g in muon_opt.param_groups):
            raise ValueError("attn_lr_scale needs the optimizer built with attn_group=True")
        self.attn_lr_scale = attn_lr_scale
        self.attn_lr_scale_steps = attn_lr_scale_steps
        self.muon_opt = muon_opt
        self.adamw_opt = adamw_opt
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_start_step = decay_start_step(total_steps, decay_start_pct)
        self.decay_type = decay_type

        # Store base LRs
        self.muon_base_lr = muon_opt.defaults["lr"]
        self.adamw_base_lr = adamw_opt.defaults["lr"]

    def get_lr_multiplier(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup
            return step / self.warmup_steps
        elif step < self.decay_start_step:
            # Stable phase
            return 1.0
        else:
            # Decay to zero
            decay_steps = self.total_steps - self.decay_start_step
            progress = (step - self.decay_start_step) / max(decay_steps, 1)

            if self.decay_type == "sqrt":
                return 1.0 - progress ** 0.5
            elif self.decay_type == "linear":
                return 1.0 - progress
            elif self.decay_type == "cosine":
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
            else:
                raise ValueError(f"Unknown decay type: {self.decay_type}")

    def step(self, step: int) -> None:
        mult = self.get_lr_multiplier(step)
        for group in self.muon_opt.param_groups:
            scale = (self.attn_lr_scale if group.get("tag") == "attn" and step < self.attn_lr_scale_steps
                     else 1.0)
            group["lr"] = self.muon_base_lr * mult * scale
        for group in self.adamw_opt.param_groups:
            group["lr"] = self.adamw_base_lr * mult * group.get("lr_scale", 1.0)

    def get_last_lr(self) -> dict[str, float]:
        muon_lr = (
            self.muon_opt.param_groups[0]["lr"]
            if self.muon_opt.param_groups
            else 0.0
        )
        adamw_lr = (
            self.adamw_opt.param_groups[0]["lr"]
            if self.adamw_opt.param_groups
            else 0.0
        )
        return {"muon_lr": muon_lr, "adamw_lr": adamw_lr}
