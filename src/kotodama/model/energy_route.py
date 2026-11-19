"""Energy routing for Block AttnRes: the routing weights at a site minimise a learned energy over the simplex.

Today's router is one softmax of a static query against the RMS-normalised sources. That softmax is the exact
minimiser of E0(a) = -sum_s a_s l_s + sum_s a_s log a_s, with l_s the routing logits. Energy routing adds a term that
scores the mixture of sources rather than each source alone:

    E(a) = E0(a) - 1/2 * sum_j sigma_j c_j^2,    c = sum_s a_s P_s,    P_s = U^T k_s,

where k_s is the normalised key the router already computes (norm gain * RMS-normalised source), U (D x r) is shared by
every site and sigma (r) is per site. sigma > 0 rewards agreement between the chosen sources, sigma < 0 rewards
complementary ones. ``steps`` rounds of mirror descent with unit step on the simplex have the closed form

    a^(0) = softmax(l),    a^(k+1) = softmax(l + P (sigma * c^(k))),

whose fixed point is a stationary point of E. The gradient of E is analytic, so training needs no second-order
autodiff. With sigma = 0 (its initialisation) the weights are exactly today's softmax.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def project_sources(sources: torch.Tensor, rsqrt: torch.Tensor, norm_weight: torch.Tensor,
                    U: torch.Tensor) -> torch.Tensor:
    """P_s = U^T (norm_weight * source_s * rsqrt_s), without materialising the normalised sources.

    sources: (S, ..., D); rsqrt: (S, ...); norm_weight: (D,); U: (D, r). Returns (S, ..., r)."""
    w = (norm_weight.unsqueeze(-1) * U).to(sources.dtype)                     # (D, r)
    return torch.matmul(sources, w) * rsqrt.unsqueeze(-1).to(sources.dtype)


def energy_weights(logits: torch.Tensor, P: torch.Tensor, sigma: torch.Tensor,
                   steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Routing weights after ``steps`` mirror-descent steps on E. Returns (a_final, a_0), each (S, ...).

    logits: (S, ...), -inf for inactive sources (they stay at weight 0); P: (S, ..., r); sigma: (r,)."""
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")
    dt = torch.promote_types(logits.dtype, torch.float32)   # at least fp32; fp64 stays fp64 (tests)
    lf = logits.to(dt)
    Pf = P.to(dt)
    sig = sigma.to(dt)
    a0 = F.softmax(lf, dim=0)
    a = a0
    for _ in range(steps):
        c = (a.unsqueeze(-1) * Pf).sum(0)                                      # (..., r)
        shift = (Pf * (sig * c).unsqueeze(0)).sum(-1)                          # (S, ...)
        a = F.softmax(lf + shift, dim=0)
    return a.to(logits.dtype), a0.to(logits.dtype)


def energy(a: torch.Tensor, logits: torch.Tensor, P: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """E(a) per token, (...): for tests and diagnostics. a must be strictly positive where logits are finite."""
    finite = torch.isfinite(logits)
    lin = torch.where(finite, a * logits, torch.zeros_like(a)).sum(0)
    ent = torch.where(a > 0, a * torch.log(a.clamp_min(1e-30)), torch.zeros_like(a)).sum(0)
    c = (a.unsqueeze(-1) * P).sum(0)
    return -lin + ent - 0.5 * (sigma * c * c).sum(-1)


def energy_grad(a: torch.Tensor, logits: torch.Tensor, P: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Closed-form dE/da, (S, ...): -l + log a + 1 - P (sigma * c)."""
    c = (a.unsqueeze(-1) * P).sum(0)
    return -logits + torch.log(a.clamp_min(1e-30)) + 1.0 - (P * (sigma * c).unsqueeze(0)).sum(-1)


def route_energy(sources: torch.Tensor, qw: torch.Tensor, norm_weight: torch.Tensor, eps: float,
                 U: torch.Tensor, sigma: torch.Tensor, steps: int,
                 active_mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The routed mixture over ``sources`` (S, B, T, D) with energy routing.

    qw = query * norm_weight (the router's pre-weighted query, as today). Returns (h, a_final, a_0); h is (B, T, D) and
    the weights are (S, B, T). ``active_mask`` (S,) marks real sources in a padded buffer (None: all are real)."""
    # The (S, B, T, D) sources are only ever read by reductions (matmuls), never copied: memory as today's router.
    rsqrt, logits, P = source_terms(sources, qw, norm_weight, eps, U)
    if active_mask is not None:
        logits = logits.masked_fill(~active_mask.view(-1, *([1] * (logits.dim() - 1))), float("-inf"))
    a, a0 = energy_weights(logits, P, sigma, steps)
    h = torch.einsum("s...,s...d->...d", a.to(sources.dtype), sources)
    return h, a, a0


def source_terms(sources: torch.Tensor, qw: torch.Tensor, norm_weight: torch.Tensor, eps: float,
                 U: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(rsqrt, logits, P) for ``sources`` (S, ..., D), as small fp32 tensors (S, ...), (S, ...), (S, ..., r).

    logits match today's router: (source . qw) * rsqrt(mean(source^2) + eps)."""
    D = sources.shape[-1]
    ms = torch.linalg.vector_norm(sources, dim=-1).float().pow(2) / D
    rsqrt = torch.rsqrt(ms + eps)
    logits = torch.matmul(sources, qw.to(sources.dtype)).float() * rsqrt
    P = project_sources(sources, rsqrt, norm_weight, U).float()
    return rsqrt, logits, P


def route_energy_split(committed: torch.Tensor, partial: torch.Tensor, qw: torch.Tensor, norm_weight: torch.Tensor,
                       eps: float, U: torch.Tensor, sigma: torch.Tensor, steps: int) -> torch.Tensor:
    """route_energy over [committed (n, B, T, D)..., partial (B, T, D)] without concatenating the big tensors.

    Only the small per-source terms (logits, P) are joined; the mixture is the committed einsum plus the partial's
    weighted term. Returns h (B, T, D)."""
    _rc, lc, Pc = source_terms(committed, qw, norm_weight, eps, U)
    _rp, lp, Pp = source_terms(partial.unsqueeze(0), qw, norm_weight, eps, U)
    a, _a0 = energy_weights(torch.cat([lc, lp], 0), torch.cat([Pc, Pp], 0), sigma, steps)
    a = a.to(partial.dtype)
    n = committed.shape[0]
    return torch.einsum("s...,s...d->...d", a[:n], committed) + a[n].unsqueeze(-1) * partial
