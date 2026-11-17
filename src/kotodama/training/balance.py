"""Cost-balanced window assignment across data-parallel ranks, for document-masked (varlen) training.

Every micro-step ends in collectives (FSDP2's reduce-scatter, DDP's all-reduce), so each micro-step waits for the rank
whose windows cost the most. Attention over a window costs about Σ L² over its documents, so a rank that drew a few
long documents runs much longer than one that drew many short ones. Here the ranks pool a whole optimizer step's
windows (every rank's, every micro-step's) and deal them out again: sorted by cost, the heaviest ``world ×
micro_batch`` windows train together in one micro-step, the next heaviest in the next, and so on, and inside each
micro-step the windows go longest first to the least-loaded rank that still has room. Pooling the step, not one
micro-step, is what makes this work at long context, where a micro-step holds one window per rank.

- The step trains the same windows (only which rank and micro-step holds which changes), so the update is the same up
  to summation order.
- Every rank still holds exactly ``micro_batch`` windows in every micro-step, so each rank's mean loss covers the same
  number of tokens and the average over ranks and micro-steps is the batch mean, as before.
- The deal is a pure function of the pooled windows (costs, then global index for ties), so a resume deals the same
  windows the same way; each rank's dataset position is untouched.

The windows' documents are found again from the EOS tokens on the receiving rank (:func:`compute_doc_boundaries`, as
the dataset does), so only token ids move between ranks.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist

from kotodama.data.dataset import PackedSample, collate_packed, compute_doc_boundaries


def window_attention_cost(tokens: np.ndarray, eos_id: int = 0) -> float:
    """Σ L² over the documents of one window (the varlen attention cost, up to a constant)."""
    cu, _, _ = compute_doc_boundaries(tokens, eos_id=eos_id)
    lengths = np.diff(cu).astype(np.float64)
    return float((lengths * lengths).sum())


def assign_equal_count_lpt(costs: Sequence[float], n_ranks: int) -> list[list[int]]:
    """Deal ``len(costs)`` windows to ``n_ranks`` ranks, ``len(costs) // n_ranks`` each, levelling the summed cost:
    longest first, each to the least-loaded rank that still has room (ties: lower rank). Returns each rank's window
    indices in ascending order. Deterministic in ``costs`` alone."""
    n = len(costs)
    if n_ranks < 1 or n % n_ranks:
        raise ValueError(f"{n} windows do not split evenly over {n_ranks} ranks")
    per_rank = n // n_ranks
    order = sorted(range(n), key=lambda i: (-float(costs[i]), i))
    load = [0.0] * n_ranks
    held: list[list[int]] = [[] for _ in range(n_ranks)]
    for i in order:
        r = min((r for r in range(n_ranks) if len(held[r]) < per_rank), key=lambda r: (load[r], r))
        held[r].append(i)
        load[r] += float(costs[i])
    return [sorted(h) for h in held]


def plan_step(costs: Sequence[float], n_ranks: int, n_micro: int) -> list[list[list[int]]]:
    """The deal for one optimizer step. ``costs`` holds every window of the step, indexed rank-major then micro-step
    then slot (window ``g = (rank · n_micro + micro) · per_micro + slot``). Returns ``plan[rank][micro]``, the global
    indices that rank trains in that micro-step: cost-alike windows share a micro-step, and each micro-step's windows
    are dealt equal-count by cost across the ranks. Deterministic in ``costs`` alone."""
    n = len(costs)
    if n % (n_ranks * n_micro):
        raise ValueError(f"{n} windows do not split evenly over {n_ranks} ranks x {n_micro} micro-steps")
    per_group = n // n_micro
    order = sorted(range(n), key=lambda i: (-float(costs[i]), i))
    plan: list[list[list[int]]] = [[[] for _ in range(n_micro)] for _ in range(n_ranks)]
    for m in range(n_micro):
        group = order[m * per_group:(m + 1) * per_group]
        held = assign_equal_count_lpt([costs[g] for g in group], n_ranks)
        for r in range(n_ranks):
            plan[r][m] = sorted(group[i] for i in held[r])
    return plan


def step_wait_ratio(costs: Sequence[float], layout: list[list[list[int]]]) -> float:
    """Σ over micro-steps of the slowest rank's cost, over Σ of the mean rank's: how much longer the step's attention
    takes than if every micro-step were perfectly level (1.0). ``layout[rank][micro]`` lists window indices."""
    n_ranks, n_micro = len(layout), len(layout[0])
    slow = sum(max(sum(costs[g] for g in layout[r][m]) for r in range(n_ranks)) for m in range(n_micro))
    mean = sum(costs) / n_ranks
    return slow / mean if mean > 0 else 1.0


def rebalance_step(batches: list[dict[str, Any]], seq_len: int, device: torch.device, group: Optional[Any] = None,
                   eos_id: int = 0, stats: Optional[dict[str, float]] = None) -> list[dict[str, Any]]:
    """This rank's micro-batches for one optimizer step after the cost-levelling deal (see the module docstring).
    ``batches`` are the step's :func:`collate_packed` micro-batches, each of whole windows of ``seq_len`` tokens; the
    result is as many micro-batches of the same size, on the CPU like the input. ``stats``, when given, receives
    :func:`step_wait_ratio` before and after the deal (``balance/step_wait_ratio_{before,after}``). Collective over
    ``group`` (the data-parallel ranks); a no-op on one rank."""
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size(group) < 2:
        return batches
    n_micro = len(batches)
    rows = []
    for b in batches:
        ids = b["input_ids"].reshape(-1)
        if ids.numel() % seq_len:
            raise ValueError(f"a micro-batch of {ids.numel()} tokens is not whole windows of {seq_len}")
        rows.append(ids.view(-1, seq_len))
    per_micro = rows[0].shape[0]
    if any(r.shape[0] != per_micro for r in rows):
        raise ValueError("micro-batches of one step must hold the same number of windows")
    windows = torch.cat(rows)  # local window (micro, slot) = micro * per_micro + slot
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)

    costs = torch.tensor([window_attention_cost(w, eos_id) for w in windows.numpy()], dtype=torch.float64,
                         device=device)
    all_costs = [torch.empty_like(costs) for _ in range(world)]
    dist.all_gather(all_costs, costs, group=group)
    pooled_costs = torch.cat(all_costs).tolist()
    plan = plan_step(pooled_costs, world, n_micro)
    if stats is not None:
        n_local = n_micro * per_micro
        identity = [[list(range(r * n_local + m * per_micro, r * n_local + (m + 1) * per_micro))
                     for m in range(n_micro)] for r in range(world)]
        stats["balance/step_wait_ratio_before"] = step_wait_ratio(pooled_costs, identity)
        stats["balance/step_wait_ratio_after"] = step_wait_ratio(pooled_costs, plan)

    local_dev = windows.to(device=device, dtype=torch.int32)
    all_ids = [torch.empty_like(local_dev) for _ in range(world)]
    dist.all_gather(all_ids, local_dev, group=group)
    pooled = torch.cat(all_ids).cpu()

    out = []
    for m in range(n_micro):
        samples = []
        for g in plan[rank][m]:
            raw = pooled[g].numpy()
            cu, pos, max_len = compute_doc_boundaries(raw, eos_id=eos_id)
            samples.append(PackedSample(input_ids=pooled[g].to(torch.int64), cu_seqlens=torch.from_numpy(cu),
                                        position_ids=torch.from_numpy(pos).long(), max_seqlen=max_len))
        out.append(collate_packed(samples))
    return out
