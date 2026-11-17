"""Cost-balanced window assignment across data-parallel ranks, for document-masked (varlen) training.

Every micro-step ends in collectives (FSDP2's reduce-scatter, DDP's all-reduce), so the step waits for the rank whose
windows cost the most. Attention over a window costs about Σ L² over its documents, so a rank that drew a few long
documents runs much longer than one that drew many short ones. Here the ranks pool one micro-step's windows and deal
them out again so every rank's attention cost is level:

- the same set of windows trains in the same step (only which rank holds which changes);
- every rank still holds exactly ``micro_batch`` windows, so each rank's mean loss covers the same number of tokens and
  the average over ranks is the batch mean, as before;
- the deal is a pure function of the pooled windows (costs, then global index for ties), so a resume deals the same
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


def rebalance_batch(batch: dict[str, Any], seq_len: int, device: torch.device, group: Optional[Any] = None,
                    eos_id: int = 0, stats: Optional[dict[str, float]] = None) -> dict[str, Any]:
    """This rank's share of the pooled micro-step after a cost-levelling deal (see the module docstring). ``batch`` is
    a :func:`collate_packed` batch of whole windows of ``seq_len`` tokens; the result is the same kind of batch, on the
    CPU like its input. ``stats``, when given, receives the slowest rank's attention cost over the mean rank's, before
    and after the deal (``balance/cost_max_over_mean_{before,after}``). Collective over ``group`` (the data-parallel
    ranks); a no-op on one rank."""
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size(group) < 2:
        return batch
    ids = batch["input_ids"].reshape(-1)
    if ids.numel() % seq_len:
        raise ValueError(f"a micro-batch of {ids.numel()} tokens is not whole windows of {seq_len}")
    windows = ids.view(-1, seq_len)
    n_local = windows.shape[0]
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)

    local_np = windows.numpy()
    costs = torch.tensor([window_attention_cost(w, eos_id) for w in local_np], dtype=torch.float64, device=device)
    all_costs = [torch.empty_like(costs) for _ in range(world)]
    dist.all_gather(all_costs, costs, group=group)
    pooled_costs = torch.cat(all_costs).tolist()  # global window g = rank * n_local + j

    deal = assign_equal_count_lpt(pooled_costs, world)
    mine = deal[rank]
    if stats is not None:
        mean = sum(pooled_costs) / world
        before = max(sum(pooled_costs[r * n_local:(r + 1) * n_local]) for r in range(world))
        after = max(sum(pooled_costs[g] for g in held) for held in deal)
        stats["balance/cost_max_over_mean_before"] = before / mean if mean > 0 else 1.0
        stats["balance/cost_max_over_mean_after"] = after / mean if mean > 0 else 1.0
    local_dev = windows.to(device=device, dtype=torch.int32)
    all_ids = [torch.empty_like(local_dev) for _ in range(world)]
    dist.all_gather(all_ids, local_dev, group=group)
    pooled = torch.cat(all_ids).cpu()

    samples = []
    for g in mine:
        raw = pooled[g].numpy()
        cu, pos, max_len = compute_doc_boundaries(raw, eos_id=eos_id)
        samples.append(PackedSample(input_ids=pooled[g].to(torch.int64), cu_seqlens=torch.from_numpy(cu),
                                    position_ids=torch.from_numpy(pos).long(), max_seqlen=max_len))
    assert len(samples) == n_local
    return collate_packed(samples)
