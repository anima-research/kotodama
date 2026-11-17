"""Cost-balanced window assignment: every rank keeps micro_batch windows, the pooled step trains the same windows,
the deal is deterministic, and the received windows carry the document boundaries the dataset would give them."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kotodama.data.dataset import PackedSample, collate_packed, compute_doc_boundaries
from kotodama.training.balance import assign_equal_count_lpt, rebalance_batch, window_attention_cost

SEQ = 32


def test_window_cost_is_the_sum_of_squared_document_lengths():
    w = np.array([5] * 10 + [0] + [7] * 20 + [0], dtype=np.int64)  # docs of 11 and 21 tokens (EOS ends a doc)
    cu, _, _ = compute_doc_boundaries(w, eos_id=0)
    assert window_attention_cost(w) == float((np.diff(cu) ** 2).sum())


def test_equal_count_lpt_levels_cost_and_keeps_counts():
    costs = [100.0, 1.0, 1.0, 1.0, 90.0, 1.0, 1.0, 1.0]  # identity split (4 + 4) loads 103 vs 93; one rank gets both big
    held = assign_equal_count_lpt(costs, 2)
    assert sorted(i for h in held for i in h) == list(range(8))
    assert all(len(h) == 4 for h in held)
    loads = [sum(costs[i] for i in h) for h in held]
    assert max(loads) - min(loads) <= 10.0
    assert held == assign_equal_count_lpt(costs, 2)  # deterministic
    with pytest.raises(ValueError, match="evenly"):
        assign_equal_count_lpt(costs[:7], 2)


def test_equal_count_lpt_beats_the_identity_split_on_skewed_windows():
    rng = np.random.default_rng(0)
    costs = list(rng.pareto(1.2, 16 * 8) + 1.0)
    identity = [sum(costs[r * 8:(r + 1) * 8]) for r in range(16)]
    dealt = [sum(costs[i] for i in h) for h in assign_equal_count_lpt(costs, 16)]
    assert max(dealt) < max(identity)
    # A window can cost more than a whole rank's mean load, which no deal levels; the rank that holds the biggest
    # window then gets only the cheapest leftovers (every other rank fills first).
    cheapest_rest = sum(sorted(costs)[:7])
    assert max(dealt) <= max(max(costs) + cheapest_rest, 1.5 * sum(costs) / 16)


def _windows(rank: int, n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(100 + rank)
    out = []
    for _ in range(n):
        w = rng.integers(1, 50, SEQ)
        w[rng.choice(SEQ, rng.integers(0, 6), replace=False)] = 0  # 0..5 EOS: documents of mixed length
        out.append(w.astype(np.int64))
    return out


def _batch(ws: list[np.ndarray]) -> dict:
    samples = []
    for w in ws:
        cu, pos, m = compute_doc_boundaries(w, eos_id=0)
        samples.append(PackedSample(input_ids=torch.from_numpy(w.copy()), cu_seqlens=torch.from_numpy(cu),
                                    position_ids=torch.from_numpy(pos).long(), max_seqlen=m))
    return collate_packed(samples)


def _rebalance_worker(rank: int, world: int, init_file: str, out_dir: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    mine = _batch(_windows(rank, 4))
    stats: dict[str, float] = {}
    a = rebalance_batch(mine, SEQ, torch.device("cpu"), stats=stats)
    assert stats["balance/cost_max_over_mean_after"] <= stats["balance/cost_max_over_mean_before"]
    assert stats["balance/cost_max_over_mean_after"] >= 1.0
    b = rebalance_batch(mine, SEQ, torch.device("cpu"))
    for k in ("input_ids", "cu_seqlens", "position_ids"):
        assert torch.equal(a[k], b[k]), k  # deterministic
    rows = a["input_ids"].view(-1, SEQ)
    assert rows.shape[0] == 4  # every rank keeps micro_batch windows
    want = _batch([r.numpy() for r in rows])  # the boundaries the dataset gives these very windows
    for k in ("cu_seqlens", "position_ids"):
        assert torch.equal(a[k], want[k]), k
    assert a["max_seqlen"] == want["max_seqlen"]
    torch.save(rows, f"{out_dir}/rank{rank}.pt")
    dist.destroy_process_group()


def test_rebalanced_ranks_hold_exactly_the_pooled_windows(tmp_path):
    world = 3
    mp.spawn(_rebalance_worker, args=(world, str(tmp_path / "pg"), str(tmp_path)), nprocs=world, join=True)
    got = sorted(tuple(r.tolist()) for k in range(world) for r in torch.load(tmp_path / f"rank{k}.pt"))
    pooled = sorted(tuple(w.tolist()) for k in range(world) for w in _windows(k, 4))
    assert got == pooled
    costs = [window_attention_cost(np.array(w)) for k in range(world) for w in _windows(k, 4)]
    loads = [sum(window_attention_cost(r.numpy()) for r in torch.load(tmp_path / f"rank{k}.pt")) for k in range(world)]
    identity = [sum(costs[k * 4:(k + 1) * 4]) for k in range(world)]
    assert max(loads) <= max(identity)
