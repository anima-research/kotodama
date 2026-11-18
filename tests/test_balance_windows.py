"""Cost-balanced window assignment: the step's windows are pooled over ranks and micro-steps, every rank keeps
micro_batch windows per micro-step, the step trains the same windows, the deal is deterministic, and the received
windows carry the document boundaries the dataset would give them."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kotodama.data.dataset import PackedSample, collate_packed, compute_doc_boundaries
from kotodama.training.balance import (assign_equal_count_lpt, plan_step, rebalance_step, step_wait_ratio,
                                        window_attention_cost)

SEQ = 32


def test_window_cost_is_the_sum_of_squared_document_lengths():
    w = np.array([5] * 10 + [0] + [7] * 20 + [0], dtype=np.int64)  # docs of 11 and 21 tokens (EOS ends a doc)
    cu, _, _ = compute_doc_boundaries(w, eos_id=0)
    assert window_attention_cost(w) == float((np.diff(cu) ** 2).sum())


def test_equal_count_lpt_levels_cost_and_keeps_counts():
    costs = [100.0, 1.0, 1.0, 1.0, 90.0, 1.0, 1.0, 1.0]
    held = assign_equal_count_lpt(costs, 2)
    assert sorted(i for h in held for i in h) == list(range(8))
    assert all(len(h) == 4 for h in held)
    loads = [sum(costs[i] for i in h) for h in held]
    assert max(loads) - min(loads) <= 10.0
    assert held == assign_equal_count_lpt(costs, 2)  # deterministic
    with pytest.raises(ValueError, match="evenly"):
        assign_equal_count_lpt(costs[:7], 2)


def _identity(n_ranks: int, n_micro: int, per_micro: int) -> list[list[list[int]]]:
    n_local = n_micro * per_micro
    return [[list(range(r * n_local + m * per_micro, r * n_local + (m + 1) * per_micro)) for m in range(n_micro)]
            for r in range(n_ranks)]


def test_step_plan_keeps_counts_and_pools_across_micro_steps():
    rng = np.random.default_rng(0)
    ranks, micro, per = 6, 10, 1  # long context: one window per rank per micro-step
    costs = list(rng.pareto(1.2, ranks * micro * per) + 1.0)
    plan = plan_step(costs, ranks, micro)
    assert sorted(g for r in plan for m in r for g in m) == list(range(len(costs)))  # the same windows
    assert all(len(plan[r][m]) == per for r in range(ranks) for m in range(micro))
    assert plan == plan_step(costs, ranks, micro)  # deterministic
    before = step_wait_ratio(costs, _identity(ranks, micro, per))
    after = step_wait_ratio(costs, plan)
    assert after < before and after >= 1.0
    # a per-micro-step deal can't move anything at one window per rank: pooling the step is what helps
    per_micro_only = [[assign_equal_count_lpt([costs[g] for g in [_identity(ranks, micro, per)[q][m][0]
                                                                   for q in range(ranks)]], ranks)[r]
                       for m in range(micro)] for r in range(ranks)]
    assert all(len(x) == 1 for row in per_micro_only for x in row)


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


N_MICRO, PER = 3, 2


def _rebalance_worker(rank: int, world: int, init_file: str, out_dir: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    ws = _windows(rank, N_MICRO * PER)
    step = [_batch(ws[m * PER:(m + 1) * PER]) for m in range(N_MICRO)]
    stats: dict[str, float] = {}
    a = rebalance_step(step, SEQ, torch.device("cpu"), stats=stats)
    b = rebalance_step(step, SEQ, torch.device("cpu"))
    assert 1.0 <= stats["balance/step_wait_ratio_after"] <= stats["balance/step_wait_ratio_before"]
    assert 0.0 <= stats["balance/windows_moved_frac"] <= 1.0
    assert len(a) == N_MICRO
    rows = []
    for x, y in zip(a, b):
        for k in ("input_ids", "cu_seqlens", "position_ids"):
            assert torch.equal(x[k], y[k]), k  # deterministic
        r = x["input_ids"].view(-1, SEQ)
        assert r.shape[0] == PER  # micro_batch windows per micro-step, every rank
        want = _batch([w.numpy() for w in r])  # the boundaries the dataset gives these very windows
        for k in ("cu_seqlens", "position_ids"):
            assert torch.equal(x[k], want[k]), k
        assert x["max_seqlen"] == want["max_seqlen"]
        rows.append(r)
    torch.save(torch.cat(rows), f"{out_dir}/rank{rank}.pt")
    dist.destroy_process_group()


def test_rebalanced_step_holds_exactly_the_pooled_windows(tmp_path):
    world = 3
    mp.spawn(_rebalance_worker, args=(world, str(tmp_path / "pg"), str(tmp_path)), nprocs=world, join=True)
    got = sorted(tuple(r.tolist()) for k in range(world) for r in torch.load(tmp_path / f"rank{k}.pt"))
    pooled = sorted(tuple(w.tolist()) for k in range(world) for w in _windows(k, N_MICRO * PER))
    assert got == pooled
