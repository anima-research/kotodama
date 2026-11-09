"""Sharded geometric probes (MonitorConfig.sharded) reproduce the single-process metrics.

Two CPU processes (gloo) each hold half of a probe's rows; every sharded reduction is checked against the unsharded
computation on all the rows: RankMe through the summed Gram matrix, attention entropy, the token-mean probe stats
(including routing_entropy rebuilt from per-site mean routing weights), the EoC power iteration over block-diagonal
(per-sequence) Jacobians, and the row-sample gather.
"""

from __future__ import annotations

import math
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kotodama.monitoring import geometric
from kotodama.monitoring.geometric import GeometricMonitor, MonitorConfig

WORLD = 2


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(2)])


def _layer_fn(A: torch.Tensor):
    # Mixes tokens within a sequence (cumulative mean over T) and features (A): Jacobian block-diagonal per sequence.
    def fn(h: torch.Tensor) -> torch.Tensor:
        c = h.cumsum(dim=1) / torch.arange(1, h.shape[1] + 1, dtype=h.dtype).view(1, -1, 1)
        return torch.tanh(c @ A) + h
    return fn


def _worker(rank: int, init_file: str, out: dict) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD)
    try:
        torch.manual_seed(0)  # identical data on every rank; each keeps its own rows
        B, T, D, H = 4, 6, 8, 3
        X = torch.randn(B, T, D)
        sharded = GeometricMonitor(_Tiny(), MonitorConfig(sharded=True))
        plain = GeometricMonitor(_Tiny(), MonitorConfig(sharded=False))
        shard = sharded._shard
        assert shard.on and shard.world == WORLD
        rows = shard.rows(B)
        X_local = shard.take(X)
        assert torch.equal(X_local, X[rows])

        # RankMe: Gram over ranks == SVD of all rows
        r_sh = geometric._rankme_gram(X_local.reshape(-1, D), shard)
        r_full = geometric._rankme(X.reshape(-1, D))
        assert abs(r_sh - r_full) <= 1e-4 * r_full, (r_sh, r_full)

        # attention entropy over every rank's sequences
        A = torch.softmax(torch.randn(B, H, T, T), dim=-1)
        e_sh = geometric._attention_entropy_stats_sharded(A[rows], H, shard, torch.device("cpu"))
        e_full = geometric._attention_entropy_stats(A)
        assert abs(e_sh[0] - e_full[0]) < 1e-5 and abs(e_sh[1] - e_full[1]) < 1e-5, (e_sh, e_full)

        # token-mean stats + routing entropy from per-site mean weights
        n_local = len(rows) * T
        alphas_full = torch.softmax(torch.randn(3, B * T), dim=0)  # 3 sources, per-token weights
        mine = alphas_full.view(3, B, T)[:, rows].reshape(3, -1)
        sharded._mlp_mean_abs = {0: X_local.abs().reshape(-1, D).mean(0)}
        sharded._attn_res_diagnostics = {"attnres/block_norm/0": X_local.norm(dim=-1).mean().item()}
        for s_idx, a in enumerate(mine.mean(1).tolist()):
            sharded._attn_res_diagnostics[f"attnres/_alpha_mean/layer_1/pre_mlp/{s_idx}"] = a
        sharded._attn_res_diagnostics["attnres/routing_entropy/layer_1/pre_mlp"] = float("nan")  # local, replaced
        sharded._reduce_probe_stats(n_local, torch.device("cpu"))
        assert torch.allclose(sharded._mlp_mean_abs[0], X.abs().reshape(-1, D).mean(0), atol=1e-6)
        d = sharded._attn_res_diagnostics
        assert abs(d["attnres/block_norm/0"] - X.norm(dim=-1).mean().item()) < 1e-5
        am = alphas_full.mean(1).double()
        want = (-(am * (am + 1e-10).log()).sum() / math.log(3)).item()
        got = d["attnres/routing_entropy/layer_1/pre_mlp"]
        assert abs(got - want) < 1e-6, (got, want)
        assert not any(k.startswith("attnres/_alpha_mean/") for k in d)

        # EoC power iteration: sharded over sequences == on all of them (both converged)
        A_mix = torch.randn(D, D) * 0.5
        cpu = torch.device("cpu")
        sharded._eoc_n_global = B
        sharded._eoc_local_global = rows
        sig_sh, gain_sh, it_sh, _ = sharded._power_iterate(X_local, _layer_fn(A_mix), cpu, 400)
        sig_full, gain_full, it_full, _ = plain._power_iterate(X, _layer_fn(A_mix), cpu, 400)
        assert abs(sig_sh - sig_full) <= 1e-3 * sig_full, (sig_sh, sig_full)
        assert abs(gain_sh - gain_full) <= 1e-3 * gain_full, (gain_sh, gain_full)
        assert it_sh == it_full, (it_sh, it_full)

        # ... and the SAME iteration, not only the same limit: each rank starts from its rows of the unsharded seeded
        # start (drawn whole on every rank, no RNG traffic), so a 3-step, unconverged σ agrees to rounding
        assert torch.equal(sharded._eoc_start((B, T, D))[rows], plain._eoc_start((B, T, D))[rows])
        short_sh = sharded._power_iterate(X_local, _layer_fn(A_mix), cpu, 3)
        short_full = plain._power_iterate(X, _layer_fn(A_mix), cpu, 3)
        assert short_sh.iters == short_full.iters == 3
        assert abs(short_sh.sigma - short_full.sigma) <= 1e-5 * short_full.sigma, (short_sh, short_full)
        assert abs(short_sh.sigma - sig_full) > 1e-4 * sig_full  # 3 steps is short of the limit: the start matters

        # a rank holding none of the EoC rows still takes part
        empty = X_local[:0] if rank == 1 else X_local
        sharded._eoc_local_global = [] if rank == 1 else rows
        sig_e = sharded._power_iterate(empty, _layer_fn(A_mix), cpu, 400).sigma
        sig_r0 = plain._power_iterate(X[[0, 2]], _layer_fn(A_mix), cpu, 400).sigma  # rank 0's rows
        assert abs(sig_e - sig_r0) <= 1e-3 * sig_r0, (sig_e, sig_r0)

        # sample_rows draws exactly the unsharded sample (rank 0's randperm over all rows), so order-invariant
        # consumers agree to rounding
        torch.manual_seed(123)
        a_sh = geometric._anisotropy(shard.sample_rows(X_local, B, 10), max_samples=10)
        torch.manual_seed(123)
        a_full = geometric._anisotropy(X.reshape(-1, D), max_samples=10)
        assert abs(a_sh - a_full) < 1e-5, (a_sh, a_full)

        out[rank] = "ok"
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_sharded_probe_reductions_match_unsharded():
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "init")
        manager = mp.Manager()
        out = manager.dict()
        mp.spawn(_worker, args=(init_file, out), nprocs=WORLD, join=True)
        assert dict(out) == {0: "ok", 1: "ok"}


def test_unsharded_by_default():
    m = GeometricMonitor(_Tiny(), MonitorConfig())
    assert m._shard.on is False and m._shard.world == 1
