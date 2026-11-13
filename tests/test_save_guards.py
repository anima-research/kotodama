"""Trainer safety rails: a non-finite loss stops every rank before the optimizer step, and no save path writes
non-finite weights (so rotation can never evict the last good checkpoint for a bad one)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kotodama.training.checkpoint import CheckpointManager, require_finite_params, save_weights_only
from kotodama.training.muon import HybridScheduler, build_hybrid_optimizer


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4))


def test_finite_params_pass_and_nan_params_refuse():
    m = _model()
    require_finite_params(m, "probe")
    with torch.no_grad():
        m[1].weight[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="1 non-finite"):
        require_finite_params(m, "probe")


def test_nan_save_keeps_the_last_good_checkpoint(tmp_path):
    m = _model()
    muon, adamw = build_hybrid_optimizer(m)
    sched = HybridScheduler(muon, adamw, warmup_steps=1, total_steps=10)
    mgr = CheckpointManager(tmp_path, rank=0, keep_last_n=1)
    good = mgr.save(step=1, model=m, muon_opt=muon, adamw_opt=adamw, scheduler=sched)
    assert good is not None and good.exists()
    with torch.no_grad():
        m[0].weight.fill_(float("inf"))
    with pytest.raises(RuntimeError, match="refusing to save"):
        mgr.save(step=2, model=m, muon_opt=muon, adamw_opt=adamw, scheduler=sched)
    with pytest.raises(RuntimeError, match="refusing to save"):
        save_weights_only(tmp_path / "weights", 2, m, tokens_consumed=0)
    assert sorted(p.name for p in tmp_path.glob("step_*.pt")) == [good.name]  # rotation did not run
    assert not (tmp_path / "weights").exists() or not list((tmp_path / "weights").iterdir())


def _loss_worker(rank: int, world: int, init_file: str) -> None:
    from kotodama.training.train import check_finite_loss

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    check_finite_loss(2.5, step=3, device=torch.device("cpu"), checkpoint_dir="ckpt")  # all finite: passes
    loss = math.nan if rank == 1 else 2.5  # only rank 1 sees it; every rank must stop
    with pytest.raises(RuntimeError, match="non-finite loss at step 4"):
        check_finite_loss(loss, step=4, device=torch.device("cpu"), checkpoint_dir="ckpt")
    dist.destroy_process_group()


def test_non_finite_loss_on_one_rank_stops_every_rank(tmp_path):
    mp.spawn(_loss_worker, args=(3, str(tmp_path / "pg-init")), nprocs=3, join=True)
