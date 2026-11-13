"""FSDP2 + sharded Muon against the unsharded model, on CPU (gloo).

The model is a small looped AttnRes stack whose every matrix shards unevenly over 3 ranks. Checked:

- sharded Newton-Schulz returns the replicated result bitwise (uneven and empty shards included);
- the Muon / AdamW / no-decay split is name-for-name the unsharded one;
- every gradient equals the unsharded model's, and a loop-shared weight's gradient is the sum over its passes;
- the total grad norm and the telemetry readers equal the unsharded values;
- over several Muon steps the sharded run stays within the unsharded run's own reduction-order noise
  (the same model with its rank losses summed in a different order);
- HSDP replicas stay bitwise identical;
- a sharded checkpoint resumes bitwise, also at a different world size;
- the rank-data guard raises when two ranks hold the same batch.
"""

from __future__ import annotations

import contextlib
import copy
import io
import math
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
from kotodama.training.fsdp import apply_fsdp2, full_model_state_dict, params_to_fp32, shard_rows
from kotodama.training.muon import (
    HybridScheduler,
    _orthogonalize_by_shape,
    assign_owners,
    build_hybrid_optimizer,
    orthogonalize_sharded,
)

LR = dict(muon_lr=0.02, adamw_lr=1e-3, muon_fp32_master=True, adamw_no_decay_norms=True,
          muon_ns_coefficients="gram_ns")
BATCH, SEQ = 2, 24


def _cfg() -> KotodamaModelConfig:
    # q 128 / kv 64 / ffn 320 / vocab 256 rows: none divides by 3.
    return KotodamaModelConfig(hidden_size=128, num_layers=6, num_attention_heads=2, num_kv_heads=1, head_dim=64,
                               intermediate_size=320, vocab_size=256, max_position_embeddings=64, attn_res=True,
                               attn_res_boundaries=[0, 1, 3, 5], loop_start=1, loop_len=2, loop_times=2,
                               attn_impl="sdpa")


def _reference() -> KotodamaModel:
    torch.manual_seed(0)
    model = KotodamaModel(_cfg())
    with torch.no_grad():  # non-trivial routing (fresh queries are zero)
        for name, p in model.named_parameters():
            if name.endswith("res_query") and p.requires_grad:
                p.normal_(0.0, 0.05)
    return model


def _sharded(ref: KotodamaModel, mesh) -> KotodamaModel:
    model = copy.deepcopy(ref)
    params_to_fp32(model)
    apply_fsdp2(model, mesh, param_dtype=None)
    return model


def _optimizers(model):
    with contextlib.redirect_stdout(io.StringIO()):
        return build_hybrid_optimizer(model, **LR)


def _batch(step: int, rank: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 * step + rank)
    return torch.randint(0, _cfg().vocab_size, (BATCH, SEQ), generator=g)


def _mean_loss(model, step: int, ranks) -> torch.Tensor:
    ranks = list(ranks)
    return sum(model(_batch(step, r), labels=_batch(step, r))["loss"] for r in ranks) / len(ranks)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def _full(t):
    return t.full_tensor() if hasattr(t, "full_tensor") else t


def _init(rank: int, world: int, init_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)


def _spawn(fn, world: int, tmp_path: Path, *args) -> None:
    mp.spawn(fn, args=(world, str(tmp_path / "pg-init"), *args), nprocs=world, join=True)


# ── pure ─────────────────────────────────────────────────────────────────────


def test_shard_rows_is_dtensor_chunking():
    for world in range(1, 9):
        for n in range(0, 40):
            chunks = [len(c) for c in torch.arange(n).chunk(world)] if n else []
            assert shard_rows(n, world) == chunks + [0] * (world - len(chunks)), (n, world)


def test_owner_plan_is_deterministic_and_balanced():
    shapes = [(128, 128), (64, 128), (64, 128), (320, 128), (128, 320)] * 4
    a, b = assign_owners(shapes, 3), assign_owners(list(shapes), 3)
    assert a == b and set(a) == {0, 1, 2}
    cost = [0, 0, 0]
    for (r, c), o in zip(shapes, a):
        cost[o] += r * c * min(r, c)
    assert max(cost) / min(cost) < 1.5


# ── sharded Newton-Schulz ────────────────────────────────────────────────────

NS_SHAPES = [(128, 128), (64, 128), (64, 128), (320, 128), (128, 320), (4, 7), (5, 3), (1, 9), (320, 128)]


def _ns_worker(rank: int, world: int, init_file: str) -> None:
    _init(rank, world, init_file)
    g = torch.Generator().manual_seed(7)
    fulls = [torch.randn(*s, generator=g) * 1e-4 for s in NS_SHAPES]
    locals_ = []
    for M, (n, _c) in zip(fulls, NS_SHAPES):
        rows = shard_rows(n, world)
        start = sum(rows[:rank])
        locals_.append(M[start:start + rows[rank]].clone())
    got = orthogonalize_sharded(locals_, NS_SHAPES, None, world, rank, 5, "gram_ns")
    want = _orthogonalize_by_shape(dict(enumerate(fulls)), 5, "gram_ns")
    for i, (n, _c) in enumerate(NS_SHAPES):
        rows = shard_rows(n, world)
        start = sum(rows[:rank])
        assert torch.equal(got[i], want[i][start:start + rows[rank]]), (world, rank, NS_SHAPES[i])
    dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 3, 4])
def test_sharded_newton_schulz_is_bitwise_replicated(world, tmp_path):
    _spawn(_ns_worker, world, tmp_path)


# ── the model, sharded vs unsharded ──────────────────────────────────────────


def _parity_worker(rank: int, world: int, init_file: str) -> None:
    from torch.distributed.device_mesh import init_device_mesh

    from kotodama.training.train import _layer_health_metrics, _norm_gain_metrics, _per_layer_grad_norms

    _init(rank, world, init_file)
    cfg = _cfg()
    ref = _reference()
    reordered = copy.deepcopy(ref)
    unrolled = copy.deepcopy(ref)
    for i in range(cfg.num_layers):  # an un-aliased copy: every loop pass gets its own modules
        src = cfg.layer_source(i)
        if src != i:
            unrolled.layers[i].attn = copy.deepcopy(unrolled.layers[src].attn)
            unrolled.layers[i].ffn = copy.deepcopy(unrolled.layers[src].ffn)
    model = _sharded(ref, init_device_mesh("cpu", (world,)))
    assert model.layers[3].attn is model.layers[1].attn

    opts = {"ref": _optimizers(ref), "reordered": _optimizers(reordered), "sharded": _optimizers(model)}

    # G3: the param-group split, by name
    def split(m, pair):
        names = {id(p): n for n, p in m.named_parameters()}
        return [[names[id(p)] for p in g["params"]] for o in pair for g in o.param_groups]
    assert split(model, opts["sharded"]) == split(ref, opts["ref"])
    for n in split(model, opts["sharded"])[0]:
        assert dict(model.named_parameters())[n].ndim == 2, n

    for step in range(6):
        _mean_loss(ref, step, range(world)).backward()
        _mean_loss(reordered, step, reversed(range(world))).backward()
        model(_batch(step, rank), labels=_batch(step, rank))["loss"].backward()

        if step == 0:
            # G4: every gradient, and the shared core weights as the sum over both passes
            _mean_loss(unrolled, step, range(world)).backward()
            ref_params = dict(ref.named_parameters())
            for name, p in model.named_parameters():
                want = ref_params[name].grad
                if want is None:
                    assert p.grad is None, name
                    continue
                assert _rel(_full(p.grad), want) < 1e-5, name
            unrolled_params = dict(unrolled.named_parameters())
            shared = [n for n in ref_params if n.startswith(f"layers.{cfg.loop_start}.") and (".attn." in n or ".ffn." in n)]
            assert shared
            for name in shared:
                passes = [name.replace(f"layers.{cfg.loop_start}.", f"layers.{cfg.loop_start + k * cfg.loop_len}.")
                          for k in range(cfg.loop_times)]
                summed = sum(unrolled_params[q].grad for q in passes)
                assert _rel(_full(dict(model.named_parameters())[name].grad), summed) < 1e-5, name
            # G7: telemetry readers (collective on the sharded model) equal the unsharded values
            for reader in (_norm_gain_metrics, _layer_health_metrics, _per_layer_grad_norms):
                got, want = reader(model), reader(ref)
                assert got.keys() == want.keys(), reader.__name__
                for k in want:
                    assert math.isclose(got[k], want[k], rel_tol=1e-5, abs_tol=1e-9), (reader.__name__, k)

        # G7: total grad norm (DTensor in, one number out)
        gn_ref = torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0).item()
        gn_re = torch.nn.utils.clip_grad_norm_(reordered.parameters(), 1.0).item()
        gn_sh = _full(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)).item()
        if step == 0:  # same weights: the same norm
            assert math.isclose(gn_sh, gn_ref, rel_tol=1e-5), (gn_sh, gn_ref)
        else:  # weights now differ by reduction-order noise (`noise`, measured on the parameters last step)
            assert abs(gn_sh - gn_ref) / gn_ref <= 10.0 * noise, (step, gn_sh, gn_ref, gn_re, noise)

        for pair in opts.values():
            for o in pair:
                o.step()
                o.zero_grad(set_to_none=True)

        # G1: sharded vs unsharded stays inside the unsharded run's own reduction-order noise
        sharded_div = max(_rel(_full(p.detach()), q.detach())
                          for p, q in zip(model.parameters(), ref.parameters()))
        noise = max(_rel(p.detach(), q.detach()) for p, q in zip(reordered.parameters(), ref.parameters()))
        assert sharded_div <= 3.0 * noise + 1e-6, (step, sharded_div, noise)

    full = full_model_state_dict(model)
    if rank == 0:
        assert set(full) == set(ref.state_dict())
    dist.destroy_process_group()


def test_fsdp2_matches_unsharded_model(tmp_path):
    _spawn(_parity_worker, 3, tmp_path)


# ── deferred gradient sync ───────────────────────────────────────────────────


def _defer_worker(rank: int, world: int, init_file: str) -> None:
    from torch.distributed.device_mesh import init_device_mesh

    _init(rank, world, init_file)
    mesh = init_device_mesh("cpu", (world,))
    ref = _reference()
    every, deferred = _sharded(ref, mesh), _sharded(ref, mesh)
    micros = 3
    for m, defer in ((every, False), (deferred, True)):
        for k in range(micros):
            if defer:
                m.set_requires_gradient_sync(k == micros - 1)
            ids = _batch(k, rank)
            (m(ids, labels=ids)["loss"] / micros).backward()
    for (name, p), q in zip(every.named_parameters(), deferred.parameters()):
        if p.grad is not None:
            assert _rel(_full(q.grad), _full(p.grad)) < 1e-5, name
    dist.destroy_process_group()


def test_deferred_grad_sync_matches_per_micro_sync(tmp_path):
    _spawn(_defer_worker, 3, tmp_path)


# ── compile inside the units ─────────────────────────────────────────────────


def _compile_worker(rank: int, world: int, init_file: str) -> None:
    from torch.distributed.device_mesh import init_device_mesh

    from kotodama.training.fsdp import compile_fsdp_units, fsdp_units

    _init(rank, world, init_file)
    mesh = init_device_mesh("cpu", (world,))
    ref = _reference()
    plain, compiled = _sharded(ref, mesh), _sharded(ref, mesh)
    assert compile_fsdp_units(fsdp_units(compiled), backend="eager") == 2  # GQAttention, SwiGLUFFN
    for m in (plain, compiled):
        m(_batch(0, rank), labels=_batch(0, rank))["loss"].backward()
    for (name, p), q in zip(plain.named_parameters(), compiled.parameters()):
        if p.grad is not None:
            assert torch.equal(_full(p.grad), _full(q.grad)), name
    dist.destroy_process_group()


def test_compiled_units_match_eager_units(tmp_path):
    _spawn(_compile_worker, 2, tmp_path)


# ── in-loop evals: uneven forward counts in lockstep ─────────────────────────


def _lockstep_worker(rank: int, world: int, init_file: str) -> None:
    from torch.distributed.device_mesh import init_device_mesh

    from kotodama.training.inloop import Lockstep

    _init(rank, world, init_file)
    ref = _reference().eval()
    model = _sharded(ref, init_device_mesh("cpu", (world,))).eval()
    step = Lockstep(lambda ids: model(ids)["logits"], torch.device("cpu"))
    with torch.no_grad():
        for k in range(rank + 1):  # rank r makes r + 1 real forwards
            ids = _batch(k, rank)
            assert torch.allclose(step(ids), ref(ids)["logits"], atol=1e-5, rtol=1e-4)
        step.drain()
    assert step.real_calls == rank + 1 and step.real_calls + step.pad_calls == world
    dist.destroy_process_group()


def test_lockstep_pads_uneven_forward_counts(tmp_path):
    _spawn(_lockstep_worker, 3, tmp_path)


# ── HSDP ─────────────────────────────────────────────────────────────────────


def _hsdp_worker(rank: int, world: int, init_file: str) -> None:
    from torch.distributed.device_mesh import init_device_mesh

    from kotodama.training.fsdp import replicate_group
    from kotodama.training.train import param_checksum_spread

    _init(rank, world, init_file)
    mesh = init_device_mesh("cpu", (2, world // 2), mesh_dim_names=("replicate", "shard"))
    ref = _reference()
    model = _sharded(ref, mesh)
    muon, adamw = _optimizers(model)
    _mean_loss(ref, 0, range(world)).backward()
    for step in range(3):
        model(_batch(step, rank), labels=_batch(step, rank))["loss"].backward()
        if step == 0:
            ref_params = dict(ref.named_parameters())
            for name, p in model.named_parameters():
                if p.grad is not None:
                    assert _rel(_full(p.grad), ref_params[name].grad) < 1e-5, name
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in (muon, adamw):
            o.step()
            o.zero_grad(set_to_none=True)
    group = replicate_group(next(model.parameters()))
    peers = dist.get_world_size(group)
    tensors = [p.to_local() for p in model.parameters()] + [s["momentum_buffer"].to_local() for s in muon.state.values()]
    for t in tensors:
        gathered = [torch.empty_like(t) for _ in range(peers)]
        dist.all_gather(gathered, t.contiguous(), group=group)
        assert all(torch.equal(g, gathered[0]) for g in gathered)
    assert param_checksum_spread(model) == 0.0
    dist.destroy_process_group()


def test_hsdp_replicas_stay_identical(tmp_path):
    _spawn(_hsdp_worker, 4, tmp_path)


# ── sharded checkpoints ──────────────────────────────────────────────────────


def _train_steps(model, muon, adamw, sched, rank: int, start: int, n: int) -> None:
    for step in range(start, start + n):
        sched.step(step)
        model(_batch(step, rank), labels=_batch(step, rank))["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in (muon, adamw):
            o.step()
            o.zero_grad(set_to_none=True)


def _ckpt_worker(rank: int, world: int, init_file: str, ckpt_dir: str, out_dir: str, phase: str) -> None:
    from torch.distributed.device_mesh import init_device_mesh

    from kotodama.training.checkpoint import ShardedCheckpointManager

    _init(rank, world, init_file)
    model = _sharded(_reference(), init_device_mesh("cpu", (world,)))
    muon, adamw = _optimizers(model)
    sched = HybridScheduler(muon, adamw, warmup_steps=2, total_steps=20, decay_start_pct=0.5)
    mgr = ShardedCheckpointManager(ckpt_dir, rank=rank, keep_last_n=2)
    if phase == "save":
        _train_steps(model, muon, adamw, sched, rank, 0, 3)
        mgr.save(step=2, model=model, muon_opt=muon, adamw_opt=adamw, scheduler=sched, tokens_consumed=123,
                 data_state={"pos": 7}, extra={"probe_batch": torch.arange(4)})
        saved = full_model_state_dict(model)
        _train_steps(model, muon, adamw, sched, rank, 3, 3)
        final = full_model_state_dict(model)
        if rank == 0:
            torch.save({"saved": saved, "final": final}, Path(out_dir) / "uninterrupted.pt")
    else:
        meta = mgr.load_latest(model, muon, adamw, torch.device("cpu"))
        assert meta["step"] == 2 and meta["tokens_consumed"] == 123 and meta["data"] == {"pos": 7}
        assert torch.equal(meta["probe_batch"], torch.arange(4))
        loaded = full_model_state_dict(model)
        _train_steps(model, muon, adamw, sched, rank, 3, 3)
        final = full_model_state_dict(model)
        if rank == 0:
            want = torch.load(Path(out_dir) / "uninterrupted.pt")
            for k, v in want["saved"].items():
                assert torch.equal(loaded[k], v), ("load", k)
            if phase == "resume-same-world":  # resume reproduces the uninterrupted run exactly
                for k, v in want["final"].items():
                    assert torch.equal(final[k], v), ("resume", k)
    dist.destroy_process_group()


def test_sharded_checkpoint_resumes_bitwise(tmp_path):
    ckpt, out = tmp_path / "ckpt", tmp_path / "out"
    out.mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "c").mkdir()
    _spawn(_ckpt_worker, 3, tmp_path / "a", str(ckpt), str(out), "save")
    assert (ckpt / "step_00000002.dcp" / "meta.pt").is_file()
    assert not list(ckpt.glob("*.tmp"))
    _spawn(_ckpt_worker, 3, tmp_path / "b", str(ckpt), str(out), "resume-same-world")
    _spawn(_ckpt_worker, 2, tmp_path / "c", str(ckpt), str(out), "resume-other-world")


def test_sharded_manager_refuses_unsharded_checkpoints(tmp_path):
    from kotodama.training.checkpoint import ShardedCheckpointManager

    (tmp_path / "step_00000010.pt").write_bytes(b"")
    mgr = ShardedCheckpointManager(tmp_path, rank=0)
    with pytest.raises(RuntimeError, match="unsharded"):
        mgr.load_latest(torch.nn.Linear(1, 1), None, None, torch.device("cpu"))  # type: ignore[arg-type]


# ── the rank-data guard ──────────────────────────────────────────────────────


def _guard_worker(rank: int, world: int, init_file: str, duplicate: bool) -> None:
    from kotodama.training.train import check_rank_batches_distinct

    _init(rank, world, init_file)
    ids = _batch(0, 0 if (duplicate and rank in (0, 2)) else rank)
    if duplicate:
        with pytest.raises(RuntimeError, match=r"\[\[0, 2\]\]"):
            check_rank_batches_distinct(ids, None, torch.device("cpu"), step=5, micro=1)
    else:
        digests = check_rank_batches_distinct(ids, None, torch.device("cpu"), step=5, micro=1)
        assert len(set(digests)) == world
    dist.destroy_process_group()


@pytest.mark.parametrize("duplicate", [False, True])
def test_rank_data_guard(duplicate, tmp_path):
    _spawn(_guard_worker, 3, tmp_path, duplicate)


def test_fsdp2_flag_refuses_unsupported_combinations():
    import argparse

    from kotodama.training.train import validate_fsdp2_args

    ok = argparse.Namespace(fsdp2=True, fsdp2_replicas=2, tp_size=1, fp8=False, geo_monitor=False, inloop_evals="")
    validate_fsdp2_args(ok)
    validate_fsdp2_args(argparse.Namespace(**{**vars(ok), "inloop_evals": "riders,induction"}))
    for bad in (dict(tp_size=2), dict(fp8=True), dict(geo_monitor=True)):
        with pytest.raises(ValueError, match="does not support"):
            validate_fsdp2_args(argparse.Namespace(**{**vars(ok), **bad}))
    with pytest.raises(ValueError, match="needs --fsdp2"):
        validate_fsdp2_args(argparse.Namespace(fsdp2=False, fsdp2_replicas=2))
