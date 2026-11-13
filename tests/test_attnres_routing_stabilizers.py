"""Opt-in AttnRes routing stabilizers: unit-norm routing queries with a learned temperature (--attnres_query_norm)
and a separate AdamW LR for the routing queries (--attnres_query_lr_scale).

A unit-norm model starts with every temperature at 0, so its first forward equals a zero-query model's with the same
seed (routing uniform, every other weight drawn identically). The LR scale must reach exactly the routing queries."""

from __future__ import annotations

import pytest
import torch

from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
from kotodama.training import train
from kotodama.training.muon import HybridScheduler, build_hybrid_optimizer


def _cfg(query_norm: bool = False, gain: str = "folded") -> KotodamaModelConfig:
    return KotodamaModelConfig(
        hidden_size=64, num_layers=6, num_attention_heads=2, num_kv_heads=1, head_dim=32, intermediate_size=128,
        vocab_size=97, max_position_embeddings=32, attn_res=True, attn_res_boundaries=[0, 1, 3, 5],
        loop_start=1, loop_len=2, loop_times=2, attn_impl="sdpa", attn_res_gain=gain, attn_res_query_norm=query_norm,
    )


def _model(query_norm: bool, seed: int = 0) -> KotodamaModel:
    torch.manual_seed(seed)
    return KotodamaModel(_cfg(query_norm))


def _ids() -> torch.Tensor:
    return torch.randint(0, 97, (2, 32), generator=torch.Generator().manual_seed(1))


def _temperatures(m: KotodamaModel) -> list[torch.nn.Parameter]:
    temps = m.routing_temperatures()
    assert len(temps) == 2 * m.config.num_layers + 1
    return temps


def test_unit_norm_init_matches_zero_query_model():
    plain, unit = _model(False), _model(True)
    shared = {k: v for k, v in unit.state_dict().items() if "res_query" not in k}
    ref = {k: v for k, v in plain.state_dict().items() if "res_query" not in k}
    assert shared.keys() == ref.keys()
    assert all(torch.equal(shared[k], ref[k]) for k in ref)  # the global RNG stream is untouched
    with torch.no_grad():
        assert torch.equal(plain.eval()(_ids())["logits"], unit.eval()(_ids())["logits"])


def test_temperatures_are_one_element_vectors():
    assert all(t.shape == (1,) for t in _temperatures(_model(True)))  # FSDP2's fully_shard refuses 0-dim params


def test_effective_query_is_temperature_times_unit_direction():
    m = _model(True)
    with torch.no_grad():
        m.layers[2].parametrizations["mlp_res_query"][0].temperature.fill_(2.5)
    q = m.layers[2].mlp_res_query
    assert torch.allclose(q.norm(), torch.tensor(2.5))
    d = m.layers[2].parametrizations["mlp_res_query"].original
    assert torch.allclose(q, 2.5 * d / d.norm())


def test_gradients_reach_temperature_then_direction():
    m = _model(True)
    m(_ids(), labels=_ids())["loss"].backward()
    temps = _temperatures(m)
    first = m.layers[0].parametrizations["attn_res_query"]
    assert first.original.requires_grad is False and first[0].temperature.requires_grad is False
    live = [t for t in temps if t.requires_grad]
    assert len(live) == len(temps) - 1
    assert all(t.grad is not None and t.grad.abs() > 0 for t in live)
    # At temperature 0 the direction's gradient is exactly zero; it flows once the temperature moves.
    d = m.layers[2].parametrizations["attn_res_query"].original
    assert d.grad is None or bool((d.grad == 0).all())
    m.zero_grad(set_to_none=True)
    with torch.no_grad():
        for t in live:
            t.fill_(1.0)
    m(_ids(), labels=_ids())["loss"].backward()
    assert d.grad is not None and d.grad.abs().sum() > 0


def test_state_dict_round_trip():
    a = _model(True, seed=3)
    with torch.no_grad():
        for i, t in enumerate(_temperatures(a)):
            t.fill_(0.1 * (i + 1))
    b = _model(True, seed=4)
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        assert torch.equal(a.eval()(_ids())["logits"], b.eval()(_ids())["logits"])


def test_learned_gain_rejected():
    with pytest.raises(ValueError, match="attn_res_query_norm"):
        _cfg(True, gain="learned")


def test_reset_and_fold():
    m = _model(True)
    with torch.no_grad():
        for t in _temperatures(m):
            t.fill_(1.5)
    assert m.fold_attn_res_gains() == 0  # gains are 1: nothing to fold, nothing changes
    m.reset_routing(range(m.config.num_layers), final=True)
    assert all(float(t.detach()) == 0.0 for t in _temperatures(m))
    with torch.no_grad():
        m.layers[3].attn_res_norm.weight.fill_(2.0)
    with pytest.raises(ValueError, match="unit-norm"):
        m.fold_attn_res_gains()


def _group_of(opt: torch.optim.Optimizer, p: torch.Tensor) -> dict:
    return next(g for g in opt.param_groups if any(q is p for q in g["params"]))


@pytest.mark.parametrize("query_norm", [False, True])
def test_query_lr_scale_groups_and_schedule(query_norm):
    m = _model(query_norm)
    _muon, adamw = build_hybrid_optimizer(m, adamw_lr=1e-3, res_query_lr_scale=0.25)
    sched = HybridScheduler(_muon, adamw, warmup_steps=10, total_steps=100)
    sched.step(50)
    main = adamw.param_groups[0]
    assert main["lr"] == pytest.approx(1e-3)
    q = (m.layers[2].parametrizations["attn_res_query"].original if query_norm else m.layers[2].attn_res_query)
    g = _group_of(adamw, q)
    assert g["lr"] == pytest.approx(0.25e-3) and g["weight_decay"] == main["weight_decay"]
    assert not any(p is m.layers[2].attn_res_norm.weight for p in g["params"])  # gains are not queries
    n_queries = sum(p.numel() > 1 and p.requires_grad for p in g["params"])
    assert n_queries == 2 * m.config.num_layers  # every site but the frozen first-boundary pre-attn one
    if query_norm:
        tg = _group_of(adamw, m.layers[2].parametrizations["attn_res_query"][0].temperature)
        assert tg["weight_decay"] == 0.0 and tg["lr"] == pytest.approx(0.25e-3)


def test_default_scale_keeps_one_group_for_plain_queries():
    _muon, adamw = build_hybrid_optimizer(_model(False))
    assert len(adamw.param_groups) == 1


def test_unit_norm_temperatures_never_decay_at_scale_one():
    m = _model(True)
    _muon, adamw = build_hybrid_optimizer(m, adamw_lr=1e-3)
    tg = _group_of(adamw, m.layers[2].parametrizations["attn_res_query"][0].temperature)
    assert tg["weight_decay"] == 0.0 and tg.get("lr_scale", 1.0) == 1.0
    q = m.layers[2].parametrizations["attn_res_query"].original
    assert _group_of(adamw, q) is adamw.param_groups[0]


def test_bad_scale_rejected():
    with pytest.raises(ValueError):
        build_hybrid_optimizer(_model(False), res_query_lr_scale=0.0)


@pytest.mark.parametrize("query_norm", [False, True])
def test_router_telemetry_sites_and_no_false_dead(query_norm):
    from kotodama.training import train as T

    m = _model(query_norm)
    m(_ids(), labels=_ids())["loss"].backward()
    h = T._layer_health_metrics(m)
    n_sites = 2 * m.config.num_layers + 1
    assert sum(k.endswith("/query_norm") for k in h if k.startswith("router/")) == n_sites
    assert sum(k.endswith("/grad_norm") for k in h if k.startswith("router/")) == n_sites - 1
    assert "router/final/query_norm" in h and "router/layer_2/mlp/query_norm" in h
    if query_norm:
        assert h["router/layer_2/mlp/query_norm"] == 0.0  # |temperature|, which starts at 0
        assert h["router/dead_sites"] == 0.0  # the temperatures carry gradient from step 0


def _args(*extra: str):
    import sys

    old = sys.argv
    try:
        sys.argv = ["train.py", "--model_size", "proxy", *extra]
        return train.parse_args()
    finally:
        sys.argv = old


def test_flags_parse_and_default_off():
    a = _args()
    assert a.attnres_query_norm is False and a.attnres_query_lr_scale == 1.0
    a = _args("--attnres_query_norm", "--attnres_res_gain", "folded", "--attnres_query_lr_scale", "0.25")
    assert a.attnres_query_norm is True and a.attnres_query_lr_scale == 0.25
