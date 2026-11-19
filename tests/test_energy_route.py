"""Energy routing (kotodama.model.energy_route) on a tiny looped AttnRes model.

Gates: at sigma = 0 an energy-routed model is today's model (same init, same logits); the iteration's fixed point is a
stationary point of the energy; the closed-form gradient is autograd's; gradients reach sigma at once and U after
sigma moves; U trains with AdamW, not Muon; the first boundary site's unused sigma is frozen; the eval loader infers
the energy fields from a checkpoint and refuses a mismatch; the capture walker refuses an energy-routed model; the
eval routing weights carry the energy and log the TV distance from the one-softmax weights.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import pytest
import torch

from kotodama.model import energy_route as er
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig

BASE = dict(hidden_size=64, num_layers=7, num_attention_heads=4, num_kv_heads=2, head_dim=16,
            intermediate_size=128, vocab_size=199, max_position_embeddings=96, attn_impl="sdpa",
            attn_res=True, attn_res_boundaries=[0, 2, 5], use_liger=False,
            loop_start=1, loop_len=2, loop_times=2)
ENERGY = dict(attn_res_energy_steps=2, attn_res_energy_rank=8)


def _model(seed: int = 20261118, **kw) -> KotodamaModel:
    torch.manual_seed(seed)
    return KotodamaModel(KotodamaModelConfig(**BASE, **kw))


def _queries_nonzero(m: KotodamaModel, seed: int = 7) -> None:
    """Routing queries start at 0 (uniform routing); give them values so routing is non-trivial."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.endswith("res_query"):
                p.copy_(torch.randn(p.shape, generator=g))


@pytest.fixture(scope="module")
def ids() -> torch.Tensor:
    torch.manual_seed(3)
    return torch.randint(3, 199, (2, 24), dtype=torch.long)


def test_sigma_zero_is_todays_model(ids):
    base, en = _model(), _model(**ENERGY)
    sb, se = base.state_dict(), en.state_dict()
    for k, v in sb.items():                       # same init for every shared weight (U has its own generator)
        assert torch.equal(v, se[k]), k
    _queries_nonzero(base); _queries_nonzero(en)
    for mode in ("train", "eval"):
        getattr(base, mode)(); getattr(en, mode)()
        with torch.no_grad():
            lb, le = base(ids)["logits"], en(ids)["logits"]
        torch.testing.assert_close(le, lb, atol=1e-5, rtol=1e-5)


def test_energy_changes_routing_when_sigma_moves(ids):
    a, b = _model(**ENERGY), _model(**ENERGY)
    _queries_nonzero(a); _queries_nonzero(b)
    with torch.no_grad():
        for name, p in b.named_parameters():
            if name.endswith("res_energy_sigma") and p.requires_grad:
                p.fill_(2.0)
    a.eval(); b.eval()
    with torch.no_grad():
        assert not torch.allclose(a(ids)["logits"], b(ids)["logits"], atol=1e-4)


def test_fixed_point_is_stationary():
    g = torch.Generator().manual_seed(0)
    S, N, r = 5, 11, 4
    logits = torch.randn(S, N, generator=g, dtype=torch.float64)
    logits[4, :3] = float("-inf")                 # an inactive source on some tokens
    P = torch.randn(S, N, r, generator=g, dtype=torch.float64)
    sigma = 0.2 * torch.randn(r, generator=g, dtype=torch.float64)
    a, _ = er.energy_weights(logits, P, sigma, steps=300)
    grad = er.energy_grad(a, logits, P, sigma)
    active = torch.isfinite(logits)
    # Stationary on the simplex: the gradient is constant across the active sources of each token
    for t in range(N):
        gt = grad[active[:, t], t]
        assert (gt - gt.mean()).abs().max() < 1e-9
    assert torch.all(a[~active] == 0)


def test_closed_form_gradient_matches_autograd():
    g = torch.Generator().manual_seed(1)
    S, N, r = 4, 6, 3
    logits = torch.randn(S, N, generator=g, dtype=torch.float64)
    P = torch.randn(S, N, r, generator=g, dtype=torch.float64)
    sigma = torch.randn(r, generator=g, dtype=torch.float64)
    a = torch.softmax(torch.randn(S, N, generator=g, dtype=torch.float64), 0).requires_grad_(True)
    er.energy(a, logits, P, sigma).sum().backward()
    torch.testing.assert_close(a.grad, er.energy_grad(a.detach(), logits, P, sigma))


def test_steps_zero_is_softmax():
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(3, 5, generator=g)
    P = torch.randn(3, 5, 4, generator=g)
    a, a0 = er.energy_weights(logits, P, torch.ones(4), steps=0)
    torch.testing.assert_close(a, torch.softmax(logits, 0))
    torch.testing.assert_close(a0, a)


def test_split_router_matches_stacked():
    g = torch.Generator().manual_seed(4)
    committed, partial = torch.randn(3, 2, 5, 16, generator=g), torch.randn(2, 5, 16, generator=g)
    qw, nw, U = torch.randn(16, generator=g), 1 + 0.1 * torch.randn(16, generator=g), torch.randn(16, 4, generator=g)
    sigma = torch.randn(4, generator=g)
    h_split = er.route_energy_split(committed, partial, qw, nw, 1e-6, U, sigma, 2)
    h_stack = er.route_energy(torch.cat([committed, partial.unsqueeze(0)]), qw, nw, 1e-6, U, sigma, 2)[0]
    torch.testing.assert_close(h_split, h_stack, atol=1e-5, rtol=1e-5)


def test_gradients_reach_sigma_then_u(ids):
    m = _model(**ENERGY); _queries_nonzero(m); m.train()
    m(ids, labels=ids)["loss"].backward()
    sig_grads = [p.grad for n, p in m.named_parameters() if n.endswith("res_energy_sigma") and p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in sig_grads)
    # At sigma = 0, U has no effect, so no gradient; once sigma moves, it does
    assert m.attn_res_energy_U.grad is None or m.attn_res_energy_U.grad.abs().sum() == 0
    m.zero_grad(set_to_none=True)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("res_energy_sigma") and p.requires_grad:
                p.fill_(0.5)
    m(ids, labels=ids)["loss"].backward()
    assert m.attn_res_energy_U.grad is not None and m.attn_res_energy_U.grad.abs().sum() > 0


def test_first_boundary_sigma_frozen():
    m = _model(**ENERGY)
    assert not m.layers[0].attn_res_energy_sigma.requires_grad
    assert m.layers[0].mlp_res_energy_sigma.requires_grad and m.final_res_energy_sigma.requires_grad


def test_u_trains_with_adamw_not_muon():
    from kotodama.training.muon import build_hybrid_optimizer
    m = _model(**ENERGY)
    muon_opt, adamw_opt = build_hybrid_optimizer(m, muon_lr=0.02, adamw_lr=1e-3)
    muon_ids = {id(p) for g in muon_opt.param_groups for p in g["params"]}
    adamw_ids = {id(p) for g in adamw_opt.param_groups for p in g["params"]}
    assert id(m.attn_res_energy_U) in adamw_ids and id(m.attn_res_energy_U) not in muon_ids
    for n, p in m.named_parameters():
        if "res_energy_sigma" in n and p.requires_grad:
            assert id(p) in adamw_ids, n


def test_loader_infers_and_refuses(tmp_path):
    from kotodama.eval.model_loader import attn_res_geometry, load_model
    m = _model(**ENERGY)
    state = m.state_dict()
    geo = attn_res_geometry({"attn_res": True, "attn_res_boundaries": [0, 2, 5]}, state, {}, None)
    assert geo["attn_res_energy_steps"] == 2 and geo["attn_res_energy_rank"] == 8
    # A checkpoint without the energy keys loaded into an energy model: refused, not silently dropped
    plain = {k: v for k, v in state.items() if "res_energy" not in k}
    path = tmp_path / "plain.pt"
    torch.save({"model": plain}, path)
    with pytest.raises(ValueError, match="energy-routing keys"):
        load_model(path, config_section="proxy", device="cpu",
                   attn_res_config={"attn_res": True, "attn_res_boundaries": [0, 2, 5],
                                    "attn_res_energy_steps": 2, "attn_res_energy_rank": 8},
                   overrides={k: v for k, v in BASE.items() if k not in ("attn_res", "attn_res_boundaries")})


def test_capture_refuses(ids):
    from kotodama.model.capture import instrumented_forward
    m = _model(**ENERGY); m.eval()
    with pytest.raises(NotImplementedError, match="energy"):
        instrumented_forward(m, ids)


def test_eval_alphas_log_tv(ids):
    m = _model(**ENERGY); _queries_nonzero(m)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("res_energy_sigma") and p.requires_grad:
                p.fill_(3.0)
    m.eval(); m._energy_tv = []
    with torch.no_grad():
        m(ids)
    assert len(m._energy_tv) > 0
    assert all(tv.shape == ids.shape for _site, tv in m._energy_tv)
    assert max(float(tv.mean()) for _site, tv in m._energy_tv) > 0


def test_energy_telemetry(ids):
    from kotodama.training.train import EnergyTelemetry
    m = _model(**ENERGY); _queries_nonzero(m)
    tel = EnergyTelemetry(m)
    out0 = tel(m, 1)
    assert out0["energy/sigma_abs_mean"] == 0 and out0["energy/sigma_update_ratio"] == 0 and out0["energy/u_drift"] == 0
    assert "energy/sigma_abs/final" in out0 and "energy/sigma_abs/layer_0/mlp" in out0
    assert "energy/sigma_abs/layer_0/attn" not in out0          # frozen (never used), not logged
    with torch.no_grad():
        m.layers[1].attn_res_energy_sigma.fill_(0.5)
        m.attn_res_energy_U.add_(0.1)
    out1 = tel(m, 2)
    assert out1["energy/sigma_abs/layer_1/attn"] == pytest.approx(0.5)
    assert out1["energy/sigma_step_abs_mean"] > 0 and out1["energy/sigma_update_ratio"] > 0 and out1["energy/u_drift"] > 0


class _StubTok:
    """Just what the induction batch needs: special ids and no eos (the tiny model's vocab is 199)."""
    all_special_ids = [0]
    eos_token_id = None

    def __len__(self) -> int:
        return 199


def test_inloop_energy_suite(tmp_path):
    from kotodama.training.inloop import InloopConfig, InloopEvaluator
    m = _model(**ENERGY); _queries_nonzero(m)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("res_energy_sigma") and p.requires_grad:
                p.fill_(2.0)
    m.train()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    cfg = InloopConfig(suites=["energy"], max_batch_tokens=64, induction_n_seqs=4, induction_len=16)
    ev = InloopEvaluator(cfg, out_dir=tmp_path, rank=0, world_size=1, device=torch.device("cpu"),
                         max_seq_len=96, vocab_size=199, tokenizer=_StubTok())
    out = ev.run(0, m)
    assert out is not None and out["inloop/errors"] == 0
    for k in ("induction_gain_normalized", "loss_first", "loss_second"):
        assert out[f"inloop/energy/{k}_delta"] == pytest.approx(out[f"inloop/energy/{k}_on"] - out[f"inloop/energy/{k}_off"])
    assert out["inloop/energy/loss_second_delta"] != 0                    # sigma = 2 changes the routing
    assert out["inloop/energy/tv_first_mean"] > 0 and out["inloop/energy/tv_second_mean"] > 0
    assert any(k.startswith("inloop/energy/tv_second/layer_") for k in out)
    assert m.training and m.config.attn_res_energy_steps == 2 and getattr(m, "_energy_tv", None) is None
    for k, v in m.state_dict().items():                                    # eval only: no weight moved
        assert torch.equal(v, before[k]), k


def test_inloop_energy_suite_refuses_plain_model(tmp_path):
    from kotodama.training.inloop import InloopConfig, InloopEvaluator
    m = _model(); m.train()
    cfg = InloopConfig(suites=["energy"], max_batch_tokens=64, induction_n_seqs=4, induction_len=16)
    ev = InloopEvaluator(cfg, out_dir=tmp_path, rank=0, world_size=1, device=torch.device("cpu"),
                         max_seq_len=96, vocab_size=199, tokenizer=_StubTok())
    out = ev.run(0, m)
    assert out is not None and out["inloop/errors"] == 1                    # reported, not raised


def test_frozen_energy_sigmas_flags_a_dead_energy_path(ids):
    """The fail-fast guard: sigmas still exactly zero after optimizer steps mean no forward path routed with the
    energy. A real energy forward moves them on the first AdamW step; a forward that skips the energy leaves them."""
    from kotodama.training.train import frozen_energy_sigmas
    m = _model(**ENERGY); _queries_nonzero(m); m.train()
    n_sig = sum(1 for n, p in m.named_parameters() if n.endswith("res_energy_sigma") and p.requires_grad)
    assert n_sig > 0 and len(frozen_energy_sigmas(m)) == n_sig          # at init, every sigma is zero
    opt = torch.optim.AdamW([p for n, p in m.named_parameters() if "res_energy" in n], lr=1e-3, weight_decay=0.0)
    m(ids, labels=ids)["loss"].backward(); opt.step()
    assert frozen_energy_sigmas(m) == []                                 # the energy path trains every sigma
    with torch.no_grad():                                                # one site that skipped the energy
        name, p = next((n, p) for n, p in m.named_parameters() if n.endswith("mlp_res_energy_sigma") and p.requires_grad)
        p.zero_()
    assert frozen_energy_sigmas(m) == [name]
    dead = _model(**ENERGY); _queries_nonzero(dead); dead.train()
    dead.config.attn_res_energy_steps = 0                                # a forward that routes without the energy
    opt = torch.optim.AdamW([p for n, p in dead.named_parameters() if "res_energy" in n], lr=1e-3, weight_decay=0.0)
    dead(ids, labels=ids)["loss"].backward(); opt.step()
    assert len(frozen_energy_sigmas(dead)) == n_sig
