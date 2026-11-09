"""EoC ruler: a seeded, fixed-budget power iteration whose readout depends only on the model and the probe (not on the
probe's call history or the training RNG), its drift reported beside σ, scale-free relative gain, and the MLP sub-layer
probed at its own (pre-MLP) routed input."""
from __future__ import annotations

import math

import pytest
import torch

from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
from kotodama.monitoring.geometric import GeometricMonitor, MonitorConfig


def _tiny(attn_res: bool = True) -> KotodamaModel:
    torch.manual_seed(0)
    kw = dict(hidden_size=64, num_layers=2, num_attention_heads=2, num_kv_heads=1, head_dim=32,
              intermediate_size=128, vocab_size=256, max_position_embeddings=128, attn_impl="sdpa")
    if attn_res:
        kw.update(attn_res=True, attn_res_boundaries=[0, 1])
    return KotodamaModel(KotodamaModelConfig(**kw))


def _mon(model: KotodamaModel) -> GeometricMonitor:
    return GeometricMonitor(model, MonitorConfig(tier1_sample_layers=[0, 1], tier2_twonn_layers=[0]))


def test_seeded_power_iteration_converges():
    mon = _mon(_tiny())
    scale = torch.tensor([2.0, 1.5] + [0.5] * 62)
    fn = lambda h: h * scale  # symmetric J: dominant |eigenvalue| = 2.0
    h = torch.randn(1, 8, 64)
    r = mon._power_iterate(h, fn, torch.device("cpu"), mon.EOC_ITERS)
    assert r.sigma == pytest.approx(2.0, rel=5e-3)
    assert r.iters == mon.EOC_ITERS and r.drift <= mon.EOC_DRIFT_TOL


def test_short_budget_reads_as_unconverged():
    """Four steps from a random start leave σ still climbing toward |λ1|: the drift says so, and the aggregate counts
    every such sub-layer, instead of passing the partial σ off as a converged one."""
    mon = _mon(_tiny())
    scale = torch.tensor([2.0, 1.99] + [0.5] * 62)
    h = torch.randn(1, 8, 64)
    r = mon._power_iterate(h, lambda x: x * scale, torch.device("cpu"), 4)
    assert r.iters == 4 and r.drift > mon.EOC_DRIFT_TOL


def test_power_iteration_is_continuous_in_the_weights():
    """A fixed budget from a fixed start makes σ a continuous function of the operator: a small perturbation of the
    weights moves σ by a small amount (an early stop on a per-step tolerance does not have this property: where the
    stopping step changes, σ jumps)."""
    mon = _mon(_tiny())
    torch.manual_seed(2)
    W, E = torch.randn(64, 64) / 8, torch.randn(64, 64) / 8
    h = torch.randn(2, 8, 64)
    sig = [mon._power_iterate(h, lambda x, _W=W + t * E: x @ _W, torch.device("cpu"), 20).sigma
           for t in (0.0, 1e-5, 2e-5)]
    assert abs(sig[1] - sig[0]) <= 1e-3 * sig[0] and abs(sig[2] - sig[1]) <= 1e-3 * sig[0]
    assert sig[0] != sig[1] != sig[2]  # the perturbation reaches σ


def _snap(model: KotodamaModel) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in model.state_dict().items()}


def _eoc(m: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in m.items() if k.startswith("eoc/") and k != "eoc/time_s"}


def test_eoc_readout_is_independent_of_call_history():
    """The same model and probe read the same σ (to the bit) whatever the monitor measured before, and however many
    times it measured: a start vector carried between calls, or re-drawn on a call-count cadence, steps every series
    at the redraw."""
    model = _tiny(attn_res=True)
    ids = torch.randint(1, 256, (4, 16), generator=torch.Generator().manual_seed(3))
    state_a = _snap(model)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(1.05)
    state_b = _snap(model)

    seen = _mon(model)
    seen.set_probe_batch(ids)
    history = []
    for state in (state_a, state_b, state_a, state_b, state_b):
        model.load_state_dict(state)
        history.append(_eoc(seen.compute_jacobian_spectral(step=0)))

    fresh = _mon(model)
    fresh.set_probe_batch(ids)
    model.load_state_dict(state_b)
    want_b = _eoc(fresh.compute_jacobian_spectral(step=0))
    assert history[1] == history[3] == history[4] == want_b
    assert history[0] == history[2] != want_b
    assert want_b["eoc/iters"] == fresh.EOC_ITERS


def test_eoc_probe_leaves_training_rng_untouched():
    """The start vector comes from the monitor's own generator: a probe call between two training draws leaves the
    global CPU (and CUDA) RNG streams where they were."""
    model = _tiny(attn_res=True)
    mon = _mon(model)
    mon.set_probe_batch(torch.randint(1, 256, (4, 16), generator=torch.Generator().manual_seed(3)))
    torch.manual_seed(11)
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    m = mon.compute_jacobian_spectral(step=0)
    assert "eoc/sigma_max" in m
    assert torch.equal(torch.get_rng_state(), cpu_state)
    if cuda_state is not None:
        assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), cuda_state))


def test_start_vector_is_fixed_by_shape_and_seed():
    a, b = _mon(_tiny()), _mon(_tiny())
    va = a._eoc_start((4, 8, 64))
    assert torch.equal(va, b._eoc_start((4, 8, 64)))
    assert va is a._eoc_start((4, 8, 64))  # drawn once per shape
    assert va.dtype == torch.float32 and va.device.type == "cpu"


def test_relative_gain_is_scale_free():
    """Norm-then-linear (layer 0 on the raw embedding): raw σ ∝ 1/‖h‖, the relative gain
    σ·‖h‖/‖f(h)‖ does not move when the input is rescaled."""
    mon = _mon(_tiny())
    torch.manual_seed(1)
    W = torch.randn(64, 64) / 8
    W = (W + W.T) / 2
    fn = lambda h: (h / h.pow(2).mean(-1, keepdim=True).sqrt()) @ W
    h = torch.randn(1, 8, 64) * 0.03
    s_small, g_small, _, _ = mon._power_iterate(h, fn, torch.device("cpu"), 30)
    s_big, g_big, _, _ = mon._power_iterate(h * 100, fn, torch.device("cpu"), 30)
    assert s_small / s_big == pytest.approx(100, rel=0.05)
    assert g_small == pytest.approx(g_big, rel=0.05)


def test_mlp_probed_at_its_own_routed_input():
    model = _tiny(attn_res=True).eval()
    mon = _mon(model)
    ids = torch.randint(1, 256, (2, 16))
    hs, _ = mon._probe_forward(ids)
    assert set(mon._mlp_inputs) == {0, 1}
    # layer 0: pre-attn input is the embedding; pre-MLP input includes attn's output
    assert not torch.allclose(mon._mlp_inputs[0], hs[0])
    assert not model.layers[0].ffn_norm._forward_pre_hooks  # hooks removed
    m = mon._eoc_metrics(hs, torch.device("cpu"), n_iters=5)
    for i in (0, 1):
        for sub in ("attn", "mlp"):
            assert math.isfinite(m[f"eoc/jacobian_gain/layer_{i}/{sub}"])
            assert m[f"eoc/jacobian_sigma/layer_{i}/{sub}"] > 0
            assert m[f"eoc/jacobian_drift/layer_{i}/{sub}"] >= 0
    assert {"eoc/gain_max", "eoc/gain_mean", "eoc/sigma_max", "eoc/layer0_sigma"} <= set(m)
    assert m["eoc/iters"] == 5 and 0 <= m["eoc/unconverged"] <= 4
    assert m["eoc/drift_max"] == max(v for k, v in m.items() if k.startswith("eoc/jacobian_drift/"))
