"""Fused energy routing (--attn_res_energy_impl fused, kotodama.model.energy_fused). CPU: the impl option, the
dispatch (a CPU or unsupported call falls back to the PyTorch op, so a CPU model gives the same output either way).
GPU with Triton: the fused op against the PyTorch op computed in fp64, judged against the bf16 PyTorch op's own error,
at σ = 0 and σ ≠ 0, and through torch.compile."""

from __future__ import annotations

import pytest
import torch

from kotodama.model import energy_fused as ef
from kotodama.model import energy_route as er
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
from kotodama.training import train

BASE = dict(hidden_size=64, num_layers=7, num_attention_heads=4, num_kv_heads=2, head_dim=16,
            intermediate_size=128, vocab_size=199, max_position_embeddings=96, attn_impl="sdpa",
            attn_res=True, attn_res_boundaries=[0, 2, 5], use_liger=False,
            loop_start=1, loop_len=2, loop_times=2, attn_res_energy_steps=2, attn_res_energy_rank=8)


def _model(impl: str) -> KotodamaModel:
    torch.manual_seed(20261119)
    m = KotodamaModel(KotodamaModelConfig(**BASE, attn_res_energy_impl=impl))
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.endswith("res_query") or name.endswith("res_energy_sigma"):
                p.copy_(torch.randn(p.shape, generator=g) * (0.5 if "sigma" in name else 1.0))
    return m


def test_impl_defaults_to_torch_and_is_validated():
    assert KotodamaModelConfig().attn_res_energy_impl == "torch"
    with pytest.raises(ValueError, match="attn_res_energy_impl"):
        KotodamaModelConfig(**BASE, attn_res_energy_impl="triton")


def test_flag_reaches_the_config():
    import sys

    old = sys.argv
    try:
        sys.argv = ["train.py", "--model_size", "proxy"]
        assert train.parse_args().attn_res_energy_impl == "torch"
        sys.argv = ["train.py", "--model_size", "proxy", "--attn_res_energy_impl", "fused"]
        assert train.parse_args().attn_res_energy_impl == "fused"
    finally:
        sys.argv = old


def test_energy_for_carries_the_impl():
    m = _model("fused")
    U, sigma, steps, impl = m._energy_for(m.layers[2], "attn")
    assert impl == "fused" and steps == 2 and U.shape == (64, 8) and sigma.shape == (8,)


def test_cpu_call_falls_back_to_the_torch_op():
    g = torch.Generator().manual_seed(0)
    c, p = torch.randn(3, 2, 8, 64, generator=g), torch.randn(2, 8, 64, generator=g)
    qw, nw = torch.randn(64, generator=g), 1 + 0.1 * torch.randn(64, generator=g)
    U, sigma = torch.randn(64, 8, generator=g) / 8, torch.randn(8, generator=g)
    assert not ef.fused_supported(c, p, U, 2)
    assert torch.equal(ef.route_energy_split_fused(c, p, qw, nw, 1e-5, U, sigma, 2),
                       er.route_energy_split(c, p, qw, nw, 1e-5, U, sigma, 2))


def test_cpu_model_is_the_same_with_either_impl():
    ids = torch.randint(3, 199, (2, 24), generator=torch.Generator().manual_seed(3))
    a, b = _model("torch").eval(), _model("fused").eval()
    with torch.no_grad():
        assert torch.equal(a(ids)["logits"], b(ids)["logits"])


def _gpu_ok() -> bool:
    return torch.cuda.is_available() and ef.AVAILABLE


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300))


@pytest.mark.skipif(not _gpu_ok(), reason="needs CUDA + Triton")
@pytest.mark.parametrize("n,sigma_scale", [(1, 0.0), (3, 0.0), (3, 0.3), (5, 0.3)])
def test_fused_matches_the_torch_op_within_its_own_error(n, sigma_scale):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(100 * n + int(10 * sigma_scale))
    D, B, T, r = 256, 2, 128, 32
    rn = lambda *s: torch.randn(*s, device=dev, generator=g, dtype=torch.float64)
    vals64 = [rn(n, B, T, D), rn(B, T, D), rn(D) * (0.3 / D ** 0.5), 1 + 0.1 * rn(D), rn(D, r) * D ** -0.5,
              rn(r) * sigma_scale]
    dh = rn(B, T, D)
    bf = [v.to(torch.bfloat16) for v in vals64]

    def run(fn, leaves, grad):
        xs = [t.detach().clone().requires_grad_() for t in leaves]
        h = fn(xs[0], xs[1], xs[2], xs[3], 1e-5, xs[4], xs[5], 2)
        h.backward(grad.to(h.dtype))
        return [h.detach()] + [x.grad for x in xs]

    truth = run(er.route_energy_split, [v.double() for v in bf], dh.double())
    ref = run(er.route_energy_split, bf, dh.to(torch.bfloat16))
    assert ef.fused_supported(bf[0], bf[1], bf[4], 2)
    fused = run(ef.route_energy_split_fused, bf, dh.to(torch.bfloat16))
    for i, (f, r_, t) in enumerate(zip(fused, ref, truth)):
        if t.norm() == 0:                      # dnw / dU at σ = 0
            assert f.double().norm() < 1e-6, i
            continue
        assert _rel(f, t) <= max(1.5 * _rel(r_, t), 2e-3), (i, _rel(f, t), _rel(r_, t))


@pytest.mark.skipif(not _gpu_ok(), reason="needs CUDA + Triton")
def test_fused_runs_under_torch_compile():
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(5)
    D, B, T, r = 256, 1, 128, 32
    bf = lambda *s, sc=1.0: (torch.randn(*s, device=dev, generator=g) * sc).to(torch.bfloat16)
    leaves = [bf(2, B, T, D).requires_grad_(), bf(B, T, D).requires_grad_(), bf(D, sc=0.02).requires_grad_(),
              (1 + 0.1 * bf(D)).requires_grad_(), bf(D, r, sc=D ** -0.5).requires_grad_(), bf(r, sc=0.3).requires_grad_()]
    eager = ef.route_energy_split_fused(leaves[0], leaves[1], leaves[2], leaves[3], 1e-5, leaves[4], leaves[5], 2)
    comp = torch.compile(ef.route_energy_split_fused, fullgraph=True)(
        leaves[0], leaves[1], leaves[2], leaves[3], 1e-5, leaves[4], leaves[5], 2)
    assert torch.equal(eager, comp)
    comp.float().sum().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad.float()).all() for x in leaves)


def test_layer_scope_regions_route_with_energy():
    """compile_regions(scope="layer") (and FSDP2's layer scope, which runs the same regions) routes every site with
    energy: same loss and same energy/query gradients as the eager per-sublayer path, and U receives a gradient
    (a region that dropped the energy tuple would route with the plain router and leave U without one)."""
    ids = torch.randint(3, 199, (2, 24), generator=torch.Generator().manual_seed(3))

    def run(regions: bool):
        m = _model("torch").train()
        if regions:
            m.compile_regions(scope="layer")
        loss = m(ids, labels=ids)["loss"]
        loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in m.named_parameters()
                 if p.grad is not None and ("energy" in n or "res_query" in n)}
        return loss.detach(), grads

    l_eager, g_eager = run(False)
    l_region, g_region = run(True)
    assert "attn_res_energy_U" in g_region and g_region["attn_res_energy_U"].abs().sum() > 0
    torch.testing.assert_close(l_region, l_eager, rtol=1e-4, atol=1e-5)
    assert g_region.keys() == g_eager.keys()
    for n in g_eager:
        torch.testing.assert_close(g_region[n], g_eager[n], rtol=2e-3, atol=1e-5, msg=n)
