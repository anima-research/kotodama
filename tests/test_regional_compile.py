"""Regional compile (KotodamaModel.compile_regions): two compiled sublayer functions shared by every layer
(scope "sublayer"), or one compiled routed layer per committed-block count and boundary (scope "layer").

Gates: the number of dynamo graphs is fixed (sublayer: attn with/without a grad-carrying partial, mlp; layer: two per
block depth) whatever the depth, looped repeats included, and a second batch adds none; train-mode loss and grads match the eager model (aot_eager
backend: same ops, no fusion) with and without activation checkpointing; eval mode runs the eager sublayers (no
dynamo frames — in-loop evals and monitors keep eager numerics, as under whole-model compile); the state_dict is
unchanged; non-AttnRes models refuse.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch._dynamo
from torch._dynamo.utils import counters

from kotodama.model.llama import KotodamaModel, KotodamaModelConfig

BASE = dict(hidden_size=64, num_attention_heads=4, num_kv_heads=2, head_dim=16, intermediate_size=128,
            vocab_size=199, max_position_embeddings=96, attn_impl="sdpa", attn_res=True, use_liger=False)
SHAPES = {
    "plain-7": dict(num_layers=7, attn_res_boundaries=[0, 2, 5]),
    "looped-12": dict(num_layers=12, attn_res_boundaries=[0, 2, 5, 9], loop_start=1, loop_len=3, loop_times=2),
}


# Graphs per scope: sublayer = attn (grad-carrying partial or not) + mlp; layer = (boundary, inner) per block depth.
GRAPHS = {("sublayer", "plain-7"): 3, ("sublayer", "looped-12"): 3, ("layer", "plain-7"): 6, ("layer", "looped-12"): 8}


def _pair(shape: str, ac: bool, scope: str = "sublayer") -> tuple[KotodamaModel, KotodamaModel]:
    torch.manual_seed(20261022)
    ref = KotodamaModel(KotodamaModelConfig(**BASE, **SHAPES[shape], activation_checkpointing=ac))
    ref.train()
    reg = copy.deepcopy(ref)
    torch._dynamo.reset()
    counters.clear()
    reg.compile_regions(scope=scope, backend="aot_eager")
    return ref, reg


def _loss_and_grads(m: KotodamaModel, ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    m.zero_grad(set_to_none=True)
    loss = m(ids, labels=ids)["loss"]
    loss.backward()
    return loss.detach(), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("scope", ["sublayer", "layer"])
@pytest.mark.parametrize("ac", [False, True])
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_regional_matches_eager_with_fixed_graph_count(shape: str, ac: bool, scope: str) -> None:
    ref, reg = _pair(shape, ac, scope)
    torch.manual_seed(1)
    ids = torch.randint(3, 199, (2, 24))
    l_ref, g_ref = _loss_and_grads(ref, ids)
    l_reg, g_reg = _loss_and_grads(reg, ids)
    assert torch.equal(l_ref, l_reg)
    assert g_ref.keys() == g_reg.keys()
    for n in g_ref:
        torch.testing.assert_close(g_reg[n], g_ref[n], rtol=1e-5, atol=1e-7, msg=n)
    assert counters["stats"]["unique_graphs"] == GRAPHS[scope, shape]
    _loss_and_grads(reg, torch.randint(3, 199, (2, 24)))
    assert counters["stats"]["unique_graphs"] == GRAPHS[scope, shape]


@pytest.mark.parametrize("scope", ["sublayer", "layer"])
def test_eval_mode_runs_eager(scope: str) -> None:
    ref, reg = _pair("looped-12", ac=False, scope=scope)
    ref.eval()
    reg.eval()
    ids = torch.randint(3, 199, (2, 24))
    with torch.no_grad():
        out_ref, out_reg = ref(ids)["logits"], reg(ids)["logits"]
    assert torch.equal(out_ref, out_reg)
    assert counters["stats"]["unique_graphs"] == 0


def test_state_dict_unchanged() -> None:
    ref, reg = _pair("looped-12", ac=False)
    sd_ref, sd_reg = ref.state_dict(), reg.state_dict()
    assert list(sd_ref) == list(sd_reg)
    assert all(torch.equal(sd_ref[k], sd_reg[k]) for k in sd_ref)


def test_non_attn_res_refuses() -> None:
    cfg = KotodamaModelConfig(**{**BASE, "attn_res": False}, num_layers=3)
    with pytest.raises(ValueError, match="AttnRes"):
        KotodamaModel(cfg).compile_regions(backend="aot_eager")


def test_bad_scope_refuses() -> None:
    cfg = KotodamaModelConfig(**BASE, **SHAPES["plain-7"])
    with pytest.raises(ValueError, match="scope"):
        KotodamaModel(cfg).compile_regions(scope="block", backend="aot_eager")
