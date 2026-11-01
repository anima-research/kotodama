"""KotodamaModel(config, init_weights=False): the loaders' construction, for weights that a checkpoint supplies.

It skips the init pass, and a looped model's repeat positions build no attention or FFN of their own (they are
aliased to the core's), so construction holds the unique weights once instead of once per effective position. The
default construction still builds every position's modules and draws the init pass, so a seed gives the same model
it always has. CPU, a tiny looped model.
"""

from __future__ import annotations

import pytest
import torch

from kotodama.model import llama
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig

CFG = dict(hidden_size=64, num_layers=8, num_attention_heads=4, num_kv_heads=2, head_dim=16, intermediate_size=128,
           vocab_size=256, max_position_embeddings=64, loop_start=2, loop_len=2, loop_times=3, attn_impl="sdpa")


def _built_attention(monkeypatch, init_weights: bool) -> list[str]:
    """The device of every GQAttention's q_proj as it is built."""
    devices = []
    orig = llama.GQAttention.__init__

    def record(self, config):
        orig(self, config)
        devices.append(self.q_proj.weight.device.type)

    monkeypatch.setattr(llama.GQAttention, "__init__", record)
    KotodamaModel(KotodamaModelConfig(**CFG), init_weights=init_weights)
    return devices


def test_loader_construction_builds_only_the_unique_layers(monkeypatch) -> None:
    cfg = KotodamaModelConfig(**CFG)
    lean = _built_attention(monkeypatch, init_weights=False)
    assert lean.count("cpu") == cfg.num_unique_layers and lean.count("meta") == cfg.num_layers - cfg.num_unique_layers


def test_default_construction_is_unchanged(monkeypatch) -> None:
    assert _built_attention(monkeypatch, init_weights=True) == ["cpu"] * CFG["num_layers"]  # the seed's draws
    torch.manual_seed(0)
    a = KotodamaModel(KotodamaModelConfig(**CFG)).state_dict()
    torch.manual_seed(0)
    b = KotodamaModel(KotodamaModelConfig(**CFG)).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("attn_res", [False, True])
def test_loaded_model_matches_the_saved_one(attn_res: bool) -> None:
    cfg = dict(CFG, attn_res=attn_res, attn_res_boundaries=[0, 2, 4, 6] if attn_res else None)
    torch.manual_seed(0)
    saved = KotodamaModel(KotodamaModelConfig(**cfg)).eval()
    loaded = KotodamaModel(KotodamaModelConfig(**cfg), init_weights=False)
    loaded.load_state_dict(saved.state_dict(), strict=True)
    loaded.eval()
    assert not any(t.is_meta for t in list(loaded.parameters()) + list(loaded.buffers()))
    for i in range(cfg["num_layers"]):
        src = loaded.config.layer_source(i)
        assert loaded.layers[i].attn is loaded.layers[src].attn and loaded.layers[i].ffn is loaded.layers[src].ffn
    ids = torch.randint(0, cfg["vocab_size"], (2, 16))
    with torch.no_grad():
        assert torch.equal(loaded(ids)["logits"], saved(ids)["logits"])
