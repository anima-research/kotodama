"""LoRA adapters on a tiny LOOPED model (cf. tests/test_posttrain_engines.py): layer groups, attach, save, merge, an
SFT run that trains only the adapter, and the server merging an adapter at load. CPU, fp32."""
from __future__ import annotations

import json

import pytest
import torch

from kotodama import presets
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
from kotodama.posttrain import lora

pytest.importorskip("peft")
pytest.importorskip("safetensors")

# prelude 0-1, a 2-layer core run twice (2-5), coda 6, at the SmolLM2 vocab so the real tokenizer's ids fit.
TINY = dict(hidden_size=32, num_layers=7, num_attention_heads=2, num_kv_heads=1, head_dim=16, intermediate_size=64,
            vocab_size=49152, max_position_embeddings=512, loop_start=2, loop_len=2, loop_times=2)
BOUNDS = "0,1,2,4,6"


def _config() -> KotodamaModelConfig:
    return KotodamaModelConfig(**TINY, attn_res=True, attn_res_boundaries=[int(x) for x in BOUNDS.split(",")],
                               attn_impl="sdpa", rope_theta=10000.0)


def _model(seed: int = 0) -> KotodamaModel:
    torch.manual_seed(seed)
    model = KotodamaModel(_config()).eval()
    with torch.no_grad():  # routing off its zero init so every block matters
        for n, p in model.named_parameters():
            if "res_query" in n:
                p.normal_(0, 0.5)
    return model


def _ids() -> torch.Tensor:
    return torch.randint(5, 49152, (2, 20), generator=torch.Generator().manual_seed(3))


def _randomize_b(trainable: dict[str, torch.nn.Parameter]) -> None:
    """B starts at zero (an identity adapter); give it values so the adapter changes the model."""
    g = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for n, p in trainable.items():
            if "lora_B" in n:
                p.copy_(0.05 * torch.randn(p.shape, generator=g))


def test_layer_groups_follow_the_loop() -> None:
    cfg = _config()
    assert lora.layer_groups(cfg) == {"prelude": [0, 1], "core": [2, 3], "coda": [6]}
    assert lora.resolve_layers("all", cfg) == [0, 1, 2, 3, 6]
    assert lora.resolve_layers("core,coda", cfg) == [2, 3, 6]
    assert lora.resolve_layers("0-2", cfg) == [0, 1, 2]
    with pytest.raises(ValueError, match="loop repeat"):
        lora.resolve_layers("4", cfg)  # the core's second pass


def test_attach_is_identity_at_init_and_trains_only_the_adapter() -> None:
    model = _model()
    with torch.no_grad():
        before = model(_ids())["logits"]
    trainable = lora.attach(model, rank=4, alpha=8.0, targets=list(lora.TARGETS), layers=[2, 3, 6])
    with torch.no_grad():
        after = model(_ids())["logits"]
    assert torch.equal(before, after)
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    assert set(trainable) == {n for n, p in model.named_parameters() if p.requires_grad}
    assert not any(".layers.0." in f".{n}" or ".layers.1." in f".{n}" for n in trainable)  # prelude untouched


def test_core_adapter_is_one_update_on_both_passes() -> None:
    model = _model()
    lora.attach(model, rank=4, alpha=8.0, targets=["q_proj"], layers=[2])
    assert model.layers[4].attn.q_proj is model.layers[2].attn.q_proj
    assert hasattr(model.layers[4].attn.q_proj, "lora_A")
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names == ["layers.2.attn.q_proj.lora_A.default.weight", "layers.2.attn.q_proj.lora_B.default.weight"]


def test_save_then_merge_reproduces_the_adapted_model(tmp_path) -> None:
    adapted = _model()
    trainable = lora.attach(adapted, rank=4, alpha=8.0, targets=list(lora.TARGETS), layers=lora.resolve_layers("all", adapted.config))
    _randomize_b(trainable)
    with torch.no_grad():  # the saved adapter is bf16; compare against the adapter as saved
        for p in trainable.values():
            p.copy_(p.to(torch.bfloat16).float())
    lora.save(tmp_path / "a.adapter", adapted, {"rank": 4, "alpha": 8.0})

    merged = _model()
    meta = lora.merge(merged, tmp_path / "a.adapter")
    assert meta["rank"] == 4
    with torch.no_grad():
        want, got, base = adapted(_ids())["logits"], merged(_ids())["logits"], _model()(_ids())["logits"]
    assert not torch.allclose(want, base, atol=1e-3)  # the adapter really changes the model
    assert torch.allclose(got, want, atol=1e-4)


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(presets.TOKENIZER_NAME)
    except OSError as e:
        pytest.skip(f"tokenizer unavailable: {e}")


@pytest.fixture()
def tiny_ckpt(tmp_path, monkeypatch):
    monkeypatch.setitem(presets.SHAPES, "tiny-loop", dict(TINY))
    path = tmp_path / "base.pt"
    torch.save({"step": 0, "model": _model().state_dict()}, path)
    return path


def test_sft_lora_trains_and_saves_only_an_adapter(tiny_ckpt, tok, tmp_path) -> None:
    from kotodama.eval.model_loader import load_model
    from kotodama.posttrain import sft

    conv = {"messages": [{"role": "user", "content": "say the word"},
                         {"role": "assistant", "content": "lantern lantern lantern"}]}
    data = tmp_path / "chat.jsonl"
    data.write_text("\n".join(json.dumps(conv) for _ in range(16)) + "\n")
    out = tmp_path / "sft"
    rc = sft.main(["--checkpoint", str(tiny_ckpt), "--model_size", "tiny-loop", "--attn_res_boundaries", BOUNDS,
                   "--device", "cpu", "--attn_impl", "sdpa", "--no-rope_fp32", "--out_dir", str(out),
                   "--format", "chatml", "--train_jsonl", str(data), "--eval_jsonl", str(data), "--max_seq_len", "64",
                   "--micro_batch_size", "2", "--grad_accum", "1", "--max_steps", "12", "--muon_lr", "0.05",
                   "--adamw_lr", "1e-2", "--warmup_ratio", "0", "--save_every", "6",
                   "--lora_rank", "4", "--lora_layers", "core,coda"])
    assert rc == 0
    log = [json.loads(x) for x in (out / "train_log.jsonl").read_text().splitlines()]
    evals = [r for r in log if "eval/loss" in r]
    assert evals[-1]["eval/loss"] < evals[0]["eval/loss"]
    final = json.loads((out / "final.json").read_text())
    assert final["adapter"] is True
    adapter = tmp_path / "sft" / "checkpoints" / "step_00000012.adapter"
    assert str(adapter) == final["checkpoint"]
    assert (adapter / lora.WEIGHTS_FILE).exists() and (out / "checkpoints" / "step_00000006.adapter").exists()
    assert not list((out / "checkpoints").glob("*.pt"))  # no full-weights checkpoint: the base stays the base
    meta = json.loads((adapter / lora.META_FILE).read_text())
    assert meta["rank"] == 4 and meta["layers"] == [2, 3, 6] and meta["step"] == 12

    bounds = {"attn_res": True, "attn_res_boundaries": [int(x) for x in BOUNDS.split(",")]}
    base = load_model(tiny_ckpt, config_section="tiny-loop", device="cpu", attn_res_config=bounds,
                      overrides={"attn_impl": "sdpa"})
    tuned = load_model(tiny_ckpt, config_section="tiny-loop", device="cpu", attn_res_config=bounds,
                       overrides={"attn_impl": "sdpa"})
    lora.merge(tuned, adapter)
    ids = torch.tensor([tok.encode("user: say the word\nassistant: lantern lantern")])
    with torch.no_grad():
        assert not torch.allclose(base(ids)["logits"], tuned(ids)["logits"], atol=1e-4)


def test_server_merges_an_adapter_at_load(tiny_ckpt, tok, tmp_path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from kotodama.serve import server

    adapted = _model()
    trainable = lora.attach(adapted, rank=4, alpha=8.0, targets=list(lora.TARGETS), layers=[0, 2, 6])
    _randomize_b(trainable)
    lora.save(tmp_path / "wry.adapter", adapted, {"rank": 4, "alpha": 8.0})

    cfg = dict(TINY, attn_res=True, attn_res_boundaries=[int(x) for x in BOUNDS.split(",")], attn_impl="sdpa",
               rope_theta=10000.0)
    monkeypatch.setitem(server.MODEL_CONFIGS, "tiny-loop", cfg)
    monkeypatch.setattr(server, "_model_size_arg", "tiny-loop")
    monkeypatch.setattr(server, "_adapter_arg", str(tmp_path / "wry.adapter"))
    served, _ = server.load_model(str(tiny_ckpt), device="cpu", model_size="tiny-loop", engine="reference",
                                  adapter=str(tmp_path / "wry.adapter"))
    want = _model()
    lora.merge(want, tmp_path / "wry.adapter")
    with torch.no_grad():
        assert torch.allclose(served(_ids())["logits"], want(_ids())["logits"], atol=1e-5)
    assert server._model_name().endswith("+wry")
