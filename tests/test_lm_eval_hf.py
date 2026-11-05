"""The HF lm-eval runner scores an HF causal LM through KotodamaEvalLM's batched,
right-padded, mask-free path; that path must equal scoring each request alone."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("lm_eval")
transformers = pytest.importorskip("transformers")

from scripts.eval.run_lm_eval_hf import load_tokenizer, make_hf_eval_lm, weight_fingerprint  # noqa: E402

VOCAB = 64


class ByteTok:
    """Minimal tokenizer: one token per character, mod VOCAB, ids >= 1."""

    eos_token = "<eos>"
    eos_token_id = 0
    pad_token = None
    pad_token_id = None

    def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
        return [1 + (ord(c) % (VOCAB - 1)) for c in s]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(chr(i) for i in ids)


def _tiny_model() -> torch.nn.Module:
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
    )
    return transformers.LlamaForCausalLM(cfg).eval()


def _lm(model, batch_size: int = 4, max_length: int = 256):
    tok = ByteTok()
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    return make_hf_eval_lm(model, tok, "cpu", batch_size, max_length, 32768, "tiny")


def _alone(model, ctx: list[int], cont: list[int]) -> float:
    ids = torch.tensor([ctx + cont])
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    lp = F.log_softmax(logits[len(ctx) - 1 : -1], dim=-1)
    return lp.gather(1, torch.tensor(cont).unsqueeze(-1)).sum().item()


def test_batched_loglikelihood_matches_unbatched():
    model = _tiny_model()
    lm = _lm(model)
    tok = ByteTok()
    pairs = [("The cat sat on", " the mat"), ("A", " b"), ("Longer context here, much longer", " yes"),
             ("Mid length", " continuation text"), ("x" * 90, " y")]
    reqs = [((c, x), tok.encode(c), tok.encode(x)) for c, x in pairs]
    got = lm._loglikelihood_tokens(reqs)
    for (_, ctx, cont), (ll, _greedy) in zip(reqs, got):
        assert ll == pytest.approx(_alone(model, ctx, cont), abs=1e-4)


def test_fingerprint_tracks_weights():
    a, b = _tiny_model(), _tiny_model()
    assert weight_fingerprint(a)["sha256"] == weight_fingerprint(b)["sha256"]
    with torch.no_grad():
        next(b.parameters()).add_(1e-3)
    assert weight_fingerprint(a)["sha256"] != weight_fingerprint(b)["sha256"]


def test_generation_refused():
    with pytest.raises(NotImplementedError):
        _lm(_tiny_model()).generate_until([])


@pytest.mark.parametrize("trust", [False, True])
def test_load_tokenizer_forwards_trust_remote_code(monkeypatch, trust):
    """A custom-code tokenizer loads only with trust_remote_code, on the revision's load and on main's fallback."""
    calls: list[dict] = []

    class Empty(ByteTok):
        def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
            return [] if calls[-1]["revision"] != "main" else super().encode(s)

    def fake(repo, **kw):
        calls.append(kw)
        return Empty()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(fake))
    tok, rev = load_tokenizer("org/model", "step-100", None, trust_remote_code=trust)
    assert rev == "main" and tok.encode("ab")
    assert [c["revision"] for c in calls] == ["step-100", "main"]
    assert all(c["trust_remote_code"] is trust for c in calls)
