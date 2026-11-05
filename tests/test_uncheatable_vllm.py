"""The vLLM Uncheatable runner without vLLM: assembling chunk NLLs from both
``prompt_logprobs`` layouts (the start position never scored), and the batched
path reproducing ``kotodama.eval.uncheatable.evaluate`` on a toy causal model."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch
import torch.nn.functional as F

from kotodama.eval import uncheatable as U
from scripts.eval.run_uncheatable_vllm import (
    ScoringError,
    chunk_nll_from_prompt_logprobs,
    doc_chunks,
    eval_domain_batched,
    evaluate_batched,
)

V = 300  # byte b -> id b + 1; id 0 = the document-start token


def encode(text: str) -> list[int]:
    return [b + 1 for b in text.encode("utf-8")]


@dataclass
class FakeLogprob:
    logprob: float


@dataclass
class FakeFlat:
    """The attribute layout of vLLM's ``FlatLogprobs``."""
    start_indices: list[int] = field(default_factory=list)
    end_indices: list[int] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)

    def append(self, entry: dict[int, float] | None) -> None:
        self.start_indices.append(len(self.logprobs))
        for t, lp in (entry or {}).items():
            self.token_ids.append(t)
            self.logprobs.append(lp)
        self.end_indices.append(len(self.logprobs))

    def __len__(self) -> int:
        return len(self.start_indices)


torch.manual_seed(0)
W = torch.randn(V, V)  # toy causal model: logits at t depend on ids[t] and t only


def toy_logits(ids: torch.Tensor) -> torch.Tensor:
    ids = ids.reshape(-1)
    pos = torch.arange(ids.numel(), dtype=torch.float32).unsqueeze(1)
    return W[ids] + 0.01 * pos


def fake_prompt_logprobs(chunk: list[int], flat: bool, extra_top: bool = False):
    """What vLLM returns for ``prompt_logprobs=0``: None at position 0, then the
    prompt token's fp32 log-softmax at each later position (``extra_top`` adds a
    second, unrelated token the way ``prompt_logprobs=k>0`` would)."""
    lsm = F.log_softmax(toy_logits(torch.tensor(chunk)).float(), dim=-1)
    entries: list[dict[int, float] | None] = [None]
    for i in range(1, len(chunk)):
        e = {chunk[i]: float(lsm[i - 1, chunk[i]])}
        if extra_top:
            other = (chunk[i] + 1) % V
            e = {other: float(lsm[i - 1, other]), **e}
        entries.append(e)
    if flat:
        out = FakeFlat()
        for e in entries:
            out.append(e)
        return out
    return [None if e is None else {t: FakeLogprob(lp) for t, lp in e.items()}
            for e in entries]


def fake_scorer(flat: bool, seen: list[list[int]] | None = None):
    def score(chunks):
        if seen is not None:
            seen.extend(chunks)
        return [chunk_nll_from_prompt_logprobs(c, fake_prompt_logprobs(c, flat)) for c in chunks]
    return score


# ── assembly ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("extra_top", [False, True])
def test_chunk_nll_matches_cross_entropy_and_skips_start(flat, extra_top):
    chunk = [0, 5, 17, 17, 200, 3]
    got = chunk_nll_from_prompt_logprobs(chunk, fake_prompt_logprobs(chunk, flat, extra_top))
    ids = torch.tensor(chunk)
    want = F.cross_entropy(toy_logits(ids)[:-1].float(), ids[1:], reduction="sum").item()
    assert got == pytest.approx(want, rel=1e-6)


def test_chunk_nll_rejects_missing_or_misaligned_logprobs():
    chunk = [0, 5, 17]
    good = fake_prompt_logprobs(chunk, flat=False)
    with pytest.raises(ScoringError):
        chunk_nll_from_prompt_logprobs(chunk, None)
    with pytest.raises(ScoringError):
        chunk_nll_from_prompt_logprobs(chunk + [9], good)  # length mismatch
    wrong = [None, {6: FakeLogprob(-1.0)}, good[2]]          # not the prompt token
    with pytest.raises(ScoringError):
        chunk_nll_from_prompt_logprobs(chunk, wrong)
    flat_wrong = FakeFlat()
    for e in (None, {6: -1.0}, {17: -1.0}):
        flat_wrong.append(e)
    with pytest.raises(ScoringError):
        chunk_nll_from_prompt_logprobs(chunk, flat_wrong)
    with pytest.raises(ScoringError):
        chunk_nll_from_prompt_logprobs(chunk, [None, {5: FakeLogprob(float("nan"))}, good[2]])


def test_doc_chunks_match_doc_nll_forward_inputs():
    ids = list(range(1, 12))
    seen: list[list[int]] = []

    def spy(t: torch.Tensor) -> torch.Tensor:
        seen.append(t.reshape(-1).tolist())
        return toy_logits(t)

    U.doc_nll(spy, ids, bos_id=0, chunk_size=4, loss_precision="fp32")
    assert doc_chunks(ids, 0, 4) == seen == [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11]]
    with pytest.raises(ValueError):
        doc_chunks([7], 0, 4)


# ── batched path == the module's evaluate ────────────────────────────────────

DOCS = {
    "alpha": ["hello world, this is a document", "tiny bit", "ümlaut ç 字 ok"],
    "beta": ["x" * 50, "print('hi')\n" * 3],
}


def _settings(**kw) -> U.Settings:
    base = dict(checkpoint="stub", config_section="hf", attn_res=False,
                attn_res_boundaries=None, rope_fp32=False, tokenizer="stub",
                bos_mode="tokenizer_bos", bos_id=0, chunk_size=7,
                loss_precision="fp32", max_docs_per_domain=None,
                max_tokens_per_domain=None, data_source="mem", device="cpu",
                static_shapes=False)
    base.update(kw)
    return U.Settings(**base)


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("batch_chunks", [1, 3, 1000])
@pytest.mark.parametrize("trim", [{}, {"max_docs_per_domain": 2}, {"max_tokens_per_domain": 20}])
def test_batched_result_equals_module_evaluate(flat, batch_chunks, trim):
    settings = _settings(**trim)
    ref = U.evaluate(toy_logits, encode, DOCS, settings)
    got = evaluate_batched(fake_scorer(flat), encode, DOCS, settings, batch_chunks)
    assert got.keys() == ref.keys()
    assert got["schema"] == U.SCHEMA
    assert got["mean_bpb"] == pytest.approx(ref["mean_bpb"], rel=1e-6)
    assert (got["n_domains"], got["trimmed"]) == (ref["n_domains"], ref["trimmed"])
    for name, r in ref["domains"].items():
        g = got["domains"][name]
        assert g.keys() == r.keys()
        for k in ("n_docs", "n_docs_total", "n_tokens", "n_bytes", "n_chars", "trimmed"):
            assert g[k] == r[k], (name, k)
        for k in ("bpb", "bpc", "nll_mean"):
            assert g[k] == pytest.approx(r[k], rel=1e-6), (name, k)


def test_every_token_scored_once_with_start_token_per_chunk():
    seen: list[list[int]] = []
    texts = ["abcdefghij", "klmnopqrstuvwxyz0123"]
    r = eval_domain_batched(fake_scorer(False, seen), encode, texts, bos_id=0, chunk_size=6,
                            batch_chunks=2)
    assert all(c[0] == 0 and 2 <= len(c) <= 7 for c in seen)
    assert sum(len(c) - 1 for c in seen) == r.n_tokens == 30


def test_scorer_returning_wrong_count_is_an_error():
    with pytest.raises(ScoringError):
        eval_domain_batched(lambda chunks: [1.0], encode, ["abcdefgh"], 0, chunk_size=3,
                            batch_chunks=4)

