"""Batch-invariant prefill (kotodama.model.regional.block_attend, DecodeEngine.extend_padded / prompt_kv_only), on a
tiny looped model, CPU fp32.

The kernel property itself (a row's bytes do not depend on the chunk that holds it) is the GPU's to show:
scripts/benchmark/kv_session_check.py and the split probe in the serving records. What is pinned here is the
engine's side of it:
  * ``extend_padded`` reads the cache's earlier positions without rewriting them, chunks greedily with the last
    chunk right-padded, returns the logits of the last real position (``block_last``, not the pad row), and leaves
    the engine at the real end with nothing reusable as prompt KV;
  * ``prefill_cached(prompt_kv_only=True)`` reuses only KV a prefill wrote, never a decode step's;
  * without flash-attn (CPU) the engine reports blocks as not batch-invariant, and block attention is SDPA.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn.attention import SDPBackend

from kotodama.model.decode_engine import DecodeEngine
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig

CFG = dict(hidden_size=64, num_layers=11, num_attention_heads=4, num_kv_heads=2, head_dim=16,
           intermediate_size=128, vocab_size=211, max_position_embeddings=128, rope_theta=10000.0,
           qk_norm=True, tie_word_embeddings=True, attn_impl="sdpa", use_liger=False,
           attn_res=True, attn_res_boundaries=[0, 1, 3, 6, 9], loop_start=3, loop_len=3, loop_times=2)
MAX_SEQ = 96


@pytest.fixture(scope="module")
def model() -> KotodamaModel:
    torch.manual_seed(20261104)
    return KotodamaModel(KotodamaModelConfig(**CFG)).eval()


def _ids(n: int, seed: int) -> list[int]:
    return torch.randint(14, CFG["vocab_size"], (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _engine(model: KotodamaModel, buckets=(16, 8)) -> DecodeEngine:
    eng = DecodeEngine(model, max_seq_len=MAX_SEQ, dtype=torch.float32, prefill_backends=[SDPBackend.MATH],
                       block_buckets=buckets)
    eng._compiled_block = eng._forward_block  # the block forward, uncompiled
    return eng


def test_on_cpu_blocks_are_not_batch_invariant(model: KotodamaModel) -> None:
    eng = DecodeEngine(model, max_seq_len=MAX_SEQ, dtype=torch.float32, block_buckets=(64, 16, 4))
    assert not eng.block_invariant and eng.block_span_max == 768 and eng.block_buckets == (64, 16, 4)
    with pytest.raises(RuntimeError, match="batch-invariant"):
        eng.extend_padded(torch.tensor([_ids(30, 1)]), 10)


@torch.inference_mode()
def test_extend_padded_reads_the_cache_and_returns_the_last_real_rows_logits(model: KotodamaModel) -> None:
    eng = _engine(model)
    eng.block_invariant = True  # the engine's side under test; the kernel property is the GPU's
    P = _ids(45, 2)
    n = 20
    eng.prefill(torch.tensor([P[:n]]))
    k0 = [c[0, :, :n].clone() for c in eng.k_caches]
    logits = eng.extend_padded(torch.tensor([P]), n)  # plan: 16 at 20, 8 at 36, then 1 left: an 8 padded by 7
    assert all(torch.equal(c[0, :, :n], k) for c, k in zip(eng.k_caches, k0))  # earlier positions not rewritten
    with torch.inference_mode():
        want = model(torch.tensor([P]))["logits"][0, -1:].float()
    torch.testing.assert_close(logits, want, atol=2e-4, rtol=2e-4)  # the last real row, not the pad row
    assert eng._pos_cpu == 45 and int(eng.pos) == 45 and eng.prompt_kv_len == 0
    assert eng.token_history[:45].tolist() == P and not eng.token_history[45:52].any()
    assert bool((eng.attn_bias[..., :45] == 0).all()) and bool(torch.isinf(eng.attn_bias[..., 45:]).all())
    # the pad row's logits are something else: block_last is what picks the real one
    eng.prefill(torch.tensor([P[:n]]))
    eng.block_last.fill_(7)
    eng.pos.fill_(44)
    pad = eng._forward_block(torch.tensor([P[44:] + [P[-1]] * 7]))[0, -1:].float()
    assert not torch.allclose(pad, want, atol=1e-3)
    with pytest.raises(ValueError):
        eng.extend_padded(torch.tensor([P]), 0)


@torch.inference_mode()
def test_prompt_kv_only_reuses_only_prefill_written_kv(model: KotodamaModel) -> None:
    eng = _engine(model, buckets=(16, 8))
    P = _ids(40, 3)
    eng.prefill(torch.tensor([P[:20]]))  # at least min_cached_prefix (16) tokens, so a hit is used
    for t in P[20:25]:  # decode steps write positions 20..24
        eng.step_logits(torch.tensor([[t]]))
    assert eng.prompt_kv_len == 20 and eng._pos_cpu == 25
    _, info = eng.prefill_cached(torch.tensor([P]), prompt_kv_only=True)
    assert info["common_prefix"] == 20
    eng.prefill(torch.tensor([P[:20]]))
    for t in P[20:25]:
        eng.step_logits(torch.tensor([[t]]))
    _, info = eng.prefill_cached(torch.tensor([P]))
    assert info["common_prefix"] == 25  # without it, the decode steps' KV is reused too
