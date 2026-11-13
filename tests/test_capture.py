"""CPU faithfulness gates for kotodama.model.capture at raw level: the cached walker
is bit-exact against the uncached one, prefill+continue matches the full forward, a
truncated prefix changes the reading, and the replay contract has the banking shapes."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
# The replay-contract test needs anamnesis; point ANAMNESIS_PL_PATH at a checkout
# of its pipeline/ to run it (skipped otherwise).
if os.environ.get("ANAMNESIS_PL_PATH") and os.environ["ANAMNESIS_PL_PATH"] not in sys.path:
    sys.path.insert(0, os.environ["ANAMNESIS_PL_PATH"])

import pytest
import torch

from kotodama.model.llama import KotodamaModel, KotodamaModelConfig

from kotodama.model.capture import (
    instrumented_forward,
    KotoCaptureError,
    cached_forward,
    hidden_rows,
    replay_extract_cached_koto,
    replay_extract_koto,
)

BOUNDARIES = [0, 1, 3]  # 3 committed blocks over 6 layers, DD-shape


@pytest.fixture(scope="module", params=[False, True], ids=["plain-query", "unit-norm-query"])
def model(request):
    """The plain model, and one with unit-norm routing queries (--attnres_query_norm) at nonzero temperatures, so the
    capture walkers read the parametrized query on non-uniform routes."""
    torch.manual_seed(20260925)
    cfg = KotodamaModelConfig(
        hidden_size=64,
        num_layers=6,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=199,
        max_position_embeddings=96,
        attn_impl="sdpa",
        attn_res=True,
        attn_res_boundaries=BOUNDARIES,
        use_liger=False,
        attn_res_gain="folded" if request.param else "learned",
        attn_res_query_norm=request.param,
    )
    m = KotodamaModel(cfg)
    if request.param:
        with torch.no_grad():
            for t in m.routing_temperatures():
                t.copy_(torch.randn(()) * 3)
    m.eval()
    return m


@pytest.fixture(scope="module")
def ids():
    torch.manual_seed(7)
    return torch.randint(3, 199, (1, 40), dtype=torch.long)


def _same(a: torch.Tensor, b: torch.Tensor, tol: float, what: str) -> None:
    d = (a.float() - b.float()).abs().max().item()
    assert d <= tol, f"{what}: max abs diff {d} > {tol}"


def test_cached_walker_is_the_instrumented_walker(model, ids):
    ref_logits, ref_cap = instrumented_forward(model, ids)
    logits, cap, cache = cached_forward(model, ids, past=None, light=False)
    _same(logits, ref_logits, 0.0, "logits")
    for key in ("keys", "values", "queries", "attn", "gate",
                "residual_partial", "committed"):
        assert len(cap[key]) == len(ref_cap[key]), key
        for i, (a, b) in enumerate(zip(cap[key], ref_cap[key])):
            _same(a, b, 0.0, f"{key}[{i}]")
    assert len(cap["routing"]) == len(ref_cap["routing"])
    for (t1, l1, w1), (t2, l2, w2) in zip(cap["routing"], ref_cap["routing"]):
        assert (t1, l1) == (t2, l2)
        _same(w1, w2, 0.0, f"routing {t1}@{l1}")
    # the additions the smoke does not have
    assert len(cap["attn_out"]) == model.config.num_layers
    assert cache.length == ids.shape[1]


def test_eager_prefill_plus_continuation_is_pure_windowing(model, ids):
    P = 25
    full_logits, full_cap, _ = cached_forward(model, ids, past=None, light=False)
    _, _, cache = cached_forward(model, ids[:, :P], past=None, light=False)
    cont_logits, cont_cap, _ = cached_forward(
        model, ids[:, P:], past=cache, position_offset=P, light=False
    )
    n = ids.shape[1] - P
    _same(cont_logits, full_logits[:, P:], 1e-5, "continuation logits")
    for key in ("keys", "values", "queries", "gate", "attn_out"):
        for l in range(model.config.num_layers):
            a, b = cont_cap[key][l], full_cap[key][l]
            if a.ndim == 4:  # (1, heads, n|S, hd)
                _same(a, b[:, :, P:, :], 1e-5, f"{key} L{l}")
            else:  # (1, n|S, width)
                _same(a, b[:, P:, :], 1e-5, f"{key} L{l}")
    for l in range(model.config.num_layers):
        a = cont_cap["attn"][l]  # (1, nh, n, P+n)
        b = full_cap["attn"][l]  # (1, nh, S, S)
        for i in range(n):
            _same(a[0, :, i, : P + i + 1], b[0, :, P + i, : P + i + 1],
                  1e-5, f"attn row L{l} i{i}")
    # hidden-state convention rows agree at the continuation positions
    hr_cont = hidden_rows(cont_cap, model.config.num_layers)
    hr_full = hidden_rows(full_cap, model.config.num_layers)
    for l, (a, b) in enumerate(zip(hr_cont, hr_full)):
        _same(a, b[:, P:, :], 1e-5, f"hidden row {l}")


def test_parity_prefill_matches_eager_prefill_exactly(model, ids):
    """The production prefill (eager_nocapture) must put the cache on the
    SAME fp trajectory as the eager capture path — the substrate of the cached == full gate.
    Also asserts it banks nothing (that is its point: parity without the
    O(S²) memory)."""
    P = 25
    _, cap_e, cache_e = cached_forward(model, ids[:, :P], past=None, light=False)
    logits_p, cap_p, cache_p = cached_forward(
        model, ids[:, :P], past=None, light=False,
        capture_hidden=False, eager_nocapture=True,
    )
    for (ke, ve), (kp, vp) in zip(cache_e.layers, cache_p.layers):
        _same(ke, kp, 0.0, "cache k")
        _same(ve, vp, 0.0, "cache v")
    assert not cap_p["keys"] and not cap_p["attn"] and not cap_p["routing"]
    assert not cap_p["residual_partial"]
    # and the continuation on a parity cache equals the full eager forward
    full_logits, _, _ = cached_forward(model, ids, past=None, light=False)
    cont_logits, _, _ = cached_forward(
        model, ids[:, P:], past=cache_p, position_offset=P, light=False
    )
    _same(cont_logits, full_logits[:, P:], 1e-5, "parity-prefill continuation")


def test_light_prefill_delta_is_fp_dust(model, ids):
    P = 25
    full_logits, _, _ = cached_forward(model, ids, past=None, light=False)
    _, _, cache = cached_forward(
        model, ids[:, :P], past=None, light=True, capture_hidden=False
    )
    cont_logits, _, _ = cached_forward(
        model, ids[:, P:], past=cache, position_offset=P, light=False
    )
    _same(cont_logits, full_logits[:, P:], 1e-3, "light-prefill continuation logits")
    assert torch.equal(
        cont_logits.argmax(-1), full_logits[:, P:].argmax(-1)
    ), "light prefill changed a top-1 token — that is not dust"


def test_truncated_prefix_changes_the_reading(model, ids):
    P = 25
    full_logits, _, _ = cached_forward(model, ids, past=None, light=False)
    _, _, short = cached_forward(
        model, ids[:, P - 8 : P], past=None, light=False
    )  # 8 tokens rotated at positions 0..7 — a corrupted prefix
    cont_logits, _, _ = cached_forward(
        model, ids[:, P:], past=short, position_offset=P, light=False
    )
    d = (cont_logits.float() - full_logits[:, P:].float()).abs().max().item()
    assert d > 1e-2, (
        f"truncated prefix moved logits by only {d} — the cached path is not "
        "actually reading its cache, which is the failure this gate exists to catch"
    )


def test_offset_below_cache_length_refuses(model, ids):
    _, _, cache = cached_forward(model, ids[:, :20], past=None, light=True,
                                 capture_hidden=False)
    with pytest.raises(KotoCaptureError):
        cached_forward(model, ids[:, 20:], past=cache, position_offset=10)


def test_replay_functions_share_the_anamnesis_contract(model, ids):
    pytest.importorskip("anamnesis")
    P = 25
    n = ids.shape[1] - P
    t = n - 1
    L = model.config.num_layers

    raw_full = replay_extract_koto(model, ids[0].tolist(), P)
    _, _, cache = cached_forward(model, ids[:, :P], past=None, light=False)
    raw_cached = replay_extract_cached_koto(model, cache, ids[:, P:], P)

    for raw in (raw_full, raw_cached):
        assert len(raw.hidden_states) == t
        assert raw.hidden_states[0].shape == (L + 1, model.config.hidden_size)
        assert len(raw.attentions) == t
        assert raw.attentions[0].shape[0] == L
        assert raw.attentions[0].shape[2] == P + 1
        assert raw.attentions[t - 1].shape[2] == P + t
        assert len(raw.logits) == t
        assert list(raw.chosen_token_ids.astype(int)) == ids[0, P + 1 :].tolist()
        assert raw.prompt_length == P
        assert len(raw.attn_res_committed) == len(BOUNDARIES)
        assert raw.attn_res_committed[0].shape == (t, model.config.hidden_size)
        tags = [tag for tag, _, _ in raw.attn_res_routing]
        assert tags[-1] == "final"
        assert raw.attn_res_routing[-1][2].shape[0] == t

    # cached == full at raw level: eager prefill makes cached == full, field by field
    import numpy as np

    for i in range(t):
        assert np.allclose(raw_cached.hidden_states[i], raw_full.hidden_states[i],
                           atol=1e-5), f"hidden step {i}"
        assert np.allclose(raw_cached.attentions[i], raw_full.attentions[i],
                           atol=1e-5), f"attn step {i}"
        assert np.allclose(raw_cached.logits[i], raw_full.logits[i], atol=1e-4), \
            f"logits step {i}"
    for l in range(L):
        for i in range(t):
            assert np.allclose(raw_cached.pre_rope_keys[l][i],
                               raw_full.pre_rope_keys[l][i], atol=1e-5)
            assert np.allclose(raw_cached.gate_activations[l][i],
                               raw_full.gate_activations[l][i], atol=1e-5)
    for (t1, l1, w1), (t2, l2, w2) in zip(raw_cached.attn_res_routing,
                                          raw_full.attn_res_routing):
        assert (t1, l1) == (t2, l2)
        assert np.allclose(w1, w2, atol=1e-5)


def test_generate_light_hf_layout(model):
    from kotodama.model.capture import generate_light

    torch.manual_seed(11)
    prompt = torch.randint(3, 199, (2, 12), dtype=torch.long)
    out = generate_light(model, prompt, max_new_tokens=5, do_sample=True,
                         temperature=0.9, top_p=0.95, eos_token_id=None)
    L = model.config.num_layers
    assert out.sequences.shape == (2, 17)
    assert len(out.hidden_states) == 6  # prefill + 5 steps
    assert len(out.hidden_states[0]) == L + 1
    assert out.hidden_states[0][0].shape == (2, 12, 64)
    for step in out.hidden_states[1:]:
        assert len(step) == L + 1
        assert step[0].shape == (2, 1, 64)
    # determinism under the caller's seed, as calibrate relies on
    torch.manual_seed(11)
    prompt2 = torch.randint(3, 199, (2, 12), dtype=torch.long)
    out2 = generate_light(model, prompt2, max_new_tokens=5, do_sample=True,
                          temperature=0.9, top_p=0.95, eos_token_id=None)
    assert torch.equal(out.sequences, out2.sequences)


def test_rope_ceiling_refuses(model):
    long_ids = torch.randint(3, 199, (1, 97), dtype=torch.long)
    with pytest.raises(KotoCaptureError):
        cached_forward(model, long_ids, past=None, light=True, capture_hidden=False)
