"""The calibration append: the head is the base's bytes, a thin tail stays zero, the seam is
read against the new fit's own noise, and the append's replay is the fit's replay byte for byte."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import pytest
import torch

from kotodama.eval import anamnesis_calibrate as cal
from kotodama.eval.calibration_append import CalibrationAppendError, means_from, seam_report, splice
from kotodama.model.capture import cached_forward, hidden_rows
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig


def _tables(depth: int = 3, width: int = 12, hidden: int = 4, reach: int = 5):
    rng = np.random.default_rng(0)
    base = np.zeros((depth, width, hidden), np.float32)
    base[:, :reach] = rng.standard_normal((depth, reach, hidden)).astype(np.float32)
    counts = np.zeros((depth, width), np.int64)
    counts[:, :reach] = 9
    return base, counts


def test_means_leave_a_thin_position_at_zero():
    sums = np.ones((2, 4, 3)) * np.array([10.0, 20.0, 30.0, 40.0])[None, :, None]
    counts = np.array([10, 10, 3, 0])
    m = means_from(sums, counts, floor=5)
    assert m.dtype == np.float32
    assert np.allclose(m[:, :2], [[1.0], [2.0]]) and not m[:, 2:].any()


def test_the_splice_keeps_the_base_head_byte_for_byte():
    base, base_counts = _tables()
    new = np.full((3, 20, 4), 7.0, np.float32)
    new_counts = np.full(20, 200, np.int64)
    new_counts[18:] = 40                                   # a thin tail end
    means, counts = splice(base, base_counts, 5, new, new_counts, floor=50)
    assert means[:, :5].tobytes() == base[:, :5].tobytes()
    assert np.array_equal(counts[:, :5], base_counts[:, :5])
    assert (means[:, 5:18] == 7.0).all() and not means[:, 18:].any()
    assert (counts[:, 5:18] == 200).all() and not counts[:, 18:].any()


def test_the_splice_refuses_a_table_that_reaches_no_further_or_does_not_match():
    base, base_counts = _tables()
    with pytest.raises(CalibrationAppendError, match="no further"):
        splice(base, base_counts, 5, np.zeros((3, 5, 4), np.float32), np.ones(5, np.int64), 1)
    with pytest.raises(CalibrationAppendError, match="layers or width"):
        splice(base, base_counts, 5, np.zeros((2, 20, 4), np.float32), np.ones(20, np.int64), 1)
    with pytest.raises(CalibrationAppendError, match="dtype"):
        splice(base, base_counts, 5, np.zeros((3, 20, 4), np.float64), np.ones(20, np.int64), 1)


def test_the_seam_reads_against_the_new_fits_own_noise():
    rng = np.random.default_rng(1)
    truth = rng.standard_normal((2, 40, 8)).astype(np.float32) * 10
    even = truth + rng.standard_normal(truth.shape).astype(np.float32)
    odd = truth + rng.standard_normal(truth.shape).astype(np.float32)
    new = (even + odd) / 2
    base = truth.copy()
    report = seam_report(base, 20, new, even, odd, new, {}, 0, span=16)
    assert report["rows"] == [4, 19]
    assert 0.3 < report["ratio_median"] < 3.0                 # base and new differ by noise only
    shifted = seam_report(base + 5.0, 20, new, even, odd, new, {}, 0, span=16)
    assert shifted["ratio_median"] > 3.0                      # a real offset reads as one


def test_a_step_in_the_corrected_norm_across_the_reach_is_reported():
    means = np.zeros((1, 40, 4), np.float32)
    flat = np.ones((1, 40, 4), np.float32) + np.random.default_rng(2).standard_normal((1, 40, 4)).astype(np.float32) * 0.01
    stepped = flat.copy()
    stepped[:, 20:] += 3.0
    rep = seam_report(means, 20, means, means, means, means, {0: flat, 1: stepped}, 0, span=16)
    by = {s["stream"]: s["max_abs_step_in_sd"] for s in rep["corrected_norm_steps"]}
    assert by[1] > 20 * by[0]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(20260925)
    cfg = KotodamaModelConfig(hidden_size=64, num_layers=6, num_attention_heads=4, num_kv_heads=2,
                              head_dim=16, intermediate_size=128, vocab_size=199,
                              max_position_embeddings=96, attn_impl="sdpa", attn_res=True,
                              attn_res_boundaries=[0, 1, 3], use_liger=False)
    return KotodamaModel(cfg).eval()


def test_the_append_replay_is_the_fits_replay_byte_for_byte(model):
    torch.manual_seed(7)
    ids = torch.randint(3, 199, (1, 40), dtype=torch.long)
    with torch.no_grad():
        _, cap, _ = cached_forward(model, ids, past=None, position_offset=0, light=False,
                                   capture_hidden=True)
        ref = torch.stack([h[0] for h in hidden_rows(cap, 6)])
    got = cal._replay_states(model, ids[0].tolist(), 6, "cpu")
    assert got.dtype == ref.dtype and torch.equal(got, ref)


def test_append_options_parse():
    a = cal.parser().parse_args(["--checkpoint", "c.pt", "--out_dir", "o", "--append_from", "b",
                                 "--max_new_tokens", "4096", "--suppress_eos", "--shard", "2/8"])
    assert str(a.append_from) == "b" and a.shard == "2/8" and a.count_floor == 50
