"""build_model_config sizes the RoPE table to the training context: --sequence_length past the preset's
max_position_embeddings grows it (positions reset per document, so a long document in a long window reaches past the
preset's length), and the positions below the preset's length are unchanged."""

from __future__ import annotations

import sys

import torch

from kotodama.model.llama import precompute_rope_frequencies
from kotodama.training import train


def _args(*extra: str):
    old = sys.argv
    try:
        sys.argv = ["train.py", "--model_size", "proxy", *extra]
        return train.parse_args()
    finally:
        sys.argv = old


def test_preset_length_kept_at_or_below_it():
    preset = train.MODEL_CONFIGS["proxy"]["max_position_embeddings"]
    assert train.build_model_config(_args("--sequence_length", "2048")).max_position_embeddings == preset
    assert train.build_model_config(_args("--sequence_length", str(preset))).max_position_embeddings == preset


def test_longer_context_grows_the_table():
    preset = train.MODEL_CONFIGS["proxy"]["max_position_embeddings"]
    cfg = train.build_model_config(_args("--sequence_length", str(4 * preset)))
    assert cfg.max_position_embeddings == 4 * preset


def test_growing_leaves_the_existing_positions_unchanged():
    short_cos, short_sin = precompute_rope_frequencies(128, 4096, 500000.0)
    long_cos, long_sin = precompute_rope_frequencies(128, 32768, 500000.0)
    assert torch.equal(long_cos[:4096], short_cos) and torch.equal(long_sin[:4096], short_sin)


def test_model_flags_still_reach_the_config():
    cfg = train.build_model_config(_args("--attn_res", "--attn_res_n_blocks", "3", "--attnres_res_gain", "folded"))
    assert cfg.attn_res and cfg.attn_res_n_blocks == 3 and cfg.attn_res_gain == "folded" and cfg.qk_norm
