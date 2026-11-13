"""Model shapes, AttnRes block boundaries, and the tokenizer: the one table.

Everything else (rope theta, norm eps, QK-norm, tied embeddings, z-loss) is a
``KotodamaModelConfig`` default. Training, serving, and eval all build their
configs from here, so a new model size is one entry below.
"""

from __future__ import annotations

from typing import Any

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"

# Block AttnRes boundaries (first layer of each block).
# DD-3B: re-derived from the 3B phase-1 geometry; the shipped 3B uses it.
# DD-v1: the proxy-scale (108M) configuration.
DD3B: tuple[int, ...] = (0, 1, 3, 7, 15, 19, 24)
DDV1: tuple[int, ...] = (0, 3, 7, 12, 21, 25)
BOUNDARIES: dict[str, tuple[int, ...]] = {"dd3b": DD3B, "ddv1": DDV1}

_COMMON = dict(vocab_size=49152, max_position_embeddings=4096)

SHAPES: dict[str, dict[str, Any]] = {
    "3b": dict(hidden_size=3072, num_layers=28, num_attention_heads=24,
               num_kv_heads=8, head_dim=128, intermediate_size=8192, **_COMMON),
    # ~7.18B params at vocab 49,152: the Llama-3-8B skeleton; its extra ~1B is
    # purely the 128K vocab.
    "7b": dict(hidden_size=4096, num_layers=32, num_attention_heads=32,
               num_kv_heads=8, head_dim=128, intermediate_size=14336, **_COMMON),
    # The formation-ladder rung: ~1.42B, benched 590K tok/s
    # at mb8 on 8xB200 (14.1h per 30B-token arm). DD-3B boundaries verbatim.
    "1b": dict(hidden_size=2048, num_layers=28, num_attention_heads=16,
               num_kv_heads=8, head_dim=128, intermediate_size=5632, **_COMMON),
    # Depth arm: the 1b's parameter budget (1.4255B vs 1.4222B total, tied) spent on depth —
    # d1536 x 52 layers (1.86x deeper), heads 12 / kv 6 at head_dim 128, FFN 4096 (~8/3 d).
    "1b-deep": dict(hidden_size=1536, num_layers=52, num_attention_heads=12,
                    num_kv_heads=6, head_dim=128, intermediate_size=4096, **_COMMON),
    # Looped depth: 1b-deep's width and 52 EFFECTIVE layers from 28 unique ones — 8 prelude + a 12-layer
    # core run 3x + 8 coda (~0.80B params). Compute/depth-matched to 1b-deep: does serial depth or unique params
    # carry the capacity effect?
    "1b-deep-loop": dict(hidden_size=1536, num_layers=52, num_attention_heads=12,
                         num_kv_heads=6, head_dim=128, intermediate_size=4096,
                         loop_start=8, loop_len=12, loop_times=3, **_COMMON),
    # Loop-aligned: the loop placed INSIDE the 52L boundary fit's long block (CF52_FIT 19→39, where the
    # lookup forms) — 19 prelude + a 10-layer core run 2x (positions 19–38) + 13 coda = 42 unique (~1.17B params).
    "1b-deep-loopfit": dict(hidden_size=1536, num_layers=52, num_attention_heads=12,
                            num_kv_heads=6, head_dim=128, intermediate_size=4096,
                            loop_start=19, loop_len=10, loop_times=2, **_COMMON),
    # kotodama-tsumugi (Shape A) (configs/tsumugi-1b.yaml): 1b-deep's 52 unique layers (~1.42B) with the CF52_FIT
    # long block (19–38) run 2x → 72 effective = 19 prelude + 20-layer core x2 + 13 coda; loop-aligned boundaries
    # TSUMUGI_LOOP.
    "1b-tsumugi": dict(hidden_size=1536, num_layers=72, num_attention_heads=12,
                             num_kv_heads=6, head_dim=128, intermediate_size=4096,
                             loop_start=19, loop_len=20, loop_times=2, **_COMMON),
    # A ~9.2B deep-narrow benchmark shape for multi-GPU throughput (not a trained model): d3584, 80 effective
    # layers from 64 unique ones = 24 prelude + a 16-layer core run 2x + 24 coda; heads 28 / kv 14, FFN 9472.
    "10b-bench-loop": dict(hidden_size=3584, num_layers=80, num_attention_heads=28,
                           num_kv_heads=14, head_dim=128, intermediate_size=9472,
                           loop_start=24, loop_len=16, loop_times=2, **_COMMON),
    # d1024 x 28 (~0.38B with tied embeddings) — between the proxy and the 1b.
    "intermediate": dict(hidden_size=1024, num_layers=28, num_attention_heads=8,
                         num_kv_heads=4, head_dim=128, intermediate_size=2816, **_COMMON),
    # The 108M proxy the ablation matrices run on.
    "proxy": dict(hidden_size=512, num_layers=28, num_attention_heads=4,
                  num_kv_heads=2, head_dim=128, intermediate_size=1408, **_COMMON),
    "smoke": dict(hidden_size=256, num_layers=4, num_attention_heads=4,
                  num_kv_heads=2, head_dim=64, intermediate_size=512,
                  vocab_size=1024, max_position_embeddings=2048),
}

# kotodama-standard (configs/standard-1b.yaml): the 1b rung's shape trained on the tsumugi recipe, with its own
# boundary fit (STANDARD_FIT). The same dict as "1b": one shape, two boundary sets.
SHAPES["1b-standard"] = SHAPES["1b"]

# Older names still found in configs and scripts, and public model names for their shapes.
# "1b-moonshot-loop" is kotodama-tsumugi's pre-release name; checkpoints, published config.json files and older scripts carry it.
ALIASES: dict[str, str] = {"full": "3b", "model": "3b", "8b": "7b", "1b-moonshot-loop": "1b-tsumugi"}

# The boundaries each shape was trained with, where there is a shipped model
# (the 1b ladder runs DD-3B verbatim by design — same 28-layer block geometry).
# 1b-deep: DD-3B's block starts scaled by 52/28 and rounded (same 7-block geometry, stretched over depth).
DD3B_DEEP52: tuple[int, ...] = (0, 2, 6, 13, 28, 35, 45)
# 52L boundaries fit by scripts/eval/fit_boundaries.py with no AttnRes and no decay (robust cuts + the late cut).
CF52_FIT: tuple[int, ...] = (0, 1, 7, 9, 17, 19, 39)
# Shape A's effective-position boundaries: CF52_FIT's prelude cuts + a block at every loop pass (19, 39) + the coda (59).
TSUMUGI_LOOP: tuple[int, ...] = (0, 1, 7, 9, 17, 19, 39, 59)
# 1b-standard's boundaries: scripts/eval/fit_boundaries.py on a no-AttnRes, no-decay counterfactual of the shape to
# 10B tokens (rules_true, settled from 8.5B): the cuts present in >= 90% of the second-half fits (0, 1, 4), the two
# strongest late cuts (20, 23), and the one middle pair that keeps layer 14 in a block of >= 8 layers (10, 12).
STANDARD_FIT: tuple[int, ...] = (0, 1, 4, 10, 12, 20, 23)
DEFAULT_BOUNDARIES: dict[str, tuple[int, ...]] = {"3b": DD3B, "proxy": DDV1,
                                                  "1b": DD3B, "1b-deep": DD3B_DEEP52,
                                                  "1b-deep-loop": DD3B_DEEP52,  # same effective geometry as deep
                                                  "1b-deep-loopfit": CF52_FIT, "1b-tsumugi": TSUMUGI_LOOP,
                                                  "1b-standard": STANDARD_FIT}
# an alias reads the same boundaries as its shape (callers index this table by the name they were given)
DEFAULT_BOUNDARIES.update({a: DEFAULT_BOUNDARIES[k] for a, k in ALIASES.items() if k in DEFAULT_BOUNDARIES})


def resolve(name: str) -> str:
    """Canonical shape name for ``name`` (accepts aliases)."""
    key = ALIASES.get(name, name)
    if key not in SHAPES:
        raise KeyError(f"unknown model size {name!r}; choose from {sorted(SHAPES) + sorted(ALIASES)}")
    return key


def shape(name: str) -> dict[str, Any]:
    """A fresh copy of the shape kwargs for ``name`` (accepts aliases)."""
    return dict(SHAPES[resolve(name)])


def names() -> list[str]:
    """Every accepted size name, canonical first."""
    return list(SHAPES) + list(ALIASES)
