"""Unified model loading for evaluation and analysis.

One loader for every eval entry point (lm-eval, Uncheatable, induction, NCA gates, post-training): a ``.pt`` /
``.pt.zst`` path, a ``kotodama.ckpt`` registry name, a published-weights directory or ``hf:REPO[@REVISION]``. It also
decides the AttnRes geometry, so callers do not restate boundaries:

  1. ``attn_res_config`` fields the caller states win (``attn_res``, ``attn_res_boundaries``, ``attn_res_n_blocks``);
  2. ``attn_res`` not stated: read from the weights (AttnRes query keys present or not);
  3. AttnRes on with no boundaries stated: the published weights' ``config.json``, else the preset's trained
     boundaries (``kotodama.presets.DEFAULT_BOUNDARIES``), else the legacy 7 equal blocks, with a warning.

Published weights whose ``config.json`` names a different shape than ``config_section`` are refused.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import yaml

from kotodama import ckpt, presets
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig

logger = logging.getLogger(__name__)


def load_model_config(
    config_path: Path | str | None = None,
    section: str = "proxy",
) -> dict[str, Any]:
    """Model architecture kwargs for KotodamaModelConfig.

    With no ``config_path``, ``section`` names a shape in ``kotodama.presets``
    (aliases like "model" -> "3b" accepted). A YAML path is still honoured for
    one-off shapes that are not presets.

    Raises:
        FileNotFoundError: If a given config file doesn't exist.
        KeyError: If the section / preset is unknown.
    """
    if config_path is None:
        return presets.shape(section)
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with open(config_path) as f:
        full_config = yaml.safe_load(f)

    if section not in full_config:
        available = list(full_config.keys())
        raise KeyError(
            f"Section '{section}' not found in {config_path}. "
            f"Available: {available}"
        )

    return dict(full_config[section])


def attn_res_geometry(stated: dict[str, Any] | None, state: dict[str, Any], published: dict[str, Any],
                      preset: str | None) -> dict[str, Any]:
    """The AttnRes fields for a model: what the caller stated, then the weights, the published config, the preset's
    trained boundaries (``preset`` None: a YAML shape, no preset lookup), the legacy 7 blocks (module docstring)."""
    out = dict(stated or {})
    if "attn_res" not in out:
        out["attn_res"] = any("attn_res_query" in k for k in state)
    if out["attn_res"] and "attn_res_boundaries" not in out and "attn_res_n_blocks" not in out:
        bounds = published.get("attn_res_boundaries")
        source = "the published config"
        if bounds is None and preset is not None:
            bounds, source = presets.DEFAULT_BOUNDARIES.get(preset), f"preset {preset!r}"
        if bounds is not None:
            out["attn_res_boundaries"] = list(bounds)
            logger.info("AttnRes boundaries from %s: %s", source, list(bounds))
        else:
            out["attn_res_n_blocks"] = 7
            logger.warning("AttnRes on with no boundaries stated, published or preset: 7 equal blocks. "
                           "Pass the boundaries the model was trained with.")
    logger.info("AttnRes config: %s", out)
    return out


def load_model(
    checkpoint_path: Path | str,
    config_path: Path | str | None = None,
    config_section: str = "proxy",
    attn_res_config: dict[str, Any] | None = None,
    device: str = "cuda:0",
    meta: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> KotodamaModel:
    """Load a model from a checkpoint file.

    Handles:
    - torch.compile ``_orig_mod.`` prefix stripping
    - DDP ``module.`` prefix stripping
    - AttnRes configuration (explicit or auto-detected)
    - State dict wrapped in {"model": ...} or bare

    AttnRes geometry: :func:`attn_res_geometry` (stated fields, then the weights, the published config, the
    preset's trained boundaries).

    Args:
        checkpoint_path: A .pt/.pt.zst path, a kotodama.ckpt registry name, a published-weights directory or
            ``hf:REPO[@REVISION]``.
        config_path: Optional model YAML; default = kotodama.presets.
        config_section: Preset name (3b, proxy, ...) or YAML section.
        attn_res_config: AttnRes fields the caller states (attn_res, attn_res_n_blocks, attn_res_boundaries);
            any field left out is decided by :func:`attn_res_geometry`.
        device: Target device for the model.
        meta: If given, filled with the checkpoint's bookkeeping (``step``,
            ``tokens_consumed``; None when the file does not carry them).
        overrides: Extra KotodamaModelConfig fields applied last, independent of
            the AttnRes handling (e.g. ``{"rope_fp32": True}`` for the ladder arms).

    Returns:
        Model in eval mode on the specified device.

    Raises:
        FileNotFoundError: If checkpoint or config doesn't exist.
        ValueError: Published weights of a different shape than ``config_section``.
    """
    published: dict[str, Any] = {}
    if ckpt.is_hf_ref(checkpoint_path) or ckpt.is_hf_layout(checkpoint_path):
        name = str(checkpoint_path)
        state = ckpt.load(checkpoint_path)
        published = dict(state.get("config") or {})
    else:
        checkpoint_path = ckpt.decompress_cached(ckpt.resolve(checkpoint_path))
        name = checkpoint_path.name
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    shape = published.get("model_size")
    if shape and config_path is None and presets.resolve(shape) != presets.resolve(config_section):
        raise ValueError(f"{name} holds a {shape!r} model; pass config_section={shape!r} "
                         f"(--config-section {shape}), not {config_section!r}")

    # Load model architecture config
    cfg = load_model_config(config_path, config_section)
    cleaned = ckpt.model_state(state)
    if meta is not None:
        wrapped = isinstance(state, dict) and "model" in state
        meta["step"] = state.get("step") if wrapped else None
        meta["tokens_consumed"] = state.get("tokens_consumed") if wrapped else None

    cfg.update(attn_res_geometry(attn_res_config, cleaned, published,
                                 config_section if config_path is None else None))

    if overrides:
        cfg.update(overrides)
        logger.info("Model config overrides: %s", overrides)

    # Build model — filter out YAML keys that aren't KotodamaModelConfig fields
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(KotodamaModelConfig)}
    cfg = {k: v for k, v in cfg.items() if k in valid_fields}
    model = KotodamaModel(KotodamaModelConfig(**cfg), init_weights=False)  # the checkpoint supplies the weights
    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    # Report key mismatches (filter out expected AttnRes key mismatches)
    real_missing = [k for k in missing if "attn_res" not in k]
    if real_missing:
        logger.warning("Missing keys in checkpoint: %s", real_missing[:10])
    if unexpected:
        logger.warning("Unexpected keys in checkpoint: %s", unexpected[:10])

    model.eval()
    model.to(device)
    if "cuda" in str(device):
        model.bfloat16()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        "Loaded model: %s (%.1fM params, device=%s, attn_res=%s)",
        name,
        param_count / 1e6,
        device,
        cfg.get("attn_res", False),
    )

    return model
