"""LoRA adapters on kotodama models: attach for training, save, and merge into the base weights for serving.

An adapter adds a low-rank update ``(alpha / rank) * B @ A`` to chosen linear projections of chosen unique layers.
On a looped model the core's repeats alias the core's attention and FFN modules, so an adapter on a core layer is one
update applied on every pass. Layers are named by group or index of a unique layer: ``prelude`` (before the loop),
``core`` (the looped block, by its first-pass indices), ``coda`` (after the last repeat), ``all``, or ranges such as
``0-3,9``; a repeat's own index (a second-pass position) is refused, because it names a module the core owns.

An adapter directory holds two files: WEIGHTS_FILE, the A and B matrices under peft's parameter names (the module's
name followed by lora_A or lora_B), and META_FILE, the JSON metadata (rank, alpha, targets, layers, base checkpoint,
step). Attaching for training goes through peft (the ``training`` extra); merging needs only torch and safetensors, so
the server merges an adapter without peft.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

TARGETS: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
WEIGHTS_FILE = "adapter.safetensors"
META_FILE = "adapter.json"
_A, _B = ".lora_A.default.weight", ".lora_B.default.weight"


def layer_groups(config: Any) -> dict[str, list[int]]:
    """Unique-layer indices by group. A model without looped depth is all prelude."""
    n = config.num_layers
    if config.loop_start is None:
        return {"prelude": list(range(n)), "core": [], "coda": []}
    start, length = config.loop_start, config.loop_len
    end = start + length * config.loop_times
    return {"prelude": list(range(start)), "core": list(range(start, start + length)), "coda": list(range(end, n))}


def resolve_layers(spec: str, config: Any) -> list[int]:
    """Unique-layer indices for a layer spec (see the module docstring). Raises ValueError on a repeat's index or an
    index past the last layer."""
    groups = layer_groups(config)
    unique = sorted(i for g in groups.values() for i in g)
    if spec.strip() == "all":
        return unique
    chosen: set[int] = set()
    for part in (p.strip() for p in spec.split(",") if p.strip()):
        if part in groups:
            chosen.update(groups[part])
        elif "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            chosen.update(range(lo, hi + 1))
        else:
            chosen.add(int(part))
    outside = sorted(chosen - set(unique))
    if outside:
        spans = ", ".join(f"{name} {g[0]}-{g[-1]}" for name, g in groups.items() if g)
        raise ValueError(f"layers {outside} are not unique layers of this model ({spans}); "
                         "a loop repeat runs the core's modules")
    return sorted(chosen)


def add_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("LoRA (train an adapter instead of the full model)")
    g.add_argument("--lora_rank", type=int, default=0, help="adapter rank; 0 trains the full model")
    g.add_argument("--lora_alpha", type=float, default=None, help="update scale numerator (default: 2 x rank)")
    g.add_argument("--lora_targets", default=",".join(TARGETS), help="comma list of projections to adapt")
    g.add_argument("--lora_layers", default="all", help="'all', prelude/core/coda, or unique-layer ranges like 0-3,9")
    g.add_argument("--lora_dropout", type=float, default=0.0)


def _target_pattern(targets: list[str], layers: list[int]) -> str:
    return rf"layers\.({'|'.join(map(str, layers))})\.(attn|ffn)\.({'|'.join(targets)})"


def attach(model: torch.nn.Module, rank: int, alpha: float, targets: list[str], layers: list[int],
           dropout: float = 0.0) -> dict[str, torch.nn.Parameter]:
    """Inject a LoRA adapter (B starts at zero, so the model's outputs are unchanged), freeze every other parameter,
    and keep the adapter in fp32. Returns the trainable parameters by name."""
    from peft import LoraConfig, inject_adapter_in_model

    unknown = sorted(set(targets) - set(TARGETS))
    if unknown:
        raise ValueError(f"unknown LoRA targets {unknown}; choose from {list(TARGETS)}")
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=_target_pattern(targets, layers))
    inject_adapter_in_model(cfg, model)
    trainable: dict[str, torch.nn.Parameter] = {}
    for name, param in model.named_parameters():
        is_lora = "lora_" in name
        param.requires_grad_(is_lora)
        if is_lora:
            param.data = param.data.float()
            trainable[name] = param
    if not trainable:
        raise ValueError("no LoRA parameters were created; check the targets and layers")
    return trainable


def save(directory: Path, model: torch.nn.Module, meta: dict[str, Any]) -> Path:
    """Write the adapter's weights (bf16) and ``meta`` into ``directory``; returns the directory."""
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    weights = {n: p.detach().to("cpu", torch.bfloat16).contiguous()
               for n, p in model.named_parameters() if "lora_" in n}
    tmp = directory / (WEIGHTS_FILE + ".tmp")
    save_file(weights, str(tmp))
    os.replace(tmp, directory / WEIGHTS_FILE)
    (directory / META_FILE).write_text(json.dumps(meta, indent=1, default=str))
    logger.info("saved adapter %s", directory)
    return directory


def merge(model: torch.nn.Module, directory: str | Path) -> dict[str, Any]:
    """Add the adapter in ``directory`` into ``model``'s weights in place: W <- W + (alpha / rank) * B @ A, summed in
    fp32 and stored in W's dtype. Returns the adapter's metadata.

    Rounding the sum into a bf16 weight discards the part of the update below that weight's precision, which peft's
    unmerged forward keeps.
    """
    from safetensors.torch import load_file

    directory = Path(directory)
    meta = json.loads((directory / META_FILE).read_text())
    scale = meta["alpha"] / meta["rank"]
    weights = load_file(str(directory / WEIGHTS_FILE))
    modules = dict(model.named_modules())
    merged = 0
    with torch.no_grad():
        for key, a in weights.items():
            if not key.endswith(_A):
                continue
            prefix = key[: -len(_A)]
            if prefix not in modules:
                raise KeyError(f"adapter module {prefix!r} is not in the model")
            w = modules[prefix].weight
            b = weights[prefix + _B].to(w.device, torch.float32)
            w.copy_((w.float() + scale * (b @ a.to(w.device, torch.float32))).to(w.dtype))
            merged += 1
    logger.info("merged adapter %s: %d modules, rank %d, alpha %g", directory, merged, meta["rank"], meta["alpha"])
    return meta
