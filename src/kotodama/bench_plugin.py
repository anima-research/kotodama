"""kotodama's side of the binding benchmark: the ``koto:`` model loader, and its registration.

The benchmark (the ``binding-bench`` package, kotodama's ``bench`` extra) scores Hugging Face models by itself and
loads every other checkpoint format through a plugin. :func:`load_koto` is the plugin for ``koto:`` specs, and
:func:`register` hands it to the benchmark in this process. ``python -m kotodama.eval.binding`` registers it and then
runs the benchmark's CLI. On a machine where kotodama is on ``PYTHONPATH`` rather than installed, the same loader is
named by ``BINDING_BENCH_LOADERS=koto=kotodama.bench_plugin:load_koto``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from binding_bench.binding.models import LoadedModel, ModelSpec


def load_koto(spec: "ModelSpec", device: str) -> "LoadedModel":
    """A kotodama checkpoint as a scorer: logits from the parity-tested capture forward (AttnRes models) or the
    model's own forward (plain residual), with the kotodama tokenizer and ``rope_fp32`` on."""
    import torch
    from binding_bench.binding.models import LoadedModel
    from transformers import AutoTokenizer

    from kotodama.model import capture
    from kotodama.presets import TOKENIZER_NAME
    from kotodama.serve import server

    server._rope_fp32_override = True
    server._attn_res_override = spec.attn_res
    server._attn_res_boundaries_override = spec.bounds
    model, _ = server.load_model(spec.path, device=device, model_size=spec.size)
    model.eval()
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if getattr(model.config, "attn_res", False) and getattr(model.config, "attn_res_energy_steps", 0) <= 0:
        def fwd(ids: torch.Tensor) -> torch.Tensor:
            return capture.instrumented_forward(model, ids, light=True)[0]
    else:   # the capture walker covers the one-softmax AttnRes router only; plain-residual and energy-routed models
        # score through their own forward, which the capture walker is parity-tested against
        def fwd(ids: torch.Tensor) -> torch.Tensor:
            return model(ids)["logits"]
    msl = getattr(model.config, "max_seq_len", None) or getattr(model.config, "max_position_embeddings", None)
    return LoadedModel(spec, fwd, tok, next(model.parameters()).device, msl)


def register() -> None:
    """Make :func:`load_koto` the benchmark's ``koto:`` loader in this process."""
    from binding_bench.binding.models import register_loader

    register_loader("koto", load_koto)
