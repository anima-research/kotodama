"""FSDP2 (``fully_shard``) for :class:`~kotodama.model.llama.KotodamaModel`.

Sharding plan:

- **One FSDP unit per unique attention and per unique FFN module.** The AttnRes forward calls each block's
  ``attn_sublayer`` / ``mlp_sublayer`` rather than ``TransformerBlock.forward``, so FSDP's forward hooks belong on
  the modules those sublayers call: ``block.attn`` and ``block.ffn``. Loop repeats alias their core position's
  ``attn`` / ``ffn`` module objects, so wrapping by module identity shards each weight set exactly once. A shared
  unit runs (and all-gathers) once per pass, and its sharded gradient accumulates over the passes.
- **Everything else is the root unit**: embeddings (read directly as the tied LM head), the per-position pre-norms,
  the AttnRes pseudo-queries and their norms, the final norm. These are all-gathered when the root forward starts
  and stay gathered through backward.
- **Parameters are fp32 and are the optimizer's masters.** FSDP2 casts them to bf16 for compute and reduces
  gradients in fp32. ``cast_forward_inputs`` is off: the units receive the same activations and RoPE tables
  (fp32 under ``rope_fp32``) as the unsharded model.

Muon on the sharded matrices is :func:`kotodama.training.muon.orthogonalize_sharded`. Readers of sharded
parameters (telemetry) reduce local shards with :func:`tensor_stats` on every rank.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

MESH_DIM_SHARD = "shard"
MESH_DIM_REPLICATE = "replicate"


def is_dtensor(t: Any) -> bool:
    """True for a DTensor (a sharded parameter, its gradient, or optimizer state laid out like it)."""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # pragma: no cover - torch without distributed
        return False
    return isinstance(t, DTensor)


def shard_rows(n: int, world: int) -> list[int]:
    """Rows of an ``n``-row tensor that each of ``world`` ranks holds under a dim-0 ``Shard`` (DTensor splits with
    ``torch.chunk``: ceil(n / world) rows per rank, the tail rank short, ranks past the end empty)."""
    size = -(-n // world) if n else 0
    return [max(0, min(size, n - r * size)) for r in range(world)]


def shard_group(t: Any) -> tuple[Any, int, int]:
    """(process group, world, rank) of the mesh dimension a DTensor is row-sharded over.

    FSDP2 lays parameters out as ``Shard(0)`` over its mesh (HSDP: ``(Replicate(), Shard(0))``); any other layout
    is refused."""
    from torch.distributed.tensor import Replicate, Shard

    shard_dims = [d for d, pl in enumerate(t.placements) if isinstance(pl, Shard)]
    if (len(shard_dims) != 1 or t.placements[shard_dims[0]].dim != 0
            or any(not isinstance(pl, (Shard, Replicate)) for pl in t.placements)):
        raise ValueError(f"expected a dim-0 Shard (optionally beside Replicate dims); got {t.placements}")
    d = shard_dims[0]
    mesh = t.device_mesh
    return mesh.get_group(mesh_dim=d), mesh.size(d), mesh.get_local_rank(mesh_dim=d)


def replicate_group(t: Any) -> Optional[Any]:
    """The process group across a DTensor's replicas (HSDP's replicate dimension), or None without one."""
    from torch.distributed.tensor import Replicate

    dims = [d for d, pl in enumerate(t.placements) if isinstance(pl, Replicate)]
    if not dims:
        return None
    if len(dims) > 1:
        raise ValueError(f"expected at most one Replicate dim; got {t.placements}")
    return t.device_mesh.get_group(mesh_dim=dims[0])


def tensor_stats(tensors: list[torch.Tensor]) -> list[tuple[float, float, int]]:
    """``(sum, sum of squares, element count)`` of each whole tensor, in float64, with one host sync.

    A DTensor contributes its local rows and the partial sums are all-reduced over its shard group (one collective
    for all of them), so the call is collective whenever any input is a DTensor: every rank of the mesh must make
    it with the same list. Plain tensors are read as they are."""
    import torch.distributed as dist

    if not tensors:
        return []
    rows: list[torch.Tensor] = []
    counts: list[int] = []
    sharded: list[int] = []
    group: Any = None
    for i, t in enumerate(tensors):
        if is_dtensor(t):
            g = shard_group(t)[0]
            if group is not None and g is not group:
                raise ValueError("tensor_stats: DTensors on different shard groups")
            group = g
            sharded.append(i)
            x = t.to_local().detach().double()
        else:
            x = t.detach().double()
        rows.append(torch.stack([x.sum(), x.pow(2).sum()]) if x.numel() else x.new_zeros(2))
        counts.append(t.numel())
    device = rows[0].device
    out = torch.stack([r.to(device) for r in rows])
    if sharded:
        idx = torch.tensor(sharded, device=device)
        part = out.index_select(0, idx)
        dist.all_reduce(part, group=group)
        out.index_copy_(0, idx, part)
    return [(s, ss, n) for (s, ss), n in zip(out.tolist(), counts)]


def fsdp_units(model: nn.Module) -> list[nn.Module]:
    """The modules that get their own FSDP unit: every unique ``attn`` and ``ffn`` module object across
    ``model.layers``, in effective-layer order (loop repeats contribute nothing new)."""
    seen: set[int] = set()
    units: list[nn.Module] = []
    for layer in model.layers:  # type: ignore[attr-defined]
        for mod in (layer.attn, layer.ffn):
            if id(mod) not in seen:
                seen.add(id(mod))
                units.append(mod)
    return units


def fsdp_layer_groups(model: nn.Module, layers_per_group: int = 1) -> list[list[nn.Module]]:
    """Shard groups of ``layers_per_group`` consecutive unique layers: each group is the ``[attn, ffn, ...]`` of
    those layers, gathered and resharded together (fewer hook calls and collectives than one unit per module; the
    lever for deep, narrow shapes, whose fixed per-group cost scales with depth while the work per gather scales
    with width). Groups never mix the looped core with the layers outside it, so a loop pass gathers only core
    weights. Loop repeats contribute nothing new."""
    if layers_per_group < 1:
        raise ValueError(f"layers_per_group must be >= 1, got {layers_per_group}")
    cfg = getattr(model, "config", None)
    looped = ({cfg.layer_source(i) for i in range(cfg.num_layers) if cfg.layer_source(i) != i}
              if cfg is not None else set())
    groups: list[list[nn.Module]] = []
    seen: set[int] = set()
    current: list[nn.Module] = []
    current_layers = 0
    current_core: Optional[bool] = None
    for i, layer in enumerate(model.layers):  # type: ignore[attr-defined]
        a, f = id(layer.attn), id(layer.ffn)
        if (a in seen) != (f in seen):
            raise ValueError(f"layer {i}: attn and ffn are not shared together; cannot group them")
        if a in seen:
            continue
        seen.update((a, f))
        core = i in looped
        if current and (current_layers == layers_per_group or core != current_core):
            groups.append(current)
            current, current_layers = [], 0
        current += [layer.attn, layer.ffn]
        current_layers += 1
        current_core = core
    if current:
        groups.append(current)
    return groups


@torch.no_grad()
def params_to_fp32(model: nn.Module) -> int:
    """Turn every floating-point parameter into fp32 in place (buffers keep their dtype). On a bf16 model this
    gives fp32 values equal to the bf16 ones, which is where the DDP path's fp32 masters start. Returns the count."""
    n = 0
    for p in model.parameters():
        if p.is_floating_point() and p.dtype != torch.float32:
            p.data = p.data.float()
            n += 1
    return n


def build_mesh(world_size: int, replicas: int = 1, device_type: str = "cuda") -> Any:
    """Device mesh for FSDP2: 1-D ``("shard",)`` over every rank, or HSDP's 2-D ``("replicate", "shard")`` with
    ``replicas`` copies of a ``world_size // replicas``-way shard (shard within a node, replicate across)."""
    from torch.distributed.device_mesh import init_device_mesh

    if replicas < 1 or world_size % replicas:
        raise ValueError(f"replicas={replicas} must divide world_size={world_size}")
    if replicas == 1:
        return init_device_mesh(device_type, (world_size,), mesh_dim_names=(MESH_DIM_SHARD,))
    return init_device_mesh(device_type, (replicas, world_size // replicas),
                            mesh_dim_names=(MESH_DIM_REPLICATE, MESH_DIM_SHARD))


def apply_fsdp2(
    model: nn.Module,
    mesh: Any,
    *,
    param_dtype: Optional[torch.dtype] = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    reshard_after_forward: bool = True,
    layers_per_group: int = 0,
) -> list[nn.Module]:
    """Shard ``model`` in place per the module docstring's plan; returns the non-root units.

    Call after every weight is loaded and every parameter's ``requires_grad`` is final, and after
    :func:`params_to_fp32`. ``param_dtype=None`` computes in the parameters' own dtype (tests compare against an
    fp32 reference that way). ``layers_per_group`` >= 1 shards that many consecutive unique layers as one group
    (:func:`fsdp_layer_groups`); 0 = one unit per attn / ffn module."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    for name, p in model.named_parameters():
        if p.is_floating_point() and p.dtype != torch.float32:
            raise ValueError(f"apply_fsdp2: {name} is {p.dtype}; call params_to_fp32 first (sharded params are "
                             "the fp32 masters)")
    mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=False)
    units = fsdp_units(model)
    shard_calls: list[Any] = fsdp_layer_groups(model, layers_per_group) if layers_per_group > 0 else units
    for unit in shard_calls:
        fully_shard(unit, mesh=mesh, mp_policy=mp, reshard_after_forward=reshard_after_forward)
    fully_shard(model, mesh=mesh, mp_policy=mp, reshard_after_forward=reshard_after_forward)
    return units


def compile_fsdp_units(units: list[nn.Module], **compile_kwargs: Any) -> int:
    """torch.compile inside the FSDP units: each unit's ``forward`` is replaced by one compiled function per module
    class, shared by every unit of that class (parameters are graph inputs, so the loop repeats and every layer reuse
    the trace). FSDP's gather/reshard hooks run in ``__call__`` around ``forward``, outside every compiled region;
    the AttnRes routing, pre-norms and residual adds between units run eagerly. Returns the number of classes
    compiled."""
    import types

    compiled: dict[Any, Any] = {}
    for unit in units:
        base = _base_forward(unit)
        if base not in compiled:
            compiled[base] = torch.compile(base, **compile_kwargs)
        unit.forward = types.MethodType(compiled[base], unit)
    return len(compiled)


def _base_forward(unit: nn.Module) -> Any:
    """The unit's own class ``forward`` (unbound), skipping the FSDP subclass that ``fully_shard`` mixes in."""
    return next(c for c in type(unit).__mro__ if "forward" in vars(c) and not c.__name__.startswith("FSDP")).forward


def compile_fsdp_sublayers(model: nn.Module, **compile_kwargs: Any) -> int:
    """torch.compile each AttnRes sublayer whole, under FSDP2: the compiled region is the pre-norm, the attn / ffn
    unit's body and the residual add (the scope of :meth:`KotodamaModel.compile_regions`), and FSDP's gather /
    reshard hooks run around it. Each unique attn / ffn unit gets a method ``kotodama_residual(norm, h, partial,
    ...)`` registered as an FSDP forward method (``register_fsdp_forward_method``), and every layer's
    ``attn_sublayer`` / ``mlp_sublayer`` calls it; the unit's ``forward`` and ``__call__`` are not used on this path.
    One compiled function per sublayer kind and unit class, shared by every layer and loop repeat (parameters are
    graph inputs). The AttnRes routing between sublayers runs as before. Returns the number of compiled functions."""
    import types

    from torch.distributed.fsdp import register_fsdp_forward_method

    compiled: dict[tuple[str, type], Any] = {}

    def _compiled(kind: str, unit: nn.Module) -> Any:
        fwd = _base_forward(unit)
        key = (kind, fwd)
        if key not in compiled:
            if kind == "attn":
                def attn_residual(self: nn.Module, norm: nn.Module, h: torch.Tensor, partial: torch.Tensor,
                                  rope_cos: torch.Tensor, rope_sin: torch.Tensor, mask: Optional[torch.Tensor] = None,
                                  position_ids: Optional[torch.Tensor] = None,
                                  cu_seqlens: Optional[torch.Tensor] = None, max_seqlen: Optional[int] = None,
                                  block_mask: Any = None) -> torch.Tensor:
                    return partial + fwd(self, norm(h), rope_cos, rope_sin, mask, position_ids=position_ids,
                                         cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, block_mask=block_mask)
                compiled[key] = torch.compile(attn_residual, **compile_kwargs)
            else:
                def mlp_residual(self: nn.Module, norm: nn.Module, h: torch.Tensor,
                                 partial: torch.Tensor) -> torch.Tensor:
                    return partial + fwd(self, norm(h))
                compiled[key] = torch.compile(mlp_residual, **compile_kwargs)
        return compiled[key]

    done: set[int] = set()
    for layer in model.layers:  # type: ignore[attr-defined]
        for kind, unit in (("attn", layer.attn), ("mlp", layer.ffn)):
            if id(unit) not in done:
                done.add(id(unit))
                unit.kotodama_residual = types.MethodType(_compiled(kind, unit), unit)
                register_fsdp_forward_method(unit, "kotodama_residual")

    def attn_sublayer(blk: nn.Module, h: torch.Tensor, partial: torch.Tensor, rope_cos: torch.Tensor,
                      rope_sin: torch.Tensor, mask: Optional[torch.Tensor] = None,
                      position_ids: Optional[torch.Tensor] = None, cu_seqlens: Optional[torch.Tensor] = None,
                      max_seqlen: Optional[int] = None, block_mask: Any = None) -> torch.Tensor:
        return blk.attn.kotodama_residual(blk.attn_norm, h, partial, rope_cos, rope_sin, mask, position_ids,
                                          cu_seqlens, max_seqlen, block_mask)

    def mlp_sublayer(blk: nn.Module, h: torch.Tensor, partial: torch.Tensor) -> torch.Tensor:
        return blk.ffn.kotodama_residual(blk.ffn_norm, h, partial)

    for layer in model.layers:  # type: ignore[attr-defined]
        layer.attn_sublayer = types.MethodType(attn_sublayer, layer)
        layer.mlp_sublayer = types.MethodType(mlp_sublayer, layer)
    return len(compiled)


def compile_fsdp_layers(model: nn.Module, **compile_kwargs: Any) -> None:
    """torch.compile each whole AttnRes layer with both of its routes, under FSDP2: the regions of
    :meth:`KotodamaModel.compile_regions` ``(scope="layer")`` (one compiled copy of ``routed_layer`` per
    committed-block count and boundary), each run through a method registered as an FSDP forward method on the
    layer's attn unit. That unit's pre-forward gathers its shard group, which holds the layer's attn and ffn, so it
    needs layer groups (``layers_per_group`` >= 1). Inside the region the units run through their class forwards
    (``attn_fn`` / ``ffn_fn``), so no FSDP hook fires inside a compiled graph. The layer's routing queries and norms
    belong to the root unit, gathered for the whole forward."""
    import types

    from torch.distributed.fsdp import register_fsdp_forward_method

    for i, layer in enumerate(model.layers):  # type: ignore[attr-defined]
        if layer.attn._get_fsdp_state() is not layer.ffn._get_fsdp_state():
            raise ValueError(f"compile_fsdp_layers: layer {i}'s attn and ffn are separate FSDP units; shard by layer "
                             "groups (--fsdp2_layers_per_group >= 1)")
    model.compile_regions(scope="layer", **compile_kwargs)  # type: ignore[attr-defined]
    regions = model._layer_region  # type: ignore[attr-defined]

    def kotodama_layer(unit: nn.Module, region: Any, layer: nn.Module, *args: Any) -> torch.Tensor:
        return region(layer, *args, attn_fn=unit._kotodama_body, ffn_fn=layer.ffn._kotodama_body)

    done: set[int] = set()
    for layer in model.layers:  # type: ignore[attr-defined]
        for unit in (layer.attn, layer.ffn):
            if id(unit) not in done:
                done.add(id(unit))
                unit._kotodama_body = types.MethodType(_base_forward(unit), unit)
        if not hasattr(layer.attn, "kotodama_layer"):
            layer.attn.kotodama_layer = types.MethodType(kotodama_layer, layer.attn)
            register_fsdp_forward_method(layer.attn, "kotodama_layer")

    def fsdp_region(n_committed: int, boundary: bool) -> Any:
        region = regions(n_committed, boundary)

        def run(layer: nn.Module, *args: Any) -> torch.Tensor:
            return layer.attn.kotodama_layer(region, layer, *args)
        return run

    model._layer_region = fsdp_region  # type: ignore[attr-defined]


def is_fsdp2(model: nn.Module) -> bool:
    """True when ``model`` (or the module it wraps) has been through :func:`apply_fsdp2`."""
    from torch.distributed.fsdp import FSDPModule

    inner = getattr(model, "_orig_mod", model)
    return isinstance(inner, FSDPModule)


def full_model_state_dict(model: nn.Module, dtype: Optional[torch.dtype] = None) -> dict[str, torch.Tensor]:
    """The whole model's state dict as plain CPU tensors on rank 0 (empty on other ranks), keyed like the unsharded
    model's ``state_dict()``, loop-repeat aliases included. ``dtype`` casts the floating tensors (``bfloat16``
    gives the DDP path's on-disk format). Collective: every rank must call it."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if dtype is not None:
        sd = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in sd.items()}
    return sd
