"""K/V operations on a session's cache, as pure functions: evict a range and renumber the survivors, export rows, and
import rows at a position, each moved key re-rotated through the model's own RoPE table (kotodama.serve.kv_sessions
holds the sessions these run on).

Layout: K and V of one or more layers, ``[layers, n_kv_heads, positions, head_dim]`` (a single layer is
``[1, ...]``; any float dtype, bf16 on the engine), keys stored after QK-norm and RoPE, values without position. A
session cache is compact: the key at index i is rotated to position i. ``cos``/``sin`` are the model's own RoPE
tables (``[max_positions, head_dim]``, the engine's ``rope_cos_t``/``rope_sin_t``), never rebuilt from the inverse
frequencies: on CUDA a rebuilt table differs from the model's CPU-built one by about 2e-3 at 4K positions.

Rounding: a moved key is cast to fp32, un-rotated at its old position (cos, -sin), rotated at its new one (cos,
sin), both through the table in fp32, and cast back to the key's dtype once. Keys that do not move, and all values,
stay bit-identical. A session row's lane names this file's digest and this convention (``ROTATION``), since both
decide its bytes.
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor

# The rounding convention above, as a session row's capture identity names it.
ROTATION = "fp32-once: unrotate(cos,-sin)@old, rotate(cos,sin)@new, cast back once; unmoved rows bit-identical"


def _rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """kotodama.model.llama.apply_rope for a 2-D table (SDPA layout), in x's dtype."""
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos.unsqueeze(0).unsqueeze(0) + rotated * sin.unsqueeze(0).unsqueeze(0)


def move_keys(k: Tensor, old_pos: Tensor, new_pos: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Keys ``k`` (``[L, h, n, d]``) rotated to ``old_pos`` (n,) moved to ``new_pos`` (n,). Rows with old == new are
    returned bit-identical; the others go through fp32 once (un-rotate, rotate) and back to k's dtype."""
    if k.dim() != 4 or k.shape[-2] != old_pos.numel() or old_pos.shape != new_pos.shape:
        raise ValueError(f"keys {tuple(k.shape)} vs positions {tuple(old_pos.shape)} -> {tuple(new_pos.shape)}")
    o, n = old_pos.to(k.device).long(), new_pos.to(k.device).long()
    both = torch.cat([o, n])
    if both.numel() and (int(both.max()) >= cos.shape[0] or int(both.min()) < 0):
        raise ValueError(f"positions outside the {cos.shape[0]}-row RoPE table")
    moved = (o != n).nonzero().flatten()
    if moved.numel() == 0:
        return k
    c, s = cos.to(k.device).float(), sin.to(k.device).float()
    out = k.clone()
    x = k[..., moved, :].float()
    x = _rope(x, c[o[moved]], -s[o[moved]])
    out[..., moved, :] = _rope(x, c[n[moved]], s[n[moved]]).to(k.dtype)
    return out


def evict_rerotate(k: Tensor, v: Tensor, drop: range, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Drop positions ``drop`` (a contiguous range, step 1) from a compact session cache; survivors renumber to
    0..n-len(drop)-1, so everything after the range moves down by len(drop) and is re-rotated. Rows before it are
    untouched (keep an attention sink, such as a header, before any range you drop)."""
    n = k.shape[-2]
    if v.shape[-2] != n:
        raise ValueError(f"k has {n} positions, v {v.shape[-2]}")
    if drop.step != 1 or not 0 <= drop.start < drop.stop <= n:
        raise ValueError(f"bad drop range {drop} for {n} positions")
    a, b = drop.start, drop.stop
    tail_old = torch.arange(b, n, device=k.device)
    tail_k = move_keys(k[..., b:, :], tail_old, tail_old - (b - a), cos, sin)
    return (torch.cat([k[..., :a, :], tail_k], dim=-2).contiguous(),
            torch.cat([v[..., :a, :], v[..., b:, :]], dim=-2).contiguous())


def export_rows(k: Tensor, v: Tensor, rows: range) -> tuple[Tensor, Tensor, int]:
    """Copies of positions ``rows`` and the position their first key is rotated to (``rows.start`` in a compact
    session). Archive all three: an import needs to know where the keys were rotated to."""
    n = k.shape[-2]
    if rows.step != 1 or not 0 <= rows.start < rows.stop <= n:
        raise ValueError(f"bad row range {rows} for {n} positions")
    return k[..., rows.start:rows.stop, :].clone(), v[..., rows.start:rows.stop, :].clone(), rows.start


def import_rows(k: Tensor, v: Tensor, rows_k: Tensor, rows_v: Tensor, at: int, rows_from: int,
                cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Insert rows before position ``at`` of a compact session cache (``at == n`` appends). The rows' keys are
    rotated now to ``rows_from..rows_from+m-1`` (as exported); they move, as a rigid block, to ``at..at+m-1``, and
    every row at or after ``at`` moves up by m. Nothing is recomputed: each row keeps what it attended to when it
    was made."""
    n, m = k.shape[-2], rows_k.shape[-2]
    if v.shape[-2] != n or rows_v.shape[-2] != m or m == 0:
        raise ValueError(f"shapes: k {tuple(k.shape)}, v {tuple(v.shape)}, rows {tuple(rows_k.shape)}")
    if rows_k.shape[:2] != k.shape[:2] or rows_k.shape[-1] != k.shape[-1]:
        raise ValueError(f"rows {tuple(rows_k.shape)} don't fit the cache {tuple(k.shape)}")
    if not 0 <= at <= n:
        raise ValueError(f"at={at} outside a cache of {n} positions")
    if n + m > cos.shape[0]:
        raise ValueError(f"{n + m} positions exceed the {cos.shape[0]}-row RoPE table")
    dev = k.device
    span_k = move_keys(rows_k.to(dev), torch.arange(rows_from, rows_from + m, device=dev),
                       torch.arange(at, at + m, device=dev), cos, sin)
    tail_old = torch.arange(at, n, device=dev)
    tail_k = move_keys(k[..., at:, :], tail_old, tail_old + m, cos, sin)
    return (torch.cat([k[..., :at, :], span_k.to(k.dtype), tail_k], dim=-2).contiguous(),
            torch.cat([v[..., :at, :], rows_v.to(dev, v.dtype), v[..., at:, :]], dim=-2).contiguous())
