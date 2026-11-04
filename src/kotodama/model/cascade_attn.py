"""Shared-prefix (cascade) decode attention: attention over a key set split in two parts, each part attended on its
own and the two results merged exactly by log-sum-exp.

Softmax attention over keys K = K1 ∪ K2 is a weighted mean of the attention over each part:
    out = (out1 · e^lse1 + out2 · e^lse2) / (e^lse1 + e^lse2) = out2 + (out1 − out2) · sigmoid(lse1 − lse2)
where out_i and lse_i (the log of the part's softmax denominator) come from attending to K_i alone. Rows that share
a prompt therefore read the prompt's keys once for all of them (part 1: the rows' queries stacked as several query
positions of one sequence) and their own decoded keys apart (part 2: one query per row), and get the attention over
prompt + own tail to within kernel rounding. The idea is FlashInfer's cascade state merge
(https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/cascade.py), written here in torch.

``attend`` is the part kernel: every item's queries attend to that item's first ``lens`` keys and return the output
with its log-sum-exp. On CUDA with flash-attn installed it is flash-attn's KV-cache kernel behind a torch custom op
(``torch.ops.kotodama.kv_attend_lse``, opaque to the compiler, so it captures in a CUDA graph with its lengths read
from device memory); anywhere else it is the fp32 reference below, which the CPU tests pin the merge against.
Layouts follow flash-attn: queries (N, Lq, Hq, D), keys and values (N, S, Hk, D) with Hq a multiple of Hk (grouped
queries: query head h reads key head h // (Hq // Hk)), lengths (N,) int32, log-sum-exp (N, Hq, Lq) fp32 in nats.
"""

from __future__ import annotations

import functools
import logging
import math
from typing import Callable

import torch

logger = logging.getLogger(__name__)

AttendFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def attend_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention of every item's queries over its first ``lens`` keys, in fp32. q (N, Lq, Hq, D), k/v (N, S, Hk, D),
    lens (N,). Returns (out (N, Lq, Hq, D) in q's dtype, lse (N, Hq, Lq) fp32). An item with no keys gets out 0 and
    lse -inf, so a merge gives it no weight."""
    n, _, hq, d = q.shape
    s, hk = k.shape[1], k.shape[2]
    if hq % hk:
        raise ValueError(f"query heads {hq} are not a multiple of key heads {hk}")
    kf = k.float().repeat_interleave(hq // hk, dim=2)  # (N, S, Hq, D)
    vf = v.float().repeat_interleave(hq // hk, dim=2)
    scores = torch.einsum("nqhd,nshd->nhqs", q.float(), kf) / math.sqrt(d)
    valid = torch.arange(s, device=q.device).view(1, s) < lens.view(n, 1).to(q.device)
    scores = scores.masked_fill(~valid.view(n, 1, 1, s), float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)  # (N, Hq, Lq)
    p = torch.exp(scores - torch.where(torch.isinf(lse), torch.zeros_like(lse), lse).unsqueeze(-1))
    out = torch.einsum("nhqs,nshd->nqhd", p, vf)
    return out.to(q.dtype), lse


@torch.library.custom_op("kotodama::kv_attend_lse", mutates_args=())
def _kv_attend_lse(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from flash_attn import flash_attn_with_kvcache

    out, lse = flash_attn_with_kvcache(q.contiguous(), k, v, cache_seqlens=lens, return_softmax_lse=True)
    return out, lse


@_kv_attend_lse.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(q, memory_format=torch.contiguous_format), \
        q.new_empty((q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32)


@torch.library.custom_op("kotodama::prefill_attend", mutates_args=())
def _prefill_attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    from flash_attn import flash_attn_with_kvcache

    # one split: each query row reduces over its keys in one fixed order, whatever number of queries the call holds
    return flash_attn_with_kvcache(q.contiguous(), k, v, cache_seqlens=lens, causal=True, num_splits=1)


@_prefill_attend.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(q, memory_format=torch.contiguous_format)


def prefill_flash(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """Causal attention of a block of S queries over a KV cache (flash-attn's KV-cache kernel, one split): query i
    sits at position lens - S + i and attends keys < its position + 1. q (1, S, Hq, D), k/v (1, T, Hk, D) with a unit
    last stride, lens (1,) int32. A row's output does not depend on how many queries the call holds or where the block
    starts (one split fixes each row's reduction order), so a prompt prefilled in any chunking gets the same bytes."""
    return torch.ops.kotodama.prefill_attend(q, k, v, lens)


def attend_flash(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``attend_reference``'s contract on flash-attn's KV-cache kernel (CUDA, bf16/fp16, ``lens`` int32, keys and
    values with a unit last stride). An item with ``lens`` 0 gets out 0 and lse -inf (the kernel itself writes +inf
    there, which a merge would read as all the weight)."""
    out, lse = torch.ops.kotodama.kv_attend_lse(q, k, v, lens)
    return out, lse.masked_fill((lens == 0).view(-1, 1, 1), float("-inf"))


@functools.lru_cache(maxsize=None)
def flash_available() -> bool:
    """Whether flash-attn's KV-cache kernel imports (it needs a CUDA build of flash-attn)."""
    try:
        from flash_attn import flash_attn_with_kvcache  # noqa: F401
    except Exception as exc:  # any import failure (missing package, ABI mismatch) means "not available"
        logger.info("flash-attn KV-cache kernel unavailable (%r): cascade attention uses the fp32 reference", exc)
        return False
    return True


def attend_for(device: torch.device, dtype: torch.dtype) -> tuple[str, AttendFn]:
    """The part kernel for tensors on ``device`` in ``dtype``: (name, function). flash-attn on CUDA in bf16/fp16 when
    it imports, else the fp32 reference (correct everywhere; on a GPU it reads every key slot in fp32, so it is a
    fallback, not a fast path)."""
    if device.type == "cuda" and dtype in (torch.bfloat16, torch.float16) and flash_available():
        return "flash", attend_flash
    return "reference", attend_reference


def merge(o1: torch.Tensor, lse1: torch.Tensor, o2: torch.Tensor, lse2: torch.Tensor) -> torch.Tensor:
    """Attention over two disjoint key sets from the attention over each: o (..., H, D), lse (..., H). Exact up to
    the rounding of o1 and o2; part 2 must hold at least one key (lse2 finite), part 1 may be empty (lse1 -inf).
    Returns o in o2's dtype."""
    w1 = torch.sigmoid(lse1 - lse2).unsqueeze(-1)  # e^lse1 / (e^lse1 + e^lse2)
    o2f = o2.float()
    return (o2f + (o1.float() - o2f) * w1).to(o2.dtype)


def pack_group_queries(q_rows: torch.Tensor, members: torch.Tensor, n_kv: int) -> torch.Tensor:
    """Group-major queries for the shared part: q_rows (B, H, D) and members (G, m) row indices -> (G, m * r, Hk, D)
    with r = H // Hk, where query position i * r + j of key head c is row members[g, i]'s query head c * r + j. Every
    query head of a group then reads its key head in one pass (multi-head attention over the key heads), instead of
    each of the r query heads sharing a key head reading it again."""
    g, m = members.shape
    h, d = q_rows.shape[1], q_rows.shape[2]
    r = h // n_kv
    qg = q_rows[members]  # (G, m, H, D)
    return qg.view(g, m, n_kv, r, d).permute(0, 1, 3, 2, 4).reshape(g, m * r, n_kv, d)


def unpack_group_outputs(out: torch.Tensor, lse: torch.Tensor, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of ``pack_group_queries`` on the shared part's result: out (G, m * r, Hk, D), lse (G, Hk, m * r) ->
    per group slot (G * m, H, D) and (G * m, H), slot g * m + i holding row members[g, i]."""
    g, mr, n_kv, d = out.shape
    r = mr // m
    o = out.view(g, m, r, n_kv, d).permute(0, 1, 3, 2, 4).reshape(g * m, n_kv * r, d)
    lo = lse.view(g, n_kv, m, r).permute(0, 2, 1, 3).reshape(g * m, n_kv * r)
    return o, lo
