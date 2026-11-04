"""Static-shape single-stream decode engine for KotodamaModel (AttnRes).

Serving-only fast path. Borrows weight tensors from a loaded KotodamaModel —
llama.py remains the training/reference implementation and the parity oracle.

Design notes
------------
The decode step is written to be torch.compile + CUDA-graph friendly:

  * All persistent state (KV cache, attention bias, position, sampling
    presence) lives in registered nn.Module buffers → static addresses,
    so inductor's cudagraph trees accept the in-place mutations.
  * KV cache: (n_layers, B, n_kv_heads, max_seq_len, head_dim), written in
    place at position `pos` (a CUDA tensor — never a Python int).
  * Attention: SDPA over the FULL cache length with an additive mask that
    opens one position per step. Shape-static.
  * AttnRes routing is FUNCTIONAL: committed block outputs and the running
    partial are plain locals (mirroring llama._forward_attn_res_cached
    exactly, minus the padding/stacking — source counts per routing call
    site are compile-time constants, so every site unrolls into its own
    fused kernel). No buffer mutation → no functionalization blowups.
  * Sampling: on-GPU, matching server.py semantics — presence-based
    repetition penalty over GENERATED tokens only, pure temperature
    (top_k/top_p intentionally unsupported: prod runs 0/0).
  * Steering (optional, enabled at construction): a registered per-layer
    steer_buf (n_layers, hidden) added row-wise to the running `partial`
    after attention at EVERY layer of an enabled engine (zero rows = off).
    set_steer() populates rows per write: the site row + every later block
    entry (block-persistent, the koto primitive — single-site writes die at
    DD-3B boundaries); multiple sites just populate different rows. Decode
    steps only; prefill/extend never inject. A disabled engine traces zero
    new ops. kotodama/serve/steering.py has the API + the GPU gates.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from kotodama.model.gemv_kernels import decode_gemv, use_custom_gemv
from kotodama.model.levers import LeverTables
from kotodama.model.llama import KotodamaModel, apply_rope
from kotodama.model.regional import (StepGraph, block_attend, build_regions, cache_attend, check_norms,
                                     layer_signatures, region_options, run_regions)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerSchedule:
    """Per-layer static schedule: is this layer an AttnRes block boundary?"""

    layer_idx: int
    is_boundary: bool


def compute_steer_points(steer_site: int, boundaries: set[int] | list[int], n_layers: int) -> tuple[int, ...]:
    """Layer indices that receive the block-persistent steering write.

    A single-site residual add is destroyed at the next DD-3B block
    boundary (commit-and-reset + routing recombination attenuate it ~100x, KL
    0.002 vs 0.78 nats persistent), so the koto write primitive re-asserts the
    vector at the first after-attn position of every block AFTER the site's.

    In the sublayer convention of the steering npz keys (``_s{sublayer}``),
    sites are SUBLAYER indices (after-attn of layer L = sublayer 2*L);
    persist points {2*L} + {2*b : boundary b, 2*b > 2*L} reduce in layer terms
    to: the site layer itself + every boundary layer strictly after it.
    """
    if not 0 <= steer_site < n_layers:
        raise ValueError(f"steer_site {steer_site} out of range [0, {n_layers})")
    return (steer_site, *sorted(b for b in set(boundaries) if steer_site < b < n_layers))


class SamplingParams:
    """server.py sampling controls supported by the fast path.

    top_k == 0, top_p in {0.0, 1.0} and min_p == 0.0 mean "disabled" (server.py
    convention). min_p keeps the tokens whose probability (after temperature,
    top-k and top-p) is at least min_p times the most likely token's.
    Truncated sampling (top_k > 0 or 0 < top_p < 1) routes through a separate
    compiled sampler variant; the pure-temperature graph is untouched.

    ``penalty_scope`` says which tokens the repetition penalty counts:
    "generated" (the default) the row's generated tokens only, "row" every
    token of the row, prompt included (the CTRL / HF convention). Either way
    the penalty reads one presence set; "row" seeds it with the prompt ids.
    """

    def __init__(
        self,
        temperature: float = 0.9,
        repetition_penalty: float = 1.2,
        top_k: int = 0,
        top_p: float = 0.0,
        penalty_scope: str = "generated",
        min_p: float = 0.0,
    ) -> None:
        if penalty_scope not in ("generated", "row"):
            raise ValueError(f"penalty_scope must be 'generated' or 'row', got {penalty_scope!r}")
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        if not 0.0 <= top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {top_p}")
        if not 0.0 <= min_p < 1.0:
            raise ValueError(f"min_p must be in [0, 1), got {min_p}")
        self.temperature = float(temperature)
        self.repetition_penalty = float(repetition_penalty)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.penalty_scope = penalty_scope
        self.min_p = float(min_p)

    @property
    def truncated(self) -> bool:
        return self.top_k > 0 or 0.0 < self.top_p < 1.0 or self.min_p > 0.0


def _graph_input(tokens: torch.Tensor) -> torch.Tensor:
    """``tokens`` as a fresh offset-0 tensor with canonical strides, for a compiled graph.

    A (1, n) slice of a (1, S) prompt keeps stride (S, 1) and its storage offset, and ``.contiguous()`` returns it
    as-is (a size-1 row already counts as contiguous). Compiled graphs guard on both, so a new offset recompiles, and
    once they are dynamic, inductor's cudagraph trees record a graph with its own memory pool for each distinct value:
    graph memory then grows with every new prompt length. A contiguous-format clone has one layout per shape.
    """
    return tokens.clone(memory_format=torch.contiguous_format)


def decode_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """F.linear or the custom decode GEMV, decided per weight shape.

    The branch resolves at trace time (decode shapes are static), so the
    compiled graph bakes in one path per call site with zero replay cost.
    """
    if use_custom_gemv(w):
        return decode_gemv(x, w)
    return F.linear(x, w)


class DecodeEngine(nn.Module):
    """Single-stream (B=1) static-shape decode engine.

    Usage:
        engine = DecodeEngine(model, max_seq_len=4096)
        engine.compile_step()                          # optional
        logits = engine.prefill(input_ids)             # eager reference prefill
        tok = engine.sample_first(logits, params)
        while ...:
            tok = engine.step(tok, params)             # static decode step

    Contracts:
      * Weights are BOUND AT CONSTRUCTION: fused QKV/gate-up and rope tables
        are copies. After mutating model weights (checkpoint hot-reload),
        rebuild the engine — a stale engine emits silently wrong logits.
      * step()/sample_first() return the persistent cur_token buffer;
        step_logits() under compile returns CUDA-graph pool memory. Both are
        valid only until the next step — consume or copy immediately.
      * Do not call engine.to()/.half() — buffer lists hold raw tensor refs.
    """

    def __init__(
        self,
        model: KotodamaModel,
        max_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        prefill_backends: Optional[list[SDPBackend]] = None,
        steer_enabled: bool = False,
        levers: int = 0,
        emulate_casts: bool = False,
        block_buckets: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if not model.config.attn_res:
            raise ValueError("DecodeEngine requires an attn_res model")
        if max_seq_len is not None and max_seq_len > model.config.max_position_embeddings:
            raise ValueError(
                f"max_seq_len {max_seq_len} exceeds rope table length "
                f"{model.config.max_position_embeddings}"
            )
        self.model = model
        self.config = model.config
        # compiled regions round intermediates as eager does (regional.region_options), the engine's and every
        # decoder's built on it: signing decoders then decode what unsigned ones do
        self.emulate_casts = bool(emulate_casts)
        self.device = device or next(model.parameters()).device
        self.dtype = dtype
        self.max_seq_len = int(max_seq_len or model.config.max_position_embeddings)
        cfg = model.config

        self.n_layers = cfg.num_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.hidden = cfg.hidden_size
        self.eps = cfg.norm_eps

        # ── Static state (registered buffers → stable addresses for graphs) ──
        # Per-layer KV buffers (NOT one stacked tensor): in-place index_copy_
        # must target a graph input directly. Writing into a slice view of a
        # stacked buffer gets functionalized by inductor into full-buffer
        # materializations (~470 MB of copies per step, measured ~1.3 ms).
        kv_shape = (1, self.n_kv_heads, self.max_seq_len, self.head_dim)
        for li in range(self.n_layers):
            self.register_buffer(f"k_cache_{li}", torch.zeros(kv_shape, device=self.device, dtype=dtype), persistent=False)
            self.register_buffer(f"v_cache_{li}", torch.zeros(kv_shape, device=self.device, dtype=dtype), persistent=False)
        self.k_caches = [getattr(self, f"k_cache_{li}") for li in range(self.n_layers)]
        self.v_caches = [getattr(self, f"v_cache_{li}") for li in range(self.n_layers)]
        self.register_buffer(
            "attn_bias",
            torch.full((1, 1, 1, self.max_seq_len), float("-inf"), device=self.device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer("pos", torch.zeros((1,), device=self.device, dtype=torch.long), persistent=False)
        # The block row whose logits a block forward returns (its last row, unless the chunk is right-padded):
        # read in-graph, set eagerly before each replay.
        self.register_buffer("block_last", torch.zeros((1,), device=self.device, dtype=torch.long), persistent=False)
        self.register_buffer(
            "presence", torch.zeros((cfg.vocab_size,), device=self.device, dtype=torch.bool), persistent=False
        )
        self.register_buffer("cur_token", torch.zeros((1, 1), device=self.device, dtype=torch.long), persistent=False)
        # Token ids whose KV is in the cache: token_history[:pos] <-> k/v[:, :, :pos].
        # Written in-graph by _forward_step (consumed token at its position) and
        # by prefill/extend (prompt ids), so the invariant cannot drift from the
        # cache contents — this is what prefix matching trusts.
        self.register_buffer(
            "token_history",
            torch.zeros((self.max_seq_len,), device=self.device, dtype=torch.long),
            persistent=False,
        )
        # Below this, an extend is not worth the bookkeeping — do a full prefill.
        self.min_cached_prefix = 16
        # Leading cache positions whose KV a prefill or extend wrote (prompt positions). Decode steps write KV
        # with steering injected, so a cache keyed by tokens alone (kotodama.model.prefix_cache) pools only
        # positions below this mark. Host int; prefix-closed: an extend on top of decode-written KV attends to it,
        # so the mark never moves past the first decode-written position.
        self.prompt_kv_len = 0

        # RoPE tables (read-only).
        self.register_buffer("rope_cos_t", model.rope_cos.to(self.device), persistent=False)
        self.register_buffer("rope_sin_t", model.rope_sin.to(self.device), persistent=False)

        # ── Static schedule from the frozen boundary list ───────────────────
        boundary_set = model._attn_res_boundary_set
        self.schedule = [LayerSchedule(i, i in boundary_set) for i in range(self.n_layers)]

        # ── Steering v2 (block-persistent residual write; enabled at construction) ──
        # PER-LAYER buffer (n_layers, hidden). Rows are populated at request
        # time by set_steer(): each write (site_layer, vec, scale) lands
        # scale*vec in the site row AND in every later block-entry row
        # (compute_steer_points — applied at CONTENT time, not trace
        # time), so multi-site stacks are just multiple populated rows.
        # DECODE STEPS ONLY — prefill/extend paths never inject, which gives
        # generated-positions-only semantics for free (re-prefilled prior
        # turns stay unsteered, by design).
        #
        # COMPILE SAFETY: the buffer is created ONCE and only ever mutated
        # in-place (copy_/zero_) — torch.compile(max-autotune)/cudagraph trees
        # specialize on tensor identity, so reassignment would silently detach
        # the graphs from the live buffer. A steering-enabled engine bakes ONE
        # row-add per layer into its graphs UNCONDITIONALLY (zero rows = off),
        # so set_steer() never changes graph structure; a steering-disabled
        # engine (steer_enabled=False, a trace-time constant) traces ZERO new
        # ops — the off-gate stays bitwise by construction, never by an
        # "adding zeros is identity" argument.
        self.steer_enabled = bool(steer_enabled)
        self._steer_active = False
        self._boundaries = tuple(sorted(boundary_set))
        if self.steer_enabled:
            self.register_buffer(
                "steer_buf",
                torch.zeros((self.n_layers, self.hidden), device=self.device, dtype=dtype),
                persistent=False,
            )
        else:
            self.steer_buf = None

        # ── Per-request levers (kotodama.model.levers; enabled at construction with levers > 0) ──
        # Lever tables per site kind, one slot per bound lever, slot 0 the -0.0 identity. lever_slot is the slot
        # this engine's own forwards read (prefill and the single-stream step); the batched decoders keep one slot
        # per row. levers == 0 (a trace-time constant) traces no lever op at all.
        self.levers = None
        if int(levers) > 0:
            self.levers = LeverTables(self.n_layers, self.hidden, int(levers), dtype, self.device,
                                      intermediate=cfg.intermediate_size)
            self.register_buffer("lever_attn", self.levers.tables["attn"], persistent=False)
            self.register_buffer("lever_mlp", self.levers.tables["mlp"], persistent=False)
            self.register_buffer("lever_act_keep", self.levers.tables["act_keep"], persistent=False)
            self.register_buffer("lever_act_add", self.levers.tables["act_add"], persistent=False)
            self.register_buffer("lever_slot", torch.zeros((1,), device=self.device, dtype=torch.long),
                                 persistent=False)
        # Digest of the uniform lever that wrote the prompt KV in the cache (None: unsteered). A cached prefix is
        # reused only by a request of the same lever.
        self._kv_lever: Optional[str] = None
        # Where the positions the last prefill wrote end, and whether the decode steps since then ran steered (a
        # lever slot, or broadcast rows engaged by set_steer): those steps wrote steered KV past that point, which no
        # later prompt may reuse.
        self._prefill_end = 0
        self._gen_steered = False
        # Eager taps (kotodama.model.levers.capture): None outside a capture; compiled graphs never run inside one.
        self._taps = None

        # ── Fused projection weights (decode-only) ──────────────────────────
        # QKV and gate/up merges: fewer, larger GEMVs. Output rows are
        # independent dot products, so results match the unfused projections.
        # Costs ~3.7 GB extra (originals stay live for the reference prefill).
        for li, layer in enumerate(model.layers):
            w_qkv = torch.cat(
                [layer.attn.q_proj.weight, layer.attn.k_proj.weight, layer.attn.v_proj.weight], dim=0
            )
            w_gate_up = torch.cat([layer.ffn.gate_proj.weight, layer.ffn.up_proj.weight], dim=0)
            self.register_buffer(f"w_qkv_{li}", w_qkv.contiguous(), persistent=False)
            self.register_buffer(f"w_gate_up_{li}", w_gate_up.contiguous(), persistent=False)
        self.w_qkvs = [getattr(self, f"w_qkv_{li}") for li in range(self.n_layers)]
        self.w_gate_ups = [getattr(self, f"w_gate_up_{li}") for li in range(self.n_layers)]
        # NOTE: pre-transposing o_proj/down_proj and using x @ W_t is slower
        # (decode 394 → 348 tok/s) — the NN orientation
        # dispatches an even slower path than nvjet's 27%-BW TNN config.
        # The o/down GEMV inefficiency needs a custom kernel, not a layout dodge.
        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim

        # Sampling controls as 0-dim CUDA buffers: Python floats inside the
        # compiled step would become dynamo guards → recompile per value.
        self.register_buffer("inv_temp", torch.tensor(1.0 / 0.9, device=self.device, dtype=torch.float32), persistent=False)
        self.register_buffer("rep_penalty", torch.tensor(1.2, device=self.device, dtype=torch.float32), persistent=False)
        self._cur_temp = 0.9
        self._cur_rp = 1.2
        # Truncation dials for the truncated sampler variant, as buffers so one
        # graph serves every (k, p): disabled values are algebraic no-ops
        # (k = vocab -> the kth threshold is the min logit, masks nothing;
        # p = 2.0 -> cumprob - prob >= 2.0 is never true, masks nothing) —
        # exactly the branches server.py skips. p MUST be 2.0, not 1.0: fp32
        # cumsum saturates to exactly 1.0 partway through the tail (measured:
        # ~11k of 49k tokens masked at p=1.0), and inside a k-only law the
        # renormalized survivor cumsum can saturate the same way.
        self.register_buffer(
            "top_k_eff", torch.tensor(cfg.vocab_size, device=self.device, dtype=torch.long), persistent=False
        )
        self.register_buffer("top_p_val", torch.tensor(2.0, device=self.device, dtype=torch.float32), persistent=False)
        # log(min_p); -inf (disabled) keeps every token
        self.register_buffer("log_min_p", torch.tensor(float("-inf"), device=self.device, dtype=torch.float32),
                             persistent=False)
        self._cur_top_k = 0
        self._cur_top_p = 0.0
        self._cur_min_p = 0.0

        self._compiled_forward = None
        self._compiled_step_sampled = None
        self._compiled_step_truncated = None
        # Block-forward buckets for compiled prefill/extend, descending. One
        # compiled callable — dynamo specializes a graph per static S. The
        # eager reference prefill measured FLAT ~23-26ms at T<=1024 (launch
        # bound); compiled blocks replace it and the 33-40ms MATH extend.
        # Batch-invariant blocks (regional.block_attend: flash-attn's causal KV-cache kernel with one split, on CUDA
        # in bf16/fp16): a position's KV and logits come out with the same bytes in any chunk, so every prompt and
        # extend runs through the blocks (right-padded, never re-forwarding a cached position) and a cached prefix
        # extends to exactly what a cold prefill writes. Their default is ONE bucket width: each width compiles its
        # own kernels, and the row reductions of two widths can sum in different orders. Otherwise blocks attend
        # the whole cache under an additive mask (SDPA), whose arithmetic depends on the chunk's width, and spans
        # above block_span_max take the eager paths (faster there).
        from kotodama.model.cascade_attn import flash_available

        self.block_invariant: bool = (self.device.type == "cuda" and dtype in (torch.bfloat16, torch.float16)
                                      and flash_available())
        default_buckets = (256,) if self.block_invariant else (256, 64, 16)
        self.block_buckets: tuple[int, ...] = tuple(sorted({int(b) for b in (block_buckets or default_buckets)},
                                                           reverse=True))
        if not self.block_buckets or self.block_buckets[-1] < 1:
            raise ValueError(f"block buckets {self.block_buckets} must be positive")
        self.block_span_max: int = self.max_seq_len if self.block_invariant else 768
        # Extends (start > 0) of a span the buckets hold in one chunk take that one chunk, right-aligned (it
        # re-forwards up to bucket - span cached tokens), instead of the greedy cover: a block replay costs about the
        # same at any bucket width here, so one replay beats several. Off keeps the greedy cover; the prefix KV pool
        # (kotodama.model.prefix_cache), whose hits extend short suffixes, turns it on.
        self.extend_single_chunk: bool = False
        self._compiled_block = None
        # CPU mirror of self.pos for bounds checking without device syncs.
        self._pos_cpu = 0
        # Prefill SDPA backend pin (see prefill()); parameterized so tests can
        # run non-bf16 models on MATH. Prod default is the validated set.
        self._prefill_backends = (
            list(prefill_backends)
            if prefill_backends
            else [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        )
        # Multi-token EXTEND needs an explicit mask, which eliminates the fast
        # backends (measured on B200/torch2.x: FLASH rejects any attn_mask,
        # EFFICIENT rejects enable_gqa, CUDNN accepts but re-plans ~300ms per
        # novel (S, L) shape — and extend shapes are novel every turn). MATH
        # costs only ~2-5ms at our suffix sizes and is shape-oblivious.
        self._extend_backends = [SDPBackend.MATH]

        gemv_sites = {
            "qkv": self.w_qkvs[0],
            "o": model.layers[0].attn.o_proj.weight,
            "gate_up": self.w_gate_ups[0],
            "down": model.layers[0].ffn.down_proj.weight,
            "lm_head": model.get_lm_head_weight(),
        }
        custom_sites = [name for name, w in gemv_sites.items() if use_custom_gemv(w)]
        logger.info(
            "DecodeEngine: %d layers, max_seq_len=%d, %d boundaries, kv cache %.0f MB, "
            "custom GEMV: %s",
            self.n_layers, self.max_seq_len, len(boundary_set),
            sum(b.numel() for b in self.k_caches) * 2 * 2 / 1e6,
            custom_sites or "off",
        )
        if self.steer_enabled:
            logger.info(
                "DecodeEngine steering ENABLED: per-layer buffer (%d, %d), "
                "block-persistent rows written by set_steer (decode steps only, "
                "zero rows = off)",
                self.n_layers, self.hidden,
            )

    # ── Linear dispatch: custom Triton GEMV for tuned shapes ────────────────

    def _linear(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return decode_linear(x, w)

    # ── Routing (functional; matches llama._route_static math exactly) ──────

    def _route(self, sources: list[torch.Tensor], query: torch.Tensor, norm_weight: torch.Tensor) -> torch.Tensor:
        """Attention-residual routing over n sources. n is static per call site."""
        qw = query * norm_weight  # (D,)
        stacked = torch.stack(sources, dim=0)                        # (n, 1, 1, D)
        rsqrt = torch.rsqrt(stacked.pow(2).mean(-1) + self.eps)      # (n, 1, 1)
        logits = (stacked * qw).sum(-1) * rsqrt                      # (n, 1, 1)
        weights = F.softmax(logits, dim=0)
        return (weights.unsqueeze(-1) * stacked).sum(0)              # (1, 1, D)

    # ── One attention sublayer with static cache ─────────────────────────────

    def _attn(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        attn = self.model.layers[layer_idx].attn

        qkv = self._linear(x, self.w_qkvs[layer_idx])  # (1, 1, q_dim + 2*kv_dim)
        q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        q = q.view(1, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(1, 1, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(1, 1, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if attn.qk_norm:
            q = attn.q_norm(q)
            k = attn.k_norm(k)

        cos = self.rope_cos_t.index_select(0, self.pos)  # (1, head_dim)
        sin = self.rope_sin_t.index_select(0, self.pos)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k_cache = self.k_caches[layer_idx]
        v_cache = self.v_caches[layer_idx]
        k_cache.index_copy_(2, self.pos, k)
        v_cache.index_copy_(2, self.pos, v)

        out = F.scaled_dot_product_attention(
            q,
            k_cache,
            v_cache,
            attn_mask=self.attn_bias,
            enable_gqa=True,
        )
        out = out.transpose(1, 2).contiguous().view(1, 1, -1)
        return self._linear(out, attn.o_proj.weight)

    # ── Block forward: S tokens against the full cache (compiled prefill/extend) ──

    def _attn_block(
        self,
        layer_idx: int,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """_attn generalized to S queries writing K/V at `positions`."""
        attn = self.model.layers[layer_idx].attn
        S = x.shape[1]

        qkv = self._linear(x, self.w_qkvs[layer_idx])  # (1, S, q_dim + 2*kv_dim)
        q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        q = q.view(1, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(1, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(1, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if attn.qk_norm:
            q = attn.q_norm(q)
            k = attn.k_norm(k)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        out = block_attend(q, k, v, (self.k_caches[layer_idx], self.v_caches[layer_idx]), (bias, positions))
        return self._linear(out, attn.o_proj.weight)

    def _forward_block(self, tokens: torch.Tensor) -> torch.Tensor:
        """S-token block forward. Compiled per static S (the block buckets).

        Mirrors _forward_step at S > 1: the causal-offset attention bias and
        rope positions derive from `pos` IN-GRAPH (query row i at absolute
        position pos+i attends keys <= pos+i), so the persistent attn_bias
        buffer is not consulted — callers restore it eagerly after chunking.
        Rows beyond pos+i (stale KV from longer past requests) stay masked.
        Returns last-position logits (1, 1, V); advances pos by S.

        No steer_buf write: blocks forward PROMPT positions (prefill/extend),
        and that write is generated-positions-only, which also keeps
        re-prefilled prior turns unsteered — see the steering block in
        __init__. Lever tables (when built) are read at ``lever_slot``, which
        prefill sets to a uniform lever's slot and to 0 otherwise.
        """
        model = self.model
        positions, bias, cos, sin = self._block_inputs(tokens)
        embed = model.embed_tokens(tokens)  # (1, S, D)
        committed: list[torch.Tensor] = []
        if self.levers is not None:
            # every layer's lever rows in one gather per forward ([n_layers, 1, hidden]): each layer adds its slice
            lever_attn = self.lever_attn.index_select(1, self.lever_slot)
            lever_mlp = self.lever_mlp.index_select(1, self.lever_slot)
            act_keep = self.lever_act_keep.index_select(1, self.lever_slot)
            act_add = self.lever_act_add.index_select(1, self.lever_slot)
        partial = embed

        for ls in self.schedule:
            layer = model.layers[ls.layer_idx]

            if committed:
                h_attn = self._route(
                    committed + [partial], layer.attn_res_query, layer.attn_res_norm.weight
                )
            else:
                h_attn = partial

            if ls.is_boundary:
                committed.append(partial)
                partial = torch.zeros_like(embed)

            attn_out = self._attn_block(
                ls.layer_idx, layer.attn_norm(h_attn), cos, sin, positions, bias
            )
            if self.levers is not None:
                attn_out = attn_out + lever_attn[ls.layer_idx]
            partial = partial + attn_out

            h_mlp = self._route(committed + [partial], layer.mlp_res_query, layer.mlp_res_norm.weight)
            x_mlp = layer.ffn_norm(h_mlp)
            if self._taps is not None:
                self._taps.record(ls.layer_idx, partial, x_mlp)
            gu = self._linear(x_mlp, self.w_gate_ups[ls.layer_idx])
            gate, up = gu.split(self.config.intermediate_size, dim=-1)
            act = F.silu(gate) * up
            if self.levers is not None:
                act = act * act_keep[ls.layer_idx] + act_add[ls.layer_idx]
            mlp_out = self._linear(act, layer.ffn.down_proj.weight)
            if self.levers is not None:
                mlp_out = mlp_out + lever_mlp[ls.layer_idx]
            partial = partial + mlp_out

        return self._block_tail(committed, partial)

    def _block_inputs(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """A block's positions (S,), causal-offset attention bias (1, 1, S, max_seq_len) and rope rows (S, head_dim);
        records the block's tokens in token_history."""
        S = tokens.shape[1]
        positions = self.pos + torch.arange(S, device=tokens.device)  # (S,)
        key_idx = torch.arange(self.max_seq_len, device=tokens.device)
        bias = torch.where(
            key_idx.view(1, 1, 1, -1) <= positions.view(1, 1, S, 1),
            torch.zeros((), device=tokens.device, dtype=self.dtype),
            torch.full((), float("-inf"), device=tokens.device, dtype=self.dtype),
        )
        cos = self.rope_cos_t.index_select(0, positions)  # (S, head_dim)
        sin = self.rope_sin_t.index_select(0, positions)
        self.token_history.index_copy_(0, positions, tokens.view(S))
        return positions, bias, cos, sin

    def _block_tail(self, committed: list[torch.Tensor], partial: torch.Tensor) -> torch.Tensor:
        """A block's last-position logits (1, 1, V); advances pos by the block's length."""
        model = self.model
        S = partial.shape[1]
        # Routing is per-position independent — only one position's logits
        # are consumed (block_last: the last real row), so slice before the
        # final route + lm_head.
        last = [c.index_select(1, self.block_last) for c in committed] + [partial.index_select(1, self.block_last)]
        x = self._route(last, model.final_res_query, model.final_res_norm.weight)
        x = model.norm(x)
        logits = self._linear(x, model.get_lm_head_weight())  # (1, 1, V)

        self.pos.add_(S)
        return logits

    # ── The static decode step ────────────────────────────────────────────────

    def _forward_step(self, token: torch.Tensor) -> torch.Tensor:
        """One decode forward. Mirrors llama._forward_attn_res_cached with locals."""
        model = self.model

        # Open the current position in the attention bias BEFORE attending,
        # and record the consumed token at its slot (keeps token_history in
        # lockstep with the KV cache for prefix matching).
        self.attn_bias.index_fill_(3, self.pos, 0.0)
        self.token_history.index_copy_(0, self.pos, token.view(1))

        embed = model.embed_tokens(token)  # (1, 1, D)
        committed: list[torch.Tensor] = []
        if self.levers is not None:
            # every layer's lever rows in one gather per forward ([n_layers, 1, hidden]): each layer adds its slice
            lever_attn = self.lever_attn.index_select(1, self.lever_slot)
            lever_mlp = self.lever_mlp.index_select(1, self.lever_slot)
            act_keep = self.lever_act_keep.index_select(1, self.lever_slot)
            act_add = self.lever_act_add.index_select(1, self.lever_slot)
        partial = embed

        for ls in self.schedule:
            layer = model.layers[ls.layer_idx]

            # Pre-attention routing (committed + current partial).
            if committed:
                h_attn = self._route(
                    committed + [partial], layer.attn_res_query, layer.attn_res_norm.weight
                )
            else:
                h_attn = partial

            if ls.is_boundary:
                committed.append(partial)
                partial = torch.zeros_like(embed)

            attn_out = self._attn(ls.layer_idx, layer.attn_norm(h_attn))
            if self.levers is not None:
                attn_out = attn_out + lever_attn[ls.layer_idx]
            partial = partial + attn_out

            # Block-persistent steering write (decode/generated tokens only).
            # steer_enabled is a trace-time constant: a disabled engine traces
            # NO new ops here; an enabled one bakes ONE row-add per layer in
            # unconditionally (zero rows = off; set_steer populates the site
            # row + later block-entry rows per write), so toggling content
            # never changes the compiled graph.
            if self.steer_enabled:
                partial = partial + self.steer_buf[ls.layer_idx]

            # Pre-MLP routing.
            h_mlp = self._route(committed + [partial], layer.mlp_res_query, layer.mlp_res_norm.weight)
            x_mlp = layer.ffn_norm(h_mlp)
            if self._taps is not None:
                self._taps.record(ls.layer_idx, partial, x_mlp)
            gu = self._linear(x_mlp, self.w_gate_ups[ls.layer_idx])
            gate, up = gu.split(self.config.intermediate_size, dim=-1)
            act = F.silu(gate) * up
            if self.levers is not None:
                act = act * act_keep[ls.layer_idx] + act_add[ls.layer_idx]
            mlp_out = self._linear(act, layer.ffn.down_proj.weight)
            if self.levers is not None:
                mlp_out = mlp_out + lever_mlp[ls.layer_idx]
            partial = partial + mlp_out

        return self._step_tail(committed, partial)

    def _step_tail(self, committed: list[torch.Tensor], partial: torch.Tensor) -> torch.Tensor:
        """The step's final routing, norm and LM head: logits (1, 1, V); advances pos by one."""
        model = self.model
        x = self._route(committed + [partial], model.final_res_query, model.final_res_norm.weight)
        x = model.norm(x)
        logits = self._linear(x, model.get_lm_head_weight())  # (1, 1, V)

        self.pos.add_(1)
        return logits

    def _step_sampled(self, token: torch.Tensor) -> torch.Tensor:
        """Forward + Gumbel-max sampling, all in-graph.

        argmax(logits/T + Gumbel noise) is exact categorical sampling from
        softmax(logits/T) (Gumbel-max theorem) — identical distribution to
        server.py's softmax+multinomial, fused by inductor, philox RNG is
        cudagraph-safe. Penalty math mirrors server.py: fp32 cast first,
        >0 divide / else multiply, presence over generated ids only.
        rep_penalty == 1.0 passes through unchanged (identity by algebra),
        so no Python branch is needed.
        """
        return self._sample_plain(self._forward_step(token))

    def _sample_plain(self, logits: torch.Tensor) -> torch.Tensor:
        """_step_sampled's sampler on the step's logits (1, 1, V)."""
        logits = logits[0, -1].float()  # (V,)

        rp = self.rep_penalty
        penalized = torch.where(logits > 0, logits / rp, logits * rp)
        logits = torch.where(self.presence, penalized, logits)

        scaled = logits * self.inv_temp
        u = torch.rand_like(scaled)
        gumbel = -torch.log((-torch.log(u.clamp_min(1e-20))).clamp_min(1e-20))
        next_token = (scaled + gumbel).argmax(keepdim=True)

        self.presence.scatter_(0, next_token, True)
        self.cur_token.copy_(next_token.view(1, 1))
        return self.cur_token

    def _step_sampled_truncated(self, token: torch.Tensor) -> torch.Tensor:
        """Forward + top-k/top-p + Gumbel-max sampling, all in-graph.

        Mirrors server.py sample_next_token exactly: penalty -> temperature ->
        top-k (kth-value threshold — ties with the kth logit survive) ->
        top-p over the RENORMALIZED survivor distribution (exclusive cumsum
        < p keeps the first token that crosses) -> categorical draw.
        Disabled dials are algebraic no-ops (top_k_eff = vocab masks nothing;
        top_p_val = 2.0 never triggers), so one graph serves every law.
        Costs one full-vocab sort (~121 us measured on B200 @49152) over the
        pure-temperature step, which keeps its own unchanged graph.
        """
        return self._sample_truncated(self._forward_step(token))

    def _sample_truncated(self, logits: torch.Tensor) -> torch.Tensor:
        """_step_sampled_truncated's sampler on the step's logits (1, 1, V)."""
        logits = logits[0, -1].float()  # (V,)

        rp = self.rep_penalty
        penalized = torch.where(logits > 0, logits / rp, logits * rp)
        logits = torch.where(self.presence, penalized, logits)
        scaled = logits * self.inv_temp

        sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
        kth = sorted_logits.index_select(0, (self.top_k_eff - 1).view(1))  # (1,)
        sorted_logits = torch.where(
            sorted_logits < kth, torch.full_like(sorted_logits, float("-inf")), sorted_logits
        )
        # -inf suffix -> softmax renormalizes over the top-k survivors, which
        # is what server.py's second softmax sees.
        probs = sorted_logits.softmax(dim=-1)
        cutoff = probs.cumsum(dim=-1) - probs >= self.top_p_val
        sorted_logits = torch.where(
            cutoff, torch.full_like(sorted_logits, float("-inf")), sorted_logits
        )
        floor = sorted_logits[:1] + self.log_min_p  # min-p: probability >= min_p * the top token's
        sorted_logits = torch.where(
            sorted_logits < floor, torch.full_like(sorted_logits, float("-inf")), sorted_logits
        )
        final = torch.full_like(scaled, float("-inf")).scatter(0, sorted_idx, sorted_logits)

        u = torch.rand_like(final)
        gumbel = -torch.log((-torch.log(u.clamp_min(1e-20))).clamp_min(1e-20))
        next_token = (final + gumbel).argmax(keepdim=True)

        self.presence.scatter_(0, next_token, True)
        self.cur_token.copy_(next_token.view(1, 1))
        return self.cur_token

    # ── Sampling (server.py sample_next_token semantics, on-GPU) ─────────────

    def _sample(self, logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        """Sample from (1, 1, V) logits → (1, 1) token. Updates presence."""
        logits = logits[0, -1].float()  # (V,) — server.py casts to float first

        rp = params.repetition_penalty
        if rp != 1.0:
            penalized = torch.where(logits > 0, logits / rp, logits * rp)
            logits = torch.where(self.presence, penalized, logits)

        if params.temperature == 0.0:
            token = logits.argmax(keepdim=True)
        else:
            logits = logits / params.temperature
            if params.top_k > 0:
                top_k = min(params.top_k, logits.size(-1))
                kth_val = logits.topk(top_k).values[-1]
                logits = logits.masked_fill(logits < kth_val, float("-inf"))
            if 0.0 < params.top_p < 1.0:
                sorted_logits, sorted_indices = logits.sort(descending=True)
                probs_sorted = sorted_logits.softmax(dim=-1)
                mask = probs_sorted.cumsum(dim=-1) - probs_sorted >= params.top_p
                sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_indices, sorted_logits)
            if params.min_p > 0.0:
                logits = logits.masked_fill(logits < logits.max() + math.log(params.min_p), float("-inf"))
            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1)

        self.presence.scatter_(0, token, True)
        return token.view(1, 1)

    # ── Public API ───────────────────────────────────────────────────────────

    @torch.inference_mode()
    def reset(self) -> None:
        """Clear per-request state.

        KV contents are also zeroed (~1 GB of writes, ~0.25 ms): masked-out
        stale slots are weight-0 in SDPA, but 0 * inf = NaN, so one request
        that ever wrote a non-finite K/V row would otherwise poison every
        subsequent request until those slots are overwritten.
        """
        self.pos.zero_()
        self._pos_cpu = 0
        self.prompt_kv_len = 0
        self.attn_bias.fill_(float("-inf"))
        self.presence.zero_()
        for li in range(self.n_layers):
            self.k_caches[li].zero_()
            self.v_caches[li].zero_()
        self._kv_lever = None
        self._prefill_end = 0
        self._gen_steered = False
        if self.levers is not None:
            self.lever_slot.zero_()
        # NOTE: steer_buf is deliberately NOT cleared here — it is per-request
        # state owned by the caller (server.py engages it before prefill and
        # clears it in a finally after every request).

    @torch.inference_mode()
    def set_steer(self, writes: Optional[list[tuple[int, torch.Tensor, float]]], rule: str = "K",
                  norms: Optional[dict[int, float]] = None) -> None:
        """Install (or clear) steering writes for subsequent decode steps.

        `writes` is None (or an empty list) to clear, else a list of
        (site_layer, vec, scale) triples: scale*vec is added into the
        per-layer buffer at row site_layer AND at every block-entry row
        after site_layer's block (compute_steer_points — the block-persistent
        primitive, applied per write). Writes at different sites populate
        different rows; overlapping rows SUM. The buffer is mutated IN-PLACE
        (staged in fp32, cast to engine dtype) — compiled graphs hold it by
        identity, so this never recompiles or swaps a graph. Callers own
        scaling: pass scale = alpha * that site's median residual norm.

        ``rule`` sets what lands at the later block entries: "K" the site row
        unchanged (block-persistent, the default), "N" the site row times
        norms[row] / norms[site] (re-added at the scale of the partial it lands
        in; ``norms`` = per-layer median post-attention partial norm), "S" no
        re-add (a single write at the site).

        Raises on a steering-disabled engine (except set_steer(None), which is
        a no-op there so callers can clear unconditionally in a finally).
        """
        if rule not in ("K", "N", "S"):
            raise ValueError(f"steer rule must be K, N or S, got {rule!r}")
        if not self.steer_enabled:
            if writes is None:
                return
            raise RuntimeError(
                "Engine was built without steering (steer_enabled=False); "
                "rebuild with steer_enabled=True to enable set_steer()."
            )
        if not writes:  # None or []
            self.steer_buf.zero_()
            self._steer_active = False
            return
        staged = torch.zeros((self.n_layers, self.hidden), device=self.device, dtype=torch.float32)
        active = False
        for i, item in enumerate(writes):
            try:
                site_layer, vec, scale = item
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"writes[{i}] must be a (site_layer, vec, scale) triple, got {item!r}"
                ) from exc
            if not isinstance(vec, torch.Tensor):
                raise TypeError(f"writes[{i}]: vec must be a torch.Tensor, got {type(vec).__name__}")
            if vec.numel() != self.hidden:
                raise ValueError(
                    f"writes[{i}]: vec has {vec.numel()} elements, expected hidden={self.hidden}"
                )
            scale = float(scale)
            if not math.isfinite(scale):
                raise ValueError(f"writes[{i}]: non-finite scale {scale}")
            v = vec.detach().reshape(self.hidden).to(device=self.device, dtype=torch.float32)
            if not bool(torch.isfinite(v).all().item()):
                raise ValueError(f"writes[{i}]: vec contains non-finite values")
            site = int(site_layer)
            rows = (site,) if rule == "S" else compute_steer_points(site, self._boundaries, self.n_layers)
            for row in rows:
                factor = 1.0
                if rule == "N" and row != site:
                    if not norms or site not in norms or row not in norms:
                        raise ValueError(f"steer rule N needs partial norms for layers {site} and {row}")
                    factor = norms[row] / norms[site]
                staged[row] += scale * factor * v
            if scale != 0.0:
                active = True
        self.steer_buf.copy_(staged.to(self.dtype))
        # Host-side activity flag (NO device sync — derived from the writes
        # themselves): gates the prefix-cache suffix==1 shortcut, which must
        # not run the injecting compiled step on a prompt position while any
        # row is live. Conservative: a nonzero-scale write of an all-zero
        # vector still reads active (harmless — only skips the shortcut).
        self._steer_active = active

    def _chunk_plan(self, start: int, end: int) -> list[tuple[int, int]] | None:
        """Decompose [start, end) into block-bucket chunks.

        Greedy largest-bucket cover; a remainder smaller than the smallest
        bucket becomes a RIGHT-ALIGNED overlap chunk (re-forwarding a few
        already-cached tokens writes identical K/V — harmless, and it keeps
        every chunk a compiled static shape). Returns None when the span (or
        prompt head, for the overlap) can't be covered — caller falls back
        to the eager reference path.

        With batch-invariant blocks the plan never overlaps: greedy chunks,
        the last RIGHT-PADDED to a bucket (``_run_chunks`` pads the ids; the
        pad rows write KV past ``end``, masked until a later write replaces
        it), so no cached position is re-forwarded and a prompt of any
        length runs through the blocks.
        """
        if self.block_invariant:
            b_min = self.block_buckets[-1]
            plan = []
            i = start
            while i < end:
                b = next((B for B in self.block_buckets if B <= end - i), b_min)
                plan.append((i, b))
                i += b
            if plan and plan[-1][0] + plan[-1][1] > self.max_seq_len:
                # no room to pad at the cache's end: the last chunk right-aligned instead (one kernel per width, so
                # the re-forwarded cached positions come out with the bytes they hold)
                s0, b = plan[-1]
                if end - b < 0:
                    return None
                plan[-1] = (end - b, b)
            return plan or None
        if self.extend_single_chunk and start > 0:
            fit = [B for B in self.block_buckets if end - start <= B <= end]
            if fit:
                B = min(fit)
                return [(end - B, B)]
        b_min = self.block_buckets[-1]
        plan: list[tuple[int, int]] = []
        i = start
        while end - i >= b_min:
            b = next(B for B in self.block_buckets if B <= end - i)
            plan.append((i, b))
            i += b
        if i < end:
            if end - b_min < 0:
                return None  # prompt shorter than the smallest bucket
            plan.append((end - b_min, b_min))
        return plan

    @torch.inference_mode()
    def _run_chunks(self, input_ids: torch.Tensor, plan: list[tuple[int, int]], eager: bool = False) -> torch.Tensor:
        """Execute a chunk plan via the compiled block forward (``eager``: the same block forward uncompiled, for
        spans no bucket covers).

        Each chunk pins pos eagerly (overlap chunks rewind it), replays the
        bucket graph, and leaves pos at chunk end. Restores the persistent
        attn_bias/pos state the decode step expects afterwards.
        """
        block = self._forward_block if eager else self._compiled_block
        assert block is not None
        end = input_ids.shape[1]
        need = max(s + b for s, b in plan)
        ids = input_ids.to(self.device)
        if need > end:  # a right-padded last chunk (batch-invariant plans): its pad rows repeat the last id
            ids = torch.cat([ids, ids[:, -1:].expand(1, need - end)], dim=1)
        logits = None
        for s, b in plan:
            self.pos.fill_(s)
            self.block_last.fill_(min(b, end - s) - 1)
            logits = block(_graph_input(ids[:, s: s + b]))
        if need > end:
            self.token_history[end:need].fill_(0)
        # Block graphs derive their bias in-graph; re-establish the decode
        # invariant on the persistent buffer: [0, end) open, rest closed.
        self.attn_bias.fill_(float("-inf"))
        self.attn_bias[..., :end] = 0.0
        self.pos.fill_(end)
        self._pos_cpu = end
        return logits[0, -1:].float()

    def _prompt_slot(self, slot: int) -> tuple[int, Optional[str]]:
        """The slot a request's PROMPT positions read and the digest of the lever that writes them: a uniform
        lever's own slot, else the identity (a decode-span lever writes generated positions only)."""
        slot = int(slot)
        if self.levers is None:
            if slot != 0:
                raise RuntimeError("Engine was built without lever tables (levers=0); a lever slot needs levers > 0")
            return 0, None
        self.levers.check_slots([slot])
        if slot == 0 or self.levers.span(slot) != "uniform":
            return 0, None
        return slot, self.levers.plan(slot).digest()

    def _arm_decode_slot(self, slot: int) -> None:
        """The slot the single-stream decode step reads from here on: the request's lever, whatever its span. Called
        at the end of every prefill, so it also records where the prefill-written positions end."""
        self._prefill_end = self._pos_cpu
        steered = self._steer_active  # broadcast rows engaged for this request (set_steer before the prefill)
        if self.levers is not None:
            self.lever_slot.fill_(int(slot))
            steered = steered or bool(slot)
        self._gen_steered = steered

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor, slot: int = 0) -> torch.Tensor:
        """Full prefill into the static cache; returns last logits (1, V) fp32.

        Routes through the compiled block forward when available (chunked,
        ~launch-free); the eager reference path (prefill_reference) covers
        un-compiled engines and prompts shorter than the smallest bucket.

        ``slot`` (an engine built with levers): the request's lever. A uniform
        lever's prefill writes the prompt through the block forward at that
        slot (compiled when a bucket plan covers the prompt, else the same
        block forward eager); its prompt KV is then steered, so it is not
        pooled by tokens (``prompt_kv_len`` stays 0). The decode steps that
        follow read the slot whatever its span.
        """
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"Expected (1, S) input_ids, got {tuple(input_ids.shape)}")
        prompt_len = input_ids.shape[1]
        self.reset()
        if prompt_len >= self.max_seq_len:
            raise ValueError(f"Prompt too long: {prompt_len} >= {self.max_seq_len}")
        pslot, digest = self._prompt_slot(slot)
        plan = (
            self._chunk_plan(0, prompt_len)
            if self._compiled_block is not None and prompt_len <= self.block_span_max and self._taps is None
            else None
        )
        if self._taps is not None:  # one eager block over the whole prompt, so every position is tapped once
            if self.levers is not None:
                self.lever_slot.fill_(pslot)
            logits = self._run_chunks(input_ids.to(self.device), [(0, prompt_len)], eager=True)
            self.prompt_kv_len = 0 if pslot else prompt_len
            self._kv_lever = digest
            self._arm_decode_slot(slot)
            return logits
        if pslot:
            self.lever_slot.fill_(pslot)
            if plan is None:
                logits = self._run_chunks(input_ids.to(self.device), [(0, prompt_len)], eager=True)
            else:
                logits = self._run_chunks(input_ids.to(self.device), plan)
            self.prompt_kv_len = 0
            self._kv_lever = digest
            self._arm_decode_slot(slot)
            return logits
        if plan is None:
            logits = self._prefill_reference_inner(input_ids, prompt_len)
        else:
            logits = self._run_chunks(input_ids.to(self.device), plan)
            self.prompt_kv_len = prompt_len
        self._arm_decode_slot(slot)
        return logits

    @torch.inference_mode()
    def prefill_reference(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Reference eager prefill (parity oracle; pre-block behavior)."""
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"Expected (1, S) input_ids, got {tuple(input_ids.shape)}")
        prompt_len = input_ids.shape[1]

        # Reset BEFORE the length check: a caller that catches the error and
        # keeps stepping must not silently continue the previous request.
        self.reset()
        if prompt_len >= self.max_seq_len:
            raise ValueError(f"Prompt too long: {prompt_len} >= {self.max_seq_len}")
        return self._prefill_reference_inner(input_ids, prompt_len)

    def _prefill_reference_inner(self, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:

        # Pin prefill to flash (efficient as backstop): cuDNN's per-shape plan
        # selection costs ~300 ms on every novel prompt length, and prefill
        # shapes are arbitrary. This also makes prefill numerics independent
        # of process-global SDPA toggles (parity was measured on this kernel).
        with sdpa_kernel(self._prefill_backends):
            out = self.model(input_ids.to(self.device), use_cache=True)
        for li, (k, v) in enumerate(out["past_kv"]):
            self.k_caches[li][:, :, :prompt_len].copy_(k)
            self.v_caches[li][:, :, :prompt_len].copy_(v)
        self.attn_bias[..., :prompt_len] = 0.0
        self.token_history[:prompt_len].copy_(input_ids[0])
        self.pos.fill_(prompt_len)
        self._pos_cpu = prompt_len
        self.prompt_kv_len = prompt_len
        return out["logits"][0, -1:].float()

    # ── Prefix-cached prefill (extend-from-pos) ──────────────────────────────

    def cached_prefix_len(self, input_ids: torch.Tensor) -> int:
        """How many leading tokens of ``input_ids`` (1, S) the cache holds now (token-exact; one host sync)."""
        return self._common_prefix_len(input_ids)

    @torch.inference_mode()
    def adopt_prefix(self, input_ids: torch.Tensor, n: int) -> None:
        """Declare that cache positions [0, n) hold prefill-written KV of ``input_ids[0, :n]``: a caller that wrote
        that KV itself (kotodama.model.prefix_cache loads pooled blocks) calls this, and the next ``prefill_cached``
        of a prompt starting with those tokens extends from position n. Positions from n on are left as they are
        (closed in the mask by the extend)."""
        if not 0 < n < self.max_seq_len or n > input_ids.shape[1]:
            raise ValueError(f"adopt_prefix: n={n} outside (0, {min(self.max_seq_len - 1, input_ids.shape[1])}]")
        self.token_history[:n].copy_(input_ids[0, :n].to(self.device))
        self.pos.fill_(n)
        self._pos_cpu = n
        self.prompt_kv_len = n
        self._kv_lever = None

    def _common_prefix_len(self, input_ids: torch.Tensor) -> int:
        """Longest common prefix between the cached sequence and a new prompt.

        Token-exact comparison on GPU; one host sync (turn granularity).
        """
        n = min(self._pos_cpu, input_ids.shape[1])
        if n <= 0:
            return 0
        ids = input_ids[0, :n].to(self.device)
        neq = (self.token_history[:n] != ids).nonzero()
        return n if neq.numel() == 0 else int(neq[0, 0].item())

    @torch.inference_mode()
    def prefill_cached(self, input_ids: torch.Tensor, slot: int = 0,
                       prompt_kv_only: bool = False) -> tuple[torch.Tensor, dict]:
        """Prefill reusing any cached common prefix from the previous request.

        Token-exact match against token_history; on a hit, only the suffix is
        forwarded (reference path, explicit causal-offset mask) and its KV is
        appended at the right positions. On a miss (or < min_cached_prefix
        tokens shared) this is exactly prefill().

        Per-request semantics preserved: presence resets here; capacity rules
        match prefill(). The stale-KV NaN insurance that reset() provides is
        replaced by an explicit finiteness gate on the appended suffix KV
        (non-finite -> warn + full re-prefill, which zeroes).

        ``slot``: the request's lever, as in prefill(). A cached prefix is
        reused only when the lever that wrote it (none, or the same uniform
        lever by content digest) is the one this prompt is written with, and
        never past the previous prompt when that request decoded steered (a
        lever, or broadcast rows: its generated positions hold steered KV); a uniform lever's extend
        runs the block forward, which reads its slot. Inside a capture every
        prompt is a full eager prefill.

        ``prompt_kv_only``: reuse only KV a prefill wrote (never a decode step's, whose kernels write other bytes):
        with batch-invariant blocks the result is then exactly a cold prefill's.

        Returns (last-position logits (1, V) float32, info dict with
        prefix_hit / common_prefix / suffix_len for observability).
        """
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"Expected (1, S) input_ids, got {tuple(input_ids.shape)}")
        prompt_len = input_ids.shape[1]
        if prompt_len >= self.max_seq_len:
            # Mirror prefill(): never leave a caller able to silently continue
            # the previous request after catching the error.
            self.reset()
            raise ValueError(f"Prompt too long: {prompt_len} >= {self.max_seq_len}")

        pslot, digest = self._prompt_slot(slot)
        # Cap at plen-1: even a fully-cached prompt must re-forward its last
        # token to produce next-token logits (regenerate / undo-to-midpoint).
        common = min(self._common_prefix_len(input_ids), prompt_len - 1)
        if digest != self._kv_lever:
            common = 0  # the cached prompt KV was written under another lever
        if self._gen_steered:
            common = min(common, self._prefill_end)  # the decode steps after it wrote steered KV
        if prompt_kv_only:
            common = min(common, self.prompt_kv_len)
        if self._taps is not None:
            common = 0  # a capture taps every position: one eager prefill of the whole prompt
        if common < self.min_cached_prefix or (pslot and self._compiled_block is None):
            logits = self.prefill(input_ids, slot=slot)
            return logits, {"prefix_hit": False, "common_prefix": 0, "suffix_len": prompt_len}
        if self.levers is not None:
            self.lever_slot.fill_(pslot)
        # The suffix attends to the reused prefix, so its KV is prompt KV only when the whole prefix is.
        prompt_kv_after = prompt_len if common <= self.prompt_kv_len else self.prompt_kv_len

        suffix = input_ids[:, common:].to(self.device)
        suffix_len = prompt_len - common

        # Per-request state. KV rows beyond prompt_len go stale but stay
        # masked by the re-closed bias; they are finite by the same gate below.
        self.presence.zero_()
        self.attn_bias.fill_(float("-inf"))

        if suffix_len == 1 and not self._steer_active and not self.block_invariant:
            # A 1-token extend IS a decode step (regenerate / undo-to-tip):
            # run it through the compiled step — no mask machinery, ~2.5ms.
            # Steering caveat: the compiled step injects steer_buf rows, but this
            # token is a PROMPT position — with a nonzero vector engaged we
            # skip this shortcut (falls through to the block/eager extend,
            # which never injects) to preserve generated-positions-only
            # semantics. Vector off -> unchanged fast path.
            # The step opens bias at its own position and records history.
            # The suffix is a slice with a nonzero storage offset; _graph_input
            # gives the compiled step the layout it was captured with (see
            # _graph_input for what a varying offset costs).
            self.attn_bias[..., :common] = 0.0
            self.pos.fill_(common)
            self._pos_cpu = common
            logits = self.step_logits(_graph_input(suffix))
            self.prompt_kv_len = 0 if pslot else prompt_kv_after
            self._arm_decode_slot(slot)
            return logits[0, -1:].float(), {
                "prefix_hit": True, "common_prefix": common, "suffix_len": 1,
            }

        # Compiled block extend: chunked block forwards replace the eager
        # MATH-SDPA suffix path (a 33-40ms launch floor -> block replays).
        # Overlap chunks may rewind into the cached region; those tokens are
        # present in input_ids, so re-forwarding writes identical K/V. Falls
        # through to the eager path when uncompiled.
        if self._compiled_block is not None and suffix_len <= self.block_span_max:
            plan = self._chunk_plan(common, prompt_len)
            if plan is not None:
                self.presence.zero_()
                lo = min(s for s, _ in plan)
                try:
                    logits = self._run_chunks(input_ids.to(self.device), plan)
                except Exception:
                    self.reset()  # same contract as the eager extend below
                    raise
                flags = []
                for li in range(self.n_layers):
                    flags.append(self.k_caches[li][:, :, lo:prompt_len].isfinite().all())
                    flags.append(self.v_caches[li][:, :, lo:prompt_len].isfinite().all())
                finite_ok = bool(torch.stack(flags).all().item())  # one host sync
                if not finite_ok:
                    logger.warning(
                        "prefill_cached(block): non-finite KV at common=%d suffix=%d — "
                        "falling back to full re-prefill", common, suffix_len,
                    )
                    logits = self.prefill(input_ids, slot=slot)
                    return logits, {"prefix_hit": False, "common_prefix": 0, "suffix_len": prompt_len}
                self.prompt_kv_len = 0 if pslot else prompt_kv_after
                self._arm_decode_slot(slot)
                return logits, {
                    "prefix_hit": True, "common_prefix": common, "suffix_len": suffix_len,
                }

        if pslot:  # the eager extend runs model.forward, which reads no lever: prefill the whole prompt steered
            logits = self.prefill(input_ids, slot=slot)
            return logits, {"prefix_hit": False, "common_prefix": 0, "suffix_len": prompt_len}
        past = [
            (self.k_caches[li][:, :, :common], self.v_caches[li][:, :, :common])
            for li in range(self.n_layers)
        ]
        # Causal-with-offset mask (True = attend): suffix query i (absolute
        # position common+i) sees keys j <= common+i. Required: the reference
        # SDPA path sets is_causal=False whenever past_kv is given, which is
        # only correct for single-token steps.
        mask = torch.ones(suffix_len, prompt_len, dtype=torch.bool, device=self.device)
        mask = mask.tril_(diagonal=common)

        try:
            with sdpa_kernel(self._extend_backends):
                out = self.model(suffix, use_cache=True, past_kv=past, mask=mask)
        except Exception:
            # Don't leave a half-extended state (bias closed, pos stale) for a
            # caller that catches and keeps going — same contract as prefill().
            self.reset()
            raise

        finite_flags = []
        for li, (k, v) in enumerate(out["past_kv"]):
            k_suf = k[:, :, common:]
            v_suf = v[:, :, common:]
            self.k_caches[li][:, :, common:prompt_len].copy_(k_suf)
            self.v_caches[li][:, :, common:prompt_len].copy_(v_suf)
            finite_flags.append(k_suf.isfinite().all())
            finite_flags.append(v_suf.isfinite().all())
        if not bool(torch.stack(finite_flags).all().item()):
            logger.warning(
                "prefill_cached: non-finite suffix KV at common=%d suffix=%d — "
                "falling back to full re-prefill", common, suffix_len,
            )
            logits = self.prefill(input_ids, slot=slot)
            return logits, {"prefix_hit": False, "common_prefix": 0, "suffix_len": prompt_len}

        self.attn_bias[..., :prompt_len] = 0.0
        self.token_history[:prompt_len].copy_(input_ids[0])
        self.pos.fill_(prompt_len)
        self._pos_cpu = prompt_len
        self.prompt_kv_len = prompt_kv_after
        self._arm_decode_slot(slot)
        return out["logits"][0, -1:].float(), {
            "prefix_hit": True, "common_prefix": common, "suffix_len": suffix_len,
        }

    @torch.inference_mode()
    def extend_padded(self, input_ids: torch.Tensor, start: int) -> torch.Tensor:
        """Forward positions [start, S) of ``input_ids`` (1, S) over the KV the cache holds at [0, start), without
        rewriting any of it: greedy bucket chunks from ``start``, the last one right-padded (its pad rows write KV
        past S, which stays masked until a later write replaces it). The caller has set the cache's first ``start``
        positions (and their token history); a KV session's turn extends this way. Batch-invariant blocks only (a
        padded chunk's real rows then come out as in any other chunking). Returns the logits at position S - 1
        (1, V) fp32; leaves the engine holding S positions, nothing of it reusable as prompt KV."""
        if self._compiled_block is None or not self.block_invariant:
            raise RuntimeError("extend_padded needs compiled, batch-invariant blocks")
        end = input_ids.shape[1]
        plan = self._chunk_plan(start, end) if 0 < start < end else None
        if plan is None:
            raise ValueError(f"extend_padded: positions [{start}, {end}) outside the cache ({self.max_seq_len})")
        logits = self._run_chunks(input_ids, plan)
        self.presence.zero_()
        self.prompt_kv_len = 0
        self._arm_decode_slot(0)
        return logits

    @torch.inference_mode()
    def sample_first(self, prefill_logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        """Sample the first token from prefill logits. Returns (1, 1) CUDA tensor. Under
        ``penalty_scope="row"`` the presence set starts as the prompt's ids."""
        if params.penalty_scope == "row" and self._pos_cpu > 0:
            self.presence.index_fill_(0, self.token_history[: self._pos_cpu], True)
        token = self._sample(prefill_logits.view(1, 1, -1), params)
        self.cur_token.copy_(token)
        return self.cur_token

    def _check_capacity(self) -> None:
        if self._pos_cpu >= self.max_seq_len:
            raise RuntimeError(
                f"Decode position {self._pos_cpu} would exceed max_seq_len "
                f"{self.max_seq_len}; caller must stop at capacity (an OOB "
                "index_copy_ is a sticky device-side assert)."
            )

    def _sync_sampling_buffers(self, params: SamplingParams) -> None:
        if params.temperature != self._cur_temp:
            self.inv_temp.fill_(1.0 / params.temperature)
            self._cur_temp = params.temperature
        if params.repetition_penalty != self._cur_rp:
            self.rep_penalty.fill_(params.repetition_penalty)
            self._cur_rp = params.repetition_penalty
        if params.top_k != self._cur_top_k:
            # 0 (disabled) -> vocab_size: the kth threshold becomes the min
            # logit and masks nothing (server.py skips the block entirely).
            k = params.top_k if params.top_k > 0 else self.config.vocab_size
            self.top_k_eff.fill_(min(k, self.config.vocab_size))
            self._cur_top_k = params.top_k
        if params.min_p != self._cur_min_p:
            self.log_min_p.fill_(math.log(params.min_p) if params.min_p > 0.0 else float("-inf"))
            self._cur_min_p = params.min_p
        if params.top_p != self._cur_top_p:
            # 0.0/1.0 (disabled) -> 2.0: unreachable even when fp32 cumsum
            # saturates to 1.0 (which it does — see the buffer comment).
            p = params.top_p if 0.0 < params.top_p < 1.0 else 2.0
            self.top_p_val.fill_(p)
            self._cur_top_p = params.top_p

    @torch.inference_mode()
    def step(self, token: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        """One decode step: forward(token) → sample → next token (1, 1) on GPU.

        Returns the engine's persistent ``cur_token`` buffer — valid only
        until the next step. Consume immediately (``.item()`` / compare);
        never accumulate the returned tensor across steps.
        """
        self._check_capacity()
        # Fully-fused path (forward + sampler in one graph) for T > 0.
        # Truncated laws (top-k/top-p) use their own compiled variant; the
        # pure-temperature graph is byte-identical to before they existed.
        # (At temperature 0 server.py argmaxes BEFORE truncation, so truncation
        # is a no-op there — the greedy path below is correct for all laws.)
        if params.temperature > 0.0 and self._taps is None:
            fused = (
                self._compiled_step_truncated if params.truncated else self._compiled_step_sampled
            )
            if fused is not None:
                self._sync_sampling_buffers(params)
                out = fused(token)
                self._pos_cpu += 1
                return out
        forward = self._compiled_forward if self._compiled_forward is not None and self._taps is None \
            else self._forward_step
        logits = forward(token)
        self._pos_cpu += 1
        next_token = self._sample(logits, params)
        self.cur_token.copy_(next_token)
        return self.cur_token

    @torch.inference_mode()
    def step_logits(self, token: torch.Tensor) -> torch.Tensor:
        """Forward only (parity testing): logits (1, 1, V) without sampling.

        Under a compiled step the result lives in CUDA-graph pool memory and
        is CLOBBERED by the next step call — copy (``.float()``/``.clone()``)
        before stepping again.
        """
        self._check_capacity()
        forward = self._compiled_forward if self._compiled_forward is not None and self._taps is None \
            else self._forward_step
        out = forward(token)
        self._pos_cpu += 1
        return out

    def compile_step(self, mode: str = "reduce-overhead", regional: bool = False, backend="inductor",
                     cuda_graph: Optional[bool] = None, block_mode: Optional[str] = None) -> None:
        """torch.compile the decode-step variants and the block forward.

        Default: each is one graph of the whole depth under ``mode`` (CUDA graphs come from the mode).
        ``regional`` (kotodama.model.regional): each layer runs as the compiled graph of its signature, one graph per
        distinct signature shared by the forward, sampled and truncated steps, and one set per block bucket; the
        tails and samplers are graphs of their own, all without CUDA graphs (``mode`` maps to its no-cudagraphs
        form). With ``cuda_graph`` (default: on a CUDA engine) each step variant and each block bucket is captured
        whole as one CUDA graph on its third call and replayed after. ``backend`` (regional only): the torch.compile
        backend of the regions. ``block_mode``: the block forward's mode (default ``mode``); prefill blocks gain little
        from kernel benchmarking and their buckets multiply its cost."""
        block_mode = block_mode or mode
        if regional:
            self._compile_regional(mode, backend, cuda_graph, block_mode)
            return
        if backend != "inductor" or cuda_graph is not None:
            raise ValueError("backend and cuda_graph apply to the regional step only (regional=True)")
        logger.info("Compiling decode step (mode=%s)...", mode)
        import torch._dynamo

        # cur_token is fed back as the next step's input; pinning its address
        # lets cudagraph trees record it directly instead of copying per replay.
        torch._dynamo.mark_static_address(self.cur_token)
        self._compiled_forward = torch.compile(self._forward_step, mode=mode, fullgraph=True)
        self._compiled_step_sampled = torch.compile(self._step_sampled, mode=mode, fullgraph=True)
        self._compiled_step_truncated = torch.compile(self._step_sampled_truncated, mode=mode, fullgraph=True)
        self._compiled_block = torch.compile(self._forward_block, mode=block_mode, fullgraph=True)

    # ── The regional step and block forward (compile_step(regional=True)) ──

    def _regional_levers(self) -> Optional[tuple[torch.Tensor, ...]]:
        """Every layer's lever rows at the engine's slot: (attn, mlp, act keep, act add), each [n_layers, 1, width]."""
        if self.levers is None:
            return None
        return tuple(t.index_select(1, self.lever_slot)
                     for t in (self.lever_attn, self.lever_mlp, self.lever_act_keep, self.lever_act_add))

    def _regional_step_forward(self, token: torch.Tensor) -> torch.Tensor:
        """``_forward_step`` with each layer run as its signature's compiled region: token (1, 1) -> logits
        (1, 1, V)."""
        self.attn_bias.index_fill_(3, self.pos, 0.0)
        self.token_history.index_copy_(0, self.pos, token.view(1))
        embed = self.model.embed_tokens(token)
        cos = self.rope_cos_t.index_select(0, self.pos)
        sin = self.rope_sin_t.index_select(0, self.pos)
        committed, partial = run_regions(self._step_regions, embed, cos, sin, (self.attn_bias, self.pos),
                                         self._regional_levers(), self.steer_buf if self.steer_enabled else None, None)
        return self._regional_step_tail(committed, partial)

    def _regional_block_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``_forward_block`` over the regions of the bucket ``tokens.shape[1]``: last-position logits (1, 1, V)."""
        positions, bias, cos, sin = self._block_inputs(tokens)
        embed = self.model.embed_tokens(tokens)
        committed, partial = run_regions(self._block_regions[tokens.shape[1]], embed, cos, sin, (bias, positions),
                                         self._regional_levers(), None, None)
        return self._regional_block_tail(committed, partial)

    def _regional_block(self, tokens: torch.Tensor) -> torch.Tensor:
        S = tokens.shape[1]
        if S not in self._block_runs:
            raise ValueError(f"no regional block forward for {S} tokens (buckets {self.block_buckets})")
        return self._block_runs[S](tokens)

    def _compile_regional(self, mode: str, backend, cuda_graph: Optional[bool], block_mode: str) -> None:
        check_norms(self.model, self.eps)
        on_cuda = self.device.type == "cuda"
        cuda_graph = on_cuda if cuda_graph is None else bool(cuda_graph)
        if cuda_graph and not on_cuda:
            raise ValueError("whole-step CUDA graph capture needs a CUDA engine")
        opts = dict(fullgraph=True, dynamic=False, backend=backend)
        block_opts = dict(opts)
        if backend == "inductor":
            opts["options"] = region_options(mode, self.emulate_casts)
            block_opts["options"] = region_options(block_mode, self.emulate_casts)
        kvs = list(zip(self.k_caches, self.v_caches))
        sigs = [(c, b, False, False, 0) for c, b in layer_signatures(self._boundaries, self.n_layers)]
        build = dict(attend=cache_attend, linear=decode_linear, **opts)
        self._step_regions, self.regional_variants = build_regions(
            self, sigs, kvs, "engine_step_layer", steer=self.steer_enabled, **build)
        self._regional_step_tail = torch.compile(self._step_tail, **opts)
        self._regional_block_tail = torch.compile(self._block_tail, **block_opts)
        pool = torch.cuda.graph_pool_handle() if cuda_graph else None

        def step_with(head: Optional[Callable[[torch.Tensor], torch.Tensor]]) -> Callable[[torch.Tensor], torch.Tensor]:
            compiled = torch.compile(head, **opts) if head is not None else None

            def run(token: torch.Tensor) -> torch.Tensor:
                logits = self._regional_step_forward(token)
                return compiled(logits) if compiled is not None else logits

            return StepGraph(run, self.cur_token, pool) if cuda_graph else run

        self._compiled_forward = step_with(None)
        self._compiled_step_sampled = step_with(self._sample_plain)
        self._compiled_step_truncated = step_with(self._sample_truncated)
        # Each block bucket: its own regions (a static S), and on CUDA its own captured graph reading its own input.
        self._block_regions, self._block_runs = {}, {}
        for b in self.block_buckets:
            self._block_regions[b], _ = build_regions(self, sigs, kvs, f"engine_block{b}_layer", steer=False,
                                                      attend=block_attend, linear=decode_linear, **block_opts)
            if cuda_graph:
                buf = torch.zeros((1, b), device=self.device, dtype=torch.long)
                self._block_runs[b] = StepGraph(self._regional_block_forward, buf, pool)
            else:
                self._block_runs[b] = self._regional_block_forward
        self._compiled_block = self._regional_block
        region_mode = f"{mode}{' (emulated casts)' if self.emulate_casts else ''}" if backend == "inductor" else backend
        logger.info("DecodeEngine: regional step and block forward, %d layer signatures over %d layers (blocks %s), "
                    "regions %s, %s", self.regional_variants, self.n_layers, self.block_buckets,
                    region_mode,
                    "whole-forward CUDA graphs" if cuda_graph else "no CUDA graph")

    @torch.inference_mode()
    def warm_blocks(self) -> None:
        """Trigger the per-bucket block-graph compiles (call once at startup).

        Each bucket size is a separate dynamo specialization; an unwarmed
        bucket would pay its compile on the first real request.
        """
        if self._compiled_block is None:
            return
        g = torch.Generator().manual_seed(3)
        for b in self.block_buckets:
            ids = torch.randint(4, self.config.vocab_size, (1, b + 1), generator=g).to(self.device)
            self.prefill(ids[:, :b])          # exact-bucket plan
            self.prefill(ids)                 # overlap-remainder plan
        self.reset()

    @torch.inference_mode()
    def generate_greedy(self, input_ids: torch.Tensor, max_new_tokens: int) -> list[int]:
        """Raw-argmax generation helper for PARITY TESTING ONLY.

        Not a serving surrogate: applies no repetition penalty and no stop
        tokens (server.py at temperature 0 still penalizes and stops).
        Prefill pinned to the reference path — parity harnesses gate the
        decode step from an identical cache; blocks have their own battery.
        """
        logits = self.prefill_reference(input_ids)
        token = logits.argmax(-1).view(1, 1)
        out_ids = [int(token.item())]
        budget = min(max_new_tokens - 1, self.max_seq_len - self._pos_cpu)
        for _ in range(budget):
            self.cur_token.copy_(token)
            logits = self.step_logits(self.cur_token)
            token = logits[0, -1].argmax().view(1, 1)
            out_ids.append(int(token.item()))
        return out_ids
