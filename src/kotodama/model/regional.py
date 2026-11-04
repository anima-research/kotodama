"""Regional compile of the decode forwards: one compiled graph per layer signature instead of one graph of the whole
depth.

The whole-step compile traces every effective layer into one graph, so its compile time grows with depth and every
bucket width (rows, or block tokens) compiles the whole depth again. A layer's code is the same at every depth; what
differs between layers is only static structure: how many committed AttnRes sources its routing reads, whether it
opens a block, whether its MLP input is captured and whether it signs (kotodama.model.signing). The regional path compiles one graph per such signature, with
every weight, norm gain, routing query, KV buffer and lever row passed in as a tensor (no module attribute is read
inside a region, so nothing guards on which layer's module it is), and runs the depth loop in Python. Every caller's
signatures get code objects of their own, so each code object holds one compiled graph and dynamo's per-code-object
recompile limit stays out of reach however many signatures and widths a server builds.

The regions run without CUDA graphs; on a CUDA engine the caller captures its whole forward (embedding, depth loop,
tail, sampler) as one CUDA graph (``StepGraph``), so a replay stays launch-free.

Used by kotodama.model.multi_decode (the row decoders) and kotodama.model.decode_engine (the single-stream step and
block forward).
"""

from __future__ import annotations

import types
from typing import Callable, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F

from kotodama.model.llama import RMSNorm, apply_rope
from kotodama.model.signing import stats_write


def layer_signatures(boundaries: Iterable[int], n_layers: int) -> list[tuple[int, bool]]:
    """Each effective layer's static decode structure: (committed sources its attention routing reads, whether it is a
    block boundary). Layer i reads the blocks committed before it, one per boundary below i."""
    bset = {int(b) for b in boundaries}
    out, committed = [], 0
    for i in range(n_layers):
        out.append((committed, i in bset))
        committed += i in bset
    return out


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """kotodama.model.llama.RMSNorm's forward, on its weight."""
    norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * norm * weight


def route_weights(sources: list[torch.Tensor], query: torch.Tensor, norm_weight: torch.Tensor,
                  eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """DecodeEngine._route's arithmetic, with eps passed in: (the routed mixture, the weights (n, B, S))."""
    qw = query * norm_weight
    stacked = torch.stack(sources, dim=0)
    rsqrt = torch.rsqrt(stacked.pow(2).mean(-1) + eps)
    logits = (stacked * qw).sum(-1) * rsqrt
    weights = F.softmax(logits, dim=0)
    return (weights.unsqueeze(-1) * stacked).sum(0), weights


def _route(sources: list[torch.Tensor], query: torch.Tensor, norm_weight: torch.Tensor, eps: float) -> torch.Tensor:
    return route_weights(sources, query, norm_weight, eps)[0]


def sign_write(sign: tuple, q: Optional[torch.Tensor], k: Optional[torch.Tensor], v: Optional[torch.Tensor],
               gate: Optional[torch.Tensor], w_attn: Optional[torch.Tensor], w_mlp: torch.Tensor) -> None:
    """A signing layer's last position into the sign buffers (kotodama.model.signing): sign = (route buffer, row ids,
    each row's slot, the layer's column (0-d), then the q, k, v, gate buffers and the sampled column, or Nones for a
    layer that writes its routing only, then the partial buffer and the layer's column in it (0-d), or Nones; the
    layer writes its output partial there itself). q / k / v are (B, heads, S, head_dim) before RoPE, gate
    (B, S, intermediate), the routing weights (n, B, S) (``w_attn`` None: the attention input reads no routing)."""
    route_buf, row_ids, slot, layer_col, q_buf, k_buf, v_buf, gate_buf, sampled_col = sign[:9]
    B = w_mlp.shape[1]
    width = route_buf.shape[-1] // 2
    halves = []
    for w in (w_attn, w_mlp):
        n = 0 if w is None else w.shape[0]
        if n:
            halves.append(w[:, :, -1].transpose(0, 1).to(route_buf.dtype))
        if n < width:
            halves.append(route_buf.new_zeros((B, width - n)))
    route_buf.index_put_((row_ids, slot, layer_col.expand(B)), torch.cat(halves, dim=-1))
    if q_buf is None:
        return
    idx = (row_ids, slot, sampled_col.expand(B))
    q_buf.index_put_(idx, q[:, :, -1].reshape(B, -1))
    k_buf.index_put_(idx, k[:, :, -1].reshape(B, -1))
    v_buf.index_put_(idx, v[:, :, -1].reshape(B, -1))
    gate_buf.index_put_(idx, gate[:, -1])


def cache_attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv: tuple[torch.Tensor, ...],
                 state: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """DecodeEngine's attention on passed tensors (its decode step and its block forward): the S new keys and values
    are written at ``positions`` of the one-row cache, then every query attends the whole cache under ``mask``.
    kv = (k_cache, v_cache), state = (mask, positions). Returns (1, S, n_heads * head_dim)."""
    B, _, S, _ = q.shape
    k_cache, v_cache = kv
    mask, positions = state
    k_cache.index_copy_(2, positions, k)
    v_cache.index_copy_(2, positions, v)
    out = F.scaled_dot_product_attention(q, k_cache, v_cache, attn_mask=mask, enable_gqa=True)
    return out.transpose(1, 2).contiguous().view(B, S, -1)


def block_attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv: tuple[torch.Tensor, ...],
                 state: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """DecodeEngine's block attention (prefill and extend chunks), ``cache_attend``'s contract with the queries at
    consecutive ``positions``. On CUDA in bf16/fp16 with flash-attn, the causal KV-cache kernel with one split
    (kotodama.model.cascade_attn.prefill_flash): every row comes out with the same bytes whatever chunk holds it,
    so a prompt's KV does not depend on how it was chunked or what a cache held. Elsewhere, ``cache_attend``."""
    from kotodama.model.cascade_attn import flash_available, prefill_flash

    if not (q.is_cuda and q.dtype in (torch.bfloat16, torch.float16) and flash_available()):
        return cache_attend(q, k, v, kv, state)
    B, _, S, _ = q.shape
    k_cache, v_cache = kv
    _, positions = state
    k_cache.index_copy_(2, positions, k)
    v_cache.index_copy_(2, positions, v)
    lens = (positions[-1:] + 1).to(torch.int32)
    out = prefill_flash(q.transpose(1, 2), k_cache.transpose(1, 2), v_cache.transpose(1, 2), lens)
    return out.reshape(B, S, -1)


def make_layer(name: str, *, is_boundary: bool, capture: bool, eps: float, n_heads: int, n_kv_heads: int,
                       head_dim: int, intermediate: int, qk_norm: bool, levers: bool, steer: bool,
                       attend: Callable, linear: Callable = F.linear, sign: int = 0, stats: int = 0,
                       probs: Optional[Callable] = None, stats_fn: Optional[Callable] = None) -> Callable:
    """One layer as a function of tensors, mirroring the loop body of MultiSampleDecoder._forward_step and of
    DecodeEngine's step and block forwards: pre-attention routing over ``committed + [partial]``, the attention
    sublayer (KV write and attend through ``attend(q, k, v, kv, state)``), the lever and broadcast-steering adds,
    pre-MLP routing, the MLP (its activation rewritten by the act lever) and its lever add. A boundary layer commits ``partial`` (the caller appends it to its list
    after the call) and starts the new block from zero. ``partial`` is (B, S, D): B rows of one token, or one row of
    S tokens; ``cos`` / ``sin`` are the rope rows of its positions, as apply_rope takes them; ``linear(x, w)`` is the
    projection (the engine's GEMV dispatch).

    Returns the layer's output partial, and with ``capture`` also the MLP input (B, S, D). With ``sign`` the layer
    writes its routing (1) and a sampled layer's surfaces too (2) through ``sign_write`` (``sign_args``: its sign
    tuple); the arithmetic of everything it returns is the same as without. With ``stats`` (1: every layer's statistics, 2: a sampled layer's too) the layer
    writes its attention statistics through kotodama.model.signing.stats_write from ``probs(q, kv, state)`` (the
    attention's probabilities in absolute column order; ``stats_args``: its stats tuple), or with ``stats_fn`` through
    ``stats_fn(stats_args, q, kv, state, sampled, lse)``, which writes the same statistics itself (a fused kernel)
    given the attention's per-row log-normalisers (``attend(..., lse=True)`` returns them beside its output). The function gets a code
    object of its own named ``name``, so its compiled graphs are cached apart from every other signature's."""
    q_dim, kv_dim = n_heads * head_dim, n_kv_heads * head_dim

    def decode_layer(partial, committed, attn_q, attn_g, mlp_q, mlp_g, attn_norm_w, ffn_norm_w, q_norm_w, k_norm_w,
                     w_qkv, w_o, w_gate_up, w_down, cos, sin, kv, state, lever_a, lever_m, act_keep, act_add,
                     steer_row, sign_args=None, stats_args=None):
        w_attn = None
        if committed:
            h_attn, w_attn = route_weights(committed + [partial], attn_q, attn_g, eps)
        else:
            h_attn = partial
        if is_boundary:
            sources = committed + [partial]
            base = torch.zeros_like(partial)
        else:
            sources = committed
            base = partial
        x = _rms(h_attn, attn_norm_w, eps)
        B, S = x.shape[0], x.shape[1]
        qkv = linear(x, w_qkv)
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.view(B, S, n_heads, head_dim).transpose(1, 2)
        k = k.view(B, S, n_kv_heads, head_dim).transpose(1, 2)
        v = v.view(B, S, n_kv_heads, head_dim).transpose(1, 2)
        if qk_norm:
            q = _rms(q, q_norm_w, eps)
            k = _rms(k, k_norm_w, eps)
        q_pre, k_pre = q, k
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        lse = None
        if stats and stats_fn is not None:
            attended, lse = attend(q, k, v, kv, state, lse=True)
        else:
            attended = attend(q, k, v, kv, state)
        if stats and stats_fn is None:
            stats_write(stats_args, probs(q, kv, state), stats == 2)
        attn_out = linear(attended, w_o)
        if levers:
            attn_out = attn_out + lever_a.unsqueeze(1)
        new = base + attn_out
        if steer:
            new = new + steer_row
        h_mlp, w_mlp = route_weights(sources + [new], mlp_q, mlp_g, eps)
        x_mlp = _rms(h_mlp, ffn_norm_w, eps)
        gate, up = linear(x_mlp, w_gate_up).split(intermediate, dim=-1)
        if sign:
            sign_write(sign_args, q_pre, k_pre, v, gate, w_attn, w_mlp)
        act = F.silu(gate) * up
        if levers:
            act = act * act_keep.unsqueeze(1) + act_add.unsqueeze(1)
        mlp_out = linear(act, w_down)
        if levers:
            mlp_out = mlp_out + lever_m.unsqueeze(1)
        out = new + mlp_out
        if sign and sign_args[9] is not None:  # the output partial (the depth path, or a sampled layer's hidden)
            sign_args[9][sign_args[1], sign_args[2], sign_args[10].expand(B)] = out[:, -1]
        if lse is not None:
            # last: a fused kernel is opaque to the compiler, and launched mid-layer it would split the fusions of
            # everything after it (the layer's arithmetic would then differ from the unsigned layer's)
            stats_fn(stats_args, q, kv, state, stats == 2, lse)
        if capture:
            return out, x_mlp
        return out

    code = decode_layer.__code__.replace(co_name=name, co_qualname=name)
    return types.FunctionType(code, decode_layer.__globals__, name, decode_layer.__defaults__,
                              decode_layer.__closure__)


def own_code(fn: Callable, name: str) -> Callable:
    """``fn`` behind a function with a code object of its own named ``name``: torch.compile caches graphs per code
    object, so a bound method compiled for many instances (each decoder's sampler and tail) would otherwise pile one
    cache entry per instance onto one code object."""
    def call(*args):
        return fn(*args)

    code = call.__code__.replace(co_name=name, co_qualname=name)
    return types.FunctionType(code, call.__globals__, name, None, call.__closure__)


# The region mode for each whole-step mode: the same autotuning, without CUDA graphs (the whole step is captured as
# one graph around the regions instead).
REGION_MODES = {"reduce-overhead": "default", "max-autotune": "max-autotune-no-cudagraphs"}


def region_options(mode: str, emulate_casts: bool = False) -> dict:
    """The inductor options of a region compiled for whole-step ``mode``: its region mode's (REGION_MODES), and with
    ``emulate_casts`` intermediates rounded to their dtype where eager rounds them (``emulate_precision_casts``).
    Inductor otherwise keeps a fused chain's intermediates in fp32, so where a fusion boundary falls changes the
    arithmetic, and a signing layer's extra stores move fusion boundaries: with the rounding fixed, a signing decoder
    computes what an unsigned one does, bit for bit, under one kernel pin table
    (scripts/benchmark/check_signed_generation.py). It costs the 1B's 128-row step ~4%."""
    import torch._inductor

    opts = dict(torch._inductor.list_mode_options(REGION_MODES.get(mode, mode)))
    if emulate_casts:
        opts["emulate_precision_casts"] = True
    return opts


class StepGraph:
    """A whole decode step replayed as one CUDA graph. The first ``warmup`` calls run the step eagerly on a side
    stream (each is a real step: it compiles the regions and settles their workspaces), the next call captures the
    step and replays it once, and every later call replays it. Every tensor the step reads or writes lives at a fixed
    address (the decoder's and engine's buffers), so a replay is the step on whatever those buffers hold now; the
    step's result is one of those buffers. ``tokens`` is the input buffer the graph reads; a call with
    another tensor copies it in first."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor], tokens: torch.Tensor, pool, warmup: int = 2) -> None:
        self.fn, self.tokens, self.pool, self.warmup = fn, tokens, pool, int(warmup)
        self.calls = 0
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.out: Optional[torch.Tensor] = None

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.data_ptr() != self.tokens.data_ptr():  # the graph reads its own buffer: put the input there
            self.tokens.copy_(tokens)
            tokens = self.tokens
        if self.graph is not None:
            self.graph.replay()
            return self.out
        if self.calls < self.warmup:
            self.calls += 1
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                out = self.fn(tokens)
            torch.cuda.current_stream().wait_stream(side)
            return out
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            out = self.fn(tokens)
        self.graph, self.out = graph, out
        graph.replay()
        return out


def check_norms(model, eps: float) -> None:
    """The regions read every norm as kotodama RMSNorm's arithmetic on its weight; any other norm class (Liger's
    kernels) would compute something else, so it is refused."""
    norms = [model.norm, model.final_res_norm]
    for layer in model.layers:
        norms += [layer.attn_norm, layer.ffn_norm, layer.attn_res_norm, layer.mlp_res_norm]
        if layer.attn.qk_norm:
            norms += [layer.attn.q_norm, layer.attn.k_norm]
    bad = {type(n).__name__ for n in norms if type(n) is not RMSNorm or n.eps != eps}
    if bad:
        raise ValueError(f"regional decode needs kotodama RMSNorm norms at eps {eps}; the model has {sorted(bad)} "
                         "(build it with use_liger=False)")


Region = tuple[Callable, bool, int, tuple, tuple, Optional[tuple], Optional[tuple]]


def build_regions(eng, signatures: Sequence[tuple[int, bool, bool, int, int]],
                  kvs: Sequence[tuple[torch.Tensor, ...]], prefix: str, *, attend: Callable, steer: bool,
                  linear: Callable = F.linear, capture_col: Optional[Callable[[int], int]] = None,
                  sign_args: Optional[Callable[[int], tuple]] = None, probs: Optional[Callable] = None,
                  stats_args: Optional[Callable[[int], tuple]] = None, stats_fn: Optional[Callable] = None,
                  **opts) -> tuple[list[Region], int]:
    """One compiled layer per distinct signature (committed sources, boundary, capture, sign, stats) over a
    DecodeEngine's weights, and each effective layer's (compiled layer, boundary, capture column or -1, its tensors,
    its KV buffers, its sign tuple or None, its stats tuple or None). The compiled functions are named
    ``{prefix}_c{committed}[_boundary][_capture][_sign{level}][_stats{level}]``. ``opts`` go to torch.compile.
    ``capture_col(layer)``: a captured layer's column in the caller's capture buffer; ``sign_args(layer)``: a signing
    layer's sign tuple (``sign_write``); ``probs`` / ``stats_args(layer)``: the attention probabilities and a layer's
    stats tuple (``make_layer``). Returns (regions, number of distinct signatures)."""
    model = eng.model
    static = dict(eps=eng.eps, n_heads=eng.n_heads, n_kv_heads=eng.n_kv_heads, head_dim=eng.head_dim,
                  intermediate=eng.config.intermediate_size, qk_norm=bool(eng.config.qk_norm),
                  levers=eng.levers is not None, steer=bool(steer), attend=attend, linear=linear)
    fns: dict[tuple[int, bool, bool], Callable] = {}
    regions: list[Region] = []
    for li, sig in enumerate(signatures):
        c, boundary, capture, sign, stats = sig
        if sig not in fns:
            name = (f"{prefix}_c{c}{'_boundary' if boundary else ''}{'_capture' if capture else ''}"
                    f"{f'_sign{sign}' if sign else ''}{f'_stats{stats}' if stats else ''}")
            fns[sig] = torch.compile(make_layer(name, is_boundary=boundary, capture=capture, sign=sign, stats=stats,
                                                probs=probs, stats_fn=stats_fn, **static), **opts)
        layer = model.layers[li]
        attn = layer.attn
        tensors = (layer.attn_res_query, layer.attn_res_norm.weight, layer.mlp_res_query, layer.mlp_res_norm.weight,
                   layer.attn_norm.weight, layer.ffn_norm.weight,
                   attn.q_norm.weight if attn.qk_norm else None, attn.k_norm.weight if attn.qk_norm else None,
                   eng.w_qkvs[li], attn.o_proj.weight, eng.w_gate_ups[li], layer.ffn.down_proj.weight)
        col = capture_col(li) if capture else -1
        regions.append((fns[sig], boundary, col, tensors, kvs[li], sign_args(li) if sign else None,
                        stats_args(li) if stats else None))
    return regions, len(fns)


def run_regions(regions: Sequence[Region], embed: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                state: tuple[torch.Tensor, ...], levers: Optional[tuple[torch.Tensor, ...]],
                steer_buf: Optional[torch.Tensor],
                capture_buf: Optional[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor]:
    """The depth loop over ``regions`` from the embedding: each layer's lever rows (``levers``: attn, mlp, act keep,
    act add, each indexed by layer), broadcast steering row (``steer_buf[li]``) and, for a captured layer, its MLP
    input written to ``capture_buf[:, col]`` (the last position's). Returns (the committed block outputs, the last
    partial) for the tail."""
    committed: list[torch.Tensor] = []
    partial = embed
    for li, (fn, boundary, col, tensors, kv, sign, stats) in enumerate(regions):
        rows = (None,) * 4 if levers is None else tuple(t[li] for t in levers)
        steer_row = steer_buf[li] if steer_buf is not None else None
        if sign is None and stats is None:
            out = fn(partial, committed, *tensors, cos, sin, kv, state, *rows, steer_row)
        else:
            out = fn(partial, committed, *tensors, cos, sin, kv, state, *rows, steer_row, sign, stats)
        if col >= 0:
            out, x_mlp = out
            capture_buf[:, col] = x_mlp[:, -1]
        if boundary:
            committed.append(partial)
        partial = out
    return committed, partial
