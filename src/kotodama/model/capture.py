"""Instrumented AttnRes forward passes for extraction and readout.

Two layers, one convention:

* ``instrumented_forward`` — a faithful mirror of ``KotodamaModel._forward_attn_res``
  (non-cached, eval) that runs attention eagerly and captures every surface:
  pre-RoPE keys/queries (k_norm/q_norm outputs), values, softmax weights, gate
  pre-activations, the running residual after each sublayer, committed block
  snapshots, and per-token AttnRes routing weights. Gate: its logits match
  ``model(input_ids)["logits"]``.
* ``KotoKVCache`` / ``cached_forward`` / ``generate_light`` — the KV-cached
  extension (prefix reuse, windowed capture, light generation), bit-exact against
  the uncached walker (``tests/test_capture.py``).

The ``replay_extract_*`` adapters at the bottom speak the anamnesis banking
convention and import anamnesis lazily; the rest is pure torch. Notes on the
cached extension sit above ``KotoKVCache``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from kotodama.model.llama import apply_rope


# ── the uncached walker (convention of record) ─────────────────────────────────

def _route_capture(committed_stack, n_committed, partial, query, norm):
    """PyTorch AttnRes routing (matches `_route_p12` fallback) + return per-token weights.
    Returns (routed_output, weights[(n_committed+1, B, T)] | None)."""
    if n_committed == 0:
        return partial, None
    qw = query * norm.weight
    eps = norm.eps
    rsqrt_c = torch.rsqrt(committed_stack.pow(2).mean(-1) + eps)          # (n_c, B, T)
    logits_c = (committed_stack * qw).sum(-1) * rsqrt_c                   # (n_c, B, T)
    rsqrt_p = torch.rsqrt(partial.pow(2).mean(-1, keepdim=True) + eps)
    logit_p = ((partial * qw).sum(-1, keepdim=True) * rsqrt_p).squeeze(-1)  # (B, T)
    all_logits = torch.cat([logits_c, logit_p.unsqueeze(0)], dim=0)       # (n_c+1, B, T)
    weights = torch.softmax(all_logits, dim=0)
    all_src = torch.cat([committed_stack, partial.unsqueeze(0)], dim=0)
    return (weights.unsqueeze(-1) * all_src).sum(0), weights


def _eager_attention(attn, x_normed, rope_cos, rope_sin):
    """Eager GQA attention matching `_forward_sdpa` math; returns (out, k_pre_rope, v, q_pre_rope, weights).
    k/v/q captured per-head pre-RoPE/operative; weights = softmax(QKᵀ/√d), fp32 softmax (SDPA parity)."""
    B, S, _ = x_normed.shape
    nh, nkv, hd = attn.num_heads, attn.num_kv_heads, attn.head_dim
    q = attn.q_proj(x_normed).view(B, S, nh, hd).transpose(1, 2)          # (B, nh, S, hd)
    k = attn.k_proj(x_normed).view(B, S, nkv, hd).transpose(1, 2)         # (B, nkv, S, hd)
    v = attn.v_proj(x_normed).view(B, S, nkv, hd).transpose(1, 2)
    if attn.qk_norm:
        q = attn.q_norm(q)
        k = attn.k_norm(k)                                                # operative pre-RoPE key
    cap_k, cap_v, cap_q = k.detach(), v.detach(), q.detach()
    cos, sin = rope_cos[:S], rope_sin[:S]
    qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    g = nh // nkv
    kr_e = kr.repeat_interleave(g, dim=1)                                 # GQA expand → (B, nh, S, hd)
    v_e = v.repeat_interleave(g, dim=1)
    scores = (qr @ kr_e.transpose(-1, -2)) / math.sqrt(hd)               # (B, nh, S, S)
    causal = torch.triu(torch.ones(S, S, dtype=torch.bool, device=scores.device), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    weights = torch.softmax(scores.float(), dim=-1).to(v.dtype)          # fp32 softmax (SDPA parity)
    out = (weights @ v_e).transpose(1, 2).contiguous().view(B, S, nh * hd)
    out = attn.o_proj(out)
    return out, cap_k, cap_v, cap_q, weights.detach()


def _sdpa_attention(attn, x_normed, rope_cos, rope_sin):
    """Light attention: SDPA (no softmax-weight materialization) — returns ONLY `out`. Mathematically equals
    `_eager_attention`'s output (fp aside); used by `light=True` (calibration / residual-only) to skip the
    O(S²) weight capture + per-head k/v/q transfers."""
    B, S, _ = x_normed.shape
    nh, nkv, hd = attn.num_heads, attn.num_kv_heads, attn.head_dim
    q = attn.q_proj(x_normed).view(B, S, nh, hd).transpose(1, 2)
    k = attn.k_proj(x_normed).view(B, S, nkv, hd).transpose(1, 2)
    v = attn.v_proj(x_normed).view(B, S, nkv, hd).transpose(1, 2)
    if attn.qk_norm:
        q = attn.q_norm(q)
        k = attn.k_norm(k)
    cos, sin = rope_cos[:S], rope_sin[:S]
    qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    g = nh // nkv
    out = F.scaled_dot_product_attention(qr, kr.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1),
                                         is_causal=True)
    return attn.o_proj(out.transpose(1, 2).contiguous().view(B, S, nh * hd))


@torch.no_grad()
def instrumented_forward(model, input_ids, light=False, routing_only=False):
    """Faithful mirror of `_forward_attn_res` (eval/non-cached/non-batched) with eager attn + full capture.
    `light=True`: SDPA attn + capture ONLY residual_partial + committed (skip keys/values/queries/attn/gate/
    routing) — for positional-mean calibration where only the residual is needed (≈10× faster, low-memory).
    `routing_only=True`: SDPA attn + capture ONLY `routing` — for layer × source routing reads over many
    tokens (no O(S²) attention maps held)."""
    if light and routing_only:
        raise ValueError("light and routing_only capture disjoint surfaces; pick one")
    keep_routing = not light
    full = not light and not routing_only
    layers = model.layers
    rope_cos, rope_sin = model.rope_cos, model.rope_sin
    block_ranges = model._attn_res_block_ranges
    cap = {k: [] for k in ("keys", "values", "queries", "attn", "gate",
                           "residual_partial", "committed", "routing")}

    embed = model.embed_tokens(input_ids)
    committed: list[torch.Tensor] = []
    committed_stack = None
    partial = embed

    for (block_start, block_end) in block_ranges:
        n_committed = len(committed)
        if n_committed == 0:
            h_attn = partial
        else:
            h_attn, w = _route_capture(committed_stack, n_committed, partial,
                                       layers[block_start].attn_res_query, layers[block_start].attn_res_norm)
            if keep_routing:
                cap["routing"].append(("pre_attn", block_start, w.detach()))
        # commit at boundary
        committed.append(partial)
        if not routing_only:
            cap["committed"].append(partial.detach())
        partial = torch.zeros_like(embed)
        committed_stack = torch.stack(committed, dim=0)
        n_committed = len(committed)

        for i in range(block_start, block_end):
            lyr = layers[i]
            if i != block_start:
                h_attn, w = _route_capture(committed_stack, n_committed, partial,
                                           lyr.attn_res_query, lyr.attn_res_norm)
                if keep_routing:
                    cap["routing"].append(("pre_attn", i, w.detach()))
            if not full:
                attn_out = _sdpa_attention(lyr.attn, lyr.attn_norm(h_attn), rope_cos, rope_sin)
            else:
                attn_out, ck, cv, cq, aw = _eager_attention(lyr.attn, lyr.attn_norm(h_attn), rope_cos, rope_sin)
                cap["keys"].append(ck); cap["values"].append(cv); cap["queries"].append(cq); cap["attn"].append(aw)
            partial = partial + attn_out
            if not routing_only:
                cap["residual_partial"].append(partial.detach())

            h_mlp, w = _route_capture(committed_stack, n_committed, partial,
                                      lyr.mlp_res_query, lyr.mlp_res_norm)
            ffn_in = lyr.ffn_norm(h_mlp)
            if keep_routing:
                cap["routing"].append(("pre_mlp", i, w.detach()))
            if full:
                cap["gate"].append(lyr.ffn.gate_proj(ffn_in).detach())       # pre-SiLU gate
            partial = partial + lyr.ffn(ffn_in)
            if not routing_only:
                cap["residual_partial"].append(partial.detach())

    h_final, w = _route_capture(committed_stack, len(committed), partial,
                                model.final_res_query, model.final_res_norm)
    if keep_routing:
        cap["routing"].append(("final", -1, w.detach()))
    logits = F.linear(model.norm(h_final), model.get_lm_head_weight())
    return logits, cap


# ── the KV-cached extension ────────────────────────────────────────────────────
#
# Cached instrumented AttnRes forward for kotodama KotodamaModel — the koto
# replay surface for signature capture.
#
# Extends `instrumented_forward` (the convention of record for
# every banked koto signature) with the two things a replay battery needs
# beyond the uncached walker:
#
#   1. **KV-cached teacher-forced continuation** — prefill a prefix (no capture),
#      then run the continuation tokens in ONE forward that attends over the
#      cache, capturing every surface for the continuation positions only. This
#      mirrors `anamnesis.extraction.replay_cached.replay_extract_cached`'s
#      banking contract exactly (T = n-1 steps, positions offset..offset+n-2),
#      which is what the cached-vs-full gate (cached == full) and the
#      truncated-prefix gate (a truncated prefix MUST differ) check.
#   2. **HF-layout generation** (`generate_light`) — the light/SDPA path driving
#      a sampling loop and returning `hidden_states` in the HF generate layout
#      (prefill tuple + one per-step tuple), so a positional-means calibration
#      written against HF models runs on koto checkpoints unchanged.
#
# WHAT "HIDDEN STATE" MEANS HERE — the koto convention of record
# (`hidden_rows`): index 0 = the embedding; index l+1 = the
# post-MLP running `partial` of layer l. AttnRes RESETS the partial at committed
# block boundaries, so this is NOT a monotone global residual; every koto
# signature and the koto calibration share this convention, which is what makes
# them comparable to each other.
#
# CACHE CONVENTION — the model's own (`GQAttention._forward_sdpa`, read at
# source): per layer a `(k, v)` tuple with k post-QK-norm and rotated at its
# absolute position, v raw, both `(B, n_kv, S, hd)`. AttnRes routing state
# (committed/partial) is POSITION-LOCAL and is NOT cached: continuation tokens
# compute their own block structure; only attention reads back through the KV
# cache. That is the whole reason a cached AttnRes replay is tractable.
#
# CAPTURED SURFACES per continuation position (capture=True):
#   keys/queries  k_norm/q_norm output, PRE-RoPE, per-head  (operative content
#                 key — a raw k_proj hook would read pre-norm keys)
#   values        v_proj output, per-KV-head
#   attn          eager softmax(QKᵀ/√d) rows over cache+self, fp32 softmax
#   gate          ffn.gate_proj output (pre-SiLU)
#   attn_out      o_proj output (the attention sublayer's residual contribution)
#                 — anamnesis's `attn_outputs` family reads it
#   residual      running partial after EACH sublayer + committed snapshots
#   routing       per-token AttnRes routing softmax at every routing point
#
# Faithfulness gates (tests/test_capture.py, CPU, tiny config):
# cached-with-empty-cache == `instrumented_forward` bit-for-bit surfaces;
# prefill+continue logits == full-forward logits at the continuation positions;
# and the same equality for every captured surface. The same equalities are
# re-checked on the real checkpoint and GPU before a capture run is used.
#
# Determinism: every capture entry point asserts `CUBLAS_WORKSPACE_CONFIG` is
# set — an unset workspace is a build/device default, not a pinnable identity.
#


class KotoCaptureError(RuntimeError):
    """Raised when the cached capture cannot be wired safely."""


def assert_lane_pinned() -> dict[str, str]:
    """Refuse to capture under an unpinned arithmetic lane.

    Returns the lane identity to be recorded in provenance: torch version,
    CUDA version, device capability, and the workspace pin.
    """
    ws = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if not ws:
        raise KotoCaptureError(
            "CUBLAS_WORKSPACE_CONFIG is not set. An unset workspace is a "
            "build/device default, not a pinnable identity, and on this "
            "hardware the two known values differ by more than any tolerance "
            "anyone will set. Export "
            "CUBLAS_WORKSPACE_CONFIG=:16:8 (the standing pin) before any "
            "capture."
        )
    ident = {"cublas_workspace": ws, "torch": torch.__version__}
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        ident["cuda"] = str(torch.version.cuda)
        ident["sm"] = f"sm_{cap[0]}{cap[1]}"
        ident["device_name"] = torch.cuda.get_device_name()
    return ident


@dataclass
class KotoKVCache:
    """Per-layer (k, v): k post-QK-norm AND rotated, v raw, (B, n_kv, S, hd).

    The model's own cache convention. `length` is the one thing the
    replay contract reads (`cache_length`).
    """

    layers: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)

    @property
    def length(self) -> int:
        if not self.layers:
            raise KotoCaptureError("cache has no layers")
        return int(self.layers[0][0].shape[-2])

    @property
    def batch(self) -> int:
        if not self.layers:
            raise KotoCaptureError("cache has no layers")
        return int(self.layers[0][0].shape[0])


def _rope_rows(model, start: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE table rows for absolute positions start..start+n-1.

    The tables are built to `max_position_embeddings`; walking past the end is
    the RoPE analog of reading uninitialized memory, so it is a refusal.
    """
    cos, sin = model.rope_cos, model.rope_sin
    if start + n > cos.shape[0]:
        raise KotoCaptureError(
            f"positions {start}..{start + n - 1} exceed the RoPE table "
            f"({cos.shape[0]} rows = max_position_embeddings). The koto "
            "context ceiling is a hard fact of the body, not a tunable."
        )
    return cos[start : start + n], sin[start : start + n]


def _cont_mask(n: int, cache_len: int, device) -> torch.Tensor | None:
    """Bool mask (n, cache_len+n): continuation token i sees cache fully and
    itself + earlier continuation tokens (True = MASKED, additive -inf sites).

    Built explicitly because SDPA's `is_causal` aligns TOP-LEFT when
    q_len != kv_len, which silently blinds later continuation tokens to the
    cache tail — the well-formed-wrong-numbers failure mode.
    """
    if cache_len == 0:
        return None  # square case: plain causal handled by the caller
    full = torch.zeros(n, cache_len + n, dtype=torch.bool, device=device)
    full[:, cache_len:] = torch.triu(
        torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1
    )
    return full


def _attn_cached(
    attn,
    x_normed: torch.Tensor,
    cos_rows: torch.Tensor,
    sin_rows: torch.Tensor,
    past: tuple[torch.Tensor, torch.Tensor] | None,
    eager: bool,
    keep_caps: bool = True,
):
    """One attention sublayer over cache+self. Returns
    (out, new_kv, cap) where cap is None on the light path (or when
    `keep_caps=False`) and otherwise (k_pre_rope, v, q_pre_rope, weights)
    for the NEW positions only.

    Mirrors `_forward_sdpa`'s math (QK-norm -> RoPE -> concat cache) and
    `instrumented_forward`'s eager capture semantics (pre-RoPE post-norm k/q,
    fp32 softmax weights).

    `eager=True, keep_caps=False` is the PARITY mode: eager arithmetic
    (identical to the full-replay reference, so a prefill built this way
    keeps deep-layer k/v on the same fp trajectory — what the cached-vs-full
    gate requires),
    with the O(S²) weights discarded immediately instead of banked.
    """
    B, S, _ = x_normed.shape
    nh, nkv, hd = attn.num_heads, attn.num_kv_heads, attn.head_dim
    q = attn.q_proj(x_normed).view(B, S, nh, hd).transpose(1, 2)
    k = attn.k_proj(x_normed).view(B, S, nkv, hd).transpose(1, 2)
    v = attn.v_proj(x_normed).view(B, S, nkv, hd).transpose(1, 2)
    if attn.qk_norm:
        q = attn.q_norm(q)
        k = attn.k_norm(k)
    cap_k, cap_v, cap_q = (
        (k.detach(), v.detach(), q.detach()) if (eager and keep_caps) else (None,) * 3
    )

    qr = apply_rope(q, cos_rows, sin_rows)
    kr = apply_rope(k, cos_rows, sin_rows)
    if past is not None:
        kr = torch.cat([past[0], kr], dim=2)
        v_all = torch.cat([past[1], v], dim=2)
    else:
        v_all = v
    new_kv = (kr.detach(), v_all.detach())
    cache_len = int(kr.shape[2]) - S

    if not eager:
        mask = _cont_mask(S, cache_len, x_normed.device)
        out = F.scaled_dot_product_attention(
            qr, kr, v_all,
            attn_mask=None if mask is None else ~mask.view(1, 1, S, cache_len + S),
            is_causal=(mask is None),
            enable_gqa=True,
        )
        out = out.transpose(1, 2).contiguous().view(B, S, nh * hd)
        return attn.o_proj(out), new_kv, None

    g = nh // nkv
    kr_e = kr.repeat_interleave(g, dim=1)
    v_e = v_all.repeat_interleave(g, dim=1)
    scores = (qr @ kr_e.transpose(-1, -2)) / math.sqrt(hd)
    mask = _cont_mask(S, cache_len, scores.device)
    if mask is None:
        mask = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=scores.device), diagonal=1
        )
    scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores.float(), dim=-1).to(v_all.dtype)  # fp32 softmax (SDPA parity)
    out = (weights @ v_e).transpose(1, 2).contiguous().view(B, S, nh * hd)
    if not keep_caps:
        return attn.o_proj(out), new_kv, None
    return attn.o_proj(out), new_kv, (cap_k, cap_v, cap_q, weights.detach())


@torch.no_grad()
def cached_forward(
    model,
    input_ids: torch.Tensor,
    past: KotoKVCache | None = None,
    position_offset: int = 0,
    light: bool = False,
    capture_hidden: bool = True,
    eager_nocapture: bool = False,
    attention_sink=None,
    capture_layers=None,
):
    """Instrumented AttnRes forward over `input_ids`, attending through `past`.

    With `past=None, position_offset=0, light=False` this is
    `instrumented_forward` plus the cache/attn_out additions — the equality is
    asserted by the tests, not assumed.

    `eager_nocapture=True` is the PARITY-PREFILL mode (an SDPA prefill puts
    deep-layer k/v on a different fp trajectory than the eager full-replay
    reference, far enough to fail the cached-vs-full gate on deep PCA
    features): eager arithmetic
    identical to the capture path, per-token caps skipped and the O(S²)
    weights freed per layer, so a cache built this way is on the SAME fp
    trajectory as `replay_extract_koto`'s forward. `light` must be False
    with it.

    `attention_sink(layer, weights)`, when given, receives each layer's
    attention weights `(B, heads, S, cache + S)` as soon as they are computed,
    and `cap["attn"]` holds None for every layer instead: the all-layer
    weights never coexist (an on-device reduction consumes them per layer).
    `capture_layers`, when given, keeps keys, values, queries, gate
    pre-activations and attention outputs only at those layers; the others
    hold None, so every list stays indexed by layer. Routing, committed blocks
    and residual snapshots are kept for every layer either way. With neither
    given the capture is exactly the default one.

    Returns (logits, cap, new_cache):
      logits    (B, S, V) for the input positions
      cap       dict of captured surfaces for the INPUT positions only
                (light=True or eager_nocapture=True: residual_partial +
                committed only, and only when `capture_hidden`)
      new_cache KotoKVCache extended by these positions
    """
    if light and eager_nocapture:
        raise KotoCaptureError("light and eager_nocapture are mutually exclusive")
    if past is not None and position_offset < past.length:
        raise KotoCaptureError(
            f"position_offset {position_offset} < cache length {past.length}: "
            "continuation positions would overlap the cache. (offset > length "
            "is ALLOWED, mirroring anamnesis — it is exactly what the "
            "truncated-prefix gate injects, and the whole point of that "
            "gate is that such a cache must change the reading, not refuse.)"
        )
    capture = (not light) and (not eager_nocapture)
    layers = model.layers
    block_ranges = model._attn_res_block_ranges
    B, S = input_ids.shape
    cos_rows, sin_rows = _rope_rows(model, position_offset, S)

    cap: dict[str, list] = {
        k: []
        for k in (
            "keys", "values", "queries", "attn", "gate", "attn_out",
            "residual_partial", "committed", "routing",
        )
    }
    new_layers: list[tuple[torch.Tensor, torch.Tensor]] = []

    embed = model.embed_tokens(input_ids)
    committed: list[torch.Tensor] = []
    committed_stack = None
    partial = embed
    li = 0  # global layer index, for reading the right cache slot

    for (block_start, block_end) in block_ranges:
        n_committed = len(committed)
        if n_committed == 0:
            h_attn = partial
        else:
            h_attn, w = _route_capture(
                committed_stack, n_committed, partial,
                layers[block_start].attn_res_query, layers[block_start].attn_res_norm,
            )
            if capture:
                cap["routing"].append(("pre_attn", block_start, w.detach()))
        committed.append(partial)
        if capture_hidden or capture:
            cap["committed"].append(partial.detach())
        partial = torch.zeros_like(embed)
        committed_stack = torch.stack(committed, dim=0)
        n_committed = len(committed)

        for i in range(block_start, block_end):
            lyr = layers[i]
            if i != block_start:
                h_attn, w = _route_capture(
                    committed_stack, n_committed, partial,
                    lyr.attn_res_query, lyr.attn_res_norm,
                )
                if capture:
                    cap["routing"].append(("pre_attn", i, w.detach()))
            past_i = past.layers[li] if past is not None else None
            attn_out, new_kv, acap = _attn_cached(
                lyr.attn, lyr.attn_norm(h_attn), cos_rows, sin_rows, past_i,
                eager=not light, keep_caps=capture,
            )
            new_layers.append(new_kv)
            if acap is not None:
                ck, cv, cq, aw = acap
                keep = capture_layers is None or i in capture_layers
                cap["keys"].append(ck if keep else None)
                cap["values"].append(cv if keep else None)
                cap["queries"].append(cq if keep else None)
                if attention_sink is not None:
                    attention_sink(i, aw)
                    cap["attn"].append(None)
                else:
                    cap["attn"].append(aw)
                cap["attn_out"].append(attn_out.detach() if keep else None)
            partial = partial + attn_out
            if capture_hidden or capture:
                cap["residual_partial"].append(partial.detach())

            h_mlp, w = _route_capture(
                committed_stack, n_committed, partial,
                lyr.mlp_res_query, lyr.mlp_res_norm,
            )
            ffn_in = lyr.ffn_norm(h_mlp)
            if capture:
                cap["routing"].append(("pre_mlp", i, w.detach()))
                keep = capture_layers is None or i in capture_layers
                cap["gate"].append(lyr.ffn.gate_proj(ffn_in).detach() if keep else None)
            partial = partial + lyr.ffn(ffn_in)
            if capture_hidden or capture:
                cap["residual_partial"].append(partial.detach())
            li += 1

    h_final, w = _route_capture(
        committed_stack, len(committed), partial,
        model.final_res_query, model.final_res_norm,
    )
    if capture:
        cap["routing"].append(("final", -1, w.detach()))
    logits = F.linear(model.norm(h_final), model.get_lm_head_weight())
    return logits, cap, KotoKVCache(layers=new_layers)


def hidden_rows(cap: dict, num_layers: int) -> list[torch.Tensor]:
    """[L+1] tensors of (B, S, H): embed + post-MLP partial per layer — the
    koto hidden-state convention of record.
    """
    rp = cap["residual_partial"]
    if len(rp) != 2 * num_layers:
        raise KotoCaptureError(
            f"expected {2 * num_layers} residual snapshots (2/layer), got "
            f"{len(rp)} — capture_hidden was off, or the block walk changed"
        )
    return [cap["committed"][0]] + [rp[2 * l + 1] for l in range(num_layers)]


class _GenOut:
    """Duck-typed `generate` output: `.sequences` + HF-layout `.hidden_states`."""

    __slots__ = ("sequences", "hidden_states")

    def __init__(self, sequences, hidden_states):
        self.sequences = sequences
        self.hidden_states = hidden_states


@torch.no_grad()
def generate_light(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float | None = None,
    eos_token_id=None,
    output_hidden_states: bool = True,
):
    """Sampling loop on the light cached forward, HF `generate` layout.

    `hidden_states[0]` = prefill tuple(L+1) of (B, P, H); `hidden_states[t]`
    (t>=1) = tuple(L+1) of (B, 1, H) for the step whose INPUT is generated
    token t-1 at absolute position P+t-1 — exactly the layout
    `calibrate.run_calibration` indexes (`abs_pos = prompt_length + t - 1`).

    EOS handling matches calibrate's suppression contract: `eos_token_id=None`
    never stops early; a token id or list stops a SEQUENCE at first emission
    (whole-batch stop only when every row has stopped). Sampling is
    temperature -> nucleus(top_p) -> multinomial, seeded by the CALLER
    (`torch.manual_seed`), matching the HF semantics calibrate relies on.
    """
    L = int(model.config.num_layers)
    device = input_ids.device
    B, P = input_ids.shape
    if P < 1:
        raise KotoCaptureError("empty prompt")
    if isinstance(eos_token_id, int):
        eos_ids = [eos_token_id]
    else:
        eos_ids = list(eos_token_id) if eos_token_id else []

    logits, cap, cache = cached_forward(
        model, input_ids, past=None, position_offset=0, light=True,
        capture_hidden=output_hidden_states,
    )
    all_hidden = []
    if output_hidden_states:
        all_hidden.append(tuple(h for h in hidden_rows(cap, L)))

    seqs = input_ids
    alive = torch.ones(B, dtype=torch.bool, device=device)
    step_logits = logits[:, -1, :]

    for _ in range(max_new_tokens):
        if seqs.shape[1] >= model.rope_cos.shape[0]:
            break  # RoPE table exhausted — the context ceiling, not an error
        if do_sample:
            probs = torch.softmax(step_logits.float() / max(temperature, 1e-6), dim=-1)
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_p, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cum = torch.cumsum(sorted_p, dim=-1)
                cut = cum - sorted_p >= top_p  # keep tokens until mass >= top_p
                sorted_p = sorted_p.masked_fill(cut, 0.0)
                sorted_p = sorted_p / sorted_p.sum(-1, keepdim=True)
                pick = torch.multinomial(sorted_p, 1)
                nxt = sorted_idx.gather(-1, pick)
            else:
                nxt = torch.multinomial(probs, 1)
        else:
            nxt = step_logits.argmax(-1, keepdim=True)
        # A finished row keeps emitting its eos so the batch stays rectangular
        # (HF pads; constant-eos is equivalent for calibration, which only
        # reads hidden states, and those stop being accumulated per row is
        # NOT implemented — eos is suppressed in every koto recipe anyway).
        if eos_ids:
            done = ~alive
            if done.any():
                nxt[done] = eos_ids[0]
        seqs = torch.cat([seqs, nxt], dim=1)

        logits, cap, cache = cached_forward(
            model, nxt, past=cache, position_offset=seqs.shape[1] - 1, light=True,
            capture_hidden=output_hidden_states,
        )
        if output_hidden_states:
            all_hidden.append(tuple(h for h in hidden_rows(cap, L)))
        step_logits = logits[:, -1, :]
        if eos_ids:
            alive &= ~torch.isin(nxt.squeeze(1), torch.as_tensor(eos_ids, device=device))
            if not alive.any():
                break

    return _GenOut(sequences=seqs, hidden_states=all_hidden if output_hidden_states else None)


# ── replay contract (anamnesis banking convention) ──────────────────────────

def replay_extract_koto(
    model,
    full_token_ids,
    prompt_length: int,
    positional_means=None,
):
    """Full-document teacher-forced capture on the ANAMNESIS `replay_extract`
    banking convention: T = N-1 steps at positions P..P+N-2 (the state AT
    generated token g_i, whose logits predict g_{i+1}).

    NOTE this is deliberately NOT the P-1 slice (which starts one step
    earlier, at the prompt-boundary state): the cached-vs-full gate compares
    this function's output against the cached path, and both must live on the
    anamnesis convention or the gate compares different steps. Signatures
    banked on the P-1 slice are not step-aligned with these; a comparison
    across the two must say which slice each side is on.
    """
    import numpy as np

    from anamnesis.extraction.state_extractor import RawGenerationData

    assert_lane_pinned()
    device = next(model.parameters()).device
    ids = torch.as_tensor(full_token_ids, dtype=torch.long, device=device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.shape[0] != 1:
        raise KotoCaptureError(f"replay expects batch=1, got {ids.shape[0]}")
    S = int(ids.shape[1])
    P = int(prompt_length)
    N = S - P
    if P <= 0 or P >= S:
        raise KotoCaptureError(f"prompt_length {P} out of range for length {S}")
    if N < 2:
        raise KotoCaptureError(f"need >= 2 generated tokens, got N={N}")
    t_steps = N - 1

    logits, cap, _ = cached_forward(
        model, ids, past=None, position_offset=0, light=False, capture_hidden=True,
    )
    L = int(model.config.num_layers)

    def _np(t):
        return t.float().cpu().numpy()

    hr = hidden_rows(cap, L)
    hidden_states = [
        np.stack([_np(hr[l][0, P + i]) for l in range(L + 1)]) for i in range(t_steps)
    ]
    attentions = [
        np.stack([_np(cap["attn"][l][0, :, P + i, : P + i + 1]) for l in range(L)])
        for i in range(t_steps)
    ]
    logits_rows = [_np(logits[0, P + i]) for i in range(t_steps)]
    chosen = _np(ids[0, P + 1 : P + N]).astype(np.float32)

    positions = list(range(P, P + t_steps))
    return RawGenerationData(
        hidden_states=hidden_states,
        attentions=attentions,
        logits=logits_rows,
        chosen_token_ids=chosen,
        pre_rope_keys={
            l: [_np(cap["keys"][l][0, :, p, :]) for p in positions] for l in range(L)
        },
        prompt_length=P,
        positional_means=positional_means,
        gate_activations={
            l: [_np(cap["gate"][l][0, p, :]) for p in positions] for l in range(L)
        },
        v_proj_values={
            l: [_np(cap["values"][l][0, :, p, :]) for p in positions] for l in range(L)
        },
        queries={
            l: [_np(cap["queries"][l][0, :, p, :]) for p in positions] for l in range(L)
        },
        attn_outputs={
            l: [_np(cap["attn_out"][l][0, p, :]) for p in positions] for l in range(L)
        },
        attn_res_routing=[
            (tag, int(layer), _np(w[:, 0, :][:, positions]).T)
            for tag, layer, w in cap["routing"]
        ],
        attn_res_committed=[_np(c[0, positions, :]) for c in cap["committed"]],
    )


def capture_identity(model) -> dict:
    """What a kotodama capture's arithmetic is made of, for a reduction's lane id:
    this module's and the model code's source digests, the pinned lane, the
    weights' dtype, the AttnRes and loop geometry, the fp64-eps compat mode, and
    a merged LoRA adapter's identity when the model carries one
    (``adapter_identity``, set by the server's load): the adapter changes the
    weights, so its captures are a lane of their own, not the base model's."""
    import hashlib
    import os
    from pathlib import Path

    here = Path(__file__)
    cfg = model.config
    identity = dict(
        producer="kotodama.model.capture",
        source_sha256={
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (here, here.with_name("llama.py"))
        },
        lane=assert_lane_pinned(),
        dtype=str(next(model.parameters()).dtype),
        num_layers=int(cfg.num_layers),
        attn_res_boundaries=list(getattr(cfg, "attn_res_boundaries", None) or []),
        loop=[cfg.loop_start, cfg.loop_len, cfg.loop_times],
        rope_fp32=bool(getattr(cfg, "rope_fp32", False)),
        attnres_eps_fp64_compat=os.environ.get("KOTODAMA_ATTNRES_EPS_FP64_COMPAT", "0") == "1",
    )
    adapter = getattr(model, "adapter_identity", None)
    if adapter is not None:  # absent for a base model, whose lane id is unchanged
        identity["adapter"] = dict(adapter)
    return identity


@torch.no_grad()
def replay_features_gpu(model, full_token_ids, prompt_length: int, lane):
    """One span's anamnesis feature vector, captured and reduced on the model's device.

    The capture is :func:`replay_extract_koto`'s, tensor for tensor: one
    ``cached_forward`` over the whole row, the same span positions P..P+N-2, the
    same hidden-state convention (:func:`hidden_rows`), pre-RoPE keys and
    queries, values, pre-SiLU gates, routing weights and committed blocks. It is
    handed to anamnesis's fast-lane reducers (``ExternalSpan``) on the device
    instead of being copied to numpy per position: each layer's attention
    weights are reduced as the forward computes them and dropped, and keys,
    values, queries and gates are kept only at the lane's sampled layers.

    ``lane`` is an anamnesis ``GpuFeatureLane`` on the model's device (its
    configuration, calibration and schema). Returns anamnesis's
    ``GpuFeatureResult``: the vector (the fast lane's battery, then the
    attn_res family), names, mean log-probability and a receipt whose
    ``lane_id`` names this capture's identity.
    """
    from anamnesis.extraction.fast.external import ExternalSpan

    assert_lane_pinned()
    device = next(model.parameters()).device
    ids = torch.as_tensor(full_token_ids, dtype=torch.long, device=device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.shape[0] != 1:
        raise KotoCaptureError(f"replay expects batch=1, got {ids.shape[0]}")
    S = int(ids.shape[1])
    P = int(prompt_length)
    if P <= 0 or S - P < 2:
        raise KotoCaptureError(f"prompt_length {P} leaves fewer than 2 generated tokens in {S}")
    t = S - P - 1
    L = int(model.config.num_layers)
    sampled = list(lane.config.sampled_layers)
    span = ExternalSpan(lane, start=P, end=S, num_layers=L)
    logits, cap, _ = cached_forward(
        model, ids, past=None, position_offset=0, light=False, capture_hidden=True,
        attention_sink=span.attention, capture_layers=set(sampled),
    )
    hr = hidden_rows(cap, L)
    rows = slice(P, P + t)

    def heads(x):
        return x[0, :, rows, :].permute(1, 0, 2)

    return span.finish(
        hidden=torch.stack([hr[l + 1][0, rows] for l in range(L)]),
        keys={l: heads(cap["keys"][l]) for l in sampled},
        values={l: heads(cap["values"][l]) for l in sampled},
        queries={l: heads(cap["queries"][l]) for l in sampled},
        gates={l: cap["gate"][l][0, rows] for l in sampled},
        logits=logits[0, rows],
        token_ids=ids[0].tolist(),
        capture_identity=capture_identity(model),
        routing=[(tag, int(layer), w[:, 0, rows].T) for tag, layer, w in cap["routing"]],
        committed=[c[0, rows, :] for c in cap["committed"]],
    )


def replay_extract_cached_koto(
    loaded_model,
    cache: KotoKVCache,
    cont_ids,
    position_offset: int,
    positional_means=None,
):
    """Teacher-force `cont_ids` against a prefilled koto cache; return
    `RawGenerationData` on the anamnesis banking contract, plus the
    kotodama-native AttnRes fields.

    The contract, mirrored from `anamnesis.extraction.replay.cached`:
    T = n-1 steps for n continuation tokens; `logits[i]` is the
    distribution produced AT continuation index i (abs position offset+i);
    `chosen_token_ids = cont[1:n]`; attention row i keeps columns
    0..cache_len+i; `prompt_length = cache_len`. Positional means are
    remapped through anamnesis's own helper (identity here, because the koto
    path refuses offset != cache length by construction).
    """
    import numpy as np

    try:  # current anamnesis layout
        from anamnesis.extraction.replay.cached import remap_positional_means
    except ImportError:  # older layout: anamnesis.extraction.replay_cached
        from anamnesis.extraction.replay_cached import remap_positional_means
    from anamnesis.extraction.state_extractor import RawGenerationData

    assert_lane_pinned()
    model = loaded_model
    device = next(model.parameters()).device
    ids = torch.as_tensor(cont_ids, dtype=torch.long, device=device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.shape[0] != 1:
        raise KotoCaptureError(f"replay expects batch=1, got {ids.shape[0]}")
    n = int(ids.shape[1])
    if n < 2:
        raise KotoCaptureError(f"need >= 2 continuation tokens, got {n}")
    t_steps = n - 1
    cache_len = cache.length
    pm_eff = remap_positional_means(positional_means, cache_len, position_offset, t_steps)

    logits, cap, _ = cached_forward(
        model, ids, past=cache, position_offset=position_offset,
        light=False, capture_hidden=True,
    )
    L = int(model.config.num_layers)

    def _np(t):
        return t.float().cpu().numpy()

    hr = hidden_rows(cap, L)  # [L+1] of (1, n, H)
    hidden_states = [
        np.stack([_np(hr[l][0, i]) for l in range(L + 1)]) for i in range(t_steps)
    ]
    attentions = [
        np.stack([_np(cap["attn"][l][0, :, i, : cache_len + i + 1]) for l in range(L)])
        for i in range(t_steps)
    ]
    logits_rows = [_np(logits[0, i]) for i in range(t_steps)]
    chosen = _np(ids[0, 1:n]).astype(np.float32)

    def _per_layer_heads(key):
        return {
            l: [_np(cap[key][l][0, :, i, :]) for i in range(t_steps)]
            for l in range(L)
        }

    def _per_layer_seq(key):
        return {
            l: [_np(cap[key][l][0, i, :]) for i in range(t_steps)]
            for l in range(L)
        }

    positions = list(range(t_steps))
    routing = [
        (tag, int(layer), _np(w[:, 0, :][:, positions]).T)
        for tag, layer, w in cap["routing"]
    ]
    committed = [_np(c[0, positions, :]) for c in cap["committed"]]

    return RawGenerationData(
        hidden_states=hidden_states,
        attentions=attentions,
        logits=logits_rows,
        chosen_token_ids=chosen,
        pre_rope_keys=_per_layer_heads("keys"),
        prompt_length=cache_len,
        positional_means=pm_eff,
        gate_activations=_per_layer_seq("gate"),
        v_proj_values=_per_layer_heads("values"),
        queries=_per_layer_heads("queries"),
        attn_outputs=_per_layer_seq("attn_out"),
        attn_res_routing=routing,
        attn_res_committed=committed,
    )
