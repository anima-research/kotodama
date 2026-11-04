"""
Inference server for kotodama models (3B-first; one replica = one model on one GPU).

Usage:
    python -m kotodama.serve.server --checkpoint /path/to/model.pt [--prefix-cache]
    python -m kotodama.serve.server --checkpoint /path/to/instruct.pt --mode chat   # mode auto-detected
                                                                    # from 'instruct' in path
    python -m kotodama.serve.server --checkpoint ... --steer-npz algebra.npz,reppen.npz,binding.npz
                                        # per-request MULTI-SITE residual steering in the
                                        # fast path (per-key site from the _s{sublayer}
                                        # suffix; block-persistent; kotodama/serve/steering.py)

Two decode paths, deliberately only two:
  * fast (default)  — DecodeEngine: static KV cache, max-autotune compiled +
    CUDA-graphed step, all sampling laws in-graph (temperature, repetition
    penalty, top-k/top-p), compiled block prefill/extend, optional token-exact
    prefix cache. Single-stream decode is kernel-count bound (per-token cost
    grows with effective depth), so throughput comes from the batched row
    decoders (--multi_rows / --xbatch_rows; serve profiles turn them on).
    Startup pays a one-time compile per decode graph (minutes warm, much longer
    cold with many row buckets) — see gateway warmup timeouts.
  * reference       — plain eager llama.py loop. The parity oracle and debug
    path (~10x slower); NOT for production traffic.

The fleet story (kotodama.serve.gateway) runs many single-GPU replicas of this server;
concurrency comes from the fleet, not from batching within a replica.

Requires: fastapi, uvicorn, transformers (tokenizer only), torch
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import contextvars
import functools
import gc
import hashlib
import html
import json
import logging
import math
import os
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

# CPU threading: 2 threads is optimal for single-request GPU inference.
# cuDNN SDPA needs >= 2 threads; beyond that, overhead is kernel launch
# latency (not parallelizable). Avoids 128-thread default on large Xeons.
_SERVE_THREADS = int(os.environ.get("LUXIA_SERVE_THREADS", "2"))
os.environ.setdefault("OMP_NUM_THREADS", str(_SERVE_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_SERVE_THREADS))

# expandable_segments cuts allocator fragmentation under variable-length decode
# and bursty concurrency (reserved-but-unallocated headroom). Set before the
# torch import so it applies regardless of how the server is launched.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

torch.set_num_threads(_SERVE_THREADS)
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from transformers import AutoTokenizer

from kotodama import ckpt, presets
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig

# Fast decode engine (static cache + compiled/CUDA-graphed step). Optional.
try:
    from kotodama.model.decode_engine import DecodeEngine, SamplingParams as EngineSamplingParams
    from kotodama.model.multi_decode import CascadeDecoder, MultiSampleDecoder, decode_rows
    from kotodama.model.prefix_cache import PrefixKVCache
    from kotodama.model.regional import check_norms as regional_check_norms
    from kotodama.serve.xbatch import XBatchJob, XBatchRunner, XBatchScheduler
    _ENGINE_AVAILABLE = True
except Exception as _engine_exc:
    DecodeEngine = None  # type: ignore[assignment,misc]
    EngineSamplingParams = None  # type: ignore[assignment,misc]
    _ENGINE_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────────────────

TOKENIZER_NAME = presets.TOKENIZER_NAME
DDV1_BOUNDARIES = list(presets.DDV1)
DD3B_BOUNDARIES = list(presets.DD3B)

from kotodama.serve import compile_cache  # noqa: E402
from kotodama.serve.steering import (  # noqa: E402
    STEER_DEFAULT_ALPHA,
    act_injection_to_plan,
    injection_to_plan,
    load_partial_norms,
    load_steer_banks,
    writes_to_plan,
)
from kotodama.serve.chatml import (  # noqa: E402
    BASE_STOP_TOKEN_IDS,
    CHAT_STOP_TOKEN_IDS,
    CHATML_TEMPLATE,
)

# Sampling defaults — shared by the native /generate and the OpenAI-compatible
# endpoints so every entrypoint agrees on the server's default behavior.
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_K = 0
DEFAULT_TOP_P = 0.0
DEFAULT_REPETITION_PENALTY = 1.2

# Architecture-size label that goes into the served model id
# (e.g. "kotodama-3b-instruct"). Falls back to the raw --model_size string.
SIZE_LABELS = {"proxy": "108m", "1b": "1b", "3b": "3b"}


PROXY_CONFIG = dict(
    **presets.shape("proxy"),
    rope_theta=500000.0,
    norm_eps=1e-5,
    qk_norm=True,
    tie_word_embeddings=True,
    z_loss_weight=0.0,
    use_liger=False,
    attn_impl="sdpa",
    attn_res=True,
    attn_res_boundaries=DDV1_BOUNDARIES,
)

CONFIG_3B = dict(
    **presets.shape("3b"),
    rope_theta=500000.0,
    norm_eps=1e-5,
    qk_norm=True,
    tie_word_embeddings=True,
    z_loss_weight=0.0,
    use_liger=False,
    attn_impl="sdpa",
    attn_res=True,
    attn_res_boundaries=DD3B_BOUNDARIES,
)

# 1B formation-ladder shape (presets "1b"); DD-3B boundaries like the 3B. The
# ladder arms trained with rope_fp32 — serve them with --rope_fp32.
CONFIG_1B = dict(CONFIG_3B, **presets.shape("1b"))

# Depth arm (presets "1b-deep": d1536 x 52 layers, the 1b's budget); DD-3B scaled to 52 layers.
CONFIG_1B_DEEP = dict(CONFIG_3B, **presets.shape("1b-deep"),
                      attn_res_boundaries=list(presets.DEFAULT_BOUNDARIES["1b-deep"]))

CONFIG_1B_DEEP_LOOP = dict(CONFIG_3B, **presets.shape("1b-deep-loop"),
                           attn_res_boundaries=list(presets.DEFAULT_BOUNDARIES["1b-deep-loop"]))

CONFIG_1B_DEEP_LOOPFIT = dict(CONFIG_3B, **presets.shape("1b-deep-loopfit"),
                              attn_res_boundaries=list(presets.DEFAULT_BOUNDARIES["1b-deep-loopfit"]))

CONFIG_1B_TSUMUGI = dict(CONFIG_3B, **presets.shape("1b-tsumugi"),
                               attn_res_boundaries=list(presets.DEFAULT_BOUNDARIES["1b-tsumugi"]))

# kotodama-standard (presets "1b-standard"): the 1b shape with its own fitted boundaries. It routes in the kernels' own
# precision, so its replicas run without KOTODAMA_ATTNRES_EPS_FP64_COMPAT (tsumugi's serve profile sets it).
CONFIG_1B_STANDARD = dict(CONFIG_3B, **presets.shape("1b-standard"),
                          attn_res_boundaries=list(presets.DEFAULT_BOUNDARIES["1b-standard"]))

MODEL_CONFIGS = {"proxy": PROXY_CONFIG, "1b": CONFIG_1B, "1b-deep": CONFIG_1B_DEEP,
                 "1b-deep-loop": CONFIG_1B_DEEP_LOOP, "1b-deep-loopfit": CONFIG_1B_DEEP_LOOPFIT,
                 "1b-tsumugi": CONFIG_1B_TSUMUGI, "1b-standard": CONFIG_1B_STANDARD, "3b": CONFIG_3B}


def model_config_for(model_size: str) -> dict | None:
    """The serving config for a size name or one of its public aliases (``presets.ALIASES``, e.g. ``1b-tsumugi``)."""
    return MODEL_CONFIGS.get(model_size) or MODEL_CONFIGS.get(presets.ALIASES.get(model_size, ""))


# ── Global state ────────────────────────────────────────────────────────────────

_model: KotodamaModel | None = None
_tokenizer: AutoTokenizer | None = None
_attn_res_override: str = "config"   # config | on | off (CLI --attn_res)
_attn_res_boundaries_override: str | None = None
_rope_fp32_override: bool = False   # CLI --rope_fp32 (1B ladder arms)
_device: torch.device = torch.device("cpu")
_serve_mode: str = "base"
_stop_token_ids: frozenset[int] = BASE_STOP_TOKEN_IDS
_max_seq_len = CONFIG_3B["max_position_embeddings"]
# Fast decode engine (single-stream): guarded by _engine_lock.
_engine: "DecodeEngine | None" = None
_engine_lock = asyncio.Lock()
# Batched multi-sample decoders over the engine (--multi_rows 4,8): an OpenAI request with 1 < n <= the largest bucket
# and a pure-temperature law decodes its n candidates in lockstep after one prefill, on the smallest bucket that fits
# (a step costs about the same at any row count, so the bucket sets the throughput). Empty = every candidate is serial.
_multi: "dict[int, MultiSampleDecoder]" = {}
# Cross-request batched decoding (--xbatch_rows 16,32,64): concurrent pure-temperature requests that share a sampling
# law and a steering config decode together, each request's rows filled from its own prefill (kotodama.serve.xbatch).
# None = off: every request takes the paths above.
_xbatch: "XBatchScheduler | None" = None
# Cascade attention in the row decoders (--cascade_attn): rows of one prompt share one copy of its KV and each row
# keeps only the KV it decodes, at most this many tokens; a request asking for more takes the serial path. 0 = the
# row decoders hold whole-context rows (no cap).
_row_token_cap: int = 0
_cascade_info: dict | None = None
# Prefix caching (engine extend-from-pos prefill); set by load_model.
_prefix_cache = False
# Multi-entry prefix KV pool (--prefix_pool_tokens): recent prompts' KV in fixed-size token blocks, so a prompt that
# shares a prefix with any pooled prompt (not only the previous request's) prefills only its suffix. None = off.
_prefix_pool: "PrefixKVCache | None" = None
# Residual steering (--steer-npz, comma-separated banks); set by load_model.
# The bank maps key -> (site_layer, UNIT vector fp32 on-device); per-site
# median norms are keyed by LAYER. The engine receives a per-request WRITES
# list [(site_layer, unit_vec, alpha * site_median_norm), ...] via set_steer.
_steer_loaded: bool = False
_steer_bank: dict[str, tuple[int, "torch.Tensor"]] = {}
_steer_meds: dict[int, float] = {}
_steer_aliases: dict[str, tuple[str | None, float, str]] = {}
# Per-layer median post-attention partial norms (``partial_norm_l{L}`` keys in any --steer-npz file, e.g. the
# scripts/steer/partial_norms.py sidecar): the scale table of the N write rule. Empty = rule N unavailable (400).
_steer_partial_norms: dict[int, float] = {}
# Per-request levers (--levers N, kotodama.model.levers): every steered request binds its own lever slot in the
# engine's tables for its lifetime (N slots at once), so steered and unsteered requests decode in one batch and a
# request may steer its prompt too (steer_span "uniform"). 0 = off: steering takes the engine's broadcast set_steer.
_levers: int = 0
# Seeded row decoders (--seeded_rows): a request's `seed` gives each candidate its own reproducible noise stream.
_seeded_rows: bool = False
# torch.compile mode of the engine and the row decoders (--compile_mode), and whether warmup compiles the truncated
# (top-p / min-p) step variants or leaves them to their first request (--warm_truncated), and whether both compile
# regionally (--regional).
_compile_mode: str = "max-autotune"
_warm_truncated: bool = True
_regional: bool = False
# Row buckets narrower than this compile without kernel benchmarking (--autotune_min_rows): it buys them nothing.
_autotune_min_rows: int = 0
# Signing (--sign_spool, kotodama.serve.sign): a request that sets `sign` leaves one record per candidate in the
# spool; with --sign_reducer the server also signs the spool whenever it is idle (one batch at a time, on the
# generation thread under the engine lock). The weights digest names the served model in every record.
_sign_spool: "SignSpool | None" = None
_signer: "RecordSigner | None" = None
_sign_writer: str = ""
_sign_weights: str = ""
_sign_stats: dict[str, Any] = {"records": 0, "batches": 0, "signed": 0, "skipped": 0, "errors": 0,
                               "opted_out": 0, "unsigned": {}, "skipped_reasons": {}, "inflight_deferred": {}}
# In-flight signing (--sign_inflight): signed requests decode on the signer's decoder (the widest bucket's geometry,
# so their tokens are the unsigned ones'), each candidate's span is exported as it ends, and the rows are reduced on
# the generation thread in batches of the signer's width (between steps once that many wait, else when idle) and
# written to the spool as signed batches. (record, captured span or None, lever plan or None) per candidate:
_sign_pending: list[tuple[Any, Any, Any]] = []
# reduced batches (results, plans, identity) the spool's drain lock held off
_sign_unwritten: list[tuple[list, list | None, dict]] = []
_sign_pending_lock = threading.Lock()
_sign_reducing = 0  # rows taken from _sign_pending and not yet reduced
_sign_timing: dict[str, float] = {"reduce_s": 0.0, "write_s": 0.0}  # in-flight batches, seconds in each part
# Kernel pins (--kernel_pins, kotodama.model.kernel_pins): the pin session, recording or replaying.
_pins: Any = None
# Startup: when this process started, how long it took to be ready (and against what budget), for /info.
_PROCESS_T0 = time.time()
_startup_info: dict | None = None
_compile_store_state = "off"
# Server-wide default repetition penalty (CLI --default_repetition_penalty):
# used by /generate when the field is absent and by the OpenAI routes when
# frequency_penalty is absent.
_default_repetition_penalty: float = DEFAULT_REPETITION_PENALTY


def resolve_oai_repetition_penalty(frequency_penalty: float | None) -> float:
    """OpenAI frequency_penalty -> kotodama repetition penalty.

    Explicit v maps to 1 + v (0 = off); absent falls back to the server's
    default (--default_repetition_penalty).
    """
    if frequency_penalty is not None:
        return 1.0 + frequency_penalty
    return _default_repetition_penalty


def _engine_lock_contended() -> bool:
    """Whether a coroutine waits on the engine lock (read from the generation thread: a length read of the lock's
    waiter deque, which the lock creates lazily). The batcher never waits on the lock while it holds it, so a waiter
    is a request outside the batch."""
    return bool(getattr(_engine_lock, "_waiters", None))


def _engine_prefill(input_ids: "torch.Tensor", slot: int = 0,
                    fresh: bool = False) -> tuple["torch.Tensor", dict | None]:
    """Engine prefill, via the prefix cache when enabled.

    ``slot``: the request's lever slot (--levers). A uniform lever's prompt KV is steered, so it bypasses the pool
    (keyed by tokens alone); the engine's own cached prefix is reused only under the same lever.

    ``fresh``: a prefill whose bytes must not depend on what the caches hold (seeded and signed requests). With
    batch-invariant blocks (``DecodeEngine.block_invariant``) the caches still serve it, from prompt KV only: a
    cached prefix extends to exactly what a cold prefill writes. Otherwise it is a cold prefill: a prefix hit would
    forward only the suffix, through other kernels.

    Returns (last-position logits, prefix-cache info dict or None; ``invariant`` marks a cache read whose bytes equal a
    cold prefill's).
    """
    assert _engine is not None
    if fresh and not _engine.block_invariant:
        return _engine.prefill(input_ids, slot=slot), None
    if fresh:
        uniform = bool(slot) and _engine.levers is not None and _engine.levers.span(slot) == "uniform"
        if _prefix_pool is not None and not uniform:
            logits, info = _prefix_pool.prefill(input_ids, prompt_kv_only=True)
            _engine._arm_decode_slot(slot)
        elif _prefix_cache:
            logits, info = _engine.prefill_cached(input_ids, slot=slot, prompt_kv_only=True)
        else:
            return _engine.prefill(input_ids, slot=slot), None
        return logits, {**info, "invariant": True}
    uniform = bool(slot) and _engine.levers is not None and _engine.levers.span(slot) == "uniform"
    if _prefix_pool is not None and not uniform:
        out = _prefix_pool.prefill(input_ids)
        _engine._arm_decode_slot(slot)
        return out
    if _prefix_cache:
        return _engine.prefill_cached(input_ids, slot=slot)
    return _engine.prefill(input_ids, slot=slot), None


# Cap concurrent in-flight streams so total KV/activation memory stays within
# GPU limits. Unbounded streaming concurrency + mid-stream disconnects caused a
# full-GPU OOM wedge. Override via LUXIA_MAX_CONCURRENT_STREAMS.
_MAX_CONCURRENT_STREAMS = int(os.environ.get("LUXIA_MAX_CONCURRENT_STREAMS", "4"))
_stream_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_STREAMS)

# Non-streaming generation is a fully synchronous CUDA decode loop; run directly
# on the event loop thread it starves /health + /memory for the whole generation
# (under saturation the gateway's health checker sees >15s probe silence and
# restarts busy replicas). Offload it to ONE dedicated worker thread: the loop
# stays responsive, while max_workers=1 keeps generations strictly serialized
# (the engine's static
# KV cache and the compiled decode path are stateful — never run two generations
# concurrently in this process; each replica owns a single GPU anyway).
#
# HARD INVARIANT: with --engine fast, ALL
# fast-engine activity — compile/cudagraph capture at startup warmup, replay,
# and every prefill/sample/step (streaming included) — must run on THIS thread.
# Inductor's cudagraph_trees keeps its tree-manager containers in THREAD-LOCAL
# storage; capture on one thread + replay on another dies with
# `assert torch._C._is_key_in_tls(attr_name)` in cudagraph_trees.get_obj.
# If you add a new engine call site, route it through _run_generation /
# run_in_executor(_generate_executor, ...). Never widen max_workers past 1.
_generate_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="koto-generate")


async def _run_generation(fn: Callable[..., "GenerateResponse"], /, *args: Any, **kwargs: Any) -> "GenerateResponse":
    """Run a blocking generation function on the dedicated generation thread.

    Exceptions (including HTTPException, e.g. prompt-too-long 400s) propagate
    unchanged to the awaiting handler.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_generate_executor, functools.partial(fn, *args, **kwargs))


@torch.inference_mode()
def _warmup_sdpa_cache(model: KotodamaModel, max_seq_len: int = 4096, step: int = 64) -> None:
    """Prime SDPA plan cache by decoding across representative KV lengths.

    cuDNN SDPA selects an algorithm per unique KV shape. Without warmup,
    each novel shape costs ~300ms of CPU-side plan selection. This runs a
    single decode sweep so all shapes are cached before serving.
    """
    logger.info("Warming up SDPA plan cache (up to %d tokens, step %d)...", max_seq_len, step)
    t0 = time.time()

    dummy_ids = torch.zeros(1, 1, dtype=torch.long, device=_device)
    out = model(dummy_ids, use_cache=True)
    past_kv = out["past_kv"]
    tok = out["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)

    target_positions = list(range(step, max_seq_len, step))
    pos = 1
    for target in target_positions:
        while pos < target:
            out = model(tok, use_cache=True, past_kv=past_kv)
            past_kv = out["past_kv"]
            tok = out["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)
            pos += 1

    torch.cuda.synchronize()
    logger.info("SDPA warmup done in %.1fs (%d positions)", time.time() - t0, pos)


def _load_checkpoint(checkpoint_path: Path | str) -> Any:
    """Load a checkpoint dict (.pt, streamed .pt.zst, or published weights: a directory or hf:REPO) —
    see kotodama.ckpt.load."""
    return ckpt.load(checkpoint_path)


def load_model(checkpoint_path: str, device: str = "cuda", mode: str = "base", model_size: str = "3b", engine: str = "fast", prefix_cache: bool = False, warmup_sdpa: bool = False, steer_npz: str | None = None, max_seq_len: int | None = None, multi_rows: tuple[int, ...] = (), xbatch_rows: tuple[int, ...] = (), xbatch_refill: bool = False, prefix_pool_tokens: int = 0, prefix_pool_block: int = 16, cascade_attn: bool = False, cascade_groups: int = 0, cascade_tail_tokens: int = 1024, levers: int = 0, seeded_rows: bool = False, compile_store: str | None = None, compile_mode: str = "max-autotune", warm_truncated: bool = True, regional: bool = False, autotune_min_rows: int = 0, emulate_casts: bool = False, prefill_buckets: tuple[int, ...] = ()) -> tuple[KotodamaModel, AutoTokenizer]:
    global _device, _serve_mode, _stop_token_ids, _engine, _prefix_cache, _max_seq_len, _multi, _xbatch, _prefix_pool
    global _row_token_cap, _cascade_info, _levers, _seeded_rows, _compile_mode, _warm_truncated, _regional
    global _compile_store_state, _autotune_min_rows
    _compile_mode, _warm_truncated, _regional = compile_mode, bool(warm_truncated), bool(regional)
    _autotune_min_rows = int(autotune_min_rows)
    global _steer_loaded, _steer_bank, _steer_meds, _steer_aliases, _steer_partial_norms
    _serve_mode = mode
    _stop_token_ids = CHAT_STOP_TOKEN_IDS if mode == "chat" else BASE_STOP_TOKEN_IDS
    # first: a tokenizer that cannot load fails the start in seconds, not after the engine has compiled
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if mode == "chat":
        tokenizer.chat_template = CHATML_TEMPLATE
    logger.info("Tokenizer loaded: %s (vocab %d, mode=%s)", TOKENIZER_NAME, len(tokenizer), mode)
    if device.startswith("cuda") and torch.cuda.is_available():
        _device = torch.device(device)
    else:
        _device = torch.device("cpu")
    logger.info("Device: %s", _device)

    model_cfg = model_config_for(model_size)
    if model_cfg is None:
        raise ValueError(f"Unknown model_size {model_size!r}; choose from {sorted(MODEL_CONFIGS)}")
    # Serve non-AttnRes checkpoints (e.g. proxy ablation arms) without
    # editing MODEL_CONFIGS — override via --attn_res off /
    # --attn_res_boundaries "0,3,7,12,21,25".
    model_cfg = dict(model_cfg)
    if _attn_res_override == "off":
        model_cfg["attn_res"] = False
    elif _attn_res_override == "on":
        model_cfg["attn_res"] = True
    if _attn_res_boundaries_override:
        model_cfg["attn_res_boundaries"] = [
            int(x) for x in _attn_res_boundaries_override.split(",")]
    if _rope_fp32_override:
        model_cfg["rope_fp32"] = True
    config = KotodamaModelConfig(**model_cfg)
    _max_seq_len = config.max_position_embeddings if max_seq_len is None else max_seq_len
    if not 16 <= _max_seq_len <= config.max_position_embeddings:
        raise ValueError(
            f"max_seq_len must be between 16 and {config.max_position_embeddings}, got {_max_seq_len}"
        )
    logger.info("Model config (%s): %dM params", model_size, config.param_count() // 1_000_000)
    logger.info("Serving context limit: %d tokens", _max_seq_len)

    model = KotodamaModel(config)

    ckpt_path = checkpoint_path if ckpt.is_hf_ref(checkpoint_path) else Path(checkpoint_path)
    if not ckpt.is_hf_ref(ckpt_path) and not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info("Loading checkpoint: %s", ckpt_path)
    loaded = _load_checkpoint(ckpt_path)

    state_dict = loaded.get("model", loaded)
    model.load_state_dict(state_dict, strict=True)
    logger.info("Checkpoint loaded (step %s, %s tokens)", loaded.get("step", "?"), loaded.get("tokens_consumed", "?"))
    del state_dict
    del loaded
    gc.collect()

    model = model.to(_device).eval()
    if _device.type == "cuda":
        model = model.bfloat16()

    use_engine = engine == "fast" and _device.type == "cuda" and _ENGINE_AVAILABLE and config.attn_res
    if engine == "fast" and not use_engine:
        logger.warning("--engine fast requested but unavailable (cuda=%s, import=%s, attn_res=%s) — using reference path",
                       _device.type == "cuda", _ENGINE_AVAILABLE, config.attn_res)

    # ── Steering config: validate + load the banks BEFORE the (minutes-long)
    # engine compile so a bad npz aborts immediately, not after warmup. ──
    if steer_npz is not None:
        if not use_engine:
            # The reference path has no injection hook — refuse loudly rather
            # than silently serving unsteered generations.
            raise RuntimeError(
                "--steer-npz requires the fast engine (cuda + --engine fast + attn_res model)"
            )
        _steer_bank, _steer_meds, _steer_aliases = load_steer_banks(
            steer_npz.split(","), config.num_layers, config.hidden_size, _device
        )
        _steer_partial_norms = load_partial_norms(steer_npz.split(","), config.num_layers)
        _steer_loaded = True
        logger.info(
            "Steering banks loaded: %d vectors %s @ site layers %s (meds %s), aliases %s",
            len(_steer_bank), sorted(_steer_bank),
            sorted({s for s, _ in _steer_bank.values()}),
            {layer: round(v) for layer, v in sorted(_steer_meds.items())},
            sorted(_steer_aliases),
        )

    if _device.type == "cuda" and not use_engine:
        # Reference path only: the engine's masked SDPA relies on cuDNN flash.
        torch.backends.cuda.enable_cudnn_sdp(False)
        logger.info("cuDNN SDP disabled (using flash/math backend)")

    _levers = int(levers) if use_engine else 0
    _seeded_rows = bool(seeded_rows) and use_engine
    if levers and not use_engine:
        raise ValueError("--levers needs the fast engine (cuda + --engine fast + attn_res model)")
    store_key = store_fields = None
    store_hit = False
    store_claim = None
    if use_engine and compile_store:
        # The compiled decode graphs depend on the model code, the runtime and these flags, not on the weights:
        # a stored blob for the same key turns the compiles below into cache hits.
        store_fields = compile_cache.key_fields({
            "model_size": model_size, "rope_fp32": _rope_fp32_override, "attn_res": _attn_res_override,
            "attn_res_boundaries": _attn_res_boundaries_override, "max_seq_len": _max_seq_len, "engine": engine,
            "mode": mode, "prefix_cache": prefix_cache, "multi_rows": list(multi_rows),
            "xbatch_rows": list(xbatch_rows), "xbatch_refill": xbatch_refill,
            "prefix_pool_tokens": prefix_pool_tokens, "prefix_pool_block": prefix_pool_block,
            "cascade_attn": cascade_attn, "cascade_groups": cascade_groups, "cascade_tail_tokens": cascade_tail_tokens,
            "levers": levers, "seeded_rows": seeded_rows, "steer_npz": steer_npz is not None,
            "compile_mode": compile_mode, "warm_truncated": bool(warm_truncated), "regional": bool(regional),
            "autotune_min_rows": int(autotune_min_rows),
            # present only when on, so the stores of servers without it keep their keys
            **({"emulate_casts": True} if emulate_casts else {}),
            **({"prefill_buckets": sorted(prefill_buckets, reverse=True)} if prefill_buckets else {}),
        })
        store_key = compile_cache.cache_key(store_fields)
        store_hit, store_claim = compile_cache.acquire(compile_store, store_key)
        _compile_store_state = "hit" if store_hit else "miss"
    if use_engine:
        logger.info("Building fast decode engine (compile may take minutes on cold inductor cache)...")
        t0 = time.time()

        def _build_and_warm_engine() -> "DecodeEngine":
            # steer_enabled=False -> the engine's code paths (and compiled
            # graphs) are bit-identical to a pre-steering build; enabled bakes
            # one per-layer row-add in unconditionally (zero rows = off) so
            # per-request writes never touch the graphs. Warmup below runs
            # with the zero buffer — the graphs it captures ARE the steered
            # graphs.
            # --levers: per-request lever tables replace the broadcast rows (banks resolve into levers).
            eng = DecodeEngine(model, max_seq_len=_max_seq_len, steer_enabled=steer_npz is not None and not levers,
                               levers=levers, emulate_casts=emulate_casts, block_buckets=prefill_buckets or None)
            global _regional
            if _regional:
                try:
                    regional_check_norms(model, eng.eps)
                except ValueError as exc:  # the decoders follow the engine's choice
                    logger.warning("--regional: %s; compiling the whole-step graphs instead", exc)
                    _regional = False
            # prefill blocks compile without kernel benchmarking: it buys their replays little (measured within a
            # few % on the 1B) and costs minutes per bucket
            eng.compile_step(mode=_compile_mode, regional=_regional, block_mode=_without_autotune(_compile_mode))
            # Compile every block-forward bucket (incl. overlap plans) now:
            # an unwarmed bucket would pay its compile on a live request.
            eng.warm_blocks()
            with torch.inference_mode():
                warm_params = EngineSamplingParams(temperature=0.9, repetition_penalty=1.2)
                trunc_params = EngineSamplingParams(temperature=0.9, repetition_penalty=1.2, top_p=0.9)
                warm_lengths = sorted({min(plen, _max_seq_len - 1) for plen in [16, 256]})
                for plen in warm_lengths:
                    dummy = torch.randint(4, config.vocab_size, (1, plen), device=_device)
                    logits = eng.prefill(dummy)
                    tok = eng.sample_first(logits, warm_params)
                    for _ in range(4):
                        tok = eng.step(tok, warm_params)
                    for _ in range(2 if _warm_truncated else 0):
                        # Warm the truncated (top-k/top-p) sampler graph too —
                        # a cold compile on the first truncated request would
                        # stall long enough to flap gateway health.
                        tok = eng.step(tok, trunc_params)
                    eng.step_logits(tok)  # warm the greedy/forward variant too
                if prefix_cache and _max_seq_len > 112:
                    # Warm the extend path (MATH-SDPA suffix forward + match logic):
                    # the first real extend otherwise pays ~0.6s of lazy dispatch.
                    base_ids = torch.randint(4, config.vocab_size, (1, 64), device=_device)
                    eng.prefill(base_ids)
                    ext = torch.cat(
                        [base_ids, torch.randint(4, config.vocab_size, (1, 48), device=_device)], dim=1
                    )
                    eng.prefill_cached(ext)
                    # And the suffix==1 (regenerate) variant, which routes through
                    # the compiled step.
                    eng.prefill_cached(ext)
                elif prefix_cache:
                    logger.warning("Prefix-cache warmup skipped: max_seq_len=%d is too small", _max_seq_len)
                torch.cuda.synchronize(_device)
            eng.reset()
            return eng

        def _build_and_warm_pool(eng: "DecodeEngine", n_tokens: int, block: int) -> "PrefixKVCache":
            pool = PrefixKVCache(eng, n_tokens, block_size=block)
            # A pool hit's suffix (the prompt's last partial block plus whatever differs, often one short turn)
            # extends in one block replay.
            eng.extend_single_chunk = True
            if not prefix_cache and _max_seq_len > 112:
                # The pool extends through prefill_cached; warm its eager extend like --prefix-cache does.
                base_ids = torch.randint(4, config.vocab_size, (1, 64), device=_device)
                eng.prefill(base_ids)
                eng.prefill_cached(torch.cat([base_ids, torch.randint(4, config.vocab_size, (1, 48),
                                                                      device=_device)], dim=1))
            with torch.inference_mode():
                # One store and one load (pool hit onto another prompt in the engine) warm the gather/scatter
                # kernels; the pool is emptied afterwards.
                n = min(4 * block + 8, _max_seq_len - 1)
                ids = torch.randint(4, config.vocab_size, (1, n), device=_device)
                pool.prefill(ids)
                eng.prefill(torch.randint(4, config.vocab_size, (1, 32), device=_device))
                pool.prefill(ids)
                torch.cuda.synchronize(_device)
            pool.clear()
            pool.stats.update({k: 0 for k in pool.stats})
            eng.reset()
            return pool

        # Build/compile/capture on the SAME single worker thread that serves all
        # generation (_generate_executor). max-autotune uses inductor's
        # cudagraph_trees, whose tree-manager containers live in THREAD-LOCAL
        # storage: a graph captured here on the main thread asserts
        # (torch._C._is_key_in_tls) the moment the worker thread replays it
        # (a 28ms 500 on every non-stream request of a warmed-at-startup
        # replica). load_model is sync (called
        # from lifespan before serving starts), so block on the future.
        _engine = _generate_executor.submit(_build_and_warm_engine).result()
        # One decoder per row width (--multi_rows and --xbatch_rows together); the narrower ones view the widest
        # one's KV storage, since only one decoder holds live rows at a time (the engine lock).
        decoders: dict[int, MultiSampleDecoder] = {}
        for rows in sorted(set(multi_rows) | set(xbatch_rows), reverse=True):
            # Same thread as the engine: its cudagraph trees are thread-local too.
            widest = decoders[max(decoders)] if decoders else None
            decoders[rows] = _generate_executor.submit(
                build_and_warm_row_decoder, _engine, rows, widest,
                (cascade_groups, cascade_tail_tokens) if cascade_attn else None).result()
        _multi = {r: decoders[r] for r in sorted(set(multi_rows))}
        if cascade_attn and decoders:
            widest = decoders[max(decoders)]
            assert isinstance(widest, CascadeDecoder)
            _row_token_cap = widest.tail_tokens
            _cascade_info = {"groups": {r: d.groups for r, d in sorted(decoders.items())},
                             "group_rows": {r: d.group_rows for r, d in sorted(decoders.items())},
                             "tail_tokens": widest.tail_tokens, "attention": widest.attend_backend,
                             "kv_gb": round(widest.kv_bytes() / 1e9, 2)}
            logger.info("Cascade attention ENABLED: %s", _cascade_info)
        if _multi:
            logger.info("Batched multi-sample decode ENABLED: row buckets %s", sorted(_multi))
        if xbatch_rows:
            runner = XBatchRunner({r: decoders[r] for r in sorted(set(xbatch_rows))}, prefill=_engine_prefill,
                                  set_steer=_engine.set_steer,
                                  decode=lambda ids: _tokenizer.decode(ids, skip_special_tokens=True),
                                  stop_ids=_stop_token_ids, max_seq_len=_max_seq_len, device=_device,
                                  refill=xbatch_refill, should_yield=_engine_lock_contended,
                                  levers=_engine.levers)
            _xbatch = XBatchScheduler(runner, _engine_lock, _run_generation)
            logger.info("Cross-request batched decode ENABLED: row buckets %s, refill %s", sorted(set(xbatch_rows)),
                        "on" if xbatch_refill else "off")
        # Disable cuDNN SDPA only AFTER the engine compiled: the captured
        # decode graph keeps its baked cuDNN flash kernels, while the eager
        # FALLBACK path (non-engine requests) avoids cuDNN's ~300ms per-shape
        # plan-selection stalls by dispatching to pytorch flash instead.
        torch.backends.cuda.enable_cudnn_sdp(False)
        _prefix_cache = prefix_cache or prefix_pool_tokens > 0  # the pool extends through prefill_cached
        if _prefix_cache:
            logger.info("Prefix caching ENABLED (engine extend-from-pos prefill)")
        if prefix_pool_tokens > 0:
            _prefix_pool = _generate_executor.submit(_build_and_warm_pool, _engine, prefix_pool_tokens,
                                                     prefix_pool_block).result()
            logger.info("Prefix KV pool ENABLED: %d blocks of %d tokens", _prefix_pool.n_blocks, prefix_pool_block)
        logger.info("Fast decode engine ready in %.1fs (compile store: %s; %s)", time.time() - t0,
                    "hit" if store_hit else ("miss" if store_key else "off"), compile_cache.cache_counters())
        if store_key and not store_hit:
            compile_cache.save(compile_store, store_key, store_fields)
        if store_claim is not None:
            store_claim.release()
    elif _device.type == "cuda" and warmup_sdpa:
        # Reference path only: prime cuDNN plan cache so per-shape ~300ms
        # stalls don't land on live requests. ~100s; opt-in for debug use.
        _warmup_sdpa_cache(model, _max_seq_len, step=64)

    # Run a few prefills at different lengths to warm any remaining caches
    if _device.type == "cuda":
        logger.info("Warming up prefill path...")
        with torch.inference_mode():
            for plen in sorted({min(plen, _max_seq_len - 1) for plen in [1, 16, 128, 512]}):
                dummy = torch.zeros(1, plen, dtype=torch.long, device=_device)
                model(dummy, use_cache=True)
            torch.cuda.synchronize(_device)
        logger.info("Prefill warmup done")

        # Release warmup workspace back to the CUDA allocator
        torch.cuda.empty_cache()
        alloc = torch.cuda.memory_allocated(_device) / 1e9
        reserved = torch.cuda.memory_reserved(_device) / 1e9
        logger.info("Post-warmup memory: %.3f GB allocated, %.3f GB reserved", alloc, reserved)


    return model, tokenizer


# ── Request/response schemas ────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class SteerVectorSpec(BaseModel):
    """One member of a per-request steering stack ({"name": alias-or-raw-npz-key,
    "alpha": dose}). alpha omitted -> the
    alias default (or STEER_DEFAULT_ALPHA for raw keys)."""

    name: str
    alpha: float | None = None


class InjectionSpec(BaseModel):
    """A raw steering write (--levers): ``alpha * vectors[j]`` at effective layer ``sites[j]``, the vectors already
    carrying their dose. ``kind`` "attn" writes the attention output (block-persistent by ``rule``), "mlp" the MLP
    output at the site only (rule S). ``kind`` "act" moves MLP neurons toward fixed values instead: entry j is
    neuron ``neurons[j]`` at effective layer ``sites[j]``, whose activation (the input of down_proj) becomes
    ``act + alpha * (mu[j] - act)`` (alpha 1 replaces it with ``mu[j]``); it takes no vectors."""

    sites: list[int]
    vectors: list[list[float]] = Field(default_factory=list)
    alpha: float = 1.0
    rule: Literal["K", "N", "S"] = "K"
    kind: Literal["attn", "mlp", "act"] = "attn"
    neurons: list[int] = Field(default_factory=list)
    mu: list[float] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # kept so _refuse_unserved can see them
    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    # The prompt as token ids, used as sent (the client's own tokenization; no text round trip).
    prompt_ids: list[int] | None = None
    # Seeded candidates (--seeded_rows): candidate i of a request draws from its own noise stream (this seed and i),
    # whatever else shares its batch. Pure-temperature laws on the batched decoders.
    seed: int | None = Field(default=None, ge=0)
    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=2048)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=0)
    top_p: float = Field(default=DEFAULT_TOP_P, ge=0.0, le=1.0)
    # Keep the tokens whose probability is at least min_p times the most likely token's (0 = off).
    min_p: float = Field(default=0.0, ge=0.0, lt=1.0)
    repetition_penalty: float = Field(
        default_factory=lambda: _default_repetition_penalty, ge=1.0, le=2.0)
    stop_strings: list[str] = Field(default_factory=list)
    stream: bool = False
    # Which tokens the repetition penalty counts: "generated" (default) the completion's own tokens, "row" the
    # prompt's too (the CTRL / Hugging Face convention).
    penalty_scope: Literal["generated", "row"] = "generated"
    # ── Residual steering (fast engine + --steer-npz only; else -> 400). ──
    # 'vectors' composes a weighted stack — each member scaled by alpha * ITS
    # OWN site's median norm; members may live at DIFFERENT sites (v2 multi-
    # site). 'steer_model' (+ optional 'steer_alpha') is single-vector use.
    vectors: list[SteerVectorSpec] | None = None
    steer_alpha: float | None = None
    steer_model: str | None = None
    # What lands at the later block entries: K = the site row unchanged (default), N = scaled to the partial norm
    # it lands in (needs partial_norm_l* in a bank), S = a single write at the site.
    steer_rule: Literal["K", "N", "S"] = "K"
    # Which positions the write reaches: "decode" = generated tokens only (default), "uniform" = the prompt too
    # (needs --levers).
    steer_span: Literal["decode", "uniform"] = "decode"
    # A raw write instead of bank vectors (needs --levers).
    injection: InjectionSpec | None = None
    # Sign the candidates (--sign_spool): each leaves a record in the sign spool, signed in flight (--sign_inflight)
    # or later by a drain (the server when idle, or a job); ``sign_meta`` travels with the record (the caller's ids).
    # None: the server's default (--sign_default); False opts out.
    sign: bool | None = None
    sign_meta: dict[str, Any] | None = None
    # A KV session's turn (--kv_sessions): the prompt is the turn's new tokens only, prefilled against the session's
    # KV; ``append_from`` first joins that candidate of the session's last turn (kotodama.serve.kv_sessions).
    kv_session: str | None = None
    append_from: int | None = Field(default=None, ge=0)


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: float
    timing: dict[str, float] | None = None
    # exact token IDs for offline replay (e.g. signature capture) —
    # always populated (cheap, already in hand); surfaced in the OAI response
    # only when the request sets return_token_ids
    prompt_ids: list[int] | None = None
    completion_ids: list[int] | None = None
    # the stop token that ended the completion (never in completion_ids); None when it ended otherwise
    stop_id: int | None = None
    # the candidate's sign record id (requests that set `sign`)
    sign_id: str | None = None


class ModelInfo(BaseModel):
    name: str
    mode: str
    params: int
    config: dict
    device: str
    checkpoint: str
    max_seq_len: int
    fast_engine: bool
    prefix_cache: bool = False
    steering: dict | None = None
    default_repetition_penalty: float | None = None
    xbatch: dict | None = None
    prefix_pool: dict | None = None
    cascade: dict | None = None
    startup: dict | None = None
    sign: dict | None = None
    kernel_pins: dict | None = None
    request_log: dict | None = None  # --request_log: its dir, writer and row counts
    kv_sessions: dict | None = None  # --kv_sessions: the open sessions, their lengths and cache states
    memory: dict | None = None


# ── /logprobs: batched prefill-only scoring (the four-register rider battery) ──
# One forward per BATCH, no decode loop (the computation lives in
# kotodama.serve.logprobs, shared with the training loop's in-loop riders).
# This is the fast path for logprob evals; the decode engine is irrelevant here
# because nothing is generated.

from kotodama.serve.logprobs import (  # noqa: E402
    LOGPROB_MAX_BATCH,
    LOGPROB_MAX_TOTAL_TOKENS,
    LogprobLimitError,
    score_logprob_batch,
    score_token_nll_batch,
)


class LogprobItem(BaseModel):
    id: str
    prompt: str
    candidates: list[str] = Field(..., min_length=1, max_length=64)


class LogprobRequest(BaseModel):
    items: list[LogprobItem] = Field(..., min_length=1, max_length=LOGPROB_MAX_BATCH)
    top_n: int = Field(default=5, ge=0, le=50)


@torch.inference_mode()
def _score_logprobs(req: LogprobRequest) -> dict[str, Any]:
    try:
        results = score_logprob_batch(
            lambda batch: _model(batch)["logits"], _tokenizer, req.items, req.top_n,
            _device, max_seq_len=_max_seq_len, max_total_tokens=LOGPROB_MAX_TOTAL_TOKENS)
    except LogprobLimitError as e:
        raise HTTPException(400, str(e)) from e
    return {"results": results, "model": _model_name()}


class TokenScoreItem(BaseModel):
    id: str
    prompt_ids: list[int] = Field(..., min_length=1)
    completion_ids: list[int] = Field(default_factory=list)


class TokenScoreRequest(BaseModel):
    items: list[TokenScoreItem] = Field(..., min_length=1, max_length=LOGPROB_MAX_BATCH)


@torch.inference_mode()
def _score_tokens(req: TokenScoreRequest) -> dict[str, Any]:
    try:
        results = score_token_nll_batch(
            lambda batch: _model(batch)["logits"], req.items, _device,
            vocab_size=_model.config.vocab_size, max_seq_len=_max_seq_len,
            max_total_tokens=LOGPROB_MAX_TOTAL_TOKENS)
    except LogprobLimitError as e:
        raise HTTPException(400, str(e)) from e
    return {"results": results, "model": _model_name()}


def _prompt_ids(request: GenerateRequest, tokenizer: AutoTokenizer) -> list[int]:
    """The request's prompt as token ids: ``prompt_ids`` as sent (the client's own tokenization, never re-encoded:
    a text round trip can split a boundary differently), else the prompt text or chat messages, encoded once."""
    if request.prompt_ids is not None:
        if request.prompt is not None or request.messages is not None:
            raise HTTPException(400, "Pass one of 'prompt_ids', 'prompt' or 'messages'")
        ids = list(request.prompt_ids)
        vocab = int(_model.config.vocab_size) if _model is not None else None
        if not ids or (vocab is not None and not all(0 <= t < vocab for t in ids)):
            raise HTTPException(400, f"prompt_ids must be a non-empty list of ids in [0, {vocab})")
        return ids
    return tokenizer.encode(_resolve_prompt(request, tokenizer))


def _resolve_prompt(request: GenerateRequest, tokenizer: AutoTokenizer) -> str:
    """Resolve prompt text from either raw prompt or messages array."""
    if request.messages is not None:
        if _serve_mode != "chat":
            raise HTTPException(400, "messages field requires --mode chat")
        msgs = [{"role": m.role, "content": m.content} for m in request.messages]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
    if request.prompt is not None:
        return request.prompt
    raise HTTPException(400, "Either 'prompt' or 'messages' must be provided")


# ── Steering composition ────────────────────────────────────────────────────────

def _engine_params(request: GenerateRequest) -> "EngineSamplingParams":
    """The request's sampling law for the engine and the row decoders."""
    return EngineSamplingParams(temperature=request.temperature, repetition_penalty=request.repetition_penalty,
                                top_k=request.top_k, top_p=request.top_p, penalty_scope=request.penalty_scope,
                                min_p=request.min_p)


def _penalty_ids(request: GenerateRequest, input_ids: "torch.Tensor", generated_ids: list[int]) -> list[int]:
    """The tokens the repetition penalty counts on the reference path (``penalty_scope``)."""
    return input_ids[0].tolist() + generated_ids if request.penalty_scope == "row" else generated_ids


def _request_wants_steering(request: GenerateRequest) -> bool:
    return (bool(request.vectors) or request.steer_model is not None or request.steer_alpha is not None
            or request.injection is not None or request.steer_span != "decode")


def _resolve_steer_member(name: str, alpha: float | None) -> tuple[str | None, float]:
    """Alias-or-raw-key -> (bank key or None for control, effective alpha)."""
    if name in _steer_aliases:
        key, default_alpha, _desc = _steer_aliases[name]
        if key is None:  # "control" — explicitly unsteered
            return None, 0.0
    else:
        key, default_alpha = name, STEER_DEFAULT_ALPHA
    if key not in _steer_bank:
        raise HTTPException(
            400,
            f"Unknown steering vector {name!r}; aliases {sorted(_steer_aliases)}, "
            f"raw keys {sorted(_steer_bank)}",
        )
    eff = alpha if alpha is not None else default_alpha
    if not math.isfinite(eff):
        raise HTTPException(400, f"Non-finite alpha for steering vector {name!r}")
    return key, float(eff)


def _steer_rule_kwargs(request: GenerateRequest) -> dict[str, Any]:
    """set_steer's rule arguments for a request; rule N without a partial-norm table is a 400."""
    if request.steer_rule == "N" and not _steer_partial_norms:
        raise HTTPException(400, "steer_rule N needs per-layer partial norms (partial_norm_l* keys) in --steer-npz")
    return {"rule": request.steer_rule, "norms": _steer_partial_norms or None}


def _request_lever(request: GenerateRequest) -> "LeverPlan | None":
    """A request's steering as one lever plan (--levers), or None when it is unsteered. Bank vectors resolve with
    the same dose math as ``_compose_steer_writes``; an ``injection`` is taken as given. 400 on anything the plan
    refuses."""
    assert _engine is not None and _engine.levers is not None
    from kotodama.model.levers import LeverError

    cfg = _engine.config
    boundaries = list(_engine._boundaries)
    try:
        if request.injection is not None:
            if request.vectors or request.steer_model is not None or request.steer_alpha is not None:
                raise HTTPException(400, "Pass either 'injection' or bank vectors, not both")
            inj = request.injection
            if inj.rule == "N" and not _steer_partial_norms:
                raise HTTPException(400, "injection rule N needs per-layer partial norms (partial_norm_l* keys)")
            if inj.kind == "act":
                return act_injection_to_plan(inj.sites, inj.neurons, inj.mu, inj.alpha, inj.vectors,
                                             request.steer_span, cfg.num_layers, cfg.intermediate_size)
            return injection_to_plan(inj.sites, inj.vectors, inj.alpha, inj.rule, inj.kind, request.steer_span,
                                     _steer_partial_norms or None, boundaries, cfg.num_layers, cfg.hidden_size)
        writes = _compose_steer_writes(request) if _request_wants_steering(request) and (
            request.vectors or request.steer_model is not None or request.steer_alpha is not None) else None
        if writes is None:
            return None
        kw = _steer_rule_kwargs(request)
        return writes_to_plan(writes, kw["rule"], kw["norms"], boundaries, cfg.num_layers, request.steer_span)
    except LeverError as exc:
        raise HTTPException(400, f"Steering request refused: {exc}") from None


def _check_span(request: GenerateRequest) -> None:
    if request.steer_span != "decode" and not _levers:
        raise HTTPException(400, "steer_span 'uniform' needs a server started with --levers")
    if request.injection is not None and not _levers:
        raise HTTPException(400, "'injection' needs a server started with --levers")


@contextlib.contextmanager
def _steering(request: GenerateRequest, timing: dict[str, float]):
    """Engage a request's steering for the block and yield its lever slot (0: none). Without --levers this is the
    broadcast set_steer path (slot 0); with it, the request binds a lever slot, released on exit. Everything that
    can 400 resolves before anything is engaged. Run on the generation thread."""
    assert _engine is not None
    _check_span(request)
    if not _levers:
        steer_writes = _compose_steer_writes(request)
        kwargs = _steer_rule_kwargs(request) if steer_writes is not None else {}
        try:
            if steer_writes is not None:
                _engine.set_steer(steer_writes, **kwargs)
                timing["steered"] = 1.0
                timing["steer_writes"] = float(len(steer_writes))
            yield 0
        finally:
            # ALWAYS clear (no-op on a steering-disabled engine): steering state must never leak into the next
            # request.
            _engine.set_steer(None)
        return
    plan = _request_lever(request)
    if plan is None:
        yield 0
        return
    if not _engine.levers.free:
        raise HTTPException(503, "every lever slot is in use; retry")
    slot = _engine.levers.bind(plan)
    timing["steered"] = 1.0
    timing["lever_slot"] = float(slot)
    try:
        yield slot
    finally:
        _engine.levers.release(slot)


def _compose_steer_writes(request: GenerateRequest) -> "list[tuple[int, torch.Tensor, float]] | None":
    """Resolve a request's steering fields into an engine WRITES list.

    Returns None when the request is unsteered (no fields, or an explicit
    'control'), else [(site_layer, unit_vec, alpha * that site's median
    norm), ...] — one write per resolved stack member; members at the same
    site simply sum row-wise inside engine.set_steer. Dose math per member is
    alpha * median_norm * unit, with each member scaled by ITS OWN site's
    median norm. Raises HTTPException(400)
    for steering use without --steer-npz, unknown names, or conflicting
    fields.
    """
    if not _request_wants_steering(request):
        return None
    if not _steer_loaded:
        raise HTTPException(
            400,
            "Steering fields (vectors/steer_model/steer_alpha) require a server "
            "started with --steer-npz",
        )
    if request.vectors:
        if request.steer_model is not None or request.steer_alpha is not None:
            raise HTTPException(400, "Pass either 'vectors' or steer_model/steer_alpha, not both")
        writes: list[tuple[int, torch.Tensor, float]] = []
        for item in request.vectors:
            key, alpha = _resolve_steer_member(item.name, item.alpha)
            if key is None:
                continue
            site_layer, unit = _steer_bank[key]
            writes.append((site_layer, unit, alpha * _steer_meds[site_layer]))
        return writes or None  # all-control stack -> None -> unsteered
    if request.steer_model is None:
        raise HTTPException(400, "steer_alpha requires steer_model (or use 'vectors')")
    key, alpha = _resolve_steer_member(request.steer_model, request.steer_alpha)
    if key is None:
        return None
    site_layer, unit = _steer_bank[key]
    return [(site_layer, unit, alpha * _steer_meds[site_layer])]


# ── Sampling ────────────────────────────────────────────────────────────────────

def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    generated_ids: list[int],
    min_p: float = 0.0,
) -> int:
    logits = logits.float()

    if repetition_penalty != 1.0 and generated_ids:
        penalty_ids = torch.tensor(generated_ids, device=logits.device, dtype=torch.long).unique()
        penalty_logits = logits[penalty_ids]
        penalty_logits = torch.where(
            penalty_logits > 0,
            penalty_logits / repetition_penalty,
            penalty_logits * repetition_penalty,
        )
        logits[penalty_ids] = penalty_logits

    if temperature == 0.0:
        return logits.argmax().item()

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_val = logits.topk(top_k).values[-1]
        logits = logits.masked_fill(logits < kth_val, float("-inf"))

    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = logits.sort(descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        mask = cumulative_probs - sorted_logits.softmax(dim=-1) >= top_p
        sorted_logits[mask] = float("-inf")
        logits = sorted_logits.scatter(0, sorted_indices, sorted_logits)

    if min_p > 0.0:
        logits = logits.masked_fill(logits < logits.max() + math.log(min_p), float("-inf"))

    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


# ── Model forward (fast or standard) ───────────────────────────────────────────

@torch.inference_mode()
def model_forward(model: KotodamaModel, input_ids: torch.Tensor) -> torch.Tensor:
    """Plain eager forward, returning logits (reference/debug path)."""
    return model(input_ids)["logits"]


# ── Generation ──────────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate(
    model: KotodamaModel,
    tokenizer: AutoTokenizer,
    request: GenerateRequest,
) -> GenerateResponse:
    timing: dict[str, float] = {}

    # The reference path has no injection hook — steering is engine-only.
    # (With --steer-npz loaded the engine always exists, so this only fires
    # for steering fields on an engine-less server.)
    if _request_wants_steering(request):
        raise HTTPException(400, "Steering requires the fast engine (--engine fast with --steer-npz)")

    # Tokenization
    t_tok = time.perf_counter()
    input_ids = torch.tensor([_prompt_ids(request, tokenizer)], device=_device)
    prompt_len = input_ids.shape[1]
    timing["tokenize_ms"] = (time.perf_counter() - t_tok) * 1000

    if prompt_len >= _max_seq_len:
        raise HTTPException(400, f"Prompt too long: {prompt_len} tokens (max {_max_seq_len})")

    generated_ids: list[int] = []
    t0 = time.perf_counter()

    decode_model = model

    # Prefill: process entire prompt, cache KV (always eager — variable prompt shapes)
    if _device.type == "cuda":
        torch.cuda.synchronize(_device)
    t_prefill = time.perf_counter()
    output = model(input_ids, use_cache=True)
    if _device.type == "cuda":
        torch.cuda.synchronize(_device)
    timing["prefill_ms"] = (time.perf_counter() - t_prefill) * 1000

    logits = output["logits"]
    past_kv = output.get("past_kv")

    t_sample = time.perf_counter()
    next_logits = logits[0, -1]
    token_id = sample_next_token(
        next_logits, request.temperature, request.top_k, request.top_p,
        request.repetition_penalty, _penalty_ids(request, input_ids, generated_ids), request.min_p,
    )
    timing["first_sample_ms"] = (time.perf_counter() - t_sample) * 1000

    if token_id in _stop_token_ids:
        elapsed = time.perf_counter() - t0
        return GenerateResponse(text="", prompt_tokens=prompt_len,
                                completion_tokens=0, tokens_per_second=0.0,
                                timing=timing,
                                prompt_ids=input_ids[0].tolist(),
                                completion_ids=[])

    generated_ids.append(token_id)
    next_input = torch.tensor([[token_id]], device=_device)

    # Decode: one token at a time with KV cache (compiled if available)
    decode_forward_ms = 0.0
    decode_sample_ms = 0.0
    decode_stop_ms = 0.0
    decode_text_ms = 0.0

    for _ in range(request.max_new_tokens - 1):
        if prompt_len + len(generated_ids) >= _max_seq_len:
            break

        if _device.type == "cuda":
            torch.cuda.synchronize(_device)
        t_fwd = time.perf_counter()
        output = decode_model(next_input, use_cache=True, past_kv=past_kv)
        if _device.type == "cuda":
            torch.cuda.synchronize(_device)
        decode_forward_ms += (time.perf_counter() - t_fwd) * 1000

        logits = output["logits"]
        past_kv = output.get("past_kv")

        t_samp = time.perf_counter()
        next_logits = logits[0, -1]
        token_id = sample_next_token(
            next_logits, request.temperature, request.top_k, request.top_p,
            request.repetition_penalty, _penalty_ids(request, input_ids, generated_ids), request.min_p,
        )
        decode_sample_ms += (time.perf_counter() - t_samp) * 1000

        if token_id in _stop_token_ids:
            break

        generated_ids.append(token_id)
        next_input = torch.tensor([[token_id]], device=_device)

        if request.stop_strings:
            t_stop = time.perf_counter()
            decoded_so_far = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if any(s in decoded_so_far for s in request.stop_strings):
                decode_stop_ms += (time.perf_counter() - t_stop) * 1000
                break
            decode_stop_ms += (time.perf_counter() - t_stop) * 1000

    elapsed = time.perf_counter() - t0

    t_decode_text = time.perf_counter()
    completion_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    timing["final_decode_ms"] = (time.perf_counter() - t_decode_text) * 1000

    n_decode_tokens = max(len(generated_ids) - 1, 0)
    timing["decode_forward_total_ms"] = round(decode_forward_ms, 2)
    timing["decode_sample_total_ms"] = round(decode_sample_ms, 2)
    timing["decode_stop_total_ms"] = round(decode_stop_ms, 2)
    if n_decode_tokens > 0:
        timing["decode_forward_per_token_ms"] = round(decode_forward_ms / n_decode_tokens, 2)
        timing["decode_sample_per_token_ms"] = round(decode_sample_ms / n_decode_tokens, 2)
    timing["total_ms"] = round(elapsed * 1000, 2)

    tps = len(generated_ids) / elapsed if elapsed > 0 else 0.0

    return GenerateResponse(
        text=completion_text,
        prompt_tokens=prompt_len,
        completion_tokens=len(generated_ids),
        tokens_per_second=round(tps, 1),
        timing=timing,
        prompt_ids=input_ids[0].tolist(),
        completion_ids=list(generated_ids),
    )


# ── Fast-engine generation ──────────────────────────────────────────────────

def _engine_compatible(request: GenerateRequest) -> bool:
    """Engine handles the full law surface: temperature + rep penalty + top-k/top-p.

    Truncated laws (top_k > 0 or 0 < top_p < 1) run the engine's dedicated
    compiled sampler variant rather than the reference path (which costs ~10x
    in decode).
    """
    return _engine is not None


@torch.inference_mode()
def engine_generate(request: GenerateRequest) -> GenerateResponse:
    """Non-streaming generation on the fast engine. Caller must hold _engine_lock."""
    assert _engine is not None and _tokenizer is not None and _model is not None
    timing: dict[str, float] = {}
    max_pos = _max_seq_len

    t_tok = time.perf_counter()
    input_ids = torch.tensor([_prompt_ids(request, _tokenizer)], device=_device)
    prompt_len = input_ids.shape[1]
    timing["tokenize_ms"] = (time.perf_counter() - t_tok) * 1000
    if prompt_len >= max_pos:
        raise HTTPException(400, f"Prompt too long: {prompt_len} tokens (max {max_pos})")

    params = _engine_params(request)
    t0 = time.perf_counter()
    with _steering(request, timing) as slot:
        t_prefill = time.perf_counter()
        prefill_logits, pc_info = _engine_prefill(input_ids, slot)
        tok = _engine.sample_first(prefill_logits, params)
        token_id = int(tok.item())
        timing["prefill_ms"] = (time.perf_counter() - t_prefill) * 1000
        if pc_info is not None:
            timing["prefix_cache_hit"] = pc_info["prefix_hit"]
            timing["prefix_common_tokens"] = pc_info["common_prefix"]
            timing["prefill_suffix_tokens"] = pc_info["suffix_len"]
            if "pool_tokens" in pc_info:
                timing["prefix_pool_tokens"] = pc_info["pool_tokens"]

        generated_ids: list[int] = []
        stop_id: int | None = token_id if token_id in _stop_token_ids else None
        if stop_id is None:
            generated_ids.append(token_id)
            for _ in range(request.max_new_tokens - 1):
                if prompt_len + len(generated_ids) >= max_pos:
                    break
                tok = _engine.step(tok, params)
                token_id = int(tok.item())
                if token_id in _stop_token_ids:
                    stop_id = token_id
                    break
                generated_ids.append(token_id)
                if request.stop_strings:
                    decoded_so_far = _tokenizer.decode(generated_ids, skip_special_tokens=True)
                    if any(s in decoded_so_far for s in request.stop_strings):
                        break

    elapsed = time.perf_counter() - t0
    completion_text = _tokenizer.decode(generated_ids, skip_special_tokens=True)
    timing["total_ms"] = round(elapsed * 1000, 2)
    n_decode = max(len(generated_ids) - 1, 0)
    if n_decode > 0:
        timing["decode_per_token_ms"] = round((elapsed - timing["prefill_ms"] / 1000) * 1000 / n_decode, 3)
    tps = len(generated_ids) / elapsed if elapsed > 0 else 0.0

    return GenerateResponse(
        text=completion_text,
        prompt_tokens=prompt_len,
        completion_tokens=len(generated_ids),
        tokens_per_second=round(tps, 1),
        timing=timing,
        prompt_ids=input_ids[0].tolist(),
        completion_ids=list(generated_ids),
        stop_id=stop_id,
    )


@torch.inference_mode()
async def engine_generate_stream(
    request: GenerateRequest,
    http_request: Request | None = None,
    sign: bool = False,
):
    """Streaming generation on the fast engine. ``sign``: the stream's generated ids (and the stop token that ended
    it) leave a sign record when it ends, a disconnect included (the drain signs it); the final event carries its
    sign_id.

    Emits text via windowed incremental detokenization: pending token ids are
    decoded together and flushed as soon as the text does not end in an incomplete
    UTF-8 sequence (U+FFFD) — O(window) per step instead of O(n) full redecode.

    THREADING: every engine call below is dispatched to _generate_executor —
    the single thread the engine compiled/captured on. _engine.step (T > 0)
    replays the cudagraph-captured fused step, and _engine_prefill can replay
    it too (--prefix-cache suffix==1 regenerate path); cudagraph_trees keeps
    its tree managers in THREAD-LOCAL storage, so touching either from the
    event loop thread asserts (torch._C._is_key_in_tls). sample_first/prefill
    are eager but mutate shared engine buffers, so they take the same thread.
    Cost: one ~50-100us executor hop per token. _engine_lock is held across
    the whole stream, so worker-thread engine ops never interleave requests.
    """
    assert _engine is not None and _tokenizer is not None and _model is not None
    async with _engine_lock:
        loop = asyncio.get_running_loop()
        max_pos = _max_seq_len
        steer_stack = contextlib.ExitStack()
        try:
            # Steering engages on the engine's thread (eager buffer mutation, no cudagraph replay, but the same
            # conservative discipline as every other engine call here); a 400 there is the stream's error.
            try:
                slot = await loop.run_in_executor(_generate_executor, steer_stack.enter_context,
                                                  _steering(request, {}))
            except HTTPException as exc:
                yield f"data: {json.dumps({'error': str(exc.detail)})}\n\n"
                return
            prompt_list = _prompt_ids(request, _tokenizer)
            input_ids = torch.tensor([prompt_list], device=_device)
            prompt_len = input_ids.shape[1]
            if prompt_len >= max_pos:
                yield f'data: {{"error": "Prompt too long: {prompt_len} tokens"}}\n\n'
                return
            generated_ids: list[int] = []

            def _record(stop: int | None) -> dict:
                if not sign:
                    return {}
                resp = GenerateResponse(text="", prompt_tokens=prompt_len, completion_tokens=len(generated_ids),
                                        tokens_per_second=0.0, prompt_ids=prompt_list,
                                        completion_ids=list(generated_ids), stop_id=stop)
                _spool_records(request, [resp])
                return {"sign_id": resp.sign_id}

            params = _engine_params(request)
            t_prefill = time.perf_counter()
            prefill_logits, pc_info = await loop.run_in_executor(
                _generate_executor, _engine_prefill, input_ids, slot
            )
            if pc_info is not None:
                logger.debug(
                    "stream prefill: hit=%s common=%d suffix=%d in %.1fms",
                    pc_info["prefix_hit"], pc_info["common_prefix"],
                    pc_info["suffix_len"], (time.perf_counter() - t_prefill) * 1000,
                )
            tok = await loop.run_in_executor(
                _generate_executor, _engine.sample_first, prefill_logits, params
            )
            token_id = int(tok.item())

            if token_id in _stop_token_ids:
                done = {'done': True, 'prompt_tokens': prompt_len, 'completion_tokens': 0, **_record(token_id)}
                yield f"data: {json.dumps(done)}\n\n"
                return

            generated_count = 0
            emitted_text = ""
            pending: list[int] = []

            def _flush() -> str:
                nonlocal emitted_text
                text = _tokenizer.decode(pending, skip_special_tokens=True)
                if text and not text.endswith("�"):
                    emitted_text += text
                    pending.clear()
                    return text
                return ""

            generated_count = 1
            pending.append(token_id)
            generated_ids.append(token_id)
            delta = _flush()
            if delta:
                yield f"data: {json.dumps({'token': delta})}\n\n"

            stopped = False
            interrupted = False
            stop_id: int | None = None
            for _ in range(request.max_new_tokens - 1):
                if prompt_len + generated_count >= max_pos:
                    break
                if http_request is not None and await http_request.is_disconnected():
                    interrupted = True
                    break

                tok = await loop.run_in_executor(
                    _generate_executor, _engine.step, tok, params
                )
                token_id = int(tok.item())
                if token_id in _stop_token_ids:
                    stop_id = token_id
                    break
                generated_count += 1
                pending.append(token_id)
                generated_ids.append(token_id)
                delta = _flush()
                if delta:
                    yield f"data: {json.dumps({'token': delta})}\n\n"
                if request.stop_strings and any(s in emitted_text for s in request.stop_strings):
                    stopped = True
                    break

            if not interrupted:
                # Flush any pending tail (possibly with incomplete bytes dropped
                # by skip_special decode semantics).
                if pending and not stopped:
                    tail = _tokenizer.decode(pending, skip_special_tokens=True)
                    if tail:
                        yield f"data: {json.dumps({'token': tail})}\n\n"
                done = {'done': True, 'prompt_tokens': prompt_len, 'completion_tokens': generated_count,
                        **_record(stop_id)}
                yield f"data: {json.dumps(done)}\n\n"
            else:
                _record(stop_id)  # what was generated before the client left is still served inference
        finally:
            # ALWAYS release steering (the broadcast rows, or the request's lever slot) so state never leaks
            # between requests — including the disconnect / GeneratorExit paths. Prefer the engine thread; fall
            # back to a direct call if the loop is tearing down (both are eager buffer mutations, safe off-thread
            # while the engine is idle).
            try:
                await loop.run_in_executor(_generate_executor, steer_stack.close)
            except BaseException:
                # Loop teardown / GeneratorExit / cancellation mid-await:
                # release synchronously, then let the exception propagate.
                steer_stack.close()
                _engine.set_steer(None)
                raise


@torch.inference_mode()
async def generate_stream(
    model: KotodamaModel,
    tokenizer: AutoTokenizer,
    request: GenerateRequest,
    http_request: Request | None = None,
):
    # THREADING: this reference streaming path is safe on the event loop
    # thread — it never touches the fast engine (fully eager decode, no
    # cudagraph_trees thread-local state).
    # Hold a slot for the whole stream so concurrent streams stay bounded.
    async with _stream_semaphore:
        past_kv = None
        output = None
        interrupted = False
        try:
            if _request_wants_steering(request):
                # Reference path: no injection hook (see generate()).
                yield f"data: {json.dumps({'error': 'Steering requires the fast engine (--engine fast with --steer-npz)'})}\n\n"
                return
            input_ids = torch.tensor([_prompt_ids(request, tokenizer)], device=_device)
            prompt_len = input_ids.shape[1]

            if prompt_len >= _max_seq_len:
                yield f'data: {{"error": "Prompt too long: {prompt_len} tokens"}}\n\n'
                return

            generated_ids: list[int] = []
            prev_text = ""

            decode_model = model

            # Prefill (always eager)
            output = model(input_ids, use_cache=True)
            past_kv = output.get("past_kv")
            next_logits = output["logits"][0, -1]

            token_id = sample_next_token(
                next_logits, request.temperature, request.top_k, request.top_p,
                request.repetition_penalty, _penalty_ids(request, input_ids, generated_ids), request.min_p,
            )

            if token_id in _stop_token_ids:
                yield f"data: {json.dumps({'done': True, 'prompt_tokens': prompt_len, 'completion_tokens': 0})}\n\n"
                return

            generated_ids.append(token_id)
            next_input = torch.tensor([[token_id]], device=_device)

            current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            delta = current_text
            prev_text = current_text
            if delta:
                yield f"data: {json.dumps({'token': delta})}\n\n"

            # Decode with KV cache (compiled if available)
            for _ in range(request.max_new_tokens - 1):
                if prompt_len + len(generated_ids) >= _max_seq_len:
                    break

                # Stop early for clients that have disconnected. This keeps us off
                # the mid-stream exception path, which would otherwise pin this
                # request's CUDA tensors via the traceback reference cycle.
                if http_request is not None and await http_request.is_disconnected():
                    interrupted = True
                    break

                output = decode_model(next_input, use_cache=True, past_kv=past_kv)
                past_kv = output.get("past_kv")
                next_logits = output["logits"][0, -1]

                token_id = sample_next_token(
                    next_logits, request.temperature, request.top_k, request.top_p,
                    request.repetition_penalty, _penalty_ids(request, input_ids, generated_ids), request.min_p,
                )

                if token_id in _stop_token_ids:
                    break

                generated_ids.append(token_id)
                next_input = torch.tensor([[token_id]], device=_device)

                current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                delta = current_text[len(prev_text):]
                prev_text = current_text

                if delta:
                    yield f"data: {json.dumps({'token': delta})}\n\n"

                if request.stop_strings:
                    if any(s in current_text for s in request.stop_strings):
                        break

            if not interrupted:
                yield f"data: {json.dumps({'done': True, 'prompt_tokens': prompt_len, 'completion_tokens': len(generated_ids)})}\n\n"
        except BaseException:
            # GeneratorExit / CancelledError on client disconnect arrive here.
            interrupted = True
            raise
        finally:
            # Drop refs to this request's CUDA tensors; on the interrupted path
            # force a cyclic GC so traceback-pinned tensors are reclaimed now
            # instead of accumulating across disconnects into a full-GPU OOM.
            past_kv = None
            output = None
            if interrupted:
                gc.collect()


# ── App ─────────────────────────────────────────────────────────────────────────

_checkpoint_path = ""
_mode_arg = "base"
_model_size_arg = "3b"
_served_name_override: str | None = None
_engine_arg = "fast"
_max_seq_len_arg: int | None = None
_prefix_cache_arg = False
_warmup_sdpa_arg = False
_steer_npz_arg: str | None = None
_multi_rows_arg: tuple[int, ...] = ()
_xbatch_rows_arg: tuple[int, ...] = ()
_prefill_buckets_arg: tuple[int, ...] = ()
_xbatch_refill_arg = False
_prefix_pool_tokens_arg = 0
_prefix_pool_block_arg = 16
_cascade_attn_arg = False
_cascade_groups_arg = 0
_cascade_tail_tokens_arg = 1024
_levers_arg = 0
_seeded_rows_arg = False
_compile_store_arg: str | None = None
_compile_mode_arg = "max-autotune"
_warm_truncated_arg = True
_regional_arg = True
_startup_budget_s_arg = 1800.0
_autotune_min_rows_arg = 0
_emulate_casts_arg = False
_kernel_pins_arg: str | None = None
_sign_spool_arg: str | None = None
_sign_reducer_arg: str | None = None
_sign_reducer_kwargs_arg = "{}"
_sign_layers_arg: tuple[int, ...] = ()
_sign_tail_arg = 256
_sign_width_arg = 0
_sign_rows_arg = 128
_sign_groups_arg = 16
_sign_inflight_arg = False
_sign_audit_every_arg = 100
_sign_default_arg = False
# below this much free space on the spool's filesystem, /info's sign.disk reads low (archive the spool:
# kotodama.serve.sign_archive)
_sign_disk_floor_gb_arg = 150.0
_sign_disk_low = False
_profile_name: str | None = None
# The request log (--request_log, kotodama.serve.request_log): one row per request, off unless the flag is given.
_request_log_arg: str | None = None
_request_log: "Any" = None
_request_log_weights: str | None = None
# the candidates a request produced (their exact ids, stop and sign ids), gathered for its log row
_logged_candidates: contextvars.ContextVar[list | None] = contextvars.ContextVar("logged_candidates", default=None)
_sign_engine_digest: str | None = None  # sha256 of the signer's engine identity at startup (records' meta)
# KV sessions (--kv_sessions, kotodama.serve.kv_sessions): conversations whose KV lives across their requests
_kv_sessions_arg = 0
_kv_session_idle_s_arg = 3600.0
_kv_sessions: "Any" = None


def attnres_routing() -> dict[str, Any]:
    """The AttnRes forward ops' eps choice as they run it (bound when they were imported) beside the environment
    variable that names it: a serve profile's env applies after the import, so the two can differ."""
    from kotodama.model.flash_attn_res.ops import phase_1 as p1ops
    from kotodama.model.flash_attn_res.ops import phase_2 as p2ops

    return {"eps_fp64_compat": bool(p1ops.EPS_FP64_COMPAT and p2ops.EPS_FP64_COMPAT),
            "env": os.environ.get("KOTODAMA_ATTNRES_EPS_FP64_COMPAT")}


def _record_startup() -> None:
    """Seconds from process start to ready, against --startup_budget_s (a start over budget is logged loudly)."""
    global _startup_info
    seconds = time.time() - _PROCESS_T0
    _startup_info = {"seconds": round(seconds, 1), "budget_s": _startup_budget_s_arg, "regional": _regional,
                     "attnres_routing": attnres_routing(),
                     "compile_mode": _compile_mode, "autotune_min_rows": _autotune_min_rows,
                     "compile_store": _compile_store_state}
    if _startup_budget_s_arg and seconds > _startup_budget_s_arg:
        logger.warning("STARTUP OVER BUDGET: ready after %.0fs, budget %.0fs (%s). A start this slow usually means "
                       "the compiled graphs multiplied: see the compile log above, or start with --regional",
                       seconds, _startup_budget_s_arg, _startup_info)
    else:
        logger.info("startup: ready after %.0fs (budget %.0fs; %s)", seconds, _startup_budget_s_arg, _startup_info)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _pins
    if _kernel_pins_arg:
        from kotodama.model import kernel_pins

        # before anything compiles: the pins decide every autotuned kernel from the first launch
        _pins = kernel_pins.install(_kernel_pins_arg, "replay" if Path(_kernel_pins_arg).exists() else "record",
                                    allow_missing=True)
    _model, _tokenizer = load_model(
        _checkpoint_path, _device_arg, mode=_mode_arg,
        model_size=_model_size_arg, engine=_engine_arg, prefix_cache=_prefix_cache_arg,
        warmup_sdpa=_warmup_sdpa_arg,
        steer_npz=_steer_npz_arg,
        max_seq_len=_max_seq_len_arg,
        multi_rows=_multi_rows_arg,
        xbatch_rows=_xbatch_rows_arg,
        prefill_buckets=_prefill_buckets_arg,
        xbatch_refill=_xbatch_refill_arg,
        prefix_pool_tokens=_prefix_pool_tokens_arg,
        prefix_pool_block=_prefix_pool_block_arg,
        cascade_attn=_cascade_attn_arg,
        cascade_groups=_cascade_groups_arg,
        cascade_tail_tokens=_cascade_tail_tokens_arg,
        levers=_levers_arg,
        seeded_rows=_seeded_rows_arg,
        compile_store=_compile_store_arg,
        compile_mode=_compile_mode_arg,
        warm_truncated=_warm_truncated_arg,
        regional=_regional_arg,
        autotune_min_rows=_autotune_min_rows_arg,
        emulate_casts=_emulate_casts_arg,
    )
    sign_task = _start_signing()
    _open_request_log()
    _open_kv_sessions()
    if _pins is not None and _pins.mode == "record":
        logger.info("kernel pins recorded: %s (%d kernels, %s)", _pins.save(), len(_pins.table), _pins.digest[:12])
    elif _pins is not None and _pins.misses:
        logger.warning("kernel pins: %d kernels missing from %s took their first config (digest now %s)",
                       len(_pins.misses), _pins.path, _pins.digest[:12])
    _record_startup()
    try:
        yield
    finally:
        # Uvicorn's graceful shutdown has already drained in-flight requests by
        # the time we get here; drop anything still queued (its awaiting request
        # was cancelled) so the non-daemon worker thread can't stall exit.
        if sign_task is not None:
            sign_task.cancel()
            await _sign_flush_at_shutdown()
        if _request_log is not None:
            _request_log.close()
        _generate_executor.shutdown(wait=False, cancel_futures=True)


def _open_kv_sessions() -> None:
    global _kv_sessions
    if _kv_sessions_arg > 0 and _engine is not None and _xbatch is not None:
        from kotodama.serve.kv_sessions import SessionStore

        _kv_sessions = SessionStore(_kv_sessions_arg, idle_s=_kv_session_idle_s_arg)
        logger.info("KV sessions ENABLED: up to %d, closed after %.0f s idle when room is needed", _kv_sessions_arg,
                    _kv_session_idle_s_arg)
    elif _kv_sessions_arg > 0:
        logger.warning("--kv_sessions needs the fast engine with --xbatch_rows; KV sessions are off")


def _open_request_log() -> None:
    """Open the request log (--request_log): rows name the served weights by their digest."""
    global _request_log, _request_log_weights
    if not _request_log_arg:
        return
    from kotodama.serve.request_log import RequestLog

    if _sign_weights:
        _request_log_weights = _sign_weights
    elif _model is not None:
        from kotodama.model.signing import weights_digest

        _request_log_weights = weights_digest(_model)
    _request_log = RequestLog(_request_log_arg)
    logger.info("request log: %s (writer %s)", _request_log.root, _request_log.writer)


def _note_candidates(results: "list[GenerateResponse]") -> None:
    """Hand a request's candidates to its request-log row (a no-op without --request_log)."""
    got = _logged_candidates.get()
    if got is None:
        return
    for r in results:
        got.append({"prompt_ids": r.prompt_ids, "completion_ids": r.completion_ids, "stop_id": r.stop_id,
                    "sign_id": r.sign_id, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens})


class _RequestLogMiddleware:
    """ASGI middleware writing one request-log row per logged request: the body the client sent, the response it got
    (a stream's events joined into its text), status, latency and the candidates the endpoint noted. A request whose
    body sets ``"log": false`` is answered and not written. Client addresses are not recorded."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        from kotodama.serve.request_log import LOGGED_PATHS

        log = _request_log
        if log is None or scope.get("type") != "http" or scope.get("method") != "POST" \
                or scope.get("path") not in LOGGED_PATHS:
            await self.app(scope, receive, send)
            return
        t0 = time.time()
        body = bytearray()
        sent = bytearray()
        status = [0]
        streamed = [False]

        async def recv() -> dict:
            msg = await receive()
            if msg.get("type") == "http.request":
                body.extend(msg.get("body", b""))
            return msg

        async def snd(msg: dict) -> None:
            if msg.get("type") == "http.response.start":
                status[0] = int(msg.get("status", 0))
                ctype = dict(msg.get("headers") or []).get(b"content-type", b"")
                streamed[0] = b"text/event-stream" in ctype
            elif msg.get("type") == "http.response.body":
                sent.extend(msg.get("body", b""))
            await send(msg)

        token = _logged_candidates.set([])
        try:
            await self.app(scope, recv, snd)
        finally:
            candidates = _logged_candidates.get() or []
            _logged_candidates.reset(token)
            try:
                _write_request_row(log, scope["path"], bytes(body), bytes(sent), status[0], streamed[0], t0,
                                   candidates)
            except Exception:  # the log never fails a request
                log.stats["errors"] += 1
                logger.exception("request log: building a row failed")


def _parse_json(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode("utf-8", errors="replace")


def _stream_text(raw: bytes) -> dict:
    """A server-sent-event response as the text it streamed and its last event."""
    text, last, events = [], None, 0
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            ev = json.loads(line[6:])
        except ValueError:
            continue
        events += 1
        last = ev
        if isinstance(ev.get("token"), str):
            text.append(ev["token"])
        for ch in ev.get("choices") or []:
            piece = (ch.get("delta") or {}).get("content") if "delta" in ch else ch.get("text")
            if isinstance(piece, str):
                text.append(piece)
    return {"text": "".join(text), "events": events, "last": last}


def _write_request_row(log: Any, path: str, body: bytes, sent: bytes, status: int, streamed: bool, t0: float,
                       candidates: list) -> None:
    request = _parse_json(body)
    if isinstance(request, dict) and request.get("log") is False:
        log.opted_out()
        return
    log.write({"ts": t0, "latency_ms": round((time.time() - t0) * 1000, 1), "endpoint": path, "status": status,
               "writer": log.writer, "model": _model_name(), "profile": _profile_name,
               "weights": _request_log_weights, "request": request,
               "response": _stream_text(sent) if streamed else _parse_json(sent), "stream": streamed,
               "candidates": candidates})


def _start_signing() -> "asyncio.Task | None":
    """Open the sign spool (--sign_spool) and, with --sign_reducer, build the signing decoder on the generation thread
    and start the idle drain. Returns the drain task (None without a reducer)."""
    global _sign_spool, _signer, _sign_writer, _sign_weights, _sign_engine_digest
    if not _sign_spool_arg:
        return None
    import socket

    from kotodama.model.signing import load_reducer, weights_digest
    from kotodama.serve.sign import SignSpool

    if _engine is None:
        raise RuntimeError("--sign_spool needs the fast engine (--engine fast on cuda)")
    _sign_spool = SignSpool(_sign_spool_arg)
    _sign_writer = f"{socket.gethostname()}-{os.getpid()}"
    _sign_weights = weights_digest(_engine.model)
    logger.info("sign spool: %s (writer %s, weights %s)", _sign_spool.root, _sign_writer, _sign_weights[:12])
    if not _sign_reducer_arg:
        return None
    from kotodama.model.signer import RecordSigner

    reducer = load_reducer(_sign_reducer_arg, **json.loads(_sign_reducer_kwargs_arg or "{}"))

    def build() -> RecordSigner:
        signer = RecordSigner.build(_engine, reducer, layers=_sign_layers_arg, tail=_sign_tail_arg,
                                    width=_sign_width_arg or _max_seq_len, rows=_sign_rows_arg,
                                    groups=_sign_groups_arg, compile_mode=_without_autotune(_compile_mode),
                                    pins=_pins, audit_every=_sign_audit_every_arg)
        signer.warm()
        return signer

    _signer = _generate_executor.submit(
        functools.partial(build_inflight_signer, reducer) if _sign_inflight_arg else build).result()
    if _sign_inflight_arg:
        assert _xbatch is not None
        _xbatch.runner.sign_decoder = _signer.dec
        _xbatch.runner.between_steps = _sign_between_steps
    _sign_engine_digest = hashlib.sha256(json.dumps(_signer.identity, sort_keys=True, default=str)
                                         .encode("utf-8")).hexdigest()
    logger.info("sign: %sdraining %s when idle (%d-row signing decoder, %d groups, %d-token tails, reducer %s)",
                "signing in flight and " if _sign_inflight_arg else "", _sign_spool.root, _signer.dec.rows,
                _signer.dec.groups, _signer.dec.tail_tokens, getattr(reducer, "reducer_identity", "?"))
    return asyncio.get_running_loop().create_task(_sign_when_idle())


def build_inflight_signer(reducer: "Any" = None) -> "RecordSigner":
    """The in-flight signer (--sign_inflight; call on the generation thread): a signing CascadeDecoder with the
    widest xbatch bucket's geometry and compile mode, viewing its KV storage (one decoder holds live rows at a time),
    so a signed request decodes the tokens the unsigned bucket would (one kernel pin table, --emulate_casts). Its
    generation and scoring steps are warmed (and, with kernel pins recording, recorded); the idle drain signs records
    on it too, so a drained record and an in-flight row sign alike."""
    from kotodama.model.signer import RecordSigner
    from kotodama.model.signing import SignSpec, load_reducer

    if reducer is None:
        reducer = load_reducer(_sign_reducer_arg, **json.loads(_sign_reducer_kwargs_arg or "{}"))
    if _sign_rows_arg != 128 or _sign_groups_arg != 16:
        logger.info("sign: --sign_rows / --sign_groups do not apply in flight (the signer takes the widest bucket's "
                    "geometry)")
    assert _xbatch is not None and _engine is not None
    widest = _xbatch.runner.decoders[_xbatch.runner.max_rows]
    if not isinstance(widest, CascadeDecoder):
        raise RuntimeError("--sign_inflight needs --cascade_attn (the signing decoder is a CascadeDecoder)")
    spec = SignSpec(_sign_layers_arg, tail=_sign_tail_arg, depth=True, attn=True, width=_sign_width_arg or _max_seq_len)
    dec = CascadeDecoder(_engine, widest.rows, groups=widest.groups, group_rows=widest.group_rows,
                         tail_tokens=widest.tail_tokens, kv_from=widest, seeded=_seeded_rows, sign=spec)
    mode = _row_compile_mode(widest.rows)
    dec.compile_step(mode=mode, regional=_regional)
    with torch.inference_mode():  # the generation step variants, as build_and_warm_row_decoder warms a bucket's
        params = EngineSamplingParams(temperature=0.9, repetition_penalty=1.2)
        trunc = EngineSamplingParams(temperature=0.9, repetition_penalty=1.2, top_p=0.9, min_p=0.05)
        for plen in sorted({min(plen, _engine.max_seq_len - 8) for plen in [16, 256]}):
            logits = _engine.prefill(torch.randint(4, _engine.config.vocab_size, (1, plen), device=_engine.device))
            for law in (params, trunc) if _warm_truncated else (params,):
                dec.start(logits, law)
                for _ in range(4):
                    dec.step(law)
        torch.cuda.synchronize(_engine.device)
    signer = RecordSigner(dec, reducer, compile_mode=mode, pins=_pins, audit_every=_sign_audit_every_arg)
    signer.warm(reduce=True)
    dec.release(range(dec.rows))
    _engine.reset()
    return signer


def _model_name() -> str:
    """Canonical served model id, e.g. 'kotodama-3b-instruct'.

    Priority: explicit --served_model_name > derived from architecture size
    (--model_size) + serving mode.
    """
    if _served_name_override:
        return _served_name_override
    size_label = SIZE_LABELS.get(_model_size_arg, _model_size_arg)
    mode_label = "instruct" if _serve_mode == "chat" else "base"
    return f"kotodama-{size_label}-{mode_label}"


app = FastAPI(title="kotodama", lifespan=lifespan)
app.add_middleware(_RequestLogMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/info")
async def info():
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    steering = None
    if _steer_loaded:
        sites = sorted({s for s, _ in _steer_bank.values()})
        steering = {
            "sites": sites,  # LAYER indices; npz suffix sublayer = 2*layer
            "sublayers": [2 * s for s in sites],
            "median_norms": {str(layer): v for layer, v in sorted(_steer_meds.items())},
            "aliases": {
                a: {"key": k, "alpha": al, "desc": d,
                    "site_layer": _steer_bank[k][0] if k is not None else None}
                for a, (k, al, d) in sorted(_steer_aliases.items())
            },
            "keys": {k: {"site_layer": s} for k, (s, _) in sorted(_steer_bank.items())},
            "note": "block-persistent injection (site row + later block entries), "
                    "decode/generated positions only; multi-site stacks compose row-wise",
        }
    if _levers and _engine is not None and _engine.levers is not None:
        steering = dict(steering or {})
        steering["levers"] = {"slots": _levers, "free": _engine.levers.free, "spans": ["decode", "uniform"],
                              "sites": ["attn", "mlp"], "keyed_on": "effective layer",
                              "note": "per-request levers: each steered request binds its own slot"}
    return ModelInfo(
        name=_model_name(),
        mode=_serve_mode,
        params=_model.config.param_count(),
        config=asdict(_model.config),
        device=str(_device),
        checkpoint=_checkpoint_path,
        max_seq_len=_max_seq_len,
        fast_engine=_engine is not None,
        prefix_cache=_prefix_cache,
        steering=steering,
        default_repetition_penalty=_default_repetition_penalty,
        xbatch=None if _xbatch is None else {"rows": sorted(_xbatch.runner.decoders),
                                             "refill": _xbatch.runner.refill,
                                             "stats": dict(_xbatch.runner.stats)},
        prefix_pool=None if _prefix_pool is None else {"blocks": _prefix_pool.n_blocks,
                                                       "block_size": _prefix_pool.block_size,
                                                       "cached_blocks": len(_prefix_pool.index),
                                                       "stats": dict(_prefix_pool.stats)},
        cascade=_cascade_info,
        startup=_startup_info,
        sign=_sign_info(),
        request_log=_request_log.info() if _request_log is not None else None,
        kv_sessions=_kv_sessions.info() if _kv_sessions is not None else None,
        kernel_pins=None if _pins is None else {"mode": _pins.mode, "path": str(_pins.path), "kernels": len(_pins.table),
                                                "hits": _pins.hits, "recorded": _pins.recorded,
                                                "misses": len(_pins.misses), "digest": _pins.digest},
        memory=_memory_info(),
    )


@app.get("/memory")
async def memory():
    if _device.type != "cuda":
        return {"device": "cpu"}
    return {
        "device": str(_device),
        "allocated_gb": round(torch.cuda.memory_allocated(_device) / 1e9, 3),
        "reserved_gb": round(torch.cuda.memory_reserved(_device) / 1e9, 3),
        "peak_gb": round(torch.cuda.max_memory_allocated(_device) / 1e9, 3),
    }


# ── /replay: a signed record's raw surfaces, re-captured bitwise (classical tools read them offline) ──
# The schema is anamnesis's replay export (anamnesis-replay-export-v1): Tier A, the surfaces the signature reduced,
# under the SignedRow contract's names and axes; the JSON header (provenance and shapes) rides in the safetensors
# metadata under "anamnesis_replay", and alone with format "json".
REPLAY_SCHEMA = "anamnesis-replay-export-v1"
_REPLAY_AXES = {
    "depth": ["step", "layer", "hidden"], "committed": ["block", "step", "hidden"],
    "route": ["step", "layer", "input", "source"], "route_final": ["step", "source"],
    "q": ["step", "sampled_layer", "head", "head_dim"], "k": ["step", "sampled_layer", "kv_head", "head_dim"],
    "v": ["step", "sampled_layer", "kv_head", "head_dim"], "gate": ["step", "sampled_layer", "intermediate"],
    "head_entropy": ["step", "layer", "head"], "mean_entropy": ["step", "layer"],
    "head_scalars": ["step", "sampled_layer", "scalar", "head"], "mean_rows": ["step", "sampled_layer", "column"],
    "logit_stats": ["step", "stat"],
}
# Tier B (on request): derived from Tier A by the engine's own routing rule (kotodama.model.signing.resid_pre)
_REPLAY_EXTRAS = {"resid_pre": ["step", "layer", "hidden"]}


class ReplayRequest(BaseModel):
    sign_id: str = Field(min_length=1)
    format: Literal["safetensors", "json"] = "safetensors"
    # Tier B extras: "resid_pre", each effective layer's attention input (fp32; see signing.resid_pre)
    extras: list[Literal["resid_pre"]] = []


def _replay_export(rec: Any, row: dict, extras: tuple[str, ...] = ()) -> tuple[dict, dict]:
    """``rec`` re-captured on the signer's decoder and checked against its signed row (generation thread): the
    re-reduced features must equal the stored ones bit for bit, or the export is refused (409). Returns the export's
    JSON header and its tensors (CPU)."""
    import numpy as np

    from kotodama.model.signer import deterministic_algorithms

    assert _signer is not None and _sign_spool is not None
    _signer.plans.update(_sign_spool.plans())
    try:
        signed = _signer.capture(rec)
    except ValueError as exc:
        raise HTTPException(409, f"this server cannot replay {rec.request_id}: {exc}") from exc
    identity = _signer.identity
    with deterministic_algorithms():
        reduced = _signer.reducer.reduce([signed], engine=identity,
                                         token_ids=[rec.prompt_ids + rec.generated_ids])[0]
    x = reduced.features
    x = x.detach().float().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, np.float32)
    stored = np.load(Path(row["batch_dir"]) / row["file"])[row["index"]]
    if not np.array_equal(x.reshape(-1).astype(np.float32), stored):
        raise HTTPException(409, f"the replay of {rec.request_id} does not reproduce its signed row (another "
                                 "engine, pins or reducer than signed it); no export")
    tensors: dict[str, torch.Tensor] = {}
    for name in _REPLAY_AXES:
        t = signed.surfaces.get(name)
        if t is None:
            continue
        t = torch.stack(t) if isinstance(t, list) else t
        tensors[name] = t.detach().contiguous().cpu()
    if "resid_pre" in extras:
        from kotodama.model.signing import resid_pre

        tensors["resid_pre"] = resid_pre(signed).contiguous().cpu()
    axes = {**_REPLAY_AXES, **_REPLAY_EXTRAS}
    header = {
        "schema": REPLAY_SCHEMA, "sign_id": rec.request_id, "request_id": rec.request_id,
        "token_ids": list(rec.prompt_ids + rec.generated_ids), "prompt_length": len(rec.prompt_ids),
        "start": signed.start, "positions": signed.positions, "lever_slot": signed.lever_slot,
        "weights_sha256": rec.weights,
        "engine_identity_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True, default=str)
                                                 .encode("utf-8")).hexdigest(),
        "pins_sha256": _pins.digest if _pins is not None else None,
        "lever_plan_sha256": rec.lever, "seed": rec.meta.get("seed"), "profile": rec.meta.get("profile"),
        "lane_id": row.get("lane_id"), "replayed_bitwise": True,
        "layers": {"effective": len(signed.route_sources), "sampled": list(signed.layers)},
        "route_sources": [list(x) for x in signed.route_sources], "final_sources": signed.final_sources,
        "tensors": {name: {"shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch."),
                           "axes": axes[name]} for name, t in tensors.items()},
        "extras": sorted(set(extras)),
        "derived": {"resid_pre": "each effective layer's attention input: its captured pre-attention routing over the "
                                 "blocks committed before it and the partial entering it (the embedding at layer 0, "
                                 "else depth[:, l-1]), fp32; matches the engine's mix to the captured weights' "
                                 "precision"} if "resid_pre" in extras else {},
        "meta": rec.meta,
    }
    return header, tensors


def _sign_waiting(sign_id: str) -> bool:
    """Whether ``sign_id`` is an in-flight row still in memory (pending its reduce, or reduced and not yet written)."""
    with _sign_pending_lock:
        if any(rec.request_id == sign_id for rec, _, _ in _sign_pending):
            return True
    return any(res.record.request_id == sign_id for results, _, _ in _sign_unwritten for res in results)


@app.post("/replay")
async def replay_endpoint(request: ReplayRequest):
    """A signed record's Tier A export (see REPLAY_SCHEMA): its raw surfaces re-captured on this server's signer,
    returned only when the replay reproduces its signed row bit for bit. 404: no such record in this server's
    spool; 409: not signed (yet, or skipped), or not reproducible here."""
    if _signer is None or _sign_spool is None:
        raise HTTPException(400, "/replay needs a server that signs (--sign_spool and --sign_reducer)")
    from kotodama.serve.sign import find

    rec, row = await asyncio.to_thread(find, _sign_spool, request.sign_id)
    if rec is None and _sign_waiting(request.sign_id):
        raise HTTPException(409, f"{request.sign_id} has no signed row: not signed yet (its in-flight batch is "
                                 "waiting to be reduced)")
    if rec is None:
        raise HTTPException(404, f"no record {request.sign_id!r} in this server's spool")
    if "cache_state" in rec.meta:
        raise HTTPException(409, f"{request.sign_id} is a KV session's row: its KV was the session's, which no replay "
                                 "of its ids reproduces (meta.cache_state names its condition)")
    if row is None or "file" not in row:
        why = (row or {}).get("skipped") or "not signed yet (the drain has not reached it)"
        raise HTTPException(409, f"{request.sign_id} has no signed row: {why}")
    async with _engine_lock:
        header, tensors = await _run_generation(_replay_export, rec, row, tuple(request.extras))
    if request.format == "json":
        return header
    from safetensors.torch import save

    blob = save(tensors, metadata={"anamnesis_replay": json.dumps(header, default=str)})
    return Response(content=blob, media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{request.sign_id}.safetensors"'})


@app.post("/logprobs")
async def logprobs_endpoint(request: LogprobRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(503, "Model not loaded")
    # Same dedicated thread as generation: the GPU is single-tenant per replica
    # and eager batch prefill must not interleave with engine activity.
    return await _run_generation(_score_logprobs, request)


@app.post("/score_tokens")
async def score_tokens_endpoint(request: TokenScoreRequest):
    """Per-token NLL of exact token-id continuations under the plain model
    (temperature 1, no penalty, no truncation) — the coherence read for
    sampled text (ids as returned by /generate's completion_ids)."""
    if _model is None or _tokenizer is None:
        raise HTTPException(503, "Model not loaded")
    return await _run_generation(_score_tokens, request)


@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(request: GenerateRequest, http_request: Request):
    if _model is None or _tokenizer is None:
        raise HTTPException(503, "Model not loaded")
    _refuse_unserved(request)
    request = _resolve_sign(request)

    if request.kv_session is not None or request.seed is not None or (request.sign and not request.stream):
        if request.stream:
            raise HTTPException(400, "a seeded request cannot stream")
        out = await _batched_candidates(request, 1)
        if out:
            _note_candidates(out[:1])
            return out[0]
        if _engine_compatible(request):  # a signed request no batched path takes: served serially, drain-signed
            async with _engine_lock:
                result = await _run_generation(engine_generate, request)
        else:
            result = await _run_generation(generate, request=request, model=_model, tokenizer=_tokenizer)
        _sign_serial(request, [result])
        _note_candidates([result])
        return result

    if request.stream:
        if request.sign:
            _check_sign(request)
        if _engine_compatible(request):
            return StreamingResponse(
                engine_generate_stream(request, http_request, sign=request.sign),
                media_type="text/event-stream",
            )
        if request.sign:
            _count_unsigned("streamed by the eager reference path")
        return StreamingResponse(
            generate_stream(_model, _tokenizer, request, http_request),
            media_type="text/event-stream",
        )

    if _engine_compatible(request):
        async with _engine_lock:
            result = await _run_generation(engine_generate, request)
    else:
        result = await _run_generation(generate, request=request, model=_model, tokenizer=_tokenizer)
    _note_candidates([result])
    return result


# ── OpenAI-compatible /v1/completions (for Loom/Loomsidian) ─────────────────────

# Request fields that only route or label a request (the gateway's affinity key): accepted, and they change nothing
# that is generated.
_PASSTHROUGH_FIELDS = frozenset({"user", "log"})  # log: false keeps a request out of --request_log


def _refuse_unserved(request: BaseModel) -> None:
    """400 on a field this server would otherwise accept and ignore: an unknown field that is on (any value but
    null / false / 0 / empty), a nonzero presence_penalty (the repetition law is multiplicative: frequency_penalty
    maps to repetition_penalty), best_of above 1. A client's law must never silently differ from what it sent."""
    bad = sorted(k for k, v in (request.model_extra or {}).items()
                 if k not in _PASSTHROUGH_FIELDS and v not in (False, 0, "", [], {}))
    if getattr(request, "presence_penalty", 0.0):
        bad.append("presence_penalty")
    if (getattr(request, "best_of", None) or 1) > 1:
        bad.append("best_of")
    if bad:
        raise HTTPException(400, f"not served on this endpoint (the server would otherwise ignore them): {bad}")


def _drop_nulls(data: Any) -> Any:
    """OpenAI clients (e.g. Loom) send unset sampling fields as JSON null; treat null as "not sent" so the field's
    default applies instead of a 422."""
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if v is not None}
    return data


class OAICompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # kept so _refuse_unserved can see them
    # Text, or the OpenAI token form: a list of token ids (sent as the prompt's ids, never re-encoded).
    prompt: str | list[int]
    model: str = ""
    max_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=2048)
    n: int = Field(default=1, ge=1, le=64)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=DEFAULT_TOP_P, ge=0.0, le=1.0)
    frequency_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=0.0, le=2.0)
    best_of: int | None = None
    stop: list[str] | str | None = None
    return_token_ids: bool = False
    # Residual steering, as on /generate (needs the fast engine + --steer-npz; else 400).
    vectors: list[SteerVectorSpec] | None = None
    steer_model: str | None = None
    steer_alpha: float | None = None
    steer_rule: Literal["K", "N", "S"] = "K"
    steer_span: Literal["decode", "uniform"] = "decode"
    injection: InjectionSpec | None = None
    penalty_scope: Literal["generated", "row"] = "generated"
    seed: int | None = Field(default=None, ge=0)
    min_p: float = Field(default=0.0, ge=0.0, lt=1.0)
    # Sign the choices (as on /generate; needs --sign_spool): each choice then carries its sign_id. None: the
    # server's default (--sign_default); False opts out.
    sign: bool | None = None
    sign_meta: dict[str, Any] | None = None
    # A KV session's turn, as on /generate.
    kv_session: str | None = None
    append_from: int | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _tolerant(cls, data: Any) -> Any:
        data = _drop_nulls(data)
        if isinstance(data, dict) and isinstance(data.get("prompt"), list):
            p = data["prompt"]
            if p and all(isinstance(t, int) and not isinstance(t, bool) for t in p):
                return data  # token ids
            if len(p) == 1 and isinstance(p[0], list) and p[0] and all(isinstance(t, int) for t in p[0]):
                return {**data, "prompt": p[0]}  # one token-id prompt in a batch of one
            if len(p) != 1 or not isinstance(p[0], str):
                raise ValueError("prompt: send one string or one list of token ids (several prompts are not "
                                 "supported; use n)")
            data = {**data, "prompt": p[0]}
        return data


@torch.inference_mode()
def engine_generate_multi(request: GenerateRequest, n: int) -> list[GenerateResponse]:
    """n candidates of one prompt: one engine prefill (prefix cache included), then the batched decoder. Same
    tokenization, steering, stop and capacity rules as engine_generate per candidate. Caller must hold _engine_lock."""
    dec = _multi_bucket(n, seeded=request.seed is not None)
    assert _engine is not None and dec is not None and _tokenizer is not None
    timing: dict[str, float] = {}
    input_ids = torch.tensor([_prompt_ids(request, _tokenizer)], device=_device)
    prompt_len = input_ids.shape[1]
    if prompt_len >= _max_seq_len:
        raise HTTPException(400, f"Prompt too long: {prompt_len} tokens (max {_max_seq_len})")
    params = _engine_params(request)
    stops = request.stop_strings

    def stop_check(ids: list[int]) -> bool:
        return bool(stops) and any(s in _tokenizer.decode(ids, skip_special_tokens=True) for s in stops)

    t0 = time.perf_counter()
    with _steering(request, timing) as slot:
        prefill_logits, pc_info = _engine_prefill(input_ids, slot, fresh=request.seed is not None)
        timing["prefill_ms"] = (time.perf_counter() - t0) * 1000
        if pc_info is not None:
            timing["prefix_cache_hit"] = pc_info["prefix_hit"]
            timing["prefix_common_tokens"] = pc_info["common_prefix"]
        budget = min(request.max_new_tokens, _max_seq_len - prompt_len)
        ended_on: list[int | None] = [None] * n
        rows = decode_rows(dec, prefill_logits, params, n, budget, _stop_token_ids, stop_check,
                           slots=slot if _levers else None, seed=request.seed, stops=ended_on)
    elapsed = time.perf_counter() - t0
    timing["total_ms"] = round(elapsed * 1000, 2)
    timing["batched_rows"] = float(dec.rows)
    total = sum(len(r) for r in rows)
    tps = round(total / elapsed, 1) if elapsed > 0 else 0.0
    prompt_ids = input_ids[0].tolist()
    return [GenerateResponse(text=_tokenizer.decode(r, skip_special_tokens=True), prompt_tokens=prompt_len,
                             completion_tokens=len(r), tokens_per_second=tps, timing=dict(timing),
                             prompt_ids=prompt_ids, completion_ids=list(r), stop_id=stop)
            for r, stop in zip(rows, ended_on)]


def _multi_bucket(n: int, seeded: bool = False) -> "MultiSampleDecoder | None":
    """The smallest row bucket that holds n candidates (seeded: the widest, so a seed's draws never depend on the
    width its batch happened to take), or None."""
    fits = [r for r in _multi if r >= n]
    if not fits:
        return None
    return _multi[max(fits) if seeded else min(fits)]


def build_cascade_decoder(eng: "DecodeEngine", rows: int, groups: int, tail_tokens: int,
                          kv_from: "MultiSampleDecoder | None" = None) -> "CascadeDecoder":
    """A cascade row decoder of ``rows`` rows. ``groups`` (0 = rows // 4) is the widest bucket's group count; a
    narrower bucket viewing it (``kv_from``) keeps the same rows per group, within the widest one's groups."""
    if kv_from is not None:
        assert isinstance(kv_from, CascadeDecoder)
        groups = min(kv_from.groups, max(1, kv_from.groups * rows // kv_from.rows))
    return CascadeDecoder(eng, rows, groups=groups, tail_tokens=tail_tokens, kv_from=kv_from, seeded=_seeded_rows)


def _without_autotune(mode: str) -> str:
    """``mode`` without its kernel benchmarking (its CUDA-graph choice kept)."""
    if mode.startswith("max-autotune"):
        return "reduce-overhead" if mode == "max-autotune" else "default"
    return mode


def _row_compile_mode(rows: int) -> str:
    """The compile mode of a ``rows``-row decoder: the server's, without kernel benchmarking below
    --autotune_min_rows."""
    return _without_autotune(_compile_mode) if rows < _autotune_min_rows else _compile_mode


def build_and_warm_row_decoder(eng: "DecodeEngine", rows: int, kv_from: "MultiSampleDecoder | None" = None,
                               cascade: tuple[int, int] | None = None) -> "MultiSampleDecoder":
    """A compiled, warmed row decoder over ``eng`` (call on the generation thread: its CUDA graphs are thread-local).
    ``cascade`` = (groups, tail_tokens) builds a ``CascadeDecoder`` (``build_cascade_decoder``), None a per-row one."""
    if cascade is not None:
        dec = build_cascade_decoder(eng, rows, cascade[0], cascade[1], kv_from)
    else:
        dec = MultiSampleDecoder(eng, rows, kv_from=kv_from, seeded=_seeded_rows)
    dec.compile_step(mode=_row_compile_mode(rows), regional=_regional)
    with torch.inference_mode():
        params = EngineSamplingParams(temperature=0.9, repetition_penalty=1.2)
        trunc = EngineSamplingParams(temperature=0.9, repetition_penalty=1.2, top_p=0.9, min_p=0.05)
        for plen in sorted({min(plen, eng.max_seq_len - 8) for plen in [16, 256]}):
            logits = eng.prefill(torch.randint(4, eng.config.vocab_size, (1, plen), device=eng.device))
            for law in (params, trunc) if _warm_truncated else (params,):  # plain (and truncated) step variants
                dec.start(logits, law)
                for _ in range(4):
                    dec.step(law)
        torch.cuda.synchronize(eng.device)
    eng.reset()
    return dec


def _within_row_cap(request: GenerateRequest) -> bool:
    return not _row_token_cap or request.max_new_tokens <= _row_token_cap


def _multi_compatible(request: GenerateRequest, n: int) -> bool:
    return ((n > 1 or request.seed is not None) and _multi_bucket(n) is not None and _within_row_cap(request)
            and MultiSampleDecoder.supports(_engine_params(request)))


def _xbatch_compatible(request: GenerateRequest, n: int) -> bool:
    return (_xbatch is not None and 1 <= n <= _xbatch.runner.max_rows and _within_row_cap(request)
            and MultiSampleDecoder.supports(_engine_params(request)))


def _xbatch_key(request: GenerateRequest, steered: bool) -> tuple:
    """Requests with equal keys may share a batch: the same steering config (with --levers every request carries its
    own lever, so none). Each row carries its own sampling law, so the law is not part of the key."""
    steer = None
    if steered and not _levers:
        steer = (tuple((v.name, v.alpha) for v in request.vectors or ()), request.steer_model, request.steer_alpha,
                 request.steer_rule)
    return (steer,)


async def engine_generate_xbatch(request: GenerateRequest, n: int, signed_out: list | None = None,
                                 session: "tuple[Any, list[int]] | None" = None,
                                 kv_out: list | None = None) -> list[GenerateResponse]:
    """n candidates of one prompt through the cross-request batcher: same tokenization, steering, stop and capacity
    rules as engine_generate per candidate. Bad prompts and steering stacks 400 before the request queues.
    ``signed_out`` (in-flight signing): the job decodes on the signing decoder and the list receives each
    candidate's captured span. ``session`` (a KV session's turn: the session and the turn's new ids): the job prefills
    against the session and ``kv_out`` receives each candidate's decode-written KV."""
    assert _xbatch is not None and _tokenizer is not None
    t0 = time.perf_counter()
    job_prefill = None
    if session is not None:
        from kotodama.serve import kv_sessions

        sess, new_ids = session
        prompt_ids, _ = kv_sessions.turn_ids(sess, new_ids, request.append_from)
        job_prefill = functools.partial(kv_sessions.prefill, _engine, sess, new_ids, request.append_from)
    else:
        prompt_ids = _prompt_ids(request, _tokenizer)
    if len(prompt_ids) >= _max_seq_len:
        raise HTTPException(400, f"Prompt too long: {len(prompt_ids)} tokens (max {_max_seq_len})")
    _check_span(request)
    lever = _request_lever(request) if _levers else None
    steer_writes = _compose_steer_writes(request) if not _levers else None
    steer_kwargs = _steer_rule_kwargs(request) if steer_writes is not None else {}
    params = _engine_params(request)
    steered = steer_writes is not None or lever is not None
    job = XBatchJob(key=_xbatch_key(request, steered), prompt_ids=prompt_ids, n=n,
                    max_new_tokens=request.max_new_tokens, params=params, done=lambda _outcome: None,
                    stop_strings=tuple(request.stop_strings), steer_writes=steer_writes, steer_kwargs=steer_kwargs,
                    lever=lever, seed=request.seed, sign=signed_out is not None, prefill=job_prefill,
                    admitted=(functools.partial(kv_sessions.poison, _engine, session[0].id)
                              if session is not None else None),
                    keep_kv=session is not None)
    result = await _xbatch.submit(job)
    if signed_out is not None:
        signed_out.extend(result.signed)
    if kv_out is not None:
        kv_out.extend(result.kv)
    elapsed = time.perf_counter() - t0
    timing = dict(result.timing)
    if steered:
        timing["steered"] = 1.0
    total = sum(len(r) for r in result.rows)
    tps = round(total / elapsed, 1) if elapsed > 0 else 0.0
    stops = result.stops or [None] * len(result.rows)
    return [GenerateResponse(text=_tokenizer.decode(r, skip_special_tokens=True), prompt_tokens=len(prompt_ids),
                             completion_tokens=len(r), tokens_per_second=tps, timing=dict(timing),
                             prompt_ids=list(prompt_ids), completion_ids=list(r), stop_id=stop)
            for r, stop in zip(result.rows, stops)]


def _resolve_sign(request: GenerateRequest, candidates: int = 1) -> GenerateRequest:
    """``request`` with ``sign`` decided: unset takes the server's default (--sign_default, on a server with a
    spool); an explicit False on a signing-by-default server is counted (``opted_out``, by candidate)."""
    if request.sign is None:
        return request.model_copy(update={"sign": bool(_sign_default_arg and _sign_spool is not None)})
    if request.sign is False and _sign_default_arg:
        _sign_stats["opted_out"] = _sign_stats.get("opted_out", 0) + candidates
    return request


def _count_unsigned(reason: str, candidates: int = 1) -> None:
    """Candidates a signing server generated without a record, by reason (/info sign.stats.unsigned)."""
    unsigned = _sign_stats.setdefault("unsigned", {})
    unsigned[reason] = unsigned.get(reason, 0) + candidates
    logger.warning("sign: %d candidate(s) served unsigned: %s", candidates, reason)


def _sign_serial(request: GenerateRequest, responses: list[GenerateResponse]) -> None:
    """Records for candidates a signed request got from the serial paths (the drain signs them): the fast engine's
    responses carry their ids; the eager reference path's do not carry the stop token, so they are counted
    unsigned instead."""
    if not request.sign or not responses:
        return
    if not _engine_compatible(request):
        _count_unsigned("served by the eager reference path", len(responses))
        return
    _spool_records(request, responses)


def _check_sign(request: GenerateRequest) -> None:
    """400 on a signed request the sign path cannot reproduce: no spool, or broadcast steering (only levers travel
    with a record, by their plan)."""
    if _sign_spool is None:
        raise HTTPException(400, "'sign' needs a server started with --sign_spool")
    if _request_wants_steering(request) and not _levers:
        raise HTTPException(400, "'sign' with steering needs --levers (a record carries its lever plan; the "
                                 "broadcast steering rows do not travel)")


def _spool_records(request: GenerateRequest, responses: list[GenerateResponse]) -> None:
    """One sign record per candidate (kotodama.model.signer.SignRecord) into the spool; each response gets its
    sign_id."""
    assert _sign_spool is not None
    records, plan = _sign_records(request, responses)
    _sign_spool.add(_sign_writer, records, plans=[plan] if plan is not None else None)
    _sign_stats["records"] += len(records)


def _sign_records(request: GenerateRequest, responses: list[GenerateResponse],
                  cache_state: dict | None = None) -> tuple[list, "Any"]:
    """One sign record per candidate (kotodama.model.signer.SignRecord), and the request's lever plan (None:
    unsteered); each response gets its sign_id. The generated ids end with the stop token that ended the candidate,
    if one did: its position was fed."""
    from kotodama.model.signer import SignRecord

    plan = _request_lever(request) if _levers and _request_wants_steering(request) else None
    base = uuid.uuid4().hex[:16]
    law = {"temperature": request.temperature, "repetition_penalty": request.repetition_penalty,
           "penalty_scope": request.penalty_scope, "min_p": request.min_p}
    records = []
    for i, resp in enumerate(responses):
        resp.sign_id = f"{base}-{i}"
        generated = list(resp.completion_ids or []) + ([resp.stop_id] if resp.stop_id is not None else [])
        records.append(SignRecord(request_id=resp.sign_id, prompt_ids=tuple(resp.prompt_ids or ()),
                                  generated_ids=tuple(generated), weights=_sign_weights,
                                  lever=plan.digest() if plan is not None else None,
                                  meta={**(request.sign_meta or {}), "candidate": i, "seed": request.seed,
                                        "law": law, "served_by": _sign_writer, **_sign_provenance(),
                                        # prompt tokens whose KV the prefill took from a prefix cache or pool
                                        "prefix_reused": int((resp.timing or {}).get("prefix_common_tokens", 0)),
                                        # a KV session's row: the session's condition (kotodama.serve.kv_sessions)
                                        **({"cache_state": cache_state} if cache_state is not None else {})}))
    return records, plan


def _sign_add_inflight(request: GenerateRequest, responses: list[GenerateResponse], signed: list,
                       cache_state: dict | None = None) -> None:
    """A signed request's records (as _spool_records names them) and its candidates' captured spans, queued for
    the in-flight reduce. ``cache_state``: a KV session's turn (its rows are signed in flight or not at all)."""
    records, plan = _sign_records(request, responses, cache_state=cache_state)
    if len(signed) != len(records):
        signed = [None] * len(records)
    # a prefill that reused cached KV exports no span (kotodama.serve.xbatch): the drain signs those records cold
    deferred = sum(1 for rec, row in zip(records, signed) if row is None and rec.meta.get("prefix_reused"))
    if deferred:
        reasons = _sign_stats["inflight_deferred"]
        reasons["prefix KV reused"] = reasons.get("prefix KV reused", 0) + deferred
    with _sign_pending_lock:
        _sign_pending.extend((rec, row, plan) for rec, row in zip(records, signed))
    _sign_stats["records"] += len(records)


def _sign_between_steps() -> None:
    """The xbatch runner's hook (generation thread): reduce the pending in-flight rows once a signer's batch of them
    waits."""
    if len(_sign_pending) >= _signer.dec.rows or _sign_unwritten:
        _sign_flush(full_only=True)


def _sign_flush(full_only: bool = False) -> None:
    """Reduce pending in-flight rows, a signer's batch at a time (``full_only``: only full batches), and write each
    reduced batch to the spool (a batch the drain lock holds off waits for the next call). A record whose span was not
    captured but which a drain can sign (a span as long as the sign tail) goes to the spool's records instead, so the
    drain signs it. Generation thread only."""
    global _sign_reducing
    assert _signer is not None and _sign_spool is not None
    while _sign_unwritten:
        results, plans, identity = _sign_unwritten[0]
        summary = _sign_spool.add_signed(_sign_writer, results, identity, plans=plans)
        if summary is None:
            return
        _sign_unwritten.pop(0)
        _sign_count(summary)
    width = _signer.dec.rows
    while True:
        with _sign_pending_lock:
            if not _sign_pending or (full_only and len(_sign_pending) < width):
                return
            take = _sign_pending[:width]
            del _sign_pending[:width]
            _sign_reducing = len(take)
        try:
            _sign_reduce_batch(take)
        finally:
            _sign_reducing = 0
        if _sign_unwritten:
            return


# Time a stopping replica spends reducing its pending in-flight rows (inside the gateway's 10 s termination grace).
_SIGN_SHUTDOWN_S = 7.0


async def _sign_flush_at_shutdown(timeout: float = _SIGN_SHUTDOWN_S) -> None:
    """Write the in-flight rows still waiting for their reduce before the server exits, so a request answered just
    before shutdown keeps its signature: reduced and written in flight when that finishes within ``timeout``, else
    their records go to the spool's records and the drain signs them. A row neither path can keep is counted and
    logged."""
    if _signer is None or _sign_spool is None or not (_sign_pending or _sign_reducing or _sign_unwritten):
        return
    waiting = len(_sign_pending) + _sign_reducing + sum(len(r) for r, _, _ in _sign_unwritten)

    async def flush() -> None:
        async with _engine_lock:
            await _run_generation(_sign_flush)

    try:
        await asyncio.wait_for(flush(), timeout)
    except Exception:  # a timeout or a failed reduce: what is still pending falls back to the drain
        logger.exception("sign: shutdown flush of %d in-flight rows did not finish; their records go to the drain",
                         waiting)
    with _sign_pending_lock:
        left = list(_sign_pending)
        del _sign_pending[:]
    for rec, _, _ in left:
        if "cache_state" in rec.meta:
            _count_session_unsigned(rec, "pending at shutdown")
    records = [(rec, plan) for rec, _, plan in left if "cache_state" not in rec.meta and _signer.why_not(rec) is None]
    if records:
        try:
            _sign_spool.add(_sign_writer, [rec for rec, _ in records],
                            plans=list({p.digest(): p for _, p in records if p is not None}.values()) or None)
        except Exception:
            logger.exception("sign: %d in-flight records could not be written to the spool at shutdown", len(records))
            records = []
    # rows still being reduced or held off by another drain's lock when the server exits are not written
    unfinished = _sign_reducing + sum(len(r) for r, _, _ in _sign_unwritten)
    lost = len(left) - len(records) + unfinished
    if lost:
        _sign_stats["errors"] += 1
        logger.error("sign: %d in-flight rows were neither signed nor spooled before shutdown", lost)
    logger.info("sign: shutdown: %d in-flight rows were pending: %d signed in flight, %d spooled for the drain, "
                "%d lost", waiting, waiting - len(left) - unfinished, len(records), lost)


def _sign_reduce_batch(take: list) -> None:
    """One batch of pending in-flight rows (see _sign_flush): reduced and written, or queued in _sign_unwritten
    while another drain holds the spool. A KV session's rows reduce apart from the others, under the session lane's
    identity (kotodama.serve.kv_sessions.session_identity); one that was not captured is counted unsigned, never
    spooled for the drain (no cold replay of its ids reproduces its KV)."""
    from kotodama.serve.kv_sessions import session_identity

    session = ["cache_state" in rec.meta for rec, _, _ in take]
    for (rec, row, _), sess in zip(take, session):
        if sess and row is None:
            _count_session_unsigned(rec, "its span was not captured")
    drainable = [row is None and not sess and _signer.why_not(rec) is None
                 for (rec, row, _), sess in zip(take, session)]
    to_drain = [(rec, plan) for (rec, _, plan), d in zip(take, drainable) if d]
    if to_drain:
        _sign_spool.add(_sign_writer, [rec for rec, _ in to_drain],
                        plans=list({p.digest(): p for _, p in to_drain if p is not None}.values()) or None)
    cold = [t for t, d, sess in zip(take, drainable, session) if not d and not sess]
    sessioned = [t for t, sess in zip(take, session) if sess and t[1] is not None]
    del take
    for group, extra in ((cold, None), (sessioned, session_identity(_signer.dec.engine))):
        if not group:
            continue
        identity = _signer.identity if extra is None else {**_signer.identity, **extra}
        plans = list({p.digest(): p for _, _, p in group if p is not None}.values())
        _signer.plans.update({p.digest(): p for p in plans})
        t0 = time.perf_counter()
        results = _signer.sign_captured([rec for rec, _, _ in group], [row for _, row, _ in group],
                                        identity=identity if extra is not None else None)
        t1 = time.perf_counter()
        if extra is not None:
            for res in results:
                if res.reduced is None:
                    _count_session_unsigned(res.record, f"not reduced: {res.skipped}")
        summary = _sign_spool.add_signed(_sign_writer, results, identity, plans=plans or None)
        t2 = time.perf_counter()
        _sign_timing["reduce_s"] += t1 - t0
        _sign_timing["write_s"] += t2 - t1
        if summary is None:
            _sign_unwritten.append((results, plans or None, identity))
            logger.info("sign: in-flight batch of %d rows reduced; the spool's drain lock is held, written later",
                        len(results))
            continue
        _sign_count(summary)
        logger.info("sign: in-flight batch %s: %d signed, %d skipped in %.2fs (reduce %.2fs, write %.2fs)%s",
                    summary.get("batch"), summary.get("signed", 0), summary.get("skipped", 0), t2 - t0, t1 - t0,
                    t2 - t1, " (KV session rows)" if extra is not None else "")


def _count_session_unsigned(rec: Any, reason: str) -> None:
    """A KV session's candidate left unsigned: counted in /info (sign.stats.unsigned) and on its session."""
    _count_unsigned(f"KV session row: {reason}")
    sid = (rec.meta.get("cache_state") or {}).get("session")
    if _kv_sessions is not None:
        _kv_sessions.stats["unsigned"] += 1
        try:
            _kv_sessions.get(sid).unsigned += 1
        except ValueError:  # the session closed meanwhile
            pass


def _sign_count(summary: dict, inflight: bool = True) -> None:
    _sign_stats["batches"] += 1
    _sign_stats["signed"] += summary.get("signed", 0)
    _sign_stats["skipped"] += summary.get("skipped", 0)
    reasons = _sign_stats.setdefault("skipped_reasons", {})
    for reason, n in (summary.get("skip_reasons") or {}).items():
        reasons[reason] = reasons.get(reason, 0) + n
    if inflight:
        _sign_stats["inflight_batches"] = _sign_stats.get("inflight_batches", 0) + 1


def _memory_info() -> dict | None:
    """The device's memory as the server holds it (GB): allocated and reserved now and at their peaks since start,
    and the large fixed holders: the engine's and the row decoders' KV, the signing decoder's sign buffers."""
    if _device is None or getattr(_device, "type", None) != "cuda" or not torch.cuda.is_available():
        return None
    gb = 1e9
    out: dict[str, Any] = {
        "allocated_gb": round(torch.cuda.memory_allocated(_device) / gb, 2),
        "reserved_gb": round(torch.cuda.memory_reserved(_device) / gb, 2),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated(_device) / gb, 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved(_device) / gb, 2),
        "device_total_gb": round(torch.cuda.get_device_properties(_device).total_memory / gb, 2),
    }
    if _xbatch is not None:
        widest = _xbatch.runner.decoders[_xbatch.runner.max_rows]
        out["row_kv_gb"] = round(widest.kv_bytes() / gb, 2)  # the narrower buckets view the widest one's storage
    if _signer is not None and _signer.dec.sign is not None:
        dec = _signer.dec
        out["sign_buffers_gb"] = round(sum(getattr(dec, f"sign_{n}").numel() * getattr(dec, f"sign_{n}").element_size()
                                           for n in dec.sign_names) / gb, 2)
    return out


def _sign_provenance() -> dict[str, Any]:
    """What every record names about the server that generated it: its serve profile, its kernel pin table's digest
    (as it stands), the signer's engine identity digest (at startup; None without an in-process signer) and whether
    it signs in flight."""
    return {"profile": _profile_name, "pins": _pins.digest if _pins is not None else None,
            "engine_identity": _sign_engine_digest, "inflight": _sign_inflight_arg}


def _sign_disk() -> dict:
    """Free space on the spool's filesystem against --sign_disk_floor_gb (a statvfs; logged once per crossing)."""
    global _sign_disk_low
    assert _sign_spool is not None
    try:
        free = shutil.disk_usage(_sign_spool.root).free / 1024 ** 3
    except OSError as exc:
        return {"error": str(exc), "floor_gb": _sign_disk_floor_gb_arg}
    low = free < _sign_disk_floor_gb_arg
    if low and not _sign_disk_low:
        logger.warning("sign spool: %s has %.0f GB free, under the %.0f GB floor (archive signed batches: python -m "
                       "kotodama.serve.sign_archive)", _sign_spool.root, free, _sign_disk_floor_gb_arg)
    _sign_disk_low = low
    return {"free_gb": round(free, 1), "floor_gb": _sign_disk_floor_gb_arg, "low": low}


def _sign_info() -> dict | None:
    if _sign_spool is None:
        return None
    return {"spool": str(_sign_spool.root), "writer": _sign_writer, "weights": _sign_weights,
            "disk": _sign_disk(),
            "drains_when_idle": _signer is not None, "inflight": _sign_inflight_arg,
            # rows captured and not yet in a written signed batch (waiting, being reduced, or held off by a drain)
            "pending_inflight": len(_sign_pending) + _sign_reducing + sum(len(r) for r, _, _ in _sign_unwritten),
            "reducer": getattr(_signer.reducer, "reducer_identity", None) if _signer is not None else None,
            "stats": dict(_sign_stats),
            "inflight_timing": {**{k: round(v, 3) for k, v in _sign_timing.items()},
                                **({k: round(_xbatch.runner.stats[k], 3) for k in ("signed_rows", "export_s", "reused_kv_rows")}
                                   if _xbatch is not None else {})} if _sign_inflight_arg else None}


# A replica signs its spool when it has been idle this long (no request holds the engine, nothing queued).
_SIGN_IDLE_S = 2.0


async def _sign_when_idle() -> None:
    """Drain the sign spool one batch at a time whenever the engine is idle (--sign_reducer). A batch runs on the
    generation thread under the engine lock, so a request that arrives meanwhile waits for one batch at most."""
    assert _sign_spool is not None and _signer is not None
    idle_since = time.monotonic()
    while True:
        await asyncio.sleep(0.5)
        busy = _engine_lock.locked() or (_xbatch is not None and len(_xbatch.runner.queue) > 0)
        if busy:
            idle_since = time.monotonic()
            continue
        if time.monotonic() - idle_since < _SIGN_IDLE_S:
            continue
        try:
            if _sign_pending or _sign_unwritten:
                async with _engine_lock:
                    await _run_generation(_sign_flush)
                continue
            if not await asyncio.to_thread(_sign_spool.has_pending):
                continue
            async with _engine_lock:
                summary = await _run_generation(_sign_spool.drain_once, _signer)
        except Exception:  # a failed batch must not kill the loop or the server: log, count, back off
            logger.exception("sign: a spool batch failed; retrying in 30s")
            _sign_stats["errors"] += 1
            await asyncio.sleep(30.0)
            continue
        if summary:
            _sign_count(summary, inflight=False)


async def _batched_candidates(request: GenerateRequest, n: int) -> list[GenerateResponse]:
    """All n candidates from one batched path (cross-request batcher, else the n-row decoder), or [] when neither
    takes the request and the caller generates candidates one at a time. A seeded request that no batched path
    takes is a 400: only the seeded row decoders draw from per-candidate seeds. A request that sets ``sign`` takes the
    batched paths only, and its candidates' records go to the sign spool. A KV session's turn takes the batcher only
    (_session_candidates)."""
    if request.kv_session is not None:
        return await _session_candidates(request, n)
    if request.sign:
        _check_sign(request)
        if _sign_inflight_arg and _xbatch_compatible(request, n):
            signed: list = []
            out = await engine_generate_xbatch(request, n, signed_out=signed)
            _sign_add_inflight(request, out, signed)
            return out
        # not capturable in flight (another law, a candidate past the row cap, ...): generated as unsigned, its
        # records signed by the drain; [] when no batched path takes it (the caller serves it serially and calls
        # _sign_serial)
        out = await _batched_candidates(request.model_copy(update={"sign": False}), n)
        if out:
            _spool_records(request, out)
        return out
    if request.seed is not None:
        if not _seeded_rows:
            raise HTTPException(400, "'seed' needs a server started with --seeded_rows")
        if not (_xbatch_compatible(request, n) or _multi_compatible(request, n)):
            raise HTTPException(400, "'seed' needs a pure-temperature law (temperature > 0, no top-k/top-p) and "
                                     "the batched decoders (--xbatch_rows or --multi_rows)")
    if _xbatch_compatible(request, n):
        return await engine_generate_xbatch(request, n)
    if _multi_compatible(request, n):
        async with _engine_lock:
            return await _run_generation(engine_generate_multi, request, n)
    if _xbatch is not None and MultiSampleDecoder.supports(_engine_params(request)):
        # a law the batcher samples, refused anyway (a candidate longer than the row cap, more candidates than the
        # widest bucket): served one candidate at a time on the single-stream engine
        logger.info("xbatch: n=%d, max_new_tokens=%d served single-stream (row cap %s, widest bucket %d)",
                    n, request.max_new_tokens, _row_token_cap or "none", _xbatch.runner.max_rows)
    return []


def _session_error(exc: Exception) -> HTTPException:
    return HTTPException(getattr(exc, "status", 400), str(exc))


def _session_new_ids(request: GenerateRequest) -> list[int]:
    """A session turn's new tokens: ``prompt_ids`` as sent, or the prompt text encoded with no special tokens (a
    turn continues a sequence; it does not start one). May be empty when ``append_from`` names a candidate."""
    if request.messages is not None:
        raise HTTPException(400, "a KV session's turn sends its new tokens as 'prompt' or 'prompt_ids', not messages")
    if request.prompt_ids is not None:
        if request.prompt is not None:
            raise HTTPException(400, "Pass one of 'prompt_ids' or 'prompt'")
        vocab = int(_model.config.vocab_size) if _model is not None else None
        if vocab is not None and not all(0 <= t < vocab for t in request.prompt_ids):
            raise HTTPException(400, f"prompt_ids must be ids in [0, {vocab})")
        return list(request.prompt_ids)
    return list(_tokenizer.encode(request.prompt or "", add_special_tokens=False))


async def _session_candidates(request: GenerateRequest, n: int) -> list[GenerateResponse]:
    """A KV session's turn (kotodama.serve.kv_sessions): its new tokens prefilled against the session's KV, n
    candidates decoded by the cross-request batcher, each candidate's decode-written KV kept for the next turn. A
    signed turn is signed in flight only."""
    from kotodama.serve import kv_sessions

    if _kv_sessions is None:
        raise HTTPException(400, "'kv_session' needs a server started with --kv_sessions")
    if request.stream:
        raise HTTPException(400, "a KV session's turn cannot stream")
    if _request_wants_steering(request) or request.steer_span != "decode":
        raise HTTPException(400, "a KV session's turn decodes unsteered (steering is not served on sessions)")
    if not _xbatch_compatible(request, n):
        raise HTTPException(400, "a KV session's turn needs the cross-request batcher (--xbatch_rows) and a law it "
                                 "samples, with n within its widest bucket and max_new_tokens within the row cap")
    if request.sign:
        _check_sign(request)
        if not _sign_inflight_arg:
            raise HTTPException(400, "a signed KV session's turn is signed in flight only (--sign_inflight): no "
                                     "drain can replay a session's KV from its ids")
    new_ids = _session_new_ids(request)
    try:
        sess = _kv_sessions.claim(request.kv_session)
    except kv_sessions.SessionError as exc:
        raise _session_error(exc) from exc
    try:
        kv: list = []
        signed: list | None = [] if request.sign else None
        try:
            out = await engine_generate_xbatch(request, n, signed_out=signed, session=(sess, new_ids), kv_out=kv)
        except kv_sessions.SessionError as exc:
            raise _session_error(exc) from exc
        if request.sign:
            _sign_add_inflight(request, out, signed, cache_state=sess.cache_state())
        sess.pending = {i: kv_sessions.Candidate(ids=list(r.completion_ids or []), k=kv[i][0], v=kv[i][1],
                                                 sign_id=r.sign_id)
                        for i, r in enumerate(out) if i < len(kv) and kv[i] is not None}
        _kv_sessions.stats["turns"] += 1
        for r in out:
            r.timing = {**(r.timing or {}), "kv_session_tokens": float(sess.length)}
        return out
    finally:
        _kv_sessions.release(sess)


class KVSessionRequest(BaseModel):
    session: str | None = None
    candidate: int | None = Field(default=None, ge=0)
    start: int | None = Field(default=None, ge=0)
    stop: int | None = Field(default=None, ge=0)


def _need_sessions() -> "Any":
    from kotodama.serve import kv_sessions

    if _kv_sessions is None:
        raise HTTPException(400, "KV sessions need a server started with --kv_sessions")
    return kv_sessions


async def _session_op(session_id: str | None, fn: Callable[..., Any], *args: Any) -> Any:
    """Run ``fn(session, *args)`` on the generation thread, the session claimed for its duration."""
    kv = _need_sessions()
    if not session_id:
        raise HTTPException(400, "name the 'session'")
    try:
        sess = _kv_sessions.claim(session_id)
    except kv.SessionError as exc:
        raise _session_error(exc) from exc
    try:
        async with _engine_lock:
            return await _run_generation(fn, sess, *args)
    except kv.SessionError as exc:
        raise _session_error(exc) from exc
    finally:
        _kv_sessions.release(sess)


@app.post("/kv_session/open")
async def kv_session_open(request: KVSessionRequest):
    """Open a KV session (its id: ``session``, or a fresh one); it starts empty."""
    kv = _need_sessions()
    try:
        sess = _kv_sessions.open(request.session)
    except kv.SessionError as exc:
        raise _session_error(exc) from exc
    return {"session": sess.id, "max_seq_len": _max_seq_len}


@app.post("/kv_session/close")
async def kv_session_close(request: KVSessionRequest):
    kv = _need_sessions()
    try:
        _kv_sessions.close(request.session or "")
    except kv.SessionError as exc:
        raise _session_error(exc) from exc
    return {"closed": request.session}


@app.get("/kv_session/{session_id}")
async def kv_session_info(session_id: str):
    """A session's state: its length, unfed tokens, the last turn's candidates, its cache state and op log, its
    token ids and its unsigned-row count."""
    kv = _need_sessions()
    try:
        sess = _kv_sessions.get(session_id)
    except kv.SessionError as exc:
        raise _session_error(exc) from exc
    return {**sess.info(), "ids": list(sess.ids), "unfed_ids": list(sess.unfed),
            "candidates": {c: {"ids": cand.ids, "fed": cand.fed, "sign_id": cand.sign_id}
                           for c, cand in sess.pending.items()},
            "ops": sess.cache_state()["ops"]}


@app.post("/kv_session/append")
async def kv_session_append(request: KVSessionRequest):
    """Join ``candidate`` of the last turn to the session (as a turn's ``append_from`` does), so session ops can
    run on it before the next turn."""
    kv = _need_sessions()
    if request.candidate is None:
        raise HTTPException(400, "name the 'candidate'")
    await _session_op(request.session, kv.append, request.candidate)
    return _kv_sessions.get(request.session).info()


@app.post("/kv_session/evict")
async def kv_session_evict(request: KVSessionRequest):
    """Drop positions [start, stop) and renumber the survivors, each moved key re-rotated through the model's RoPE
    table (kotodama.model.session_kv.evict_rerotate)."""
    kv = _need_sessions()
    if request.start is None or request.stop is None:
        raise HTTPException(400, "name 'start' and 'stop'")
    await _session_op(request.session, functools.partial(kv.evict, _engine), request.start, request.stop)
    return _kv_sessions.get(request.session).info()


@app.post("/kv_session/export")
async def kv_session_export(request: KVSessionRequest):
    """Positions [start, stop) as safetensors (k, v; their ids and rotation in the metadata)."""
    kv = _need_sessions()
    if request.start is None or request.stop is None:
        raise HTTPException(400, "name 'start' and 'stop'")
    blob = await _session_op(request.session, kv.export, request.start, request.stop)
    return Response(content=blob, media_type="application/octet-stream")


@app.post("/kv_session/import")
async def kv_session_import(http_request: Request, session: str, at: int, rows_from: int | None = None):
    """Insert an export's rows before position ``at`` (the body: the export's bytes; ``rows_from`` overrides where
    its keys are rotated to). The rows move as a block, re-rotated to their new positions."""
    kv = _need_sessions()
    try:
        k, v, ids, frm, meta = kv.parse_rows(await http_request.body())
    except kv.SessionError as exc:
        raise _session_error(exc) from exc
    vocab = int(_model.config.vocab_size) if _model is not None else None
    if vocab is not None and not all(0 <= t < vocab for t in ids):
        raise HTTPException(400, f"the export's ids must be in [0, {vocab})")
    await _session_op(session, functools.partial(kv.import_, _engine), k, v, ids, at,
                      frm if rows_from is None else rows_from, meta.get("session"))
    return _kv_sessions.get(session).info()


def oai_completion_to_generate(request: OAICompletionRequest) -> GenerateRequest:
    """The /generate request one /v1/completions choice runs: sampling, stops and steering carried over."""
    stop_strings: list[str] = []
    if isinstance(request.stop, str):
        stop_strings = [request.stop]
    elif isinstance(request.stop, list):
        stop_strings = request.stop
    ids = request.prompt if isinstance(request.prompt, list) else None
    return GenerateRequest(
        prompt=None if ids is not None else request.prompt,
        prompt_ids=ids,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=resolve_oai_repetition_penalty(request.frequency_penalty),
        stop_strings=stop_strings,
        vectors=request.vectors,
        steer_model=request.steer_model,
        steer_alpha=request.steer_alpha,
        steer_rule=request.steer_rule,
        steer_span=request.steer_span,
        injection=request.injection,
        penalty_scope=request.penalty_scope,
        seed=request.seed,
        min_p=request.min_p,
        sign=request.sign,
        sign_meta=request.sign_meta,
        kv_session=request.kv_session,
        append_from=request.append_from,
    )


@app.post("/v1/completions")
async def oai_completions(request: OAICompletionRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(503, detail="Model not loaded")
    _refuse_unserved(request)

    choices = []
    gen_req = _resolve_sign(oai_completion_to_generate(request), request.n)
    results = await _batched_candidates(gen_req, request.n)
    if not results:
        for _ in range(request.n):
            if _engine_compatible(gen_req):
                async with _engine_lock:
                    results.append(await _run_generation(engine_generate, gen_req))
            else:
                results.append(await _run_generation(generate, model=_model, tokenizer=_tokenizer, request=gen_req))
        _sign_serial(gen_req, results)
    _note_candidates(results)
    for i, result in enumerate(results):
        choice = {
            "text": html.unescape(result.text),
            "index": i,
            "logprobs": None,
            "finish_reason": "length" if result.completion_tokens >= request.max_tokens else "stop",
        }
        if request.return_token_ids:
            choice["token_ids"] = {"prompt": result.prompt_ids,
                                   "completion": result.completion_ids}
        if result.sign_id is not None:
            choice["sign_id"] = result.sign_id
        choices.append(choice)

    prompt_tokens = (len(request.prompt) if isinstance(request.prompt, list)
                     else _tokenizer.encode(request.prompt, return_tensors="pt").shape[1])
    completion_tokens = sum(
        _tokenizer.encode(c["text"], return_tensors="pt").shape[1] for c in choices
    )

    return {
        "id": f"cmpl-kotodama-{int(time.time())}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": _model_name(),
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.get("/v1/models")
async def oai_models():
    # One model is loaded per process; report the one actually being served.
    # Steering aliases ride in a separate non-standard key rather than as
    # extra data[] entries — the gateway routes on the model field, and fake
    # model ids would be routable.
    active = _model_name()
    resp: dict[str, Any] = {
        "object": "list",
        "data": [
            {
                "id": active,
                "object": "model",
                "owned_by": "aethera-gp",
                "active": True,
            }
        ],
    }
    if _steer_loaded:
        resp["steering"] = {
            "sites": sorted({s for s, _ in _steer_bank.values()}),
            "aliases": sorted(_steer_aliases),
            "keys": sorted(_steer_bank),
        }
    return resp


# ── OpenAI-compatible /v1/chat/completions (chat mode) ────────────────────────


class OAIChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # kept so _refuse_unserved can see them
    messages: list[ChatMessage]
    model: str = ""
    max_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=2048)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=DEFAULT_TOP_P, ge=0.0, le=1.0)
    frequency_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=0.0, le=2.0)
    stop: list[str] | str | None = None
    stream: bool = False
    n: int = Field(default=1, ge=1, le=64)
    return_token_ids: bool = False
    # Sign the choices (as on /generate; needs --sign_spool): each choice then carries its sign_id. None: the
    # server's default (--sign_default); False opts out.
    sign: bool | None = None
    sign_meta: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _tolerant(cls, data: Any) -> Any:
        return _drop_nulls(data)


@app.post("/v1/chat/completions")
async def oai_chat_completions(request: OAIChatRequest, http_request: Request):
    if _model is None or _tokenizer is None:
        raise HTTPException(503, detail="Model not loaded")
    if _serve_mode != "chat":
        raise HTTPException(400, detail="/v1/chat/completions requires --mode chat")
    _refuse_unserved(request)

    msgs = [{"role": m.role, "content": m.content} for m in request.messages]
    prompt_text = _tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    )

    stop_strings: list[str] = []
    if isinstance(request.stop, str):
        stop_strings = [request.stop]
    elif isinstance(request.stop, list):
        stop_strings = request.stop

    repetition_penalty = resolve_oai_repetition_penalty(request.frequency_penalty)

    if request.stream:
        gen_req = GenerateRequest(
            prompt=prompt_text,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=repetition_penalty,
            stop_strings=stop_strings,
            stream=True,
            sign=request.sign,
            sign_meta=request.sign_meta,
        )
        gen_req = _resolve_sign(gen_req)
        if gen_req.sign:
            _check_sign(gen_req)

        if _engine_compatible(gen_req):
            source_stream = engine_generate_stream(gen_req, http_request, sign=gen_req.sign)
        else:
            if gen_req.sign:
                _count_unsigned("streamed by the eager reference path")
            source_stream = generate_stream(_model, _tokenizer, gen_req, http_request)

        async def chat_stream():
            async for chunk in source_stream:
                data = json.loads(chunk.removeprefix("data: ").strip())
                if "token" in data:
                    oai_chunk = {
                        "id": f"chatcmpl-kotodama-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": _model_name(),
                        "choices": [{
                            "index": 0,
                            "delta": {"content": data["token"]},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(oai_chunk)}\n\n"
                elif data.get("done"):
                    final_chunk = {
                        "id": f"chatcmpl-kotodama-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": _model_name(),
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

        return StreamingResponse(chat_stream(), media_type="text/event-stream")

    choices = []
    total_completion_tokens = 0
    gen_req = GenerateRequest(
        prompt=prompt_text,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=repetition_penalty,
        stop_strings=stop_strings,
        sign=request.sign,
        sign_meta=request.sign_meta,
    )
    gen_req = _resolve_sign(gen_req, request.n)
    results = await _batched_candidates(gen_req, request.n)
    if not results:
        for _ in range(request.n):
            if _engine_compatible(gen_req):
                async with _engine_lock:
                    results.append(await _run_generation(engine_generate, gen_req))
            else:
                results.append(await _run_generation(generate, model=_model, tokenizer=_tokenizer, request=gen_req))
        _sign_serial(gen_req, results)
    _note_candidates(results)
    for i, result in enumerate(results):
        total_completion_tokens += result.completion_tokens
        choice = {
            "index": i,
            "message": {"role": "assistant", "content": html.unescape(result.text)},
            "finish_reason": "length" if result.completion_tokens >= request.max_tokens else "stop",
        }
        if request.return_token_ids:
            choice["token_ids"] = {"prompt": result.prompt_ids,
                                   "completion": result.completion_ids}
        if result.sign_id is not None:
            choice["sign_id"] = result.sign_id
        choices.append(choice)

    prompt_tokens = _tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").shape[1]

    return {
        "id": f"chatcmpl-kotodama-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_name(),
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": prompt_tokens + total_completion_tokens,
        },
    }


# ── CLI ─────────────────────────────────────────────────────────────────────────

_device_arg = "cuda"

def build_arg_parser() -> argparse.ArgumentParser:
    """The server's command line (every flag a serve profile may set)."""
    parser = argparse.ArgumentParser(description="kotodama inference server")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint .pt/.pt.zst file")
    parser.add_argument("--device", default="cuda", help="Device: cuda, cuda:N, or cpu")
    parser.add_argument("--mode", choices=["base", "chat"], default=None,
                        help="Serving mode (auto-detected from checkpoint path if omitted)")
    parser.add_argument("--engine", choices=["fast", "reference"], default="fast",
                        help="'fast' (default) = static-cache CUDA-graph DecodeEngine, all "
                             "sampling laws incl. top-k/top-p; 'reference' = eager debug/"
                             "parity path (~10x slower decode)")
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="Serving context limit (16-4096 for the 3B model). Lower values "
                             "reduce the fast engine's static KV-cache memory; default is 4096")
    parser.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=False,
                        help="Reuse the engine KV cache across requests sharing a token-exact "
                             "prompt prefix (multi-turn TTFT win; requires --engine fast and "
                             "per-conversation replica pinning)")
    parser.add_argument("--model_size", choices=list(MODEL_CONFIGS) + [a for a, k in presets.ALIASES.items() if k in MODEL_CONFIGS], default="3b",
                        help="Model architecture size (3b=2.97B, 1b=1.42B ladder, proxy=108M legacy)")
    parser.add_argument("--served_model_name", default=None,
                        help="Override the model id reported by /info and /v1/models "
                             "(default: kotodama-<size>-<base|instruct>)")
    parser.add_argument("--warmup-sdpa", action=argparse.BooleanOptionalAction, default=False,
                        help="Reference engine only: pre-pay the ~100s cuDNN SDPA plan-cache "
                             "sweep at startup instead of ~300ms per novel shape at runtime")
    parser.add_argument("--steer-npz", default=None,
                        help="COMMA-SEPARATED npz banks of unit steering vectors + per-site "
                             "median norms (vector keys like "
                             "'Alg_echo_base_s46' + 'median_norm_s46'; per-key site from the "
                             "_s{sublayer} suffix, sublayer = 2*layer after-attn). Enables "
                             "per-request multi-site residual steering on the fast engine. "
                             "See kotodama/serve/steering.py")
    parser.add_argument("--attn_res", choices=["config", "on", "off"], default="config",
                        help="Override the size-config's attn_res (proxy 2x2 arms: "
                             "baseline/NCA-only checkpoints have no AttnRes params)")
    parser.add_argument("--attn_res_boundaries", default=None,
                        help="Override boundaries, comma ints (proxy DD-v1: 0,3,7,12,21,25)")
    parser.add_argument("--rope_fp32", action=argparse.BooleanOptionalAction, default=False,
                        help="Model config rope_fp32=True (the 1B ladder arms trained with it)")
    parser.add_argument("--default_repetition_penalty", type=float,
                        default=DEFAULT_REPETITION_PENALTY,
                        help="Repetition penalty used when a request does not set one "
                             "(/generate without repetition_penalty, OpenAI routes without "
                             "frequency_penalty). 1.0 = off; range [1.0, 2.0]")
    parser.add_argument("--multi_rows", default="",
                        help="Batched multi-sample decode for OpenAI-route n > 1 (pure-temperature laws): comma row "
                             "buckets, e.g. 4,8 — n candidates decode in lockstep after one prefill on the smallest "
                             "bucket that fits (~0.9 GB of KV per row at 4096 context on the 1B). Empty = off (serial)")
    parser.add_argument("--xbatch_rows", default="",
                        help="Cross-request batched decode for the OpenAI routes (pure-temperature laws): comma row "
                             "buckets, e.g. 16,32,64 — concurrent requests with the same sampling law and steering "
                             "config decode together, each filled from its own prefill, on the narrowest bucket that "
                             "holds their rows. All buckets (and --multi_rows) share the widest one's KV memory. "
                             "Empty = off")
    parser.add_argument("--prefill_buckets", default="",
                        help="The compiled prefill chunk widths, comma tokens, e.g. 1024,256,64,16 (default "
                             "256,64,16). With batch-invariant blocks (CUDA, flash-attn) every prompt runs through "
                             "them, so wider buckets prefill long prompts in fewer chunks; each costs a compile")
    parser.add_argument("--xbatch_refill", action=argparse.BooleanOptionalAction, default=False,
                        help="With --xbatch_rows: refill rows freed by finished requests from the queue between "
                             "decode steps (continuous batching) instead of waiting for the batch's longest row")
    parser.add_argument("--prefix_pool_tokens", type=int, default=0,
                        help="Multi-entry prefix KV cache: pool this many prompt tokens of KV (fixed-size blocks, "
                             "least recently used evicted) so a prompt sharing a prefix with ANY recently prefilled "
                             "prompt prefills only its suffix (~0.22 MB per token on the 1B at bf16). Implies "
                             "--prefix-cache. 0 = off")
    parser.add_argument("--levers", type=int, default=0,
                        help="Per-request steering: lever tables with this many slots (kotodama.model.levers). Every "
                             "steered request binds its own slot, so steered and unsteered requests batch together, "
                             "a request may steer its prompt (steer_span uniform) or send a raw 'injection'; "
                             "--steer-npz banks resolve into levers. 0 = off (the broadcast steering rows)")
    parser.add_argument("--compile_store", default=os.environ.get("KOTODAMA_COMPILE_STORE") or None,
                        help="Directory of compiled engine artifacts keyed by model code, runtime, GPU and engine flags "
                             "(kotodama.serve.compile_cache): a stored key is loaded before compiling (the compiles "
                             "become cache hits), a missing one is saved after warmup. Default $KOTODAMA_COMPILE_STORE; "
                             "unset = off")
    parser.add_argument("--compile_mode", default="max-autotune",
                        choices=["max-autotune", "reduce-overhead", "max-autotune-no-cudagraphs", "default"],
                        help="torch.compile mode of the engine and the row decoders: max-autotune benchmarks kernel "
                             "choices for every graph (slowest to start, fastest steps); reduce-overhead keeps the "
                             "CUDA graphs and skips the benchmarking")
    parser.add_argument("--warm_truncated", action=argparse.BooleanOptionalAction, default=True,
                        help="Warmup compiles the truncated (top-p / min-p) step variants; --no-warm_truncated leaves "
                             "each to its first request, which then waits for that compile")
    parser.add_argument("--regional", action=argparse.BooleanOptionalAction, default=True,
                        help="Regional compile (kotodama.model.regional): the engine and the row decoders compile one "
                             "graph per layer signature instead of one graph of the whole depth per variant and "
                             "width, and capture each step whole as one CUDA graph; a fraction of the compile time at "
                             "the same or better step time. --no-regional compiles the whole-step graphs (a model "
                             "whose norms the regions cannot run falls back to them, logged)")
    parser.add_argument("--emulate_casts", action=argparse.BooleanOptionalAction, default=False,
                        help="Compiled decode regions round intermediates as eager does (kotodama.model.regional."
                             "region_options): a signing decoder then decodes the tokens an unsigned one does, bit for "
                             "bit, under one --kernel_pins table (~4%% of a step)")
    parser.add_argument("--autotune_min_rows", type=int, default=0,
                        help="Row buckets narrower than this compile without kernel benchmarking (max-autotune's "
                             "choices buy nothing at narrow widths on the regional step, and cost minutes each); "
                             "0 = every bucket compiles under --compile_mode")
    parser.add_argument("--startup_budget_s", type=float, default=1800.0,
                        help="Seconds from process start to ready this server should take; a start over it is logged "
                             "as a warning and reported in /info (startup). 0 = no budget")
    parser.add_argument("--seeded_rows", action=argparse.BooleanOptionalAction, default=False,
                        help="Row decoders draw each candidate's noise from its own seed and step: a request's 'seed' "
                             "then reproduces its candidates whatever shares their batch. Seeded requests prefill cold "
                             "and decode on the widest bucket (decoders of different widths round the logits "
                             "differently, which can flip a draw). Without it a 'seed' is a 400")
    parser.add_argument("--prefix_pool_block", type=int, default=16,
                        help="Token block size of the prefix KV pool (>= 16): prefixes are shared in whole blocks")
    parser.add_argument("--cascade_attn", action=argparse.BooleanOptionalAction, default=False,
                        help="Row decoders (--multi_rows, --xbatch_rows) share each prompt's KV among its rows and "
                             "attend to it once per group (cascade attention, merged by log-sum-exp): less attention "
                             "time per step and KV memory per row. A row then generates at most "
                             "--cascade_tail_tokens tokens; requests asking for more take the serial path. Start the "
                             "buckets at 16: on the 1B the 8-row cascade step compiles to a slower step than the "
                             "16-row one")
    parser.add_argument("--cascade_groups", type=int, default=0,
                        help="With --cascade_attn: prompts the widest bucket holds at once (each up to 32 rows; "
                             "~0.9 GB of KV per prompt slot on the 1B at 4096 context). 0 = rows // 4")
    parser.add_argument("--cascade_tail_tokens", type=int, default=1024,
                        help="With --cascade_attn: tokens a row can generate (its own KV slot)")
    parser.add_argument("--kernel_pins", default=None,
                        help="Kernel pin table (kotodama.model.kernel_pins): replayed when the file exists, else "
                             "recorded at startup and saved, so every compiled kernel launches the same config in "
                             "every process (restart-reproducible arithmetic; needed for signing). A kernel missing "
                             "from a replayed table takes its first config, and /info reports the misses")
    parser.add_argument("--sign_spool", default=None,
                        help="Sign spool directory (kotodama.serve.sign): requests that set `sign` leave one record "
                             "per candidate there, signed later by a drain (python -m kotodama.serve.sign) or, with "
                             "--sign_reducer, by this server when idle")
    parser.add_argument("--sign_reducer", default=None,
                        help="With --sign_spool: a SignatureReducer (package.module:attr) this server signs its spool "
                             "with whenever it is idle (a 128-row signing decoder beside the serving ones)")
    parser.add_argument("--sign_reducer_kwargs", default=None, help="JSON keyword arguments for --sign_reducer")
    parser.add_argument("--sign_layers", default="",
                        help="With --sign_reducer: the sampled effective layers, comma-separated")
    parser.add_argument("--sign_tail", type=int, default=256,
                        help="With --sign_reducer: positions signed per row (a longer span is skipped)")
    parser.add_argument("--sign_width", type=int, default=0,
                        help="With --sign_reducer: head-mean row columns (0 = the serving context limit)")
    parser.add_argument("--sign_rows", type=int, default=128,
                        help="With --sign_reducer: the signing decoder's width (the lane's canonical 128)")
    parser.add_argument("--sign_groups", type=int, default=16,
                        help="With --sign_reducer: prompts one signing batch holds")
    parser.add_argument("--sign_inflight", action=argparse.BooleanOptionalAction, default=False,
                        help="With --sign_reducer and --xbatch_rows (cascade): sign requests as they generate. A "
                             "signed request decodes on a signing decoder with the widest bucket's geometry (the "
                             "tokens the unsigned bucket decodes), each candidate's span is captured as it ends and "
                             "reduced in this process, and its signatures land in the spool as signed batches (the "
                             "records too); the idle drain signs on the same decoder")
    parser.add_argument("--request_log", default=None, metavar="DIR",
                        help="Write one JSON line per generation or scoring request to DIR (kotodama.serve.request_log): "
                             "the request as sent, the response, latency, the weights digest and each candidate's "
                             "exact token ids and sign id, in daily files compressed once their day is over. Off "
                             "unless given; a request's \"log\": false keeps it out. A gateway fleet turns it on for "
                             "every replica with serve_args: [\"--request_log\", \"DIR\"]")
    parser.add_argument("--kv_sessions", type=int, default=0, metavar="N",
                        help="Hold up to N KV sessions (kotodama.serve.kv_sessions): conversations whose KV lives "
                             "across their requests (a turn sends only its new tokens with 'kv_session'; "
                             "/kv_session/{open,close,append,evict,export,import} work on a session between turns). "
                             "Needs --xbatch_rows. 0 (default): off")
    parser.add_argument("--kv_session_idle_s", type=float, default=3600.0,
                        help="A KV session idle this long is closed when a new session needs its room")
    parser.add_argument("--sign_default", action=argparse.BooleanOptionalAction, default=False,
                        help="Sign every generation request that does not set `sign` (an explicit `sign: false` "
                             "opts out and is counted); /logprobs and /score_tokens are never signed. A server that "
                             "signs (this, or --sign_inflight) takes the site's signer where its flags leave it unset: "
                             "$KOTODAMA_SIGN_SPOOL, $KOTODAMA_SIGN_REDUCER, $KOTODAMA_SIGN_REDUCER_KWARGS, and "
                             "$KOTODAMA_SIGN_PYTHONPATH (where the reducer's package lives, put on sys.path) and "
                             "$KOTODAMA_KERNEL_PINS (the fleet's kernel pin table); one that signs by default refuses "
                             "to start without a spool, a reducer, --sign_layers and an existing pin table (replayed, "
                             "never recorded: every replica's rows are then one lane)")
    parser.add_argument("--sign_audit_every", type=int, default=100,
                        help="With --sign_reducer: keep the raw signed surfaces of 1 row in N (0: none)")
    parser.add_argument("--sign_disk_floor_gb", type=float, default=150.0,
                        help="With --sign_spool: /info's sign.disk reads low (and a warning is logged) while the "
                             "spool's filesystem has less than this free; python -m kotodama.serve.sign_archive moves "
                             "signed batches to long-term storage")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--profile", default=None,
                        help="Serve profile: a name in configs/serve/ (e.g. tsumugi-1b) or a YAML path whose keys are "
                             "these flags; it supplies their defaults and the server's environment "
                             "(kotodama.serve.profiles), and any flag given here overrides it")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    _profile_parser = argparse.ArgumentParser(add_help=False)
    _profile_parser.add_argument("--profile", default=None)
    _profile_arg = _profile_parser.parse_known_args()[0].profile
    if _profile_arg:
        from kotodama.serve import profiles as _profiles
        _profiles.apply(parser, _profile_arg)
    args = parser.parse_args()

    _checkpoint_path = args.checkpoint
    inferred_mode = "chat" if "instruct" in args.checkpoint else "base"

    _served_name_override = args.served_model_name
    _device_arg = args.device
    _mode_arg = args.mode if args.mode is not None else inferred_mode
    _model_size_arg = args.model_size
    _attn_res_override = args.attn_res
    _attn_res_boundaries_override = args.attn_res_boundaries
    _rope_fp32_override = args.rope_fp32
    _engine_arg = args.engine
    _max_seq_len_arg = args.max_seq_len
    _prefix_cache_arg = args.prefix_cache
    _warmup_sdpa_arg = args.warmup_sdpa
    if not 1.0 <= args.default_repetition_penalty <= 2.0:
        parser.error("--default_repetition_penalty must be in [1.0, 2.0]")
    _default_repetition_penalty = args.default_repetition_penalty
    if _prefix_cache_arg and _engine_arg != "fast":
        logger.warning("--prefix-cache requires --engine fast; ignoring")
        _prefix_cache_arg = False

    if args.steer_npz is not None and _engine_arg != "fast":
        # Unlike --prefix-cache this is FATAL, not ignored: a steering server
        # that silently serves unsteered generations poisons downstream data.
        parser.error("--steer-npz requires --engine fast")
    _steer_npz_arg = args.steer_npz
    try:
        _multi_rows_arg = tuple(int(x) for x in args.multi_rows.split(",") if x.strip())
    except ValueError:
        parser.error(f"--multi_rows: comma-separated ints, got {args.multi_rows!r}")
    if any(r < 2 for r in _multi_rows_arg):
        parser.error("--multi_rows buckets must be >= 2")
    if _multi_rows_arg and _engine_arg != "fast":
        parser.error("--multi_rows requires --engine fast")
    try:
        _xbatch_rows_arg = tuple(int(x) for x in args.xbatch_rows.split(",") if x.strip())
    except ValueError:
        parser.error(f"--xbatch_rows: comma-separated ints, got {args.xbatch_rows!r}")
    if any(r < 2 for r in _xbatch_rows_arg):
        parser.error("--xbatch_rows buckets must be >= 2")
    try:
        _prefill_buckets_arg = tuple(int(x) for x in args.prefill_buckets.split(",") if x.strip())
    except ValueError:
        parser.error(f"--prefill_buckets: comma-separated ints, got {args.prefill_buckets!r}")
    if any(b < 1 for b in _prefill_buckets_arg):
        parser.error("--prefill_buckets must be >= 1")
    if _xbatch_rows_arg and _engine_arg != "fast":
        parser.error("--xbatch_rows requires --engine fast")
    if args.xbatch_refill and not _xbatch_rows_arg:
        parser.error("--xbatch_refill needs --xbatch_rows")
    _xbatch_refill_arg = args.xbatch_refill
    if args.prefix_pool_tokens < 0 or args.prefix_pool_block < 16:
        parser.error("--prefix_pool_tokens must be >= 0 and --prefix_pool_block >= 16")
    if args.prefix_pool_tokens and _engine_arg != "fast":
        parser.error("--prefix_pool_tokens requires --engine fast")
    if 0 < args.prefix_pool_tokens < args.prefix_pool_block:
        parser.error("--prefix_pool_tokens must hold at least one --prefix_pool_block")
    _prefix_pool_tokens_arg = args.prefix_pool_tokens
    _prefix_pool_block_arg = args.prefix_pool_block
    if args.cascade_attn and not (_multi_rows_arg or _xbatch_rows_arg):
        parser.error("--cascade_attn needs --multi_rows or --xbatch_rows")
    if args.cascade_groups < 0 or args.cascade_tail_tokens < 1:
        parser.error("--cascade_groups must be >= 0 and --cascade_tail_tokens >= 1")
    _cascade_attn_arg = args.cascade_attn
    _cascade_groups_arg = args.cascade_groups
    _cascade_tail_tokens_arg = args.cascade_tail_tokens
    if args.levers < 0 or (args.levers and _engine_arg != "fast"):
        parser.error("--levers needs the fast engine and a slot count >= 0")
    _levers_arg = args.levers
    _seeded_rows_arg = args.seeded_rows
    _compile_store_arg = args.compile_store
    _compile_mode_arg = args.compile_mode
    _warm_truncated_arg = args.warm_truncated
    _regional_arg = args.regional
    _startup_budget_s_arg = args.startup_budget_s
    _autotune_min_rows_arg = args.autotune_min_rows
    _emulate_casts_arg = args.emulate_casts
    _kernel_pins_arg = args.kernel_pins
    if args.sign_default or args.sign_inflight:  # the site's signer, where the flags leave it unset
        args.kernel_pins = args.kernel_pins or os.environ.get("KOTODAMA_KERNEL_PINS") or None
        args.sign_spool = args.sign_spool or os.environ.get("KOTODAMA_SIGN_SPOOL") or None
        args.sign_reducer = args.sign_reducer or os.environ.get("KOTODAMA_SIGN_REDUCER") or None
        if args.sign_reducer_kwargs is None:
            args.sign_reducer_kwargs = os.environ.get("KOTODAMA_SIGN_REDUCER_KWARGS") or None
        for extra in (os.environ.get("KOTODAMA_SIGN_PYTHONPATH") or "").split(os.pathsep):
            if extra and args.sign_reducer and extra not in sys.path:
                sys.path.append(extra)  # after the server's own packages: nothing there shadows them
    args.sign_reducer_kwargs = args.sign_reducer_kwargs or "{}"
    if args.sign_default and not (args.sign_spool and args.sign_reducer and args.sign_layers):
        parser.error("this server signs every request (--sign_default, set by its profile) but has no "
                     f"{'sign spool' if not args.sign_spool else 'reducer' if not args.sign_reducer else 'sign layers'}"
                     ": set KOTODAMA_SIGN_SPOOL, KOTODAMA_SIGN_REDUCER and KOTODAMA_SIGN_REDUCER_KWARGS (site.env), or "
                     "pass --no-sign_default to serve without signatures")
    if args.sign_default and not (args.kernel_pins and Path(args.kernel_pins).is_file()):
        # every replica of a fleet replays one table, so their rows are one lane; a missing table would be recorded
        # here, and this server would sign a lane of its own
        parser.error("this server signs every request (--sign_default) but has "
                     f"{'no kernel pin table' if not args.kernel_pins else f'no pin table at {args.kernel_pins}'}: "
                     "set KOTODAMA_KERNEL_PINS (site.env) to the fleet's canonical replay-only table, or pass "
                     "--no-sign_default to serve without signatures")
    _kernel_pins_arg = args.kernel_pins  # with the site's table filled in above
    if args.sign_reducer and not args.sign_spool:
        parser.error("--sign_reducer needs --sign_spool")
    if args.sign_spool and _engine_arg != "fast":
        parser.error("--sign_spool needs --engine fast")
    try:
        _sign_layers_arg = tuple(int(x) for x in args.sign_layers.split(",") if x.strip())
        json.loads(args.sign_reducer_kwargs)
    except ValueError as exc:
        parser.error(f"--sign_layers / --sign_reducer_kwargs: {exc}")
    if args.sign_reducer and not _sign_layers_arg:
        parser.error("--sign_reducer needs --sign_layers")
    if args.sign_inflight and not args.sign_spool and not args.sign_default:
        print("no sign spool and --no-sign_default: --sign_inflight is off (this server signs nothing)")
        args.sign_inflight = False
    if args.sign_inflight and not (args.sign_reducer and _xbatch_rows_arg and args.cascade_attn):
        parser.error("--sign_inflight needs --sign_reducer, --xbatch_rows and --cascade_attn")
    if args.sign_inflight and _autotune_min_rows_arg <= max(_xbatch_rows_arg) and _compile_mode_arg.startswith(
            "max-autotune"):
        # Kernel benchmarking chooses each kernel's config per graph by timing: the signing decoder's fused kernels
        # differ in source from the unsigned bucket's (they store the signed surfaces), so they would time and keep
        # their own configs (a reduction's block size sets its summation order), and a signed request would decode
        # other tokens. Without it every graph takes the configs its sizes give, the same in both.
        print(f"--sign_inflight: the row buckets compile without kernel benchmarking (--autotune_min_rows "
              f"{max(_xbatch_rows_arg) + 1}), so signed and unsigned requests decode the same tokens")
        _autotune_min_rows_arg = max(_xbatch_rows_arg) + 1
    if args.sign_reducer:
        # the reducer runs under deterministic algorithms, which need cuBLAS's fixed workspace before CUDA starts
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    _sign_spool_arg = args.sign_spool
    _sign_reducer_arg = args.sign_reducer
    _sign_reducer_kwargs_arg = args.sign_reducer_kwargs
    _sign_tail_arg = args.sign_tail
    _sign_width_arg = args.sign_width
    _sign_rows_arg = args.sign_rows
    _sign_groups_arg = args.sign_groups
    _sign_inflight_arg = args.sign_inflight
    _sign_default_arg = args.sign_default
    _profile_name = _profile_arg
    _request_log_arg = args.request_log
    if args.kv_sessions and not _xbatch_rows_arg:
        parser.error("--kv_sessions needs --xbatch_rows (a session's turn decodes on the cross-request batcher)")
    _kv_sessions_arg = args.kv_sessions
    _kv_session_idle_s_arg = args.kv_session_idle_s
    _sign_audit_every_arg = args.sign_audit_every
    _sign_disk_floor_gb_arg = args.sign_disk_floor_gb

    import uvicorn

    from kotodama.serve.logs import uvicorn_log_config
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", log_config=uvicorn_log_config())
