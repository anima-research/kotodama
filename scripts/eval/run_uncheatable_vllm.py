"""Uncheatable Eval BPB for a public Hugging Face model, scored through vLLM.

The same measurement as ``scripts/eval/run_uncheatable_hf.py``, with the
forward pass run by a vLLM engine instead of ``transformers``. vLLM ships fused
kernels for hybrid and state-space architectures (Falcon-H1, Bamba, Nemotron-H,
Falcon-Mamba) that native ``transformers`` runs through slow reference scans,
and it batches many chunks per call.

Shared with the HF runner and ``kotodama.eval.uncheatable``, by import:

- data: ``load_domains`` (same ``--data-dir`` default), whole-document
  trimming by ``select_docs``;
- tokenization: the model's own tokenizer (``load_tokenizer``), no special
  tokens;
- the document-start token (``document_start_token``: BOS, else EOS) and the
  chunk size (``chunk_size_for``: 4000, else ``context - 1``), with the context
  read from the model config as the HF runner reads it;
- chunking: every chunk is scored as ``[start] + chunk``, so each document
  token is predicted exactly once; the start position has no prediction and is
  never scored;
- the per-domain reduction ``bpb_from_sums`` and the result JSON (schema
  ``kotodama.uncheatable/v1``, ``DomainResult`` / ``Settings``).

What differs is the per-token loss. vLLM returns each prompt token's log-prob
as an fp32 log-softmax of the model-dtype logits (``SamplingParams(
prompt_logprobs=0)``); the per-chunk sums are taken in fp64. The HF runner's
default rounds per-token losses and their sum to bf16. ``settings.loss_precision``
records which (``fp32_logsoftmax_<dtype>_logits``), and ``settings.extra`` the
backend, vLLM version and tensor-parallel size.

Usage:
    KOTODAMA_VENV=<vllm venv> tools/run_py.sh -m scripts.eval.run_uncheatable_vllm \\
        --model tiiuae/Falcon-H1-0.5B-Base --out <json>
    KOTODAMA_VENV=<vllm venv> tools/run_py.sh -m scripts.eval.run_uncheatable_vllm \\
        --model <local model dir> --tp 2 --trust-remote-code --out <json>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from kotodama.eval.uncheatable import (
    CHUNK_SIZE,
    DATA_ENV,
    DEFAULT_DATASET,
    SCHEMA,
    DomainResult,
    Settings,
    bpb_from_sums,
    load_domains,
    select_docs,
)
from scripts.eval.run_lm_eval_hf import REFERENCE_MAX_LENGTH, load_tokenizer
from scripts.eval.run_uncheatable_hf import chunk_size_for, document_start_token

logger = logging.getLogger(__name__)

EXIT_LOAD_FAILED = 3     # vLLM could not build an engine for the model
EXIT_SCORING_FAILED = 4  # the engine ran but returned unusable log-probs

ChunkScorer = Callable[[Sequence[list[int]]], list[float]]  # chunks -> summed NLL (nats) each
EncodeFn = Callable[[str], list[int]]


class ModelLoadError(RuntimeError):
    """vLLM refused or failed to load the model."""


class ScoringError(RuntimeError):
    """The engine's output does not cover the prompt it was given."""


# ── Chunking and log-prob assembly (pure; no vLLM) ────────────────────────────

def doc_chunks(ids: Sequence[int], bos_id: int, chunk_size: int = CHUNK_SIZE) -> list[list[int]]:
    """``[bos] + ids[i:i + chunk_size]`` for every chunk, as ``doc_nll`` forms them."""
    if len(ids) <= 1:
        raise ValueError(f"document too short ({len(ids)} tokens)")
    if chunk_size <= 2:
        raise ValueError("chunk_size must be > 2")
    return [[bos_id, *ids[b:b + chunk_size]] for b in range(0, len(ids), chunk_size)]


def _position_logprob(entry: Mapping[int, Any] | None, token: int, pos: int) -> float:
    if not entry or token not in entry:
        raise ScoringError(f"prompt position {pos}: no log-prob for its token {token}")
    lp = entry[token]
    return float(getattr(lp, "logprob", lp))


def chunk_nll_from_prompt_logprobs(prompt_ids: Sequence[int], prompt_logprobs: Any) -> float:
    """Summed NLL (nats, fp64) of ``prompt_ids[1:]`` from vLLM's ``prompt_logprobs``.

    ``prompt_logprobs`` has one entry per prompt position: position 0 (the
    document-start token) has nothing to condition on, carries ``None`` (or an
    empty entry) and is skipped; position ``i`` maps token ids to log-probs and
    must hold ``prompt_ids[i]``. Both vLLM layouts are read: a list of
    ``{token_id: Logprob}`` dicts, and ``FlatLogprobs`` (parallel flat lists
    indexed by ``start_indices`` / ``end_indices``).
    """
    n = len(prompt_ids)
    if prompt_logprobs is None:
        raise ScoringError("the engine returned no prompt log-probs")
    if len(prompt_logprobs) != n:
        raise ScoringError(f"{len(prompt_logprobs)} prompt log-prob positions for a "
                           f"{n}-token prompt")
    if n < 2:
        raise ScoringError("a chunk needs the start token and at least one scored token")
    total = 0.0
    flat = all(hasattr(prompt_logprobs, a)
               for a in ("start_indices", "end_indices", "token_ids", "logprobs"))
    if flat:
        starts, ends = prompt_logprobs.start_indices, prompt_logprobs.end_indices
        tok_ids, lps = prompt_logprobs.token_ids, prompt_logprobs.logprobs
        for i in range(1, n):
            want = prompt_ids[i]
            for j in range(starts[i], ends[i]):
                if tok_ids[j] == want:
                    total -= float(lps[j])
                    break
            else:
                raise ScoringError(f"prompt position {i}: no log-prob for its token {want}")
    else:
        for i in range(1, n):
            total -= _position_logprob(prompt_logprobs[i], prompt_ids[i], i)
    if math.isnan(total):
        raise ScoringError("NaN in the prompt log-probs")
    return total


# ── Batched scoring with the module's reduction ───────────────────────────────

def eval_domain_batched(score_chunks: ChunkScorer, encode: EncodeFn, texts: Sequence[str],
                        bos_id: int, chunk_size: int = CHUNK_SIZE,
                        max_docs: int | None = None, max_tokens: int | None = None,
                        batch_chunks: int = 512, name: str = "") -> DomainResult:
    """``eval_domain``'s result, with the chunks scored ``batch_chunks`` at a time."""
    if batch_chunks < 1:
        raise ValueError("batch_chunks must be >= 1")
    t0 = time.time()
    token_ids = [encode(t) for t in texts]
    n = select_docs([len(x) for x in token_ids], max_docs, max_tokens)
    chunks: list[list[int]] = []
    owner: list[int] = []  # document index of each chunk
    for i in range(n):
        for c in doc_chunks(token_ids[i], bos_id, chunk_size):
            chunks.append(c)
            owner.append(i)
    nlls = [0.0] * n
    for b in range(0, len(chunks), batch_chunks):
        batch = chunks[b:b + batch_chunks]
        sums = score_chunks(batch)
        if len(sums) != len(batch):
            raise ScoringError(f"scorer returned {len(sums)} sums for {len(batch)} chunks")
        for k, s in enumerate(sums):
            nlls[owner[b + k]] += s
        logger.info("  %s: %d/%d chunks (%.1fs)", name, min(b + batch_chunks, len(chunks)),
                    len(chunks), time.time() - t0)
    kept = texts[:n]
    n_bytes = [len(t.encode("utf-8")) for t in kept]
    n_chars = [len(t) for t in kept]
    mean_nll = sum(nlls) / n
    return DomainResult(
        bpb=bpb_from_sums(nlls, n_bytes),
        bpc=(mean_nll / (sum(n_chars) / n)) * (1 / math.log(2)),
        nll_mean=mean_nll,
        n_docs=n, n_docs_total=len(texts),
        n_tokens=sum(len(token_ids[i]) for i in range(n)),
        n_bytes=sum(n_bytes), n_chars=sum(n_chars),
        trimmed=n < len(texts),
        seconds=round(time.time() - t0, 2),
    )


def evaluate_batched(score_chunks: ChunkScorer, encode: EncodeFn,
                     domains: dict[str, list[str]], settings: Settings,
                     batch_chunks: int = 512) -> dict[str, Any]:
    """``evaluate``'s result JSON (schema ``SCHEMA``), scored in batches."""
    if not domains:
        raise ValueError("no domains to evaluate")
    per: dict[str, dict[str, Any]] = {}
    for name, texts in domains.items():
        r = eval_domain_batched(score_chunks, encode, texts, settings.bos_id,
                                settings.chunk_size, settings.max_docs_per_domain,
                                settings.max_tokens_per_domain, batch_chunks, name)
        per[name] = asdict(r)
        logger.info("%-24s bpb %.6f  (%d/%d docs, %d tok, %.1fs)", name, r.bpb,
                    r.n_docs, r.n_docs_total, r.n_tokens, r.seconds)
    bpbs = [d["bpb"] for d in per.values()]
    return {
        "schema": SCHEMA,
        "checkpoint": settings.checkpoint,
        "step": None,
        "tokens": None,
        "mean_bpb": sum(bpbs) / len(bpbs),
        "n_domains": len(per),
        "trimmed": any(d["trimmed"] for d in per.values()),
        "domains": per,
        "settings": asdict(settings),
    }


# ── vLLM engine ───────────────────────────────────────────────────────────────

def model_context(model_id: str, revision: str, cache_dir: str | None,
                  trust_remote_code: bool) -> int:
    """Max positions, read as the HF runner reads them (``max_position_embeddings``,
    else ``REFERENCE_MAX_LENGTH``), so both runners pick the same chunk size."""
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(model_id, revision=revision, cache_dir=cache_dir,
                                         trust_remote_code=trust_remote_code)
    except Exception as exc:
        hint = "" if trust_remote_code else " (a custom architecture may need --trust-remote-code)"
        raise ModelLoadError(f"{model_id}: cannot read its config: {exc}{hint}") from exc
    return int(getattr(cfg, "max_position_embeddings", None) or REFERENCE_MAX_LENGTH)


def resolve_commit(model_id: str, revision: str) -> tuple[bool, str | None]:
    """(is a local directory, commit): a hub-cache snapshot dir is named by its commit."""
    if Path(model_id).is_dir():
        resolved = Path(model_id).resolve()
        return True, resolved.name if resolved.parent.name == "snapshots" else None
    from huggingface_hub import HfApi, snapshot_download

    try:
        return False, HfApi().model_info(model_id, revision=revision).sha
    except Exception as exc:  # metadata only; the engine load is authoritative
        try:  # offline: the cached snapshot the engine will load is named by its commit
            local_dir = snapshot_download(model_id, revision=revision, local_files_only=True)
            return False, Path(local_dir).name
        except Exception:
            logger.warning("could not resolve commit for %s@%s: %s", model_id, revision, exc)
            return False, None


def build_llm(args: argparse.Namespace, max_model_len: int) -> Any:
    """An offline ``vllm.LLM`` for scoring, or ``ModelLoadError`` saying why not."""
    try:
        import vllm
    except ImportError as exc:
        raise ModelLoadError(f"vllm is not importable in this environment: {exc}") from exc
    kwargs: dict[str, Any] = dict(
        model=args.model,
        revision=None if Path(args.model).is_dir() else args.revision,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tp,
        dtype=args.dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,  # every chunk is scored from scratch
        safetensors_load_strategy="prefetch",
        enforce_eager=args.enforce_eager,
        model_impl=args.model_impl,
        seed=0,
    )
    if args.cache_dir:
        kwargs["download_dir"] = args.cache_dir
    if args.max_num_batched_tokens:
        kwargs["max_num_batched_tokens"] = max(args.max_num_batched_tokens, max_model_len)
    if args.max_num_seqs:
        kwargs["max_num_seqs"] = args.max_num_seqs
    try:
        return vllm.LLM(**kwargs)
    except Exception as exc:
        raise ModelLoadError(f"vLLM {vllm.__version__} could not load {args.model}: "
                             f"{type(exc).__name__}: {exc}") from exc


def engine_architecture(llm: Any) -> str | None:
    """The model class the engine resolved (a native vLLM class, or a Transformers
    backend wrapper), for the record."""
    try:
        info = llm.llm_engine.model_config._model_info
        return str(info.architecture)
    except Exception:  # provenance only
        return None


def make_vllm_scorer(llm: Any) -> ChunkScorer:
    """Summed NLL per chunk from one ``llm.generate`` call over the batch."""
    from vllm import SamplingParams

    base = dict(max_tokens=1, temperature=0.0, prompt_logprobs=0, detokenize=False)
    try:
        params = SamplingParams(**base, flat_logprobs=True)
    except TypeError:  # versions without FlatLogprobs return the list layout
        params = SamplingParams(**base)

    def score(chunks: Sequence[list[int]]) -> list[float]:
        prompts = [{"prompt_token_ids": list(c)} for c in chunks]
        outs = llm.generate(prompts, params, use_tqdm=False)
        if len(outs) != len(chunks):
            raise ScoringError(f"{len(outs)} outputs for {len(chunks)} prompts")
        sums: list[float] = []
        for c, o in zip(chunks, outs):
            if list(o.prompt_token_ids) != list(c):
                raise ScoringError("an output's prompt does not match its request "
                                   "(outputs out of order)")
            sums.append(chunk_nll_from_prompt_logprobs(c, o.prompt_logprobs))
        return sums

    return score


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, help="HF repo id, or a local model directory")
    p.add_argument("--revision", default="main")
    p.add_argument("--out", required=True, help="result JSON path")
    p.add_argument("--data-dir", default=os.environ.get(DATA_ENV, DEFAULT_DATASET),
                   help=f"domain files / parquet / HF dataset id (default ${DATA_ENV}, "
                        f"else {DEFAULT_DATASET})")
    p.add_argument("--domains", default=None, help="comma-separated subset (default: all)")
    p.add_argument("--max-docs-per-domain", type=int, default=None)
    p.add_argument("--max-tokens-per-domain", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                   help="upper bound; capped at the model's context - 1")
    p.add_argument("--cache-dir", default=None, help="HF hub cache for the model")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="allow the model repo's own code (config and engine)")
    p.add_argument("--tp", type=int, default=1, help="tensor-parallel size (GPUs)")
    p.add_argument("--dtype", default="bfloat16", help="model dtype in the engine")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                   help="vLLM's memory fraction; prompt log-probs need fp32 logits "
                        "(tokens x vocab) beyond what vLLM reserves")
    p.add_argument("--max-num-batched-tokens", type=int, default=8192,
                   help="tokens per engine step (bounds the fp32 logits); raised to "
                        "chunk + 1 when smaller; 0 = vLLM's default")
    p.add_argument("--max-num-seqs", type=int, default=0, help="0 = vLLM's default")
    p.add_argument("--batch-chunks", type=int, default=512,
                   help="chunks submitted per generate call")
    p.add_argument("--enforce-eager", action="store_true", help="no CUDA graphs / compile")
    p.add_argument("--model-impl", choices=("vllm", "auto", "transformers"), default="vllm",
                   help="vllm = native vLLM classes only: an architecture vLLM does not "
                        "implement fails to load instead of falling back to the "
                        "Transformers modeling backend")
    args = p.parse_args(argv)
    if args.tp < 1:
        p.error("--tp must be >= 1")
    if args.batch_chunks < 1:
        p.error("--batch-chunks must be >= 1")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    domains = load_domains(args.data_dir)
    if args.domains:
        want = [d.strip() for d in args.domains.split(",") if d.strip()]
        missing = [d for d in want if d not in domains]
        if missing:
            raise SystemExit(f"unknown domains {missing}; have {sorted(domains)}")
        domains = {d: domains[d] for d in want}
    logger.info("data: %d domains, %d docs from %s", len(domains),
                sum(len(v) for v in domains.values()), args.data_dir)

    tok, tok_rev = load_tokenizer(args.model, args.revision, args.cache_dir)
    bos_mode, bos_id = document_start_token(tok)
    context = model_context(args.model, args.revision, args.cache_dir, args.trust_remote_code)
    chunk = chunk_size_for(context, args.chunk_size)
    local, commit = resolve_commit(args.model, args.revision)
    logger.info("document start %s = %d (%r); context %d; chunk %d", bos_mode, bos_id,
                tok.convert_ids_to_tokens(bos_id), context, chunk)

    def encode(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    t_load = time.time()
    llm = build_llm(args, max_model_len=chunk + 1)
    load_seconds = round(time.time() - t_load, 1)
    logger.info("engine up in %.1fs", load_seconds)
    import vllm

    settings = Settings(
        checkpoint=args.model, config_section="hf", attn_res=False,
        attn_res_boundaries=None, rope_fp32=False, tokenizer=args.model,
        bos_mode=bos_mode, bos_id=bos_id, chunk_size=chunk,
        loss_precision=f"fp32_logsoftmax_{args.dtype}_logits",
        max_docs_per_domain=args.max_docs_per_domain,
        max_tokens_per_domain=args.max_tokens_per_domain,
        data_source=str(args.data_dir), device=f"cuda (vllm tp={args.tp})",
        static_shapes=False,
        extra={
            "backend": "vllm",
            "vllm_version": vllm.__version__,
            "tensor_parallel_size": args.tp,
            "revision": None if local else args.revision,
            "bos_token": tok.convert_ids_to_tokens(bos_id),
            "dtype": args.dtype,
            "loss_sum": "fp64 over fp32 per-token log-probs",
            "local": local,
            "commit": commit,
            "tokenizer_revision": tok_rev,
            "context": context,
            "max_model_len": chunk + 1,
            "trust_remote_code": args.trust_remote_code,
            "model_impl": args.model_impl,
            "engine_architecture": engine_architecture(llm),
            "engine_load_seconds": load_seconds,
        },
    )
    t0 = time.time()
    result = evaluate_batched(make_vllm_scorer(llm), encode, domains, settings,
                              args.batch_chunks)
    result["settings"]["extra"]["score_seconds"] = round(time.time() - t0, 1)
    return result


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    try:
        result = run(args)
    except ModelLoadError as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        sys.exit(EXIT_LOAD_FAILED)
    except ScoringError as exc:
        print(f"error: scoring {args.model}: {exc}", file=sys.stderr, flush=True)
        sys.exit(EXIT_SCORING_FAILED)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    width = max(len(d) for d in result["domains"])
    for name, d in result["domains"].items():
        print(f"{name:<{width}}  {d['bpb']:.6f}  ({d['n_docs']}/{d['n_docs_total']} docs)")
    print(f"{'MEAN':<{width}}  {result['mean_bpb']:.6f}  over {result['n_domains']} domains"
          f"{' (trimmed)' if result['trimmed'] else ''} -> {out}", flush=True)


if __name__ == "__main__":
    main()
