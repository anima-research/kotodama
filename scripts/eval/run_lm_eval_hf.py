"""Run the lm-eval benchmarks on a public Hugging Face model, scored exactly as
``run_lm_eval.py`` scores KotodamaModel checkpoints.

The HF model is driven through ``KotodamaEvalLM``'s own scoring code: the same
lm-eval request path (``TemplateLM``), the same right-padded length-sorted
batching (no attention mask; exact under causal attention), the same rolling
windows for perplexity tasks, the same default task list, n-shot and batch
settings. Only the network and its tokenizer differ. The model runs in bf16, as
``model_loader`` runs ours.

The rolling window is ``min(4096, model context)``: a model with a shorter
context (Pythia: 2048) scores wikitext with shorter windows than ours.

Writes ``<output-dir>/results.json`` (same shape as ``run_lm_eval.py``'s) and
``<output-dir>/meta.json`` (repo, revision, resolved commit, lm-eval version,
window length, weight fingerprint). The fingerprint is comparable across
revisions of one repo: two revisions with equal fingerprints hold the same
weights.

Usage:
    tools/run_py.sh -m scripts.eval.run_lm_eval_hf --model EleutherAI/pythia-1.4b \\
        --revision step14000 --output-dir out/pythia-1.4b-step14000
    tools/run_py.sh -m scripts.eval.run_lm_eval_hf --model /path/to/local/model \\
        --output-dir out/local-model
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import torch

from scripts.eval.run_lm_eval import DEFAULT_TASKS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# KotodamaEvalLM's window length; HF models are capped at their own context.
REFERENCE_MAX_LENGTH = 4096


def make_hf_eval_lm(
    model: torch.nn.Module,
    tokenizer: Any,
    device: str,
    batch_size: int,
    max_length: int,
    max_batch_tokens: int,
    name: str,
):
    """A ``KotodamaEvalLM`` whose network is an HF causal LM.

    HF causal-LM outputs index as ``out["logits"]``, which is the one contract
    the inherited scoring methods rely on.
    """
    from lm_eval.api.model import TemplateLM

    from kotodama.eval.lm_eval_adapter import KotodamaEvalLM

    class HFEvalLM(KotodamaEvalLM):
        def __init__(self) -> None:
            TemplateLM.__init__(self)
            self.model = model
            self.tokenizer = tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self._device = torch.device(device)
            self._batch_size = batch_size
            self._max_length = max_length
            self._max_batch_tokens = max_batch_tokens
            self._checkpoint_name = name

        def generate_until(self, requests, disable_tqdm: bool = False):
            raise NotImplementedError(
                "generative tasks are not scored by this runner; the default "
                "task list is loglikelihood-only"
            )

    return HFEvalLM()


def load_tokenizer(repo: str, revision: str, cache_dir: str | None, trust_remote_code: bool = False):
    """The revision's tokenizer, or main's when the revision ships none.

    Some intermediate revisions carry weights and config but no tokenizer
    files, and transformers then builds an empty tokenizer (encode -> []).
    A training run's tokenizer is fixed, so main's is the right one.
    ``trust_remote_code`` reaches both loads: a repository whose tokenizer is
    custom code (Kimi-K2's tiktoken tokenizer) refuses to load without it.
    """
    from transformers import AutoTokenizer

    probe = "The quick brown fox"
    try:
        tok = AutoTokenizer.from_pretrained(repo, revision=revision, cache_dir=cache_dir,
                                            trust_remote_code=trust_remote_code)
        if tok.encode(probe, add_special_tokens=False):
            return tok, revision
        logger.warning("%s@%s: tokenizer encodes to [] — using main's", repo, revision)
    except (OSError, ValueError) as exc:
        logger.warning("%s@%s: no usable tokenizer (%s) — using main's", repo, revision, exc)
    tok = AutoTokenizer.from_pretrained(repo, revision="main", cache_dir=cache_dir,
                                        trust_remote_code=trust_remote_code)
    if not tok.encode(probe, add_special_tokens=False):
        raise RuntimeError(f"{repo}: main's tokenizer also encodes to []")
    return tok, "main"


def weight_fingerprint(model: torch.nn.Module) -> dict[str, Any]:
    """sha256 over every parameter's bytes (name-sorted) plus summary stats."""
    h = hashlib.sha256()
    total = 0.0
    total_abs = 0.0
    with torch.no_grad():
        for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
            flat = p.detach().reshape(-1)
            h.update(name.encode())
            h.update(flat.contiguous().view(torch.uint8).cpu().numpy().tobytes())
            total += flat.sum(dtype=torch.float64).item()
            total_abs += flat.abs().sum(dtype=torch.float64).item()
    return {"sha256": h.hexdigest(), "sum": total, "abs_sum": total_abs}


def load_hf_model(
    model_id: str, revision: str, cache_dir: str | None, weights: str, device: str
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """An HF causal LM in bf16 on ``device``, its tokenizer, and provenance.

    ``model_id`` is a hub repo id or a local directory. The provenance dict
    carries ``local``, ``commit`` (None for a plain local copy), the tokenizer's
    revision, ``context`` (max positions), ``n_params`` and the weight fingerprint.
    """
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM

    local = Path(model_id).is_dir()
    if local:
        # A hub-cache snapshot dir is named by its commit; a plain local copy has none.
        resolved = Path(model_id).resolve()
        commit = resolved.name if resolved.parent.name == "snapshots" else None
    else:
        try:
            commit = HfApi().model_info(model_id, revision=revision).sha
        except Exception as exc:  # metadata only; the load below is authoritative
            logger.warning("could not resolve commit for %s@%s: %s", model_id, revision, exc)
            commit = None

    use_st = {"auto": None, "safetensors": True, "bin": False}[weights]
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
        use_safetensors=use_st,
    )
    model.eval().to(device)
    model.config.use_cache = False  # scoring only; no KV cache per forward
    tokenizer, tok_rev = load_tokenizer(model_id, revision, cache_dir)

    ctx = getattr(model.config, "max_position_embeddings", None) or REFERENCE_MAX_LENGTH
    fingerprint = weight_fingerprint(model)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "%s@%s (commit %s): %.1fM params, context %d, fingerprint %s",
        model_id, revision, commit, n_params / 1e6, int(ctx), fingerprint["sha256"][:16],
    )
    return model, tokenizer, {
        "local": local,
        "commit": commit,
        "tokenizer_revision": tok_rev,
        "context": int(ctx),
        "n_params": n_params,
        "weight_fingerprint": fingerprint,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, help="HF repo id, or a local model directory")
    p.add_argument("--revision", default="main")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tasks", default=DEFAULT_TASKS)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-batch-tokens", type=int, default=32768)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-fewshot", type=int, default=None)
    p.add_argument(
        "--cache-dir", default=None,
        help="HF hub cache for the model and tokenizer (default: HF_HOME's); "
             "datasets keep the default cache",
    )
    p.add_argument(
        "--weights", choices=("auto", "safetensors", "bin"), default="auto",
        help="Which weight files to load when a revision ships both",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import lm_eval

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, info = load_hf_model(
        args.model, args.revision, args.cache_dir, args.weights, args.device
    )
    local, commit, tok_rev = info["local"], info["commit"], info["tokenizer_revision"]
    fingerprint, n_params = info["weight_fingerprint"], info["n_params"]
    max_length = min(REFERENCE_MAX_LENGTH, info["context"])

    lm = make_hf_eval_lm(
        model, tokenizer, args.device, args.batch_size, max_length,
        args.max_batch_tokens, f"{args.model}@{args.revision}",
    )
    eval_kwargs: dict[str, Any] = {
        "model": lm,
        "tasks": [t.strip() for t in args.tasks.split(",")],
        "log_samples": False,
    }
    if args.limit is not None:
        eval_kwargs["limit"] = args.limit
    if args.num_fewshot is not None:
        eval_kwargs["num_fewshot"] = args.num_fewshot
    results = lm_eval.simple_evaluate(**eval_kwargs)

    serializable = results.get("results", {})
    with open(out_dir / "results.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    meta = {
        "model": args.model,
        "revision": None if local else args.revision,
        "local": local,
        "commit": commit,
        "tokenizer_revision": tok_rev,
        "weights": args.weights,
        "dtype": "bfloat16",
        "n_params": n_params,
        "max_length": max_length,
        "batch_size": args.batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "limit": args.limit,
        "num_fewshot": args.num_fewshot,
        "lm_eval_version": getattr(lm_eval, "__version__", None),
        "weight_fingerprint": fingerprint,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Results saved to %s", out_dir / "results.json")

    for task_name, task_results in serializable.items():
        metrics = [
            f"{k.replace(',none', '')}={v:.4f}"
            for k, v in task_results.items()
            if k.endswith(",none") and isinstance(v, (int, float))
        ]
        if metrics:
            print(f"  {task_name}: {', '.join(metrics)}")

    del lm, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
