"""Chat SFT on a kotodama checkpoint, including the looped 1b-tsumugi shape.

    python -m kotodama.posttrain.sft --config configs/posttrain/tsumugi-sft.yaml \\
        --checkpoint <base.pt> --train_jsonl <chat.jsonl> --out_dir <dir>

Data, ``--format transcript`` (default): JSONL rows ``{"turns": [{"speaker", "text"[, "train": false]}],
"header"?}`` rendered as plain-text ``Name: text`` documents with no special tokens (kotodama.posttrain.transcript).
``--train_speakers all`` trains every turn (unmasked); a comma list trains only those speakers' turns.
``--format chatml``: rows ``{"messages": [{"role", "content"[, "train": false]}]}``; assistant content + its
``<|im_end|>`` train (kotodama.posttrain.chat_data). Either way, examples are packed into rows with per-conversation
attention segments.

``--lora_rank N`` trains a LoRA adapter (kotodama.posttrain.lora) on the frozen model instead of the full weights;
``--lora_layers`` picks unique layers by group (prelude / core / coda) or index, and checkpoints are adapter
directories that ``kotodama.serve.server --adapter`` merges at load. The hybrid optimizer sends the adapter matrices to
Muon, as it does every hidden matrix; ``--optimizer adamw`` trains them with AdamW.

Recipe: Muon + AdamW split, WSD schedule from ``--warmup_ratio`` / ``--decay_start_pct``, loss =
mean over the micro-batch's trainable tokens (z-loss included), divided by grad-accum (the "intended" scaling, so
gradient clipping is meaningful), eval loss + <|im_end|> top-1 accuracy. Not ported: the geometric probe battery
(use the riders / stage gauges on the saved checkpoints), async pinned checkpointing, resume (runs are short; a
SIGTERM saves weights and exits).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Optional

import torch

from kotodama.posttrain import chat_data as cd
from kotodama.posttrain import common
from kotodama.posttrain import lora
from kotodama.presets import TOKENIZER_NAME
from kotodama.serve.chatml import EOS_TOKEN_ID, IM_END_TOKEN_ID

logger = logging.getLogger("kotodama.posttrain.sft")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="kotodama chat SFT")
    common.add_model_args(p)
    common.add_optim_args(p, muon_lr=1e-3, adamw_lr=1e-4, warmup_ratio=0.05, decay_start_pct=1.0,
                          decay_type="sqrt")
    lora.add_args(p)
    g = p.add_argument_group("data / loop")
    g.add_argument("--format", default="transcript", choices=["transcript", "chatml"])
    g.add_argument("--train_speakers", default="all",
                   help="transcript format: 'all' = loss on every turn; or a comma list of speakers whose turns train")
    g.add_argument("--train_jsonl", default=None)
    g.add_argument("--eval_jsonl", default=None)
    g.add_argument("--tokenizer", default=TOKENIZER_NAME)
    g.add_argument("--max_seq_len", type=int, default=4096)
    g.add_argument("--micro_batch_size", type=int, default=4, help="packed rows per micro-batch")
    g.add_argument("--grad_accum", type=int, default=4)
    g.add_argument("--epochs", type=float, default=1.0, help="> 1 warns: more than one effective epoch overfits a model's style")
    g.add_argument("--max_steps", type=int, default=0, help="cap optimizer steps (0 = epochs decide)")
    g.add_argument("--no_chunk", action="store_true", help="truncate long conversations instead of chunking")
    g.add_argument("--drop_overlong", action="store_true",
                   help="transcript format: drop transcripts longer than --max_seq_len instead of truncating")
    g.add_argument("--token_cache", action=argparse.BooleanOptionalAction, default=True,
                   help="transcript format: cache the tokenized data beside the JSONL (<file>.tok-<key>.npz)")
    g.add_argument("--out_dir", default=None)
    g.add_argument("--save_every", type=int, default=0, help="weights every N steps (0 = final only)")
    g.add_argument("--eval_every", type=int, default=0)
    a = common.parse_with_config(p, argv)
    if a.compile is None:
        a.compile = True
    for k in ("checkpoint", "train_jsonl", "out_dir"):
        if not getattr(a, k):
            p.error(f"--{k} is required (flag or config)")
    return a


def _examples(a: argparse.Namespace, path: str, tok: Any) -> list[cd.Example]:
    if a.format == "chatml":
        return cd.encode_conversations(cd.load_chat_jsonl(path), tok, a.max_seq_len, chunk=not a.no_chunk)
    from kotodama.posttrain import transcript as tx
    from kotodama.serve.chatml import EOS_TOKEN_ID

    speakers = None if a.train_speakers.strip() == "all" else {s.strip() for s in a.train_speakers.split(",")}
    return tx.encode_file(path, tok, a.tokenizer, a.max_seq_len, EOS_TOKEN_ID, speakers, drop_overlong=a.drop_overlong,
                          cache=a.token_cache)


def _batches(rows: list[cd.PackedRow], mb: int, seed: int) -> list[list[cd.PackedRow]]:
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return [[rows[i] for i in order[j:j + mb]] for j in range(0, len(order), mb)]


def _to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def varlen_inputs(b: dict[str, Any], max_seq_len: int, compiled: bool) -> dict[str, Any]:
    """Forward kwargs for a packed batch. Compiled: the segment count is dynamic from the first trace and max_seqlen is
    the row length (an exact upper bound for the varlen attention kernels) — a per-batch int would recompile on every
    new value and, past dynamo's cache limit, drop the rest of the run to eager."""
    max_seqlen = b["max_seqlen"]
    if compiled:
        torch._dynamo.maybe_mark_dynamic(b["cu_seqlens"], 0)
        max_seqlen = max_seq_len
    return {"labels": b["labels"], "position_ids": b["position_ids"], "cu_seqlens": b["cu_seqlens"],
            "max_seqlen": max_seqlen}


def evaluate(model: torch.nn.Module, rows: list[cd.PackedRow], mb: int, device: str) -> dict[str, float]:
    """Token-weighted eval loss, and top-1 accuracy on the tokens that close a turn (<|im_end|>) or a document
    (end-of-text) — does the model stop where the data stops."""
    was = model.training
    model.eval()
    tot_loss = tot_tok = 0.0
    closers = {"im_end": IM_END_TOKEN_ID, "eos": EOS_TOKEN_ID}
    hit = dict.fromkeys(closers, 0)
    cnt = dict.fromkeys(closers, 0)
    try:
        for j in range(0, len(rows), mb):
            b = _to_device(cd.collate(rows[j:j + mb]), device)
            with common.autocast(device):
                out = model(b["input_ids"], labels=b["labels"], position_ids=b["position_ids"],
                            cu_seqlens=b["cu_seqlens"], max_seqlen=b["max_seqlen"])
            n = int((b["labels"][:, 1:] != cd.IGNORE_INDEX).sum())
            tot_loss += float(out["loss"]) * n
            tot_tok += n
            tgt = b["labels"][:, 1:]
            pred = None
            for k, tid in closers.items():
                m = tgt == tid
                if m.any():
                    pred = out["logits"][:, :-1].argmax(-1) if pred is None else pred
                    hit[k] += int((pred[m] == tid).sum())
                    cnt[k] += int(m.sum())
    finally:
        model.train(was)
    res = {"eval/loss": tot_loss / max(tot_tok, 1), "eval/tokens": tot_tok}
    for k in closers:
        if cnt[k]:
            res[f"eval/{k}_top1"] = hit[k] / cnt[k]
    return res


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [sft] %(message)s", datefmt="%H:%M:%S")
    a = parse_args(argv)
    torch.manual_seed(a.seed)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(a), indent=1, default=str))
    if a.epochs > 1.0:
        logger.warning("epochs=%.2f > 1: more than one effective epoch overfits a model's style", a.epochs)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    train_rows = cd.build_rows(_examples(a, a.train_jsonl, tok), a.max_seq_len, pad_id)
    if not train_rows:
        raise SystemExit(f"no trainable examples in {a.train_jsonl}")
    eval_rows: list[cd.PackedRow] = []
    if a.eval_jsonl:
        eval_rows = cd.build_rows(_examples(a, a.eval_jsonl, tok), a.max_seq_len, pad_id)

    micro_per_epoch = math.ceil(len(train_rows) / a.micro_batch_size)
    total_steps = a.max_steps or max(1, math.ceil(micro_per_epoch * a.epochs / a.grad_accum))
    logger.info("rows %d (x%d tok) | micro %d x accum %d | %d steps", len(train_rows), a.max_seq_len,
                a.micro_batch_size, a.grad_accum, total_steps)

    model = common.load_policy(a)
    if a.lora_rank:
        alpha = a.lora_alpha if a.lora_alpha is not None else 2.0 * a.lora_rank
        layers = lora.resolve_layers(a.lora_layers, model.config)
        targets = [t.strip() for t in a.lora_targets.split(",") if t.strip()]
        trainable = lora.attach(model, a.lora_rank, alpha, targets, layers, a.lora_dropout)
        logger.info("LoRA rank %d alpha %g on %d layers x %s: %.2fM trainable params", a.lora_rank, alpha, len(layers),
                    ",".join(targets), sum(p.numel() for p in trainable.values()) / 1e6)
    fwd = common.compiled(model, a)
    muon, adamw, sched = common.build_optim(model, a, total_steps)
    tel = common.Telemetry(out_dir)
    sig = common.SigtermFlag()
    meta = {"engine": "kotodama.posttrain.sft", "base": str(a.checkpoint), "train_jsonl": str(a.train_jsonl)}
    if a.lora_rank:
        meta.update(rank=a.lora_rank, alpha=alpha, targets=targets, layers=layers, model_size=a.model_size)
    ckpt_dir = out_dir / "checkpoints"

    def save(name: str) -> Path:
        """Weights at ``checkpoints/<name>``: the adapter directory under LoRA, else a full weights checkpoint."""
        if a.lora_rank:
            return lora.save(ckpt_dir / f"{name}.adapter", model, {**meta, "step": step})
        return common.save_weights(ckpt_dir / f"{name}.pt", model, step, meta)

    if eval_rows:
        ev = evaluate(model, eval_rows, a.micro_batch_size, a.device)
        logger.info("eval @0: %s", ev)
        tel.log(0, **ev)

    step = micro = 0
    loss_acc = torch.zeros((), device=a.device)
    tok_acc = torch.zeros((), dtype=torch.long, device=a.device)
    is_compiled = fwd is not model
    t0 = time.perf_counter()
    epoch = 0
    done = False
    while not done:
        for rows in _batches(train_rows, a.micro_batch_size, a.seed + epoch):
            b = _to_device(cd.collate(rows), a.device)
            with common.autocast(a.device):
                out = fwd(b["input_ids"], **varlen_inputs(b, a.max_seq_len, is_compiled))
            (out["loss"] / a.grad_accum).backward()
            loss_acc += out["loss"].detach()  # accumulated on device: one host sync per step, not per micro-batch
            tok_acc += (b["labels"] != cd.IGNORE_INDEX).sum()
            micro += 1
            if micro % a.grad_accum:
                continue
            gn = common.optimizer_step(model, muon, adamw, sched, step, a.gradient_clip)
            step += 1
            dt = time.perf_counter() - t0
            t0 = time.perf_counter()
            loss = float(loss_acc) / a.grad_accum
            lrs = sched.get_last_lr()
            rec: dict[str, Any] = {"loss": loss, "grad_norm": gn, "muon_lr": lrs["muon_lr"],
                                   "adamw_lr": lrs["adamw_lr"], "trainable_tokens": int(tok_acc),
                                   "tok_s": a.grad_accum * a.micro_batch_size * a.max_seq_len / max(dt, 1e-9)}
            logger.info("step %d/%d loss %.4f gn %.2f lr %.2e", step, total_steps, loss, gn, lrs["muon_lr"])
            if not math.isfinite(loss):
                save(f"step_{step:08d}.nonfinite")
                raise SystemExit(f"non-finite loss at step {step}")
            if a.eval_every and eval_rows and step % a.eval_every == 0:
                rec.update(evaluate(model, eval_rows, a.micro_batch_size, a.device))
                logger.info("eval @%d: loss %.4f", step, rec["eval/loss"])
            tel.log(step, **rec)
            loss_acc.zero_()
            tok_acc.zero_()
            if a.save_every and step % a.save_every == 0 and step < total_steps:
                save(f"step_{step:08d}")
            if sig.received or step >= total_steps:
                done = True
                break
        epoch += 1
        if epoch >= math.ceil(a.epochs) and not a.max_steps:
            done = True

    final = save(f"step_{step:08d}")
    if eval_rows:
        ev = evaluate(model, eval_rows, a.micro_batch_size, a.device)
        logger.info("final eval: %s", ev)
        tel.log(step, **ev)
    tel.close()
    sig.restore()
    summary: dict[str, Any] = {"checkpoint": str(final), "steps": step}
    if a.lora_rank:
        summary["adapter"] = True
    (out_dir / "final.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
