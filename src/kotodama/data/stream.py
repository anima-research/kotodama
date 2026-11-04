"""Read and reshape an existing uint16 token stream.

    python -m kotodama.data.stream inspect <train.bin> [--eos_id N] [--json out.json]
    python -m kotodama.data.stream sample <train.bin> [--n 4] [--source NAME] [--seed 0] [--max_tokens 512]
    python -m kotodama.data.stream remix <train.bin> --out <new.bin> [--drop NAME ...] \\
        [--weight NAME=W ...] [--cap NAME=TOKENS ...] [--seed 43] [--dry_run]

``inspect`` reports the stream's size, its sequences at the provenance's
``seq_len``, its token-id range, document counts and lengths (from EOS
positions), and each source's chunks and share when provenance sits beside it
(``kotodama.data.provenance``). ``sample`` decodes random chunks, optionally
from one source, splitting them at EOS into documents. ``remix`` writes a new
stream from a labelled one: each source's chunks are reselected with the same
``--drop`` / ``--weight`` / ``--cap`` rules as ``kotodama.data.curation.assemble``,
reshuffled under ``--seed``, and given a provenance of their own whose
``parent`` records the input. A remix never edits its input.

Streams read through ``numpy.memmap`` in blocks, so a stream of hundreds of
gigabytes is inspected in one pass without loading it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from kotodama.data.curation.assemble import (
    Part, build, parse_assignments, print_plan, select_chunks, selections,
)
from kotodama.data.provenance import BYTES_PER_TOKEN, ProvenanceError, load_labels, provenance_path
from kotodama.presets import TOKENIZER_NAME

logger = logging.getLogger(__name__)

SCAN_BLOCK_TOKENS = 1 << 26


def open_stream(path: Path) -> np.memmap:
    if not path.exists():
        raise FileNotFoundError(f"stream not found: {path}")
    if path.stat().st_size % BYTES_PER_TOKEN:
        raise ValueError(f"{path} has an odd byte count; it is not a uint16 token stream")
    return np.memmap(path, dtype=np.uint16, mode="r")


def inspect_stream(path: Path, eos_id: int | None = None) -> dict[str, Any]:
    """The stream's statistics as a dict (see the module docstring); one pass over the file."""
    data = open_stream(path)
    labels = load_labels(path)
    if eos_id is None and labels is not None and labels.provenance is not None:
        eos_id = labels.provenance.eos_id
    n = len(data)
    lo, hi, n_eos, last_eos, longest = 65535, 0, 0, -1, 0
    for start in range(0, n, SCAN_BLOCK_TOKENS):
        block = np.asarray(data[start:start + SCAN_BLOCK_TOKENS])
        lo, hi = min(lo, int(block.min())), max(hi, int(block.max()))
        if eos_id is not None:
            pos = np.flatnonzero(block == eos_id) + start
            if len(pos):
                gaps = np.diff(np.concatenate([[last_eos], pos]))
                longest = max(longest, int(gaps.max()) - 1)
                n_eos += len(pos)
                last_eos = int(pos[-1])
    report: dict[str, Any] = {
        "path": str(path), "tokens": n, "bytes": n * BYTES_PER_TOKEN,
        "token_id_range": [lo, hi] if n else None,
    }
    if eos_id is not None:
        report["eos_id"] = eos_id
        report["documents"] = n_eos
        report["mean_doc_tokens"] = round((n - n_eos) / n_eos, 1) if n_eos else None
        report["longest_doc_tokens"] = max(longest, n - 1 - last_eos) if n else 0
    else:
        report["documents"] = "unknown: pass --eos_id (no provenance records it)"
    if labels is None:
        report["provenance"] = None
    else:
        counts = labels.counts()
        total = sum(counts.values()) or 1
        report["seq_len"] = labels.seq_len
        report["sequences"] = len(labels.labels)
        report["provenance"] = str(provenance_path(path)) if labels.provenance else "block spans"
        report["sources"] = {s: {"chunks": c, "tokens": c * labels.seq_len, "share": round(c / total, 6)}
                             for s, c in counts.items()}
        if labels.provenance is not None:
            report["tokenizer"] = labels.provenance.tokenizer
            report["parent"] = labels.provenance.parent
    return report


def sample_chunks(path: Path, n: int, source: str | None = None, seed: int = 0,
                  seq_len: int | None = None) -> list[tuple[int, str | None, np.ndarray]]:
    """``n`` random chunks as (chunk index, source or None, tokens).

    Without provenance, ``seq_len`` sets the chunk size and sources are unknown;
    asking for a source then raises ProvenanceError.
    """
    data = open_stream(path)
    labels = load_labels(path)
    rng = np.random.default_rng(seed)
    if labels is None:
        if source is not None:
            raise ProvenanceError(f"{path} has no provenance, so it has no sources to sample from")
        if seq_len is None:
            raise ValueError("no provenance beside the stream: pass --seq_len")
        pool = np.arange(len(data) // seq_len)
        names: list[str] | None = None
    else:
        seq_len = labels.seq_len
        names = labels.sources
        if source is None:
            pool = np.arange(len(labels.labels))
        elif source in names:
            pool = np.flatnonzero(labels.labels == names.index(source))
        else:
            raise ProvenanceError(f"source {source!r} not in {names}")
    picks = np.sort(rng.choice(pool, size=min(n, len(pool)), replace=False)) if len(pool) else pool
    out = []
    for c in picks:
        src = names[labels.labels[c]] if labels is not None and names is not None else None
        out.append((int(c), src, np.asarray(data[c * seq_len:(c + 1) * seq_len])))
    return out


def split_docs(tokens: np.ndarray, eos_id: int) -> list[np.ndarray]:
    """Split a token run at EOS (the EOS itself dropped); an unterminated tail is the last piece."""
    cuts = np.flatnonzero(tokens == eos_id)
    pieces = np.split(tokens, cuts)
    return [p[1:] if i else p for i, p in enumerate(pieces) if len(p[1:] if i else p)]


def remix(path: Path, out: Path, drop: list[str], weights: dict[str, float], caps: dict[str, int],
          seed: int, dry_run: bool = False) -> list[Part]:
    """Reselect a labelled stream's chunks per source and write them as a new stream (see module docstring)."""
    if out.resolve() == path.resolve():
        raise ValueError("--out must differ from the input stream")
    labels = load_labels(path)
    if labels is None:
        raise ProvenanceError(f"{path} has no provenance; a remix needs every chunk's source")
    sel = selections(labels.sources, drop, weights, caps)
    parts = []
    for i, name in enumerate(labels.sources):
        available = np.flatnonzero(labels.labels == i).astype(np.int64)
        parts.append(Part(name, path, select_chunks(available, sel[name], labels.seq_len, seed, name),
                          len(available), sel[name]))
    print_plan(parts, labels.seq_len)
    if dry_run:
        return parts
    prov = labels.provenance
    parent = {"stream": path.name, "seed": prov.seed if prov else None,
              "provenance_sha256": _sha256(provenance_path(path)) if prov else None}
    build([p for p in parts if len(p.chunks)], out, labels.seq_len, seed,
          prov.tokenizer if prov else None, prov.eos_id if prov else None, parent=parent)
    return parts


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [stream] %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="inspect, sample and remix uint16 token streams")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("inspect", help="size, token range, documents, per-source shares")
    pi.add_argument("stream", type=Path)
    pi.add_argument("--eos_id", type=int, help="default: the provenance's EOS id")
    pi.add_argument("--json", type=Path, help="also write the report here")
    ps = sub.add_parser("sample", help="decode random chunks")
    ps.add_argument("stream", type=Path)
    ps.add_argument("--n", type=int, default=4)
    ps.add_argument("--source")
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--seq_len", type=int, help="chunk size when no provenance sits beside the stream")
    ps.add_argument("--max_tokens", type=int, default=512, help="print at most this many tokens per chunk")
    ps.add_argument("--tokenizer", help="default: the provenance's tokenizer, else the preset one")
    ps.add_argument("--eos_id", type=int, help="default: the provenance's or tokenizer's EOS id")
    pr = sub.add_parser("remix", help="drop / reweight sources into a new stream")
    pr.add_argument("stream", type=Path)
    pr.add_argument("--out", type=Path, required=True)
    pr.add_argument("--drop", nargs="*", default=[])
    pr.add_argument("--weight", action="append", default=[], help="NAME=W")
    pr.add_argument("--cap", action="append", default=[], help="NAME=TOKENS")
    pr.add_argument("--seed", type=int, default=43, help="shuffle seed of the new stream")
    pr.add_argument("--dry_run", action="store_true")
    args = ap.parse_args(argv)

    try:
        if args.cmd == "inspect":
            report = inspect_stream(args.stream, args.eos_id)
            text = json.dumps(report, indent=1)
            print(text)
            if args.json:
                args.json.write_text(text)
        elif args.cmd == "sample":
            labels = load_labels(args.stream)
            prov = labels.provenance if labels is not None else None
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.tokenizer or (prov.tokenizer if prov and prov.tokenizer
                                                                    else TOKENIZER_NAME))
            eos = args.eos_id if args.eos_id is not None else (
                prov.eos_id if prov and prov.eos_id is not None else tok.eos_token_id)
            for c, src, toks in sample_chunks(args.stream, args.n, args.source, args.seed, args.seq_len):
                print(f"===== chunk {c}" + (f" ({src})" if src else "") + " =====")
                for d, doc in enumerate(split_docs(toks[:args.max_tokens], eos)):
                    print(f"--- doc {d} ({len(doc)} tokens shown)")
                    print(tok.decode(doc.tolist()))
        else:
            remix(args.stream, args.out, args.drop, parse_assignments(args.weight, "--weight", float),
                  parse_assignments(args.cap, "--cap", int), args.seed, args.dry_run)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
