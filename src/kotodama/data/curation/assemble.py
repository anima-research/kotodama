"""Assemble per-source token streams into one shuffled training stream, with its provenance.

    python -m kotodama.data.curation.assemble --in_dir <tokenized> --out <train.bin> --seq_len 4096 \\
        [--seed 42] [--drop NAME ...] [--weight NAME=W ...] [--cap NAME=TOKENS ...] [--dry_run]

Each ``<source>.bin`` (from ``kotodama.data.curation.tokenize``) is cut into
``seq_len``-token chunks; a trailing partial chunk is left out. ``--weight``
resamples a source's chunks: 0.5 keeps a seeded random half, 2 takes every chunk
twice, 1.5 every chunk once plus a random half again. ``--cap`` bounds a
source's tokens after weighting; ``--drop`` is weight 0. The selected chunks of
all sources (sources in name order) are concatenated, and chunk ``i`` lands at
output chunk ``perm[i]`` of ``perm = RandomState(seed).permutation(total)``.
Sources are read front to back and written with ``pwrite``, which keeps reads
sequential on large inputs.

``<out stem>.provenance.json`` records the source order, each source's chunk
count, the seed and every source's selection, which is enough to recover the
source of every output chunk (``kotodama.data.provenance``). The stream is
written under a temporary name and renamed when complete.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kotodama.data.provenance import BYTES_PER_TOKEN, Provenance, provenance_path

logger = logging.getLogger(__name__)

READ_BLOCK_CHUNKS = 1024   # chunks per read from a source


@dataclass(frozen=True)
class Selection:
    """How much of a source to take: a resampling weight, then an optional token cap."""

    weight: float = 1.0
    cap_tokens: int | None = None

    def __post_init__(self) -> None:
        if not (self.weight >= 0 and np.isfinite(self.weight)):
            raise ValueError(f"weight must be a finite number >= 0, got {self.weight}")
        if self.cap_tokens is not None and self.cap_tokens < 0:
            raise ValueError(f"cap must be >= 0, got {self.cap_tokens}")


def select_chunks(available: np.ndarray, sel: Selection, seq_len: int, seed: int, name: str) -> np.ndarray:
    """The chunk indices a selection takes from ``available`` (ascending within each repeat).

    Weight 1 without a cap returns ``available`` unchanged. The random parts are
    seeded by (seed, source name), so a source's selection does not depend on
    which other sources are present.
    """
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    whole = int(sel.weight)
    k = int(np.floor(len(available) * (sel.weight - whole)))
    parts = [available] * whole
    if k:
        parts.append(np.sort(rng.choice(available, size=k, replace=False)))
    out = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    if sel.cap_tokens is not None and len(out) * seq_len > sel.cap_tokens:
        keep = np.sort(rng.choice(len(out), size=sel.cap_tokens // seq_len, replace=False))
        out = out[keep]
    return out.astype(np.int64)


@dataclass
class Part:
    """The chunks one source contributes: indices into ``path`` at ``seq_len`` granularity."""

    name: str
    path: Path
    chunks: np.ndarray
    available: int
    selection: Selection = field(default_factory=Selection)

    def describe(self, seq_len: int) -> dict[str, Any]:
        return {"path": str(self.path), "available_chunks": self.available, "weight": self.selection.weight,
                "cap_tokens": self.selection.cap_tokens, "chunks": int(len(self.chunks)),
                "tokens": int(len(self.chunks)) * seq_len}


def write_shuffled(parts: list[Part], out: Path, seq_len: int, seed: int) -> int:
    """Write the parts' chunks to ``out`` in shuffled order; returns the number of chunks written."""
    total = int(sum(len(p.chunks) for p in parts))
    if total == 0:
        raise ValueError("nothing to write: every source selected zero chunks")
    chunk_bytes = seq_len * BYTES_PER_TOKEN
    dest = np.random.RandomState(seed).permutation(total).astype(np.int64)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
    t0 = time.time()
    try:
        os.ftruncate(fd, total * chunk_bytes)
        g = 0
        for part in parts:
            n_avail = os.path.getsize(part.path) // chunk_bytes
            if len(part.chunks) and int(part.chunks.max()) >= n_avail:
                raise ValueError(f"{part.name}: chunk index beyond the end of {part.path}")
            src = np.memmap(part.path, dtype=np.uint16, mode="r", shape=(n_avail, seq_len)) if n_avail else None
            for b in range(0, len(part.chunks), READ_BLOCK_CHUNKS):
                idx = part.chunks[b:b + READ_BLOCK_CHUNKS]
                block = np.ascontiguousarray(src[idx]) if src is not None else None
                for row in range(len(idx)):
                    os.pwrite(fd, block[row].tobytes(), int(dest[g]) * chunk_bytes)
                    g += 1
            del src
            logger.info("  %-24s %d chunks written (%.0fs)", part.name, len(part.chunks), time.time() - t0)
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        tmp.unlink(missing_ok=True)
        raise
    os.close(fd)
    tmp.replace(out)
    return total


def build(parts: list[Part], out: Path, seq_len: int, seed: int, tokenizer: str | None, eos_id: int | None,
          parent: dict[str, Any] | None = None) -> Provenance:
    """Write the stream and its provenance; returns the provenance."""
    total = write_shuffled(parts, out, seq_len, seed)
    prov = Provenance(sources=[p.name for p in parts], chunks_per_source=[int(len(p.chunks)) for p in parts],
                      seq_len=seq_len, seed=seed, total_tokens=total * seq_len, tokenizer=tokenizer,
                      eos_id=eos_id, selection={p.name: p.describe(seq_len) for p in parts}, parent=parent,
                      output=out.name)
    prov.save(provenance_path(out))
    logger.info("wrote %s: %d chunks, %.2fB tokens; provenance %s", out, total, total * seq_len / 1e9,
                provenance_path(out))
    return prov


def parse_assignments(items: list[str], flag: str, kind: type) -> dict[str, Any]:
    """``NAME=VALUE`` pairs from a repeated flag; raises ValueError on a malformed item."""
    out: dict[str, Any] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name:
            raise ValueError(f"{flag} wants NAME=VALUE, got {item!r}")
        try:
            out[name] = kind(float(value)) if kind is int else kind(value)
        except ValueError as e:
            raise ValueError(f"{flag} {item!r}: {e}") from e
    return out


def selections(names: list[str], drop: list[str], weights: dict[str, float],
               caps: dict[str, int]) -> dict[str, Selection]:
    """Per-source selections from --drop / --weight / --cap; raises ValueError for an unknown source."""
    unknown = (set(drop) | set(weights) | set(caps)) - set(names)
    if unknown:
        raise ValueError(f"unknown sources {sorted(unknown)}; present: {names}")
    return {n: Selection(weight=0.0 if n in drop else weights.get(n, 1.0), cap_tokens=caps.get(n))
            for n in names}


def _sidecar(bin_path: Path) -> dict[str, Any]:
    side = bin_path.with_name(bin_path.name + ".json")
    if not side.exists():
        logger.warning("%s has no sidecar; its tokenizer and EOS id are unrecorded", bin_path.name)
        return {}
    return json.loads(side.read_text())


def plan_from_dir(in_dir: Path, seq_len: int, seed: int, sel: dict[str, Selection] | None = None,
                  ) -> tuple[list[Part], str | None, int | None]:
    """Parts for every ``<source>.bin`` in ``in_dir``, plus the tokenizer and EOS id they share.

    Raises ValueError when sources disagree on tokenizer or EOS id: their tokens
    would mean different things in one stream.
    """
    bins = sorted(p for p in in_dir.glob("*.bin") if p.is_file())
    if not bins:
        raise FileNotFoundError(f"no .bin files in {in_dir}")
    names = [p.stem for p in bins]
    sel = sel or {}
    chunk_bytes = seq_len * BYTES_PER_TOKEN
    parts: list[Part] = []
    tokenizers: set[str] = set()
    eos_ids: set[int] = set()
    for path, name in zip(bins, names):
        side = _sidecar(path)
        if "tokenizer" in side:
            tokenizers.add(str(side["tokenizer"]))
        if "eos_id" in side:
            eos_ids.add(int(side["eos_id"]))
        n = path.stat().st_size // chunk_bytes
        s = sel.get(name, Selection())
        parts.append(Part(name, path, select_chunks(np.arange(n, dtype=np.int64), s, seq_len, seed, name), n, s))
        if n == 0:
            logger.warning("%s is shorter than one %d-token chunk", name, seq_len)
    if len(tokenizers) > 1 or len(eos_ids) > 1:
        raise ValueError(f"sources disagree: tokenizers {sorted(tokenizers)}, EOS ids {sorted(eos_ids)}")
    return parts, next(iter(tokenizers), None), next(iter(eos_ids), None)


def print_plan(parts: list[Part], seq_len: int) -> None:
    total = sum(len(p.chunks) for p in parts) or 1
    print(f"{'source':<24} {'available':>14} {'taken':>14} {'share':>7}  weight  cap")
    for p in parts:
        print(f"{p.name:<24} {p.available * seq_len:>14,} {len(p.chunks) * seq_len:>14,} "
              f"{100 * len(p.chunks) / total:>6.2f}%  {p.selection.weight:<6g}  {p.selection.cap_tokens or '-'}")
    print(f"{'total':<24} {sum(p.available for p in parts) * seq_len:>14,} {total * seq_len:>14,}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [assemble] %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="per-source token streams -> one shuffled stream + provenance")
    ap.add_argument("--in_dir", type=Path, required=True, help="directory of <source>.bin (+ .bin.json)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seq_len", type=int, required=True, help="chunk size in tokens (the training seq_len)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop", nargs="*", default=[], help="sources to leave out")
    ap.add_argument("--weight", action="append", default=[], help="NAME=W resampling weight (default 1)")
    ap.add_argument("--cap", action="append", default=[], help="NAME=TOKENS upper bound after weighting")
    ap.add_argument("--dry_run", action="store_true", help="print the plan, write nothing")
    args = ap.parse_args(argv)
    if args.seq_len <= 0:
        ap.error("--seq_len must be positive")
    try:
        names = sorted(p.stem for p in args.in_dir.glob("*.bin"))
        sel = selections(names, args.drop, parse_assignments(args.weight, "--weight", float),
                         parse_assignments(args.cap, "--cap", int))
        parts, tokenizer, eos_id = plan_from_dir(args.in_dir, args.seq_len, args.seed, sel)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print_plan(parts, args.seq_len)
    if args.dry_run:
        return 0
    build([p for p in parts if len(p.chunks)], args.out, args.seq_len, args.seed, tokenizer, eos_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
