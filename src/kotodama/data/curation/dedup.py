"""Cross-source deduplication of chunked JSONL: exact, or MinHash LSH near-duplicates.

    python -m kotodama.data.curation.dedup --in_dir <cleaned> --out_dir <deduped> \\
        [--mode minhash|exact] [--priority wikipedia=10 --priority web=3 ...] [--workers N]

Three phases:

1. **Sign** every chunk in parallel. Each document gets a 64-bit hash of its text
   (exact mode) or, in minhash mode, that plus a MinHash signature: word 5-gram
   shingles of the lower-cased text, each hashed with xxh32, minimized under 112
   random affine permutations modulo a prime above 2^32. Signatures go to one
   shard per chunk in ``--index_dir``; a chunk whose size and mtime match the
   index manifest is not signed again, so adding a source re-signs only its chunks.
2. **Cluster**: documents whose hashes collide are joined (union-find). In
   minhash mode the signature is cut into 14 bands of 8 rows and two documents
   collide when any band matches, which finds pairs above roughly 0.7 Jaccard
   similarity with high probability ((1/14)^(1/8) is the threshold of the S-curve).
   In each cluster the document of the highest-priority source survives; ties go
   to the earliest document (sources in name order, then chunk, then line).
3. **Write** each chunk to ``--out_dir`` without its removed lines, in parallel,
   plus ``dedup.summary.json`` with per-source totals.

Exact duplicates always collide in minhash mode too, so ``exact`` is the cheaper
subset, not a different criterion.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xxhash

from kotodama.data.curation.chunks import ChunkWriter, discover_chunks, iter_lines, iter_records

logger = logging.getLogger(__name__)

Mode = Literal["exact", "minhash"]

N_GRAMS = 5
NUM_BANDS = 14
ROWS_PER_BAND = 8
NUM_HASHES = NUM_BANDS * ROWS_PER_BAND
SEED = 42
MAX_HASH = np.uint64(2**32 - 1)
LARGE_PRIME = np.uint64(4294967311)
SIG_BLOCK = 4096   # shingles per vectorized block; bounds memory on very long documents
INDEX_VERSION = 1

_rng = np.random.RandomState(SEED)
HASH_A = _rng.randint(1, MAX_HASH, size=NUM_HASHES, dtype=np.uint64)
HASH_B = _rng.randint(0, MAX_HASH, size=NUM_HASHES, dtype=np.uint64)
del _rng


def record_dtype(mode: Mode) -> np.dtype:
    """One shard record: the document's line in its chunk, its text hash, and (minhash) its signature."""
    parts: list[tuple[Any, ...]] = [("line", "<u4"), ("exact", "<u8")]
    if mode == "minhash":
        parts.append(("sig", "<u4", (NUM_HASHES,)))
    return np.dtype(parts)


def exact_hash(text: str) -> int:
    """64-bit hash of a document's text with whitespace runs collapsed."""
    return xxhash.xxh3_64_intdigest(" ".join(text.split()).encode())


def minhash_signature(text: str, n_grams: int = N_GRAMS) -> np.ndarray | None:
    """The (NUM_HASHES,) uint32 MinHash signature of a text; None for a text with no words.

    A text shorter than ``n_grams`` words is one shingle of all its words.
    """
    words = text.lower().split()
    if not words:
        return None
    if len(words) < n_grams:
        shingles = [" ".join(words)]
    else:
        shingles = [" ".join(words[i:i + n_grams]) for i in range(len(words) - n_grams + 1)]
    h32 = xxhash.xxh32_intdigest
    h = np.fromiter((h32(s, SEED) for s in shingles), dtype=np.uint64, count=len(shingles))
    sig = np.full(NUM_HASHES, MAX_HASH, dtype=np.uint32)
    for start in range(0, len(h), SIG_BLOCK):
        block = h[start:start + SIG_BLOCK, None]
        vals = ((HASH_A[None, :] * block + HASH_B[None, :]) % LARGE_PRIME).astype(np.uint32)
        np.minimum(sig, vals.min(axis=0), out=sig)
    return sig


def sign_chunk(chunk: Path, shard: Path, mode: Mode) -> int:
    """Write the shard of one chunk; returns the number of documents signed."""
    rows = []
    for rec in iter_records(chunk):
        if mode == "minhash":
            sig = minhash_signature(rec.text)
            if sig is None:
                continue
            rows.append((rec.line, exact_hash(rec.text), sig))
        else:
            rows.append((rec.line, exact_hash(rec.text)))
    arr = np.array(rows, dtype=record_dtype(mode))
    tmp = shard.with_name(shard.name + ".tmp")
    arr.tofile(tmp)
    tmp.replace(shard)   # a partial shard never enters the index
    return len(arr)


@dataclass
class IndexParams:
    mode: Mode
    version: int = INDEX_VERSION
    n_grams: int = N_GRAMS
    num_bands: int = NUM_BANDS
    rows_per_band: int = ROWS_PER_BAND
    seed: int = SEED


def _shard_name(chunk: Path) -> str:
    return chunk.name.removesuffix(".zst").removesuffix(".jsonl") + ".shard"


def sign_all(by_source: dict[str, list[Path]], index_dir: Path, mode: Mode, workers: int) -> dict[str, Any]:
    """Phase 1: sign every chunk the index does not already hold; returns the index manifest."""
    index_dir.mkdir(parents=True, exist_ok=True)
    mpath = index_dir / "manifest.json"
    params = IndexParams(mode=mode).__dict__
    manifest: dict[str, Any] = {"params": params, "chunks": {}}
    if mpath.exists():
        try:
            prev = json.loads(mpath.read_text())
            if prev.get("params") == params:
                manifest = prev
            else:
                logger.info("index params differ (%s); re-signing everything", prev.get("params"))
        except json.JSONDecodeError:
            logger.warning("unreadable %s; re-signing everything", mpath)
    known: dict[str, Any] = manifest["chunks"]

    todo: list[tuple[Path, Path]] = []
    for chunks in by_source.values():
        for cp in chunks:
            st = cp.stat()
            prev = known.get(cp.name)
            shard = index_dir / _shard_name(cp)
            if prev and prev["size"] == st.st_size and abs(prev["mtime"] - st.st_mtime) < 1 and shard.exists():
                continue
            todo.append((cp, shard))
    logger.info("sign: %d chunks to sign, %d reused", len(todo), sum(map(len, by_source.values())) - len(todo))

    def note(cp: Path, n: int) -> None:
        st = cp.stat()
        known[cp.name] = {"shard": _shard_name(cp), "docs": n, "size": st.st_size, "mtime": st.st_mtime}

    if workers <= 0:
        for cp, shard in todo:
            note(cp, sign_chunk(cp, shard, mode))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [(cp, pool.submit(sign_chunk, cp, shard, mode)) for cp, shard in todo]
            for cp, fut in futures:
                note(cp, fut.result())   # a failed chunk raises here and stops the run
    tmp = mpath.with_name(mpath.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1))
    tmp.replace(mpath)
    return manifest


def band_keys(sigs: np.ndarray, band: int, rows: int = ROWS_PER_BAND) -> np.ndarray:
    """One uint64 key per document for one LSH band (a mixing hash over the band's rows)."""
    part = np.ascontiguousarray(sigs[:, band * rows:(band + 1) * rows]).view(np.uint64)
    golden = np.uint64(0x9E3779B97F4A7C15)
    keys = part[:, 0] ^ np.uint64((band * 0x9E3779B97F4A7C15 + 1) & 0xFFFFFFFFFFFFFFFF)
    for i in range(1, part.shape[1]):
        keys ^= part[:, i] + golden + (keys << np.uint64(6)) + (keys >> np.uint64(2))
    return keys


def collision_groups(keys: np.ndarray) -> list[np.ndarray]:
    """Indices of documents sharing a key, one array per key held by two or more."""
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    bounds = np.flatnonzero(np.concatenate([[True], k[1:] != k[:-1], [True]]))
    return [order[s:e] for s, e in zip(bounds[:-1], bounds[1:]) if e - s > 1]


def _band_groups(path: str, n: int, band: int) -> list[np.ndarray]:
    sigs = np.memmap(path, dtype=np.uint32, mode="r", shape=(n, NUM_HASHES))
    return collision_groups(band_keys(sigs, band))


class UnionFind:
    """Union-find over 0..n-1 with path halving and union by size."""

    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = int(p[x])
        return x

    def union_group(self, members: np.ndarray) -> None:
        root = self.find(int(members[0]))
        for m in members[1:]:
            r = self.find(int(m))
            if r == root:
                continue
            if self.size[r] > self.size[root]:
                root, r = r, root
            self.parent[r] = root
            self.size[root] += self.size[r]

    def roots(self) -> np.ndarray:
        """Every element's root, by vectorized pointer jumping."""
        p = self.parent.copy()
        while True:
            pp = p[p]
            if np.array_equal(pp, p):
                return p
            p = pp


def find_duplicates(by_source: dict[str, list[Path]], index_dir: Path, manifest: dict[str, Any],
                    priority: dict[str, int], workers: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Phase 2: cluster colliding documents and pick each cluster's survivor.

    Returns ({chunk file name: sorted line numbers to remove}, per-source summary).
    """
    mode: Mode = manifest["params"]["mode"]
    dtype = record_dtype(mode)
    shards: list[np.ndarray] = []
    chunk_of: list[np.ndarray] = []
    for chunks in by_source.values():
        for cp in chunks:
            arr = np.fromfile(index_dir / manifest["chunks"][cp.name]["shard"], dtype=dtype)
            shards.append(arr)
            chunk_of.append(np.full(len(arr), len(chunk_of), dtype=np.int32))
    chunk_names = [cp.name for chunks in by_source.values() for cp in chunks]
    chunk_source = np.array([s for s, chunks in by_source.items() for _ in chunks])
    if not shards:
        return {}, {"total_docs": 0, "removed": 0, "clusters": 0, "per_source": {}}
    recs = np.concatenate(shards)
    doc_chunk = np.concatenate(chunk_of)
    n = len(recs)
    uf = UnionFind(n)
    for g in collision_groups(recs["exact"]):
        uf.union_group(g)
    if mode == "minhash" and n > 1:
        mmap_path = index_dir / "_signatures.u32"
        mm = np.memmap(mmap_path, dtype=np.uint32, mode="w+", shape=(n, NUM_HASHES))
        mm[:] = recs["sig"]
        mm.flush()
        del mm
        try:
            if workers <= 0:
                results = [_band_groups(str(mmap_path), n, b) for b in range(NUM_BANDS)]
            else:
                with ProcessPoolExecutor(max_workers=min(workers, NUM_BANDS)) as pool:
                    results = list(pool.map(_band_groups, [str(mmap_path)] * NUM_BANDS, [n] * NUM_BANDS,
                                            range(NUM_BANDS)))
            for groups in results:
                for g in groups:
                    uf.union_group(g)
        finally:
            mmap_path.unlink(missing_ok=True)

    roots = uf.roots()
    doc_prio = np.array([priority.get(str(s), 0) for s in chunk_source], dtype=np.int64)[doc_chunk]
    # Survivor = highest priority, then lowest index: sort by (root, -priority, index), keep each root's first.
    order = np.lexsort((np.arange(n), -doc_prio, roots))
    first = np.concatenate([[True], roots[order][1:] != roots[order][:-1]])
    remove = np.zeros(n, dtype=bool)
    remove[order[~first]] = True
    cluster_sizes = np.bincount(roots, minlength=n)

    removal: dict[str, list[int]] = defaultdict(list)
    for i in np.flatnonzero(remove):
        removal[chunk_names[doc_chunk[i]]].append(int(recs["line"][i]))
    per_source: dict[str, dict[str, int]] = {}
    for s in by_source:
        in_s = chunk_source[doc_chunk] == s
        per_source[s] = {"docs": int(in_s.sum()), "removed": int((remove & in_s).sum())}
    summary = {"mode": mode, "total_docs": n, "removed": int(remove.sum()),
               "clusters": int((cluster_sizes > 1).sum()), "per_source": per_source}
    return {k: np.array(sorted(v), dtype=np.uint32) for k, v in removal.items()}, summary


def write_chunk(chunk: Path, out: Path, remove: np.ndarray) -> tuple[int, int]:
    """Copy a chunk without the given (sorted) line numbers; returns (kept, removed)."""
    kept = removed = ri = 0
    with ChunkWriter(out) as w:
        for i, line in enumerate(iter_lines(chunk)):
            if ri < len(remove) and i == remove[ri]:
                ri += 1
                removed += 1
                continue
            if line.strip():
                w.write_line(line if line.endswith(b"\n") else line + b"\n")
                kept += 1
    return kept, removed


def dedup(in_dir: Path, out_dir: Path | None, index_dir: Path | None = None, mode: Mode = "minhash",
          priority: dict[str, int] | None = None, workers: int = 8) -> dict[str, Any]:
    """Run all three phases (``out_dir=None`` stops after phase 2: a dry run). Returns the summary."""
    if mode not in ("exact", "minhash"):
        raise ValueError(f"mode must be 'exact' or 'minhash', got {mode!r}")
    if out_dir is not None and out_dir.resolve() == in_dir.resolve():
        raise ValueError("out_dir must differ from in_dir (chunks are rewritten, not edited)")
    t0 = time.time()
    by_source = discover_chunks(in_dir)
    if not by_source:
        raise FileNotFoundError(f"no chunk files in {in_dir}")
    index_dir = index_dir or (out_dir or in_dir) / "dedup_index"
    manifest = sign_all(by_source, index_dir, mode, workers)
    removal, summary = find_duplicates(by_source, index_dir, manifest, priority or {}, workers)
    for s, c in summary["per_source"].items():
        logger.info("  %-24s %d docs, %d removed", s, c["docs"], c["removed"])
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        empty = np.empty(0, dtype=np.uint32)
        tasks = [(cp, out_dir / cp.name, removal.get(cp.name, empty)) for chunks in by_source.values()
                 for cp in chunks]
        if workers <= 0:
            for t in tasks:
                write_chunk(*t)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for fut in [pool.submit(write_chunk, *t) for t in tasks]:
                    fut.result()
        summary["elapsed_s"] = round(time.time() - t0, 1)
        (out_dir / "dedup.summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def parse_priority(items: list[str]) -> dict[str, int]:
    """``name=int`` pairs; raises ValueError on a malformed item."""
    out: dict[str, int] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name:
            raise ValueError(f"--priority wants name=int, got {item!r}")
        out[name] = int(value)
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [dedup] %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="exact / MinHash LSH dedup over chunked JSONL")
    ap.add_argument("--in_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, help="omit for a dry run (report only)")
    ap.add_argument("--index_dir", type=Path, help="signature index (default <out_dir>/dedup_index)")
    ap.add_argument("--mode", choices=["minhash", "exact"], default="minhash")
    ap.add_argument("--priority", action="append", default=[],
                    help="source=int; the higher-priority copy of a duplicate survives (default 0)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    try:
        priority = parse_priority(args.priority)
    except ValueError as e:
        ap.error(str(e))
    summary = dedup(args.in_dir, args.out_dir, args.index_dir, args.mode, priority, args.workers)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
