"""Ingest a source: read it through its adapter, clean and filter every document, write chunks.

    python -m kotodama.data.curation.ingest --preset smoltalk --out_dir <cleaned>
    python -m kotodama.data.curation.ingest --hf <repo> --name <src> --class web --out_dir <cleaned>
    python -m kotodama.data.curation.ingest --local <dir|file> --name <src> --out_dir <cleaned>
    python -m kotodama.data.curation.ingest --spec <source.yaml> --out_dir <cleaned>
    python -m kotodama.data.curation.ingest --list_presets

One process reads the source (adapters are sequential streams); batches of
``--docs_per_chunk`` documents go to ``--workers`` processes, each of which
cleans (``kotodama.data.curation.clean``), filters
(``kotodama.data.curation.filters``) and writes one chunk. Chunk boundaries
depend only on the document order and ``--docs_per_chunk``, so a rerun writes
the same files. ``<source>.stats.json`` records kept / repaired / dropped counts
with the reason each rule fired, and marks the source as done: a rerun skips a
source with stats unless ``--force``. A rerun of an unfinished source removes
its old chunks first.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from kotodama.data.curation.chunks import ChunkWriter, chunk_name, discover_chunks
from kotodama.data.curation.clean import clean_document
from kotodama.data.curation.filters import (
    DEFAULT_THRESHOLDS, Action, FilterThresholds, LangId, SourceClass, apply_filters, fasttext_langid,
)
from kotodama.data.curation.sources import PRESETS, RawDoc, SourceError, SourceSpec, iter_docs

logger = logging.getLogger(__name__)


@dataclass
class IngestStats:
    """Per-source accounting; ``<source>.stats.json`` holds one."""

    source: str
    source_class: str
    docs_in: int = 0
    kept: int = 0
    repaired: int = 0
    dropped: int = 0
    chars_in: int = 0
    chars_out: int = 0
    chunks: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)
    repair_reasons: dict[str, int] = field(default_factory=dict)
    elapsed_s: float = 0.0
    spec: dict[str, Any] = field(default_factory=dict)

    def merge(self, other: "IngestStats") -> None:
        for k in ("docs_in", "kept", "repaired", "dropped", "chars_in", "chars_out", "chunks"):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        for mine, theirs in ((self.drop_reasons, other.drop_reasons),
                             (self.repair_reasons, other.repair_reasons)):
            for reason, n in theirs.items():
                mine[reason] = mine.get(reason, 0) + n

    @classmethod
    def load(cls, path: Path) -> "IngestStats":
        return cls(**json.loads(path.read_text()))


_LANGID_CACHE: dict[str, LangId] = {}


def _langid(path: str | None) -> LangId | None:
    if path is None:
        return None
    if path not in _LANGID_CACHE:
        _LANGID_CACHE[path] = fasttext_langid(path)
    return _LANGID_CACHE[path]


def process_batch(docs: list[RawDoc], out_path: Path, spec: SourceSpec,
                  thresholds: FilterThresholds = DEFAULT_THRESHOLDS, langid_model: str | None = None,
                  zstd_level: int = 6) -> IngestStats:
    """Clean and filter one batch of documents and write the survivors to ``out_path`` (one chunk)."""
    st = IngestStats(source=spec.name, source_class=spec.source_class.value, chunks=1)
    drops: Counter[str] = Counter()
    repairs: Counter[str] = Counter()
    langid = _langid(langid_model)
    with ChunkWriter(out_path, zstd_level=zstd_level) as w:
        for doc in docs:
            st.docs_in += 1
            st.chars_in += len(doc.text)
            cleaned = clean_document(doc.text, min_chars=spec.min_chars, max_chars=spec.max_chars,
                                     scrub=spec.scrub_pii)
            if cleaned is None:
                st.dropped += 1
                drops["length"] += 1
                continue
            verdict = apply_filters(cleaned, spec.source_class, thresholds, langid)
            if verdict.action is Action.DROP or verdict.text is None:
                st.dropped += 1
                drops.update(verdict.reasons)
                continue
            if verdict.action is Action.REPAIR:
                st.repaired += 1
                repairs.update(verdict.reasons)
            else:
                st.kept += 1
            st.chars_out += len(verdict.text)
            w.write(verdict.text, {**doc.meta, "source": spec.name})
    st.drop_reasons, st.repair_reasons = dict(drops), dict(repairs)
    return st


def _batches(docs: Iterator[RawDoc], size: int) -> Iterator[list[RawDoc]]:
    batch: list[RawDoc] = []
    for doc in docs:
        batch.append(doc)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _take(docs: Iterator[RawDoc], limit: int | None) -> Iterator[RawDoc]:
    """Stop after ``limit`` raw documents (a cap on input; kept counts come out at most this)."""
    for i, doc in enumerate(docs):
        if limit is not None and i >= limit:
            return
        yield doc


def ingest(spec: SourceSpec, out_dir: Path, workers: int = 8, docs_per_chunk: int = 50_000,
           zstd: bool = False, force: bool = False, thresholds: FilterThresholds = DEFAULT_THRESHOLDS,
           langid_model: str | None = None) -> IngestStats:
    """Ingest one source into ``out_dir``; returns its stats (read back from disk when already done).

    ``workers=0`` runs every batch in this process. ``spec.max_docs`` caps the raw
    documents read. Raises SourceError for a bad spec and re-raises the first
    worker failure after the remaining batches finish, leaving no stats file, so
    the source reads as unfinished.
    """
    if docs_per_chunk <= 0:
        raise ValueError("docs_per_chunk must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / f"{spec.name}.stats.json"
    if stats_path.exists() and not force:
        logger.info("%s: done already (%s); --force redoes it", spec.name, stats_path)
        return IngestStats.load(stats_path)
    stats_path.unlink(missing_ok=True)
    for old in discover_chunks(out_dir).get(spec.name, []):
        old.unlink()
    if langid_model is not None:
        _langid(langid_model)   # fail on a missing model or package before reading the source

    total = IngestStats(source=spec.name, source_class=spec.source_class.value, spec=spec.to_dict())
    t0 = time.time()
    batches = enumerate(_batches(_take(iter_docs(spec), spec.max_docs), docs_per_chunk))

    def out(i: int) -> Path:
        return out_dir / chunk_name(spec.name, i, zstd)

    if workers <= 0:
        for i, batch in batches:
            total.merge(process_batch(batch, out(i), spec, thresholds, langid_model))
    else:
        errors: list[BaseException] = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending: deque[Future[IngestStats]] = deque()

            def drain(limit: int) -> None:
                while len(pending) > limit:
                    try:
                        total.merge(pending.popleft().result())
                    except Exception as e:   # collected; the source is left unfinished
                        errors.append(e)

            for i, batch in batches:
                pending.append(pool.submit(process_batch, batch, out(i), spec, thresholds, langid_model))
                drain(2 * workers)
                if i and i % 20 == 0:
                    logger.info("%s: %d chunks submitted, %d docs in, %d kept (%.0fs)", spec.name, i + 1,
                                total.docs_in, total.kept + total.repaired, time.time() - t0)
            drain(0)
        if errors:
            raise RuntimeError(f"{spec.name}: {len(errors)} batch(es) failed; first: {errors[0]!r}") from errors[0]

    total.elapsed_s = round(time.time() - t0, 1)
    tmp = stats_path.with_name(stats_path.name + ".tmp")
    tmp.write_text(json.dumps(asdict(total), indent=1))
    tmp.replace(stats_path)
    logger.info("%s: in=%d kept=%d repaired=%d dropped=%d, %d chunks, %.1fs", spec.name, total.docs_in,
                total.kept, total.repaired, total.dropped, total.chunks, total.elapsed_s)
    return total


def build_spec(args: argparse.Namespace) -> SourceSpec:
    """The spec a command line describes: one of --preset / --spec / --hf / --local / --adapter, then overrides."""
    chosen = [a for a in ("preset", "spec", "hf", "local", "adapter") if getattr(args, a)]
    if len(chosen) != 1:
        raise SourceError("give exactly one of --preset, --spec, --hf, --local, --adapter")
    overrides: dict[str, Any] = {
        "text_field": args.text_field, "split": args.split, "format": args.format,
        "min_chars": args.min_chars, "max_docs": args.max_docs,
        "source_class": SourceClass(args.source_class) if args.source_class else None,
        "configs": tuple(args.config) if args.config else None,
        "scrub_pii": False if args.no_scrub else None,
    }
    if args.preset:
        if args.preset not in PRESETS:
            raise SourceError(f"unknown preset {args.preset!r}; --list_presets shows them")
        return PRESETS[args.preset].with_overrides(name=args.name, **overrides)
    if args.spec:
        from kotodama.yaml_config import load_yaml_config

        return SourceSpec.from_dict(load_yaml_config(args.spec)).with_overrides(name=args.name, **overrides)
    if not args.name:
        raise SourceError("--name is required with --hf, --local and --adapter")
    if args.hf:
        base = SourceSpec(args.name, adapter="hf", dataset=args.hf)
    elif args.local:
        base = SourceSpec(args.name, adapter="local", dataset=args.local)
    else:
        base = SourceSpec(args.name, adapter=args.adapter)
    return base.with_overrides(**overrides)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [ingest] %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="source -> cleaned, filtered JSONL chunks")
    src = ap.add_argument_group("source (exactly one)")
    src.add_argument("--preset", help="a public dataset preset (--list_presets)")
    src.add_argument("--spec", help="a YAML source spec (keys of kotodama.data.curation.sources.SourceSpec)")
    src.add_argument("--hf", help="a Hugging Face dataset id")
    src.add_argument("--local", help="a file or directory of jsonl / parquet / txt")
    src.add_argument("--adapter", help="module:function yielding RawDoc for a custom source")
    ov = ap.add_argument_group("spec overrides")
    ov.add_argument("--name", help="source name (file stem and manifest key)")
    ov.add_argument("--class", dest="source_class", choices=[c.value for c in SourceClass])
    ov.add_argument("--config", action="append", help="HF config name (repeatable)")
    ov.add_argument("--split")
    ov.add_argument("--text_field")
    ov.add_argument("--format")
    ov.add_argument("--min_chars", type=int)
    ov.add_argument("--max_docs", type=int, help="read at most this many raw documents")
    ov.add_argument("--no_scrub", action="store_true", help="keep e-mail / phone / SSN / IP strings")
    ap.add_argument("--out_dir", type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--docs_per_chunk", type=int, default=50_000)
    ap.add_argument("--zstd", action="store_true", help="write .jsonl.zst chunks (curation extra)")
    ap.add_argument("--langid_model", help="fastText lid.176.bin: decides the non-Latin midband")
    ap.add_argument("--force", action="store_true", help="redo a source that has stats")
    ap.add_argument("--list_presets", action="store_true")
    args = ap.parse_args(argv)

    if args.list_presets:
        for name, spec in PRESETS.items():
            where = spec.dataset if spec.data_files is None else spec.data_files
            print(f"{name:<18} {spec.adapter:<6} {spec.source_class.value:<11} {where}")
        return 0
    if args.out_dir is None:
        ap.error("--out_dir is required")
    try:
        spec = build_spec(args)
    except (SourceError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    stats = ingest(spec, args.out_dir, workers=args.workers, docs_per_chunk=args.docs_per_chunk,
                   zstd=args.zstd, force=args.force, langid_model=args.langid_model)
    print(json.dumps({k: v for k, v in asdict(stats).items() if k != "spec"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
