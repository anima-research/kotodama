"""Tokenize chunked JSONL into one uint16 token stream per source.

    python -m kotodama.data.curation.tokenize --in_dir <deduped> --out_dir <tokenized> [--workers N]

Every chunk is tokenized in a worker process (one tokenizer per process), each
document followed by the tokenizer's EOS id, which is how
``kotodama.data.dataset.TokenizedDataset`` finds document boundaries. A source's
chunk outputs are then concatenated in chunk order into ``<source>.bin``, so the
result does not depend on the number of workers. ``<source>.bin.json`` records
the token and document counts, the tokenizer and the EOS id; the assembler
refuses to mix sources whose tokenizers or EOS ids differ. A source whose
sidecar exists is skipped unless ``--force``.

The tokenizer defaults to ``kotodama.presets.TOKENIZER_NAME``. A tokenizer whose
vocabulary does not fit in uint16 is refused before any work starts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from kotodama.data.curation.chunks import discover_chunks, iter_records
from kotodama.presets import TOKENIZER_NAME

logger = logging.getLogger(__name__)

UINT16_LIMIT = 1 << 16


class BatchTokenizer(Protocol):
    """The slice of a Hugging Face tokenizer this stage uses."""

    eos_token_id: int | None

    def __call__(self, texts: list[str], add_special_tokens: bool = ...) -> Any: ...

    def __len__(self) -> int: ...


def load_tokenizer(name: str) -> BatchTokenizer:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


@dataclass
class SourceTokens:
    """The sidecar ``<source>.bin.json``."""

    source: str
    tokens: int
    docs: int
    chunks: int
    tokenizer: str
    eos_id: int
    vocab_size: int
    dtype: str = "uint16"


_TOK: BatchTokenizer | None = None


def _init_worker(tokenizer: str | BatchTokenizer) -> None:
    global _TOK
    os.environ["TOKENIZERS_PARALLELISM"] = "false"   # the process pool is the parallelism
    _TOK = load_tokenizer(tokenizer) if isinstance(tokenizer, str) else tokenizer


def encode_texts(tok: BatchTokenizer, texts: list[str], eos_id: int) -> np.ndarray:
    """Token ids of ``texts`` as one uint16 array, each document followed by ``eos_id``."""
    ids = tok(texts, add_special_tokens=False)["input_ids"]
    parts: list[np.ndarray] = []
    eos = np.array([eos_id], dtype=np.uint16)
    for doc in ids:
        if not doc:
            continue
        arr = np.asarray(doc, dtype=np.int64)
        if arr.min() < 0 or arr.max() >= UINT16_LIMIT:
            raise ValueError(f"token id out of uint16 range ({arr.min()}..{arr.max()})")
        parts.append(arr.astype(np.uint16))
        parts.append(eos)
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.uint16)


def tokenize_chunk(chunk: Path, out: Path, eos_id: int, batch_docs: int = 512) -> tuple[int, int]:
    """Tokenize one chunk into ``out`` (a raw uint16 file); returns (tokens, documents).

    Runs in a worker after ``_init_worker`` (or in-process after it is called directly).
    """
    if _TOK is None:
        raise RuntimeError("tokenizer not initialized in this process")
    tokens = docs = 0
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        batch: list[str] = []
        for rec in iter_records(chunk):
            batch.append(rec.text)
            if len(batch) >= batch_docs:
                arr = encode_texts(_TOK, batch, eos_id)
                f.write(arr.tobytes())
                tokens, docs, batch = tokens + len(arr), docs + len(batch), []
        if batch:
            arr = encode_texts(_TOK, batch, eos_id)
            f.write(arr.tobytes())
            tokens, docs = tokens + len(arr), docs + len(batch)
    tmp.replace(out)
    return tokens, docs


def tokenize_dir(in_dir: Path, out_dir: Path, tokenizer: str | BatchTokenizer = TOKENIZER_NAME,
                 sources: list[str] | None = None, workers: int = 8, batch_docs: int = 512,
                 force: bool = False, eos_id: int | None = None) -> dict[str, SourceTokens]:
    """Tokenize every source (or ``sources``) of ``in_dir`` into ``out_dir``; returns each source's sidecar.

    ``tokenizer`` is a name (loaded with ``transformers.AutoTokenizer``) or a
    tokenizer object; ``workers=0`` tokenizes in this process. ``eos_id``
    defaults to the tokenizer's. Raises ValueError for a tokenizer without an EOS
    id or with a vocabulary beyond uint16, and FileNotFoundError for a missing source.
    """
    tok = load_tokenizer(tokenizer) if isinstance(tokenizer, str) else tokenizer
    tok_name = tokenizer if isinstance(tokenizer, str) else str(
        getattr(tokenizer, "name_or_path", None) or type(tokenizer).__name__)
    vocab = len(tok)
    if vocab > UINT16_LIMIT:
        raise ValueError(f"tokenizer vocabulary {vocab} does not fit the uint16 stream format")
    eos = eos_id if eos_id is not None else tok.eos_token_id
    if eos is None:
        raise ValueError("tokenizer has no EOS id; pass eos_id")

    by_source = discover_chunks(in_dir)
    if sources:
        missing = set(sources) - set(by_source)
        if missing:
            raise FileNotFoundError(f"no chunks for {sorted(missing)} in {in_dir}")
        by_source = {s: by_source[s] for s in sources}
    if not by_source:
        raise FileNotFoundError(f"no chunk files in {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, SourceTokens] = {}
    todo: dict[str, list[Path]] = {}
    for s, chunks in by_source.items():
        side = out_dir / f"{s}.bin.json"
        if side.exists() and not force:
            results[s] = SourceTokens(**json.loads(side.read_text()))
            logger.info("%s: tokenized already; --force redoes it", s)
        else:
            side.unlink(missing_ok=True)
            todo[s] = chunks
    if not todo:
        return results

    parts_dir = out_dir / ".parts"
    parts_dir.mkdir(exist_ok=True)
    tasks = [(s, cp, parts_dir / f"{cp.name}.u16") for s, chunks in todo.items() for cp in chunks]
    t0 = time.time()
    counts: dict[Path, tuple[int, int]] = {}
    if workers <= 0:
        _init_worker(tok)
        for _, cp, part in tasks:
            counts[part] = tokenize_chunk(cp, part, eos, batch_docs)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(tokenizer,)) as pool:
            futures = {part: pool.submit(tokenize_chunk, cp, part, eos, batch_docs) for _, cp, part in tasks}
            for part, fut in futures.items():
                counts[part] = fut.result()

    for s, chunks in todo.items():
        out = out_dir / f"{s}.bin"
        tmp = out.with_name(out.name + ".tmp")
        n_tok = n_docs = 0
        with open(tmp, "wb") as f:
            for cp in chunks:
                part = parts_dir / f"{cp.name}.u16"
                with open(part, "rb") as src:
                    shutil.copyfileobj(src, f, length=1 << 24)
                n_tok += counts[part][0]
                n_docs += counts[part][1]
                part.unlink()
        tmp.replace(out)
        side = SourceTokens(source=s, tokens=n_tok, docs=n_docs, chunks=len(chunks), tokenizer=tok_name,
                            eos_id=int(eos), vocab_size=vocab)
        (out_dir / f"{s}.bin.json").write_text(json.dumps(asdict(side), indent=1))
        results[s] = side
        logger.info("%s: %d docs, %.1fM tokens -> %s", s, n_docs, n_tok / 1e6, out)
    shutil.rmtree(parts_dir, ignore_errors=True)
    logger.info("tokenized %d sources in %.0fs", len(todo), time.time() - t0)
    return results


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [tokenize] %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="chunked JSONL -> one uint16 token stream per source")
    ap.add_argument("--in_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--sources", nargs="*", help="only these sources (default: all in --in_dir)")
    ap.add_argument("--tokenizer", default=TOKENIZER_NAME)
    ap.add_argument("--eos_id", type=int, help="default: the tokenizer's EOS id")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch_docs", type=int, default=512)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    try:
        res = tokenize_dir(args.in_dir, args.out_dir, args.tokenizer, args.sources, args.workers,
                           args.batch_docs, args.force, args.eos_id)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps({s: asdict(r) for s, r in res.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
