"""Chunked JSONL: the on-disk layout every curation stage reads and writes.

A source's documents live in chunk files named ``<source>.<NNNN>.jsonl`` (or
``.jsonl.zst``), one JSON object per line: ``{"text": str, "meta": {...}}``. A
chunk is the unit of parallel work in every stage, so a large source spreads over
many workers instead of one. Each file is written to a temporary name and renamed
into place, so a reader never sees a half-written chunk.

Compressed chunks need the ``zstandard`` package (the ``curation`` extra); plain
chunks need nothing beyond the standard library.
"""

from __future__ import annotations

import io
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Iterator

logger = logging.getLogger(__name__)

CHUNK_NAME_RE = re.compile(r"^(?P<source>.+?)\.(?P<chunk>\d{4,})\.jsonl(?P<zst>\.zst)?$")
SOURCE_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_\-]*$")


class ChunkError(ValueError):
    """A chunk file or source name is malformed."""


def check_source_name(name: str) -> str:
    """Return ``name`` if it is a usable source name, else raise ChunkError.

    Source names become file stems and manifest keys; dots would make a chunk
    name ambiguous, and path separators would escape the output directory.
    """
    if not SOURCE_NAME_RE.match(name):
        raise ChunkError(f"source name {name!r} must match {SOURCE_NAME_RE.pattern}")
    return name


def chunk_name(source: str, index: int, zstd: bool) -> str:
    """The file name of chunk ``index`` of ``source``."""
    return f"{check_source_name(source)}.{index:04d}.jsonl" + (".zst" if zstd else "")


def _zstandard() -> Any:
    try:
        import zstandard
    except ImportError as e:  # the optional dependency is named in the message
        raise ImportError("compressed chunks (.jsonl.zst) need the zstandard package: "
                          "install the curation extra (pip install -e '.[curation]')") from e
    return zstandard


def iter_lines(path: Path) -> Iterator[bytes]:
    """Yield the raw lines of a ``.jsonl`` or ``.jsonl.zst`` file."""
    if path.name.endswith(".zst"):
        zstd = _zstandard()
        with open(path, "rb") as raw:
            reader = zstd.ZstdDecompressor().stream_reader(raw)
            yield from io.BufferedReader(reader, buffer_size=1 << 20)
    else:
        with open(path, "rb") as f:
            yield from f


@dataclass
class Record:
    """One document of a chunk, with its line number inside the chunk."""

    line: int
    text: str
    meta: dict[str, Any]


def iter_records(path: Path) -> Iterator[Record]:
    """Yield the documents of a chunk. Malformed lines and empty texts are skipped and counted in a log line."""
    bad = 0
    for i, line in enumerate(iter_lines(path)):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            bad += 1
            continue
        text = obj.get("text") if isinstance(obj, dict) else None
        if not isinstance(text, str) or not text:
            bad += 1
            continue
        meta = obj.get("meta")
        yield Record(line=i, text=text, meta=meta if isinstance(meta, dict) else {})
    if bad:
        logger.warning("%s: skipped %d malformed or empty lines", path.name, bad)


class ChunkWriter:
    """Write a chunk atomically: lines go to ``<name>.tmp`` and the file is renamed on ``close``.

    Used as a context manager; an exception inside the block removes the temporary
    file and leaves any existing chunk of that name untouched.
    """

    def __init__(self, path: Path, zstd_level: int = 6) -> None:
        self.path = path
        self.tmp = path.with_name(path.name + ".tmp")
        self.n = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._raw: IO[bytes] = open(self.tmp, "wb")
        self._zstream: Any = None
        if path.name.endswith(".zst"):
            self._zstream = _zstandard().ZstdCompressor(level=zstd_level).stream_writer(self._raw)
        self._buf = io.BufferedWriter(self._zstream if self._zstream is not None else self._raw,
                                      buffer_size=1 << 20)

    def write(self, text: str, meta: dict[str, Any]) -> None:
        self._buf.write(json.dumps({"text": text, "meta": meta}, ensure_ascii=False).encode() + b"\n")
        self.n += 1

    def write_line(self, line: bytes) -> None:
        """Copy a raw line through unchanged (it must end with a newline)."""
        self._buf.write(line)
        self.n += 1

    def close(self) -> None:
        self._buf.flush()
        if self._zstream is not None:
            self._zstream.close()   # finishes the zstd frame and closes the file under it
        if not self._raw.closed:
            self._raw.close()
        self.tmp.replace(self.path)

    def abort(self) -> None:
        try:
            if self._zstream is not None:
                self._zstream.close()
        finally:
            if not self._raw.closed:
                self._raw.close()
            self.tmp.unlink(missing_ok=True)

    def __enter__(self) -> "ChunkWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def discover_chunks(directory: Path) -> dict[str, list[Path]]:
    """Map each source name to its chunk files, in chunk order.

    Raises ChunkError when one chunk number appears both compressed and plain,
    which would read the same documents twice.
    """
    by_source: dict[str, dict[int, Path]] = defaultdict(dict)
    for p in sorted(directory.iterdir()):
        m = CHUNK_NAME_RE.match(p.name)
        if not m or not p.is_file():
            continue
        idx = int(m.group("chunk"))
        if idx in by_source[m.group("source")]:
            raise ChunkError(f"chunk {idx} of {m.group('source')} exists twice in {directory}")
        by_source[m.group("source")][idx] = p
    return {s: [chunks[i] for i in sorted(chunks)] for s, chunks in sorted(by_source.items())}


def source_of(path: Path) -> str:
    """The source name a chunk file belongs to."""
    m = CHUNK_NAME_RE.match(path.name)
    if not m:
        raise ChunkError(f"{path.name} is not a chunk name (<source>.NNNN.jsonl[.zst])")
    return m.group("source")
