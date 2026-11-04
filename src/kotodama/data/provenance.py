"""Which source every chunk of an assembled token stream came from.

An assembled stream (``kotodama.data.curation.assemble``) is a flat uint16 file
cut into ``seq_len``-token chunks. Its sources' chunks are concatenated in
source order, and chunk ``i`` of that concatenation is written to output chunk
``perm[i]``, where ``perm = numpy.random.RandomState(seed).permutation(total_chunks)``.
The provenance file beside the stream, ``<stream>.provenance.json``, records the
source order, each source's chunk count and the seed, which is all it takes to
label every output chunk with its source (``chunk_labels``). The file is small
whatever the stream's size.

Two layouts load here:

- **chunk shuffle** (``<stem>.provenance.json`` with ``chunks_per_source``): the
  layout above. Its ``sources`` entries may be source names or the paths of the
  per-source ``.bin`` files they were read from; the name is the file stem.
- **block spans** (``<dir>/provenance.json`` + ``<dir>/manifest.json``, the stream's
  directory): a list of ``{source, global_start, global_end}`` token spans, each
  a whole number of chunks.

A stream with neither has no labels; tools that need them say so.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

FORMAT = "kotodama.chunk-shuffle/1"
BYTES_PER_TOKEN = 2   # the uint16 stream format of kotodama.data.dataset.TokenizedDataset


class ProvenanceError(ValueError):
    """A provenance file is missing, malformed, or disagrees with its stream."""


def provenance_path(stream: Path) -> Path:
    """``<dir>/<stem>.provenance.json`` for ``<dir>/<stem>.bin``."""
    return stream.with_suffix(".provenance.json")


def source_name(entry: str) -> str:
    """A ``sources`` entry as a source name: a path's file stem, a bare name unchanged."""
    return Path(entry).name.removesuffix(".bin")


@dataclass
class Provenance:
    """The provenance of one chunk-shuffled stream."""

    sources: list[str]
    chunks_per_source: list[int]
    seq_len: int
    seed: int
    total_tokens: int = 0
    tokenizer: str | None = None
    eos_id: int | None = None
    selection: dict[str, dict[str, Any]] = field(default_factory=dict)
    parent: dict[str, Any] | None = None
    output: str | None = None
    format: str = FORMAT

    def __post_init__(self) -> None:
        self.sources = [source_name(s) for s in self.sources]
        if len(self.sources) != len(self.chunks_per_source):
            raise ProvenanceError("sources and chunks_per_source differ in length")
        if len(set(self.sources)) != len(self.sources):
            raise ProvenanceError(f"duplicate source names in provenance: {self.sources}")
        if self.seq_len <= 0 or any(c < 0 for c in self.chunks_per_source):
            raise ProvenanceError("seq_len must be positive and chunk counts non-negative")

    @property
    def total_chunks(self) -> int:
        return int(sum(self.chunks_per_source))

    def chunk_labels(self) -> np.ndarray:
        """Source index of every output chunk, shape (total_chunks,), int32."""
        perm = np.random.RandomState(self.seed).permutation(self.total_chunks)
        labels = np.empty(self.total_chunks, dtype=np.int32)
        labels[perm] = np.repeat(np.arange(len(self.sources), dtype=np.int32), self.chunks_per_source)
        return labels

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_chunks"] = self.total_chunks
        return d

    def save(self, path: Path) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=1))
        tmp.replace(path)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Provenance":
        try:
            prov = cls(
                sources=list(d["sources"]), chunks_per_source=[int(c) for c in d["chunks_per_source"]],
                seq_len=int(d["seq_len"]), seed=int(d["seed"]), total_tokens=int(d.get("total_tokens", 0)),
                tokenizer=d.get("tokenizer"), eos_id=d.get("eos_id"), selection=d.get("selection") or {},
                parent=d.get("parent"), output=d.get("output"), format=d.get("format", FORMAT),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ProvenanceError(f"malformed chunk-shuffle provenance: {e!r}") from e
        if "total_chunks" in d and int(d["total_chunks"]) != prov.total_chunks:
            raise ProvenanceError(f"total_chunks {d['total_chunks']} != sum(chunks_per_source) {prov.total_chunks}")
        return prov


@dataclass
class StreamLabels:
    """A stream's chunk labels however they were recorded: names, per-chunk source index, chunk size."""

    sources: list[str]
    labels: np.ndarray
    seq_len: int
    provenance: Provenance | None = None

    def counts(self) -> dict[str, int]:
        c = np.bincount(self.labels[self.labels >= 0], minlength=len(self.sources))
        return {s: int(n) for s, n in zip(self.sources, c)}


def _span_labels(directory: Path) -> StreamLabels:
    manifest = json.loads((directory / "manifest.json").read_text())
    spans = json.loads((directory / "provenance.json").read_text())
    seq_len = int(manifest["seq_len"])
    names = sorted({s["source"] for s in spans})
    index = {n: i for i, n in enumerate(names)}
    total = max((int(s["global_end"]) for s in spans), default=0) // seq_len
    labels = np.full(total, -1, dtype=np.int32)
    for s in spans:
        a, b = int(s["global_start"]), int(s["global_end"])
        if a % seq_len or b % seq_len:
            raise ProvenanceError(f"span {a}..{b} is not aligned to seq_len {seq_len}")
        labels[a // seq_len:b // seq_len] = index[s["source"]]
    return StreamLabels(sources=names, labels=labels, seq_len=seq_len)


def load_labels(stream: Path, seq_len: int | None = None) -> StreamLabels | None:
    """The chunk labels of ``stream``, or None when no provenance sits beside it.

    Checks the labels against the stream's size and raises ProvenanceError when
    they disagree (a provenance file of another stream, or a truncated stream).
    """
    n_tokens = stream.stat().st_size // BYTES_PER_TOKEN
    p = provenance_path(stream)
    if p.exists():
        d = json.loads(p.read_text())
        if "chunks_per_source" not in d:
            raise ProvenanceError(f"{p} has no chunks_per_source (not a chunk-shuffle provenance)")
        prov = Provenance.from_json(d)
        out = StreamLabels(sources=prov.sources, labels=prov.chunk_labels(), seq_len=prov.seq_len, provenance=prov)
    elif (stream.parent / "provenance.json").exists() and (stream.parent / "manifest.json").exists():
        out = _span_labels(stream.parent)
    else:
        return None
    if seq_len is not None and seq_len != out.seq_len:
        raise ProvenanceError(f"provenance seq_len {out.seq_len} != requested {seq_len}")
    if len(out.labels) * out.seq_len != n_tokens:
        raise ProvenanceError(f"provenance describes {len(out.labels) * out.seq_len} tokens, "
                              f"{stream} holds {n_tokens}")
    return out
