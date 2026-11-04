"""Where documents come from: source specs, adapters, row formatters and public presets.

A ``SourceSpec`` names a source and says how to read it. Its ``adapter`` turns
the spec into a stream of ``RawDoc`` (text plus metadata) that
``kotodama.data.curation.ingest`` cleans, filters and writes as chunks:

- ``hf``: a Hugging Face dataset (``datasets.load_dataset``), one or more configs.
- ``local``: a file or directory of ``.jsonl`` / ``.jsonl.gz`` / ``.jsonl.zst`` /
  ``.parquet`` rows, or ``.txt`` / ``.md`` files (one document per file).
- ``oasst``: OpenAssistant message tables, flattened from reply trees into one
  document per root-to-leaf thread.
- ``module:function``: any importable callable taking the spec and yielding
  ``RawDoc``. A site-local source (an export, a crawl, an internal archive) plugs
  in here without touching this package; the spec's other fields reach it, and
  ``options`` carries anything source-specific.

For ``hf`` and ``local``, ``format`` picks how a row becomes text:

- ``text``: the ``text_field`` string.
- ``fields``: the non-empty values of ``fields``, joined by blank lines.
- ``template``: ``template.format_map(row)``; a row missing a field is skipped.
- ``messages``: a list of ``{role, content}`` turns in ``text_field``, each
  rendered with ``turn_template``.
- ``dialogue``: parallel ``dialogue`` / ``speakers`` lists with an optional
  ``narrative`` preamble.
- ``gutenberg``: ``text_field`` with Project Gutenberg headers, footers and
  licence blocks removed.

``group_by`` concatenates the formatted text of all rows sharing a field value
(line-per-row corpora); it holds the source in memory, so it suits small sources.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from kotodama.data.curation.chunks import check_source_name, iter_lines
from kotodama.data.curation.filters import SourceClass

logger = logging.getLogger(__name__)

FORMATS = ("text", "fields", "template", "messages", "dialogue", "gutenberg")
LOCAL_SUFFIXES = (".jsonl", ".jsonl.gz", ".jsonl.zst", ".json.gz", ".parquet", ".txt", ".md")


class SourceError(ValueError):
    """A source spec is malformed or names an adapter that does not exist."""


@dataclass(frozen=True)
class RawDoc:
    """A document as an adapter yields it: raw text and the metadata to carry into its chunk."""

    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceSpec:
    """How to read one source. Field defaults suit a plain text-field dataset."""

    name: str
    adapter: str = "hf"
    dataset: str | None = None            # HF repo id or builder name (hf); a path (local)
    configs: tuple[str | None, ...] = (None,)  # HF config names, read in order; several -> meta.subset
    split: str = "train"
    streaming: bool = True
    data_files: str | None = None         # HF data_files pattern (e.g. "hf://datasets/<repo>/<glob>")
    format: str = "text"
    text_field: str = "text"
    fields: tuple[str, ...] = ()          # format "fields"
    template: str | None = None           # format "template"
    turn_template: str = "[{role}] {content}"  # format "messages"
    min_turns: int = 0                    # formats "messages" / "dialogue"
    row_min: Mapping[str, float] = field(default_factory=dict)    # skip rows whose numeric field is below
    row_equals: Mapping[str, str] = field(default_factory=dict)   # keep only rows whose field equals
    subset_field: str | None = None       # row field recorded as meta.subset
    meta_fields: tuple[str, ...] = ()     # row fields copied into meta
    group_by: str | None = None
    source_class: SourceClass = SourceClass.PROSE
    min_chars: int = 100
    max_chars: int | None = None
    scrub_pii: bool = True
    max_docs: int | None = None           # stop after this many kept documents
    options: Mapping[str, Any] = field(default_factory=dict)  # free-form, for custom adapters

    def __post_init__(self) -> None:
        # YAML gives a bare string where a one-element list is meant; normalize before validating.
        if isinstance(self.configs, str) or self.configs is None:
            object.__setattr__(self, "configs", (self.configs,))
        for name in ("configs", "fields", "meta_fields"):
            value = getattr(self, name)
            object.__setattr__(self, name, (value,) if isinstance(value, str) else tuple(value or ()))
        if not self.configs:
            object.__setattr__(self, "configs", (None,))
        if not isinstance(self.source_class, SourceClass):
            try:
                object.__setattr__(self, "source_class", SourceClass(self.source_class))
            except ValueError as e:
                raise SourceError(f"{self.name}: source_class must be one of "
                                  f"{[c.value for c in SourceClass]}") from e
        check_source_name(self.name)
        if self.format not in FORMATS:
            raise SourceError(f"{self.name}: format {self.format!r} is not one of {FORMATS}")
        if self.format == "template" and not self.template:
            raise SourceError(f"{self.name}: format 'template' needs a template")
        if self.format == "fields" and not self.fields:
            raise SourceError(f"{self.name}: format 'fields' needs fields")
        if self.adapter in ("hf", "local") and not (self.dataset or self.data_files):
            raise SourceError(f"{self.name}: adapter {self.adapter!r} needs dataset (or data_files)")
        if self.min_chars < 0 or (self.max_chars is not None and self.max_chars < self.min_chars):
            raise SourceError(f"{self.name}: need 0 <= min_chars <= max_chars")
        if self.max_docs is not None and self.max_docs <= 0:
            raise SourceError(f"{self.name}: max_docs must be positive")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceSpec":
        """Build a spec from a mapping (a YAML or JSON source file), refusing unknown keys."""
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise SourceError(f"unknown source spec keys: {sorted(unknown)} (known: {sorted(known)})")
        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["source_class"] = self.source_class.value
        return d

    def with_overrides(self, **overrides: Any) -> "SourceSpec":
        """A copy with every non-None override applied."""
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})


# ---------------------------------------------------------------------------
# row -> text
# ---------------------------------------------------------------------------

_GUTENBERG_HEADERS = (
    re.compile(r"\*\*\*\s*START OF (?:THE |THIS )?PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE | re.DOTALL),
    re.compile(r"Produced by .*?(?:\n\n|\r\n\r\n)", re.DOTALL),
)
_GUTENBERG_FOOTERS = (
    re.compile(r"\*\*\*\s*END OF (?:THE |THIS )?PROJECT GUTENBERG.*", re.IGNORECASE | re.DOTALL),
    re.compile(r"End of (?:the )?Project Gutenberg.*", re.IGNORECASE | re.DOTALL),
)
_GUTENBERG_LICENSE = re.compile(
    r"(?:This (?:eBook|ebook) is for the use of anyone.*?"
    r"(?:Project Gutenberg Literary Archive Foundation|www\.gutenberg\.org).*?(?:\n\n|\Z))",
    re.IGNORECASE | re.DOTALL,
)


def strip_gutenberg(text: str) -> str:
    """Remove a Project Gutenberg header (up to the START marker), footer and licence block."""
    for pattern in _GUTENBERG_HEADERS:
        m = pattern.search(text)
        if m:
            text = text[m.end():]
            break
    for pattern in _GUTENBERG_FOOTERS:
        m = pattern.search(text)
        if m:
            text = text[:m.start()]
            break
    return _GUTENBERG_LICENSE.sub("", text).strip()


def _str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def format_row(row: Mapping[str, Any], spec: SourceSpec) -> str | None:
    """The text of one row under ``spec.format``; None when the row has nothing usable."""
    fmt = spec.format
    if fmt == "text":
        return _str(row.get(spec.text_field)) or None
    if fmt == "gutenberg":
        return strip_gutenberg(_str(row.get(spec.text_field))) or None
    if fmt == "fields":
        parts = [_str(row.get(f)) for f in spec.fields]
        return "\n\n".join(p for p in parts if p) or None
    if fmt == "template":
        try:
            text = spec.template.format_map(row) if spec.template else ""
        except (KeyError, IndexError, ValueError):
            return None
        return text.strip() or None
    if fmt == "messages":
        turns = row.get(spec.text_field)
        if not isinstance(turns, list):
            return None
        parts = [spec.turn_template.format(role=t.get("role", "unknown"), content=_str(t.get("content")))
                 for t in turns if isinstance(t, Mapping) and _str(t.get("content"))]
        return "\n\n".join(parts) if parts and len(parts) >= spec.min_turns else None
    if fmt == "dialogue":
        dialogue = row.get("dialogue") or []
        speakers = list(row.get("speakers") or [])
        if not isinstance(dialogue, list) or len(dialogue) < max(spec.min_turns, 1):
            return None
        speakers += ["unknown"] * (len(dialogue) - len(speakers))
        parts = [f"[context] {_str(row.get('narrative'))}"] if _str(row.get("narrative")) else []
        parts += [f"[{s}] {_str(t)}" for t, s in zip(dialogue, speakers) if _str(t)]
        return "\n\n".join(parts)
    raise SourceError(f"unknown format {fmt!r}")


def row_passes(row: Mapping[str, Any], spec: SourceSpec) -> bool:
    """Apply ``row_min`` and ``row_equals``; a row missing a ``row_min`` field passes that test."""
    for key, floor in spec.row_min.items():
        value = row.get(key)
        if isinstance(value, (int, float)) and value < floor:
            return False
    return all(str(row.get(k)) == str(v) for k, v in spec.row_equals.items())


def doc_id(source: str, subset: str | None, index: int) -> str:
    """A stable 16-hex-digit id from the source, subset and row index."""
    return hashlib.sha256(f"{source}:{subset or ''}:{index}".encode()).hexdigest()[:16]


def _row_docs(rows: Iterable[tuple[str | None, Mapping[str, Any]]], spec: SourceSpec) -> Iterator[RawDoc]:
    """Turn (subset, row) pairs into documents: row filters, formatting, grouping, metadata."""
    if spec.group_by:
        groups: dict[str, list[str]] = defaultdict(list)
        for _, row in rows:
            text = format_row(row, spec) if row_passes(row, spec) else None
            if text:
                groups[str(row.get(spec.group_by))].append(text)
        for i, (key, lines) in enumerate(sorted(groups.items())):
            yield RawDoc("\n".join(lines), {"doc_id": doc_id(spec.name, key, i), "subset": key})
        return
    for i, (subset, row) in enumerate(rows):
        if not row_passes(row, spec):
            continue
        text = format_row(row, spec)
        if not text:
            continue
        sub = str(row[spec.subset_field]) if spec.subset_field and row.get(spec.subset_field) is not None \
            else subset
        meta: dict[str, Any] = {"doc_id": doc_id(spec.name, sub, i)}
        if sub:
            meta["subset"] = sub
        for f in spec.meta_fields:
            if row.get(f) is not None:
                meta[f] = row[f]
        yield RawDoc(text, meta)


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------

Adapter = Callable[[SourceSpec], Iterator[RawDoc]]


def hf_rows(spec: SourceSpec) -> Iterator[tuple[str | None, Mapping[str, Any]]]:
    """(subset, row) pairs from ``datasets.load_dataset``, config by config."""
    from datasets import load_dataset

    many = len(spec.configs) > 1
    for config in spec.configs:
        kwargs: dict[str, Any] = {"path": spec.dataset or "json", "split": spec.split,
                                  "streaming": spec.streaming}
        if config is not None:
            kwargs["name"] = config
        if spec.data_files is not None:
            kwargs["data_files"] = spec.data_files
        logger.info("%s: loading %s (config=%s, split=%s)", spec.name, kwargs["path"], config, spec.split)
        ds = load_dataset(**kwargs)
        for row in ds:
            yield (config if many else None), row


def _local_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"local source not found: {root}")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name.endswith(LOCAL_SUFFIXES))
    if not files:
        raise FileNotFoundError(f"no {LOCAL_SUFFIXES} files under {root}")
    return files


def _json_rows(lines: Iterable[bytes | str], path: Path) -> Iterator[Mapping[str, Any]]:
    bad = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            bad += 1
            continue
        if isinstance(obj, dict):
            yield obj
        else:
            bad += 1
    if bad:
        logger.warning("%s: skipped %d unparseable lines", path, bad)


def local_rows(spec: SourceSpec) -> Iterator[tuple[str | None, Mapping[str, Any]]]:
    """(subset, row) pairs from local files; a text file is one row ``{text_field: content, "path": ...}``."""
    root = Path(spec.dataset or "")
    for path in _local_files(root):
        name = path.name
        rel = str(path.relative_to(root)) if root.is_dir() else name
        if name.endswith((".jsonl", ".jsonl.zst")):
            yield from ((None, r) for r in _json_rows(iter_lines(path), path))
        elif name.endswith((".jsonl.gz", ".json.gz")):
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                yield from ((None, r) for r in _json_rows(f, path))
        elif name.endswith(".parquet"):
            import pyarrow.parquet as pq

            for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
                yield from ((None, r) for r in batch.to_pylist())
        else:
            yield None, {spec.text_field: path.read_text(encoding="utf-8", errors="replace"), "path": rel}


def hf_adapter(spec: SourceSpec) -> Iterator[RawDoc]:
    return _row_docs(hf_rows(spec), spec)


def local_adapter(spec: SourceSpec) -> Iterator[RawDoc]:
    return _row_docs(local_rows(spec), spec)


def oasst_threads(messages: Iterable[Mapping[str, Any]]) -> Iterator[tuple[str, int, list[Mapping[str, Any]]]]:
    """Yield (tree id, branch index, root-to-leaf message path) for every leaf of every message tree."""
    trees: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for m in messages:
        trees[str(m.get("message_tree_id"))].append(m)
    for tree_id, msgs in trees.items():
        children: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
        for m in msgs:
            children[m.get("parent_id")].append(m)
        branch = 0
        stack: list[tuple[Mapping[str, Any], list[Mapping[str, Any]]]] = [
            (r, []) for r in reversed(children.get(None, []))]
        while stack:
            msg, path = stack.pop()
            path = [*path, msg]
            kids = children.get(msg.get("message_id"), [])
            if not kids:
                yield tree_id, branch, path
                branch += 1
            stack.extend((k, path) for k in reversed(kids))


def oasst_adapter(spec: SourceSpec) -> Iterator[RawDoc]:
    """OpenAssistant conversations: one document per thread, turns rendered with ``turn_template``.

    ``row_equals`` filters messages before the trees are built (``{"lang": "en"}``
    keeps one language); ``min_turns`` drops short threads.
    """
    messages = [row for _, row in hf_rows(spec) if row_passes(row, spec)]
    for tree_id, branch, path in oasst_threads(messages):
        if len(path) < spec.min_turns:
            continue
        text = "\n\n".join(spec.turn_template.format(role=m.get("role", "unknown"), content=_str(m.get("text")))
                           for m in path)
        yield RawDoc(text, {"doc_id": doc_id(spec.name, tree_id, branch), "subset": f"tree_{tree_id[:8]}"})


ADAPTERS: dict[str, Adapter] = {"hf": hf_adapter, "local": local_adapter, "oasst": oasst_adapter}


def resolve_adapter(name: str) -> Adapter:
    """A registered adapter, or ``module:function`` imported on demand."""
    if name in ADAPTERS:
        return ADAPTERS[name]
    if ":" not in name:
        raise SourceError(f"adapter {name!r} is neither registered ({sorted(ADAPTERS)}) nor module:function")
    module, _, attr = name.partition(":")
    try:
        fn = getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError) as e:
        raise SourceError(f"cannot import adapter {name!r}: {e}") from e
    if not callable(fn):
        raise SourceError(f"adapter {name!r} is not callable")
    return fn


def iter_docs(spec: SourceSpec) -> Iterator[RawDoc]:
    """The raw documents of a source, through its adapter."""
    for doc in resolve_adapter(spec.adapter)(spec):
        if not isinstance(doc, RawDoc):
            raise SourceError(f"adapter {spec.adapter!r} yielded {type(doc).__name__}, expected RawDoc")
        yield doc


# ---------------------------------------------------------------------------
# public presets
# ---------------------------------------------------------------------------

_SE_SITES = (
    "stackoverflow.com", "math.stackexchange.com", "stats.stackexchange.com",
    "physics.stackexchange.com", "serverfault.com", "superuser.com", "askubuntu.com",
    "unix.stackexchange.com", "cs.stackexchange.com", "datascience.stackexchange.com",
    "softwareengineering.stackexchange.com", "electronics.stackexchange.com",
    "english.stackexchange.com", "chemistry.stackexchange.com", "biology.stackexchange.com",
    "philosophy.stackexchange.com", "history.stackexchange.com", "law.stackexchange.com",
    "economics.stackexchange.com", "linguistics.stackexchange.com", "writers.stackexchange.com",
    "worldbuilding.stackexchange.com",
)

PRESETS: dict[str, SourceSpec] = {s.name: s for s in (
    SourceSpec("fineweb_edu", dataset="HuggingFaceFW/fineweb-edu", configs=("sample-10BT",),
               source_class=SourceClass.WEB),
    SourceSpec("wikipedia", dataset="wikimedia/wikipedia", configs=("20231101.en",)),
    SourceSpec("pes2o", dataset="json", data_files="hf://datasets/allenai/peS2o/data/v2/train-*.json.gz"),
    SourceSpec("openwebmath", dataset="open-web-math/open-web-math", source_class=SourceClass.WEB),
    SourceSpec("finemath", dataset="HuggingFaceTB/finemath", configs=("finemath-4plus",),
               source_class=SourceClass.WEB),
    SourceSpec("smoltalk", dataset="HuggingFaceTB/smoltalk", configs=("all",), format="messages",
               text_field="messages", min_turns=2, min_chars=200, scrub_pii=False, subset_field="source"),
    SourceSpec("soda", dataset="allenai/soda", format="dialogue", min_turns=3, min_chars=200, scrub_pii=False),
    SourceSpec("oasst", adapter="oasst", dataset="OpenAssistant/oasst2", split="train+validation",
               streaming=False, row_equals={"lang": "en"}, min_turns=2, min_chars=200),
    SourceSpec("stackexchange", dataset="HuggingFaceTB/stackexchange_2025_md", configs=_SE_SITES,
               text_field="ThreadText", row_min={"Score": 1}, min_chars=300, source_class=SourceClass.CODE),
    SourceSpec("gutenberg", dataset="common-pile/project_gutenberg", format="gutenberg", min_chars=500,
               scrub_pii=False),
    SourceSpec("us_patents", dataset="allenai/us-patents", format="fields",
               fields=("title", "abstract", "description"), min_chars=200),
    SourceSpec("writingprompts", dataset="euclaise/writingprompts", format="template",
               template="[WP] {prompt}\n\n{story}", min_chars=200),
    SourceSpec("gutenberg_poetry", dataset="biglam/gutenberg-poetry-corpus", streaming=False,
               text_field="line", group_by="gutenberg_id", min_chars=200),
)}
