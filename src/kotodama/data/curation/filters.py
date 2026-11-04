"""Quality filters: per-document metrics and a keep / repair / drop verdict.

Every document gets a ``FilterVerdict``. Which rules apply depends on the
source's ``SourceClass``:

- ``code``: exempt from the tag and entity rules, since ``#include <vector>``,
  templates and code fences are content, not extraction debris.
- ``html_repair``: comment-formatted sources whose every document carries a small
  HTML vocabulary (``<p>``, ``<i>``, links, entities); the markup is converted
  first and the repaired text is measured.
- ``web``: the tag rules plus Gopher-style repetition rules (duplicate lines, one
  word bigram covering much of the text) and a boilerplate check.
- ``prose``: the tag rules; no repetition rules (verse and transcripts repeat by design).
- ``structured``: registers and records whose ``[bracket]`` headers would trip the tag rules.

Every class gets the universal drop rules: U+FFFD and control-character density,
non-Latin script dominance, and a class-dependent length cap (a 300k-character
web page is a dump; a 300k-character parliamentary sitting is not).

Language ID is optional. Documents in the non-Latin midband are decided by a
caller-supplied ``langid`` function returning English confidence (for example a
fastText ``lid.176`` model, see ``fasttext_langid``); without one, only the
unconditional non-Latin threshold applies.
"""

from __future__ import annotations

import bisect
import html
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable


class SourceClass(str, Enum):
    WEB = "web"
    CODE = "code"
    PROSE = "prose"
    HTML_REPAIR = "html_repair"
    STRUCTURED = "structured"


@dataclass(frozen=True)
class FilterThresholds:
    """The filter battery's thresholds. Densities are per 10,000 characters of the analysed prefix."""

    tags_drop_per_10k: float = 20.0       # an extraction failure: markup soup
    tags_strip_per_10k: float = 5.0       # stray tags: stripped in place, document kept
    entities_repair_per_10k: float = 10.0  # entity-encoded text: unescaped
    boiler_drop_hits: int = 2             # web: cookie banners, "enable javascript", JSON-LD
    fffd_drop_per_10k: float = 20.0       # shredded OCR or decoded binary
    ctrl_drop_per_10k: float = 5.0
    nonlatin_drop_frac: float = 0.50      # share of letters outside Latin script: dropped outright
    nonlatin_langid_frac: float = 0.30    # midband: the langid function decides
    langid_en_min_conf: float = 0.60
    dup_line_frac_drop: float = 0.30      # web: share of non-blank lines that repeat an earlier line
    top_bigram_frac_drop: float = 0.20    # web: share of characters covered by the commonest word bigram
    max_chars: dict[SourceClass, int] = field(default_factory=lambda: {
        SourceClass.WEB: 300_000,
        SourceClass.CODE: 1_000_000,
        SourceClass.PROSE: 10_000_000,
        SourceClass.STRUCTURED: 10_000_000,
        SourceClass.HTML_REPAIR: 10_000_000,
    })


DEFAULT_THRESHOLDS = FilterThresholds()

ANALYZE_CHARS = 20_000  # metrics read this prefix; it bounds the cost of a huge document

HTML_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]{0,15}(?:\s[^<>]{0,200})?/?>")
HTML_ENTITY_RE = re.compile(r"&(?:[a-zA-Z]{2,10}|#\d{1,7}|#x[0-9a-fA-F]{1,6});")
BOILER_RE = re.compile(
    r"(?:javascript must be enabled|enable javascript|all rights reserved"
    r"|cookies? (?:policy|settings|are disabled)|\{\s*\"@context\""
    r"|<!\[CDATA\[|<!--|-->)",
    re.IGNORECASE,
)

# Script buckets: sorted (start, end, name) codepoint ranges, looked up by bisection.
_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x0041, 0x005A, "latin"), (0x0061, 0x007A, "latin"),
    (0x00C0, 0x024F, "latin"), (0x0370, 0x03FF, "greek"),
    (0x0400, 0x04FF, "cyrillic"), (0x0530, 0x058F, "armenian"),
    (0x0590, 0x05FF, "hebrew"), (0x0600, 0x06FF, "arabic"),
    (0x0700, 0x074F, "syriac"), (0x0750, 0x077F, "arabic"),
    (0x0900, 0x097F, "devanagari"), (0x0980, 0x0DFF, "indic_other"),
    (0x0E00, 0x0E7F, "thai"), (0x0E80, 0x0EFF, "lao"),
    (0x1000, 0x109F, "myanmar"), (0x10A0, 0x10FF, "georgian"),
    (0x1100, 0x11FF, "hangul"), (0x1E00, 0x1EFF, "latin"),
    (0x2600, 0x27BF, "emoji_symbols"),
    (0x3040, 0x30FF, "kana"), (0x3400, 0x4DBF, "cjk"),
    (0x4E00, 0x9FFF, "cjk"), (0xA000, 0xA4CF, "yi"),
    (0xAC00, 0xD7AF, "hangul"), (0xF900, 0xFAFF, "cjk"),
    (0x1F000, 0x1FAFF, "emoji_symbols"),
)
_STARTS = [r[0] for r in _RANGES]
_SCRIPT_CACHE: dict[str, str | None] = {}


def script_of(ch: str) -> str | None:
    """The script bucket of a character, or None for characters the filters do not count as letters."""
    if ch in _SCRIPT_CACHE:
        return _SCRIPT_CACHE[ch]
    cp = ord(ch)
    i = bisect.bisect_right(_STARTS, cp) - 1
    s = _RANGES[i][2] if i >= 0 and cp <= _RANGES[i][1] else None
    _SCRIPT_CACHE[ch] = s
    return s


@dataclass
class DocMetrics:
    chars: int
    tags_per_10k: float = 0.0
    entities_per_10k: float = 0.0
    boiler_hits: int = 0
    fffd_per_10k: float = 0.0
    ctrl_per_10k: float = 0.0
    nonlatin_letter_frac: float = 0.0
    dup_line_frac: float = 0.0
    top_bigram_frac: float = 0.0


def _repetition_metrics(t: str) -> tuple[float, float]:
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    dup_frac = 1.0 - len(set(lines)) / len(lines) if lines else 0.0
    words = t.split()
    if len(words) < 4:
        return dup_frac, 0.0
    (w1, w2), cnt = Counter(zip(words, words[1:])).most_common(1)[0]
    return dup_frac, cnt * (len(w1) + len(w2) + 1) / max(len(t), 1)


def doc_metrics(text: str, repetition: bool = False) -> DocMetrics:
    """Measure a document over its first ``ANALYZE_CHARS`` characters (``chars`` is the full length)."""
    t = text[:ANALYZE_CHARS]
    n = len(t)
    if n == 0:
        return DocMetrics(chars=0)
    per10k = 10_000 / n
    counts = Counter(t)
    ctrl = sum(c for ch, c in counts.items() if ord(ch) < 32 and ch not in "\n\t\r")
    letters = latin = 0
    for ch, c in counts.items():
        s = script_of(ch)
        if s is not None and s != "emoji_symbols":
            letters += c
            latin += c if s == "latin" else 0
    m = DocMetrics(
        chars=len(text),
        tags_per_10k=len(HTML_TAG_RE.findall(t)) * per10k,
        entities_per_10k=len(HTML_ENTITY_RE.findall(t)) * per10k,
        boiler_hits=len(BOILER_RE.findall(t)),
        fffd_per_10k=counts.get("�", 0) * per10k,
        ctrl_per_10k=ctrl * per10k,
        nonlatin_letter_frac=(letters - latin) / letters if letters else 0.0,
    )
    if repetition:
        m.dup_line_frac, m.top_bigram_frac = _repetition_metrics(t)
    return m


_P_RE = re.compile(r"<p>", re.IGNORECASE)
_INLINE_RE = re.compile(r"</?(?:i|b|em|strong|pre|code|a)(?:\s[^<>]{0,200})?>", re.IGNORECASE)


def repair_html(text: str) -> str:
    """Convert ``<p>`` to paragraph breaks, strip inline tags and links, then unescape entities.

    Tags go first, while the text is still entity-encoded, so an escaped ``&lt;b&gt;``
    written by the author survives as literal text.
    """
    return html.unescape(_INLINE_RE.sub("", _P_RE.sub("\n\n", text)))


def strip_tags(text: str) -> str:
    """Replace each tag span with a space, keeping the text around it."""
    return HTML_TAG_RE.sub(" ", text)


class Action(str, Enum):
    KEEP = "keep"
    REPAIR = "repair"   # kept, with the text transformed
    DROP = "drop"


@dataclass
class FilterVerdict:
    action: Action
    metrics: DocMetrics
    text: str | None = None          # None exactly when the action is DROP
    reasons: list[str] = field(default_factory=list)


LangId = Callable[[str], "float | None"]


def _drop(reason: str, m: DocMetrics) -> FilterVerdict:
    return FilterVerdict(action=Action.DROP, metrics=m, reasons=[reason])


def apply_filters(text: str, source_class: SourceClass,
                  thresholds: FilterThresholds = DEFAULT_THRESHOLDS,
                  langid: LangId | None = None) -> FilterVerdict:
    """Run the battery on one document; the verdict carries the (possibly repaired) text and the rules that fired."""
    th = thresholds
    reasons: list[str] = []
    if source_class is SourceClass.HTML_REPAIR:
        repaired = repair_html(text)
        if repaired != text:
            reasons.append("html_repair")
        text = repaired

    repetition = source_class is SourceClass.WEB
    m = doc_metrics(text, repetition=repetition)
    if m.chars == 0:
        return _drop("empty", m)
    if m.chars > th.max_chars[source_class]:
        return _drop("max_chars", m)
    if m.fffd_per_10k >= th.fffd_drop_per_10k:
        return _drop("fffd", m)
    if m.ctrl_per_10k >= th.ctrl_drop_per_10k:
        return _drop("ctrl", m)
    if m.nonlatin_letter_frac >= th.nonlatin_drop_frac:
        return _drop("nonlatin", m)
    if m.nonlatin_letter_frac >= th.nonlatin_langid_frac and langid is not None:
        conf = langid(text)
        if conf is not None and conf < th.langid_en_min_conf:
            return _drop("nonlatin_langid", m)

    if source_class in (SourceClass.WEB, SourceClass.PROSE):
        if m.tags_per_10k >= th.tags_drop_per_10k or (
                source_class is SourceClass.WEB and m.boiler_hits >= th.boiler_drop_hits):
            return _drop("html_soup", m)
        if m.tags_per_10k >= th.tags_strip_per_10k:
            text = strip_tags(text)
            reasons.append("tags_stripped")
        if m.entities_per_10k >= th.entities_repair_per_10k:
            text = html.unescape(text)
            reasons.append("entities_unescaped")

    if repetition and (m.dup_line_frac >= th.dup_line_frac_drop
                       or m.top_bigram_frac >= th.top_bigram_frac_drop):
        return _drop("repetition", m)

    if "�" in text:
        text = text.replace("�", "")
        reasons.append("fffd_scrubbed")

    return FilterVerdict(action=Action.REPAIR if reasons else Action.KEEP, metrics=m, text=text,
                         reasons=reasons)


def fasttext_langid(model_path: str | Path) -> LangId:
    """A ``langid`` function backed by a fastText language-ID model (e.g. ``lid.176.bin``).

    Needs the ``fasttext`` package, which is not a dependency of this repository;
    the model file is at https://fasttext.cc/docs/en/language-identification.html.
    Raises ImportError or FileNotFoundError up front rather than on the first document.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"fastText language-ID model not found: {path}")
    import fasttext  # type: ignore[import-not-found]

    model = fasttext.load_model(str(path))

    def en_confidence(text: str) -> float:
        labels, probs = model.predict(text[:2000].replace("\n", " "), k=1)
        return float(probs[0]) if labels and labels[0] == "__label__en" else 0.0

    return en_confidence
