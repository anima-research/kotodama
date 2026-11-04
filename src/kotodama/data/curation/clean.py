"""Universal document cleaning: Unicode normalization, whitespace, PII scrubbing, length.

Every source passes through ``clean_document`` before the quality filters in
``kotodama.data.curation.filters``. The per-character work runs in C: control
characters are stripped with ``str.translate`` over a lazily filled codepoint
table, so a corpus of terabytes costs one Unicode-category lookup per distinct
codepoint rather than one per character.

PII becomes lowercase bracket placeholders (``[email]``, ``[phone]``, ``[ssn]``,
``[ip]``). Angle-bracket forms would read as HTML pseudo-tags to the tag metrics
and train as markup.
"""

from __future__ import annotations

import re
import unicodedata

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (EMAIL_RE, "[email]"),
    (PHONE_RE, "[phone]"),
    (SSN_RE, "[ssn]"),
    (IP_RE, "[ip]"),
)

_WS_RUN_RE = re.compile(r"[ \t]+")
_NL_RUN_RE = re.compile(r"\n{3,}")


def scrub_pii(text: str) -> str:
    """Replace e-mail addresses, SSNs, IPv4 addresses and phone numbers with placeholders."""
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces and tabs to one space, cap blank lines at one, strip the ends."""
    text = _WS_RUN_RE.sub(" ", text)
    text = _NL_RUN_RE.sub("\n\n", text)
    return text.strip()


class _StripControl(dict[int, int | None]):
    """``str.translate`` table: a codepoint maps to None (strip) when its category is C*, else to itself.

    Newline and tab are kept; carriage return, format characters, private-use and
    unassigned codepoints are stripped. Filled on first sight of each codepoint.
    """

    def __missing__(self, cp: int) -> int | None:
        ch = chr(cp)
        verdict: int | None = cp if ch in "\n\t" or not unicodedata.category(ch).startswith("C") else None
        self[cp] = verdict
        return verdict


_STRIP_CONTROL = _StripControl()


def normalize_unicode(text: str) -> str:
    """NFC-normalize and strip control characters other than newline and tab."""
    return unicodedata.normalize("NFC", text).translate(_STRIP_CONTROL)


def clean_document(text: str, min_chars: int = 100, max_chars: int | None = None,
                   scrub: bool = True) -> str | None:
    """Clean one document; None means drop it.

    Normalizes Unicode and whitespace, applies the length bounds to the cleaned
    text (``max_chars=None`` = no upper bound), then scrubs PII when ``scrub``.
    """
    text = normalize_whitespace(normalize_unicode(text))
    if len(text) < min_chars or (max_chars is not None and len(text) > max_chars):
        return None
    return scrub_pii(text) if scrub else text
