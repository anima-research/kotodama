"""Cleaning and the quality-filter battery (kotodama.data.curation.clean / .filters)."""

from __future__ import annotations

from kotodama.data.curation.clean import (
    clean_document, normalize_unicode, normalize_whitespace, scrub_pii,
)
from kotodama.data.curation.filters import (
    Action, FilterThresholds, SourceClass, apply_filters, doc_metrics, repair_html, script_of,
)

PROSE = ("The river ran quietly past the old mill, where the miller kept his ledgers in a "
         "wooden chest and counted sacks of grain each evening before supper. ") * 4


def test_normalize_unicode_strips_controls_keeps_newline_and_tab():
    s = "a\u0000b\tc\nd\re​fg"
    assert normalize_unicode(s) == "ab\tc\ndefg"
    # NFC: e + combining acute -> one codepoint
    assert normalize_unicode("é") == "é"


def test_normalize_whitespace():
    assert normalize_whitespace("  a \t  b\n\n\n\nc  ") == "a b\n\nc"


def test_scrub_pii_placeholders():
    s = "mail me at jo.doe+x@example.org or 555-123-4567; ssn 123-45-6789 from 10.0.0.12"
    out = scrub_pii(s)
    assert out == "mail me at [email] or [phone]; ssn [ssn] from [ip]"


def test_clean_document_bounds_and_scrub_flag():
    assert clean_document("short", min_chars=10) is None
    assert clean_document("x" * 50, min_chars=10, max_chars=20) is None
    text = "contact: a@b.co " + "word " * 30
    assert "[email]" in clean_document(text, min_chars=10)
    assert "a@b.co" in clean_document(text, min_chars=10, scrub=False)


def test_script_buckets_are_ordered_lookups():
    assert script_of("a") == "latin"
    assert script_of("Ж") == "cyrillic"
    assert script_of("漢") == "cjk"
    assert script_of("☀") == "emoji_symbols"
    assert script_of("1") is None


def test_clean_prose_is_kept_untouched():
    v = apply_filters(PROSE, SourceClass.PROSE)
    assert v.action is Action.KEEP and v.text == PROSE and v.reasons == []


def test_html_soup_dropped_for_web_and_prose_but_code_exempt():
    soup = "<div><span>x</span><b>y</b></div> " * 200
    assert apply_filters(soup, SourceClass.WEB).reasons == ["html_soup"]
    assert apply_filters(soup, SourceClass.PROSE).action is Action.DROP
    assert apply_filters(soup, SourceClass.CODE).action is Action.KEEP


def test_stray_tags_stripped_and_entities_unescaped():
    text = PROSE * 6 + " <b>bold</b> <i>it</i> &amp; &quot;q&quot; " + "&lt;x&gt; " * 3
    m = doc_metrics(text)
    assert FilterThresholds().tags_strip_per_10k <= m.tags_per_10k < FilterThresholds().tags_drop_per_10k
    v = apply_filters(text, SourceClass.PROSE)
    assert v.action is Action.REPAIR
    assert set(v.reasons) == {"tags_stripped", "entities_unescaped"}
    assert "<b>" not in v.text and '"q"' in v.text


def test_html_repair_class_converts_comment_markup():
    raw = "First point.<p>Second &amp; <i>third</i> <a href=\"https://x\">link</a>."
    assert repair_html(raw) == "First point.\n\nSecond & third link."
    v = apply_filters(raw + " " + PROSE, SourceClass.HTML_REPAIR)
    assert "html_repair" in v.reasons and "<p>" not in v.text


def test_nonlatin_drop_and_langid_midband():
    cyr = "Привет мир это тест " * 20
    assert apply_filters(cyr, SourceClass.PROSE).reasons == ["nonlatin"]
    mixed = "hello world this is " * 12 + "Привет мир это " * 8
    frac = doc_metrics(mixed).nonlatin_letter_frac
    assert 0.30 <= frac < 0.50
    assert apply_filters(mixed, SourceClass.PROSE).action is not Action.DROP   # no langid: midband passes
    assert apply_filters(mixed, SourceClass.PROSE, langid=lambda t: 0.1).reasons == ["nonlatin_langid"]
    assert apply_filters(mixed, SourceClass.PROSE, langid=lambda t: 0.9).action is not Action.DROP


def test_repetition_rules_only_for_web():
    rep = "\n".join(["Buy cheap widgets now"] * 40 + ["unique line %d here" % i for i in range(10)])
    assert apply_filters(rep, SourceClass.WEB).reasons == ["repetition"]
    assert apply_filters(rep, SourceClass.PROSE).action is Action.KEEP


def test_fffd_scrub_and_drop_and_class_length_caps():
    few = PROSE + "�"
    v = apply_filters(few, SourceClass.PROSE)
    assert v.reasons == ["fffd_scrubbed"] and "�" not in v.text
    many = "ab�" * 500
    assert apply_filters(many, SourceClass.PROSE).reasons == ["fffd"]
    long_text = "word " * 70_000   # 350k characters
    assert apply_filters(long_text, SourceClass.WEB).reasons == ["max_chars"]
    assert apply_filters(long_text, SourceClass.PROSE).action is not Action.DROP


def test_empty_is_dropped():
    assert apply_filters("", SourceClass.PROSE).reasons == ["empty"]
