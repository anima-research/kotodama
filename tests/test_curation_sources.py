"""Source specs, row formatters and adapters (kotodama.data.curation.sources)."""

from __future__ import annotations

import gzip
import json
import sys
import types

import pytest

from kotodama.data.curation import sources as S
from kotodama.data.curation.filters import SourceClass
from kotodama.data.curation.sources import RawDoc, SourceError, SourceSpec, format_row, iter_docs


def test_spec_validation():
    with pytest.raises(ValueError):
        SourceSpec("bad.name", dataset="x")
    with pytest.raises(SourceError):
        SourceSpec("s", dataset="x", format="nope")
    with pytest.raises(SourceError):
        SourceSpec("s", dataset="x", format="template")
    with pytest.raises(SourceError):
        SourceSpec("s")   # hf adapter without a dataset
    with pytest.raises(SourceError):
        SourceSpec("s", dataset="x", source_class="nope")
    with pytest.raises(SourceError):
        SourceSpec.from_dict({"name": "s", "dataset": "x", "typo_key": 1})


def test_from_dict_normalizes_yaml_shapes():
    spec = SourceSpec.from_dict({"name": "s", "dataset": "x", "configs": "en", "fields": "title",
                                 "format": "fields", "source_class": "web"})
    assert spec.configs == ("en",) and spec.fields == ("title",) and spec.source_class is SourceClass.WEB
    assert SourceSpec("s", dataset="x", configs=None).configs == (None,)
    assert spec.to_dict()["source_class"] == "web"
    assert spec.with_overrides(min_chars=5, split=None).min_chars == 5


def test_every_preset_is_a_valid_spec():
    assert {"smoltalk", "soda", "oasst", "gutenberg", "us_patents", "writingprompts"} <= set(S.PRESETS)
    for name, spec in S.PRESETS.items():
        assert spec.name == name
        SourceSpec.from_dict(spec.to_dict())   # round-trips through the YAML shape


def test_format_text_fields_template():
    spec = SourceSpec("s", dataset="x")
    assert format_row({"text": "  hi  "}, spec) == "hi"
    assert format_row({"text": None}, spec) is None
    f = SourceSpec("s", dataset="x", format="fields", fields=("title", "abstract", "body"))
    assert format_row({"title": "T", "abstract": "", "body": "B"}, f) == "T\n\nB"
    t = SourceSpec("s", dataset="x", format="template", template="[WP] {prompt}\n\n{story}")
    assert format_row({"prompt": "P", "story": "S"}, t) == "[WP] P\n\nS"
    assert format_row({"story": "S"}, t) is None


def test_format_messages_and_dialogue():
    m = SourceSpec("s", dataset="x", format="messages", text_field="messages", min_turns=2)
    row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
    assert format_row(row, m) == "[user] hi\n\n[assistant] hello"
    assert format_row({"messages": row["messages"][:1]}, m) is None
    d = SourceSpec("s", dataset="x", format="dialogue", min_turns=2)
    out = format_row({"narrative": "At a cafe.", "dialogue": ["Hi", "Hey", "Bye"], "speakers": ["A", "B"]}, d)
    assert out == "[context] At a cafe.\n\n[A] Hi\n\n[B] Hey\n\n[unknown] Bye"


def test_strip_gutenberg():
    text = ("Header junk\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nChapter 1. It begins.\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK X ***\nlicence text")
    assert S.strip_gutenberg(text) == "Chapter 1. It begins."


def test_local_adapter_reads_every_file_kind(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    root = tmp_path / "src"
    root.mkdir()
    (root / "a.jsonl").write_text(json.dumps({"text": "alpha", "score": 3}) + "\nnot json\n"
                                  + json.dumps({"text": "low", "score": 0}) + "\n")
    with gzip.open(root / "b.jsonl.gz", "wt") as f:
        f.write(json.dumps({"text": "beta", "score": 5}) + "\n")
    pq.write_table(pa.table({"text": ["gamma"], "score": [9]}), root / "c.parquet")
    (root / "d.txt").write_text("delta from a text file")
    spec = SourceSpec("loc", adapter="local", dataset=str(root), row_min={"score": 1}, meta_fields=("score",))
    docs = list(iter_docs(spec))
    assert [d.text for d in docs] == ["alpha", "beta", "gamma", "delta from a text file"]
    assert docs[0].meta["score"] == 3 and "doc_id" in docs[0].meta
    assert len({d.meta["doc_id"] for d in docs}) == len(docs)
    with pytest.raises(FileNotFoundError):
        list(iter_docs(SourceSpec("loc", adapter="local", dataset=str(tmp_path / "missing"))))


def test_group_by_joins_lines(tmp_path):
    rows = [{"line": "Roses are red", "gid": 1}, {"line": "Violets are blue", "gid": 1}, {"line": "Solo", "gid": 2}]
    (tmp_path / "p.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    spec = SourceSpec("poems", adapter="local", dataset=str(tmp_path), text_field="line", group_by="gid")
    docs = list(iter_docs(spec))
    assert [d.text for d in docs] == ["Roses are red\nViolets are blue", "Solo"]
    assert [d.meta["subset"] for d in docs] == ["1", "2"]


def test_oasst_threads_flatten_every_leaf():
    msgs = [
        {"message_tree_id": "t", "message_id": "r", "parent_id": None, "role": "prompter", "text": "Q"},
        {"message_tree_id": "t", "message_id": "a1", "parent_id": "r", "role": "assistant", "text": "A1"},
        {"message_tree_id": "t", "message_id": "a2", "parent_id": "r", "role": "assistant", "text": "A2"},
        {"message_tree_id": "t", "message_id": "f", "parent_id": "a1", "role": "prompter", "text": "F"},
    ]
    paths = [[m["message_id"] for m in p] for _, _, p in S.oasst_threads(msgs)]
    assert paths == [["r", "a1", "f"], ["r", "a2"]]


def test_oasst_adapter_filters_and_renders(monkeypatch):
    rows = [
        {"message_tree_id": "t1", "message_id": "r", "parent_id": None, "role": "prompter", "text": "Q",
         "lang": "en"},
        {"message_tree_id": "t1", "message_id": "a", "parent_id": "r", "role": "assistant", "text": "A",
         "lang": "en"},
        {"message_tree_id": "t2", "message_id": "x", "parent_id": None, "role": "prompter", "text": "Q2",
         "lang": "de"},
    ]
    monkeypatch.setattr(S, "hf_rows", lambda spec: ((None, r) for r in rows))
    docs = list(iter_docs(S.PRESETS["oasst"]))
    assert [d.text for d in docs] == ["[prompter] Q\n\n[assistant] A"]


def test_custom_adapter_by_import_path(monkeypatch):
    mod = types.ModuleType("site_local_adapter")

    def read(spec: SourceSpec):
        for i in range(spec.options["n"]):
            yield RawDoc(f"doc {i}", {"doc_id": str(i)})

    mod.read = read
    mod.bad = lambda spec: iter(["not a RawDoc"])
    monkeypatch.setitem(sys.modules, "site_local_adapter", mod)
    spec = SourceSpec("site", adapter="site_local_adapter:read", options={"n": 3})
    assert [d.text for d in iter_docs(spec)] == ["doc 0", "doc 1", "doc 2"]
    with pytest.raises(SourceError):
        list(iter_docs(SourceSpec("site", adapter="site_local_adapter:bad")))
    with pytest.raises(SourceError):
        list(iter_docs(SourceSpec("site", adapter="no_such_module_xyz:read")))
    with pytest.raises(SourceError):
        list(iter_docs(SourceSpec("site", adapter="unregistered")))
