"""Ingest: adapter -> clean -> filter -> chunked JSONL (kotodama.data.curation.ingest, .chunks)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kotodama.data.curation import ingest as I
from kotodama.data.curation.chunks import ChunkError, ChunkWriter, discover_chunks, iter_records, source_of
from kotodama.data.curation.sources import SourceError, SourceSpec

GOOD = "A plain paragraph about the weather in the hills, written for testing the ingest stage. "


def _write_source(path: Path, n_good: int = 10) -> None:
    rows = [{"text": GOOD * 3 + f"Number {i}.", "id": i} for i in range(n_good)]
    rows += [{"text": "too short"}, {"text": "<div><b>x</b></div> " * 100}, {"nope": 1}]
    rows.append({"text": GOOD * 3 + "write to me@example.com"})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _spec(src: Path, **kw) -> SourceSpec:
    return SourceSpec("demo", adapter="local", dataset=str(src), meta_fields=("id",), **kw)


def test_chunk_writer_is_atomic_and_discoverable(tmp_path):
    with ChunkWriter(tmp_path / "s.0000.jsonl") as w:
        w.write("hello", {"k": 1})
    with pytest.raises(RuntimeError):
        with ChunkWriter(tmp_path / "s.0001.jsonl") as w:
            w.write("half", {})
            raise RuntimeError("boom")
    assert not (tmp_path / "s.0001.jsonl").exists() and not list(tmp_path.glob("*.tmp"))
    assert {k: [p.name for p in v] for k, v in discover_chunks(tmp_path).items()} == {"s": ["s.0000.jsonl"]}
    assert [(r.text, r.meta) for r in iter_records(tmp_path / "s.0000.jsonl")] == [("hello", {"k": 1})]
    assert source_of(tmp_path / "s.0000.jsonl") == "s"
    with pytest.raises(ChunkError):
        source_of(tmp_path / "notachunk.txt")


def test_ingest_cleans_filters_and_accounts(tmp_path):
    src = tmp_path / "raw.jsonl"
    _write_source(src)
    out = tmp_path / "cleaned"
    st = I.ingest(_spec(src), out, workers=0, docs_per_chunk=4)
    # 14 rows; one has no text, so the adapter yields 13 documents
    assert st.docs_in == 13
    assert st.kept + st.repaired == 11 and st.dropped == 2
    assert st.drop_reasons == {"length": 1, "html_soup": 1}
    chunks = discover_chunks(out)["demo"]
    assert [p.name for p in chunks] == [f"demo.{i:04d}.jsonl" for i in range(4)]
    recs = [r for c in chunks for r in iter_records(c)]
    assert len(recs) == 11
    assert all(r.meta["source"] == "demo" and "doc_id" in r.meta for r in recs)
    assert recs[0].meta["id"] == 0
    assert any("[email]" in r.text for r in recs) and not any("@example.com" in r.text for r in recs)
    saved = json.loads((out / "demo.stats.json").read_text())
    assert saved["kept"] + saved["repaired"] == 11 and saved["spec"]["name"] == "demo"


def test_rerun_skips_and_force_replaces_stale_chunks(tmp_path):
    src = tmp_path / "raw.jsonl"
    _write_source(src)
    out = tmp_path / "cleaned"
    I.ingest(_spec(src), out, workers=0, docs_per_chunk=2)
    n_before = len(discover_chunks(out)["demo"])
    stamp = (out / "demo.0000.jsonl").stat().st_mtime_ns
    I.ingest(_spec(src), out, workers=0, docs_per_chunk=2)          # done: skipped
    assert (out / "demo.0000.jsonl").stat().st_mtime_ns == stamp
    I.ingest(_spec(src), out, workers=0, docs_per_chunk=100, force=True)
    assert n_before > 1 and len(discover_chunks(out)["demo"]) == 1   # no stale higher-numbered chunks


def test_max_docs_caps_raw_documents(tmp_path):
    src = tmp_path / "raw.jsonl"
    _write_source(src)
    st = I.ingest(_spec(src, max_docs=3), tmp_path / "out", workers=0)
    assert st.docs_in == 3


def test_process_pool_matches_inline(tmp_path):
    src = tmp_path / "raw.jsonl"
    _write_source(src, n_good=30)
    a, b = tmp_path / "a", tmp_path / "b"
    I.ingest(_spec(src), a, workers=0, docs_per_chunk=7)
    I.ingest(_spec(src), b, workers=2, docs_per_chunk=7)
    names = [p.name for p in discover_chunks(a)["demo"]]
    assert names == [p.name for p in discover_chunks(b)["demo"]]
    for n in names:
        assert (a / n).read_bytes() == (b / n).read_bytes()


def test_zstd_chunks_round_trip(tmp_path):
    pytest.importorskip("zstandard")
    src = tmp_path / "raw.jsonl"
    _write_source(src)
    out = tmp_path / "z"
    I.ingest(_spec(src), out, workers=0, docs_per_chunk=5, zstd=True)
    chunks = discover_chunks(out)["demo"]
    assert all(p.name.endswith(".jsonl.zst") for p in chunks)
    assert sum(1 for c in chunks for _ in iter_records(c)) == 11


def test_build_spec_from_command_line(tmp_path):
    import argparse

    def ns(**kw):
        base = dict(preset=None, spec=None, hf=None, local=None, adapter=None, name=None, source_class=None,
                    config=None, split=None, text_field=None, format=None, min_chars=None, max_docs=None,
                    no_scrub=False)
        base.update(kw)
        return argparse.Namespace(**base)

    s = I.build_spec(ns(preset="smoltalk", min_chars=10, no_scrub=True))
    assert s.name == "smoltalk" and s.min_chars == 10 and s.scrub_pii is False
    s = I.build_spec(ns(hf="org/data", name="mine", source_class="web", config=["a", "b"]))
    assert s.dataset == "org/data" and s.configs == ("a", "b") and s.source_class.value == "web"
    yaml_path = tmp_path / "src.yaml"
    yaml_path.write_text("name: fromyaml\nadapter: local\ndataset: /data/x\nformat: fields\nfields: [a, b]\n")
    assert I.build_spec(ns(spec=str(yaml_path))).fields == ("a", "b")
    with pytest.raises(SourceError):
        I.build_spec(ns(hf="org/data"))             # needs --name
    with pytest.raises(SourceError):
        I.build_spec(ns(hf="x", local="y", name="n"))
    with pytest.raises(SourceError):
        I.build_spec(ns(preset="nope"))


def test_cli_end_to_end(tmp_path, capsys):
    src = tmp_path / "raw.jsonl"
    _write_source(src)
    rc = I.main(["--local", str(src), "--name", "cli", "--out_dir", str(tmp_path / "o"), "--workers", "0"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["docs_in"] == 13
    assert I.main(["--list_presets"]) == 0
    assert I.main(["--hf", "x", "--out_dir", str(tmp_path / "o")]) == 2


def test_audit_samples_and_reports_verdicts(tmp_path, capsys):
    from kotodama.data.curation import audit as Au
    from kotodama.data.curation.filters import SourceClass

    raw = tmp_path / "raw"
    raw.mkdir()
    texts = [GOOD * 3] * 30 + ["<div><b>x</b></div> " * 100] * 10
    with ChunkWriter(raw / "web.0000.jsonl") as w:
        for t in texts:
            w.write(t, {})
    out = tmp_path / "audit"
    sample, total = Au.reservoir(iter_records(raw / "web.0000.jsonl"), 15, seed=1)
    assert total == 40 and len(sample) == 15 and len({r.line for r in sample}) == 15
    rc = Au.main(["--in_dir", str(raw), "--source", "web", "--class", "web", "--out_dir", str(out), "--n", "100"])
    assert rc == 0
    summary = json.loads((out / "web.summary.json").read_text())
    assert summary["total_docs"] == 40 and summary["action_pct"] == {"keep": 75.0, "drop": 25.0}
    assert summary["reason_pct"] == {"html_soup": 25.0}
    assert len(json.loads((out / "web.exemplars.json").read_text())["html_soup"]) == 10
    assert sum(1 for _ in open(out / "web.metrics.jsonl")) == 40
    direct = Au.audit(iter_records(raw / "web.0000.jsonl"), "web", SourceClass.CODE, tmp_path / "a2")
    assert direct["action_pct"] == {"keep": 100.0}
    assert Au.main(["--in_dir", str(raw), "--source", "nope", "--class", "web", "--out_dir", str(out)]) == 2
    capsys.readouterr()
