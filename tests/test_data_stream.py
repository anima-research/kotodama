"""Inspect, sample and remix assembled streams (kotodama.data.stream), and the whole curation
pipeline end to end on synthetic text."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from kotodama.data import stream as St
from kotodama.data.curation import assemble as A
from kotodama.data.curation import dedup as D
from kotodama.data.curation import ingest as I
from kotodama.data.curation import tokenize as T
from kotodama.data.curation.sources import SourceSpec
from kotodama.data.dataset import TokenizedDataset
from kotodama.data.provenance import ProvenanceError, load_labels, provenance_path

SEQ = 8
EOS = 0


def _stream(tmp_path: Path, sizes: dict[str, int], seed: int = 3) -> Path:
    """An assembled stream whose chunk tokens encode (source index + 1) * 100 and carry EOS at position 3."""
    tok = tmp_path / "tok"
    tok.mkdir()
    for s_i, (name, n) in enumerate(sorted(sizes.items())):
        chunk = np.full(SEQ, (s_i + 1) * 100, dtype=np.uint16)
        chunk[3] = EOS
        np.tile(chunk, n).tofile(tok / f"{name}.bin")
        (tok / f"{name}.bin.json").write_text(json.dumps({"tokenizer": "char-test", "eos_id": EOS}))
    parts, tokenizer, eos = A.plan_from_dir(tok, SEQ, seed)
    out = tmp_path / "train.bin"
    A.build(parts, out, SEQ, seed, tokenizer, eos)
    return out


def test_inspect_reports_sources_and_documents(tmp_path):
    s = _stream(tmp_path, {"a": 3, "b": 5})
    r = St.inspect_stream(s)
    assert r["tokens"] == 8 * SEQ and r["sequences"] == 8 and r["seq_len"] == SEQ
    assert r["sources"]["a"] == {"chunks": 3, "tokens": 3 * SEQ, "share": 0.375}
    assert r["documents"] == 8 and r["eos_id"] == EOS
    assert r["longest_doc_tokens"] == SEQ - 1 and r["token_id_range"] == [0, 200]
    provenance_path(s).unlink()
    bare = St.inspect_stream(s)
    assert bare["provenance"] is None and isinstance(bare["documents"], str)
    assert St.inspect_stream(s, eos_id=EOS)["documents"] == 8


def test_sample_by_source_and_split_docs(tmp_path):
    s = _stream(tmp_path, {"a": 4, "b": 6})
    picks = St.sample_chunks(s, 10, source="b", seed=1)
    assert len(picks) == 6 and all(src == "b" and toks[0] == 200 for _, src, toks in picks)
    with pytest.raises(ProvenanceError):
        St.sample_chunks(s, 1, source="nope")
    docs = St.split_docs(np.array([5, 6, 0, 7, 0, 0, 8]), 0)
    assert [d.tolist() for d in docs] == [[5, 6], [7], [8]]


def test_remix_drop_and_weight_and_chain(tmp_path):
    s = _stream(tmp_path, {"a": 4, "b": 6, "c": 2})
    out = tmp_path / "remix.bin"
    St.remix(s, out, drop=["c"], weights={"a": 2.0}, caps={}, seed=9)
    labels = load_labels(out)
    assert labels.counts() == {"a": 8, "b": 6}
    data = np.fromfile(out, dtype=np.uint16).reshape(-1, SEQ)
    assert np.array_equal(data[:, 0] // 100 - 1, labels.labels)     # every chunk still labels true
    prov = labels.provenance
    assert prov.parent["stream"] == "train.bin" and len(prov.parent["provenance_sha256"]) == 64
    assert prov.eos_id == EOS and prov.tokenizer == "char-test"
    out2 = tmp_path / "remix2.bin"
    St.remix(out, out2, drop=[], weights={"b": 0.5}, caps={}, seed=1)   # a remix of a remix
    assert load_labels(out2).counts() == {"a": 8, "b": 3}
    with pytest.raises(ValueError):
        St.remix(s, s, [], {}, {}, 0)
    with pytest.raises(ValueError):
        St.remix(s, tmp_path / "x.bin", ["zzz"], {}, {}, 0)
    St.remix(s, tmp_path / "dry.bin", [], {}, {}, 0, dry_run=True)
    assert not (tmp_path / "dry.bin").exists()
    provenance_path(s).unlink()
    with pytest.raises(ProvenanceError):
        St.remix(s, tmp_path / "y.bin", [], {}, {}, 0)


def test_cli(tmp_path, capsys):
    s = _stream(tmp_path, {"a": 2, "b": 2})
    assert St.main(["inspect", str(s), "--json", str(tmp_path / "r.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text())["sequences"] == 4
    assert St.main(["remix", str(s), "--out", str(tmp_path / "m.bin"), "--drop", "a"]) == 0
    assert load_labels(tmp_path / "m.bin").counts() == {"b": 2}
    assert St.main(["inspect", str(tmp_path / "missing.bin")]) == 2
    capsys.readouterr()


# ---------------------------------------------------------------------------
# the whole pipeline
# ---------------------------------------------------------------------------

class CharTok:
    eos_token_id = 0
    name_or_path = "char-test"

    def __len__(self) -> int:
        return 1001

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[ord(c) % 1000 + 1 for c in t] for t in texts]}


def test_pipeline_end_to_end(tmp_path):
    rng = random.Random(0)
    words = "stone river lantern copper meadow thistle harbor violet ember willow".split()

    def para(n: int = 60) -> str:
        return " ".join(rng.choice(words) for _ in range(n)).capitalize() + "."

    raw = tmp_path / "raw"
    raw.mkdir()
    books = [para() for _ in range(30)]
    web = [para() for _ in range(20)] + books[:5] + ["<div><p>menu</p></div>" * 50]
    (raw / "books.jsonl").write_text("".join(json.dumps({"text": t}) + "\n" for t in books))
    (raw / "web.jsonl").write_text("".join(json.dumps({"body": t}) + "\n" for t in web))

    cleaned = tmp_path / "cleaned"
    I.ingest(SourceSpec("books", adapter="local", dataset=str(raw / "books.jsonl")), cleaned, workers=0,
             docs_per_chunk=10)
    st = I.ingest(SourceSpec("web", adapter="local", dataset=str(raw / "web.jsonl"), text_field="body",
                             source_class="web"), cleaned, workers=0, docs_per_chunk=10)
    assert st.drop_reasons == {"html_soup": 1}

    deduped = tmp_path / "deduped"
    summary = D.dedup(cleaned, deduped, priority={"books": 10}, workers=0)
    assert summary["per_source"]["web"]["removed"] == 5 and summary["per_source"]["books"]["removed"] == 0

    tokenized = tmp_path / "tok"
    sides = T.tokenize_dir(deduped, tokenized, CharTok(), workers=0)
    assert sides["books"].docs == 30 and sides["web"].docs == 20

    seq = 64
    parts, tok, eos = A.plan_from_dir(tokenized, seq, 42)
    out = tmp_path / "train.bin"
    A.build(parts, out, seq, 42, tok, eos)
    report = St.inspect_stream(out)
    assert set(report["sources"]) == {"books", "web"} and report["eos_id"] == 0
    ds = TokenizedDataset(out, seq_len=seq, doc_masking=True, eos_id=report["eos_id"])
    sample = next(iter(ds))
    assert sample.input_ids.shape == (seq,)
