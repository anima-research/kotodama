"""Exact and MinHash LSH deduplication (kotodama.data.curation.dedup)."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from kotodama.data.curation import dedup as D
from kotodama.data.curation.chunks import ChunkWriter, discover_chunks, iter_records

WORDS = ("amber basin cedar delta ember fjord grove harbor island jetty kettle lagoon meadow nectar orchard "
         "prairie quarry ridge summit thicket upland valley willow yarrow zephyr").split()


def _doc(seed: int, n: int = 200) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n))


def _near(text: str, k: int = 3, seed: int = 0) -> str:
    """The same text with ``k`` words replaced."""
    rng = random.Random(seed)
    words = text.split()
    for i in rng.sample(range(len(words)), k):
        words[i] = "changed"
    return " ".join(words)


def _write(d, source: str, chunks: list[list[str]]) -> None:
    for i, texts in enumerate(chunks):
        with ChunkWriter(d / f"{source}.{i:04d}.jsonl") as w:
            for t in texts:
                w.write(t, {"source": source})


def _texts(d, source: str) -> list[str]:
    return [r.text for c in discover_chunks(d).get(source, []) for r in iter_records(c)]


def test_signature_properties():
    a = _doc(1)
    assert np.array_equal(D.minhash_signature(a), D.minhash_signature(a))
    sim_near = np.mean(D.minhash_signature(a) == D.minhash_signature(_near(a)))
    sim_far = np.mean(D.minhash_signature(a) == D.minhash_signature(_doc(2)))
    assert sim_near > 0.8 and sim_far < 0.3
    assert D.minhash_signature("   ") is None
    assert D.minhash_signature("two words").shape == (D.NUM_HASHES,)
    assert D.exact_hash("a  b\nc") == D.exact_hash("a b c")


def test_union_find_roots():
    uf = D.UnionFind(6)
    uf.union_group(np.array([0, 2]))
    uf.union_group(np.array([2, 4]))
    uf.union_group(np.array([1, 5]))
    r = uf.roots()
    assert r[0] == r[2] == r[4] and r[1] == r[5] and len({r[0], r[1], r[3]}) == 3


def test_exact_mode_respects_source_priority(tmp_path):
    inp, out = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    shared, near = _doc(1), _near(_doc(1))
    _write(inp, "web", [[shared, _doc(3)], [near]])
    _write(inp, "wiki", [[shared, _doc(4)]])
    summary = D.dedup(inp, out, mode="exact", priority={"wiki": 10}, workers=0)
    assert summary["removed"] == 1 and summary["per_source"]["web"]["removed"] == 1
    assert shared in _texts(out, "wiki") and shared not in _texts(out, "web")
    assert near in _texts(out, "web")   # exact mode leaves near-duplicates
    assert json.loads((out / "dedup.summary.json").read_text())["mode"] == "exact"


def test_minhash_mode_removes_near_duplicates(tmp_path):
    inp, out = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    base = _doc(1)
    _write(inp, "a", [[base, _doc(5)], [_doc(6)]])
    _write(inp, "b", [[_near(base, seed=1), _near(base, seed=2), _doc(7)]])
    summary = D.dedup(inp, out, mode="minhash", workers=0)
    assert summary["removed"] == 2 and summary["clusters"] == 1
    assert _texts(out, "a") == [base, _doc(5), _doc(6)]    # tie on priority: the earliest document survives
    assert _texts(out, "b") == [_doc(7)]
    assert [p.name for p in discover_chunks(out)["a"]] == ["a.0000.jsonl", "a.0001.jsonl"]


def test_index_is_reused_and_dry_run_writes_nothing(tmp_path, monkeypatch):
    inp, idx = tmp_path / "in", tmp_path / "idx"
    inp.mkdir()
    _write(inp, "a", [[_doc(1), _doc(1)]])
    calls = []
    real = D.sign_chunk
    monkeypatch.setattr(D, "sign_chunk", lambda *a: calls.append(a) or real(*a))
    first = D.dedup(inp, None, index_dir=idx, workers=0)
    assert first["removed"] == 1 and len(calls) == 1
    D.dedup(inp, None, index_dir=idx, workers=0)
    assert len(calls) == 1                     # unchanged chunk: shard reused
    D.dedup(inp, None, index_dir=idx, mode="exact", workers=0)
    assert len(calls) == 2                     # different params: re-signed
    assert not (tmp_path / "out").exists()


def test_process_pool_matches_inline(tmp_path):
    inp = tmp_path / "in"
    inp.mkdir()
    base = _doc(1)
    _write(inp, "a", [[base, _doc(2)], [_near(base)]])
    _write(inp, "b", [[_doc(3), base]])
    s0 = D.dedup(inp, tmp_path / "o0", workers=0)
    s2 = D.dedup(inp, tmp_path / "o2", workers=2)
    assert s0["removed"] == s2["removed"] == 2
    for name in ("a.0000.jsonl", "a.0001.jsonl", "b.0000.jsonl"):
        assert (tmp_path / "o0" / name).read_bytes() == (tmp_path / "o2" / name).read_bytes()


def test_refusals_and_cli(tmp_path, capsys):
    inp = tmp_path / "in"
    inp.mkdir()
    with pytest.raises(FileNotFoundError):
        D.dedup(inp, tmp_path / "o", workers=0)
    _write(inp, "a", [[_doc(1)]])
    with pytest.raises(ValueError):
        D.dedup(inp, inp, workers=0)
    with pytest.raises(ValueError):
        D.parse_priority(["noequals"])
    assert D.main(["--in_dir", str(inp), "--out_dir", str(tmp_path / "o"), "--workers", "0",
                   "--priority", "a=3"]) == 0
    assert json.loads(capsys.readouterr().out)["total_docs"] == 1
