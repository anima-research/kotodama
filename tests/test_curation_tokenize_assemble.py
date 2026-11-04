"""Tokenize chunks into per-source streams and assemble them with provenance
(kotodama.data.curation.tokenize / .assemble, kotodama.data.provenance)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from kotodama.data.curation import assemble as A
from kotodama.data.curation import tokenize as T
from kotodama.data.curation.chunks import ChunkWriter
from kotodama.data.provenance import Provenance, ProvenanceError, load_labels, provenance_path


class CharTok:
    """A character tokenizer: id = codepoint % 1000 + 1, EOS = 0. Picklable, so worker processes can use it."""

    eos_token_id = 0
    name_or_path = "char-test"

    def __len__(self) -> int:
        return 1001

    def __call__(self, texts, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [[ord(c) % 1000 + 1 for c in t] for t in texts]}

    def decode(self, ids) -> str:
        return "".join(chr(i - 1) for i in ids)


class HugeTok(CharTok):
    def __len__(self) -> int:
        return 70_000


def _chunks(d: Path, source: str, docs: list[list[str]]) -> None:
    for i, texts in enumerate(docs):
        with ChunkWriter(d / f"{source}.{i:04d}.jsonl") as w:
            for t in texts:
                w.write(t, {})


def test_encode_texts_appends_eos_and_checks_range():
    arr = T.encode_texts(CharTok(), ["ab", "", "c"], eos_id=0)
    assert arr.tolist() == [98, 99, 0, 100, 0] and arr.dtype == np.uint16

    class Bad(CharTok):
        def __call__(self, texts, add_special_tokens=False):
            return {"input_ids": [[70_000]]}

    with pytest.raises(ValueError):
        T.encode_texts(Bad(), ["x"], 0)


def test_tokenize_round_trip_and_sidecar(tmp_path):
    inp, out = tmp_path / "chunks", tmp_path / "tok"
    inp.mkdir()
    docs = [["hello world", "second doc"], ["third"]]
    _chunks(inp, "s", docs)
    res = T.tokenize_dir(inp, out, CharTok(), workers=0)
    data = np.fromfile(out / "s.bin", dtype=np.uint16)
    pieces = [p for p in np.split(data, np.flatnonzero(data == 0) + 1) if len(p)]
    tok = CharTok()
    assert [tok.decode(p[:-1].tolist()) for p in pieces] == [t for c in docs for t in c]
    side = json.loads((out / "s.bin.json").read_text())
    assert side == {"source": "s", "tokens": len(data), "docs": 3, "chunks": 2, "tokenizer": "char-test",
                    "eos_id": 0, "vocab_size": 1001, "dtype": "uint16"} and res["s"].tokens == len(data)
    assert not (out / ".parts").exists()


def test_tokenize_pool_matches_inline_and_skips_done(tmp_path):
    inp = tmp_path / "chunks"
    inp.mkdir()
    _chunks(inp, "a", [["alpha " * 50] * 5, ["beta " * 40] * 3, ["gamma"]])
    _chunks(inp, "b", [["delta epsilon"] * 7])
    T.tokenize_dir(inp, tmp_path / "t0", CharTok(), workers=0)
    T.tokenize_dir(inp, tmp_path / "t2", CharTok(), workers=2, batch_docs=2)
    for s in ("a", "b"):
        assert (tmp_path / "t0" / f"{s}.bin").read_bytes() == (tmp_path / "t2" / f"{s}.bin").read_bytes()
    stamp = (tmp_path / "t0" / "a.bin").stat().st_mtime_ns
    T.tokenize_dir(inp, tmp_path / "t0", CharTok(), workers=0)
    assert (tmp_path / "t0" / "a.bin").stat().st_mtime_ns == stamp


def test_tokenize_refusals(tmp_path):
    inp = tmp_path / "chunks"
    inp.mkdir()
    _chunks(inp, "a", [["x"]])
    with pytest.raises(ValueError):
        T.tokenize_dir(inp, tmp_path / "o", HugeTok(), workers=0)

    class NoEos(CharTok):
        eos_token_id = None

    with pytest.raises(ValueError):
        T.tokenize_dir(inp, tmp_path / "o", NoEos(), workers=0)
    assert T.tokenize_dir(inp, tmp_path / "o", NoEos(), workers=0, eos_id=5)["a"].eos_id == 5
    with pytest.raises(FileNotFoundError):
        T.tokenize_dir(inp, tmp_path / "o", CharTok(), sources=["missing"], workers=0)


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------

SEQ = 8


def _bins(d: Path, sizes: dict[str, int], tokenizer: str = "char-test", eos: int = 0) -> None:
    """Per-source streams whose every token is (source index + 1) * 100 + chunk index % 100."""
    d.mkdir(parents=True, exist_ok=True)
    for s_i, (name, n_chunks) in enumerate(sorted(sizes.items())):
        chunk_vals = (s_i + 1) * 100 + (np.arange(n_chunks) % 100)
        np.repeat(chunk_vals, SEQ).astype(np.uint16).tofile(d / f"{name}.bin")
        (d / f"{name}.bin.json").write_text(json.dumps({"tokenizer": tokenizer, "eos_id": eos}))


def _reference_assemble(bins: list[Path], seq_len: int, seed: int) -> bytes:
    """Chunk i of the sources concatenated in file-name order lands at RandomState(seed).permutation(N)[i]."""
    chunks = []
    for p in sorted(bins):
        data = np.fromfile(p, dtype=np.uint16)
        n = len(data) // seq_len
        chunks.extend(data[i * seq_len:(i + 1) * seq_len] for i in range(n))
    out = [None] * len(chunks)
    for i, dst in enumerate(np.random.RandomState(seed).permutation(len(chunks))):
        out[dst] = chunks[i]
    return b"".join(c.tobytes() for c in out)


def test_assemble_matches_reference_permutation_and_labels(tmp_path):
    _bins(tmp_path / "tok", {"code": 5, "books": 3, "web": 9})
    with open(tmp_path / "tok" / "web.bin", "ab") as f:   # a trailing partial chunk is left out
        f.write(np.zeros(3, dtype=np.uint16).tobytes())
    parts, tok, eos = A.plan_from_dir(tmp_path / "tok", SEQ, seed=7)
    out = tmp_path / "train.bin"
    prov = A.build(parts, out, SEQ, 7, tok, eos)
    assert out.read_bytes() == _reference_assemble(list((tmp_path / "tok").glob("*.bin")), SEQ, 7)
    assert prov.sources == ["books", "code", "web"] and prov.chunks_per_source == [3, 5, 9]
    assert (prov.tokenizer, prov.eos_id) == ("char-test", 0)
    labels = load_labels(out)
    data = np.fromfile(out, dtype=np.uint16).reshape(-1, SEQ)
    assert np.array_equal(data[:, 0] // 100 - 1, labels.labels)
    assert labels.counts() == {"books": 3, "code": 5, "web": 9}
    assert not list(tmp_path.glob("*.tmp"))


def test_selection_rules():
    avail = np.arange(10)
    sel = lambda **kw: A.select_chunks(avail, A.Selection(**kw), SEQ, 1, "s")  # noqa: E731
    assert np.array_equal(sel(), avail)
    half = sel(weight=0.5)
    assert len(half) == 5 and len(set(half)) == 5 and np.all(np.diff(half) > 0)
    assert np.array_equal(sel(weight=0.5), half)                          # seeded
    assert len(sel(weight=2.0)) == 20 and len(sel(weight=1.5)) == 15
    assert len(sel(weight=0.0)) == 0
    assert len(sel(weight=2.0, cap_tokens=7 * SEQ + 3)) == 7
    with pytest.raises(ValueError):
        A.Selection(weight=-1)


def test_assemble_drop_weight_cap_via_cli(tmp_path, capsys):
    _bins(tmp_path / "tok", {"a": 10, "b": 10, "c": 10})
    out = tmp_path / "mix.bin"
    rc = A.main(["--in_dir", str(tmp_path / "tok"), "--out", str(out), "--seq_len", str(SEQ),
                 "--drop", "c", "--weight", "a=0.5", "--cap", f"b={4 * SEQ}"])
    assert rc == 0
    labels = load_labels(out)
    assert labels.counts() == {"a": 5, "b": 4}
    sel = labels.provenance.selection
    assert sel["a"]["weight"] == 0.5 and sel["b"]["cap_tokens"] == 4 * SEQ and "c" not in sel
    assert A.main(["--in_dir", str(tmp_path / "tok"), "--out", str(tmp_path / "x.bin"), "--seq_len",
                   str(SEQ), "--dry_run"]) == 0
    assert not (tmp_path / "x.bin").exists()
    assert A.main(["--in_dir", str(tmp_path / "tok"), "--out", str(tmp_path / "x.bin"), "--seq_len",
                   str(SEQ), "--drop", "zzz"]) == 2
    capsys.readouterr()


def test_assemble_refuses_mixed_tokenizers(tmp_path):
    _bins(tmp_path / "tok", {"a": 2})
    _bins(tmp_path / "tok2", {"b": 2}, tokenizer="other")
    os.replace(tmp_path / "tok2" / "b.bin", tmp_path / "tok" / "b.bin")
    os.replace(tmp_path / "tok2" / "b.bin.json", tmp_path / "tok" / "b.bin.json")
    with pytest.raises(ValueError):
        A.plan_from_dir(tmp_path / "tok", SEQ, 0)


def test_legacy_provenance_with_paths_and_block_spans(tmp_path):
    # chunk-shuffle provenance whose sources are .bin paths and which records no tokenizer
    data = np.arange(6 * SEQ, dtype=np.uint16)
    stream = tmp_path / "train.bin"
    data.tofile(stream)
    provenance_path(stream).write_text(json.dumps({
        "sources": ["/data/tokenized/alpha.bin", "/data/tokenized/beta.bin"], "chunks_per_source": [2, 4],
        "total_chunks": 6, "seq_len": SEQ, "seed": 42, "total_tokens": 6 * SEQ, "output": "train.bin"}))
    labels = load_labels(stream)
    assert labels.sources == ["alpha", "beta"] and labels.counts() == {"alpha": 2, "beta": 4}
    assert labels.provenance.tokenizer is None
    # block spans in the stream's directory
    spans_dir = tmp_path / "spans"
    spans_dir.mkdir()
    data.tofile(spans_dir / "train.bin")
    (spans_dir / "manifest.json").write_text(json.dumps({"seq_len": SEQ}))
    (spans_dir / "provenance.json").write_text(json.dumps([
        {"source": "x", "global_start": 0, "global_end": 2 * SEQ},
        {"source": "y", "global_start": 2 * SEQ, "global_end": 6 * SEQ}]))
    sl = load_labels(spans_dir / "train.bin")
    assert sl.labels.tolist() == [0, 0, 1, 1, 1, 1] and sl.provenance is None
    # a provenance that disagrees with its stream is refused
    provenance_path(stream).write_text(json.dumps({
        "sources": ["a"], "chunks_per_source": [3], "seq_len": SEQ, "seed": 1}))
    with pytest.raises(ProvenanceError):
        load_labels(stream)
    with pytest.raises(ProvenanceError):
        Provenance(sources=["a", "a"], chunks_per_source=[1, 1], seq_len=SEQ, seed=0)
