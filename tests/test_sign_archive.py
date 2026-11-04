"""The sign archive (kotodama.serve.sign_archive): a spool's sealed batches copied into compressed segments, read back
byte for byte, found by ``find`` after the spool's copy is dropped, and never touched while unsealed."""

from __future__ import annotations

import io
import json
import os
import tarfile
import time
from pathlib import Path

import numpy as np
import pytest

zstandard = pytest.importorskip("zstandard")

from kotodama.model.signer import SignRecord, SignResult  # noqa: E402
from kotodama.model.signing import ReducedRow  # noqa: E402
from kotodama.serve import sign_archive  # noqa: E402
from kotodama.serve.sign import SignSpool, find  # noqa: E402
from kotodama.serve.sign_archive import Archive, ArchiveError, drop_local  # noqa: E402

W = "0" * 64


def archive_spool(*args, **kwargs):
    """The archive run with no free-space floor (a test's filesystem may be small)."""
    return sign_archive.archive_spool(*args, floor_gb=kwargs.pop("floor_gb", 0.0), **kwargs)


def _batch(spool: SignSpool, tag: str, n: int = 3, raw: bool = True, age_s: float = 7200.0) -> Path:
    """One signed batch written the way a drain writes it, its files dated ``age_s`` ago (plus audit captures)."""
    rng = np.random.default_rng(abs(hash(tag)) % 2 ** 32)
    results = []
    for i in range(n):
        rec = SignRecord(request_id=f"{tag}-{i}", prompt_ids=(1, 2), generated_ids=(3, 4, 5), weights=W,
                         meta={"tag": tag})
        red = ReducedRow(features=rng.standard_normal(6).astype(np.float32), names_digest="d" * 32, lane_id="lane",
                         meta={"feature_names": [f"f{j}" for j in range(6)], "steps": 2})
        results.append(SignResult(record=rec, reduced=red, same_engine=True))
    summary = spool._write_batch(results, {"weights": W, "tag": tag})
    bdir = spool.signed_dir / summary["batch"]
    if raw:
        (bdir / "raw").mkdir()
        for i in range(n):
            (bdir / "raw" / f"{tag}-{i}.pt").write_bytes(rng.bytes(4096))
    old = time.time() - age_s
    for p in [bdir, *bdir.rglob("*")]:
        os.utime(p, (old, old))
    return bdir


def _files(d: Path) -> dict[str, bytes]:
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def test_archive_copies_sealed_batches_into_verified_segments(tmp_path) -> None:
    spool = SignSpool(tmp_path / "spool")
    batches = [_batch(spool, f"b{i}") for i in range(3)]
    before = {b.name: _files(b) for b in batches}
    out = archive_spool(spool.root, tmp_path / "arch", name="s")
    assert out["batches"] == 3 and out["stopped"] is None
    # one segment per stream (all batches are from one day), the audit captures on their own
    assert sorted(s["segment"].split("-")[3] for s in out["segments"]) == ["main", "raw"]
    arch = Archive(tmp_path / "arch" / "s")
    assert arch.verify() == {"segments": 2, "frames": 6, "batches": 3, "failures": []}
    for b in batches:  # the spool's copies stay, untouched
        assert _files(b) == before[b.name]
        held = arch.read_batch(b.name, streams=("main", "raw"))
        assert {k.split("/", 1)[1]: v for k, v in held.items()} == before[b.name]
        assert arch.read_member(b.name, "rows.jsonl") == before[b.name]["rows.jsonl"]
    # rows are located without opening a frame; the newest batch wins as in find
    fr, file, idx = arch.locate("b1-2")
    assert fr.batch == batches[1].name and idx == 2 and file.startswith("features-")
    # the whole segment is plain tools' data: its frames decompress to concatenated tars (`zstd -dc | tar -xi`)
    seg = next(s["segment"] for s in out["segments"] if "-main-" in s["segment"])
    with open(arch.segments / seg, "rb") as f:
        data = zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True).read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:", ignore_zeros=True) as tar:
        names = sorted(tar.getnames())
    assert names == sorted(f"{b.name}/{rel}" for b in batches for rel in before[b.name] if not rel.startswith("raw"))
    assert (arch.segments / f"{seg}.sha256").read_text().split() == [sign_archive._file_sha256(arch.segments / seg),
                                                                     seg]
    assert not list(arch.segments.glob(".*")) and not list(arch.index_dir.glob(".*"))


def test_archive_is_idempotent_and_leaves_unsealed_batches(tmp_path) -> None:
    spool = SignSpool(tmp_path / "spool")
    old = _batch(spool, "old")
    young = _batch(spool, "young", age_s=10.0)
    # a directory the drain renamed but never indexed (a crash in between), and one still being written
    stray = spool.signed_dir / "00000000000000000001-1"
    stray.mkdir()
    (stray / "rows.jsonl").write_text("{}\n")
    (spool.signed_dir / ".00000000000000000002-1").mkdir()
    first = archive_spool(spool.root, tmp_path / "arch", name="s")
    assert first["batches"] == 1 and first["young"] == 1
    arch = Archive(tmp_path / "arch" / "s")
    assert set(arch.batches()) == {old.name}
    again = archive_spool(spool.root, tmp_path / "arch", name="s")
    assert again["batches"] == 0 and again["already"] == 1 and again["segments"] == []
    later = archive_spool(spool.root, tmp_path / "arch", name="s", min_age_s=0)
    assert later["batches"] == 1 and set(Archive(tmp_path / "arch" / "s").batches()) == {old.name, young.name}
    assert sorted(sign_archive.archived(spool.root)) == sorted([old.name, young.name])
    # a run that died after its index and before the spool's note: the next run notes it, archiving nothing again
    lines = (spool.root / "archived.jsonl").read_text().splitlines()
    (spool.root / "archived.jsonl").write_text(lines[0] + "\n")
    healed = archive_spool(spool.root, tmp_path / "arch", name="s", min_age_s=0)
    assert healed["renoted"] == 1 and healed["batches"] == 0
    assert sorted(sign_archive.archived(spool.root)) == sorted([old.name, young.name])


def test_find_and_replay_inputs_resolve_after_the_spool_copy_is_dropped(tmp_path) -> None:
    spool = SignSpool(tmp_path / "spool", signed_dir=tmp_path / "bulk")
    spool.add("srv", [SignRecord(request_id=f"a-{i}", prompt_ids=(1, 2), generated_ids=(3, 4, 5), weights=W)
                      for i in range(3)])
    a = _batch(spool, "a")
    b = _batch(spool, "b")
    rec, row = find(spool, "a-1")
    stored = np.load(Path(row["batch_dir"]) / row["file"])[row["index"]]
    archive_spool(spool.root, tmp_path / "arch")
    assert drop_local(spool.root, dry_run=True)["dropped"] == 2 and a.is_dir()
    out = drop_local(spool.root, batches=[a.name])
    assert out["dropped"] == 1 and out["refused"] == {} and not a.exists() and b.is_dir()
    assert [json.loads(x)["batch"] for x in (spool.root / "dropped.jsonl").read_text().splitlines()] == [a.name]
    pointer = json.loads((spool.root / "ARCHIVED.json").read_text())
    assert pointer["batches_dropped"] == 1
    assert pointer["archives"] == [sign_archive.archived(spool.root)[a.name]["archive"]]
    assert "extract --archive" in pointer["extract"][0]
    rec2, row2 = find(spool, "a-1")
    assert rec2 == rec and row2["batch_dir"] == str(spool.root / "restored" / a.name)
    assert {k: v for k, v in row2.items() if k != "batch_dir"} == {k: v for k, v in row.items() if k != "batch_dir"}
    assert np.array_equal(np.load(Path(row2["batch_dir"]) / row2["file"])[row2["index"]], stored)
    assert not (spool.root / "restored" / a.name / "raw").exists()  # a replay reads the main stream only
    assert find(spool, "b-0")[1]["batch_dir"] == str(b)  # a batch still in the spool is read in place
    assert find(spool, "nobody") == (None, None)
    # the extract command restores the plain layout, audit captures included on request
    dest = tmp_path / "plain"
    assert sign_archive.main(["extract", "--archive", str(sign_archive.Archive(
        sign_archive.archived(spool.root)[a.name]["archive"]).root), "--batch", a.name, "--to", str(dest),
        "--raw"]) == 0
    assert sorted(_files(dest / a.name)) == sorted(["engine.json", "features-dddddddddddddddd.npy", "names.json",
                                                    "rows.jsonl", "raw/a-0.pt", "raw/a-1.pt", "raw/a-2.pt"])


def test_drop_local_refuses_what_the_archive_does_not_hold_byte_for_byte(tmp_path) -> None:
    spool = SignSpool(tmp_path / "spool")
    a = _batch(spool, "a")
    archive_spool(spool.root, tmp_path / "arch")
    late = _batch(spool, "late")  # written after the run: not archived
    (a / "rows.jsonl").write_text((a / "rows.jsonl").read_text() + "\n")  # changed since it was archived
    (late / "extra").write_text("x")
    out = drop_local(spool.root, batches=[a.name, late.name])
    assert out["dropped"] == 0 and set(out["refused"]) == {a.name, late.name}
    assert not (spool.root / "ARCHIVED.json").exists()  # nothing dropped, no pointer
    assert "differs" in out["refused"][a.name] and out["refused"][late.name] == "not archived"
    assert a.is_dir() and late.is_dir()
    assert sign_archive.main(["drop_local", "--spool", str(spool.root)]) == 1


def test_a_full_archive_filesystem_stops_sealing(tmp_path, monkeypatch) -> None:
    spool = SignSpool(tmp_path / "spool")
    _batch(spool, "a")
    rc = sign_archive.main(["archive", "--spool", str(spool.root), "--to", str(tmp_path / "arch"), "--name", "s",
                            "--floor_gb", "1e9"])
    assert rc == 3
    arch = Archive(tmp_path / "arch" / "s")
    assert arch.frames() == [] and not list(arch.segments.iterdir())
    assert not (spool.root / "archived.jsonl").exists()


def test_a_damaged_segment_is_reported_and_never_read_as_good(tmp_path) -> None:
    spool = SignSpool(tmp_path / "spool")
    a = _batch(spool, "a", raw=False)
    out = archive_spool(spool.root, tmp_path / "arch", name="s")
    arch = Archive(tmp_path / "arch" / "s")
    seg = arch.segments / out["segments"][0]["segment"]
    blob = bytearray(seg.read_bytes())
    blob[len(blob) // 2] ^= 0xFF
    seg.write_bytes(bytes(blob))
    report = arch.verify()
    assert len(report["failures"]) == 2  # the segment's sha256 and its one frame
    with pytest.raises(ArchiveError, match="does not match its index sha256"):
        arch.read_batch(a.name)
    assert drop_local(spool.root)["refused"][a.name].endswith("does not match its index sha256")
    assert a.is_dir()


def test_info_reports_the_spool_disk_against_its_floor(tmp_path, monkeypatch) -> None:
    from kotodama.serve import server

    monkeypatch.setattr(server, "_sign_spool", SignSpool(tmp_path / "spool"))
    monkeypatch.setattr(server, "_sign_disk_floor_gb_arg", 1e9)
    monkeypatch.setattr(server, "_sign_disk_low", False)
    disk = server._sign_disk()
    assert disk["low"] is True and disk["floor_gb"] == 1e9 and disk["free_gb"] > 0
    monkeypatch.setattr(server, "_sign_disk_floor_gb_arg", 0.0)
    assert server._sign_disk()["low"] is False
