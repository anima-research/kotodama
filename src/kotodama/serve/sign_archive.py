"""The sign archive: a sign spool's signed batches, sealed into compressed segments on long-term storage.

A spool (kotodama.serve.sign) writes one directory per signed batch. A batch is sealed once its line is in
``<spool>/index.jsonl``: the drain writes the directory under a temporary name and renames it before indexing it,
and nothing writes to it again. ``archive`` copies sealed batches into segments under ``<to>/<name>/`` (``name``
defaults to the spool's path, flattened) and leaves the spool's own copy in place. ``drop_local`` removes a spool's
copy only on request, and only for batches whose archived bytes it has just read back and matched against the local
files. Layout:

    <archive>/segments/<day>-<stream>-<run>.tar.zst   one zstd frame per batch; each frame is a tar of that batch's
                                                       files, named <batch>/<file>. The whole segment decompresses
                                                       to concatenated tars: `zstd -dc SEG | tar -xi` restores it
    <archive>/segments/<segment>.sha256               the segment's sha256 (`sha256sum -c` reads it)
    <archive>/index/<segment>.jsonl                   one line per frame: batch, stream, segment, byte offset and
                                                       length, the frame's sha256, each member's path, size and
                                                       sha256, and (main stream) each row's sign_id, features file
                                                       and row index
    <spool>/archived.jsonl                             per archived batch: where its frames went (``find`` and
                                                       ``/replay`` read a batch from there once the spool's copy is
                                                       gone)

Two streams: ``main`` holds everything a loader reads (``<batch>/rows.jsonl``, ``<batch>/names.json``,
``<batch>/features-*.npy``, ``<batch>/engine.json``); ``raw`` holds the audit captures (``<batch>/raw/*.pt``) on
their own, so they can be kept or dropped as a unit. Segments are written as ``.<name>.tmp``, fsynced and renamed,
and the index file is renamed last: a batch is archived once its index line exists, and a crashed run leaves only a
temporary file behind. Index files are only ever created, never rewritten.
Sealing stops while the archive's filesystem has less than ``floor_gb`` free.

    tools/run_py.sh -m kotodama.serve.sign_archive archive --spool <dir> --to <archive root>
    tools/run_py.sh -m kotodama.serve.sign_archive extract --archive <root>/<name> --batch <batch> --to <dir>
    tools/run_py.sh -m kotodama.serve.sign_archive verify --archive <root>/<name>
    tools/run_py.sh -m kotodama.serve.sign_archive locate --archive <root>/<name> --sign_id <id>
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import logging
import os
import shutil
import tarfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

ARCHIVE_FORMAT = "kotodama-sign-archive/1"
STREAMS = ("main", "raw")
_GB = 1024 ** 3


class ArchiveError(RuntimeError):
    """The archive cannot be written or read as asked (a full filesystem, a missing batch, bytes that disagree)."""


def _zstd() -> Any:
    try:
        import zstandard
    except ImportError as exc:  # an optional dependency: only archiving needs it
        raise ArchiveError("the sign archive needs the zstandard package (uv pip install zstandard)") from exc
    return zstandard


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fsync_write(path: Path, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def spool_name(spool_root: str | Path) -> str:
    """The default archive name of a spool: its absolute path with separators flattened (unique per node)."""
    parts = [p for p in Path(spool_root).resolve().parts if p not in ("/", "")]
    return "_".join(parts) or "spool"


def stream_of(rel: str) -> str:
    """The stream a batch member belongs to: ``raw`` for the audit captures, ``main`` for everything else."""
    return "raw" if rel.split("/", 1)[0] == "raw" else "main"


def batch_day(batch: str, fallback_mtime: float) -> str:
    """The UTC day a batch was written (its name starts with the nanosecond time the drain wrote it)."""
    head = batch.split("-", 1)[0]
    try:
        seconds = int(head) / 1e9
        if not 1e9 < seconds < 1e10:
            raise ValueError(head)
    except ValueError:
        seconds = fallback_mtime
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%d")


# ── reading an archive (the functions loaders build on) ──

@dataclass
class Frame:
    """One batch's frame in one stream: where it is and what it holds."""

    batch: str
    stream: str
    segment: str
    offset: int
    length: int
    sha256: str
    members: list[dict[str, Any]]
    rows: list[list[Any]] = field(default_factory=list)  # [sign_id, features file or None, row index or None]
    source_dir: Optional[str] = None
    spool: Optional[str] = None

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Frame":
        return cls(batch=d["batch"], stream=d["stream"], segment=d["segment"], offset=int(d["offset"]),
                   length=int(d["length"]), sha256=d["sha256"], members=list(d["members"]),
                   rows=[list(r) for r in d.get("rows", ())], source_dir=d.get("source_dir"), spool=d.get("spool"))

    def to_json(self) -> dict[str, Any]:
        return {"format": ARCHIVE_FORMAT, "batch": self.batch, "stream": self.stream, "segment": self.segment,
                "offset": self.offset, "length": self.length, "sha256": self.sha256, "members": self.members,
                "rows": self.rows, "source_dir": self.source_dir, "spool": self.spool}


class Archive:
    """One spool's archive directory (``<to>/<name>``), read through its index files."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._frames: Optional[list[Frame]] = None

    @property
    def segments(self) -> Path:
        return self.root / "segments"

    @property
    def index_dir(self) -> Path:
        return self.root / "index"

    def frames(self, reload: bool = False) -> list[Frame]:
        """Every indexed frame, oldest segment first."""
        if self._frames is None or reload:
            out: list[Frame] = []
            if self.index_dir.is_dir():
                for path in sorted(self.index_dir.glob("*.jsonl")):
                    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                        if not line.strip():
                            continue
                        try:
                            out.append(Frame.from_json(json.loads(line)))
                        except (ValueError, KeyError, TypeError) as exc:
                            raise ArchiveError(f"{path}:{n}: not an archive index line ({exc})") from exc
            self._frames = out
        return self._frames

    def batches(self) -> dict[str, dict[str, Frame]]:
        """Archived batches: batch -> stream -> its frame (the newest, if a batch was archived twice)."""
        out: dict[str, dict[str, Frame]] = {}
        for fr in self.frames():
            out.setdefault(fr.batch, {})[fr.stream] = fr
        return out

    def locate(self, sign_id: str) -> Optional[tuple[Frame, Optional[str], Optional[int]]]:
        """``sign_id``'s main-stream frame, features file and row index (None when no archived batch holds it; the
        newest batch wins, as in kotodama.serve.sign.find)."""
        hit = None
        for fr in self.frames():
            if fr.stream != "main":
                continue
            for sid, file, idx in fr.rows:
                if sid == sign_id:
                    hit = (fr, file, idx)
        return hit

    def read_frame(self, frame: Frame, verify: bool = True) -> dict[str, bytes]:
        """One frame's members (``<batch>/<file>`` -> bytes), read with one seek; ``verify`` checks the frame's and
        every member's sha256 against the index."""
        path = self.segments / frame.segment
        with open(path, "rb") as f:
            f.seek(frame.offset)
            blob = f.read(frame.length)
        if len(blob) != frame.length:
            raise ArchiveError(f"{path}: frame of {frame.batch} is cut short ({len(blob)} of {frame.length} bytes)")
        if verify and _sha256(blob) != frame.sha256:
            raise ArchiveError(f"{path}: frame of {frame.batch} does not match its index sha256")
        data = _zstd().ZstdDecompressor().decompress(blob)
        out: dict[str, bytes] = {}
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
            for info in tar:
                if not info.isfile():
                    continue
                fobj = tar.extractfile(info)
                out[info.name] = fobj.read() if fobj is not None else b""
        if verify:
            want = {m["path"]: m for m in frame.members}
            if set(want) != set(out):
                raise ArchiveError(f"{path}: frame of {frame.batch} holds {sorted(out)}, its index {sorted(want)}")
            for name, blob_m in out.items():
                if len(blob_m) != want[name]["size"] or _sha256(blob_m) != want[name]["sha256"]:
                    raise ArchiveError(f"{path}: {name} does not match its index sha256")
        return out

    def read_batch(self, batch: str, streams: tuple[str, ...] = ("main",)) -> dict[str, bytes]:
        """A batch's members in ``streams`` (``<batch>/<file>`` -> bytes), verified."""
        frames = self.batches().get(batch)
        if not frames:
            raise ArchiveError(f"no batch {batch!r} in {self.root}")
        out: dict[str, bytes] = {}
        for stream in streams:
            if stream in frames:
                out.update(self.read_frame(frames[stream]))
        return out

    def read_member(self, batch: str, rel: str) -> bytes:
        """One file of a batch (``rel``: its path inside the batch directory, the part after ``<batch>/``)."""
        members = self.read_batch(batch, streams=(stream_of(rel),))
        key = f"{batch}/{rel}"
        if key not in members:
            raise ArchiveError(f"batch {batch!r} has no {rel!r} in {self.root}")
        return members[key]

    def extract_batch(self, batch: str, dest: str | Path, streams: tuple[str, ...] = ("main",)) -> Path:
        """Write a batch back to the plain spool layout: ``<dest>/<batch>/<file>``. Files already there with the
        archived bytes are kept; a file with other bytes is refused. Returns the batch directory."""
        dest = Path(dest)
        out_dir = dest / batch
        for name, blob in self.read_batch(batch, streams).items():
            target = dest / name
            if not target.resolve().is_relative_to(out_dir.resolve()):
                raise ArchiveError(f"member {name!r} would land outside {out_dir}")
            if target.exists():
                if target.read_bytes() != blob:
                    raise ArchiveError(f"{target} exists with other bytes than the archive's")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            tmp.write_bytes(blob)
            os.replace(tmp, target)
        return out_dir

    def verify(self) -> dict[str, Any]:
        """Re-read every segment: its sha256 file, then each frame against the index. Returns counts and failures."""
        bad: list[str] = []
        segments = 0
        frames = self.frames(reload=True)
        for seg in sorted({fr.segment for fr in frames}):
            segments += 1
            path = self.segments / seg
            sidecar = self.segments / f"{seg}.sha256"
            try:
                want = sidecar.read_text().split()[0]
                if _file_sha256(path) != want:
                    bad.append(f"{seg}: sha256 differs from {sidecar.name}")
            except (OSError, IndexError) as exc:
                bad.append(f"{seg}: {exc}")
        for fr in frames:
            try:
                self.read_frame(fr)
            except (ArchiveError, OSError) as exc:
                bad.append(str(exc))
        return {"segments": segments, "frames": len(frames), "batches": len(self.batches()), "failures": bad}


# ── writing ──

@dataclass
class _Batch:
    name: str
    dir: Path
    members: list[tuple[str, Path]]  # (path relative to the batch dir, file)


def _sealed_batches(spool_root: Path, min_age_s: float) -> tuple[list[_Batch], dict[str, int]]:
    """The spool's indexed batches whose directory is present and at least ``min_age_s`` old, plus counts of what
    was passed over."""
    index = spool_root / "index.jsonl"
    seen: dict[str, Path] = {}
    if index.exists():
        for line in index.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                seen[str(entry["batch"])] = Path(entry["dir"])
            except (ValueError, KeyError, TypeError):
                logger.warning("sign archive: %s has a line that is not a batch entry; passed over", index)
    out: list[_Batch] = []
    skipped = {"missing": 0, "young": 0}
    now = time.time()
    for name, bdir in seen.items():
        if not bdir.is_dir() or not (bdir / "rows.jsonl").exists():
            skipped["missing"] += 1
            continue
        if now - bdir.stat().st_mtime < min_age_s:
            skipped["young"] += 1
            continue
        members = sorted((p.relative_to(bdir).as_posix(), p) for p in bdir.rglob("*")
                         if p.is_file() and not p.is_symlink() and not p.name.startswith("."))
        out.append(_Batch(name=name, dir=bdir, members=members))
    return out, skipped


def _tar_frame(batch: _Batch, members: list[tuple[str, Path]], level: int) -> tuple[bytes, list[dict[str, Any]]]:
    buf = io.BytesIO()
    meta: list[dict[str, Any]] = []
    with tarfile.open(fileobj=buf, mode="w:", format=tarfile.PAX_FORMAT) as tar:
        for rel, path in members:
            data = path.read_bytes()
            info = tarfile.TarInfo(f"{batch.name}/{rel}")
            info.size = len(data)
            info.mtime = int(path.stat().st_mtime)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
            meta.append({"path": info.name, "size": len(data), "sha256": _sha256(data)})
    return _zstd().ZstdCompressor(level=level).compress(buf.getvalue()), meta


def _rows_of(batch: _Batch) -> list[list[Any]]:
    rows: list[list[Any]] = []
    with open(batch.dir / "rows.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append([r.get("sign_id"), r.get("file"), r.get("index")])
    return rows


class _SegmentWriter:
    """Frames appended to one temporary segment file; ``commit`` fsyncs, renames and writes the index."""

    def __init__(self, archive: Archive, name: str, spool: str) -> None:
        self.archive, self.name, self.spool = archive, name, spool
        self.tmp = archive.segments / f".{name}.tmp"
        self.f = open(self.tmp, "wb")
        self.h = hashlib.sha256()
        self.frames: list[Frame] = []
        self.size = 0

    def add(self, frame_bytes: bytes, frame: Frame) -> None:
        frame.offset = self.size
        self.f.write(frame_bytes)
        self.h.update(frame_bytes)
        self.size += len(frame_bytes)
        self.frames.append(frame)

    def commit(self) -> list[Frame]:
        self.f.flush()
        os.fsync(self.f.fileno())
        self.f.close()
        seg = self.archive.segments / self.name
        if seg.exists():
            raise ArchiveError(f"{seg} exists; a segment is never overwritten")
        os.replace(self.tmp, seg)
        digest = self.h.hexdigest()
        sidecar_tmp = self.archive.segments / f".{self.name}.sha256.tmp"
        _fsync_write(sidecar_tmp, f"{digest}  {self.name}\n".encode())
        os.replace(sidecar_tmp, self.archive.segments / f"{self.name}.sha256")
        _fsync_dir(self.archive.segments)
        # read the committed file back before the index names it
        if _file_sha256(seg) != digest:
            raise ArchiveError(f"{seg}: the bytes read back differ from the bytes written")
        for fr in self.frames:
            self.archive.read_frame(fr)
        lines = "".join(json.dumps(fr.to_json(), sort_keys=True) + "\n" for fr in self.frames)
        index_tmp = self.archive.index_dir / f".{self.name}.jsonl.tmp"
        _fsync_write(index_tmp, lines.encode())
        os.replace(index_tmp, self.archive.index_dir / f"{self.name}.jsonl")
        _fsync_dir(self.archive.index_dir)
        return self.frames

    def abandon(self) -> None:
        """Close the temporary file and leave it (a later run never reads it; it is listed by ``archive``)."""
        try:
            self.f.close()
        except OSError:
            pass


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    with open(path, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ArchiveError(f"another archive run holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / _GB


def archived(spool_root: str | Path) -> dict[str, dict[str, Any]]:
    """The spool's archived batches (``<spool>/archived.jsonl``): batch -> {archive, streams: {stream: frame}}."""
    path = Path(spool_root) / "archived.jsonl"
    out: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if entry.get("batch") and entry.get("archive"):
                    out[entry["batch"]] = entry
    return out


def _note_archived(spool_root: Path, archive: Archive, frames: list[Frame]) -> None:
    by_batch: dict[str, dict[str, Any]] = {}
    for fr in frames:
        e = by_batch.setdefault(fr.batch, {"batch": fr.batch, "archive": str(archive.root.resolve()), "streams": {}})
        e["streams"][fr.stream] = {"segment": fr.segment, "offset": fr.offset, "length": fr.length}
    if not by_batch:
        return
    with open(spool_root / "archived.jsonl", "a", encoding="utf-8") as f:
        f.write("".join(json.dumps(e, sort_keys=True) + "\n" for e in by_batch.values()))
        f.flush()
        os.fsync(f.fileno())


def archive_spool(spool_root: str | Path, to: str | Path, *, name: Optional[str] = None, min_age_s: float = 3600.0,
                  floor_gb: float = 300.0, segment_gb: float = 4.0, level: int = 10,
                  max_batches: Optional[int] = None) -> dict[str, Any]:
    """Copy the spool's sealed batches that no segment holds yet into new segments under ``<to>/<name>``, one segment
    per UTC day and stream per run (a new one past ``segment_gb``), and note each in ``<spool>/archived.jsonl``. The
    spool's copies stay. Stops before a segment that would leave the archive's filesystem under ``floor_gb`` free.
    Idempotent: a batch any index line holds is passed over. Returns a summary."""
    spool_root = Path(spool_root)
    if not (spool_root / "index.jsonl").exists() and not (spool_root / "records").is_dir():
        raise ArchiveError(f"{spool_root} is not a sign spool (no <spool>/index.jsonl or <spool>/records/)")
    archive = Archive(Path(to) / (name or spool_name(spool_root)))
    archive.segments.mkdir(parents=True, exist_ok=True)
    archive.index_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"archive": str(archive.root), "segments": [], "batches": 0, "bytes": 0,
                               "stopped": None}
    with _locked(spool_root / "archive.lock"):
        done = archive.batches()
        noted = archived(spool_root)
        # a run that crashed after its index and before the spool's note: note those batches now
        missing = [fr for b, streams in done.items() if b not in noted for fr in streams.values()]
        _note_archived(spool_root, archive, missing)
        summary["renoted"] = len({fr.batch for fr in missing})
        stale = sorted(p.name for p in archive.segments.glob(".*.tmp"))
        if stale:
            summary["stale_tmp"] = stale
            logger.warning("sign archive: %d temporary files from an earlier run in %s (left in place): %s",
                           len(stale), archive.segments, stale)
        batches, skipped = _sealed_batches(spool_root, min_age_s)
        todo = [b for b in batches if b.name not in done]
        if max_batches is not None:
            todo = todo[:max_batches]
        summary.update(already=len(batches) - len([b for b in batches if b.name not in done]), **skipped)
        run = f"{time.time_ns():020d}-{os.getpid()}"
        writers: dict[tuple[str, str], _SegmentWriter] = {}
        rolls: dict[tuple[str, str], int] = {}

        def close(key: tuple[str, str]) -> None:
            w = writers.pop(key)
            frames = w.commit()
            _note_archived(spool_root, archive, frames)
            summary["segments"].append({"segment": w.name, "frames": len(frames), "bytes": w.size})
            summary["bytes"] += w.size

        try:
            for b in todo:
                local = sum(p.stat().st_size for _, p in b.members)
                if _free_gb(archive.root) - local / _GB < floor_gb:
                    summary["stopped"] = (f"{archive.root}'s filesystem would fall under {floor_gb:g} GB free; "
                                          "sealing stopped")
                    logger.warning("sign archive: %s", summary["stopped"])
                    break
                day = batch_day(b.name, b.dir.stat().st_mtime)
                rows = _rows_of(b)
                for stream in STREAMS:
                    members = [(rel, p) for rel, p in b.members if stream_of(rel) == stream]
                    if not members:
                        continue
                    key = (day, stream)
                    w = writers.get(key)
                    if w is not None and w.size >= segment_gb * _GB:
                        close(key)
                        w = None
                    if w is None:
                        rolls[key] = rolls.get(key, 0) + 1
                        seg = f"{day}-{stream}-{run}-{rolls[key]:03d}.tar.zst"
                        w = writers[key] = _SegmentWriter(archive, seg, str(spool_root))
                    blob, meta = _tar_frame(b, members, level)
                    w.add(blob, Frame(batch=b.name, stream=stream, segment=w.name, offset=0, length=len(blob),
                                      sha256=_sha256(blob), members=meta, rows=rows if stream == "main" else [],
                                      source_dir=str(b.dir), spool=str(spool_root)))
                summary["batches"] += 1
            for key in list(writers):
                close(key)
        except BaseException:
            for w in writers.values():
                w.abandon()
            raise
    archive.frames(reload=True)
    return summary


def drop_local(spool_root: str | Path, *, batches: Optional[list[str]] = None,
               dry_run: bool = False) -> dict[str, Any]:
    """Remove the spool's own copy of archived batches (all of them, or ``batches``): each only after every file in
    its directory was read back from the archive with the same bytes. Never runs by itself. ``<spool>/dropped.jsonl``
    lists each removal, and ``<spool>/ARCHIVED.json`` says where the dropped batches went and how to get one back (a
    pointer for anything that cites the spool's path). Returns counts and the batches refused (with why)."""
    spool_root = Path(spool_root)
    notes = archived(spool_root)
    index: dict[str, Path] = {}
    if (spool_root / "index.jsonl").exists():
        for line in (spool_root / "index.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                index[str(e["batch"])] = Path(e["dir"])
    out: dict[str, Any] = {"dropped": 0, "bytes": 0, "refused": {}, "dry_run": dry_run}
    wanted = batches if batches is not None else sorted(notes)
    with _locked(spool_root / "archive.lock"):
        archives: dict[str, Archive] = {}
        for name in wanted:
            note = notes.get(name)
            bdir = index.get(name)
            if note is None:
                out["refused"][name] = "not archived"
                continue
            if bdir is None or not bdir.is_dir():
                continue  # already gone
            if bdir.is_symlink() or bdir.name != name:
                out["refused"][name] = f"{bdir} is not the batch's own directory"
                continue
            arch = archives.setdefault(note["archive"], Archive(note["archive"]))
            try:
                held = arch.read_batch(name, streams=STREAMS)
                local = {f"{name}/{p.relative_to(bdir).as_posix()}": p for p in bdir.rglob("*") if p.is_file()}
                if set(local) != set(held):
                    raise ArchiveError(f"local files {sorted(set(local) ^ set(held))} differ from the archive's")
                for key, p in local.items():
                    if p.read_bytes() != held[key]:
                        raise ArchiveError(f"{key} differs from the archive's bytes")
            except (ArchiveError, OSError) as exc:
                out["refused"][name] = str(exc)
                continue
            size = sum(p.stat().st_size for p in local.values())
            if not dry_run:
                shutil.rmtree(bdir)
                with open(spool_root / "dropped.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"batch": name, "dir": str(bdir), "bytes": size, "archive": note["archive"],
                                        "time": time.time()}) + "\n")
            out["dropped"] += 1
            out["bytes"] += size
        if out["dropped"] and not dry_run:
            _write_pointer(spool_root, sorted({notes[n]["archive"] for n in wanted if n in notes}))
    return out


def _write_pointer(spool_root: Path, archives: list[str]) -> None:
    """``<spool>/ARCHIVED.json``: the archives holding the spool's dropped batches, how to read or extract one, and
    the totals ``<spool>/dropped.jsonl`` records (rewritten after each drop)."""
    dropped = [json.loads(x) for x in (spool_root / "dropped.jsonl").read_text(encoding="utf-8").splitlines()
               if x.strip()]
    prior = spool_root / "ARCHIVED.json"
    known: list[str] = []
    if prior.exists():
        try:
            known = list(json.loads(prior.read_text()).get("archives", []))
        except ValueError:
            known = []
    archives = sorted(set(known) | set(archives))
    pointer = {
        "format": ARCHIVE_FORMAT,
        "what": "signed batches of this spool were archived and their copies here removed; records/, plans/, "
                "index.jsonl and state.json stay, and find (and so /replay) reads a removed batch from the archive",
        "archives": archives,
        "extract": [f"tools/run_py.sh -m kotodama.serve.sign_archive extract --archive {a} --batch <batch> --to <dir> "
                    "[--raw]" for a in archives],
        "reader": "kotodama.serve.sign_archive.Archive(<archive>): locate(sign_id), read_batch(batch), "
                  "extract_batch(batch, dest)",
        "batches_dropped": len({d["batch"] for d in dropped}),
        "bytes_dropped": sum(int(d.get("bytes", 0)) for d in dropped),
        "last_drop_utc": datetime.fromtimestamp(max(float(d.get("time", 0)) for d in dropped),
                                                tz=timezone.utc).isoformat(timespec="seconds"),
    }
    tmp = spool_root / f".ARCHIVED.{os.getpid()}.json"
    tmp.write_text(json.dumps(pointer, indent=1) + "\n")
    os.replace(tmp, prior)


def restore_for_spool(spool_root: str | Path, batch: str) -> Optional[Path]:
    """A local directory holding ``batch``'s main-stream files: the spool's own when present, else the archive's
    extracted under ``<spool>/restored/`` (None when the batch is in neither)."""
    spool_root = Path(spool_root)
    note = archived(spool_root).get(batch)
    if note is None:
        return None
    dest = spool_root / "restored"
    if (dest / batch / "rows.jsonl").exists():
        return dest / batch
    return Archive(note["archive"]).extract_batch(batch, dest)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Seal a sign spool's signed batches into compressed segments on "
                                             "long-term storage, and read them back")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_arch = sub.add_parser("archive", help="copy sealed batches into new segments (the spool's copies stay)")
    a_arch.add_argument("--spool", required=True)
    a_arch.add_argument("--to", required=True, help="the archive root; the spool's archive is <to>/<name>")
    a_arch.add_argument("--name", default=None, help="the spool's archive name (default: its path, flattened)")
    a_arch.add_argument("--min_age_s", type=float, default=3600.0, help="archive batches at least this old")
    a_arch.add_argument("--floor_gb", type=float, default=300.0,
                        help="stop while the archive's filesystem has less than this free")
    a_arch.add_argument("--segment_gb", type=float, default=4.0, help="start a new segment past this size")
    a_arch.add_argument("--level", type=int, default=10, help="zstd level")
    a_arch.add_argument("--max_batches", type=int, default=None)
    a_drop = sub.add_parser("drop_local", help="remove the spool's copy of batches the archive holds byte for byte")
    a_drop.add_argument("--spool", required=True)
    a_drop.add_argument("--batch", action="append", default=None, help="only these batches (repeatable)")
    a_drop.add_argument("--dry_run", action="store_true")
    a_ext = sub.add_parser("extract", help="write a batch back to the plain layout <to>/<batch>/<file>")
    a_ext.add_argument("--archive", required=True)
    a_ext.add_argument("--batch", required=True, action="append")
    a_ext.add_argument("--to", required=True)
    a_ext.add_argument("--raw", action="store_true", help="also the raw audit captures")
    a_ver = sub.add_parser("verify", help="re-read every segment and frame against the index")
    a_ver.add_argument("--archive", required=True)
    a_loc = sub.add_parser("locate", help="where a sign_id's row is archived")
    a_loc.add_argument("--archive", required=True)
    a_loc.add_argument("--sign_id", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        if a.cmd == "archive":
            try:
                os.nice(10)
            except OSError:
                pass
            out: Any = archive_spool(a.spool, a.to, name=a.name, min_age_s=a.min_age_s, floor_gb=a.floor_gb,
                                     segment_gb=a.segment_gb, level=a.level, max_batches=a.max_batches)
            print(json.dumps(out, indent=1), flush=True)
            return 3 if out["stopped"] else 0
        if a.cmd == "drop_local":
            out = drop_local(a.spool, batches=a.batch, dry_run=a.dry_run)
            print(json.dumps(out, indent=1), flush=True)
            return 1 if out["refused"] else 0
        arch = Archive(a.archive)
        if a.cmd == "extract":
            streams = STREAMS if a.raw else ("main",)
            for b in a.batch:
                print(arch.extract_batch(b, a.to, streams=streams), flush=True)
            return 0
        if a.cmd == "verify":
            out = arch.verify()
            print(json.dumps(out, indent=1), flush=True)
            return 1 if out["failures"] else 0
        hit = arch.locate(a.sign_id)
        if hit is None:
            print(json.dumps(None))
            return 1
        fr, file, idx = hit
        print(json.dumps({"segment": fr.segment, "batch": fr.batch, "file": file, "index": idx,
                          "offset": fr.offset, "length": fr.length}))
        return 0
    except ArchiveError as exc:
        logger.error("sign archive: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
