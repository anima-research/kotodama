"""The sign spool: records of generated rows waiting to be signed, and the signatures written for them.

A server started with ``--sign_spool DIR`` appends a ``SignRecord`` (kotodama.model.signer) for every candidate of a
request that sets ``sign``; with a reducer configured it also signs the spool itself whenever it has nothing else to
do, one batch at a time. The same drain runs as a job of its own (this module's command line), for a spool whose
server never idles or was started without a reducer. Layout:

    <spool>/records/<writer>.jsonl   one record per line, appended by each writer (a server process)
    <spool>/plans/<digest>.npz       the lever plans records name (kotodama.model.signer.save_plans)
    <spool>/signed/<batch>/          per signed batch: kotodama.model.signer.write_results's files and the engine
                                     identity it was signed under (<batch>/engine.json); batches sort in drain order
    <spool>/index.jsonl              one line per signed batch: its name, directory, host, and row counts
    <spool>/state.json               per records file, the byte offset signed (or skipped) up to
    <spool>/archived.jsonl           per batch copied to a sign archive, where (kotodama.serve.sign_archive); ``find``
                                     reads a batch whose own directory is gone from there, through <spool>/restored/

Rows generated elsewhere (or a served row's tokens under another lever: worn vs unworn on the same tokens) enter a
spool through ``import_rows`` (the command line's --import_rows): one JSON object per line, ``{"id", "ids",
"prompt_length", "injection" (the server's request field, or null), "steer_span" (default "decode"), "meta"}``; an
injection becomes its lever plan by the server's own conversion, so an imported row and a served one with the same
injection name the same plan.

Records are small and belong with the run; signed batches are bulk (~19 KB per row on the 1B), so a drain can write
them elsewhere (``signed_dir``, e.g. node-local storage) and the index says where each batch went.

A drain holds an exclusive flock on ``<spool>/drain.lock`` while it signs a batch, so a server and a job can drain one
spool together. A batch is written (into a temporary directory, then renamed) before the state moves past it, so a
crash re-signs at most the batch in flight, under the same request ids.

    tools/run_py.sh -m kotodama.serve.sign --spool <dir> --checkpoint <ckpt> --model_size 1b-tsumugi \\
        --reducer <package.module:attr> --reducer_kwargs '<json>' --layers 0,18,25,... [--pins <table>]
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from kotodama.model.levers import LeverPlan
from kotodama.model.signer import RecordSigner, SignRecord, load_plans, save_plans, write_results

logger = logging.getLogger(__name__)


class SignSpool:
    """A spool directory (see the module docstring)."""

    def __init__(self, root: str | Path, signed_dir: Optional[str | Path] = None) -> None:
        self.root = Path(root)
        self.signed_dir = Path(signed_dir) if signed_dir is not None else self.root / "signed"
        for sub in ("records", "plans"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.signed_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._plans: dict[str, LeverPlan] = {}

    # ── writing ──
    def add(self, writer: str, records: list[SignRecord], plans: Optional[list[LeverPlan]] = None) -> None:
        """Append ``records`` to ``writer``'s records file, after saving any plan the spool does not hold yet."""
        for plan in plans or ():
            path = self.root / "plans" / f"{plan.digest()}.npz"
            if not path.exists():
                tmp = path.with_name(f".{path.stem}.{os.getpid()}.npz")
                save_plans(tmp, [plan])
                os.replace(tmp, path)
        lines = "".join(json.dumps(r.to_json(), separators=(",", ":")) + "\n" for r in records)
        with self._lock, open(self.root / "records" / f"{writer}.jsonl", "a", encoding="utf-8") as f:
            f.write(lines)
            f.flush()

    # ── draining ──
    def _state(self) -> dict[str, int]:
        path = self.root / "state.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def _save_state(self, state: dict[str, int]) -> None:
        tmp = self.root / f".state.{os.getpid()}.json"
        tmp.write_text(json.dumps(state, sort_keys=True))
        os.replace(tmp, self.root / "state.json")

    def has_pending(self) -> bool:
        """Whether any records file has bytes past its signed offset (a stat per file)."""
        state = self._state()
        return any(path.stat().st_size > state.get(path.name, 0)
                   for path in (self.root / "records").glob("*.jsonl"))

    def pending(self) -> int:
        """Complete record lines past each file's signed offset."""
        state = self._state()
        n = 0
        for path in sorted((self.root / "records").glob("*.jsonl")):
            with open(path, "rb") as f:
                f.seek(state.get(path.name, 0))
                n += sum(1 for line in f if line.endswith(b"\n"))
        return n

    def plans(self) -> dict[str, LeverPlan]:
        """Every plan in the spool, by digest (loaded once each)."""
        for path in (self.root / "plans").glob("*.npz"):
            if path.stem not in self._plans:
                self._plans.update(load_plans(path))
        return dict(self._plans)

    def _take(self, max_rows: int) -> tuple[list[SignRecord], dict[str, int]]:
        """Up to ``max_rows`` unsigned records, oldest file first, and the state after them."""
        state = self._state()
        out: list[SignRecord] = []
        for path in sorted((self.root / "records").glob("*.jsonl")):
            done = state.get(path.name, 0)
            with open(path, "rb") as f:
                f.seek(done)
                for line in f:
                    if len(out) >= max_rows or not line.endswith(b"\n"):  # a line still being written waits
                        break
                    try:
                        out.append(SignRecord.from_json(json.loads(line)))
                    except (ValueError, TypeError, UnicodeDecodeError) as exc:
                        logger.warning("sign spool: %s at byte %d is not a record (%s); skipped", path.name, done,
                                       exc)
                    done += len(line)
            state[path.name] = done
            if len(out) >= max_rows:
                break
        return out, state

    @contextmanager
    def _drain_lock(self) -> Iterator[bool]:
        with open(self.root / "drain.lock", "a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def drain_once(self, signer: RecordSigner, max_rows: Optional[int] = None) -> Optional[dict[str, Any]]:
        """Sign one batch of pending records (default: the signer's width) and write it. Returns the batch's summary,
        or None when nothing was pending or another drain holds the spool."""
        with self._drain_lock() as mine:
            if not mine:
                return None
            records, state = self._take(max_rows or signer.dec.rows)
            if not records and state == self._state():
                return None
            signer.plans.update(self.plans())
            t0 = time.perf_counter()
            results = signer.sign(records) if records else []
            summary = self._write_batch(results, signer.identity)
            self._save_state(state)
            summary.update(seconds=round(time.perf_counter() - t0, 2),
                           signer_timing={k: round(v, 3) for k, v in signer.timing.items()})
            logger.info("sign spool: %s", summary)
            return summary

    def _write_batch(self, results: list, identity: Any, inflight: bool = False) -> dict[str, Any]:
        """One signed batch into the signed dir (a temporary directory, then renamed) and its index line."""
        name = f"{time.time_ns():020d}-{os.getpid()}"  # sorts in drain order
        tmp = self.signed_dir / f".{name}"
        summary = write_results(tmp, results)
        (tmp / "engine.json").write_text(json.dumps(identity, sort_keys=True, default=str))
        final = self.signed_dir / name
        os.replace(tmp, final)
        line = {"batch": name, "dir": str(final.resolve()), "host": socket.gethostname(),
                "signed": summary["signed"], "skipped": summary["skipped"]}
        if inflight:
            line["inflight"] = True
        with open(self.root / "index.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
        summary["batch"] = name
        return summary

    def add_signed(self, writer: str, results: list, identity: Any,
                   plans: Optional[list[LeverPlan]] = None) -> Optional[dict[str, Any]]:
        """Records signed as they were generated (in-flight signing; ``results``: kotodama.model.signer.SignResult,
        each holding its record): appends the records to ``<writer>.inflight.jsonl`` and writes their signed batch,
        then moves that file's signed offset past them, all under the drain lock, so a drain never signs them again.
        Returns the batch's summary, or None when another drain holds the spool (nothing was written; call again)."""
        with self._drain_lock() as mine:
            if not mine:
                return None
            if not results:
                return {"signed": 0, "skipped": 0, "batch": None}
            path = self.root / "records" / f"{writer}.inflight.jsonl"
            before = path.stat().st_size if path.exists() else 0
            # a failure between the append and the state below leaves the records past the signed offset: a drain
            # then signs them (the records file stays the one account of what was generated)
            self.add(writer + ".inflight", [r.record for r in results], plans=plans)
            summary = self._write_batch(results, identity, inflight=True)
            state = self._state()
            if state.get(path.name, 0) == before:
                state[path.name] = path.stat().st_size
                self._save_state(state)
            else:
                logger.warning("sign spool: %s holds records an earlier batch left unsigned; a drain signs them "
                               "(and these again)", path.name)
            return summary


def find(spool: SignSpool, sign_id: str) -> tuple[Optional[SignRecord], Optional[dict[str, Any]]]:
    """``sign_id``'s record and its signed row in ``spool`` (each None when absent): the row is its line in
    ``<batch>/rows.jsonl``, with ``batch_dir`` added; the newest batch wins (a drain that re-signed after a crash).
    Scans the files (a replay is rare); a line is parsed only when it holds the id."""
    needle = json.dumps(sign_id)
    record = None
    for path in sorted((spool.root / "records").glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if needle in line and line.endswith("\n"):
                    rec = SignRecord.from_json(json.loads(line))
                    if rec.request_id == sign_id:
                        record = rec
    row = None
    index = spool.root / "index.jsonl"
    if index.exists():
        archived_batch = _archived_lookup(spool, sign_id)
        for entry in reversed(index.read_text(encoding="utf-8").splitlines()):
            entry = json.loads(entry)
            batch = Path(entry["dir"])
            rows = batch / "rows.jsonl"
            if not rows.exists():
                # a batch whose spool copy was dropped after archiving: read from the archive's copy
                restored = archived_batch(str(entry["batch"]))
                if restored is None:
                    continue
                batch, rows = restored, restored / "rows.jsonl"
            for line in rows.read_text(encoding="utf-8").splitlines():
                if needle in line:
                    cand = json.loads(line)
                    if cand.get("sign_id") == sign_id:
                        row = {**cand, "batch_dir": str(batch)}
            if row is not None:
                break
    return record, row


def _archived_lookup(spool: SignSpool, sign_id: str):
    """For ``find``: a function from a batch name to a local directory with its archived main-stream files, or None
    when the archive's index does not list ``sign_id`` in that batch (kotodama.serve.sign_archive)."""
    from kotodama.serve import sign_archive

    notes = sign_archive.archived(spool.root)
    archives: dict[str, sign_archive.Archive] = {}

    def lookup(batch: str) -> Optional[Path]:
        note = notes.get(batch)
        if note is None:
            return None
        arch = archives.setdefault(note["archive"], sign_archive.Archive(note["archive"]))
        main = arch.batches().get(batch, {}).get("main")
        if main is None or not any(r[0] == sign_id for r in main.rows):
            return None
        return sign_archive.restore_for_spool(spool.root, batch)

    return lookup


def import_rows(spool: SignSpool, path: str | Path, writer: str, *, weights: str, engine: Any) -> int:
    """Append the rows of JSONL ``path`` (see the module docstring) to ``spool`` as writer ``writer``'s records, each
    naming ``weights`` (the signing model's digest) and, with an injection, the lever plan the server makes of it on
    ``engine`` (a DecodeEngine with lever tables). Returns the rows imported. Refuses a writer the spool already has
    (an import is appended once) and an injection a drain cannot rebuild (rule N reads per-layer partial norms the
    drain does not load)."""
    from kotodama.serve.server import InjectionSpec
    from kotodama.serve.steering import act_injection_to_plan, injection_to_plan

    if (spool.root / "records" / f"{writer}.jsonl").exists():
        raise FileExistsError(f"the spool already has records from writer {writer!r}: an import is appended once")
    cfg = engine.config
    records: list[SignRecord] = []
    plans: dict[str, LeverPlan] = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            where = f"{path}:{n}"
            row = json.loads(line)
            ids = [int(t) for t in row["ids"]]
            plen = int(row["prompt_length"])
            if not 0 < plen < len(ids):
                raise ValueError(f"{where}: prompt_length {plen} must leave at least one generated id of {len(ids)}")
            plan = None
            if row.get("injection") is not None:
                if engine.levers is None:
                    raise ValueError(f"{where}: an injection needs a drain with lever tables (--levers)")
                inj = InjectionSpec(**row["injection"])
                span = row.get("steer_span", "decode")
                if inj.kind == "act":
                    plan = act_injection_to_plan(inj.sites, inj.neurons, inj.mu, inj.alpha, inj.vectors, span,
                                                 cfg.num_layers, cfg.intermediate_size)
                elif inj.rule == "N":
                    raise ValueError(f"{where}: injection rule N reads per-layer partial norms a drain does not load")
                else:
                    plan = injection_to_plan(inj.sites, inj.vectors, inj.alpha, inj.rule, inj.kind, span, None,
                                             list(engine._boundaries), cfg.num_layers, cfg.hidden_size)
                plans[plan.digest()] = plan
            records.append(SignRecord(request_id=str(row["id"]), prompt_ids=tuple(ids[:plen]),
                                      generated_ids=tuple(ids[plen:]), weights=weights,
                                      lever=plan.digest() if plan is not None else None,
                                      meta={**(row.get("meta") or {}), "imported_by": writer}))
    spool.add(writer, records, plans=list(plans.values()))
    return len(records)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Drain a sign spool: sign its records through a pinned signing decoder "
                                             "and a SignatureReducer")
    ap.add_argument("--spool", required=True)
    ap.add_argument("--signed_dir", default=None,
                    help="where signed batches go (default <spool>/signed; the spool's index.jsonl lists each)")
    ap.add_argument("--audit_every", type=int, default=100,
                    help="keep the raw signed surfaces of 1 in N rows for an offline re-reduction check (0: none)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_size", default="1b-tsumugi")
    ap.add_argument("--reducer", required=True, help="package.module:attr (a SignatureReducer class or factory)")
    ap.add_argument("--reducer_kwargs", default="{}", help="JSON keyword arguments for the reducer")
    ap.add_argument("--layers", required=True, help="the sampled effective layers, comma-separated")
    ap.add_argument("--tail", type=int, default=256, help="sign positions per row (longer spans are skipped)")
    ap.add_argument("--width", type=int, default=0, help="head-mean row columns (0: max_seq_len)")
    ap.add_argument("--rows", type=int, default=128, help="the signing decoder's width (the lane's canonical 128)")
    ap.add_argument("--groups", type=int, default=16)
    ap.add_argument("--group_rows", type=int, default=32)
    ap.add_argument("--tail_tokens", type=int, default=0,
                    help="the signing decoder's per-row tail (0: the sign tail + 2); a server's in-flight signer "
                         "has its widest bucket's geometry (--groups, --group_rows, --tail_tokens = the server's "
                         "--cascade_tail_tokens)")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--levers", type=int, default=4, help="lever slots (records' levers bind them per batch)")
    ap.add_argument("--compile_mode", default="default")
    ap.add_argument("--pins", default=None, help="kernel pin table: replayed when it exists, else recorded")
    ap.add_argument("--rope_fp32", action="store_true")
    ap.add_argument("--emulate_casts", action="store_true",
                    help="compiled regions round intermediates as eager does (the generating server's setting: a "
                         "signing decoder then decodes what an unsigned one does)")
    ap.add_argument("--import_rows", default=None,
                    help="JSONL of rows to append to the spool before draining (see the module docstring)")
    ap.add_argument("--import_writer", default=None, help="the writer name the imported records go under")
    ap.add_argument("--once", action="store_true", help="drain what is pending and exit (default: keep polling)")
    ap.add_argument("--poll_s", type=float, default=30.0)
    a = ap.parse_args(argv)
    # the reducers' deterministic algorithms need cuBLAS's fixed workspace, set before CUDA starts
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import torch

    from kotodama.model import kernel_pins
    from kotodama.model.decode_engine import DecodeEngine
    from kotodama.model.signing import load_reducer
    from kotodama.serve import server

    pins = None
    if a.pins:
        pins = kernel_pins.install(a.pins, "replay" if Path(a.pins).exists() else "record")
    if a.rope_fp32:
        server._rope_fp32_override = True
    model, _ = server.load_model(a.checkpoint, "cuda", mode="base", model_size=a.model_size, engine="reference",
                                 max_seq_len=a.max_seq_len)
    eng = DecodeEngine(model, max_seq_len=a.max_seq_len, levers=a.levers, emulate_casts=a.emulate_casts)
    eng.compile_step(mode=a.compile_mode, regional=True)
    reducer = load_reducer(a.reducer, **json.loads(a.reducer_kwargs))
    signer = RecordSigner.build(eng, reducer, layers=[int(x) for x in a.layers.split(",")], tail=a.tail,
                                width=a.width or a.max_seq_len, rows=a.rows, groups=a.groups,
                                group_rows=a.group_rows, tail_tokens=a.tail_tokens, compile_mode=a.compile_mode,
                                pins=pins,
                                audit_every=a.audit_every)
    if pins is not None and pins.mode == "record":
        signer.warm()
        logger.info("kernel pins recorded: %s (%d kernels, %s)", pins.save(), len(pins.table), pins.digest[:12])
    spool = SignSpool(a.spool, signed_dir=a.signed_dir)
    if a.import_rows:
        writer = a.import_writer or Path(a.import_rows).stem
        n = import_rows(spool, a.import_rows, writer, weights=signer.weights, engine=eng)
        logger.info("sign spool: imported %d rows from %s as writer %s", n, a.import_rows, writer)
    with torch.inference_mode():
        while True:
            summary = spool.drain_once(signer)
            if summary is not None:
                if pins is not None and pins.mode == "record":
                    # a batch can launch kernels the warm-up did not (its prompts' lengths): keep them in the table
                    pins.save()
                print(json.dumps(summary), flush=True)
                continue
            if a.once:
                return 0
            print(f"sign spool: {spool.pending()} pending; polling every {a.poll_s:.0f}s", flush=True)
            time.sleep(a.poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
