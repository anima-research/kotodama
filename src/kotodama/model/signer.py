"""Signing finished generations by replaying them through the decode engine: the sign path a server runs in-process
after a batch, and a signing job runs over stored records.

A generated row is described by a small ``SignRecord``: its full prompt ids, its generated ids, the lever it decoded
under (a LeverPlan digest; the plans travel beside the records, ``save_plans`` / ``load_plans``), the weights digest of
the model that generated it, and the serving engine's identity. ``RecordSigner`` re-admits each record's prompt into a
signing ``CascadeDecoder`` (kotodama.model.multi_decode) at its canonical width, teacher-forces the generated ids
through the decode step (``score``: a scored span is signed exactly as a generated one) and hands the signed rows to a
``SignatureReducer`` (kotodama.model.signing). Rows of a batch are independent: a record signs to the same bytes alone
or beside any batchmates, so records are packed by prompt and the unused rows of a batch score filler.

A record is skipped, with the reason logged, when the signer cannot reproduce or fully hold it: other weights, other
broadcast steering or adapter, a lever plan it was not given, fewer than two generated ids (no fed position), a span
longer than the sign tail, or (with attention statistics) a sequence wider than the head-mean rows. A batch the reducer
refuses is retried row by row, and the rows it still refuses are skipped with the reducer's error.
"""

from __future__ import annotations

import dataclasses
import hashlib
from contextlib import contextmanager
import json
import re
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np
import torch

from kotodama.model import signing
from kotodama.model.levers import KINDS, LeverPlan

logger = logging.getLogger(__name__)

RECORD_FORMAT = "kotodama-sign-record/1"


@dataclass(frozen=True)
class SignRecord:
    """One generated row, as much as signing it needs. ``prompt_ids``: every token the row was conditioned on (system
    and earlier turns included); ``generated_ids``: the tokens it sampled, in order (a stop token it sampled
    included); ``weights``: ``signing.weights_digest`` of the generating model; ``lever``: the LeverPlan digest it
    decoded under (None: unsteered) and ``dose`` the dose that plan was built at (metadata); ``steering``: the
    broadcast steering bank identity (None: none); ``adapter``: the merged adapter identity (None: none);
    ``engine``: the serving engine's identity (``signing.engine_identity``; None: unknown); ``meta``: anything else
    (seed, law, the caller's row id)."""

    request_id: str
    prompt_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    weights: str
    lever: Optional[str] = None
    dose: Optional[float] = None
    steering: Optional[str] = None
    adapter: Optional[str] = None
    engine: Optional[Mapping[str, Any]] = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("a sign record needs a non-empty request_id")
        for name in ("prompt_ids", "generated_ids"):
            ids = tuple(int(t) for t in getattr(self, name))
            if any(t < 0 for t in ids):
                raise ValueError(f"{self.request_id}: {name} holds a negative token id")
            object.__setattr__(self, name, ids)
        if not self.prompt_ids:
            raise ValueError(f"{self.request_id}: prompt_ids is empty")
        if not isinstance(self.weights, str) or not self.weights:
            raise ValueError(f"{self.request_id}: a sign record names its weights digest")
        if self.dose is not None:
            object.__setattr__(self, "dose", float(self.dose))
        object.__setattr__(self, "meta", dict(self.meta))
        if self.engine is not None:
            object.__setattr__(self, "engine", dict(self.engine))

    def to_json(self) -> dict[str, Any]:
        out = {"format": RECORD_FORMAT}
        out.update({f.name: getattr(self, f.name) for f in dataclasses.fields(self)})
        out["prompt_ids"], out["generated_ids"] = list(self.prompt_ids), list(self.generated_ids)
        return out

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "SignRecord":
        fmt = obj.get("format")
        if fmt != RECORD_FORMAT:
            raise ValueError(f"not a {RECORD_FORMAT} record (format {fmt!r})")
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(obj) - names - {"format"}
        if unknown:
            raise ValueError(f"unknown sign record fields {sorted(unknown)}")
        return cls(**{k: v for k, v in obj.items() if k in names})


def write_records(path: str | Path, records: Iterable[SignRecord], append: bool = True) -> int:
    """Write records as JSON lines (appending by default). Returns how many were written."""
    n = 0
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_json(), separators=(",", ":")) + "\n")
            n += 1
    return n


def read_records(path: str | Path) -> Iterator[SignRecord]:
    """The records of a JSON-lines file, in order; a malformed line raises ValueError naming it."""
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield SignRecord.from_json(json.loads(line))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{i}: {exc}") from exc


def save_plans(path: str | Path, plans: Iterable[LeverPlan]) -> list[str]:
    """Lever plans as one npz, keyed by digest (``load_plans`` reads them back). Returns the digests."""
    arrays: dict[str, np.ndarray] = {}
    digests = []
    for plan in plans:
        d = plan.digest()
        digests.append(d)
        arrays[f"{d}/span"] = np.array(plan.span)
        for kind in (*KINDS, "act"):
            for layer, row in plan.rows(kind).items():
                arrays[f"{d}/{kind}/{layer}"] = row
    np.savez(path, **arrays)
    return digests


def load_plans(path: str | Path) -> dict[str, LeverPlan]:
    """The plans ``save_plans`` wrote, by digest; a plan whose content does not hash to its key raises."""
    rows: dict[str, dict[str, Any]] = {}
    with np.load(path, allow_pickle=False) as z:
        for key in z.files:
            d, _, rest = key.partition("/")
            entry = rows.setdefault(d, {"span": None, **{k: {} for k in (*KINDS, "act")}})
            if rest == "span":
                entry["span"] = str(z[key])
            else:
                kind, _, layer = rest.partition("/")
                entry[kind][int(layer)] = z[key]
    plans = {}
    for d, entry in rows.items():
        plan = LeverPlan(attn=entry["attn"], mlp=entry["mlp"], act=entry["act"], span=entry["span"])
        if plan.digest() != d:
            raise ValueError(f"{path}: the plan stored under {d} hashes to {plan.digest()}")
        plans[d] = plan
    return plans


@dataclass
class SignResult:
    """One record's outcome: its reduced signature, or why it was skipped. ``same_engine``: the record's serving
    identity equals the signer's (its signature is then the one its generation would have captured)."""

    record: SignRecord
    reduced: Optional[signing.ReducedRow] = None
    skipped: Optional[str] = None
    same_engine: bool = False
    raw: Optional[signing.SignedRow] = None  # the signed surfaces, for the rows the audit sink samples (on the CPU)

    @property
    def steps(self) -> int:
        """The positions the row's signature covers (its generated ids but the last, which was never fed)."""
        return max(0, len(self.record.generated_ids) - 1)


def audited(request_id: str, every: int) -> bool:
    """Whether the audit sink keeps ``request_id``'s raw surfaces (a fixed 1 in ``every`` sample by id; 0: none)."""
    if every <= 0:
        return False
    return int(hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:8], 16) % every == 0


@contextmanager
def deterministic_algorithms() -> Iterator[None]:
    """torch's deterministic algorithms on for the block, then the previous setting back. Reducers run under it (a
    signature lane's reductions refuse to run without it); the capture does not, since the setting is process-wide
    and can swap the kernels a decoder's pinned arithmetic was recorded with."""
    was, warn_only = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was, warn_only=warn_only)


def _canonical(x: Any) -> Any:
    """``x`` as it reads back from JSON (tuples become lists), for comparing identities."""
    return json.loads(json.dumps(x, sort_keys=True, default=str))


def trim(row: signing.SignedRow, n: int) -> signing.SignedRow:
    """``row``'s first ``n`` positions (a row that scored filler past its own span)."""
    if n >= row.positions:
        return row
    kept = min(n, row.kept)
    surf = {name: ([t[:kept] for t in x] if isinstance(x, list) else x[:kept]) for name, x in row.surfaces.items()}
    return dataclasses.replace(row, positions=n, kept=kept, surfaces=surf)


class RecordSigner:
    """Signs records on a signing decoder (a ``CascadeDecoder`` built with ``sign=``, compiled regionally, its
    engine's lever tables holding at least as many slots as a batch has distinct levers) and a reducer.

    ``plans``: the lever plans records may name, by digest; ``steering`` / ``adapter``: the broadcast steering bank
    and merged adapter the signer's engine runs (records must match); ``compile_mode``: what the decoder's regions
    were compiled under; ``pins`` (a kotodama.model.kernel_pins.PinSession, read at every batch) or ``pins_digest``:
    the kernel pin table they replay. Both go in the engine identity. ``audit_every``: keep the raw signed surfaces of
    a fixed 1 in N sample of rows (``audited``) for an offline re-reduction check (0: none). ``timing`` accumulates
    the seconds spent capturing (prefill + scored steps + export) and reducing."""

    def __init__(self, decoder: Any, reducer: signing.SignatureReducer, *, compile_mode: str,
                 pins_digest: Optional[str] = None, plans: Optional[Mapping[str, LeverPlan]] = None,
                 steering: Optional[str] = None, adapter: Optional[str] = None, pins: Any = None,
                 audit_every: int = 0) -> None:
        from kotodama.model.multi_decode import CascadeDecoder

        if not isinstance(decoder, CascadeDecoder) or decoder.sign is None:
            raise TypeError("RecordSigner needs a CascadeDecoder built with sign=")
        if not isinstance(reducer, signing.SignatureReducer):
            raise TypeError("the reducer does not implement SignatureReducer")
        self.dec = decoder
        self.reducer = reducer
        self.spec: signing.SignSpec = decoder.sign
        self.plans = dict(plans or {})
        self.steering, self.adapter = steering, adapter
        self.weights: str = decoder._sign_weights
        self.compile_mode = compile_mode
        self._pins, self._pins_digest = pins, pins_digest
        self.audit_every = int(audit_every)
        self.timing = {"capture_s": 0.0, "reduce_s": 0.0, "rows": 0}

    @property
    def identity(self) -> dict[str, Any]:
        """The engine identity the signed rows are captured under (the pin table's digest as it stands now)."""
        dec = self.dec
        digest = self._pins.digest if self._pins is not None else self._pins_digest
        return signing.engine_identity(
            compile_mode=self.compile_mode, width=dec.rows, weights=self.weights, pins_digest=digest,
            device=dec.engine.device,
            extra={"decoder": {"class": type(dec).__name__, "rows": dec.rows, "groups": dec.groups,
                               "group_rows": dec.group_rows, "tail_tokens": dec.tail_tokens,
                               "max_seq_len": dec.max_seq_len, "seeded": dec.seeded},
                   "sign": dataclasses.asdict(self.spec), "steering": self.steering, "adapter": self.adapter,
                   "emulate_casts": dec.engine.emulate_casts})

    @classmethod
    def build(cls, engine: Any, reducer: signing.SignatureReducer, *, layers: Sequence[int], tail: int, width: int,
              rows: int = 128, groups: int = 16, group_rows: int = 32, tail_tokens: int = 0,
              compile_mode: str = "default", backend: str = "inductor", **kw: Any) -> "RecordSigner":
        """A signer over ``engine``: a seeded signing CascadeDecoder of ``rows`` rows (depth path and attention
        statistics on; ``tail_tokens`` default: the sign tail + 2, room for its longest scored span), compiled
        regionally under ``compile_mode``."""
        from kotodama.model.multi_decode import CascadeDecoder

        spec = signing.SignSpec(tuple(layers), tail=tail, depth=True, attn=True, width=width)
        dec = CascadeDecoder(engine, rows, groups=groups, group_rows=group_rows,
                             tail_tokens=tail_tokens or tail + 2, seeded=True, sign=spec)
        dec.compile_step(mode=compile_mode, regional=True, backend=backend)
        return cls(dec, reducer, compile_mode=compile_mode, **kw)

    @torch.inference_mode()
    def warm(self, prompt_len: Optional[int] = None, span: int = 8, reduce: bool = False) -> None:
        """Run one batch of filler through the scoring step: compiles and, with kernel pins recording, records the
        kernels a signed batch launches. The default prompt is as long as the prefill's block buckets together, so its
        prefill runs every bucket (a long prompt's chunks launch kernels a short one does not). ``reduce``: also hand
        the batch's rows to the reducer once, its output discarded (a reducer's first batch compiles and allocates
        what later batches reuse); a reducer that fails here is logged, not raised."""
        from kotodama.model.decode_engine import SamplingParams

        dec, eng = self.dec, self.dec.engine
        if prompt_len is None:
            prompt_len = min(sum(eng.block_buckets), eng.max_seq_len - dec.tail_tokens)
        span = max(2, min(span, self.spec.tail + 1, dec.tail_tokens - 1))
        g = torch.Generator().manual_seed(0)
        prompt = torch.randint(3, eng.config.vocab_size, (1, prompt_len), generator=g).to(eng.device)
        dec.release(range(dec.rows))
        dec.admit(list(range(dec.rows)), eng.prefill(prompt), SamplingParams(temperature=1.0, repetition_penalty=1.0),
                  seeds=[0] * dec.rows)
        tokens = torch.randint(3, eng.config.vocab_size, (1, span), generator=g)
        dec.score(tokens.to(eng.device))
        if reduce:
            rows = [dec.signed(r, device=eng.device) for r in range(dec.rows)]
            ids = tuple(prompt[0].tolist()) + tuple(tokens[0].tolist())
            t0 = time.perf_counter()
            try:
                with deterministic_algorithms():
                    self.reducer.reduce(rows, engine=self.identity, token_ids=[ids] * len(rows))
                logger.info("sign: reducer warmed on a %d-row batch in %.2fs", len(rows), time.perf_counter() - t0)
            except Exception as exc:  # noqa: BLE001 - a warm-up must not keep a server or drain from starting
                logger.warning("sign: the reducer's warm-up batch failed (%s: %s)", type(exc).__name__, exc)
            del rows
        dec.release(range(dec.rows))

    def why_not(self, rec: SignRecord) -> Optional[str]:
        """Why ``rec`` cannot be signed here (None: it can)."""
        n = len(rec.generated_ids)
        total = len(rec.prompt_ids) + n
        if rec.weights != self.weights:
            return f"weights {rec.weights[:12]}... are not the signer's {self.weights[:12]}..."
        if rec.steering != self.steering:
            return f"steering {rec.steering!r} is not the signer's {self.steering!r}"
        if rec.adapter != self.adapter:
            return f"adapter {rec.adapter!r} is not the signer's {self.adapter!r}"
        if rec.lever is not None and rec.lever not in self.plans:
            return f"lever {rec.lever} has no plan"
        if rec.lever is not None and self.dec.engine.levers is None:
            return "the signer's engine has no lever tables"
        if n < 2:
            return f"{n} generated ids: no fed position to sign"
        if n - 1 > self.spec.tail:
            return f"span {n - 1} is longer than the sign tail {self.spec.tail}"
        if n >= self.dec.tail_tokens:
            return f"{n} scored tokens do not fit the decoder's {self.dec.tail_tokens}-token tails"
        if total >= self.dec.max_seq_len:
            return f"{total} tokens do not fit max_seq_len {self.dec.max_seq_len}"
        if self.spec.attn and total - 1 > self.spec.width:
            return f"{total - 1} attended columns exceed the head-mean rows' width {self.spec.width}"
        return None

    def _batches(self, keyed: list[tuple[tuple, list[int]]],
                 n_scored: Sequence[int]) -> Iterator[list[tuple[tuple, list[int]]]]:
        """Chunks of (key, record indices) per batch: at most ``rows`` rows and, with the filler rows' groups, at
        most ``groups`` groups and ``lever slots`` distinct levers; every row scores the batch's longest span (its
        own, then filler), so the longest prompt plus that span must fit ``max_seq_len``."""
        dec = self.dec
        m, rows, groups = dec.group_rows, dec.rows, dec.groups
        slots = dec.engine.levers.free if dec.engine.levers is not None else 0
        batch: list[tuple[tuple, list[int]]] = []
        used_rows = used_groups = 0
        levers: set[str] = set()
        longest_prompt = longest_span = 0

        def fits(k: int, lever: Optional[str], prompt: int, span: int) -> bool:
            g = used_groups + -(-k // m)
            filler = rows - used_rows - k
            return (used_rows + k <= rows and g + -(-filler // m) <= groups
                    and max(longest_prompt, prompt) + max(longest_span, span) < dec.max_seq_len
                    and (lever is None or lever in levers or len(levers) < slots))

        for key, idx in keyed:
            lever, prompt = key[1], len(key[0])
            while idx:
                k = min(len(idx), rows - used_rows, m * (groups - used_groups))
                while k > 0 and not fits(k, lever, prompt, max(n_scored[i] for i in idx[:k])):
                    k -= 1
                if k == 0:
                    if not batch:
                        raise RuntimeError("a single record does not fit an empty batch (groups/levers too few)")
                    yield batch
                    batch, used_rows, used_groups, levers = [], 0, 0, set()
                    longest_prompt = longest_span = 0
                    continue
                batch.append((key, idx[:k]))
                used_rows += k
                used_groups += -(-k // m)
                longest_prompt = max(longest_prompt, prompt)
                longest_span = max(longest_span, max(n_scored[i] for i in idx[:k]))
                if lever is not None:
                    levers.add(lever)
                idx = idx[k:]
        if batch:
            yield batch

    @torch.inference_mode()
    def sign(self, records: Sequence[SignRecord]) -> list[SignResult]:
        """Sign ``records``; one result per record, in order."""
        results = [SignResult(record=r) for r in records]
        keyed: dict[tuple, list[int]] = {}
        for i, rec in enumerate(records):
            reason = self.why_not(rec)
            if reason is not None:
                results[i].skipped = reason
                logger.warning("sign: skipped %s: %s", rec.request_id, reason)
                continue
            results[i].same_engine = rec.engine is not None and _canonical(rec.engine) == _canonical(self.identity)
            keyed.setdefault((rec.prompt_ids, rec.lever), []).append(i)
        for batch in self._batches(list(keyed.items()), [len(r.generated_ids) for r in records]):
            t0 = time.perf_counter()
            order, signed = self._capture(records, batch)
            for i, row in zip(order, signed):
                if audited(records[i].request_id, self.audit_every):
                    results[i].raw = _to_cpu(row)
            t1 = time.perf_counter()
            self._reduce(records, results, order, signed)
            self.timing["capture_s"] += t1 - t0
            self.timing["reduce_s"] += time.perf_counter() - t1
            self.timing["rows"] += len(order)
        dec = self.dec
        dec.release(range(dec.rows))
        return results

    @torch.inference_mode()
    def capture(self, rec: SignRecord) -> signing.SignedRow:
        """``rec``'s signed span, re-captured on this signer's decoder as ``sign`` captures it (no reducer; on the
        decoder's device). Raises ValueError when the signer cannot reproduce it (``why_not``)."""
        reason = self.why_not(rec)
        if reason is not None:
            raise ValueError(f"{rec.request_id}: {reason}")
        batch = next(self._batches([((rec.prompt_ids, rec.lever), [0])], [len(rec.generated_ids)]))
        _, signed = self._capture([rec], batch)
        self.dec.release(range(self.dec.rows))
        return signed[0]

    def _capture(self, records: Sequence[SignRecord],
                 batch: list[tuple[tuple, list[int]]]) -> tuple[list[int], list[signing.SignedRow]]:
        """One batch of records (``_batches``) admitted and scored: the record indices in row order and their signed
        spans, each trimmed to its own fed positions (synchronised)."""
        from kotodama.model.decode_engine import SamplingParams

        dec, eng = self.dec, self.dec.engine
        law = SamplingParams(temperature=1.0, repetition_penalty=1.0)
        dev = eng.device
        bound: dict[str, int] = {}
        row_of: dict[int, int] = {}
        dec.release(range(dec.rows))
        try:
            nxt = 0
            for (prompt, lever), idx in batch:
                slot = 0
                if lever is not None:
                    if lever not in bound:
                        bound[lever] = eng.levers.bind(self.plans[lever])
                    slot = bound[lever]
                rows = list(range(nxt, nxt + len(idx)))
                nxt += len(idx)
                logits = eng.prefill(torch.tensor([prompt], device=dev), slot=slot)
                dec.admit(rows, logits, law, slots=[slot] * len(rows) if eng.levers is not None else None,
                          seeds=[0] * len(rows))
                row_of.update(zip(idx, rows))
            # the unused rows score filler from the last prompt (they stay in their own groups)
            last_slot = slot
            filler = list(range(nxt, dec.rows))
            if filler:
                dec.admit(filler, logits, law,
                          slots=[last_slot] * len(filler) if eng.levers is not None else None,
                          seeds=[0] * len(filler))
            S = max(len(records[i].generated_ids) for _, idx in batch for i in idx)
            last = records[batch[-1][1][0]].generated_ids
            tokens = torch.zeros((dec.rows, S), dtype=torch.long)
            for r in range(dec.rows):
                tokens[r] = torch.tensor((list(last) + [last[-1]] * S)[:S])
            for i, r in row_of.items():
                g = records[i].generated_ids
                tokens[r] = torch.tensor(list(g) + [g[-1]] * (S - len(g)))
            dec.score(tokens.to(dev))
            order = [i for _, idx in batch for i in idx]
            signed = [trim(dec.signed(row_of[i], device=dev), len(records[i].generated_ids) - 1) for i in order]
        finally:
            for slot in bound.values():
                eng.levers.release(slot)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        return order, signed

    @torch.inference_mode()
    def sign_captured(self, records: Sequence[SignRecord], rows: Sequence[Optional[signing.SignedRow]],
                      identity: Optional[dict[str, Any]] = None) -> list[SignResult]:
        """Reduce rows the signing decoder captured as it generated them (in-flight signing: ``rows[i]`` is
        ``records[i]``'s span, exported when its generation ended; None: not exported). One result per record, in
        order; a record is skipped for the reasons ``sign`` skips it, or when its span was not exported. Rows reduce
        in batches of the decoder's width, as ``sign`` reduces them; their signatures are the ones ``sign`` gives the
        same records on this signer (same decoder, scoring instead of sampling). ``identity``: the capture identity
        the reducer names their lane by, when it is not this signer's (rows whose KV came from somewhere a replay of
        their ids does not reach, such as a KV session, add what made it)."""
        if len(rows) != len(records):
            raise ValueError(f"{len(rows)} captured rows for {len(records)} records")
        results = [SignResult(record=r) for r in records]
        ok: list[int] = []
        for i, (rec, row) in enumerate(zip(records, rows)):
            reason = self.why_not(rec)
            if reason is None and row is None:
                reason = "its span was not captured (fewer than two generated ids, or as long as the sign tail)"
            if reason is None and row.positions != len(rec.generated_ids) - 1:
                reason = f"its captured span has {row.positions} positions, its record {len(rec.generated_ids) - 1}"
            if reason is not None:
                results[i].skipped = reason
                logger.warning("sign: skipped %s: %s", rec.request_id, reason)
                continue
            results[i].same_engine = True
            ok.append(i)
        dev = self.dec.engine.device
        for k in range(0, len(ok), self.dec.rows):
            order = ok[k:k + self.dec.rows]
            signed = [rows[i] for i in order]
            for i, row in zip(order, signed):
                if audited(records[i].request_id, self.audit_every):
                    results[i].raw = _to_cpu(row)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            self._reduce(records, results, order, signed, identity=identity)
            self.timing["reduce_s"] += time.perf_counter() - t0
            self.timing["rows"] += len(order)
        return results

    def _reduce(self, records: Sequence[SignRecord], results: list[SignResult], order: list[int],
                signed: list[signing.SignedRow], identity: Optional[dict[str, Any]] = None) -> None:
        """Reduce a batch's rows together; if the reducer refuses the batch, row by row."""
        engine = identity if identity is not None else self.identity

        def call(ix: list[int], rows: list[signing.SignedRow]) -> list[signing.ReducedRow]:
            with deterministic_algorithms():
                out = self.reducer.reduce(rows, engine=engine,
                                          token_ids=[records[i].prompt_ids + records[i].generated_ids for i in ix])
            if len(out) != len(rows):
                raise RuntimeError(f"the reducer returned {len(out)} rows for {len(rows)}")
            return out

        try:
            for i, red in zip(order, call(order, signed)):
                results[i].reduced = red
            return
        except Exception as exc:  # noqa: BLE001 - any reducer failure is retried row by row, then reported per row
            logger.warning("sign: the reducer refused a batch of %d rows (%s: %s); retrying row by row",
                           len(order), type(exc).__name__, exc)
        for i, row in zip(order, signed):
            try:
                results[i].reduced = call([i], [row])[0]
            except Exception as exc:  # noqa: BLE001
                results[i].skipped = f"reducer: {type(exc).__name__}: {exc}"
                logger.warning("sign: skipped %s: %s", records[i].request_id, results[i].skipped)


def _to_cpu(row: signing.SignedRow) -> signing.SignedRow:
    surf = {k: ([t.cpu() for t in v] if isinstance(v, list) else v.cpu()) for k, v in row.surfaces.items()}
    return dataclasses.replace(row, surfaces=surf)


def write_results(out_dir: str | Path, results: Sequence[SignResult]) -> dict[str, Any]:
    """Results into ``out_dir``:
      * ``<out_dir>/rows.jsonl``: per record, in order: sign_id, steps (positions signed), weights, lever,
        same_engine, the record's meta, and either the skip reason or lane_id, names_digest, the features file and
        row index, and the reducer's meta (feature names left out);
      * ``<out_dir>/features-<names digest>.npy``: the reduced rows of one feature layout stacked, fp32 (rows with
        few steps have fewer features, so each layout has its own file);
      * ``<out_dir>/names.json``: each layout's feature names (from the reducer's meta), once;
      * ``<out_dir>/raw/<sign_id>.pt``: the raw signed surfaces of the rows the audit sink sampled.
    Returns a summary."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    feats: dict[str, list[np.ndarray]] = {}
    names: dict[str, Any] = {}
    raw = 0
    with open(out / "rows.jsonl", "w", encoding="utf-8") as f:
        for res in results:
            rec = res.record
            line: dict[str, Any] = {"sign_id": rec.request_id, "steps": res.steps, "weights": rec.weights,
                                    "lever": rec.lever, "same_engine": res.same_engine, "meta": rec.meta}
            if res.reduced is None:
                line["skipped"] = res.skipped or "not signed"
            else:
                red = res.reduced
                x = red.features
                x = x.detach().float().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, np.float32)
                meta = dict(red.meta)
                if "feature_names" in meta:
                    names.setdefault(red.names_digest, meta.pop("feature_names"))
                rows = feats.setdefault(red.names_digest, [])
                line.update(lane_id=red.lane_id, names_digest=red.names_digest,
                            file=f"features-{red.names_digest[:16]}.npy", index=len(rows), reducer_meta=meta)
                rows.append(x.reshape(-1).astype(np.float32))
            f.write(json.dumps(line, default=str) + "\n")
            if res.raw is not None:
                (out / "raw").mkdir(exist_ok=True)
                torch.save(dataclasses.asdict(res.raw), out / "raw" / f"{rec.request_id}.pt")
                raw += 1
    for digest, rows in feats.items():
        widths = {x.shape[0] for x in rows}
        if len(widths) != 1:
            raise ValueError(f"rows of one feature layout ({digest[:16]}) have different widths {sorted(widths)}")
        np.save(out / f"features-{digest[:16]}.npy", np.stack(rows))
    if names:
        (out / "names.json").write_text(json.dumps(names, default=str))
    return {"signed": sum(len(v) for v in feats.values()), "skipped": sum(r.reduced is None for r in results),
            "layouts": len(feats), "audited": raw, "skip_reasons": skip_reasons(results)}


def skip_reasons(results: Sequence[SignResult]) -> dict[str, int]:
    """The skipped results counted by reason, digests in a reason replaced by H and other numbers by N (so one cause
    is one key)."""
    out: dict[str, int] = {}
    for res in results:
        if res.reduced is None:
            key = re.sub(r"\d+", "N", re.sub(r"\b[0-9a-f]{12,}\b", "H", res.skipped or "not signed"))
            out[key] = out.get(key, 0) + 1
    return out
