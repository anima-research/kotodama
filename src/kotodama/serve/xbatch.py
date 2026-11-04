"""Cross-request batched decoding: concurrent completion requests decoded together in one wide row batch.

A server holds one engine and decodes one request at a time, while a decode step over 64 rows costs about twice a
step over 8 (the deep looped trunk is kernel-count bound, not FLOP bound). This module gathers concurrent requests
that share a sampling law and a steering config (``XBatchJob.key``), prefills each on the engine (prefix cache
included), fills ``n`` rows of a ``MultiSampleDecoder`` from each prefill (``admit``: every row keeps its own
position, mask and presence set) and steps every row together, ending each row at its own stop token, stop string,
token budget or the context limit. A request's result is delivered the moment its last row ends.

Two parts:
  * ``XBatchRunner.run_batch`` runs on the server's generation thread, with the engine lock held. It takes the
    oldest waiting request's key group from the queue (FIFO, as many requests as fit the widest bucket), decodes it
    on the narrowest bucket that holds the group's rows, and returns when every row has ended. With ``refill`` on,
    rows freed by ended candidates are refilled between steps (continuous batching): first with the unstarted
    candidates of admitted requests, then with new requests from the queue (a request may start with fewer rows
    than its n), until a request with a different key is waiting or ``should_yield`` says another engine user is
    (the server passes "a request outside the batcher waits on the engine lock"); those then get the engine when
    the batch drains.
  * ``XBatchScheduler`` is the asyncio side: ``submit`` enqueues a job and awaits its result, and one pump task
    runs batches (each under the engine lock, on the generation thread) while the queue is non-empty.

A decoder whose rows share storage (``kotodama.model.multi_decode.CascadeDecoder``: one copy of a prompt's KV per
group of rows) bounds admission too: the runner admits only as many rows as ``admit_capacity`` allows (a request
that finds no room waits in the queue), hands rows back with ``release`` as soon as nobody will restart from them,
and caps each candidate at ``token_room``.

With lever tables (``levers``, kotodama.model.levers), steering is per request instead: a job carries its own lever
plan, binds a slot of the tables when it is admitted (a job finds no room while every slot is bound, as with
``admit_capacity``), prefills and decodes its rows on that slot, and releases it when its last row ends, so the batch
key need not include steering at all.

With a signing decoder (``sign_decoder``: a ``CascadeDecoder`` built with ``sign=``, as wide as the widest bucket and
decoding the tokens it does), a job that sets ``sign`` decodes there: a batch holding one runs on the signing decoder
(other jobs of the batch decode on it too, unsigned), and a signed job waits while another decoder runs. When a signed
candidate ends, its row's span is exported (``MultiSampleDecoder.signed``, trimmed to the positions its generated ids
fed) into ``XBatchResult.signed``, before the row can start anything else. A signed job's prompt KV must be what a
cold prefill writes, as a seeded job's must: it asks for a ``fresh`` prefill, which an engine with batch-invariant
blocks serves from the caches' prompt KV (the bytes are the same) and any other engine serves cold. A prefill that
reports reused KV without that guarantee leaves its records to the drain, which signs them cold.

The runner knows nothing about HTTP: prompts arrive tokenized and validated, steering arrives as the engine's
``set_steer`` arguments or as a lever plan, and stop strings are checked through a ``decode`` callable (the
tokenizer's).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Hashable, Optional

import torch

from kotodama.model.decode_engine import SamplingParams
from kotodama.model.multi_decode import MultiSampleDecoder, candidate_seed

logger = logging.getLogger(__name__)


@dataclass
class XBatchResult:
    """One request's outcome: ``rows[i]`` is candidate i's generated ids (stop tokens not kept); ``stops[i]`` the
    stop token that ended candidate i (None: it ended otherwise)."""

    rows: list[list[int]]
    timing: dict[str, float]
    stops: list[Optional[int]] = field(default_factory=list)
    # signed jobs: candidate i's signed span (kotodama.model.signing.SignedRow, on the decoder's device), or None when
    # it fed fewer than one position, its span does not fit the sign tail, or the job's prefill reused cached KV
    signed: list[Any] = field(default_factory=list)
    # jobs that keep their KV (``XBatchJob.keep_kv``): candidate i's decode-written KV as kotodama.serve.kv_sessions
    # .row_kv gives it, for its fed tokens ((k, v), each [layers, n_kv_heads, fed, head_dim])
    kv: list[Any] = field(default_factory=list)


@dataclass
class XBatchJob:
    """One request: ``n`` candidates of ``prompt_ids``.

    ``key`` groups requests that may share a batch (equal sampling law and steering config); ``params``,
    ``steer_writes`` and ``steer_kwargs`` must be equal across one key. ``done`` is called exactly once, on the
    generation thread, with the result or the exception that ended the request.
    """

    key: Hashable
    prompt_ids: list[int]
    n: int
    max_new_tokens: int
    params: SamplingParams
    done: Callable[[XBatchResult | BaseException], None]
    stop_strings: tuple[str, ...] = ()
    steer_writes: Optional[list[tuple[int, torch.Tensor, float]]] = None
    steer_kwargs: dict[str, Any] = field(default_factory=dict)
    lever: Any = None  # a kotodama.model.levers.LeverPlan (runner built with levers), None: unsteered
    # seeded decoders: candidate i draws with candidate_seed(seed, i), in any batch (always on the widest bucket)
    seed: Optional[int] = None
    sign: bool = False  # decode on the signing decoder and export each candidate's signed span
    # a prefill of its own instead of the runner's (a KV session's turn: kotodama.serve.kv_sessions.prefill), called
    # on the generation thread when the job is admitted; returns what the runner's prefill returns
    prefill: Optional[Callable[[], tuple]] = None
    # called right after the job's rows are admitted (its prompt KV copied into them): a session turn marks the
    # engine's KV unreusable here, once the decoder has checked how it was written
    admitted: Optional[Callable[[], None]] = None
    # copy each candidate's decode-written KV out of its row when it ends (into XBatchResult.kv)
    keep_kv: bool = False
    t_enqueue: float = field(default_factory=time.perf_counter)


class XBatchQueue:
    """Thread-safe FIFO of waiting jobs (the event loop puts, the generation thread takes)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: list[XBatchJob] = []

    def put(self, job: XBatchJob) -> None:
        with self._lock:
            self._jobs.append(job)

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)

    def take_group(self, max_rows: int) -> list[XBatchJob]:
        """The oldest job's key group, oldest first, while the rows fit ``max_rows``; later jobs of the key that
        do not fit stay queued, as do jobs of other keys."""
        with self._lock:
            if not self._jobs:
                return []
            key = self._jobs[0].key
            taken, rows = [], 0
            for job in self._jobs:
                if job.key == key and rows + job.n <= max_rows:
                    taken.append(job)
                    rows += job.n
            for job in taken:
                self._jobs.remove(job)
            return taken

    def requeue(self, jobs: list[XBatchJob]) -> None:
        """Put ``jobs`` back at the head of the queue, in their order (taken but not admitted)."""
        with self._lock:
            self._jobs[:0] = jobs

    def drain(self) -> list[XBatchJob]:
        """Remove and return every waiting job."""
        with self._lock:
            jobs, self._jobs = self._jobs, []
            return jobs

    def take_next(self, key: Hashable) -> Optional[XBatchJob]:
        """The oldest job, when it has ``key``; None otherwise (also while a job of another key waits at the head,
        so a running batch stops growing once a different request is due)."""
        with self._lock:
            if not self._jobs or self._jobs[0].key != key:
                return None
            return self._jobs.pop(0)


@dataclass
class _Live:
    """A job decoding in the batch: each candidate's output, which candidates run in which rows, how many have
    started, and the prompt state a row needs to start another candidate (its length and last-position logits)."""

    job: XBatchJob
    logits: torch.Tensor
    prompt_len: int
    budget: int
    stop_window: int
    timing: dict[str, float]
    t_admit: float
    out: list[list[int]] = field(default_factory=list)
    running: dict[int, int] = field(default_factory=dict)  # row -> candidate
    started: int = 0
    home: frozenset[int] = frozenset()  # the rows admitted from this job's prompt: only they can restart it
    slot: int = 0  # the job's lever slot (0: none)
    stops: dict[int, int] = field(default_factory=dict)  # candidate -> the stop token that ended it
    signed: dict[int, Any] = field(default_factory=dict)  # candidate -> its exported span (signed jobs)
    kv_reused: int = 0  # prompt tokens whose KV the prefill took from a prefix cache or pool (signed: not exported)
    kept_kv: dict[int, Any] = field(default_factory=dict)  # candidate -> its decode-written KV (keep_kv jobs)

    @property
    def done(self) -> bool:
        return self.started == self.job.n and not self.running


class XBatchRunner:
    """Runs batches on the generation thread. ``decoders`` maps bucket width -> decoder (all over one engine);
    ``prefill(ids (1, S)) -> (last logits (1, V), prefix-cache info or None)`` prefills the engine;
    ``set_steer(writes, **kwargs)`` installs the engine's steering rows (``None`` clears); ``decode(ids) -> str``
    is the tokenizer's decode, for stop strings; ``should_yield() -> bool``, polled between steps, stops refilling
    so the engine passes to whoever waits for it once the batch drains.

    Rows hold candidates, not requests. A request admitted with fewer free rows than its n starts its remaining
    candidates in rows its own finished candidates free: such a row still holds the request's prompt in its first
    cache slots, so it rewinds to the prompt's end and samples a fresh first token (``MultiSampleDecoder.restart``,
    no prefill and no KV copy). Freed rows go first to admitted requests' unstarted candidates, then (with
    ``refill``) to new requests from the queue."""

    def __init__(self, decoders: dict[int, MultiSampleDecoder], prefill: Callable[..., tuple],
                 set_steer: Callable[..., None], decode: Callable[[list[int]], str], stop_ids: frozenset[int],
                 max_seq_len: int, device: torch.device, refill: bool = False,
                 should_yield: Optional[Callable[[], bool]] = None, levers: Any = None,
                 sign_decoder: Optional[MultiSampleDecoder] = None,
                 between_steps: Optional[Callable[[], None]] = None) -> None:
        if not decoders:
            raise ValueError("XBatchRunner needs at least one decoder bucket")
        self.decoders = dict(sorted(decoders.items()))
        self.max_rows = max(self.decoders)
        self.prefill = prefill
        self.set_steer = set_steer
        self.decode = decode
        self.stop_ids = stop_ids
        self.max_seq_len = max_seq_len
        self.device = device
        self.refill = refill
        self.should_yield = should_yield or (lambda: False)
        # the engine's LeverTables: per-job levers, ``prefill(ids, slot)``; None: ``prefill(ids)``, set_steer
        self.levers = levers
        # signed jobs decode here (see the module docstring); as wide as the widest bucket
        if sign_decoder is not None and (getattr(sign_decoder, "sign", None) is None
                                         or sign_decoder.rows != self.max_rows):
            raise ValueError("sign_decoder must be a signing decoder as wide as the widest bucket")
        self.sign_decoder = sign_decoder
        # polled on the generation thread between steps and after each batch (the server reduces signed rows there)
        self.between_steps = between_steps or (lambda: None)
        self._dec: Optional[MultiSampleDecoder] = None  # the decoder of the running batch
        self.queue = XBatchQueue()
        self.stats: dict[str, float] = {"batches": 0, "steps": 0, "row_steps": 0, "useful_row_steps": 0,
                                        "admitted": 0, "refilled": 0, "restarts": 0, "prefill_s": 0.0,
                                        "step_s": 0.0, "host_s": 0.0, "signed_rows": 0, "export_s": 0.0,
                                        "reused_kv_rows": 0}

    def bucket_for(self, rows: int) -> MultiSampleDecoder:
        """The narrowest decoder that holds ``rows`` rows (the widest when none does)."""
        return next((d for w, d in self.decoders.items() if w >= rows), self.decoders[self.max_rows])

    @staticmethod
    def _deliver(job: XBatchJob, outcome: XBatchResult | BaseException) -> None:
        try:
            job.done(outcome)
        except Exception:  # a broken callback must not end the batch for every other request
            logger.exception("xbatch: result callback failed")

    def _admit(self, dec: MultiSampleDecoder, job: XBatchJob, free: list[int]) -> Optional[_Live]:
        """Prefill ``job`` and start its first candidates in up to ``job.n`` free rows (taken from ``free``).
        Returns None (the job already failed) when the prefill raises."""
        t0 = time.perf_counter()
        timing: dict[str, float] = {"queue_ms": round((t0 - job.t_enqueue) * 1000, 2)}
        slot = 0
        try:
            ids = torch.tensor([job.prompt_ids], dtype=torch.long, device=self.device)
            # a seeded job's draws must not depend on what a prefix cache held, nor a signed job's signature: their
            # prefill is cold, or with batch-invariant blocks reads the caches' prompt KV, equal to a cold prefill's
            fresh = {"fresh": True} if job.seed is not None or job.sign else {}
            if job.prefill is not None:  # its own prefill (a session turn: unsteered, never a cache hit)
                logits, info = job.prefill()
            elif self.levers is not None:
                slot = self.levers.bind(job.lever) if job.lever is not None else 0
                logits, info = self.prefill(ids, slot, **fresh)
            else:
                logits, info = self.prefill(ids, **fresh)
            m = min(job.n, len(free), dec.admit_capacity())
            rows = free[:m]
            kw: dict[str, Any] = {}
            if self.levers is not None:
                kw["slots"] = slot
            if job.seed is not None:
                kw["seeds"] = [candidate_seed(job.seed, c) for c in range(m)]
            first = dec.admit(rows, logits, job.params, **kw)
            if job.admitted is not None:
                job.admitted()
            del free[:m]
        except Exception as exc:
            logger.warning("xbatch: prefill/admit failed for a %d-token prompt: %r", len(job.prompt_ids), exc)
            if slot:
                self.levers.release(slot)
            self._deliver(job, exc)
            return None
        timing["prefill_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        if info is not None:
            timing["prefix_cache_hit"] = float(bool(info.get("prefix_hit")))
            timing["prefix_common_tokens"] = float(info.get("common_prefix", 0))
            if "pool_tokens" in info:
                timing["prefix_pool_tokens"] = float(info["pool_tokens"])
        timing["batched_rows"] = float(dec.rows)
        timing["rows_at_admission"] = float(len(rows))
        # A stop string that first appears at this token ends inside its text, so decoding a tail that holds the
        # longest stop several times over finds it without re-decoding the whole row every step.
        window = 4 * max((len(x) for x in job.stop_strings), default=0) + 16
        budget = min(job.max_new_tokens, self.max_seq_len - len(job.prompt_ids), dec.token_room(len(job.prompt_ids)))
        if slot:
            timing["lever_slot"] = float(slot)
        # reused KV whose bytes equal a cold prefill's (batch-invariant blocks) is no reason to withhold a span
        reused = int(info.get("common_prefix", 0)) if info and not info.get("invariant") else 0
        lv = _Live(job, logits, len(job.prompt_ids), budget, window, timing, t0, out=[[] for _ in range(job.n)],
                   home=frozenset(rows), slot=slot, kv_reused=reused)
        try:
            self._start(lv, rows, first)
        except Exception as exc:  # the job never joins the batch: its slot and its caller must not wait on it
            logger.warning("xbatch: starting a %d-token prompt's candidates failed: %r", len(job.prompt_ids), exc)
            self._release_lever(lv)
            dec.release(rows)
            self._deliver(job, exc)
            return None
        self.stats["admitted"] += 1
        self.stats["prefill_s"] += time.perf_counter() - t0
        return lv

    def _start(self, lv: _Live, rows: list[int], first: list[int]) -> None:
        """Bind the next candidates of ``lv`` to ``rows`` and process their first tokens."""
        for row, tok in zip(rows, first):
            lv.running[row] = lv.started
            lv.started += 1
            self._consume(lv, row, tok)

    def _consume(self, lv: _Live, row: int, tok: int) -> None:
        """Process one sampled token of the candidate in ``row``, ending it at a stop token (not kept), a stop
        string or the token budget (capped at the context room left after the prompt, so a live row never steps
        past the last cache slot). An ended candidate releases its row (``lv.running`` drops it)."""
        cand = lv.running[row]
        if tok in self.stop_ids:
            lv.stops[cand] = tok
            self._end(lv, row)
            return
        out = lv.out[cand]
        out.append(tok)
        self.stats["useful_row_steps"] += 1
        stops = lv.job.stop_strings
        if len(out) >= lv.budget:
            self._end(lv, row)
        elif stops:
            tail = self.decode(out[-lv.stop_window:])
            if any(s in tail for s in stops):
                self._end(lv, row)

    def _end(self, lv: _Live, row: int) -> None:
        """End the candidate in ``row``; a signed job's candidate first exports its span. Runs before the row can
        restart or take another job, and before the next step launches: the span's last fed position is the
        candidate's next-to-last generated id (the step in flight feeds its last id one slot further, which the
        trim drops; a span of the sign tail or more is not exported, since that step would overwrite its last
        slot)."""
        cand = lv.running.pop(row)
        dec = self._dec
        if lv.job.keep_kv and dec is not None:
            from kotodama.serve.kv_sessions import row_kv

            # the KV its decode steps wrote: every generated id but the last was fed, and the last too when a stop
            # token followed it (the step in flight writes one slot further, past what is copied)
            out = lv.out[cand]
            fed = len(out) if cand in lv.stops else max(len(out) - 1, 0)
            lv.kept_kv[cand] = row_kv(dec, row, lv.prompt_len, fed)
        if not lv.job.sign or dec is None or getattr(dec, "sign", None) is None:
            return
        from kotodama.model.signer import trim

        n = len(lv.out[cand]) + (cand in lv.stops)  # generated ids, the stop token included
        if n < 2 or n > dec.sign.tail:
            lv.signed[cand] = None
            return
        if lv.kv_reused:  # the drain signs its record cold (see the module docstring)
            lv.signed[cand] = None
            self.stats["reused_kv_rows"] += 1
            return
        t0 = time.perf_counter()
        lv.signed[cand] = trim(dec.signed(row, device=dec.cur_tokens.device, lever_slot=lv.slot), n - 1)
        self.stats["signed_rows"] += 1
        self.stats["export_s"] += time.perf_counter() - t0

    def _room_for(self, job: XBatchJob) -> bool:
        """Whether ``job`` can bind its lever now (always, for an unsteered job or a runner without levers)."""
        return self.levers is None or job.lever is None or self.levers.free > 0

    def _release_lever(self, lv: _Live) -> None:
        if lv.slot:
            self.levers.release(lv.slot)
            lv.slot = 0

    def _finish(self, lv: _Live) -> None:
        self._release_lever(lv)
        lv.timing["decode_ms"] = round((time.perf_counter() - lv.t_admit) * 1000 - lv.timing["prefill_ms"], 2)
        lv.timing["total_ms"] = round((time.perf_counter() - lv.job.t_enqueue) * 1000, 2)
        signed = [lv.signed.get(c) for c in range(lv.job.n)] if lv.job.sign else []
        kv = [lv.kept_kv.get(c) for c in range(lv.job.n)] if lv.job.keep_kv else []
        self._deliver(lv.job, XBatchResult(rows=lv.out, timing=lv.timing,
                                           stops=[lv.stops.get(c) for c in range(lv.job.n)], signed=signed, kv=kv))

    def _fill(self, dec: MultiSampleDecoder, active: list[_Live], free: list[int], head: XBatchJob) -> int:
        """Hand free rows out: a job's unstarted candidates restart in the job's own free rows (the only rows that
        hold its prompt), which stay reserved for it until every candidate has started; the other free rows go,
        with refill on and nobody else waiting for the engine, to new jobs of the batch's key. Delivers jobs that
        end. Returns the number of jobs taken from the queue."""
        taken = 0
        while True:
            progress = True
            while progress:  # a restarted candidate can end at its first token and free its row again
                progress = False
                busy = {row for lv in active for row in lv.running}
                for lv in active:
                    want = lv.job.n - lv.started
                    rows = [r for r in sorted(lv.home) if r not in busy][:want]
                    if rows:
                        busy.update(rows)
                        kw = {} if lv.job.seed is None else {
                            "seeds": [candidate_seed(lv.job.seed, lv.started + i) for i in range(len(rows))]}
                        self._start(lv, rows, dec.restart(rows, lv.prompt_len, lv.logits, lv.job.params, **kw))
                        self.stats["restarts"] += len(rows)
                        progress = True
            for lv in [lv for lv in active if lv.done]:
                active.remove(lv)
                self._finish(lv)
            busy = {row for lv in active for row in lv.running}
            reserved = {row for lv in active if lv.started < lv.job.n for row in lv.home}
            free[:] = [r for r in range(dec.rows) if r not in busy and r not in reserved]
            dec.release(free)  # nobody restarts from these rows: storage they shared goes back to the decoder
            if not (free and self.refill) or self.should_yield() or not dec.admit_capacity():
                return taken
            job = self.queue.take_next(head.key)
            if job is None:
                return taken
            if not self._room_for(job):  # every lever slot is bound: it waits at the head of the queue
                self.queue.requeue([job])
                return taken
            if job.seed is not None and dec.rows != self.max_rows:  # seeded jobs run on the widest bucket only
                self.queue.requeue([job])
                return taken
            if job.sign and dec is not self.sign_decoder:  # signed jobs run on the signing decoder only
                self.queue.requeue([job])
                return taken
            taken += 1
            self.stats["refilled"] += 1
            lv = self._admit(dec, job, free)
            if lv is not None:
                active.append(lv)

    def run_batch(self) -> int:
        """Decode one batch from the queue to completion; returns the number of requests it served (0 when the
        queue was empty). Call on the generation thread with the engine lock held."""
        jobs = self.queue.take_group(self.max_rows)
        if not jobs:
            return 0
        head = jobs[0]
        # A batch holding a seeded job runs the widest bucket: rows decode bit for bit whatever shares a decoder of
        # one width, but decoders of different widths round the logits differently, which can flip a seeded draw.
        seeded = any(j.seed is not None for j in jobs)
        if any(j.sign for j in jobs):
            if self.sign_decoder is None:
                for job in jobs:
                    self._deliver(job, RuntimeError("a signed job needs a runner with a signing decoder"))
                return len(jobs)
            dec = self.sign_decoder
        else:
            dec = self.decoders[self.max_rows] if seeded else self.bucket_for(sum(j.n for j in jobs))
        self._dec = dec
        free = list(range(dec.rows))
        dec.release(free)  # whatever the decoder's rows held belongs to an earlier batch
        active: list[_Live] = []
        served = 0
        try:
            if head.steer_writes and self.levers is None:
                self.set_steer(head.steer_writes, **head.steer_kwargs)
            for i, job in enumerate(jobs):
                # the decoder's shared storage, or every lever slot, is taken: the rest wait in the queue
                if not dec.admit_capacity() or not self._room_for(job):
                    self.queue.requeue(jobs[i:])
                    break
                lv = self._admit(dec, job, free)
                served += 1
                if lv is not None:
                    active.append(lv)
            self.stats["batches"] += 1
            # Software pipeline: step k+1 is launched before step k's tokens are read back and processed, so the
            # host work (stop checks, retiring, starting candidates, admitting requests) overlaps the GPU step. A
            # row whose candidate ended at step k runs one extra step whose token is ignored; a row (re)started
            # while a step is in flight joins from the next launch: its start is stream-ordered after the in-flight
            # step, and that step's token for the row is ignored because the step's snapshot of which candidate
            # runs where predates the start.
            cuda = dec.cur_tokens.is_cuda
            bufs = [torch.empty(dec.rows, dtype=torch.long, pin_memory=cuda) for _ in range(2)]
            pending: Optional[tuple[torch.Tensor, Any, list[tuple[int, _Live, int]]]] = None
            k = 0
            while True:
                served += self._fill(dec, active, free, head)
                if not active and pending is None:
                    break
                self.between_steps()
                t0 = time.perf_counter()
                launched = None
                if active:
                    dec.step(head.params)
                    buf = bufs[k % 2]
                    buf.copy_(dec.cur_tokens.view(-1), non_blocking=cuda)
                    ev = torch.cuda.Event() if cuda else None
                    if ev is not None:
                        ev.record()
                    launched = (buf, ev, [(row, lv, cand) for lv in active for row, cand in lv.running.items()])
                    k += 1
                    self.stats["steps"] += 1
                    self.stats["row_steps"] += dec.rows
                if pending is not None:
                    buf, ev, snapshot = pending
                    if ev is not None:
                        ev.synchronize()
                    t1 = time.perf_counter()
                    toks = buf.tolist()
                    for row, lv, cand in snapshot:
                        if lv.running.get(row) == cand:
                            self._consume(lv, row, toks[row])
                    self.stats["step_s"] += t1 - t0
                    self.stats["host_s"] += time.perf_counter() - t1
                pending = launched
        except Exception as exc:
            logger.exception("xbatch: batch failed with %d live requests", len(active))
            for lv in active:
                self._deliver(lv.job, exc)
        finally:
            for lv in active:  # lever slots never outlive the batch either
                self._release_lever(lv)
            active.clear()
            self.set_steer(None)  # steering state never outlives the batch
            self._dec = None
        self.between_steps()
        return served


class XBatchScheduler:
    """The asyncio side: ``submit`` enqueues a job and awaits its result; one pump task runs batches while jobs
    wait. ``run_on_worker(fn)`` runs ``fn`` on the generation thread; ``lock`` is the engine lock every engine
    user holds, so other request paths interleave between batches."""

    def __init__(self, runner: XBatchRunner, lock: asyncio.Lock,
                 run_on_worker: Callable[[Callable[[], int]], Awaitable[int]]) -> None:
        self.runner = runner
        self.lock = lock
        self.run_on_worker = run_on_worker
        self._pump_task: Optional[asyncio.Task] = None

    async def submit(self, job: XBatchJob) -> XBatchResult:
        """Queue ``job`` (its ``done`` is replaced) and wait for its result; re-raises the exception that ended it.
        A job wider than the widest bucket is refused (ValueError): no batch could ever take it."""
        if not 1 <= job.n <= self.runner.max_rows:
            raise ValueError(f"n={job.n} outside [1, {self.runner.max_rows}]")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()

        def resolve(outcome: XBatchResult | BaseException) -> None:
            if fut.done():  # the awaiting request was cancelled
                return
            if isinstance(outcome, BaseException):
                fut.set_exception(outcome)
            else:
                fut.set_result(outcome)

        job.done = lambda outcome: loop.call_soon_threadsafe(resolve, outcome)
        self.runner.queue.put(job)
        if self._pump_task is None or self._pump_task.done():
            self._pump_task = asyncio.create_task(self._pump())
        return await fut

    async def _pump(self) -> None:
        while len(self.runner.queue):
            async with self.lock:
                try:
                    await self.run_on_worker(self.runner.run_batch)
                except Exception as exc:
                    # run_batch delivers its own failures; reaching here means the generation thread itself is
                    # gone (e.g. shutdown), so nothing queued can ever run: fail it rather than spin.
                    logger.exception("xbatch: pump failed; failing %d queued requests", len(self.runner.queue))
                    for job in self.runner.queue.drain():
                        XBatchRunner._deliver(job, exc)
                    return
