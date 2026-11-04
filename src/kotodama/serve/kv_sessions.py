"""KV sessions (--kv_sessions): a KV cache that lives across the requests of one conversation, on one server.

Without a session every request re-prefills its whole prompt, so the model always sees a fresh recompute of its
window. A session keeps the KV the conversation actually lived instead:

  * a turn sends only its new tokens (``kv_session`` on /v1/completions or /generate). The server prefills them
    against the session's KV (an eager extend: every earlier position's KV is read as the session holds it, never
    rewritten), decodes the turn's n candidates, and keeps each candidate's decode-written KV until the next turn;
  * the next turn names the candidate the conversation continues with (``append_from``, or the session append endpoint):
    its generated tokens join the session with the KV their decode steps wrote. A token whose position no step fed
    (a candidate's last token, when no stop token followed it) is prefilled with the next turn;
  * between turns, ``evict`` drops a range of positions and renumbers the survivors (each moved key re-rotated
    through the model's RoPE table), and ``export`` / ``import`` take rows out and put them back at a position
    (kotodama.model.session_kv, whose rounding convention a session row's lane names).

A session's KV never reaches another request: the engine's own prefix reuse and the prefix pool both refuse it
(``poison``), and a session turn never stores blocks in the pool. A session turn decodes unsteered (steering on a
session request is refused). Its signed candidates are signed in flight only: their KV is the session's, so no cold
replay of their ids reproduces it, and a candidate that cannot be captured is counted (per session and in /info) and
left unsigned, never sent to the drain. Their records carry ``meta.cache_state`` (``cache_state``), and their lane
names the session conventions (``SESSION_IDENTITY``).

Everything here that touches a tensor runs on the server's generation thread (the engine's own state is touched
only there); the store's bookkeeping runs on the event loop.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from kotodama.model import session_kv

logger = logging.getLogger(__name__)

# What a session row's capture identity adds to the engine's (kotodama.model.signing.engine_identity): rows whose KV
# came from a session, the rotation's rounding convention and the digest of the code that moves its rows. A cold row
# adds nothing, so its lane id stays what it is.
SESSION_IDENTITY: dict[str, str] = {
    "kv_source": "session",
    "kv_rotation": session_kv.ROTATION,
    "session_ops_sha256": hashlib.sha256(Path(session_kv.__file__).read_bytes()).hexdigest(),
}


class SessionError(ValueError):
    """A session request that cannot be served as asked (the server answers 400, 404 or 409 by ``status``)."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class Candidate:
    """One candidate of a session's last turn: its generated ids (no stop token) and the KV its decode steps wrote
    for the first ``fed`` of them (``[layers, n_kv_heads, fed, head_dim]``)."""

    ids: list[int]
    k: torch.Tensor
    v: torch.Tensor
    sign_id: Optional[str] = None

    @property
    def fed(self) -> int:
        return int(self.k.shape[-2])


@dataclass
class KVSession:
    """One conversation's cache: ``ids[i]`` is the token whose KV ``k[..., i, :]`` / ``v[..., i, :]`` hold (keys rotated
    to position i); ``unfed`` the tokens that follow with no KV yet (prefilled with the next turn)."""

    id: str
    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None
    ids: list[int] = field(default_factory=list)
    unfed: list[int] = field(default_factory=list)
    pending: dict[int, Candidate] = field(default_factory=dict)
    ops: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    evicts: int = 0
    imports: int = 0
    unsigned: int = 0
    busy: bool = False
    last_used: float = field(default_factory=time.monotonic)

    @property
    def length(self) -> int:
        return len(self.ids)

    def cache_state(self) -> dict[str, Any]:
        """The session's condition, as its signed rows carry it in ``meta.cache_state``: ``kind`` fresh (no eviction,
        no import), rotated (evicted, nothing imported) or spliced (rows imported), the counts, and the op log."""
        kind = "spliced" if self.imports else ("rotated" if self.evicts else "fresh")
        return {"kind": kind, "session": self.id, "turn": self.turns, "evict_rounds": self.evicts,
                "imports": self.imports, "ops": [dict(op) for op in self.ops]}

    def info(self) -> dict[str, Any]:
        return {"session": self.id, "length": self.length, "unfed": len(self.unfed), "turns": self.turns,
                "pending": sorted(self.pending), "cache_state": self.cache_state()["kind"],
                "evict_rounds": self.evicts, "imports": self.imports, "unsigned": self.unsigned,
                "kv_bytes": (2 * self.k.numel() * self.k.element_size()) if self.k is not None else 0,
                "idle_s": round(time.monotonic() - self.last_used, 1)}


class SessionStore:
    """The server's open sessions, at most ``max_sessions``; a session idle for ``idle_s`` is closed when a new one
    needs its room."""

    def __init__(self, max_sessions: int, idle_s: float = 3600.0) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        self.max_sessions = int(max_sessions)
        self.idle_s = float(idle_s)
        self._lock = threading.Lock()
        self._sessions: dict[str, KVSession] = {}
        self.stats: dict[str, int] = {"opened": 0, "closed": 0, "expired": 0, "turns": 0, "unsigned": 0}

    def open(self, session_id: Optional[str] = None) -> KVSession:
        sid = session_id or uuid.uuid4().hex[:16]
        if not sid or len(sid) > 128:
            raise SessionError("a session id is 1 to 128 characters")
        with self._lock:
            if sid in self._sessions:
                raise SessionError(f"session {sid!r} is already open", 409)
            if len(self._sessions) >= self.max_sessions:
                now = time.monotonic()
                idle = [s for s in self._sessions.values() if not s.busy and now - s.last_used >= self.idle_s]
                for s in sorted(idle, key=lambda s: s.last_used)[:len(self._sessions) - self.max_sessions + 1]:
                    del self._sessions[s.id]
                    self.stats["expired"] += 1
                    logger.info("kv session %s closed after %.0f s idle", s.id, now - s.last_used)
            if len(self._sessions) >= self.max_sessions:
                raise SessionError(f"this server holds {self.max_sessions} open sessions; close one first", 409)
            s = KVSession(id=sid)
            self._sessions[sid] = s
            self.stats["opened"] += 1
            return s

    def get(self, session_id: str) -> KVSession:
        with self._lock:
            s = self._sessions.get(session_id)
        if s is None:
            raise SessionError(f"no open session {session_id!r} on this server", 404)
        return s

    def close(self, session_id: str) -> None:
        with self._lock:
            s = self._sessions.pop(session_id, None)
        if s is None:
            raise SessionError(f"no open session {session_id!r} on this server", 404)
        self.stats["closed"] += 1

    def claim(self, session_id: str) -> KVSession:
        """The session, marked busy (one request at a time works on a session); ``release`` ends the claim."""
        s = self.get(session_id)
        with self._lock:
            if s.busy:
                raise SessionError(f"session {session_id!r} is serving another request", 409)
            s.busy = True
        return s

    @staticmethod
    def release(s: KVSession) -> None:
        s.busy = False
        s.last_used = time.monotonic()

    def info(self) -> dict[str, Any]:
        with self._lock:
            sessions = list(self._sessions.values())
        return {"max_sessions": self.max_sessions, "idle_s": self.idle_s, "open": len(sessions),
                "stats": dict(self.stats), "sessions": [s.info() for s in sessions]}


# ── generation thread ─────────────────────────────────────────────────────────────────────────────────────────────


def poison(engine: Any, session_id: str) -> None:
    """Make the engine's cached KV unreusable by any later request: the engine reuses a cached prefix only under the
    lever digest that wrote it (``prefill_cached``), and no request carries this one; and nothing of it is prompt KV
    the prefix pool could store."""
    engine._kv_lever = f"kv-session:{session_id}"
    engine.prompt_kv_len = 0


def turn_ids(s: KVSession, new_ids: list[int], append_from: Optional[int]) -> tuple[list[int], int]:
    """A turn's whole context (the session's tokens, the appended candidate's, the unfed ones and the new ones) and
    how many leading tokens of it have KV in the session once the candidate is appended. Host only: no state
    changes."""
    ids, unfed = list(s.ids), list(s.unfed)
    if append_from is not None:
        cand = _candidate(s, append_from)
        ids += cand.ids[:cand.fed]
        unfed += cand.ids[cand.fed:]
    full = ids + unfed + list(new_ids)
    if len(full) == len(ids):
        raise SessionError("a session turn needs at least one token to prefill (new tokens, or an appended "
                           "candidate's unfed last token)")
    return full, len(ids)


def _candidate(s: KVSession, c: int) -> Candidate:
    if c not in s.pending:
        raise SessionError(f"session {s.id!r} holds no candidate {c} (it holds {sorted(s.pending)}: candidates live "
                           f"until the next turn or session op)", 409)
    if s.unfed:
        raise SessionError(f"session {s.id!r} has unfed tokens; a candidate can only follow fed KV", 409)
    return s.pending[c]


@torch.inference_mode()
def append(s: KVSession, c: int) -> None:
    """Join candidate ``c`` of the last turn to the session: its fed ids with the KV its decode steps wrote, the
    rest as unfed tokens. The turn's other candidates are dropped."""
    cand = _candidate(s, c)
    if cand.fed:
        s.k = cand.k if s.k is None else torch.cat([s.k, cand.k], dim=-2)
        s.v = cand.v if s.v is None else torch.cat([s.v, cand.v], dim=-2)
    s.ids += cand.ids[:cand.fed]
    s.unfed += cand.ids[cand.fed:]
    s.pending = {}
    s.ops.append({"op": "append", "candidate": int(c), "fed": cand.fed, "tokens": len(cand.ids),
                  **({"sign_id": cand.sign_id} if cand.sign_id else {})})


@torch.inference_mode()
def prefill(engine: Any, s: KVSession, new_ids: list[int], append_from: Optional[int]) -> tuple[torch.Tensor, dict]:
    """A session turn's prefill on ``engine``: append the chosen candidate, load the session's KV into the engine's
    cache, extend it eagerly with the unfed and new tokens (earlier positions are read, never rewritten), and take
    the extension's KV into the session. Leaves the engine holding the turn's whole prompt for the decoders' admit,
    after which the caller calls ``poison``. Returns (last-position logits (1, V) fp32, info)."""
    full, _ = turn_ids(s, new_ids, append_from)
    if len(full) >= engine.max_seq_len:
        raise SessionError(f"session {s.id!r}: the turn's context is {len(full)} tokens (max {engine.max_seq_len - 1})")
    if append_from is not None:
        append(s, append_from)
    n, p = s.length, len(full)
    engine.reset()
    dev = engine.device
    if n == 0:
        logits = engine.prefill(torch.tensor([full], dtype=torch.long, device=dev))
    else:
        for li in range(engine.n_layers):
            engine.k_caches[li][0, :, :n].copy_(s.k[li])
            engine.v_caches[li][0, :, :n].copy_(s.v[li])
        past = [(engine.k_caches[li][:, :, :n], engine.v_caches[li][:, :, :n]) for li in range(engine.n_layers)]
        suffix = torch.tensor([full[n:]], dtype=torch.long, device=dev)
        # suffix query i (position n + i) attends keys j <= n + i (the eager path's explicit causal-offset mask)
        mask = torch.ones(p - n, p, dtype=torch.bool, device=dev).tril_(diagonal=n)
        from torch.nn.attention import sdpa_kernel

        with sdpa_kernel(engine._extend_backends):
            out = engine.model(suffix, use_cache=True, past_kv=past, mask=mask)
        flags = []
        for li, (k, v) in enumerate(out["past_kv"]):
            engine.k_caches[li][:, :, n:p].copy_(k[:, :, n:])
            engine.v_caches[li][:, :, n:p].copy_(v[:, :, n:])
            flags += [k[:, :, n:].isfinite().all(), v[:, :, n:].isfinite().all()]
        if not bool(torch.stack(flags).all().item()):
            engine.reset()
            raise SessionError(f"session {s.id!r}: the turn's prefill wrote non-finite KV; the session is unchanged",
                               409)
        engine.attn_bias[..., :p] = 0.0
        engine.token_history[:p].copy_(torch.tensor(full, dtype=torch.long, device=dev))
        engine.pos.fill_(p)
        engine._pos_cpu = p
        engine._arm_decode_slot(0)
        logits = out["logits"][0, -1:].float()
    k_new = torch.stack([c[0, :, n:p] for c in engine.k_caches])
    v_new = torch.stack([c[0, :, n:p] for c in engine.v_caches])
    s.k = k_new if s.k is None else torch.cat([s.k, k_new], dim=-2)
    s.v = v_new if s.v is None else torch.cat([s.v, v_new], dim=-2)
    s.ops.append({"op": "turn", "tokens": p - n})
    s.ids, s.unfed, s.pending = full, [], {}
    s.turns += 1
    return logits, {"session": s.id, "session_tokens": n, "prefilled_tokens": p - n}


@torch.inference_mode()
def row_kv(dec: Any, row: int, prompt_len: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The KV row ``row`` of a decoder wrote for its first ``n`` generated tokens (positions prompt_len ..
    prompt_len + n - 1), copied out in the session layout ``[layers, n_kv_heads, n, head_dim]``. A CascadeDecoder
    keeps a row's own KV in its tail (position-major); a plain decoder in the row's full slot."""
    if hasattr(dec, "tail_ks"):
        k = torch.stack([t[row, :n].transpose(0, 1) for t in dec.tail_ks])
        v = torch.stack([t[row, :n].transpose(0, 1) for t in dec.tail_vs])
    else:
        k = torch.stack([c[row, :, prompt_len:prompt_len + n] for c in dec.k_caches])
        v = torch.stack([c[row, :, prompt_len:prompt_len + n] for c in dec.v_caches])
    return k.contiguous(), v.contiguous()


@torch.inference_mode()
def evict(engine: Any, s: KVSession, start: int, stop: int) -> None:
    """Drop positions [start, stop) and renumber the survivors (session_kv.evict_rerotate). The last turn's
    candidates are dropped (their KV was rotated to the old positions)."""
    if s.k is None or not 0 <= start < stop <= s.length:
        raise SessionError(f"evict [{start}, {stop}) is outside the session's {s.length} positions with KV")
    s.k, s.v = session_kv.evict_rerotate(s.k, s.v, range(start, stop), engine.rope_cos_t, engine.rope_sin_t)
    del s.ids[start:stop]
    s.pending = {}
    s.evicts += 1
    s.ops.append({"op": "evict", "start": int(start), "stop": int(stop)})


@torch.inference_mode()
def export(s: KVSession, start: int, stop: int) -> bytes:
    """Positions [start, stop) as safetensors bytes: ``k`` and ``v`` in the session layout, and in the metadata the
    rows' token ids, where their keys are rotated to (``rows_from``) and the session they came from."""
    from safetensors.torch import save

    if s.k is None or not 0 <= start < stop <= s.length:
        raise SessionError(f"export [{start}, {stop}) is outside the session's {s.length} positions with KV")
    k, v, rows_from = session_kv.export_rows(s.k, s.v, range(start, stop))
    meta = {"rows_from": str(rows_from), "ids": json.dumps(s.ids[start:stop]), "session": s.id,
            "cache_state": json.dumps({"kind": s.cache_state()["kind"], "turn": s.turns}),
            "layout": "[layers, n_kv_heads, positions, head_dim], keys after RoPE"}
    return save({"k": k.cpu(), "v": v.cpu()}, metadata=meta)


def parse_rows(data: bytes) -> tuple[torch.Tensor, torch.Tensor, list[int], int, dict[str, str]]:
    """An export's rows: (k, v, ids, rows_from, metadata). Host only."""
    import struct

    from safetensors.torch import load

    try:
        tensors = load(data)
        # the format's header: an 8-byte little-endian length, then JSON whose "__metadata__" is the string map
        (hlen,) = struct.unpack("<Q", data[:8])
        meta = dict(json.loads(data[8:8 + hlen]).get("__metadata__") or {})
    except Exception as exc:
        raise SessionError(f"the body is not a safetensors export: {type(exc).__name__}: {exc}") from exc
    if set(tensors) != {"k", "v"}:
        raise SessionError(f"an export holds tensors k and v, this one {sorted(tensors)}")
    try:
        ids = [int(t) for t in json.loads(meta["ids"])]
        rows_from = int(meta["rows_from"])
    except (KeyError, ValueError, TypeError) as exc:
        raise SessionError(f"the export's metadata needs ids and rows_from ({exc})") from exc
    k, v = tensors["k"], tensors["v"]
    if k.shape != v.shape or k.dim() != 4 or k.shape[-2] != len(ids):
        raise SessionError(f"export shapes k {tuple(k.shape)}, v {tuple(v.shape)} for {len(ids)} ids")
    return k, v, ids, rows_from, meta


@torch.inference_mode()
def import_(engine: Any, s: KVSession, k: torch.Tensor, v: torch.Tensor, ids: list[int], at: int, rows_from: int,
            source: Optional[str] = None) -> None:
    """Insert exported rows before position ``at`` (session_kv.import_rows); the last turn's candidates are dropped."""
    if s.k is None:
        if at != 0:
            raise SessionError("an empty session takes imported rows at 0 only")
        dev, dt = engine.device, engine.dtype
        shape = (engine.n_layers, engine.n_kv_heads, 0, engine.head_dim)
        s.k, s.v = torch.zeros(shape, device=dev, dtype=dt), torch.zeros(shape, device=dev, dtype=dt)
    if not 0 <= at <= s.length:
        raise SessionError(f"import at {at} is outside the session's {s.length} positions with KV")
    if s.unfed and at == s.length:
        raise SessionError("the session has unfed tokens after its KV; import before them, not after", 409)
    want = (s.k.shape[0], s.k.shape[1], s.k.shape[3])
    if (k.shape[0], k.shape[1], k.shape[3]) != want:
        raise SessionError(f"rows of shape {tuple(k.shape)} do not fit this model's cache {want}")
    if s.length + len(ids) + len(s.unfed) >= engine.max_seq_len:
        raise SessionError(f"importing {len(ids)} rows would pass the context ({engine.max_seq_len} positions)")
    s.k, s.v = session_kv.import_rows(s.k, s.v, k.to(engine.device, engine.dtype), v.to(engine.device, engine.dtype),
                                      at, rows_from, engine.rope_cos_t, engine.rope_sin_t)
    s.ids[at:at] = list(ids)
    s.pending = {}
    s.imports += 1
    s.ops.append({"op": "import", "at": int(at), "rows": len(ids), "rows_from": int(rows_from),
                  **({"source": source} if source else {})})
