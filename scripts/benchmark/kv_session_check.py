"""GPU acceptance for KV sessions (--kv_sessions) against a running server that signs in flight.

  python scripts/benchmark/kv_session_check.py --url http://127.0.0.1:PORT --spool SPOOL_DIR --out result.json

Checks, each recorded with its numbers:
  * probe: one seeded, signed fan of 8 decoded three ways (alone, beside foreign batchmates, and admitted after
    them into other rows). A candidate's decode-written KV, as the session keeps it, must be byte-identical across
    the three, and its tokens equal. Falsifier: the same tokens' KV as a prefill writes it must differ in bytes, or
    the comparison is not sensitive. The same fan decoded unsigned (another decoder) is reported beside them;
  * ops: evict a range, then import its exported rows back where they came from. Rows that do not move and every
    value stay byte-identical; moved keys come back within rotation rounding;
  * signing: the session's signed rows land in the spool with ``meta.cache_state`` and a lane apart from a cold
    signed request's; /replay refuses them; no session row is left unsigned;
  * timing: a session turn's prefill against a full re-prefill of the same context.

Exit status 0 when every check passes. The spool must be this server's own (--sign_spool): rows are read from it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import torch
from safetensors.torch import load

TEXT = ("The lighthouse keeper had not spoken to anyone in eleven days when the boat finally came. She watched it "
        "from the gallery, a grey smudge that became a hull, and went down the spiral stairs to meet it.")
FOREIGN = ["In the market at dawn the fishmongers argue about the weather.",
           "A letter arrived with no return address and a pressed flower inside.",
           "The orchestra tuned for a long time before anyone noticed the conductor was missing.",
           "Two friends meet on a train they both missed by a day.",
           "The recipe called for an ingredient nobody in the village had heard of.",
           "At the end of the corridor there was a door that had always been painted shut."]


def _rows(blob: bytes) -> tuple[torch.Tensor, torch.Tensor, dict]:
    t = load(blob)
    (n,) = struct.unpack("<Q", blob[:8])
    return t["k"], t["v"], json.loads(blob[8:8 + n]).get("__metadata__") or {}


class Client:
    def __init__(self, url: str) -> None:
        self.http = httpx.AsyncClient(base_url=url, timeout=1800.0)

    async def post(self, path: str, **kw: Any) -> httpx.Response:
        r = await self.http.post(path, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"{path}: {r.status_code} {r.text[:400]}")
        return r

    async def open(self) -> str:
        return (await self.post("/kv_session/open", json={})).json()["session"]

    async def turn(self, sid: str | None, prompt: Any, *, n: int = 8, seed: int = 1234, sign: bool = True,
                   max_tokens: int = 256, append_from: int | None = None) -> dict:
        body = {"prompt": prompt, "max_tokens": max_tokens, "n": n, "temperature": 1.0, "top_p": 1.0, "seed": seed,
                "sign": sign, "return_token_ids": True, "sign_meta": {"check": "kv_session"}}
        if sid is not None:
            body["kv_session"] = sid
        if append_from is not None:
            body["append_from"] = append_from
        return (await self.post("/v1/completions", json=body)).json()

    async def info(self, sid: str) -> dict:
        return (await self.http.get(f"/kv_session/{sid}")).json()

    async def export(self, sid: str, start: int, stop: int) -> bytes:
        return (await self.post("/kv_session/export", json={"session": sid, "start": start, "stop": stop})).content

    async def wait_signed(self, timeout: float = 600.0) -> dict:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            sign = (await self.http.get("/info")).json().get("sign") or {}
            if not sign.get("pending_inflight"):
                return sign
            await asyncio.sleep(1.0)
        raise RuntimeError("in-flight rows still pending after the timeout")


async def _probe_condition(c: Client, mode: str, sign: bool = True) -> dict:
    sid = await c.open()
    foreign = []
    if mode in ("batchmates", "permuted"):
        k = 2 if mode == "batchmates" else 6
        foreign = [asyncio.create_task(c.turn(None, FOREIGN[i], seed=900 + i)) for i in range(k)]
        await asyncio.sleep(0.3 if mode == "batchmates" else 1.0)  # permuted: the foreign fans hold the first rows
    t0 = time.perf_counter()
    out = await c.turn(sid, TEXT, sign=sign)
    wall = time.perf_counter() - t0
    if foreign:
        await asyncio.gather(*foreign)
    info = await c.info(sid)
    return {"sid": sid, "out": out, "info": info, "wall_s": wall}


def _equal(a: bytes, b: bytes) -> tuple[bool, float]:
    ka, va, _ = _rows(a)
    kb, vb, _ = _rows(b)
    same = ka.shape == kb.shape and torch.equal(ka, kb) and torch.equal(va, vb)
    diff = float(max((ka.float() - kb.float()).abs().max(), (va.float() - vb.float()).abs().max())) \
        if ka.shape == kb.shape else float("inf")
    return same, diff


async def main(args: argparse.Namespace) -> int:
    c = Client(args.url)
    result: dict[str, Any] = {"checks": {}}
    checks = result["checks"]

    # ── probe ──
    conds = {m: await _probe_condition(c, m) for m in ("alone", "batchmates", "permuted")}
    conds["unsigned"] = await _probe_condition(c, "alone", sign=False)
    tokens = {m: [ch["token_ids"]["completion"] for ch in v["out"]["choices"]] for m, v in conds.items()}
    cands = conds["alone"]["info"]["candidates"]
    pick = max(cands, key=lambda k: cands[k]["fed"])
    L0 = conds["alone"]["info"]["length"]
    exports: dict[str, bytes] = {}
    for m, v in conds.items():
        await c.post("/kv_session/append", json={"session": v["sid"], "candidate": int(pick)})
        fed = v["info"]["candidates"][pick]["fed"]
        exports[m] = await c.export(v["sid"], L0, L0 + fed)
    fed = cands[pick]["fed"]
    # falsifier: the same tokens' KV as a prefill writes it
    ids = conds["alone"]["info"]["ids"] + cands[pick]["ids"][:fed]
    sid_f = await c.open()
    await c.turn(sid_f, ids, n=1, sign=False, max_tokens=1)
    exports["prefilled"] = await c.export(sid_f, L0, L0 + fed)
    probe = {"candidate": int(pick), "fed": fed, "prompt_tokens": L0,
             "tokens_equal": {m: tokens[m] == tokens["alone"] for m in tokens}}
    for m in ("batchmates", "permuted", "unsigned", "prefilled"):
        same, diff = _equal(exports["alone"], exports[m])
        probe[f"alone_vs_{m}"] = {"bytes_equal": same, "max_abs_diff": diff}
    probe["pass"] = (probe["tokens_equal"]["batchmates"] and probe["tokens_equal"]["permuted"]
                     and probe["alone_vs_batchmates"]["bytes_equal"] and probe["alone_vs_permuted"]["bytes_equal"]
                     and not probe["alone_vs_prefilled"]["bytes_equal"])
    checks["probe"] = probe
    print(json.dumps({"probe": probe}, indent=1), flush=True)

    # ── ops: evict a range, import it back ──
    sid = conds["alone"]["sid"]
    n = (await c.info(sid))["length"]
    a, b = 8, min(40, n - 1)
    pre = await c.export(sid, 0, n)
    cut = await c.export(sid, a, b)
    await c.post("/kv_session/evict", json={"session": sid, "start": a, "stop": b})
    post = await c.export(sid, 0, n - (b - a))
    kp, vp, _ = _rows(pre)
    kq, vq, _ = _rows(post)
    evict_ok = (torch.equal(kq[..., :a, :], kp[..., :a, :]) and torch.equal(vq[..., :a, :], vp[..., :a, :])
                and torch.equal(vq[..., a:, :], vp[..., b:, :]))
    await c.post(f"/kv_session/import?session={sid}&at={a}", content=cut,
                 headers={"content-type": "application/octet-stream"})
    back = await c.export(sid, 0, n)
    kr, vr, _ = _rows(back)
    moved = float((kr[..., b:, :].float() - kp[..., b:, :].float()).abs().max()) if b < n else 0.0
    scale = float(kp[..., b:, :].float().abs().max()) if b < n else 1.0
    import_ok = torch.equal(vr, vp) and torch.equal(kr[..., :b, :], kp[..., :b, :]) and moved <= 0.02 * scale
    info = await c.info(sid)
    checks["ops"] = {"evict": [a, b], "evict_unmoved_and_values_bitwise": evict_ok,
                     "import_values_and_unmoved_bitwise": bool(torch.equal(vr, vp) and torch.equal(kr[..., :b, :],
                                                                                                 kp[..., :b, :])),
                     "moved_keys_max_abs_diff": moved, "moved_keys_abs_max": scale,
                     "cache_state": info["cache_state"], "pass": evict_ok and import_ok and info["cache_state"] == "spliced"}
    after = await c.turn(sid, " She waved.", n=4, seed=77)
    print(json.dumps({"ops": checks["ops"]}, indent=1), flush=True)

    # ── signing ──
    cold = await c.turn(None, TEXT, n=2, seed=4321)
    sign = await c.wait_signed()
    session_ids = {ch["sign_id"] for v in conds.values() if v["out"]["choices"][0].get("sign_id")
                   for ch in v["out"]["choices"]} | {ch["sign_id"] for ch in after["choices"]}
    cold_ids = {ch["sign_id"] for ch in cold["choices"]}
    lanes: dict[str, str] = {}
    states: dict[str, Any] = {}
    for rows in Path(args.spool, "signed").glob("*/rows.jsonl"):
        for line in rows.read_text().splitlines():
            row = json.loads(line)
            if row.get("sign_id") in session_ids | cold_ids:
                lanes[row["sign_id"]] = row.get("lane_id") or f"skipped: {row.get('skipped')}"
                states[row["sign_id"]] = (row.get("meta") or {}).get("cache_state", {}).get("kind")
    s_lanes = {lanes.get(i) for i in session_ids}
    c_lanes = {lanes.get(i) for i in cold_ids}
    replay = await c.http.post("/replay", json={"sign_id": sorted(session_ids)[0], "format": "json"})
    unsigned = {k: v for k, v in (sign.get("stats", {}).get("unsigned") or {}).items() if "KV session" in k}
    checks["signing"] = {"session_rows": len(session_ids), "found": sum(i in lanes for i in session_ids),
                         "session_lanes": sorted(map(str, s_lanes)), "cold_lanes": sorted(map(str, c_lanes)),
                         "session_states": sorted({str(states.get(i)) for i in session_ids}),
                         "replay_status": replay.status_code, "session_unsigned": unsigned,
                         "pass": (all(i in lanes for i in session_ids) and None not in s_lanes
                                  and not (s_lanes & c_lanes) and len(c_lanes) == 1 and replay.status_code == 409
                                  and not unsigned
                                  and all(states.get(i) in ("fresh", "spliced") for i in session_ids))}
    print(json.dumps({"signing": checks["signing"]}, indent=1), flush=True)

    # ── timing: a session turn vs a full re-prefill of the same context ──
    full_ids = (await c.info(sid))["ids"]
    t0 = time.perf_counter()
    full = await c.turn(None, full_ids + [13], n=1, sign=False, max_tokens=1)
    t_full = time.perf_counter() - t0
    t0 = time.perf_counter()
    sess = await c.turn(sid, [13], n=1, sign=False, max_tokens=1, append_from=0)
    t_sess = time.perf_counter() - t0
    checks["timing"] = {"context_tokens": len(full_ids), "full_reprefill_s": round(t_full, 3),
                        "session_turn_s": round(t_sess, 3), "pass": bool(full["choices"] and sess["choices"])}
    result["pass"] = all(v["pass"] for v in checks.values())
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"RESULT {'PASS' if result['pass'] else 'FAIL'}: " + ", ".join(f"{k} {'PASS' if v['pass'] else 'FAIL'}"
                                                                     for k, v in checks.items()), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True)
    p.add_argument("--spool", required=True, help="the server's own --sign_spool directory")
    p.add_argument("--out", required=True)
    sys.exit(asyncio.run(main(p.parse_args())))
