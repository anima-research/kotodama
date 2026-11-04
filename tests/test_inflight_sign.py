"""In-flight signing (a signed request signed as it generates) on a tiny looped model, CPU fp32.

What is pinned here:
  * the cross-request runner decodes signed jobs on its signing decoder (a signing CascadeDecoder with the widest
    bucket's geometry, viewing its storage), unsigned jobs of the batch beside them, and every job decodes the tokens
    it decodes without signing (seeded draws, a decode lever, stops, budgets, candidates restarted in freed rows);
  * each signed candidate's span, exported as it ends, is bit for bit the span the drain's replay of its record gives
    (every surface), and a span that does not fit the sign tail is not exported;
  * ``RecordSigner.sign_captured`` reduces captured rows as ``sign`` reduces records, skipping the same records;
  * ``SignSpool.add_signed`` writes the records and their signed batch together and moves the drain past them, so a
    drain never signs them again; while another drain holds the spool it writes nothing;
  * the server (--sign_inflight): a signed request's choices carry sign ids whose signatures land in the spool once
    the in-flight rows are flushed, equal to the drain's signatures of the same records.
"""

from __future__ import annotations

import asyncio
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import pytest
import torch
from torch.nn.attention import SDPBackend

from kotodama.model.decode_engine import DecodeEngine, SamplingParams
from kotodama.model.levers import LeverPlan
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
from kotodama.model.multi_decode import CascadeDecoder
from kotodama.model.signer import RecordSigner, SignRecord
from kotodama.model.signing import ReducedRow, SignSpec
from kotodama.serve.xbatch import XBatchJob, XBatchRunner, XBatchScheduler

CFG = dict(hidden_size=64, num_layers=11, num_attention_heads=4, num_kv_heads=2, head_dim=16,
           intermediate_size=128, vocab_size=211, max_position_embeddings=128, rope_theta=10000.0,
           qk_norm=True, tie_word_embeddings=True, attn_impl="sdpa", use_liger=False,
           attn_res=True, attn_res_boundaries=[0, 1, 3, 6, 9], loop_start=3, loop_len=3, loop_times=2)
MAX_SEQ = 96
SPEC = SignSpec((0, 3, 5, 8, 10), tail=10, depth=True, attn=True, width=MAX_SEQ)
HOT = SamplingParams(temperature=0.9, repetition_penalty=1.2)
STOPS = frozenset(range(4, 14))  # a few stop ids: some candidates end on one, the rest on their budgets


@pytest.fixture(scope="module")
def model() -> KotodamaModel:
    torch.manual_seed(20261029)
    m = KotodamaModel(KotodamaModelConfig(**CFG)).eval()
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for layer in m.layers:
            for name, p in layer.named_parameters():
                if not name.startswith(("attn.", "ffn.")):
                    p.add_(0.3 * torch.randn(p.shape, generator=g))
    return m


def _ids(n: int, seed: int) -> list[int]:
    return torch.randint(14, CFG["vocab_size"], (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _decoders(eng: DecodeEngine) -> tuple[dict, CascadeDecoder]:
    wide = CascadeDecoder(eng, 4, groups=4, group_rows=2, tail_tokens=32, seeded=True)
    narrow = CascadeDecoder(eng, 2, groups=2, group_rows=2, tail_tokens=32, kv_from=wide, seeded=True)
    signer = CascadeDecoder(eng, 4, groups=4, group_rows=2, tail_tokens=32, kv_from=wide, seeded=True, sign=SPEC)
    for d in (wide, narrow, signer):
        d.compile_step(regional=True, backend="aot_eager")
    return {2: narrow, 4: wide}, signer


def _runner(eng: DecodeEngine, decoders: dict, signer=None, hook=None) -> XBatchRunner:
    return XBatchRunner(decoders, prefill=lambda ids, slot=0, **kw: (eng.prefill(ids, slot=slot), None),
                        set_steer=eng.set_steer, decode=lambda ids: "", stop_ids=STOPS, max_seq_len=MAX_SEQ,
                        device=torch.device("cpu"), refill=True, levers=eng.levers, sign_decoder=signer,
                        between_steps=hook)


class _CharTok:
    """Characters <-> ids inside the tiny vocab (4 + ord % 200)."""

    def encode(self, text, return_tensors=None, **_):
        ids = [4 + ord(c) % 200 for c in text]
        return torch.tensor([ids]) if return_tensors == "pt" else ids

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + (i - 4) % 26) for i in ids)


class _Keep:
    """A stand-in reducer that keeps the rows it is handed, by token ids (features: the span length)."""

    reducer_identity = "keep-v1"

    def __init__(self) -> None:
        self.rows: dict[tuple, object] = {}

    def reduce(self, rows, *, token_ids, engine):
        out = []
        for r, ids in zip(rows, token_ids):
            self.rows[tuple(ids)] = r
            out.append(ReducedRow(features=np.array([r.positions, len(ids)], np.float32), names_digest="k",
                                  lane_id=f"keep:{engine['weights'][:8]}", meta={"steps": r.positions}))
        return out


class _Digest:
    """A reducer whose features are every surface of a row, summed in fp64 (equal rows give equal features)."""

    reducer_identity = "digest-v1"

    def reduce(self, rows, *, token_ids, engine):
        out = []
        for r in rows:
            feats = []
            for name in sorted(r.surfaces):
                x = r.surfaces[name]
                for t in (x if isinstance(x, list) else [x]):
                    feats += [float(t.double().sum()), float(t.double().abs().sum())]
            out.append(ReducedRow(features=np.array(feats, np.float64), names_digest="d", lane_id="digest"))
        return out


def _same(a, b) -> None:
    assert (a.positions, a.kept, a.start, a.lever_slot) == (b.positions, b.kept, b.start, b.lever_slot)
    for name, x in a.surfaces.items():
        y = b.surfaces[name]
        if isinstance(x, list):
            assert len(x) == len(y) and all(torch.equal(u, v) for u, v in zip(x, y)), name
        else:
            assert torch.equal(x, y), name


def _jobs(out: dict, plan: LeverPlan, sign: bool) -> list[XBatchJob]:
    """Three seeded jobs (one on a decode lever, one with more candidates than its rows get at first) and an
    unsigned one; ``sign`` marks the first three signed."""
    def job(name, prompt, n, max_new, seed, lever=None, signed=sign):
        return XBatchJob(key="k", prompt_ids=prompt, n=n, max_new_tokens=max_new, params=HOT,
                         done=lambda o: out.__setitem__(name, o), lever=lever, seed=seed, sign=signed)

    return [job("a", _ids(7, 1), 3, 8, 21), job("b", _ids(11, 2), 2, 6, 22, lever=plan),
            job("c", _ids(5, 3), 2, 12, 23), job("u", _ids(9, 4), 1, 7, 24, signed=False)]


@torch.inference_mode()
def test_signed_jobs_capture_the_spans_their_records_replay_to(model: KotodamaModel) -> None:
    eng = DecodeEngine(model, max_seq_len=MAX_SEQ, dtype=torch.float32, prefill_backends=[SDPBackend.MATH], levers=2)
    decs, signer = _decoders(eng)
    plan = LeverPlan(attn={5: np.random.default_rng(0).standard_normal(CFG["hidden_size"]).astype(np.float32)},
                     span="decode")
    calls: list[int] = []
    signed_out: dict = {}
    runner = _runner(eng, decs, signer, hook=lambda: calls.append(1))
    for job in _jobs(signed_out, plan, sign=True):
        runner.queue.put(job)
    while len(runner.queue):
        runner.run_batch()
    plain_out: dict = {}
    plain = _runner(eng, decs)
    for job in _jobs(plain_out, plan, sign=False):
        plain.queue.put(job)
    while len(plain.queue):
        plain.run_batch()
    # signing changes no token: the signed batch on the signing decoder, the unsigned one on the widest bucket
    for name in "abcu":
        assert signed_out[name].rows == plain_out[name].rows and signed_out[name].stops == plain_out[name].stops
    assert calls and runner.stats["signed_rows"] >= 5 and runner.stats["restarts"] >= 1
    assert any(s is not None for s in signed_out["a"].stops)  # a candidate ended on a stop token
    assert signed_out["u"].signed == [] and len(signed_out["a"].signed) == 3
    # every exported span is its record's replay, bit for bit
    keep = _Keep()
    rs = RecordSigner(signer, keep, compile_mode="aot_eager", plans={plan.digest(): plan})
    pairs = []
    for name, job_n in (("a", 3), ("b", 2), ("c", 2)):
        res = signed_out[name]
        prompt = {"a": _ids(7, 1), "b": _ids(11, 2), "c": _ids(5, 3)}[name]
        for c in range(job_n):
            gen = tuple(res.rows[c]) + ((res.stops[c],) if res.stops[c] is not None else ())
            rec = SignRecord(request_id=f"{name}-{c}", prompt_ids=tuple(prompt), generated_ids=gen,
                             weights=signer._sign_weights, lever=plan.digest() if name == "b" else None)
            pairs.append((rec, res.signed[c]))
            if len(gen) < 2:
                assert res.signed[c] is None
    replay = rs.sign([rec for rec, _ in pairs])
    checked = 0
    for (rec, row), res in zip(pairs, replay):
        if row is None:
            continue
        assert res.skipped is None
        _same(row, keep.rows[rec.prompt_ids + rec.generated_ids])
        checked += 1
    assert checked >= 5
    # sign_captured reduces the captured rows to what sign gives
    again = _Keep()
    rs2 = RecordSigner(signer, again, compile_mode="aot_eager", plans={plan.digest(): plan})
    cap = rs2.sign_captured([rec for rec, _ in pairs], [row for _, row in pairs])
    for (rec, row), a, b in zip(pairs, cap, replay):
        assert (a.skipped is None) == (row is not None and b.skipped is None)
        if a.reduced is not None:
            assert a.same_engine and np.array_equal(a.reduced.features, b.reduced.features)
            _same(again.rows[rec.prompt_ids + rec.generated_ids], row)
    with pytest.raises(ValueError):
        rs2.sign_captured([pairs[0][0]], [])


@torch.inference_mode()
def test_spans_past_the_sign_tail_are_not_exported(model: KotodamaModel) -> None:
    eng = DecodeEngine(model, max_seq_len=MAX_SEQ, dtype=torch.float32, prefill_backends=[SDPBackend.MATH], levers=2)
    decs, signer = _decoders(eng)
    out: dict = {}
    runner = _runner(eng, decs, signer)
    runner.stop_ids = frozenset()
    for name, budget in (("fits", SPEC.tail), ("long", SPEC.tail + 1)):
        runner.queue.put(XBatchJob(key="k", prompt_ids=_ids(6, 9), n=1, max_new_tokens=budget, params=HOT,
                                   done=lambda o, name=name: out.__setitem__(name, o), seed=5, sign=True))
    runner.run_batch()
    assert out["fits"].signed[0] is not None and out["fits"].signed[0].positions == SPEC.tail - 1
    assert out["long"].signed == [None]  # the step after its end would overwrite its last slot
    # a runner without a signing decoder fails a signed job rather than decode it unsigned
    bare = _runner(eng, decs)
    bare.queue.put(XBatchJob(key="k", prompt_ids=_ids(6, 9), n=1, max_new_tokens=3, params=HOT,
                             done=lambda o: out.__setitem__("bare", o), sign=True))
    bare.run_batch()
    assert isinstance(out["bare"], RuntimeError)
    with pytest.raises(ValueError):
        _runner(eng, decs, decs[4])  # not a signing decoder


@torch.inference_mode()
def test_signed_jobs_prefill_cold_and_one_that_reused_cached_kv_exports_nothing(model: KotodamaModel) -> None:
    # a prefix cache or pool hit hands the prompt KV computed in another prefill's split: its captured bytes would
    # depend on the cache's history, so the candidates export nothing and the drain signs their records cold
    eng = DecodeEngine(model, max_seq_len=MAX_SEQ, dtype=torch.float32, prefill_backends=[SDPBackend.MATH], levers=2)
    decs, signer = _decoders(eng)
    warm = _ids(12, 6)

    def prefill(ids, slot=0, **kw):
        hit = ids.shape[1] == len(warm)  # a cache that ignored the cold request: the guard behind it
        return eng.prefill(ids, slot=slot), {"prefix_hit": hit, "common_prefix": 8 if hit else 0}

    out: dict = {}
    runner = XBatchRunner(decs, prefill=prefill, set_steer=eng.set_steer, decode=lambda ids: "", stop_ids=frozenset(),
                          max_seq_len=MAX_SEQ, device=torch.device("cpu"), refill=True, levers=eng.levers,
                          sign_decoder=signer)
    for name, prompt in (("cold", _ids(9, 5)), ("warm", warm)):
        runner.queue.put(XBatchJob(key="k", prompt_ids=prompt, n=2, max_new_tokens=6, params=HOT,
                                   done=lambda o, name=name: out.__setitem__(name, o), sign=True))
    runner.run_batch()
    assert all(r is not None for r in out["cold"].signed)
    assert out["warm"].signed == [None, None] and runner.stats["reused_kv_rows"] == 2
    assert [len(r) for r in out["warm"].rows] == [6, 6]  # generation itself is unchanged
    # a signed job asks for a cold prefill (an unsigned one may take the cache)
    asked: list[bool] = []
    runner.prefill = lambda ids, slot=0, **kw: (asked.append(bool(kw.get("fresh"))), prefill(ids, slot, **kw))[1]
    for sign in (True, False):
        runner.queue.put(XBatchJob(key="k", prompt_ids=_ids(9, 7), n=1, max_new_tokens=3, params=HOT,
                                   done=lambda o: None, sign=sign))
        runner.run_batch()
    assert asked == [True, False]
    # a cache hit whose bytes equal a cold prefill's (batch-invariant blocks: the prefill says so) exports its spans
    runner.prefill = lambda ids, slot=0, **kw: (eng.prefill(ids, slot=slot),
                                                {"prefix_hit": True, "common_prefix": 8, "invariant": True})
    runner.queue.put(XBatchJob(key="k", prompt_ids=warm, n=2, max_new_tokens=6, params=HOT,
                               done=lambda o: out.__setitem__("invariant", o), sign=True))
    runner.run_batch()
    assert all(r is not None for r in out["invariant"].signed) and runner.stats["reused_kv_rows"] == 2


@torch.inference_mode()
def test_add_signed_writes_records_and_batch_and_the_drain_skips_them(model: KotodamaModel, tmp_path) -> None:
    from kotodama.serve.sign import SignSpool

    eng = DecodeEngine(model, max_seq_len=MAX_SEQ, dtype=torch.float32, prefill_backends=[SDPBackend.MATH], levers=2)
    decs, signer = _decoders(eng)
    out: dict = {}
    runner = _runner(eng, decs, signer)
    runner.queue.put(XBatchJob(key="k", prompt_ids=_ids(8, 3), n=3, max_new_tokens=6, params=HOT,
                               done=lambda o: out.__setitem__("j", o), seed=3, sign=True))
    runner.run_batch()
    res = out["j"]
    recs = [SignRecord(request_id=f"j-{c}", prompt_ids=tuple(_ids(8, 3)),
                       generated_ids=tuple(r) + ((s,) if s is not None else ()), weights=signer._sign_weights)
            for c, (r, s) in enumerate(zip(res.rows, res.stops))]
    rs = RecordSigner(signer, _Keep(), compile_mode="aot_eager")
    results = rs.sign_captured(recs, res.signed)
    spool = SignSpool(tmp_path / "spool")
    spool.add("other", recs[:1])  # a served record waiting for the drain
    with spool._drain_lock() as mine:
        assert mine
        assert spool.add_signed("srv", results, rs.identity) is None  # another drain holds the spool
    assert not (tmp_path / "spool" / "records" / "srv.inflight.jsonl").exists()
    summary = spool.add_signed("srv", results, rs.identity)
    assert summary["signed"] == sum(r.reduced is not None for r in results)
    assert spool.pending() == 1  # only the other writer's record
    lines = (tmp_path / "spool" / "records" / "srv.inflight.jsonl").read_text().splitlines()
    assert [json.loads(x)["request_id"] for x in lines] == [r.request_id for r in recs]
    index = [json.loads(x) for x in open(tmp_path / "spool" / "index.jsonl")]
    assert index[-1]["inflight"] is True and index[-1]["batch"] == summary["batch"]
    batch = tmp_path / "spool" / "signed" / summary["batch"]
    rows = [json.loads(x) for x in open(batch / "rows.jsonl")]
    assert [r["sign_id"] for r in rows] == [r.request_id for r in recs]
    assert all(r["same_engine"] for r in rows if "skipped" not in r)
    drained = spool.drain_once(rs)
    assert drained["signed"] + drained["skipped"] == 1 and spool.pending() == 0
    spool.add_signed("srv", results, rs.identity)  # a second in-flight batch from the same writer
    assert spool.pending() == 0


def test_server_signs_in_flight(monkeypatch, model: KotodamaModel, tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kotodama.model.signing import weights_digest
    from kotodama.serve import server
    from kotodama.serve.sign import SignSpool


    eng = DecodeEngine(model, max_seq_len=MAX_SEQ, dtype=torch.float32, prefill_backends=[SDPBackend.MATH])
    decs, signer = _decoders(eng)

    async def run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    spool = SignSpool(tmp_path / "spool")
    rs = RecordSigner(signer, _Digest(), compile_mode="aot_eager")
    # the scoring step compiles here, on the thread that replays below (the test client generates on its own thread;
    # a server compiles and runs everything on its generation thread)
    rs.warm(prompt_len=8, span=4)
    runner = XBatchRunner(decs, prefill=lambda ids, **kw: (eng.prefill(ids), None), set_steer=eng.set_steer,
                          decode=lambda ids: _CharTok().decode(ids), stop_ids=STOPS, max_seq_len=MAX_SEQ,
                          device=torch.device("cpu"), refill=True, sign_decoder=signer,
                          between_steps=server._sign_between_steps)
    sched = XBatchScheduler(runner, asyncio.Lock(), run_inline)
    for k, v in {"_model": model, "_tokenizer": _CharTok(), "_engine": eng, "_multi": {}, "_xbatch": sched,
                 "_device": torch.device("cpu"), "_max_seq_len": MAX_SEQ, "_prefix_cache": False,
                 "_stop_token_ids": STOPS, "_run_generation": run_inline, "_sign_spool": spool,
                 "_sign_writer": "replica-a", "_sign_weights": weights_digest(model), "_signer": rs,
                 "_sign_inflight_arg": True, "_sign_pending": [], "_sign_unwritten": [],
                 "_sign_stats": {"records": 0, "batches": 0, "signed": 0, "skipped": 0, "errors": 0}}.items():
        monkeypatch.setattr(server, k, v)
    client = TestClient(server.app)
    body = {"prompt": "two friends talk", "max_tokens": 8, "n": 3, "temperature": 1.1, "top_p": 0, "seed": 4,
            "return_token_ids": True, "sign": True, "sign_meta": {"round": 3}}
    r = client.post("/v1/completions", json=body)
    assert r.status_code == 200, r.text
    choices = r.json()["choices"]
    assert len(server._sign_pending) == 3 and server._sign_info()["pending_inflight"] == 3
    # an in-flight row waiting for its reduce is "not signed yet", not unknown
    r = client.post("/replay", json={"sign_id": choices[0]["sign_id"]})
    assert r.status_code == 409 and "not signed yet" in r.text
    with torch.inference_mode():  # the server stops with rows still pending: they are reduced before it exits
        asyncio.run(server._sign_flush_at_shutdown())
    assert server._sign_pending == [] and server._sign_stats["inflight_batches"] == 1
    assert server._sign_stats["errors"] == 0
    timing = server._sign_info()["inflight_timing"]
    assert timing["reduce_s"] > 0 and timing["signed_rows"] >= 2 and "write_s" in timing and "export_s" in timing
    index = [json.loads(x) for x in open(tmp_path / "spool" / "index.jsonl")]
    batch = tmp_path / "spool" / "signed" / index[-1]["batch"]
    rows = [json.loads(x) for x in open(batch / "rows.jsonl")]
    assert [row["sign_id"] for row in rows] == [c["sign_id"] for c in choices]
    assert all(row["meta"]["round"] == 3 for row in rows) and spool.pending() == 0
    # the drain's signatures of the same records equal the in-flight ones
    from kotodama.model.signer import read_records

    recs = list(read_records(tmp_path / "spool" / "records" / "replica-a.inflight.jsonl"))
    with torch.inference_mode():
        replay = rs.sign(recs)
    feats = {f: np.load(batch / f) for f in {row["file"] for row in rows if "file" in row}}
    for row, res in zip(rows, replay):
        if "skipped" in row:
            assert res.reduced is None
            continue
        assert np.array_equal(feats[row["file"]][row["index"]], res.reduced.features.astype(np.float32))
    assert sum("file" in row for row in rows) >= 2
    # /replay: a signed record's raw surfaces, re-captured on the signer and checked against its signed row
    from safetensors.torch import load as st_load

    sid = rows[0]["sign_id"]
    r = client.post("/replay", json={"sign_id": sid})
    assert r.status_code == 200, r.text
    exported = st_load(r.content)
    header = client.post("/replay", json={"sign_id": sid, "format": "json"}).json()
    assert header["schema"] == "anamnesis-replay-export-v1" and header["replayed_bitwise"] is True
    rec0 = recs[0]
    assert header["token_ids"] == list(rec0.prompt_ids + rec0.generated_ids)
    assert header["positions"] == len(rec0.generated_ids) - 1 and header["layers"]["sampled"] == list(SPEC.layers)
    assert set(exported) == set(header["tensors"]) and {"depth", "committed", "route", "q", "mean_rows"} <= set(exported)
    with torch.inference_mode():
        again = rs.capture(rec0)
    for name, t in exported.items():
        want = again.surfaces[name]
        want = torch.stack(want) if isinstance(want, list) else want
        assert torch.equal(t, want.cpu()), name
        assert list(t.shape) == header["tensors"][name]["shape"]
    assert header["lever_slot"] == again.lever_slot
    # Tier B resid_pre: each effective layer's attention input, as the model's own eager forward computes it (the
    # input of every attn_norm call, in effective-layer order, at the record's signed positions)
    extra = st_load(client.post("/replay", json={"sign_id": sid, "extras": ["resid_pre"]}).content)
    got = extra["resid_pre"]
    assert list(got.shape) == [header["positions"], len(header["route_sources"]), CFG["hidden_size"]]
    seen: list[torch.Tensor] = []
    hooks = [layer.attn_norm.register_forward_pre_hook(lambda _m, args: seen.append(args[0].detach()))
             for layer in model.layers]
    try:
        with torch.inference_mode():
            model(torch.tensor([list(rec0.prompt_ids + rec0.generated_ids[:-1])]))
    finally:
        for h in hooks:
            h.remove()
    eager = torch.stack([x[0, header["start"]:header["start"] + header["positions"]] for x in seen], dim=1)
    assert eager.shape == got.shape
    torch.testing.assert_close(got, eager.float(), atol=2e-4, rtol=2e-4)
    assert client.post("/replay", json={"sign_id": "nope"}).status_code == 404
    # a stored signature the replay does not reproduce: refused, never returned unmarked
    feats_path = batch / rows[0]["file"]
    saved = np.load(feats_path)
    np.save(feats_path, saved + 1.0)
    assert client.post("/replay", json={"sign_id": sid}).status_code == 409
    np.save(feats_path, saved)
    # a request the batcher cannot take (greedy) is served serially and signed by the drain from its record
    greedy = client.post("/v1/completions", json={**{k: v for k, v in body.items() if k != "seed"}, "temperature": 0,
                                                    "n": 2})
    assert greedy.status_code == 200 and all(c["sign_id"] for c in greedy.json()["choices"])
    assert spool.pending() == 2
    pending_id = greedy.json()["choices"][0]["sign_id"]
    r = client.post("/replay", json={"sign_id": pending_id})
    assert r.status_code == 409 and "not signed yet" in r.text  # its record waits for the drain
    # signing by default: a request that does not set `sign` is signed; `sign: false` opts out and is counted
    monkeypatch.setattr(server, "_sign_default_arg", True)
    monkeypatch.setattr(server, "_profile_name", "test-profile")
    quiet = {k: v for k, v in body.items() if k not in ("sign", "sign_meta", "seed")}
    r = client.post("/v1/completions", json={**quiet, "temperature": 0})
    assert r.status_code == 200 and all(c.get("sign_id") for c in r.json()["choices"])
    assert spool.pending() == 5
    r = client.post("/v1/completions", json={**quiet, "temperature": 0, "sign": False})
    assert r.status_code == 200 and not any(c.get("sign_id") for c in r.json()["choices"])
    assert spool.pending() == 5 and server._sign_stats["opted_out"] == 3
    # a stream leaves one record, its sign_id in the final event, its ids the ones streamed
    r = client.post("/generate", json={"prompt": "a stream", "max_new_tokens": 6, "temperature": 0, "stream": True})
    done = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")][-1]
    assert done["done"] and done["sign_id"] and spool.pending() == 6
    recs = {rec.request_id: rec for path in (tmp_path / "spool" / "records").glob("*.jsonl")
            for rec in read_records(path)}
    streamed = recs[done["sign_id"]]
    assert len(streamed.generated_ids) in (done["completion_tokens"], done["completion_tokens"] + 1)
    # every record names the server that generated it
    assert streamed.meta["profile"] == "test-profile" and streamed.meta["inflight"] is True
    assert {"pins", "engine_identity", "served_by", "seed", "law"} <= set(streamed.meta)
    # the drain signs them; spans it skips are counted by reason
    summaries = []
    with torch.inference_mode():
        while (summary := spool.drain_once(rs)) is not None:  # one batch of the decoder's width at a time
            summaries.append(summary)
    assert sum(x["signed"] + x["skipped"] for x in summaries) == 6 and spool.pending() == 0
    assert all(sum(x["skip_reasons"].values()) == x["skipped"] for x in summaries)


@torch.inference_mode()
def test_a_shutdown_whose_reduce_fails_spools_the_pending_records_for_the_drain(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    from kotodama.serve import server
    from kotodama.serve.sign import SignSpool

    class _Signer:
        def why_not(self, rec):
            return None if rec.generated_ids else "nothing generated"

    async def failing_run(fn, *args, **kwargs):
        raise RuntimeError("the reducer failed")

    recs = [SignRecord(request_id=f"r-{i}", prompt_ids=(1, 2, 3), generated_ids=(4, 5) if i else (),
                       weights="w" * 64) for i in range(3)]
    spool = SignSpool(tmp_path / "spool")
    for k, v in {"_signer": _Signer(), "_sign_spool": spool, "_sign_writer": "replica-a", "_run_generation": failing_run,
                 "_sign_pending": [(rec, None, None) for rec in recs], "_sign_unwritten": [], "_sign_reducing": 0,
                 "_sign_stats": {"records": 3, "batches": 0, "signed": 0, "skipped": 0, "errors": 0}}.items():
        monkeypatch.setattr(server, k, v)
    asyncio.run(server._sign_flush_at_shutdown())
    assert server._sign_pending == [] and spool.pending() == 2  # the record with nothing to sign is counted lost
    assert server._sign_stats["errors"] == 1


def test_warm_hands_the_reducer_one_batch(model: KotodamaModel) -> None:
    eng = DecodeEngine(model, max_seq_len=MAX_SEQ, dtype=torch.float32, prefill_backends=[SDPBackend.MATH], levers=2)
    _, signer = _decoders(eng)
    keep = _Keep()
    RecordSigner(signer, keep, compile_mode="aot_eager").warm(prompt_len=8, span=4, reduce=True)
    assert len(keep.rows) == 1  # one batch of identical filler rows
    assert next(iter(keep.rows.values())).positions == 3

    class Broken:
        reducer_identity = "broken"

        def reduce(self, rows, *, token_ids, engine):
            raise RuntimeError("no")

    RecordSigner(signer, Broken(), compile_mode="aot_eager").warm(prompt_len=8, span=4, reduce=True)  # logged


def test_a_signing_profile_refuses_to_start_unconfigured(tmp_path) -> None:
    """A server whose profile signs every request, on a site without a sign spool or reducer, exits before loading
    anything, naming the fix; --no-sign_default serves unsigned (in-flight signing then off)."""
    import subprocess
    import sys

    from pathlib import Path

    env = {k: v for k, v in os.environ.items() if not k.startswith("KOTODAMA_SIGN_") and k != "KOTODAMA_KERNEL_PINS"}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    cmd = [sys.executable, "-m", "kotodama.serve.server", "--profile", "tsumugi-1b", "--port", "1"]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 2 and "KOTODAMA_SIGN_SPOOL" in out.stderr and "--no-sign_default" in out.stderr
    # a site with a signer but no pin table (or a table path that does not exist): refused, so no replica records
    # a table of its own and signs a lane of its own
    site = {**env, "KOTODAMA_SIGN_SPOOL": str(tmp_path / "spool"), "KOTODAMA_SIGN_REDUCER": "x.y:Z"}
    out = subprocess.run(cmd, env=site, capture_output=True, text=True, timeout=300)
    assert out.returncode == 2 and "KOTODAMA_KERNEL_PINS" in out.stderr
    out = subprocess.run(cmd, env={**site, "KOTODAMA_KERNEL_PINS": str(tmp_path / "absent.json")},
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 2 and "no pin table at" in out.stderr
