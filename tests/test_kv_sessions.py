"""KV sessions (kotodama.serve.kv_sessions, --kv_sessions) on a tiny looped model, CPU fp32.

What is pinned here:
  * the session K/V operations (kotodama.model.session_kv): unmoved rows and all values stay bit-identical, a moved
    key is the key RoPE gives its new position, and an exported range imported back where it came from restores it;
  * a session turn reads the session's KV as it holds it (earlier positions are never rewritten), its logits are an
    eager forward of the new tokens over exactly that KV, and the KV a candidate's decode steps wrote is the KV kept
    for it, aligned to its generated ids (only fed positions: the last id joins as unfed unless a stop followed);
  * a session's KV never reaches another request (the engine's prefix reuse refuses it);
  * the server: turns over HTTP, append, evict, export and import, the session's op log and cache state;
  * signing: a session's rows reduce under the session lane's identity, apart from cold rows, and an uncaptured one is
    counted on its session and never spooled for the drain;
  * the gateway routes every request of a session to the replica holding it.
"""

from __future__ import annotations

import asyncio
import functools
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import pytest
import torch
from torch.nn.attention import SDPBackend

from kotodama.model import session_kv
from kotodama.model.decode_engine import DecodeEngine, SamplingParams
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig, apply_rope
from kotodama.model.multi_decode import CascadeDecoder
from kotodama.serve import kv_sessions
from kotodama.serve.xbatch import XBatchJob, XBatchRunner, XBatchScheduler

CFG = dict(hidden_size=64, num_layers=11, num_attention_heads=4, num_kv_heads=2, head_dim=16,
           intermediate_size=128, vocab_size=211, max_position_embeddings=128, rope_theta=10000.0,
           qk_norm=True, tie_word_embeddings=True, attn_impl="sdpa", use_liger=False,
           attn_res=True, attn_res_boundaries=[0, 1, 3, 6, 9], loop_start=3, loop_len=3, loop_times=2)
MAX_SEQ = 96
HOT = SamplingParams(temperature=0.9, repetition_penalty=1.0)
STOPS = frozenset(range(4, 8))


@pytest.fixture(scope="module")
def model() -> KotodamaModel:
    torch.manual_seed(20261104)
    m = KotodamaModel(KotodamaModelConfig(**CFG)).eval()
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for layer in m.layers:
            for name, p in layer.named_parameters():
                if not name.startswith(("attn.", "ffn.")):
                    p.add_(0.3 * torch.randn(p.shape, generator=g))
    return m


def _ids(n: int, seed: int) -> list[int]:
    return torch.randint(14, CFG["vocab_size"], (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _engine(model: KotodamaModel) -> DecodeEngine:
    # with lever tables, as the tsumugi profile serves: the decoders check how the prompt KV was written
    return DecodeEngine(model, max_seq_len=MAX_SEQ, dtype=torch.float32, prefill_backends=[SDPBackend.MATH], levers=2)


def _runner(eng: DecodeEngine) -> XBatchRunner:
    wide = CascadeDecoder(eng, 4, groups=4, group_rows=2, tail_tokens=32, seeded=True)
    narrow = CascadeDecoder(eng, 2, groups=2, group_rows=2, tail_tokens=32, kv_from=wide, seeded=True)
    for d in (wide, narrow):
        d.compile_step(regional=True, backend="aot_eager")
    return XBatchRunner({2: narrow, 4: wide}, prefill=lambda ids, **kw: (eng.prefill(ids), None),
                        set_steer=eng.set_steer, decode=lambda ids: "", stop_ids=STOPS, max_seq_len=MAX_SEQ,
                        device=torch.device("cpu"), refill=True)


def _stacked(eng: DecodeEngine, lo: int, hi: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.stack([c[0, :, lo:hi] for c in eng.k_caches]), torch.stack([c[0, :, lo:hi] for c in eng.v_caches]))


def _eager_kv(model: KotodamaModel, ids: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    with torch.inference_mode():
        return model(torch.tensor([ids]), use_cache=True)["past_kv"]


def test_session_ops_move_keys_by_rope_and_leave_the_rest_bit_identical(model: KotodamaModel) -> None:
    torch.manual_seed(0)
    L, h, n, d = 3, 2, 12, 16
    cos, sin = model.rope_cos.float(), model.rope_sin.float()  # the model's own tables
    raw = torch.randn(L, h, n, d)
    k = apply_rope(raw, cos[:n], sin[:n])
    v = torch.randn(L, h, n, d)
    k2, v2 = session_kv.evict_rerotate(k, v, range(3, 7), cos, sin)
    assert k2.shape[-2] == n - 4 and torch.equal(k2[..., :3, :], k[..., :3, :])
    assert torch.equal(v2, torch.cat([v[..., :3, :], v[..., 7:, :]], dim=-2))
    # a survivor's key is the key RoPE gives its raw vector at its new position
    torch.testing.assert_close(k2[..., 3:, :], apply_rope(raw[..., 7:, :], cos[3:n - 4], sin[3:n - 4]),
                               atol=1e-5, rtol=1e-5)
    ek, ev, frm = session_kv.export_rows(k, v, range(3, 7))
    k3, v3 = session_kv.import_rows(k2, v2, ek, ev, 3, frm, cos, sin)
    assert torch.equal(k3[..., :7, :], k[..., :7, :]) and torch.equal(v3, v)  # rows back where they came from
    torch.testing.assert_close(k3, k, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError):
        session_kv.evict_rerotate(k, v, range(5, 20), cos, sin)


def _turn(runner: XBatchRunner, eng: DecodeEngine, s: kv_sessions.KVSession, new_ids: list[int], n: int,
          seed: int, append_from=None, max_new: int = 9):
    out = {}
    full, _ = kv_sessions.turn_ids(s, new_ids, append_from)
    job = XBatchJob(key="k", prompt_ids=full, n=n, max_new_tokens=max_new, params=HOT, seed=seed,
                    done=lambda o: out.__setitem__("r", o), keep_kv=True,
                    prefill=functools.partial(kv_sessions.prefill, eng, s, new_ids, append_from),
                    admitted=functools.partial(kv_sessions.poison, eng, s.id))
    runner.queue.put(job)
    with torch.inference_mode():
        runner.run_batch()
    res = out["r"]
    assert not isinstance(res, BaseException), res
    s.pending = {i: kv_sessions.Candidate(ids=list(r), k=kv[0], v=kv[1]) for i, (r, kv) in
                 enumerate(zip(res.rows, res.kv))}
    return res, full


def test_a_session_turn_reads_the_lived_kv_and_keeps_its_candidates_decode_kv(model: KotodamaModel) -> None:
    eng = _engine(model)
    runner = _runner(eng)
    s = kv_sessions.KVSession(id="s1")
    first = _ids(10, 1)
    res, full = _turn(runner, eng, s, first, n=3, seed=5)
    # turn 1 starts an empty session: a cold prefill, its KV taken into the session
    assert s.ids == first and s.turns == 1 and s.cache_state()["kind"] == "fresh"
    with torch.inference_mode():
        eng.prefill(torch.tensor([first]))
    k_ref, v_ref = _stacked(eng, 0, len(first))
    assert torch.equal(s.k, k_ref) and torch.equal(s.v, v_ref)
    # each candidate's kept KV covers its fed ids and matches a recompute of them (other kernels: not bitwise)
    for c, (row, stop) in enumerate(zip(res.rows, res.stops)):
        kv = res.kv[c]
        fed = len(row) if stop is not None else max(len(row) - 1, 0)
        assert kv[0].shape == (len(eng.k_caches), CFG["num_kv_heads"], fed, CFG["head_dim"])
        if fed:
            ref = _eager_kv(model, first + row)
            want = torch.stack([k[0, :, len(first):len(first) + fed] for k, _ in ref])
            torch.testing.assert_close(kv[0], want, atol=2e-4, rtol=2e-4)
    # turn 2 continues with a candidate: the session's earlier KV is read as held, never rewritten
    c = max(range(3), key=lambda i: s.pending[i].fed)
    cand = s.pending[c]
    before_k = torch.cat([s.k, cand.k], dim=-2).clone()
    second = _ids(6, 2)
    kv_sessions.append(s, c)
    assert s.unfed == cand.ids[cand.fed:] and s.ops[-1]["op"] == "append"
    n0 = s.length
    with torch.inference_mode():
        logits, info = kv_sessions.prefill(eng, s, second, None)
    assert info["session_tokens"] == n0 and s.length == n0 + len(cand.ids) - cand.fed + len(second)
    assert torch.equal(s.k[..., :n0, :], before_k)
    assert torch.equal(torch.stack([c_[0, :, :n0] for c_ in eng.k_caches]), before_k)
    # the logits are an eager forward of the new tokens over exactly the session's KV
    suffix = s.ids[n0:]
    past = [(before_k[li].unsqueeze(0), s.v[li, :, :n0].unsqueeze(0)) for li in range(before_k.shape[0])]
    mask = torch.ones(len(suffix), s.length, dtype=torch.bool).tril_(diagonal=n0)
    from torch.nn.attention import sdpa_kernel

    with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):  # the engine's extend backend
        want = model(torch.tensor([suffix]), use_cache=True, past_kv=past, mask=mask)["logits"][0, -1:].float()
    assert torch.equal(logits, want)
    # a recompute of the same ids differs (the lived decode KV is not a prefill's)
    with torch.inference_mode():
        cold = model(torch.tensor([s.ids]))["logits"][0, -1:].float()
    assert not torch.equal(cold, logits)
    # nothing of the session is reused by a later request
    kv_sessions.poison(eng, s.id)
    with torch.inference_mode():
        _, hit = eng.prefill_cached(torch.tensor([s.ids + [20]]))
    assert not hit["prefix_hit"]


def test_evict_export_import_keep_ids_ops_and_state(model: KotodamaModel) -> None:
    eng = _engine(model)
    runner = _runner(eng)
    s = kv_sessions.KVSession(id="s2")
    _turn(runner, eng, s, _ids(20, 3), n=2, seed=9)
    ids0, k0, v0 = list(s.ids), s.k.clone(), s.v.clone()
    blob = kv_sessions.export(s, 4, 9)
    kv_sessions.evict(eng, s, 4, 9)
    assert s.ids == ids0[:4] + ids0[9:] and s.cache_state()["kind"] == "rotated" and s.pending == {}
    k, v, ids, frm, meta = kv_sessions.parse_rows(blob)
    assert ids == ids0[4:9] and frm == 4 and meta["session"] == "s2"
    kv_sessions.import_(eng, s, k, v, ids, 4, frm, source=meta["session"])
    assert s.ids == ids0 and torch.equal(s.v, v0) and torch.equal(s.k[..., :9, :], k0[..., :9, :])
    torch.testing.assert_close(s.k, k0, atol=1e-5, rtol=1e-5)
    state = s.cache_state()
    assert state["kind"] == "spliced" and state["evict_rounds"] == 1 and state["imports"] == 1
    assert [op["op"] for op in state["ops"]] == ["turn", "evict", "import"]
    with pytest.raises(kv_sessions.SessionError):
        kv_sessions.evict(eng, s, 10, 400)
    # a turn after the splice decodes over it
    res, _ = _turn(runner, eng, s, _ids(4, 4), n=2, seed=10)
    assert len(res.rows) == 2 and s.turns == 2


class _CharTok:
    def encode(self, text, return_tensors=None, **_):
        ids = [14 + ord(c) % 190 for c in text]
        return torch.tensor([ids]) if return_tensors == "pt" else ids

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + (i - 4) % 26) for i in ids)


def test_the_server_serves_session_turns_and_ops(monkeypatch, model: KotodamaModel) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kotodama.serve import server

    eng = _engine(model)
    runner = _runner(eng)

    async def run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    sched = XBatchScheduler(runner, asyncio.Lock(), run_inline)
    for k, v in {"_model": model, "_tokenizer": _CharTok(), "_engine": eng, "_multi": {}, "_xbatch": sched,
                 "_device": torch.device("cpu"), "_max_seq_len": MAX_SEQ, "_prefix_cache": False,
                 "_stop_token_ids": STOPS, "_run_generation": run_inline, "_sign_spool": None,
                 "_sign_default_arg": False, "_kv_sessions": kv_sessions.SessionStore(2)}.items():
        monkeypatch.setattr(server, k, v)
    client = TestClient(server.app)
    sid = client.post("/kv_session/open", json={}).json()["session"]
    body = {"prompt": _ids(12, 5), "max_tokens": 6, "n": 2, "temperature": 1.0, "top_p": 1.0, "seed": 3,
            "kv_session": sid, "return_token_ids": True}
    r = client.post("/v1/completions", json=body)
    assert r.status_code == 200, r.text
    info = client.get(f"/kv_session/{sid}").json()
    assert info["length"] == 12 and sorted(info["candidates"]) == ["0", "1"] and info["turns"] == 1
    r = client.post("/v1/completions", json={**body, "prompt": _ids(3, 6), "append_from": 1})
    assert r.status_code == 200, r.text
    info = client.get(f"/kv_session/{sid}").json()
    assert info["turns"] == 2 and [op["op"] for op in info["ops"]] == ["turn", "append", "turn"]
    blob = client.post("/kv_session/export", json={"session": sid, "start": 2, "stop": 6}).content
    r = client.post("/kv_session/evict", json={"session": sid, "start": 2, "stop": 6})
    assert r.status_code == 200 and r.json()["cache_state"] == "rotated"
    r = client.post(f"/kv_session/import?session={sid}&at=2", content=blob)
    assert r.status_code == 200 and r.json()["cache_state"] == "spliced", r.text
    # refusals: a missing candidate, steering, a session that does not exist, a full store
    assert client.post("/v1/completions", json={**body, "append_from": 5}).status_code == 409
    assert client.post("/v1/completions", json={**body, "kv_session": "nope"}).status_code == 404
    assert client.post("/v1/completions", json={**body, "steer_model": "x"}).status_code == 400
    client.post("/kv_session/open", json={"session": "two"})
    assert client.post("/kv_session/open", json={}).status_code == 409
    assert client.post("/kv_session/close", json={"session": "two"}).status_code == 200
    assert server._kv_sessions.info()["open"] == 1


def test_session_rows_reduce_under_the_session_lane_and_are_never_drained(monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from kotodama.model.signer import SignRecord, SignResult
    from kotodama.serve import server

    calls = {"captured": [], "signed": [], "drained": []}

    class _Signer:
        identity = {"weights": "w", "width": 4}
        plans: dict = {}

        def why_not(self, rec):
            return None

        def sign_captured(self, records, rows, identity=None):
            calls["captured"].append(([r.request_id for r in records], identity))
            return [SignResult(record=r, reduced=object()) for r in records]

    class _Spool:
        def add(self, writer, records, plans=None):
            calls["drained"] += [r.request_id for r in records]

        def add_signed(self, writer, results, identity, plans=None):
            calls["signed"].append(identity)
            return {"batch": "b", "signed": len(results), "skipped": 0}

    store = kv_sessions.SessionStore(2)
    sess = store.open("sx")
    state = sess.cache_state()

    def rec(name, session):
        return SignRecord(request_id=name, prompt_ids=(1, 2), generated_ids=(3, 4), weights="w",
                          meta={"cache_state": state} if session else {})

    for k, v in {"_signer": _Signer(), "_sign_spool": _Spool(), "_sign_writer": "w", "_kv_sessions": store,
                 "_sign_unwritten": [], "_sign_stats": {"records": 0, "batches": 0, "signed": 0, "skipped": 0,
                                                        "errors": 0}}.items():
        monkeypatch.setattr(server, k, v)
    row = object()
    server._sign_reduce_batch([(rec("cold", False), row, None), (rec("sess", True), row, None),
                               (rec("sess-lost", True), None, None), (rec("cold-lost", False), None, None)])
    assert calls["drained"] == ["cold-lost"]  # a cold row the decoder missed goes to the drain; a session's never
    assert calls["captured"][0] == (["cold"], None)
    ids, identity = calls["captured"][1]
    assert ids == ["sess"] and identity["kv_source"] == "session" and identity["weights"] == "w"
    assert identity["session_ops_sha256"] == kv_sessions.SESSION_IDENTITY["session_ops_sha256"]
    assert calls["signed"][0] == _Signer.identity and calls["signed"][1]["kv_source"] == "session"
    assert sess.unsigned == 1 and store.stats["unsigned"] == 1


def test_the_gateway_routes_a_session_to_its_replica(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    from pathlib import Path

    import httpx
    from fastapi.testclient import TestClient

    from kotodama.serve import gateway

    cfg = gateway.load_config(str(Path(__file__).resolve().parents[1] / "configs" / "gateway-tsumugi.example.yaml"),
                              [0, 1, 2])
    cfg.workdir = str(tmp_path)
    fleet = gateway.Fleet(cfg)
    for r in fleet.replicas:
        r.healthy = True
    seen = []

    def backend(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.port, request.url.path))
        if request.url.path == "/kv_session/open":
            return httpx.Response(200, json={"session": "abc"})
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(gateway, "_cfg", cfg, raising=False)
    monkeypatch.setattr(gateway, "_fleet", fleet, raising=False)
    monkeypatch.setattr(gateway, "_http", httpx.AsyncClient(transport=httpx.MockTransport(backend)), raising=False)
    client = TestClient(gateway.app)
    assert client.post("/kv_session/open", json={}).json() == {"session": "abc"}
    home = fleet.sessions["abc"].port
    for _ in range(4):
        assert client.post("/v1/completions", json={"prompt": [1, 2], "kv_session": "abc"}).status_code == 200
        assert seen[-1] == (home, "/v1/completions")
    assert client.post("/kv_session/evict", json={"session": "abc", "start": 0, "stop": 1}).status_code == 200
    assert seen[-1] == (home, "/kv_session/evict")
    assert client.get("/kv_session/abc").status_code == 200 and seen[-1] == (home, "/kv_session/abc")
    assert client.post("/v1/completions", json={"prompt": [1], "kv_session": "zzz"}).status_code == 404
    fleet.sessions["abc"].healthy = False
    assert client.post("/v1/completions", json={"prompt": [1], "kv_session": "abc"}).status_code == 410
    fleet.sessions["abc"].healthy = True
    assert client.post("/kv_session/close", json={"session": "abc"}).status_code == 200
    assert "abc" not in fleet.sessions and fleet.global_inflight == 0
