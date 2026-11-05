"""Cross-request batched decoding on a looped model (tiny moonshot geometry, CPU fp32).

The decoder: rows filled from DIFFERENT prompts of different lengths must each reproduce the full no-cache forward
of their own prompt plus their own continuation, including a row reused after a longer prompt held it. The runner:
key grouping, bucket choice, per-row stops and budgets, per-request n, greedy equivalence with the single-stream
engine, refill (continuous batching), steering installed for the batch and cleared after. The server: compatible
requests go through the batcher, anything else falls back to the existing paths; a signed request leaves one sign
record per candidate in the spool (its generated ids ending with the stop token that ended it, its sign_meta kept),
each choice carries its sign_id, the records sign through a signing decoder, and signed requests the sign path cannot
take are refused. A seeded job's prompt is prefilled cold (never through a prefix cache) and it decodes on the widest
bucket (never pulled into a narrower batch by refill), so its draws depend neither on what the cache held nor on how
wide a batch its batchmates made.
"""

from __future__ import annotations

import asyncio
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import pytest
import torch
from torch.nn.attention import SDPBackend

from kotodama.model.decode_engine import DecodeEngine, SamplingParams
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
from kotodama.model.multi_decode import MultiSampleDecoder
from kotodama.serve.xbatch import XBatchJob, XBatchQueue, XBatchResult, XBatchRunner, XBatchScheduler

CFG = dict(hidden_size=64, num_layers=11, num_attention_heads=4, num_kv_heads=2, head_dim=16,
           intermediate_size=128, vocab_size=211, max_position_embeddings=128, rope_theta=10000.0,
           qk_norm=True, tie_word_embeddings=True, attn_impl="sdpa", use_liger=False,
           attn_res=True, attn_res_boundaries=[0, 1, 3, 6, 9], loop_start=3, loop_len=3, loop_times=2)
MAX_SEQ = 96
GREEDY = SamplingParams(temperature=1e-4, repetition_penalty=1.0)


@pytest.fixture(scope="module")
def model() -> KotodamaModel:
    torch.manual_seed(20261027)
    m = KotodamaModel(KotodamaModelConfig(**CFG)).eval()
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for layer in m.layers:
            for name, p in layer.named_parameters():
                if not name.startswith(("attn.", "ffn.")):
                    p.add_(0.3 * torch.randn(p.shape, generator=g))
    return m


def _engine(model: KotodamaModel, steer: bool = False) -> DecodeEngine:
    return DecodeEngine(model, max_seq_len=MAX_SEQ, dtype=torch.float32, prefill_backends=[SDPBackend.MATH],
                        steer_enabled=steer)


def _ids(n: int, seed: int) -> list[int]:
    return torch.randint(4, CFG["vocab_size"], (n,), generator=torch.Generator().manual_seed(seed)).tolist()


@torch.inference_mode()
def test_rows_from_different_prompts_match_their_own_full_forward(model: KotodamaModel) -> None:
    eng = _engine(model)
    wide = MultiSampleDecoder(eng, 6)
    dec = MultiSampleDecoder(eng, 4, kv_from=wide)  # a narrower bucket over the wide one's KV memory
    assert dec.k_caches[0].data_ptr() == wide.k_caches[0].data_ptr()
    params = SamplingParams(temperature=0.8, repetition_penalty=1.2)
    # a long prompt first fills every row, so the shorter prompts below land on rows holding stale, longer state
    dec.admit(range(4), eng.prefill_reference(torch.tensor([_ids(40, 9)])), params)
    for _ in range(3):
        dec.step(params)
    prompts = {0: _ids(11, 1), 1: _ids(23, 2), 2: _ids(11, 1), 3: _ids(5, 3)}  # rows 0 and 2: one prompt, n=2
    first: dict[int, torch.Tensor] = {}
    for rows, key in (([3], 3), ([0, 2], 0), ([1], 1)):  # admitted out of row order, row lists non-contiguous
        logits = eng.prefill_reference(torch.tensor([prompts[key]]))
        dec.admit(rows, logits, params)
        for r in rows:
            first[r] = logits[0]
    assert [dec.row_pos(r) for r in range(4)] == [11, 23, 11, 5]
    forced = torch.stack([torch.tensor(_ids(7, 30 + r)) for r in range(4)])  # (4, 7): its own continuation per row
    got = [torch.stack([first[r] for r in range(4)])]
    for t in range(forced.shape[1] - 1):
        got.append(dec.step_logits(forced[:, t:t + 1]).float())
    got_t = torch.stack(got, dim=1)  # (4, 7, V)
    for r in range(4):
        p = torch.tensor([prompts[r]])
        full = model(torch.cat([p, forced[r:r + 1]], dim=1))["logits"][0].float()
        want = full[p.shape[1] - 1:-1]
        assert torch.allclose(got_t[r], want, atol=2e-4, rtol=1e-4), (r, (got_t[r] - want).abs().max())
    assert [dec.row_pos(r) for r in range(4)] == [17, 29, 17, 11]


@torch.inference_mode()
def test_idle_rows_park_at_the_last_slot(model: KotodamaModel) -> None:
    eng = _engine(model)
    dec = MultiSampleDecoder(eng, 2)
    params = SamplingParams(temperature=0.9, repetition_penalty=1.0)
    dec.admit([0, 1], eng.prefill_reference(torch.tensor([_ids(MAX_SEQ - 3, 4)])), params)
    for _ in range(8):  # past capacity: the device position must stop at the last slot instead of indexing past it
        dec.step(params)
    assert dec.pos.tolist() == [MAX_SEQ - 1] * 2 and dec.row_pos(0) == MAX_SEQ - 3 + 8
    with pytest.raises(ValueError):
        dec.admit([0, 0], eng.prefill_reference(torch.tensor([_ids(8, 1)])), params)
    with pytest.raises(ValueError):
        dec.admit([2], eng.prefill_reference(torch.tensor([_ids(8, 1)])), params)


class _Rec:
    """Collects job outcomes by name."""

    def __init__(self) -> None:
        self.out: dict[str, XBatchResult | BaseException] = {}

    def job(self, name: str, prompt: list[int], n: int, key="k", max_new: int = 6, stops: tuple[str, ...] = (),
            params: SamplingParams = GREEDY, steer=None) -> XBatchJob:
        return XBatchJob(key=key, prompt_ids=prompt, n=n, max_new_tokens=max_new, params=params,
                         done=lambda o: self.out.__setitem__(name, o), stop_strings=stops, steer_writes=steer,
                         steer_kwargs={"rule": "K"} if steer else {})


def _runner(eng: DecodeEngine, widths=(2, 4), refill: bool = False, stop_ids=frozenset(), decode=None,
            prefix_cache: bool = False) -> XBatchRunner:
    wide = MultiSampleDecoder(eng, max(widths))
    decs = {w: (wide if w == max(widths) else MultiSampleDecoder(eng, w, kv_from=wide)) for w in widths}
    if prefix_cache:
        def prefill(ids):
            return eng.prefill_cached(ids)
    else:
        def prefill(ids):
            return eng.prefill_reference(ids), None
    return XBatchRunner(decs, prefill=prefill, set_steer=eng.set_steer,
                        decode=decode or (lambda ids: " ".join(map(str, ids))), stop_ids=stop_ids,
                        max_seq_len=MAX_SEQ, device=torch.device("cpu"), refill=refill)


def test_queue_groups_by_key_fifo_and_fit() -> None:
    q, rec = XBatchQueue(), _Rec()
    for name, key, n in (("a1", "A", 2), ("b1", "B", 1), ("a2", "A", 3), ("a3", "A", 2)):
        q.put(rec.job(name, [5], n, key=key))
    got = q.take_group(max_rows=4)  # a1 (2) + a3 (2): a2 (3) does not fit after a1, b1 is another key
    assert [j.n for j in got] == [2, 2] and len(q) == 2
    assert q.take_next("A") is None  # B is now the head: a running A batch stops growing
    assert [j.key for j in q.take_group(max_rows=4)] == ["B"]
    assert q.take_next("A").n == 3 and q.take_next("A") is None
    assert len(q.drain()) == 0


@torch.inference_mode()
def test_batch_matches_single_stream_greedy_per_request(model: KotodamaModel) -> None:
    eng = _engine(model)
    runner, rec = _runner(eng, widths=(2, 4, 8)), _Rec()
    prompts = {"p": _ids(9, 11), "q": _ids(26, 12), "r": _ids(17, 13)}
    for name, n in (("p", 3), ("q", 1), ("r", 2)):
        runner.queue.put(rec.job(name, prompts[name], n, max_new=8))
    assert runner.run_batch() == 3 and len(runner.queue) == 0
    ref = _engine(model)
    for name, n in (("p", 3), ("q", 1), ("r", 2)):
        res = rec.out[name]
        assert isinstance(res, XBatchResult) and len(res.rows) == n
        want = ref.generate_greedy(torch.tensor([prompts[name]]), 8)
        assert all(row == want for row in res.rows), (name, res.rows, want)
        assert res.timing["batched_rows"] == 8.0  # 6 rows -> the narrowest bucket that holds them
    assert runner.stats["batches"] == 1 and runner.stats["useful_row_steps"] == 6 * 8


@torch.inference_mode()
def test_per_row_stops_budgets_and_capacity(model: KotodamaModel) -> None:
    eng = _engine(model)
    stop = frozenset(range(4, 50))  # a broad stop set: rows end at different steps
    runner, rec = _runner(eng, widths=(4,), stop_ids=stop), _Rec()
    torch.manual_seed(2)
    hot = SamplingParams(temperature=1.3, repetition_penalty=1.1)
    runner.queue.put(rec.job("a", _ids(10, 1), 3, max_new=12, params=hot))
    runner.queue.put(rec.job("edge", _ids(MAX_SEQ - 4, 2), 1, max_new=50, params=hot))  # 4 slots of room
    runner.run_batch()
    a, edge = rec.out["a"], rec.out["edge"]
    assert len(a.rows) == 3 and all(len(r) <= 12 and not set(r) & stop for r in a.rows)
    assert len(edge.rows[0]) <= 4
    # stop strings end a row on the decoded text, and the matching token is kept (as on the single-stream path)
    runner2, rec2 = _runner(_engine(model), widths=(4,), decode=lambda ids: "x" * len(ids)), _Rec()
    runner2.queue.put(rec2.job("s", _ids(8, 3), 4, max_new=20, stops=("xxx",), params=hot))
    runner2.run_batch()
    assert [len(r) for r in rec2.out["s"].rows] == [3, 3, 3, 3]
    # the tail-window check ends each row exactly where a full decode first contains a stop
    letters = lambda ids: "".join(chr(97 + i % 3) for i in ids)  # noqa: E731 - three letters: stops recur
    runner3, rec3 = _runner(_engine(model), widths=(4,), decode=letters), _Rec()
    runner3.queue.put(rec3.job("t", _ids(8, 4), 4, max_new=60, stops=("abca", "ccb"), params=hot))
    runner3.run_batch()
    for row in rec3.out["t"].rows:
        hit = [k for k in range(1, len(row) + 1) if any(s in letters(row[:k]) for s in ("abca", "ccb"))]
        assert (hit[0] if hit else 60) == len(row), row


@torch.inference_mode()
def test_keys_never_share_a_batch_and_steering_is_scoped(model: KotodamaModel) -> None:
    eng = _engine(model, steer=True)
    calls: list = []

    def set_steer(writes, **kw):
        calls.append(None if writes is None else (len(writes), kw.get("rule")))
        eng.set_steer(writes, **kw)

    runner, rec = _runner(eng, widths=(4,)), _Rec()
    runner.set_steer = set_steer
    vec = torch.randn(CFG["hidden_size"], generator=torch.Generator().manual_seed(3))
    runner.queue.put(rec.job("steered", _ids(9, 1), 2, key="S", steer=[(4, vec, 0.7)]))
    runner.queue.put(rec.job("plain", _ids(9, 1), 2, key="P"))
    assert runner.run_batch() == 1 and calls == [(1, "K"), None] and eng.steer_buf.abs().sum() == 0
    assert runner.run_batch() == 1 and calls[-1] is None and runner.run_batch() == 0
    ref = _engine(model, steer=True)
    want = ref.generate_greedy(torch.tensor([_ids(9, 1)]), 6)
    assert rec.out["plain"].rows == [want, want]
    ref.set_steer([(4, vec, 0.7)])  # steered decode differs from plain decode, as on the single-stream engine
    want_s = ref.generate_greedy(torch.tensor([_ids(9, 1)]), 6)
    assert rec.out["steered"].rows == [want_s, want_s] and want_s != want


@torch.inference_mode()
def test_refill_serves_the_queue_in_one_batch_and_keeps_parity(model: KotodamaModel) -> None:
    prompts = [_ids(6 + 5 * i, 40 + i) for i in range(5)]
    eng = _engine(model)
    runner, rec = _runner(eng, widths=(4,), refill=True, prefix_cache=True), _Rec()
    for i, p in enumerate(prompts):
        runner.queue.put(rec.job(f"j{i}", p, 2, max_new=3 + i))
    assert runner.run_batch() == 5 and runner.stats["refilled"] == 3
    ref = _engine(model)
    for i, p in enumerate(prompts):
        want = ref.generate_greedy(torch.tensor([p]), 3 + i)
        assert rec.out[f"j{i}"].rows == [want, want], i
    # without refill the same queue needs a batch per bucketful
    runner2, rec2 = _runner(_engine(model), widths=(4,)), _Rec()
    for i, p in enumerate(prompts):
        runner2.queue.put(rec2.job(f"j{i}", p, 2, max_new=3 + i))
    assert [runner2.run_batch() for _ in range(4)] == [2, 2, 1, 0]
    # another engine user waiting: the batch stops growing and drains
    runner3, rec3 = _runner(_engine(model), widths=(4,), refill=True), _Rec()
    runner3.should_yield = lambda: True
    for i, p in enumerate(prompts):
        runner3.queue.put(rec3.job(f"j{i}", p, 2, max_new=3 + i))
    assert runner3.run_batch() == 2 and runner3.stats["refilled"] == 0


def _penalized_greedy(model: KotodamaModel, prompt: list[int], steps: int, rp: float) -> list[int]:
    """Single-stream greedy under a repetition penalty over the generated tokens: the penalty makes each token
    depend on the whole continuation, so a row decoding on the wrong prompt or position cannot match it."""
    eng = _engine(model)
    logits = eng.prefill_reference(torch.tensor([prompt]))[0]
    out: list[int] = []
    for _ in range(steps):
        pen = torch.where(logits > 0, logits / rp, logits * rp)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[out] = True
        tok = int(torch.where(mask, pen, logits).argmax())
        out.append(tok)
        logits = eng.step_logits(torch.tensor([[tok]]))[0, -1].float()
    return out


@torch.inference_mode()
def test_restart_rewinds_a_row_to_its_own_prompt(model: KotodamaModel) -> None:
    eng = _engine(model)
    dec = MultiSampleDecoder(eng, 2)
    params = SamplingParams(temperature=0.8, repetition_penalty=1.0)
    a, b = _ids(17, 70), _ids(9, 71)
    la = eng.prefill_reference(torch.tensor([a]))
    dec.admit([0, 1], la, params)
    for t in _ids(5, 72):  # row 0 decodes past its prompt; then row 1 takes another prompt
        dec.step_logits(torch.tensor([[t], [t]]))
    dec.admit([1], eng.prefill_reference(torch.tensor([b])), params)
    dec.step_logits(torch.tensor([[5], [6]]))
    dec.restart([0], len(a), la, params)  # row 0: a fresh candidate of prompt a, no prefill
    assert dec.row_pos(0) == len(a) and dec.row_pos(1) == len(b) + 1
    forced = torch.tensor([_ids(6, 73), _ids(6, 74)])
    got = torch.stack([dec.step_logits(forced[:, t:t + 1]).float() for t in range(6)], dim=1)  # (2, 6, V)
    want0 = model(torch.cat([torch.tensor([a]), forced[:1]], dim=1))["logits"][0, len(a):].float()
    want1 = model(torch.tensor([b + [6] + forced[1].tolist()]))["logits"][0, len(b) + 1:].float()
    assert torch.allclose(got[0], want0, atol=2e-4, rtol=1e-4), (got[0] - want0).abs().max()
    assert torch.allclose(got[1], want1, atol=2e-4, rtol=1e-4), (got[1] - want1).abs().max()


@torch.inference_mode()
def test_candidates_restart_in_their_own_rows(model: KotodamaModel) -> None:
    # bucket 4, n=3 each: the first request takes rows 0-2, the second is refilled into row 3 alone and starts
    # its other candidates there (rewinding to its prompt, no prefill) while rows 0-2 go to the third request
    prompts = [_ids(13, 60), _ids(30, 61), _ids(7, 62)]
    steps = [4, 7, 10]
    cold = SamplingParams(temperature=1e-4, repetition_penalty=1.5)
    eng = _engine(model)
    runner, rec = _runner(eng, widths=(4,), refill=True), _Rec()
    dec = runner.decoders[4]
    holds: dict[int, int] = {}  # row -> length of the prompt it was admitted from
    real_admit, real_restart = dec.admit, dec.restart

    def admit(rows, logits, params):
        holds.update({r: eng._pos_cpu for r in rows})
        return real_admit(rows, logits, params)

    def restart(rows, plen, logits, params):
        assert all(holds[r] == plen for r in rows), (rows, plen, holds)  # a row restarts only its own prompt
        return real_restart(rows, plen, logits, params)

    dec.admit, dec.restart = admit, restart
    for i, p in enumerate(prompts):
        runner.queue.put(rec.job(f"j{i}", p, 3, max_new=steps[i], params=cold))
    assert runner.run_batch() == 3 and runner.stats["restarts"] == 2
    assert rec.out["j1"].timing["rows_at_admission"] == 1.0
    wants = [_penalized_greedy(model, p, k, 1.5) for p, k in zip(prompts, steps)]
    assert len({tuple(w[1:]) for w in wants}) == 3  # the reference really depends on the prompt
    for i in range(3):
        assert rec.out[f"j{i}"].rows == [wants[i]] * 3, i
    hot = SamplingParams(temperature=1.4, repetition_penalty=1.3)  # restarted candidates sample independently
    runner2, rec2 = _runner(_engine(model), widths=(2,), refill=True), _Rec()
    runner2.queue.put(rec2.job("h", prompts[0], 2, max_new=12, params=hot))
    runner2.queue.put(rec2.job("w", prompts[1], 2, max_new=12, params=hot))
    torch.manual_seed(3)
    runner2.run_batch()
    assert len({tuple(r) for r in rec2.out["w"].rows}) == 2


def test_engine_lock_contention_is_visible(monkeypatch) -> None:
    from kotodama.serve import server

    async def main() -> list[bool]:
        lock = asyncio.Lock()
        monkeypatch.setattr(server, "_engine_lock", lock)
        seen = [server._engine_lock_contended()]
        async with lock:
            seen.append(server._engine_lock_contended())
            waiter = asyncio.create_task(lock.acquire())
            await asyncio.sleep(0)
            seen.append(server._engine_lock_contended())
        await waiter
        lock.release()
        seen.append(server._engine_lock_contended())
        return seen

    assert asyncio.run(main()) == [False, False, True, False]


@torch.inference_mode()
def test_a_failing_prefill_fails_only_its_request(model: KotodamaModel) -> None:
    eng = _engine(model)
    runner, rec = _runner(eng, widths=(4,)), _Rec()
    runner.queue.put(rec.job("bad", _ids(MAX_SEQ + 5, 1), 2))  # longer than the cache: the engine refuses it
    runner.queue.put(rec.job("ok", _ids(9, 2), 2))
    assert runner.run_batch() == 2
    assert isinstance(rec.out["bad"], ValueError) and isinstance(rec.out["ok"], XBatchResult)


def test_scheduler_runs_concurrent_submissions_and_refuses_wide_jobs(model: KotodamaModel) -> None:
    eng = _engine(model)
    runner = _runner(eng, widths=(4,))
    batches: list[int] = []

    async def run_on_worker(fn):
        batches.append(len(runner.queue))
        return fn()

    async def main() -> list[XBatchResult]:
        sched = XBatchScheduler(runner, asyncio.Lock(), run_on_worker)
        rec = _Rec()
        with pytest.raises(ValueError):
            await sched.submit(rec.job("wide", _ids(5, 1), 5))
        jobs = [rec.job(f"c{i}", _ids(7 + i, i), 2, max_new=4) for i in range(3)]
        return await asyncio.gather(*(sched.submit(j) for j in jobs))

    with torch.inference_mode():
        results = asyncio.run(main())
    assert [len(r.rows) for r in results] == [2, 2, 2]
    assert batches == [3, 1]  # the first batch took two requests (4 rows); the third waited for the next


class _CharTok:
    """Characters <-> ids inside the tiny vocab (4 + ord % 200)."""

    def encode(self, text, return_tensors=None, **_):
        ids = [4 + ord(c) % 200 for c in text]
        return torch.tensor([ids]) if return_tensors == "pt" else ids

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + (i - 4) % 26) for i in ids)


def test_server_routes_compatible_requests_through_the_batcher(monkeypatch, model: KotodamaModel) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from kotodama.serve import server

    eng = _engine(model, steer=True)
    runner = _runner(eng, widths=(4, 8))
    runner.decode = lambda ids: _CharTok().decode(ids)
    runner.stop_ids = frozenset({0})
    used: list[str] = []

    async def run_inline(fn, *args, **kwargs):
        used.append(getattr(fn, "__name__", "?"))
        return fn(*args, **kwargs)

    sched = XBatchScheduler(runner, asyncio.Lock(), run_inline)
    for k, v in {"_model": model, "_tokenizer": _CharTok(), "_engine": eng, "_multi": {}, "_xbatch": sched,
                 "_device": torch.device("cpu"), "_max_seq_len": MAX_SEQ, "_prefix_cache": False,
                 "_stop_token_ids": frozenset({0}), "_run_generation": run_inline}.items():
        monkeypatch.setattr(server, k, v)
    client = TestClient(server.app)
    r = client.post("/v1/completions", json={"prompt": "two friends talk", "max_tokens": 6, "n": 3,
                                             "temperature": 0.9, "top_p": 0, "return_token_ids": True,
                                             "stop": ["q"]})
    assert r.status_code == 200, r.text
    choices = r.json()["choices"]
    assert used == ["run_batch"] and len(choices) == 3
    assert all(1 <= len(c["token_ids"]["completion"]) <= 6 for c in choices)
    assert all("q" not in c["text"][:-1] for c in choices)  # a stop string ends the row where it appears
    used.clear()
    r = client.post("/v1/completions", json={"prompt": "two friends talk", "max_tokens": 4, "n": 2,
                                             "temperature": 0})  # greedy: the single-stream path
    assert r.status_code == 200 and used == ["engine_generate", "engine_generate"]
    used.clear()
    r = client.post("/v1/completions", json={"prompt": "x" * MAX_SEQ, "max_tokens": 4, "n": 2, "temperature": 0.9})
    assert r.status_code == 400 and used == []  # refused before it queues
    r = client.post("/v1/completions", json={"prompt": "hey", "max_tokens": 4, "n": 2, "temperature": 0.9,
                                             "vectors": [{"name": "nope", "alpha": 1.0}]})
    assert r.status_code == 400 and used == []  # steering without a bank: refused before it queues
    assert runner.stats["batches"] == 1 and eng.steer_buf.abs().sum() == 0


def test_signed_requests_leave_records_in_the_spool(monkeypatch, model: KotodamaModel, tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kotodama.model.multi_decode import CascadeDecoder
    from kotodama.model.signer import RecordSigner, read_records
    from kotodama.model.signing import ReducedRow, SignSpec, weights_digest
    from kotodama.serve import server
    from kotodama.serve.sign import SignSpool

    eng = _engine(model)
    stop = frozenset(range(4, 40))  # a broad stop set: some candidates end on a stop token, some on the budget
    runner = _runner(eng, widths=(4,), stop_ids=stop)
    runner.decode = lambda ids: _CharTok().decode(ids)

    async def run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    spool = SignSpool(tmp_path / "spool")
    sched = XBatchScheduler(runner, asyncio.Lock(), run_inline)
    for k, v in {"_model": model, "_tokenizer": _CharTok(), "_engine": eng, "_multi": {}, "_xbatch": sched,
                 "_device": torch.device("cpu"), "_max_seq_len": MAX_SEQ, "_prefix_cache": False,
                 "_stop_token_ids": stop, "_run_generation": run_inline, "_sign_spool": spool,
                 "_sign_writer": "replica-a", "_sign_weights": weights_digest(model)}.items():
        monkeypatch.setattr(server, k, v)
    client = TestClient(server.app)
    torch.manual_seed(5)
    body = {"prompt": "two friends talk", "max_tokens": 8, "n": 4, "temperature": 1.2, "top_p": 0,
            "return_token_ids": True, "sign": True, "sign_meta": {"round": 2, "node": "n7"}}
    r = client.post("/v1/completions", json=body)
    assert r.status_code == 200, r.text
    choices = r.json()["choices"]
    recs = list(read_records(tmp_path / "spool" / "records" / "replica-a.jsonl"))
    assert [c["sign_id"] for c in choices] == [rec.request_id for rec in recs] and len(recs) == 4
    ended_on_stop = 0
    for c, rec in zip(choices, recs):
        completion = c["token_ids"]["completion"]
        assert list(rec.prompt_ids) == c["token_ids"]["prompt"] and rec.weights == weights_digest(model)
        assert rec.meta["round"] == 2 and rec.meta["node"] == "n7" and rec.meta["served_by"] == "replica-a"
        if len(rec.generated_ids) == len(completion) + 1:
            ended_on_stop += 1
            assert rec.generated_ids[-1] in stop and list(rec.generated_ids[:-1]) == completion
        else:
            assert list(rec.generated_ids) == completion and len(completion) == 8
    assert ended_on_stop >= 1
    assert server._sign_info()["stats"]["records"] >= 4
    # the records sign: each row's span is its fed positions (its generated ids but the last)

    class Steps:
        reducer_identity = "steps-v1"

        def reduce(self, rows, *, token_ids, engine):
            return [ReducedRow(features=[float(r.positions)], names_digest="s", lane_id="s") for r in rows]

    dec = CascadeDecoder(eng, 4, groups=4, group_rows=2, tail_tokens=16, seeded=True,
                         sign=SignSpec((0, 5, 10), tail=12, depth=True, attn=True, width=MAX_SEQ))
    dec.compile_step(regional=True, backend="aot_eager")
    with torch.inference_mode():
        summary = spool.drain_once(RecordSigner(dec, Steps(), compile_mode="aot_eager"))
    signed = [p for p in (tmp_path / "spool" / "signed").iterdir()]
    assert len(signed) == 1
    rows = [__import__("json").loads(line) for line in open(signed[0] / "rows.jsonl")]
    fed = {rec.request_id: len(rec.generated_ids) - 1 for rec in recs}
    for row in rows:
        if fed[row["sign_id"]] >= 1:
            assert row["reducer_meta"] == {} and "index" in row
        else:
            assert "no fed position" in row["skipped"]
    assert summary["signed"] == sum(n >= 1 for n in fed.values())
    # what the sign path cannot take is refused before anything runs
    # a law the batcher does not sample (greedy) is served serially, and its records still go to the spool
    before = server._sign_stats["records"]
    greedy = client.post("/v1/completions", json={**body, "temperature": 0, "n": 2})
    assert greedy.status_code == 200 and all(c["sign_id"] for c in greedy.json()["choices"])
    assert server._sign_stats["records"] == before + 2
    # a signed stream streams and leaves its record (the final event carries its sign_id)
    before = server._sign_stats["records"]
    r = client.post("/generate", json={"prompt": "hi", "sign": True, "stream": True, "max_new_tokens": 4})
    assert r.status_code == 200 and '"sign_id"' in r.text and server._sign_stats["records"] == before + 1
    monkeypatch.setattr(server, "_sign_spool", None)
    r = client.post("/v1/completions", json=body)
    assert r.status_code == 400 and "--sign_spool" in r.text


@torch.inference_mode()
def test_seeded_jobs_prefill_cold(monkeypatch, model: KotodamaModel) -> None:
    from kotodama.serve import server

    eng = _engine(model)
    calls: list[dict] = []

    def prefill(ids, **kw):
        calls.append(kw)
        return eng.prefill_reference(ids), None

    wide = MultiSampleDecoder(eng, 4, seeded=True)
    runner = XBatchRunner({4: wide}, prefill=prefill, set_steer=eng.set_steer, decode=lambda ids: "",
                          stop_ids=frozenset(), max_seq_len=MAX_SEQ, device=torch.device("cpu"))
    rec = _Rec()
    seeded = rec.job("s", _ids(8, 1), 2, params=SamplingParams(temperature=0.9, repetition_penalty=1.2))
    seeded.seed = 11
    runner.queue.put(seeded)
    runner.queue.put(rec.job("u", _ids(8, 1), 2))
    runner.run_batch()
    assert calls == [{"fresh": True}, {}]
    # the server's router: fresh skips the prefix cache even when it is on
    used: list[str] = []
    monkeypatch.setattr(server, "_engine", eng)
    monkeypatch.setattr(server, "_prefix_pool", None)
    monkeypatch.setattr(server, "_prefix_cache", True)
    monkeypatch.setattr(eng, "prefill", lambda ids, slot=0: used.append("cold") or eng.prefill_reference(ids))
    monkeypatch.setattr(eng, "prefill_cached", lambda ids, slot=0: used.append("cached") or (None, {}))
    server._engine_prefill(torch.tensor([_ids(8, 1)]), fresh=True)
    server._engine_prefill(torch.tensor([_ids(8, 1)]))
    assert used == ["cold", "cached"]


@torch.inference_mode()
def test_seeded_jobs_decode_on_the_widest_bucket(model: KotodamaModel) -> None:
    eng = _engine(model)
    wide = MultiSampleDecoder(eng, 8, seeded=True)
    decs = {4: MultiSampleDecoder(eng, 4, kv_from=wide, seeded=True), 8: wide}
    widths: list[int] = []
    for d in decs.values():
        real = d.admit

        def admit(*a, _real=real, _d=d, **k):
            widths.append(_d.rows)
            return _real(*a, **k)

        d.admit = admit
    late: list = []

    def prefill(ids, **kw):
        while late:  # a job that arrives while the batch is running
            runner.queue.put(late.pop())
        return eng.prefill_reference(ids), None

    runner = XBatchRunner(decs, prefill=prefill, set_steer=eng.set_steer, decode=lambda ids: "", stop_ids=frozenset(),
                          max_seq_len=MAX_SEQ, device=torch.device("cpu"), refill=True)
    rec = _Rec()
    hot = SamplingParams(temperature=0.9, repetition_penalty=1.2)

    def seeded(name: str, seed: int):
        job = rec.job(name, _ids(8, seed), 2, params=hot, max_new=3)
        job.seed = seed
        return job

    runner.queue.put(seeded("a", 1))  # 2 rows: would fit the 4-wide bucket
    runner.run_batch()
    assert widths == [8]
    widths.clear()
    runner.queue.put(rec.job("u", _ids(8, 2), 2, params=hot, max_new=6))  # unseeded: the narrow bucket
    late.append(seeded("b", 3))  # arrives mid-batch: refill must not pull it into the 4-wide batch
    runner.run_batch()
    assert widths == [4] and "b" not in rec.out
    runner.run_batch()
    assert widths == [4, 8] and "b" in rec.out


class _GatherStub:
    """The parts of an XBatchRunner the scheduler's pump reads: a queue, the bucket widths, and a run_batch that
    takes one group and records its rows."""

    def __init__(self, widths=(4, 8, 16)) -> None:
        from types import SimpleNamespace

        self.queue = XBatchQueue()
        self.max_rows = max(widths)
        self._widths = sorted(widths)
        self._ns = SimpleNamespace
        self.batches: list[int] = []

    def bucket_for(self, rows: int):
        return self._ns(rows=next((w for w in self._widths if w >= rows), self.max_rows))

    def run_batch(self) -> int:
        jobs = self.queue.take_group(self.max_rows)
        self.batches.append(sum(j.n for j in jobs))
        for job in jobs:
            XBatchRunner._deliver(job, "ok")
        return len(jobs)


def _gather_job(n: int, seed=None) -> XBatchJob:
    return XBatchJob(key="k", prompt_ids=[1], n=n, max_new_tokens=1,
                     params=SamplingParams(temperature=1.0, repetition_penalty=1.0), done=lambda _o: None, seed=seed)


def _burst(gather_s: float, jobs: int, n: int, spread_s: float, seed=None) -> tuple[list[int], dict]:
    """``jobs`` requests of ``n`` rows arriving ``spread_s`` apart; the batches' row counts and the gather stats."""
    stub = _GatherStub()

    async def go():
        async def run_inline(fn):
            return fn()

        sched = XBatchScheduler(stub, asyncio.Lock(), run_inline, gather_s=gather_s, gather_quiet_s=0.01)

        async def one(i):
            await asyncio.sleep(i * spread_s)
            return await sched.submit(_gather_job(n, seed))

        out = await asyncio.gather(*(one(i) for i in range(jobs)))
        assert out == ["ok"] * jobs
        return sched.gather_stats

    stats = asyncio.run(go())
    return stub.batches, stats


def test_a_burst_waits_to_share_one_batch_instead_of_draining_through_a_narrow_bucket() -> None:
    # 4 requests of 2 rows, 2 ms apart: without the gather the first one starts a batch alone (a narrow bucket)
    batches, stats = _burst(0.0, jobs=4, n=2, spread_s=0.002)
    assert batches[0] == 2 and stats["gathers"] == 0
    batches, stats = _burst(0.2, jobs=4, n=2, spread_s=0.002)
    assert batches == [8] and stats["gathers"] == 1 and stats["grew"] == 1
    # the wait ends once the waiting rows already take the widest bucket: no gather at all for a full burst
    batches, stats = _burst(0.2, jobs=1, n=16, spread_s=0.0)
    assert batches == [16] and stats["gathers"] == 0
    # seeded jobs always decode on the widest bucket: nothing to gather for
    batches, stats = _burst(0.2, jobs=1, n=2, spread_s=0.0, seed=7)
    assert batches == [2] and stats["gathers"] == 0
    # a lone request waits only the quiet interval, never the whole window
    batches, stats = _burst(0.5, jobs=1, n=2, spread_s=0.0)
    assert batches == [2] and stats["gathers"] == 1 and stats["waited_s"] < 0.1
