"""The in-loop setbind suite: item loading, the set-binding read against the benchmark's own grid runner and
``set_metrics`` (same items, same model, same numbers), the battery pin check, the summary rule, the evaluator
schedule end to end, and 2-process gloo == single process."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kotodama.serve.logprobs import score_logprob_items
from kotodama.training.inloop import (SETBIND_CONDITION, InloopConfig, InloopEvaluator, parse_suites,
                                      setbind_summary)

try:   # the setbind suite comes from the binding benchmark (the bench extra)
    from binding_bench.binding import batteries as bt
    from binding_bench.bindprobes import grid as bp_grid
    from binding_bench.bindprobes import items as bp_items
    from binding_bench.bindprobes.setbind import set_metrics
except ImportError:
    bt = bp_grid = bp_items = set_metrics = None  # type: ignore[assignment]
needs_bench = pytest.mark.skipif(bp_items is None, reason="the binding benchmark is not installed (kotodama[bench])")

VOCAB = 49152  # the presets tokenizer's size: candidates are real token ids


@pytest.fixture(scope="module")
def tok():
    try:
        from transformers import AutoTokenizer

        from kotodama.presets import TOKENIZER_NAME
        return AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except Exception as exc:  # offline without a cache
        pytest.skip(f"tokenizer unavailable: {exc}")


def _tiny(seed: int = 0) -> torch.nn.Module:
    from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
    torch.manual_seed(seed)
    return KotodamaModel(KotodamaModelConfig(
        hidden_size=64, num_layers=2, num_attention_heads=2, num_kv_heads=1, head_dim=32,
        intermediate_size=128, vocab_size=VOCAB, max_position_embeddings=1024,
        attn_impl="sdpa", use_liger=False, attn_res=True, attn_res_boundaries=[0, 1])).eval()


@pytest.fixture(scope="module")
def items_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Every setbind cell (static and with transfers), two base items each, plus a few attr items: the suite
    must keep only the static setbind ones."""
    split = bp_items.load_pools(None)
    its = bp_items.generate_family("setbind", split, 2, seed=11) + bp_items.generate_family("attr", split, 1, seed=11)
    path = tmp_path_factory.mktemp("bp") / "items.jsonl"
    path.write_text("".join(json.dumps(it.to_dict()) + "\n" for it in its))
    return path


def _static(path: Path) -> list[Any]:
    return [it for it in bp_items.load_items(path) if it.family == "setbind" and it.meta["n_transfers"] == 0]


def _ev(tmp_path: Path, tok: Any, items: Path, **kw: Any) -> InloopEvaluator:
    cfg = InloopConfig(**{"suites": ["setbind"], "setbind_items": str(items), "max_batch": 16,
                          "max_batch_tokens": 4096, **kw})
    return InloopEvaluator(cfg, out_dir=tmp_path, rank=0, world_size=1, device=torch.device("cpu"),
                           max_seq_len=1024, vocab_size=VOCAB, tokenizer=tok)


# ── flags and loading ───────────────────────────────────────────────────────────

def test_setbind_flag_maps_to_config(monkeypatch):
    from kotodama.training import train as train_mod
    assert parse_suites("riders,setbind") == ["riders", "setbind"]
    monkeypatch.setattr("sys.argv", ["train", "--inloop_evals", "setbind", "--inloop_setbind_items", "/x.jsonl"])
    cfg = InloopConfig.from_args(train_mod.parse_args())
    assert cfg.suites == ["setbind"] and cfg.setbind_items == "/x.jsonl"
    monkeypatch.setattr("sys.argv", ["train"])
    assert InloopConfig.from_args(train_mod.parse_args()).setbind_items is None


@needs_bench
def test_setbind_loads_only_static_setbind_items(tok, tmp_path, items_file):
    ev = _ev(tmp_path, tok, items_file)
    want = _static(items_file)
    assert [it.item_id for it in ev.s_items] == [it.item_id for it in want]
    assert {it.meta["k"] for it in ev.s_items} == {1, 2, 3}
    # the payload is the benchmark's own request item for the 3-shot condition
    assert ev.s_jobs == [bp_grid.build_payload_item(it, SETBIND_CONDITION) for it in want]
    assert all(j["prompt"].count(bp_items.family("setbind").SEPARATOR) == 3 for j in ev.s_jobs)
    assert ev.s_lengths == [len(tok.encode(j["prompt"])) for j in ev.s_jobs]
    note = ev.notes["setbind"]
    assert note["battery"] is None and note["n_prompts"] == len(want) and note["condition"] == "3shot"
    assert sum(note["n_by_k"].values()) == len(want)


@needs_bench
def test_setbind_refusals(tok, tmp_path, items_file):
    with pytest.raises(ValueError, match="setbind_items"):
        _ev(tmp_path, tok, items_file, setbind_items=None)
    dynamic = tmp_path / "dynamic.jsonl"
    dynamic.write_text("".join(ln + "\n" for ln in items_file.read_text().splitlines()
                               if json.loads(ln)["meta"].get("n_transfers", 0) != 0))
    with pytest.raises(ValueError, match="no static setbind items"):
        _ev(tmp_path, tok, dynamic)
    with pytest.raises(ValueError, match="language vocab"):          # an NCA-phase vocab
        InloopEvaluator(InloopConfig(suites=["setbind"], setbind_items=str(items_file)), out_dir=tmp_path,
                        rank=0, world_size=1, device=torch.device("cpu"), max_seq_len=1024, vocab_size=1000,
                        tokenizer=tok)


@needs_bench
def test_setbind_checks_a_pinned_battery(tok, tmp_path, items_file, monkeypatch):
    digest = bt.sha256(items_file)
    good = bt.Battery("bp-test", "bindprobes", "x/items.jsonl", digest, None, "test",
                      render_sha256=bp_items.render_sha256(bp_items.load_items(items_file)))
    monkeypatch.setattr(bt, "BATTERIES", {**bt.BATTERIES, "bp-test": good})
    assert _ev(tmp_path, tok, items_file).notes["setbind"]["battery"] == "bp-test"
    bad = bt.Battery("bp-test", "bindprobes", "x/items.jsonl", digest, None, "test", render_sha256="0" * 64)
    monkeypatch.setattr(bt, "BATTERIES", {**bt.BATTERIES, "bp-test": bad})
    with pytest.raises(bt.PinError):
        _ev(tmp_path, tok, items_file)


# ── the read ────────────────────────────────────────────────────────────────────

def test_setbind_summary_rule():
    rows = [{"k": 1, "hit1": 1.0, "exactk": 1.0, "set_margin": 0.5},
            {"k": 1, "hit1": 0.0, "exactk": 0.0, "set_margin": -1.5},
            {"k": 1, "hit1": 1.0, "exactk": 1.0, "set_margin": 0.25},
            {"k": 2, "hit1": 1.0, "exactk": 0.0, "set_margin": -0.5},
            {"k": 3, "hit1": 0.0, "exactk": 0.0, "set_margin": -2.0},
            {"k": 3, "hit1": 1.0, "exactk": 1.0, "set_margin": 0.5}]
    m = setbind_summary(rows)
    assert m["setbind/k1"] == pytest.approx(2 / 3) and m["setbind/n_k1"] == 3.0
    assert m["setbind/k1_margin"] == pytest.approx(-0.25)
    assert m["setbind/k2"] == 1.0 and m["setbind/k3"] == 0.5
    assert m["setbind/k23"] == pytest.approx(2 / 3) and m["setbind/n_k23"] == 3.0
    assert m["setbind/k23_exactk"] == pytest.approx(1 / 3)
    assert m["setbind/k23_margin"] == pytest.approx(-2 / 3)
    only_k1 = setbind_summary(rows[:3])
    assert "setbind/k23" not in only_k1 and "setbind/k2" not in only_k1 and only_k1["setbind/k1"] == m["setbind/k1"]


@needs_bench
def test_set_metrics_on_designed_logprobs():
    """The rule the suite applies to each row, on logprobs whose answer is known (keys as scored, leading space)."""
    lps = {" cup": -1.0, " pen": -0.5, " key": -3.0, " bell": -0.2}
    assert set_metrics(lps, ["bell"])["hit1"] == 1.0
    assert set_metrics(lps, ["cup"])["hit1"] == 0.0
    two = set_metrics(lps, ["bell", "pen"])
    assert two["hit1"] == 1.0 and two["exactk"] == 1.0 and two["margin"] == pytest.approx(0.5)
    assert set_metrics(lps, ["bell", "key"])["exactk"] == 0.0


def _bench_grid_reads(tmp_path: Path, tok: Any, model: torch.nn.Module, items: Path, monkeypatch) -> dict[str, Any]:
    """The benchmark's own runner over ``items`` (its /logprobs call answered by the server's scorer on ``model``,
    one prompt per forward), read with the set-binding rule: setbind rows, 3-shot, no transfers, k = 1 (and
    k = 2, 3), mean of ``set_metrics(...)["hit1"]``."""

    def request_batch(endpoint: str, payload: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
        with torch.no_grad():
            return [score_logprob_items(lambda b: model(b)["logits"], tok, [p], top_n, "cpu",
                                        max_seq_len=1024)[0] for p in payload]

    monkeypatch.setattr(bp_grid, "request_batch", request_batch)
    monkeypatch.setattr(bp_grid, "check_health", lambda endpoint, log: {"status": "ok"})
    out = tmp_path / "grid.jsonl"
    bp_grid.run_grid("tiny", bp_items.load_items(items), out, endpoint="http://unused", conditions=("3shot",),
                     families=("setbind",), log=lambda s: None)
    rows = [json.loads(ln) for ln in out.read_text().splitlines()]
    sb = [r for r in rows if r["family"] == "setbind" and r["condition"] == "3shot" and r["n_transfers"] == 0]
    hit = lambda keep: [set_metrics(r["cand_lps"], r["gold_set"])["hit1"] for r in sb if keep(r)]  # noqa: E731
    k1, k23 = hit(lambda r: r["k"] == 1), hit(lambda r: r["k"] > 1)
    return {"k1": sum(k1) / len(k1), "k23": sum(k23) / len(k23), "rows": {r["item_id"]: r for r in sb}}


@needs_bench
def test_setbind_matches_the_benchmark_grid_read(tok, tmp_path, items_file, monkeypatch):
    """Same items, same model: the in-loop k1 / k23 equal the benchmark grid's set-binding read exactly, and
    every row's candidate logprobs are the grid row's."""
    model = _tiny(3)
    ref = _bench_grid_reads(tmp_path, tok, model, items_file, monkeypatch)
    ev = _ev(tmp_path, tok, items_file, max_batch=1)
    m = ev.run(0, model)
    assert m is not None and m["inloop/errors"] == 0.0
    assert m["inloop/setbind/k1"] == ref["k1"] and m["inloop/setbind/k23"] == ref["k23"]
    lines = [json.loads(x) for x in (tmp_path / "inloop" / "step_00000000.jsonl").read_text().splitlines()]
    rows = [x for x in lines if x["suite"] == "setbind"]
    assert len(rows) == len(ref["rows"]) == len(ev.s_jobs)
    for r in rows:
        g = ref["rows"][r["item_id"]]
        assert r["cand_lps"] == g["cand_lps"] and r["gold_set"] == g["gold_set"] and r["k"] == g["k"]
        assert r["hit1"] == set_metrics(g["cand_lps"], g["gold_set"])["hit1"]
    # batching changes nothing that the read sees
    batched = _ev(tmp_path / "b", tok, items_file, max_batch=16).run(0, model)
    assert batched is not None and batched["inloop/setbind/k1"] == m["inloop/setbind/k1"]


# ── evaluator schedule and DDP ──────────────────────────────────────────────────

@needs_bench
def test_setbind_in_the_schedule(tok, tmp_path, items_file):
    logged: list[tuple[int, dict[str, float]]] = []
    cfg = InloopConfig(suites=["setbind", "induction"], setbind_items=str(items_file), max_batch=16,
                       max_batch_tokens=4096, induction_n_seqs=4, induction_len=16)
    ev = InloopEvaluator(cfg, out_dir=tmp_path, rank=0, world_size=1, device=torch.device("cpu"),
                         max_seq_len=1024, vocab_size=VOCAB, tokenizer=tok,
                         log_fn=lambda step, **m: logged.append((step, m)))
    pending = ev.plan(total_steps=100, decay_start=80, final_step=99)
    assert pending[0] == 0 and pending[-1] == 99
    model = _tiny().train()
    m = ev.run(99, model, tokens_consumed=7)
    assert m is not None and model.training and m["inloop/errors"] == 0.0
    assert len(logged) == 1 and logged[0][0] == 99                  # one batched hm_log call
    for k in ("k1", "k2", "k3", "k23"):
        assert 0.0 <= m[f"inloop/setbind/{k}"] <= 1.0
    assert m["inloop/setbind/n_k1"] + m["inloop/setbind/n_k23"] == len(ev.s_jobs)
    assert "inloop/setbind/time_s" in m and "inloop/induction/induction_gain" in m
    lines = [json.loads(x) for x in (tmp_path / "inloop" / "step_00000099.jsonl").read_text().splitlines()]
    assert lines[0]["prep"]["setbind"]["n_prompts"] == len(ev.s_jobs)
    assert len([x for x in lines if x["suite"] == "setbind"]) == len(ev.s_jobs)
    resumed = InloopEvaluator(cfg, out_dir=tmp_path, rank=0, world_size=1, device=torch.device("cpu"),
                              max_seq_len=1024, vocab_size=VOCAB, tokenizer=tok)
    assert 99 not in resumed.plan(total_steps=100, decay_start=80, final_step=99)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, world: int, port: int, out_dir: str, items: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from transformers import AutoTokenizer

        from kotodama.presets import TOKENIZER_NAME
        ev = InloopEvaluator(InloopConfig(suites=["setbind"], setbind_items=items, max_batch=16,
                                          max_batch_tokens=4096),
                             out_dir=out_dir, rank=rank, world_size=world, device=torch.device("cpu"),
                             max_seq_len=1024, vocab_size=VOCAB,
                             tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_NAME))
        m = ev.run(0, _tiny(5))
        if rank == 0:
            Path(out_dir, "metrics.json").write_text(json.dumps(m))
        else:
            assert m is None
    finally:
        dist.destroy_process_group()


@needs_bench
def test_setbind_two_process_gloo_matches_single(tok, tmp_path, items_file):
    single = _ev(tmp_path / "single", tok, items_file).run(0, _tiny(5))
    assert single is not None
    out = tmp_path / "ddp"
    mp.spawn(_worker, args=(2, _free_port(), str(out), str(items_file)), nprocs=2, join=True)
    got = json.loads((out / "metrics.json").read_text())
    for k, v in single.items():
        if k.endswith("time_s"):
            continue
        if k.endswith("_margin"):
            assert got[k] == pytest.approx(v, rel=1e-4, abs=1e-5), k
        else:
            assert got[k] == v, k                                    # discrete reads: exact
    ids = lambda p: [json.loads(x).get("item_id") for x in p.read_text().splitlines()[1:]]  # noqa: E731
    assert ids(tmp_path / "single" / "inloop" / "step_00000000.jsonl") == ids(out / "inloop" / "step_00000000.jsonl")
