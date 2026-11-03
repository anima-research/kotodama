"""Gateway pieces that need no fleet: the example config loads and validates its routing targets, the GPU choice
respects a scheduler's CUDA_VISIBLE_DEVICES grant, and cache-affinity keys follow a conversation's stable head."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from kotodama.serve import gateway, server  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[1] / "configs" / "gateway.example.yaml"


def test_example_config_loads() -> None:
    cfg = gateway.load_config(str(EXAMPLE), None)
    assert cfg.completion_pool in cfg.pool_keys()
    assert Path(cfg.workdir).is_absolute()
    assert gateway.load_config(str(EXAMPLE), [3]).gpus == [3]


def test_bad_routing_target_refused(tmp_path) -> None:
    bad = tmp_path / "g.yaml"
    bad.write_text(EXAMPLE.read_text().replace("completion_pool: base", "completion_pool: nope", 1))
    with pytest.raises(SystemExit, match="routing target"):
        gateway.load_config(str(bad), None)


def test_effective_gpus_intersects_the_grant(monkeypatch) -> None:
    cfg = gateway.load_config(str(EXAMPLE), [0, 1, 2])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert gateway.effective_gpus(cfg) == [0, 1, 2]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2,5")
    assert gateway.effective_gpus(cfg) == [1, 2]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")
    assert gateway.effective_gpus(cfg) == [6, 7]  # nothing preferred is granted: use the grant


def test_affinity_key_is_the_conversation_head() -> None:
    sys_msg = {"role": "system", "content": "s"}
    turn1 = {"messages": [sys_msg, {"role": "user", "content": "hello"}]}
    turn2 = {"messages": [sys_msg, {"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"},
                          {"role": "user", "content": "more"}]}
    other = {"messages": [sys_msg, {"role": "user", "content": "different"}]}
    assert gateway._affinity_key(turn1) == gateway._affinity_key(turn2) != gateway._affinity_key(other)
    assert gateway._affinity_key({"user": "u1", "prompt": "x"}) == "user:u1"
    assert gateway._affinity_key({"prompt": "p" * 300}) == "p" * 256
    assert gateway._affinity_key({}) is None


TSUMUGI = Path(__file__).resolve().parents[1] / "configs" / "gateway-tsumugi.example.yaml"


def test_a_pool_can_be_a_serve_profile() -> None:
    cfg = gateway.load_config(str(TSUMUGI), [0, 1])
    (spec,) = cfg.models
    assert spec.profile == "tsumugi-1b"
    assert (spec.served_model_name, spec.model_size, spec.mode) == ("kotodama-tsumugi-1b", "1b-tsumugi", "base")
    assert spec.checkpoint == "hf:LuxiaSL/kotodama-tsumugi-1b"
    # the grace covers a replica's cold start: at least the server's startup budget
    budget = server.build_arg_parser().parse_args(["--checkpoint", "x"]).startup_budget_s
    assert cfg.warmup_grace_s >= budget and cfg.affinity_max_inflight > 0


def test_one_pool_is_one_spec_and_a_spec_needs_its_fields(tmp_path) -> None:
    dup = tmp_path / "dup.yaml"
    dup.write_text("models:\n  - {key: base, profile: tsumugi-1b}\n  - {key: base, profile: tsumugi-1b}\n"
                   "chat_pool: base\ncompletion_pool: base\n")
    with pytest.raises(SystemExit, match="configured twice"):
        gateway.load_config(str(dup), None)
    bare = tmp_path / "bare.yaml"
    bare.write_text("models:\n  - {key: base, checkpoint: /x.pt}\nchat_pool: base\ncompletion_pool: base\n")
    with pytest.raises(SystemExit, match="sets no"):
        gateway.load_config(str(bare), None)


def test_replicas_run_the_profile(monkeypatch, tmp_path) -> None:
    cfg = gateway.load_config(str(TSUMUGI), [0])
    cfg.workdir = str(tmp_path)
    fleet = gateway.Fleet(cfg)
    seen = {}

    class _Proc:
        pid = 1

        def poll(self):
            return None

    def popen(cmd, **kw):
        seen["cmd"], seen["env"] = cmd, kw["env"]
        return _Proc()

    monkeypatch.setattr(gateway.subprocess, "Popen", popen)
    fleet.spawn(fleet.replicas[0])
    cmd = seen["cmd"]
    assert cmd[cmd.index("--profile") + 1] == "tsumugi-1b"
    assert cmd[cmd.index("--served_model_name") + 1] == "kotodama-tsumugi-1b"
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "0"


def test_routes_pass_through_with_their_status(monkeypatch, tmp_path) -> None:
    """Scoring routes are proxied, a backend 4xx reaches the client unchanged, and /info names the fleet."""
    import httpx
    from fastapi.testclient import TestClient

    cfg = gateway.load_config(str(TSUMUGI), [0, 1])
    cfg.workdir = str(tmp_path)
    fleet = gateway.Fleet(cfg)
    for r in fleet.replicas:
        r.healthy = True
    calls = []

    def backend(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/v1/completions":
            return httpx.Response(400, json={"detail": "Prompt too long: 9999 tokens"})
        return httpx.Response(200, json={"ok": request.url.path})

    # module globals the CLI sets at startup (annotated, unassigned until then)
    monkeypatch.setattr(gateway, "_cfg", cfg, raising=False)
    monkeypatch.setattr(gateway, "_fleet", fleet, raising=False)
    monkeypatch.setattr(gateway, "_http", httpx.AsyncClient(transport=httpx.MockTransport(backend)), raising=False)
    client = TestClient(gateway.app)  # no lifespan: nothing is spawned
    for path in ("/score_tokens", "/logprobs", "/generate"):
        r = client.post(path, json={"items": [], "prompt": "x"})
        assert r.status_code == 200 and r.json() == {"ok": path}
    r = client.post("/v1/completions", json={"prompt": "x" * 10})
    assert r.status_code == 400 and "too long" in r.text  # a client error is not retried or turned into a 5xx
    info = client.get("/info").json()
    assert info["name"] == "tsumugi-fleet" and info["pools"]["base"]["profile"] == "tsumugi-1b"
    assert info["pools"]["base"]["replicas_ready"] == 2
    assert calls == ["/score_tokens", "/logprobs", "/generate", "/v1/completions"]
    assert fleet.global_inflight == 0  # every request released its replica


def test_native_routes_route_on_model_and_forward_without_it(monkeypatch, tmp_path) -> None:
    """/generate, /score_tokens and /logprobs take no `model` field: the gateway picks the pool by it and drops it."""
    import json as _json

    import httpx
    from fastapi.testclient import TestClient

    cfg = gateway.load_config(str(TSUMUGI), [0, 1])
    cfg.models.append(cfg.models[0].model_copy(update={"key": "sft", "served_model_name": "r1sft", "gpus": [1]}))
    cfg.models[0] = cfg.models[0].model_copy(update={"gpus": [0]})
    cfg.workdir = str(tmp_path)
    fleet = gateway.Fleet(cfg)
    for r in fleet.replicas:
        r.healthy = True
    seen = []

    def backend(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.port, request.url.path, _json.loads(request.content)))
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(gateway, "_cfg", cfg, raising=False)
    monkeypatch.setattr(gateway, "_fleet", fleet, raising=False)
    monkeypatch.setattr(gateway, "_http", httpx.AsyncClient(transport=httpx.MockTransport(backend)), raising=False)
    client = TestClient(gateway.app)
    sft_port = fleet.by_pool["sft"][0].port
    for path in ("/generate", "/score_tokens", "/logprobs"):
        assert client.post(path, json={"prompt": "x", "model": "r1sft"}).status_code == 200
        port, got_path, body = seen[-1]
        assert port == sft_port and got_path == path and body == {"prompt": "x"}
    assert client.post("/generate", json={"prompt": "x", "model": "sft"}).status_code == 200  # a pool key routes too
    assert seen[-1][0] == sft_port and "model" not in seen[-1][2]
    r = client.post("/generate", json={"prompt": "x", "model": "nope"})
    assert r.status_code == 400 and "not served" in r.text and len(seen) == 4
    # the OpenAI routes keep the field (the server's OpenAI surface accepts it)
    client.post("/v1/completions", json={"prompt": "x", "model": "r1sft"})
    assert seen[-1][2]["model"] == "r1sft" and fleet.global_inflight == 0


def test_token_id_prompts_have_an_affinity_key() -> None:
    a = gateway._affinity_key({"prompt": list(range(100))})
    assert a == gateway._affinity_key({"prompt_ids": list(range(64)) + [7]}) and a.startswith("ids:")


def test_a_pool_carries_its_own_server_flags(monkeypatch, tmp_path) -> None:
    cfg = gateway.load_config(str(TSUMUGI), [0])
    cfg.workdir = str(tmp_path)
    cfg.models[0].args = ["--steer-npz", "/b/bank.npz,/b/norms.npz"]
    fleet = gateway.Fleet(cfg)
    seen = {}

    class _Proc:
        pid = 1

        def poll(self):
            return None

    monkeypatch.setattr(gateway.subprocess, "Popen", lambda cmd, **kw: seen.setdefault("cmd", cmd) and _Proc())
    fleet.spawn(fleet.replicas[0])
    cmd = seen["cmd"]
    assert cmd[-2:] == ["--steer-npz", "/b/bank.npz,/b/norms.npz"]  # after the profile and fleet flags
