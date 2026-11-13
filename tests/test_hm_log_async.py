"""hm_log posts from a background thread (a synchronous POST on rank 0 stalled every DDP rank at the next all-reduce):
points arrive in order, hm_flush waits for them, a full queue drops instead of blocking, and nothing raises."""

from __future__ import annotations

import queue
import sys
import threading
import time
import types

from kotodama.training import train


class _FakeClient:
    posts: list[tuple[str, dict]] = []
    gate = threading.Event()

    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url

    def post(self, path: str, json: dict) -> None:
        _FakeClient.gate.wait(5)
        _FakeClient.posts.append((path, json))


def _fresh(monkeypatch, maxsize: int = 4096):
    _FakeClient.posts = []
    _FakeClient.gate = threading.Event()
    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(Client=_FakeClient))
    monkeypatch.setattr(train, "_hm_job_id", "job123")
    monkeypatch.setattr(train, "_hm_queue", None)
    monkeypatch.setattr(train, "_HM_QUEUE_MAX", maxsize)


def test_async_in_order_and_flush(monkeypatch):
    _fresh(monkeypatch)
    t0 = time.time()
    for step in range(5):
        train.hm_log(step, loss=float(step), skip=None)
    assert time.time() - t0 < 1.0          # never waits on the (blocked) network
    _FakeClient.gate.set()
    train.hm_flush(timeout=5)
    assert [p[1]["step"] for p in _FakeClient.posts] == [0, 1, 2, 3, 4]
    assert all(p[0] == "/api/v1/jobs/job123/metrics" and "skip" not in p[1]["metrics"] for p in _FakeClient.posts)


def test_full_queue_drops_instead_of_blocking(monkeypatch):
    _fresh(monkeypatch, maxsize=2)
    t0 = time.time()
    for step in range(50):
        train.hm_log(step, loss=1.0)
    assert time.time() - t0 < 1.0
    _FakeClient.gate.set()
    train.hm_flush(timeout=5)
    assert 1 <= len(_FakeClient.posts) <= 4


def test_noop_without_job(monkeypatch):
    monkeypatch.setattr(train, "_hm_job_id", None)
    monkeypatch.setattr(train, "_hm_queue", None)
    train.hm_log(0, loss=1.0)
    assert train._hm_queue is None


def test_metric_prefix_gives_each_run_its_own_series(monkeypatch):
    q: queue.Queue = queue.Queue()
    monkeypatch.setattr(train, "_hm_job_id", "job")
    monkeypatch.setattr(train, "_hm_queue", q)
    monkeypatch.setattr(train, "_hm_prefix", "fsdp2-resume")
    train.hm_log(7, **{"train/loss": 2.5, "skipped": None})
    assert q.get_nowait() == (7, {"fsdp2-resume/train/loss": 2.5})
    monkeypatch.setattr(train, "_hm_prefix", "")
    train.hm_log(8, **{"train/loss": 2.4})
    assert q.get_nowait() == (8, {"train/loss": 2.4})
