"""The trainer's garbage-collector timer counts passes by generation and resets on take()."""

from __future__ import annotations

import gc

from kotodama.training.train import GcTimer


def test_gc_timer_counts_full_collections_and_resets():
    timer = GcTimer()
    gc.callbacks.append(timer)
    try:
        timer.take()
        gc.collect()  # a full (generation 2) pass
        gc.collect()
        got = timer.take()
        assert got["gc/gen2_count"] >= 2.0
        assert got["gc/gen2_ms"] >= 0.0
        assert set(got) == {f"gc/gen{g}_{k}" for g in range(3) for k in ("count", "ms")}
        assert timer.take()["gc/gen2_count"] == 0.0  # reset
    finally:
        gc.callbacks.remove(timer)
