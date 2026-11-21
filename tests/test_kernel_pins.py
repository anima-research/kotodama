"""Pinned kernel configs (kotodama.model.kernel_pins): the choice logic on stand-in launchers, and the patch of
inductor's runtime autotuner, without a GPU.

What is pinned here:
  * replay keeps exactly the recorded config, refuses a kernel the table does not hold (or, allowed, takes its first
    config and records the miss, which changes the digest), and refuses a pinned config the kernel does not have;
  * record mode leaves the choice to the autotuner and records it, and a second instance of a recorded kernel takes
    the recorded config (one config per kernel in the recording process); first mode always takes the first config;
  * the digest is a function of the table alone, and a saved table replays the same choices;
  * install patches CachingAutotuner.autotune_to_one_config so a kernel's launchers are cut to the pinned one, and
    turns inductor's autotune caches off until uninstall restores them;
  * install patches triton's Autotuner (the hand-written ``@triton.autotune`` kernels): record keeps the timed choice
    per tuning key, replay launches the recorded config without timing anything (another process whose timings
    would pick otherwise), first takes the first config, and a tuning key the table does not hold is refused;
  * install patches inductor's GEMM selection (max-autotune): record keeps the first graph's timed choice per GEMM
    key and hands it to every later graph with that key (whose own timings would pick otherwise), replay keeps the
    table's choice timing nothing, and a GEMM the table does not hold is refused.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from kotodama.model import kernel_pins as kp


class _Cfg:
    def __init__(self, xblock: int, warps: int = 4, stages: int = 1) -> None:
        self.kwargs = {"XBLOCK": xblock}
        self.num_warps = warps
        self.num_stages = stages


def _launchers(*blocks: int) -> list:
    return [types.SimpleNamespace(config=_Cfg(b)) for b in blocks]


def test_config_and_kernel_keys_are_stable() -> None:
    assert kp.config_key(_Cfg(256)) == "XBLOCK=256|w4|s1"
    assert kp.kernel_key("src", (128, 64)) == kp.kernel_key("src", (128, 64)) != kp.kernel_key("src", (64,))


def test_replay_keeps_the_recorded_config(tmp_path: Path) -> None:
    rec = kp.PinSession(mode="record", path=tmp_path / "pins.json")
    launchers = _launchers(128, 256, 512)
    assert rec.choose("k1", launchers) is None  # the autotuner decides
    rec.record("k1", launchers[1])
    # a second instance of k1 in the recording process: no second timing, the recorded config
    assert rec.choose("k1", _launchers(128, 256, 512)).config.kwargs == {"XBLOCK": 256}
    rec.record("k1", launchers[2])
    assert rec.table["k1"] == "XBLOCK=256|w4|s1" and rec.recorded == 1
    rec.save()
    import json

    rep = kp.PinSession(mode="replay", path=rec.path, table=json.loads(rec.path.read_text()))
    assert rep.choose("k1", _launchers(128, 256, 512)).config.kwargs == {"XBLOCK": 256}
    assert rep.hits == 1 and rep.digest == rec.digest
    with pytest.raises(kp.KernelPinMiss, match="not in the pin table"):
        rep.choose("k2", launchers)
    with pytest.raises(kp.KernelPinMiss, match="not among its configs"):
        rep.choose("k1", _launchers(128, 512))
    lenient = kp.PinSession(mode="replay", path=rec.path, table=dict(rep.table), allow_missing=True)
    before = lenient.digest
    assert lenient.choose("k2", _launchers(64, 128)).config.kwargs == {"XBLOCK": 64}
    assert lenient.misses == ["k2"] and lenient.digest != before


def test_first_mode_takes_the_first_config() -> None:
    assert kp.PinSession(mode="first", path=None).choose("k", _launchers(512, 128)).config.kwargs == {"XBLOCK": 512}


def test_install_patches_the_runtime_autotuner(tmp_path: Path) -> None:
    th = pytest.importorskip("torch._inductor.runtime.triton_heuristics")
    table = tmp_path / "pins.json"
    key = kp.kernel_key("def kernel(): pass", (1024,))
    table.write_text('{"%s": "XBLOCK=512|w4|s1"}' % key)
    import torch._inductor.config as ic

    before = {name: getattr(ic, name) for name in kp._CACHE_FLAGS}
    try:
        sess = kp.install(table, "replay")
        assert not any(getattr(ic, name) for name in kp._CACHE_FLAGS)
        stub = types.SimpleNamespace(fn=types.SimpleNamespace(src="def kernel(): pass"), size_hints=(1024,),
                                     launchers=_launchers(128, 512, 1024))
        th.CachingAutotuner.autotune_to_one_config(stub)
        assert [x.config.kwargs["XBLOCK"] for x in stub.launchers] == [512] and sess.hits == 1
        assert kp.session() is sess
        with pytest.raises(FileNotFoundError):
            kp.install(tmp_path / "missing.json", "replay")
    finally:
        kp.uninstall()
    assert kp.session() is None
    assert th.CachingAutotuner.autotune_to_one_config is th.CachingAutotuner._kotodama_unpinned_autotune
    assert {name: getattr(ic, name) for name in kp._CACHE_FLAGS} == before


def _autotuned(timings: dict[int, float]):
    """A triton Autotuner over a stand-in kernel (BLOCK 32/64/128, keyed on n): its benchmark returns ``timings``
    per BLOCK and its launches are logged."""
    autotuner = pytest.importorskip("triton.runtime.autotuner")
    import triton

    launched: list[tuple] = []

    def kernel(x, n, BLOCK):
        pass

    kernel.run = lambda *args, **kw: launched.append((args[1], kw["BLOCK"], kw["num_warps"]))
    configs = [triton.Config({"BLOCK": b}, num_warps=w) for b, w in ((32, 4), (64, 8), (128, 4))]
    tuner = autotuner.Autotuner(kernel, ["x", "n"], configs, key=["n"], reset_to_zero=None, restore_value=None)
    tuner._bench = lambda *args, config, **kw: timings[config.kwargs["BLOCK"]]
    return tuner, launched


def test_install_pins_triton_autotune(tmp_path: Path) -> None:
    autotuner = pytest.importorskip("triton.runtime.autotuner")
    import torch

    x = torch.zeros(4)
    try:
        sess = kp.install(tmp_path / "pins.json", "record")
        tuner, launched = _autotuned({32: 3.0, 64: 1.0, 128: 2.0})
        tuner.run(x, 100)
        tuner.run(x, 100)
        tuner.run(x, 7)
        assert [b for _, b, _ in launched] == [64, 64, 64] and sess.recorded == 2
        # another instance of the same kernel in the recording process, whose timings differ: the recorded config
        other, launched2 = _autotuned({32: 0.1, 64: 5.0, 128: 5.0})
        other.run(x, 100)
        assert launched2 == [(100, 64, 8)] and sess.recorded == 2
        assert all(k.startswith("triton:kernel:") for k in sess.table)
        sess.save()
        # another process, whose timings would pick BLOCK 32: replay launches the recorded 64, timing nothing
        rep = kp.install(tmp_path / "pins.json", "replay")
        tuner, launched = _autotuned({32: 0.5, 64: 9.0, 128: 9.0})
        tuner._bench = lambda *a, **k: pytest.fail("a pinned kernel was timed")
        tuner.run(x, 100)
        tuner.run(x, 7)
        assert launched == [(100, 64, 8), (7, 64, 8)] and rep.hits == 2 and rep.digest == sess.digest
        with pytest.raises(kp.KernelPinMiss, match="not in the pin table"):
            tuner.run(x, 5)
        kp.install(None, "first")
        tuner, launched = _autotuned({32: 9.0, 64: 0.1, 128: 9.0})
        tuner.run(x, 100)
        assert launched == [(100, 32, 4)]
    finally:
        kp.uninstall()
    assert autotuner.Autotuner.run is autotuner.Autotuner._kotodama_unpinned_run


class _Choice:
    def __init__(self, name: str) -> None:
        self.name = name

    def hash_key(self) -> str:
        return f"choice-{self.name}"


def test_install_pins_gemm_selection(tmp_path: Path, monkeypatch) -> None:
    sa = pytest.importorskip("torch._inductor.select_algorithm")
    cls = sa.AlgorithmSelectorCache
    real = getattr(cls, "_kotodama_unpinned_do_autotuning", cls.do_autotuning)
    timed: list[str] = []
    speeds: dict[str, float] = {}

    def fake(self, name, input_nodes, layout, input_gen_fns, inputs_key, choices, *args, **kwargs):
        timed.append(inputs_key)
        return {c: speeds[c.name] for c in choices}

    monkeypatch.setattr(cls, "_kotodama_unpinned_do_autotuning", fake, raising=False)
    monkeypatch.setattr(cls, "do_autotuning", real)

    def select(key: str = "[(128, 1536)]", names=("aten", "tmpl_a", "tmpl_b")):
        choices = [_Choice(n) for n in names]
        t = cls.do_autotuning(None, "mm", [], None, None, key, choices, None)
        return min(t, key=t.__getitem__).name

    try:
        sess = kp.install(tmp_path / "pins.json", "record")
        speeds.update(aten=2.0, tmpl_a=1.0, tmpl_b=3.0)
        assert select() == "tmpl_a" and len(timed) == 1
        # a later graph with the same GEMM, whose own timings would pick the extern kernel: the recorded choice
        speeds.update(aten=0.1, tmpl_a=5.0)
        assert select() == "tmpl_a" and len(timed) == 1
        assert select(key="[(64, 1536)]") == "aten" and len(timed) == 2  # another GEMM is timed
        assert all(k.startswith("gemm:mm:") for k in sess.table) and len(sess.table) == 2
        sess.save()
        rep = kp.install(tmp_path / "pins.json", "replay")
        speeds.update(aten=0.0, tmpl_a=9.0, tmpl_b=0.0)
        assert select() == "tmpl_a" and select(key="[(64, 1536)]") == "aten" and len(timed) == 2
        assert rep.hits == 2 and rep.digest == sess.digest
        with pytest.raises(kp.KernelPinMiss, match="not in the pin table"):
            select(names=("aten", "tmpl_c"))  # another candidate set is another GEMM
    finally:
        kp.uninstall()
    assert cls.do_autotuning is fake  # uninstall restores what install found
