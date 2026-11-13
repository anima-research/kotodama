"""Forward+backward cost of a model shape on one GPU, for pricing candidate shapes (width, depth, AttnRes blocks, loop).

Builds KotodamaModel directly with the training stack's settings (FA4 varlen inputs, Liger kernels + fused linear CE,
QK-norm, fp32 RoPE, whole-model torch.compile, bf16 weights), feeds random tokens, and times fwd+bwd of one
micro-batch (no optimizer step; that cost is O(params) and modelled separately). One JSON line per run on stdout.

--profile adds where the step's wall time goes: GPU-busy time (the union of kernel intervals in one traced step) vs
idle (launch gaps, host syncs), the kernel count, busy time by kernel family (GEMM, attention, other), and the GPU's
power draw and SM clock sampled over the timed steps (a power-capped GPU lowers its clock).

Pairs of runs price AttnRes routing: the same shape with --n_blocks N and with --n_blocks 0 (no AttnRes) differ by
exactly the routing work, whose per-layer share depends on N and the width, not on the depth.

  tools/run_py.sh scripts/benchmark/shape_cost.py --hidden 3072 --heads 24 --kv 8 --ffn 8192 --layers 48 --n_blocks 8
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kotodama.model import llama  # noqa: E402
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig  # noqa: E402

DEVICE = "cuda"
GEMM_MARKERS = ("gemm", "cutlass", "nvjet", "xmma", "matmul", "_mm_", "sm90_", "sm100_")
ATTN_MARKERS = ("flash", "fmha", "attention")


def _family(name: str) -> str:
    low = name.lower()
    if any(m in low for m in ATTN_MARKERS):
        return "attention"
    if any(m in low for m in GEMM_MARKERS):
        return "gemm"
    return "other"


def _union_ms(intervals: list[tuple[float, float]]) -> float:
    """Total length (ms) of the union of [start, end) intervals given in microseconds."""
    total, cur_s, cur_e = 0.0, None, None
    for s_, e_ in sorted(intervals):
        if cur_e is None or s_ > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s_, e_
        else:
            cur_e = max(cur_e, e_)
    if cur_e is not None:
        total += cur_e - cur_s
    return total / 1e3


def trace_step(step: Callable[[], None]) -> dict[str, float]:
    """One step under the torch profiler: wall (first kernel start to last kernel end), GPU-busy union, kernel count
    and busy ms per kernel family."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trace.json")
        prof.export_chrome_trace(path)
        events = json.load(open(path)).get("traceEvents", [])
    kernels = [e for e in events if e.get("cat") == "kernel" and "dur" in e]
    if not kernels:
        return {"trace_error": 1.0}
    spans = [(float(e["ts"]), float(e["ts"]) + float(e["dur"])) for e in kernels]
    wall = (max(e for _s, e in spans) - min(s_ for s_, _e in spans)) / 1e3
    busy = _union_ms(spans)
    out = {"trace_wall_ms": wall, "gpu_busy_ms": busy, "gpu_idle_frac": 1.0 - busy / wall, "kernels": float(len(kernels))}
    for fam in ("gemm", "attention", "other"):
        out[f"busy_{fam}_ms"] = _union_ms([sp for sp, e in zip(spans, kernels) if _family(e.get("name", "")) == fam])
    return out


class PowerSampler:
    """Samples power (W) and SM clock (MHz) of the current GPU every ``period`` s on a thread, via torch.cuda (NVML
    bindings) or else nvidia-smi. Reports nothing where neither works."""

    def __init__(self, period: float = 0.05) -> None:
        self.period, self.watts, self.mhz = period, [], []
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None

    @staticmethod
    def _read_nvml() -> tuple[float, float]:
        return torch.cuda.power_draw() / 1e3, float(torch.cuda.clock_rate())

    @staticmethod
    def _read_smi() -> tuple[float, float]:
        """nvidia-smi on the physical GPU behind this process's device (CUDA_VISIBLE_DEVICES maps the index)."""
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        idx = visible.split(",")[torch.cuda.current_device()] if visible else str(torch.cuda.current_device())
        out = subprocess.run(["nvidia-smi", "-i", idx, "--query-gpu=power.draw,clocks.sm",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
        w, mhz = out.stdout.strip().split(",")
        return float(w), float(mhz)

    def _run(self) -> None:
        readers = [self._read_nvml, self._read_smi]
        while not self._stop.is_set():
            while readers:
                try:
                    w, mhz = readers[0]()
                    break
                except Exception:  # noqa: BLE001 — try the next reader; none left: report nothing, keep timing
                    readers.pop(0)
            if not readers:
                return
            self.watts.append(w)
            self.mhz.append(mhz)
            time.sleep(self.period)

    def __enter__(self) -> "PowerSampler":
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._t is not None:
            self._t.join()

    def summary(self) -> dict[str, float]:
        if not self.watts:
            return {}
        return {"power_w_median": statistics.median(self.watts), "power_w_max": max(self.watts),
                "sm_mhz_median": statistics.median(self.mhz), "sm_mhz_min": min(self.mhz)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--kv", type=int, required=True)
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--ffn", type=int, required=True)
    ap.add_argument("--layers", type=int, required=True,
                    help="effective depth (num_layers; a loop reuses weights inside it, see KotodamaModelConfig)")
    ap.add_argument("--n_blocks", type=int, default=0, help="AttnRes blocks over the effective depth (0 = no AttnRes)")
    ap.add_argument("--loop_start", type=int, default=None)
    ap.add_argument("--loop_len", type=int, default=0)
    ap.add_argument("--loop_times", type=int, default=1)
    ap.add_argument("--mb", type=int, default=1)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--chain", action="store_true", help="--attnres_stack_grad_chain")
    ap.add_argument("--eager", action="store_true", help="no torch.compile (debugging)")
    ap.add_argument("--flce", action="store_true",
                    help="loss via the model's fused linear CE inside the compiled graph (as in training). Default: "
                         "the compiled model returns logits and the CE runs eagerly — under a single unsplit compiled "
                         "graph (no DDP) inductor mis-reconstructs FLCE's chunked in-place writes (an as_strided read "
                         "past a 128-element slice -> illegal address at B*T 4096).")
    ap.add_argument("--profile", action="store_true",
                    help="also trace one step (GPU busy vs idle, kernel families) and sample power + SM clock")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    if a.chain:
        llama._STACK_GRAD_CHAIN = True
    cfg = KotodamaModelConfig(
        hidden_size=a.hidden, num_layers=a.layers, num_attention_heads=a.heads, num_kv_heads=a.kv,
        head_dim=a.head_dim, intermediate_size=a.ffn, max_position_embeddings=a.seq, qk_norm=True,
        use_liger=True, rope_fp32=True, attn_impl="fa4",
        attn_res=a.n_blocks > 0, attn_res_n_blocks=max(a.n_blocks, 1),
        loop_start=a.loop_start, loop_len=a.loop_len, loop_times=a.loop_times,
    )
    torch.manual_seed(0)
    model = KotodamaModel(cfg).to(device=DEVICE, dtype=torch.bfloat16)
    model.train()
    compiled = model if a.eager else torch.compile(model)
    B, T = a.mb, a.seq
    g = torch.Generator(device=DEVICE).manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (B, T), device=DEVICE, generator=g)
    position_ids = torch.arange(T, device=DEVICE).expand(B, T)
    cu_seqlens = (torch.arange(B + 1, device=DEVICE) * T).to(torch.int32)

    def step() -> None:
        if a.flce:
            loss = compiled(ids, labels=ids, position_ids=position_ids, cu_seqlens=cu_seqlens, max_seqlen=T)["loss"]
        else:
            logits = compiled(ids, position_ids=position_ids, cu_seqlens=cu_seqlens, max_seqlen=T)["logits"]
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                                                     ids[:, 1:].reshape(-1))
        loss.backward()

    t0 = time.perf_counter()
    step()
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - t0
    for _ in range(2):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    sampler = PowerSampler()
    with (sampler if a.profile else contextlib.nullcontext()):
        for _ in range(a.iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            step()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
    times.sort()
    ms = times[len(times) // 2]
    extra: dict[str, float] = {}
    if a.profile:
        extra.update(sampler.summary())
        extra.update(trace_step(step))
    print(json.dumps({
        "tag": a.tag, "hidden": a.hidden, "heads": a.heads, "kv": a.kv, "ffn": a.ffn, "eff_layers": cfg.num_layers,
        "unique_layers": cfg.num_unique_layers, "n_blocks": a.n_blocks,
        "loop": [a.loop_start, a.loop_len, a.loop_times],
        "chain": a.chain, "flce": a.flce, "mb": B, "seq": T, "params": sum(p.numel() for p in model.parameters()),
        "fwd_bwd_ms": ms, "us_per_token": ms * 1e3 / (B * T), "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "compile_s": compile_s, **extra,
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
