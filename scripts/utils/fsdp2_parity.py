"""FSDP2 parity gate: compare the three arms of scripts/utils/fsdp2_gate.sh at one world size.

The reference is DDP; its own noise is the DDP arm at micro-batch 1 (the same data per rank, with gradients
accumulated in a different order). The FSDP2 arm passes when it differs from the reference by no more than the
reference differs from itself. The criteria are fixed here, before any run is read:

  G0  data: every logged (step, micro) batch-digest line of the FSDP2 arm equals the reference's, and no line repeats
      a digest across ranks.
  G1  loss: over all steps, mean |L_fsdp2 - L_ref| <= 2 * mean |L_noise - L_ref| + 2e-4; over the last quarter,
      max |L_fsdp2 - L_ref| <= 3 * max |L_noise - L_ref| + 2e-4 (losses are logged to 4 decimals).
  G1  weights: at every save, for every tensor, d_test = rms(W_fsdp2 - W_ref) and d_noise = rms(W_noise - W_ref);
      ratio = d_test / max(d_noise, rms(W_ref) * 2^-9). Pass: median ratio <= 2 and max ratio <= 5.
  G2  the weight table covers every tensor; tensors whose rows do not divide by the world size are flagged, never
      left out.
  G7  grad norm: the same loss rule applied to the logged grad_norm (3 decimals: floor 2e-3).

Usage: tools/run_py.sh scripts/utils/fsdp2_parity.py --ref DIR --noise DIR --test DIR --world N --out FILE.md
Each DIR holds train.log and weights/step_*.pt. Exit status 0 = every gate passes.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import torch

STEP_RE = re.compile(r"step=(\d+)/\d+ \| loss=([0-9.]+) .*?grad_norm=([0-9.]+)")
HASH_RE = re.compile(r"batch-hash step=(\d+) micro=(\d+) shape=\([^)]*\) digests=(\S+)")


def read_log(path: Path) -> tuple[dict[int, tuple[float, float]], dict[tuple[int, int], list[str]]]:
    steps: dict[int, tuple[float, float]] = {}
    hashes: dict[tuple[int, int], list[str]] = {}
    for line in path.read_text(errors="replace").splitlines():
        m = STEP_RE.search(line)
        if m:
            steps[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
            continue
        m = HASH_RE.search(line)
        if m:
            hashes[(int(m.group(1)), int(m.group(2)))] = m.group(3).split(",")
    return steps, hashes


def curve_gate(name: str, ref: dict[int, float], noise: dict[int, float], test: dict[int, float],
               floor: float) -> tuple[bool, str]:
    common = sorted(set(ref) & set(noise) & set(test))
    if not common:
        return False, f"{name}: no common steps"
    d_t = [abs(test[s] - ref[s]) for s in common]
    d_n = [abs(noise[s] - ref[s]) for s in common]
    tail = common[-max(1, len(common) // 4):]
    dt_tail = max(abs(test[s] - ref[s]) for s in tail)
    dn_tail = max(abs(noise[s] - ref[s]) for s in tail)
    ok = (statistics.mean(d_t) <= 2 * statistics.mean(d_n) + floor) and (dt_tail <= 3 * dn_tail + floor)
    return ok, (f"{name}: {len(common)} steps; mean |test-ref| {statistics.mean(d_t):.5f} vs mean |noise-ref| "
                f"{statistics.mean(d_n):.5f}; last-quarter max {dt_tail:.5f} vs {dn_tail:.5f} -> "
                f"{'PASS' if ok else 'FAIL'}")


def weight_gate(ref_dir: Path, noise_dir: Path, test_dir: Path, world: int) -> tuple[bool, list[str], list[dict]]:
    names = sorted(p.name for p in (ref_dir / "weights").glob("step_*.pt"))
    lines: list[str] = []
    rows: list[dict] = []
    ok_all = bool(names)
    for fname in names:
        if not (noise_dir / "weights" / fname).exists() or not (test_dir / "weights" / fname).exists():
            lines.append(f"{fname}: missing in an arm -> FAIL")
            ok_all = False
            continue
        load = lambda d: torch.load(d / "weights" / fname, map_location="cpu", weights_only=False)["model"]  # noqa: E731
        ref, noise, test = load(ref_dir), load(noise_dir), load(test_dir)
        if set(ref) != set(test) or set(ref) != set(noise):
            lines.append(f"{fname}: key sets differ -> FAIL")
            ok_all = False
            continue
        ratios: list[float] = []
        seen: set[int] = set()
        for k in sorted(ref):
            w = ref[k]
            if not w.is_floating_point() or w.data_ptr() in seen:
                continue  # loop-repeat aliases are one tensor
            seen.add(w.data_ptr())
            w, wn, wt = w.double(), noise[k].double(), test[k].double()
            rms = lambda x: x.pow(2).mean().sqrt().item()  # noqa: E731
            d_t, d_n = rms(wt - w), rms(wn - w)
            floor = rms(w) * 2.0 ** -9
            ratio = d_t / max(d_n, floor, 1e-30)
            ratios.append(ratio)
            rows.append({"save": fname, "tensor": k, "shape": list(w.shape), "d_test": d_t, "d_noise": d_n,
                         "ratio": ratio, "uneven": bool(w.ndim and w.shape[0] % world)})
        med, worst = statistics.median(ratios), max(ratios)
        ok = med <= 2.0 and worst <= 5.0
        ok_all &= ok
        lines.append(f"{fname}: {len(ratios)} tensors ({sum(r['uneven'] for r in rows if r['save'] == fname)} "
                     f"uneven over {world}); ratio median {med:.3f}, max {worst:.3f} -> {'PASS' if ok else 'FAIL'}")
    return ok_all, lines, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--noise", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    (r_steps, r_hash), (n_steps, _), (t_steps, t_hash) = (read_log(d / "train.log") for d in (a.ref, a.noise, a.test))
    report: list[str] = [f"# FSDP2 parity gate, world {a.world}", "",
                         f"- ref `{a.ref}`", f"- noise `{a.noise}`", f"- test `{a.test}`", ""]

    g0 = bool(r_hash) and r_hash == t_hash and all(len(set(v)) == len(v) == a.world for v in t_hash.values())
    report.append(f"- G0 data: {len(t_hash)} (step, micro) lines; FSDP2 == DDP digests: {r_hash == t_hash}; "
                  f"distinct across ranks: {all(len(set(v)) == len(v) for v in t_hash.values())} -> "
                  f"{'PASS' if g0 else 'FAIL'}")
    g1l, msg = curve_gate("loss", {s: v[0] for s, v in r_steps.items()}, {s: v[0] for s, v in n_steps.items()},
                          {s: v[0] for s, v in t_steps.items()}, 2e-4)
    report.append(f"- G1 {msg}")
    g7, msg = curve_gate("grad_norm", {s: v[1] for s, v in r_steps.items()}, {s: v[1] for s, v in n_steps.items()},
                         {s: v[1] for s, v in t_steps.items()}, 2e-3)
    report.append(f"- G7 {msg}")
    g1w, lines, rows = weight_gate(a.ref, a.noise, a.test, a.world)
    report += [f"- G1/G2 weights {line}" for line in lines]
    worst = sorted(rows, key=lambda r: -r["ratio"])[:15]
    report += ["", "Worst tensors (ratio = d_test / d_noise):", "", "| save | tensor | shape | uneven | d_test | "
               "d_noise | ratio |", "|---|---|---|---|---|---|---|"]
    report += [f"| {r['save']} | {r['tensor']} | {r['shape']} | {r['uneven']} | {r['d_test']:.3e} | "
               f"{r['d_noise']:.3e} | {r['ratio']:.3f} |" for r in worst]
    verdict = g0 and g1l and g7 and g1w
    report += ["", f"**VERDICT: {'PASS' if verdict else 'FAIL'}**", ""]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(report))
    a.out.with_suffix(".json").write_text(json.dumps({"verdict": verdict, "rows": rows}, indent=1))
    print("\n".join(report))
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
