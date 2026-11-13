#!/bin/bash
# G8: FSDP2 vs DDP throughput at the kotodama-tsumugi production recipe (compiled, 2M-token batch, 4K context) on
# one node. Runs, in order, each STEPS steps with the optimizer step timed (--time_optimizer_step) and telemetry
# readers off in every arm: ddp (micro-batch 8), fsdp2 (8), fsdp2 without reshard-after-forward (8), fsdp2 (16).
# Each run's log is OUT/<name>/train.log; the summary (mean tok/s, MFU, optimizer share and peak memory over the
# second half of the steps) is printed at the end and written to OUT/summary.md.
#
# Usage: scripts/utils/fsdp2_throughput.sh WORLD OUT DATA_PATH [STEPS] [RUNS]
#   RUNS: comma list from ddp,fsdp2,fsdp2-zero2,fsdp2-mb16 (default: all four)
set -uo pipefail
WORLD="$1"; OUT="$2"; DATA="$3"; STEPS="${4:-40}"; RUNS="${5:-ddp,fsdp2,fsdp2-zero2,fsdp2-mb16}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$OUT"
export KOTODAMA_NPROC="$WORLD"
COMMON=(--config "$ROOT/configs/tsumugi-1b.yaml" --no_geo_monitor --inloop_evals '' --mix_fraction 0
        --no_attnres_eps_fp64_compat --data_path "$DATA" --global_batch_tokens 2097152 --total_steps "$STEPS"
        --log_every 5 --norm_gain_every 0 --save_every 0 --save_schedule '' --anchor_tokens '' --time_optimizer_step)
IFS=, read -ra LIST <<< "$RUNS"
for name in "${LIST[@]}"; do
  case "$name" in
    ddp) EXTRA=(--micro_batch_size 8) ;;
    fsdp2) EXTRA=(--fsdp2 --micro_batch_size 8) ;;
    fsdp2-zero2) EXTRA=(--fsdp2 --no_fsdp2_reshard_after_forward --micro_batch_size 8) ;;
    fsdp2-mb16) EXTRA=(--fsdp2 --micro_batch_size 16) ;;
    *) echo "unknown run $name" >&2; exit 2 ;;
  esac
  mkdir -p "$OUT/$name"
  echo "=== $name $(date -u +%H:%M:%S) ==="
  "$ROOT/tools/run_train.sh" "${COMMON[@]}" "${EXTRA[@]}" --checkpoint_dir "$OUT/$name" \
    --checkpoint_shm_dir "/dev/shm/kotodama-fsdp2-g8-$name" 2>&1 | tee "$OUT/$name/train.log"
  echo "=== $name exit ${PIPESTATUS[0]} $(date -u +%H:%M:%S) ==="
done
source "$ROOT/tools/_env.sh"
python - "$OUT" "$STEPS" "${LIST[@]}" <<'EOF'
import re, statistics, sys
from pathlib import Path
out, steps, names = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
step_re = re.compile(r"step=(\d+)/\d+ .*?tok/s=(\d+) .*?mfu=([0-9.]+)% .*?gpu_mem=([0-9.]+)GB")
opt_re = re.compile(r"opt_step=([0-9.]+)s \(([0-9.]+)% of the mean step ([0-9.]+)s\)")
lines = ["| run | tok/s | MFU % | step s | opt step s | opt % of step | peak GB | status |", "|---|---|---|---|---|---|---|---|"]
for n in names:
    log = out / n / "train.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    rows = [(int(m[1]), float(m[2]), float(m[3]), float(m[4])) for m in step_re.findall(text)]
    opts = [(float(a), float(b), float(c)) for a, b, c in opt_re.findall(text)]
    late = [r for r in rows if r[0] >= steps // 2]
    late_opt = opts[-max(1, len(opts) // 2):] if opts else []
    if not late:
        lines.append(f"| {n} | | | | | | | no steps (see {log}) |")
        continue
    lines.append(f"| {n} | {statistics.mean(r[1] for r in late):.0f} | {statistics.mean(r[2] for r in late):.1f} | "
                 f"{statistics.mean(o[2] for o in late_opt):.3f} | {statistics.mean(o[0] for o in late_opt):.4f} | "
                 f"{statistics.mean(o[1] for o in late_opt):.2f} | {max(r[3] for r in rows):.1f} | "
                 f"{'complete' if 'Training complete' in text else 'incomplete'} |")
(out / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
EOF
