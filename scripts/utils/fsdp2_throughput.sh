#!/bin/bash
# G8: FSDP2 vs DDP throughput at the kotodama-tsumugi production recipe (compiled, 2M-token batch, 4K context) on
# one node. Runs, in order, each STEPS steps with the optimizer step timed (--time_optimizer_step) and telemetry
# readers off in every arm. Runs (micro-batch 8 unless named): ddp; ddp-regional (DDP with per-sublayer compile:
# the compile scope FSDP2 runs with, so the fair baseline for its comm cost); fsdp2; fsdp2-layer (one shard group per
# layer); fsdp2-layer-zero2 (that, without reshard-after-forward); fsdp2-layer-mb16 (micro-batch 16);
# fsdp2-layer-mxfp8 (MXFP8 matmuls; runs in the venv at $KOTODAMA_MXFP8_VENV, default <data root>/venv-mxfp8,
# whose torchao carries the CUDA mxfp8 kernels; skipped if absent). Also
# available: fsdp2-zero2, fsdp2-defer (reduce-scatter deferred to a step's last micro-step), fsdp2-mb16.
# Under a multi-node launch (the scheduler sets NNODES > 1) every fsdp2 run is HSDP, sharded within a node and
# replicated across nodes (--fsdp2_replicas NNODES), except the -flat runs (fsdp2-layer-flat, fsdp2-layer-flat-mxfp8),
# which shard one layer per group over every rank of every node. Interconnect settings (e.g. NCCL_IB_HCA) come from the environment.
# Each run's log is OUT/<name>/train.log and its GPU power / SM clock / throttle reasons, sampled every 5 s, are
# OUT/<name>/power.csv, for the job's own GPUs (node k: train.node<k>.log, power.node<k>.csv). The summary (over the
# second half of the steps: mean tok/s, MFU, optimizer share, peak memory, mean power and clock, and the share of
# samples throttled by the software power cap, bit 0x4) is printed at the end and written to OUT/summary.md.
#
# Usage: scripts/utils/fsdp2_throughput.sh WORLD OUT DATA_PATH [STEPS] [RUNS] [extra train flags...]
#   RUNS: comma list of run names (default: ddp,ddp-regional,fsdp2,fsdp2-layer,fsdp2-layer-zero2,fsdp2-layer-mb16,
#         fsdp2-layer-mxfp8; also fsdp2-zero2, fsdp2-defer, fsdp2-mb16, fsdp2-layer-flat, fsdp2-group2,
#         fsdp2-group4: two or four consecutive layers per shard group; fsdp2-layer-mxfp8-levers: fsdp2-layer-mxfp8
#         with the AttnRes stack-grad chain and the tuned RMSNorm launches)
#   Extra train flags come last on every run's command line, so they override the recipe and the run's own flags
#   (another shape: --model_size NAME --attn_res_n_blocks N --micro_batch_size 1 --global_batch_tokens T).
set -uo pipefail
WORLD="$1"; OUT="$2"; DATA="$3"; STEPS="${4:-40}"; RUNS="${5:-ddp,ddp-regional,fsdp2,fsdp2-layer,fsdp2-layer-zero2,fsdp2-layer-mb16,fsdp2-layer-mxfp8}"
shift $(( $# < 5 ? $# : 5 )); USER_EXTRA=("$@")
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# Several nodes: node 0 writes train.log, node k train.node<k>.log; only node 0 post-processes.
LOGSUFFIX=""
if [ "${NODE_RANK:-0}" != 0 ]; then LOGSUFFIX=".node$NODE_RANK"; fi
[ -f "$ROOT/site.env" ] && source "$ROOT/site.env"
MXVENV="${KOTODAMA_MXFP8_VENV:-${KOTODAMA_DATA_ROOT:-}/venv-mxfp8}"
mkdir -p "$OUT"
export KOTODAMA_NPROC="$WORLD"
NODES="${NNODES:-1}"; BASE_PORT="${MASTER_PORT:-29500}"; PORT_OFF=0
REPL=()
if [ "$NODES" -gt 1 ]; then REPL=(--fsdp2_replicas "$NODES"); fi
COMMON=(--config "$ROOT/configs/tsumugi-1b.yaml" --no_geo_monitor --inloop_evals '' --mix_fraction 0
        --no_attnres_eps_fp64_compat --data_path "$DATA" --global_batch_tokens 2097152 --total_steps "$STEPS"
        --log_every 5 --norm_gain_every 0 --save_every 0 --save_schedule '' --anchor_tokens '' --time_optimizer_step
        --varlen_dynamic)
IFS=, read -ra LIST <<< "$RUNS"
for name in "${LIST[@]}"; do
  case "$name" in
    ddp) EXTRA=(--micro_batch_size 8) ;;
    ddp-regional) EXTRA=(--micro_batch_size 8 --compile_regional) ;;
    fsdp2) EXTRA=(--fsdp2 --micro_batch_size 8) ;;
    fsdp2-zero2) EXTRA=(--fsdp2 --no_fsdp2_reshard_after_forward --micro_batch_size 8) ;;
    fsdp2-defer) EXTRA=(--fsdp2 --fsdp2_defer_grad_sync --micro_batch_size 8) ;;
    fsdp2-layer) EXTRA=(--fsdp2 --fsdp2_layers_per_group 1 --micro_batch_size 8) ;;
    fsdp2-layer-zero2) EXTRA=(--fsdp2 --fsdp2_layers_per_group 1 --no_fsdp2_reshard_after_forward --micro_batch_size 8) ;;
    fsdp2-layer-mb16) EXTRA=(--fsdp2 --fsdp2_layers_per_group 1 --micro_batch_size 16) ;;
    fsdp2-layer-mxfp8) EXTRA=(--fsdp2 --fsdp2_layers_per_group 1 --micro_batch_size 8 --fp8 --fp8_recipe mxfp8) ;;
    fsdp2-layer-mxfp8-levers) EXTRA=(--fsdp2 --fsdp2_layers_per_group 1 --micro_batch_size 8 --fp8 --fp8_recipe mxfp8
                                     --attnres_stack_grad_chain --liger_rmsnorm_tuned) ;;
    fsdp2-mb16) EXTRA=(--fsdp2 --micro_batch_size 16) ;;
    fsdp2-layer-flat) EXTRA=(--fsdp2 --fsdp2_layers_per_group 1 --micro_batch_size 8) ;;
    fsdp2-layer-flat-mxfp8) EXTRA=(--fsdp2 --fsdp2_layers_per_group 1 --micro_batch_size 8 --fp8 --fp8_recipe mxfp8) ;;
    fsdp2-group2) EXTRA=(--fsdp2 --fsdp2_layers_per_group 2 --micro_batch_size 8) ;;
    fsdp2-group4) EXTRA=(--fsdp2 --fsdp2_layers_per_group 4 --micro_batch_size 8) ;;
    *) echo "unknown run $name" >&2; exit 2 ;;
  esac
  if [[ "$name" == fsdp2* && "$name" != *-flat* ]]; then EXTRA+=("${REPL[@]}"); fi
  # A fresh rendezvous port per run, so a run never meets the previous one's socket still closing.
  PORT_OFF=$((PORT_OFF + 1)); export MASTER_PORT=$((BASE_PORT + PORT_OFF))
  RUN_VENV="${KOTODAMA_VENV:-}"
  if [[ "$name" == *mxfp8* ]]; then
    if [ ! -x "$MXVENV/bin/python" ]; then
      echo "=== $name SKIPPED: no mxfp8 venv at $MXVENV (set KOTODAMA_MXFP8_VENV) ==="
      continue
    fi
    RUN_VENV="$MXVENV"
  fi
  mkdir -p "$OUT/$name"
  echo "=== $name $(date -u +%H:%M:%S) ==="
  nvidia-smi ${CUDA_VISIBLE_DEVICES:+-i "$CUDA_VISIBLE_DEVICES"} \
    --query-gpu=timestamp,index,power.draw,clocks.sm,clocks_throttle_reasons.active --format=csv,noheader,nounits -l 5 > "$OUT/$name/power${LOGSUFFIX}.csv" 2>/dev/null &
  SAMPLER=$!
  KOTODAMA_SCHEDULER_METRIC_PREFIX="$name" KOTODAMA_VENV="$RUN_VENV" "$ROOT/tools/run_train.sh" "${COMMON[@]}" "${EXTRA[@]}" "${USER_EXTRA[@]}" --checkpoint_dir "$OUT/$name" \
    --checkpoint_shm_dir "/dev/shm/kotodama-fsdp2-g8-$name" 2>&1 | tee "$OUT/$name/train${LOGSUFFIX}.log"
  echo "=== $name exit ${PIPESTATUS[0]} $(date -u +%H:%M:%S) ==="
  kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null
done
[ "${NODE_RANK:-0}" = 0 ] || exit 0
source "$ROOT/tools/_env.sh"
python - "$OUT" "$STEPS" "${LIST[@]}" <<'EOF'
import re, statistics, sys
from pathlib import Path
out, steps, names = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
step_re = re.compile(r"step=(\d+)/\d+ .*?tok/s=(\d+) .*?mfu=([0-9.]+)% .*?gpu_mem=([0-9.]+)GB")
opt_re = re.compile(r"opt_step=([0-9.]+)s \(([0-9.]+)% of the mean step ([0-9.]+)s\)")
def power(path):
    """(mean W per GPU, mean SM MHz, share of samples with the SW power cap bit 0x4) over the file's second half."""
    rows = []
    for line in (path.read_text(errors="replace").splitlines() if path.exists() else []):
        f = [x.strip() for x in line.split(",")]
        try:
            rows.append((float(f[2]), float(f[3]), int(f[4], 16)))
        except (IndexError, ValueError):
            continue
    rows = rows[len(rows) // 2:]
    if not rows:
        return "", "", ""
    return (f"{statistics.mean(r[0] for r in rows):.0f}", f"{statistics.mean(r[1] for r in rows):.0f}",
            f"{100 * sum(bool(r[2] & 0x4) for r in rows) / len(rows):.0f}%")
lines = ["| run | tok/s | MFU % | step s | opt step s | opt % of step | peak GB | W/GPU | SM MHz | power-capped | status |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for n in names:
    log = out / n / "train.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    rows = [(int(m[0]), float(m[1]), float(m[2]), float(m[3])) for m in step_re.findall(text)]
    opts = [(float(a), float(b), float(c)) for a, b, c in opt_re.findall(text)]
    late = [r for r in rows if r[0] >= steps // 2]
    late_opt = opts[-max(1, len(opts) // 2):] if opts else []
    pw, mhz, capped = power(out / n / "power.csv")
    if not late:
        lines.append(f"| {n} | | | | | | | {pw} | {mhz} | {capped} | no steps (see {log}) |")
        continue
    lines.append(f"| {n} | {statistics.mean(r[1] for r in late):.0f} | {statistics.mean(r[2] for r in late):.1f} | "
                 f"{statistics.mean(o[2] for o in late_opt):.3f} | {statistics.mean(o[0] for o in late_opt):.4f} | "
                 f"{statistics.mean(o[1] for o in late_opt):.2f} | {max(r[3] for r in rows):.1f} | "
                 f"{pw} | {mhz} | {capped} | {'complete' if 'Training complete' in text else 'incomplete'} |")
(out / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
EOF
