#!/bin/bash
# A dress rehearsal for a long --fsdp2 run: shape C (presets 10b-bench-spire) on every GPU of the job's nodes at the
# launch flags measured best (layer compile scope, activation checkpointing, micro-batch 8, 10-layer groups, ZeRO-2),
# with the geometric monitor and the in-loop evals on, periodic resumable saves, and optionally one requested stop and
# resume inside the same job: node 0 creates OUT/STOP after STOP_AFTER_MIN minutes, every rank saves at the next step
# boundary and exits (--stop_file), and each node then relaunches the run, which resumes from that save.
#
# Usage: scripts/utils/fsdp2_rehearsal.sh ARM OUT DATA_PATH STEPS STOP_AFTER_MIN [extra train flags...]
#   ARM             mxfp8 | bf16
#   OUT             the checkpoint dir, on a filesystem every node sees (one sharded copy; node 0 logs to OUT/train.log,
#                   node k to OUT/train.node<k>.log)
#   STEPS           total optimizer steps (2,097,152 tokens each)
#   STOP_AFTER_MIN  0: no stop; N: one requested stop after N minutes, then the resume
# SAVE_EVERY (env, default 100 steps; 0 disables every save) sets the periodic save cadence; BOUNDARIES (env) the
# AttnRes block starts.
set -euo pipefail
ARM="$1"; OUT="$2"; DATA="$3"; STEPS="$4"; STOP_AFTER="$5"; shift 5
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
NNODES="${NNODES:-1}"; NODE_RANK="${NODE_RANK:-0}"
LOG="$OUT/train.log"
if [ "$NODE_RANK" != 0 ]; then LOG="$OUT/train.node$NODE_RANK.log"; fi
case "$ARM" in
  mxfp8) PREC=(--fp8 --fp8_recipe mxfp8) ;;
  bf16) PREC=() ;;
  *) echo "unknown arm $ARM (mxfp8 | bf16)" >&2; exit 2 ;;
esac
REPL=()
if [ "$NNODES" -gt 1 ]; then REPL=(--fsdp2_replicas "$NNODES"); fi
mkdir -p "$OUT"
# Eight AttnRes blocks cut at the loop seams (30 | 70 | 110 of the 140 effective layers), as syzygy's boundaries are.
BOUNDARIES="${BOUNDARIES:-0,1,10,20,30,70,110,125}"
# Every rank's batch digests at every step: two arms with the same data seed compare step for step, across a resume.
export KOTODAMA_DEBUG_BATCH_HASH="${KOTODAMA_DEBUG_BATCH_HASH:-$STEPS}"
FLAGS=(--config "$ROOT/configs/tsumugi-1b.yaml" --model_size 10b-bench-spire --attn_res_n_blocks 8 --attn_res_boundaries "$BOUNDARIES"
       --no_attnres_eps_fp64_compat --mix_fraction 0 --data_path "$DATA" --global_batch_tokens 2097152
       --total_steps "$STEPS" --log_every 1
       --fsdp2 "${REPL[@]}" --fsdp2_compile_scope layer --activation_checkpointing --micro_batch_size 8
       --fsdp2_layers_per_group 10 --no_fsdp2_reshard_after_forward --varlen_dynamic
       --geo_monitor --inloop_evals riders,induction
       --checkpoint_dir "$OUT" --save_every "${SAVE_EVERY:-100}" --keep_checkpoints 1 --save_schedule '' --anchor_tokens ''
       --stop_file "$OUT/STOP" "${PREC[@]}" "$@")
echo "fsdp2-rehearsal arm=$ARM nodes=$NNODES node=$NODE_RANK steps=$STEPS stop_after_min=$STOP_AFTER" | tee -a "$LOG"

run() { "$ROOT/tools/run_train.sh" "${FLAGS[@]}" 2>&1 | tee -a "$LOG"; }

TIMER=""
if [ "$STOP_AFTER" -gt 0 ] && [ "$NODE_RANK" = 0 ]; then
  ( sleep $((STOP_AFTER * 60)); echo "fsdp2-rehearsal: requesting a stop ($(date -u +%T))" | tee -a "$LOG"
    touch "$OUT/STOP" ) &
  TIMER=$!
fi
run
if [ -n "$TIMER" ]; then kill "$TIMER" 2>/dev/null || true; fi
[ "$STOP_AFTER" -gt 0 ] || exit 0

# A requested stop leaves OUT/STOP.saved once its checkpoint is on disk; a run that reached STEPS leaves none.
for _ in $(seq 120); do
  if [ -e "$OUT/STOP.saved" ]; then break; fi
  sleep 1
done
if [ ! -e "$OUT/STOP.saved" ]; then echo "fsdp2-rehearsal: finished without a stop" | tee -a "$LOG"; exit 0; fi
echo "fsdp2-rehearsal: stopped and saved; resuming ($(date -u +%T))" | tee -a "$LOG"
if [ -n "${MASTER_PORT:-}" ]; then export MASTER_PORT=$((MASTER_PORT + 1)); fi  # a fresh rendezvous port
run
