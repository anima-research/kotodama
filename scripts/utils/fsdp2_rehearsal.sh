#!/bin/bash
# A dress rehearsal for a long --fsdp2 run: shape C (presets 10b-bench-spire) on every GPU of the job's nodes at the
# launch flags measured best (layer compile scope, activation checkpointing, micro-batch 8, 10-layer groups, ZeRO-2),
# with the geometric monitor and the in-loop evals on, periodic resumable saves, and optionally one requested stop and
# resume inside the same job: node 0 creates CONTROL_DIR/STOP after STOP_AFTER_MIN minutes, every rank saves at the next
# step boundary and exits (--stop_file), and each node then relaunches the run, which resumes from that save.
#
# Usage: scripts/utils/fsdp2_rehearsal.sh ARM OUT DATA_PATH STEPS STOP_AFTER_MIN [extra train flags...]
#   ARM             mxfp8 | bf16
#   OUT             the checkpoint dir: node 0 writes it (one sharded copy; under several nodes
#                   --fsdp2_replica0_checkpoints), every node reads it at the same path, read-only is enough
#   STEPS           total optimizer steps (2,097,152 tokens each)
#   STOP_AFTER_MIN  0: no stop; N: one requested stop after N minutes, then the resume
# Environment:
#   SAVE_EVERY   periodic save cadence in steps (default 100; 0 disables every save)
#   BOUNDARIES   AttnRes block starts (default: eight blocks cut at the loop seams)
#   CONTROL_DIR  where STOP and STOP.saved live; every node must read and write it coherently (default OUT, which
#                fits one node only)
#   NODE_LOG_DIR where node k > 0 writes train.node<k>.log (default OUT); node 0 logs to OUT/train.log
#   KOTODAMA_MXFP8_VENV  the venv both arms run in (default <data root>/venv-mxfp8: a torchao build with the CUDA
#                mxfp8 cast kernels; both arms share it so the number format is the only difference)
set -euo pipefail
ARM="$1"; OUT="$2"; DATA="$3"; STEPS="$4"; STOP_AFTER="$5"; shift 5
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
NNODES="${NNODES:-1}"; NODE_RANK="${NODE_RANK:-0}"
CONTROL_DIR="${CONTROL_DIR:-$OUT}"
[ -f "$ROOT/site.env" ] && source "$ROOT/site.env"
MXVENV="${KOTODAMA_MXFP8_VENV:-${KOTODAMA_DATA_ROOT:-}/venv-mxfp8}"
if [ ! -x "$MXVENV/bin/python" ]; then echo "no mxfp8 venv at $MXVENV (set KOTODAMA_MXFP8_VENV)" >&2; exit 2; fi
export KOTODAMA_VENV="$MXVENV"
if [ "$NODE_RANK" = 0 ]; then
  mkdir -p "$OUT" "$CONTROL_DIR"
  LOG="$OUT/train.log"
else
  LOG_DIR="${NODE_LOG_DIR:-$OUT}"
  mkdir -p "$LOG_DIR"
  LOG="$LOG_DIR/train.node$NODE_RANK.log"
fi
case "$ARM" in
  mxfp8) PREC=(--fp8 --fp8_recipe mxfp8) ;;
  bf16) PREC=() ;;
  *) echo "unknown arm $ARM (mxfp8 | bf16)" >&2; exit 2 ;;
esac
REPL=()
if [ "$NNODES" -gt 1 ]; then REPL=(--fsdp2_replicas "$NNODES" --fsdp2_replica0_checkpoints); fi
# Eight AttnRes blocks cut at the loop seams (30 | 70 | 110 of the 140 effective layers), as syzygy's boundaries are.
BOUNDARIES="${BOUNDARIES:-0,1,10,20,30,70,110,125}"
# KOTODAMA_DEBUG_BATCH_HASH=STEPS logs every rank's batch digests at every step, so two arms with the same data seed
# compare step for step across a resume (off by default: four lines per step).
export KOTODAMA_DEBUG_BATCH_HASH="${KOTODAMA_DEBUG_BATCH_HASH:-0}"
FLAGS=(--config "$ROOT/configs/tsumugi-1b.yaml" --model_size 10b-bench-spire --attn_res_n_blocks 8
       --attn_res_boundaries "$BOUNDARIES" --no_attnres_eps_fp64_compat --mix_fraction 0 --data_path "$DATA"
       --global_batch_tokens 2097152 --total_steps "$STEPS" --log_every 1
       --fsdp2 "${REPL[@]}" --fsdp2_compile_scope layer --activation_checkpointing --micro_batch_size 8
       --fsdp2_layers_per_group 10 --no_fsdp2_reshard_after_forward --varlen_dynamic
       --geo_monitor --inloop_evals riders,induction
       --checkpoint_dir "$OUT" --save_every "${SAVE_EVERY:-100}" --keep_checkpoints 1 --save_schedule '' --anchor_tokens ''
       --stop_file "$CONTROL_DIR/STOP" "${PREC[@]}" "$@")
echo "fsdp2-rehearsal arm=$ARM nodes=$NNODES node=$NODE_RANK steps=$STEPS stop_after_min=$STOP_AFTER venv=$KOTODAMA_VENV" | tee -a "$LOG"

run() { "$ROOT/tools/run_train.sh" "${FLAGS[@]}" 2>&1 | tee -a "$LOG"; }

TIMER=""
stop_timer() { if [ -n "$TIMER" ]; then kill "$TIMER" 2>/dev/null || true; TIMER=""; fi; }
trap stop_timer EXIT
if [ "$STOP_AFTER" -gt 0 ] && [ "$NODE_RANK" = 0 ]; then
  # The timer polls and checks this script is still alive, so it can never outlive its run (a timer left by an
  # earlier, failed run would otherwise stop a later run sharing CONTROL_DIR).
  PARENT=$$
  ( END=$((SECONDS + STOP_AFTER * 60))
    while [ "$SECONDS" -lt "$END" ]; do sleep 5; kill -0 "$PARENT" 2>/dev/null || exit 0; done
    echo "fsdp2-rehearsal: requesting a stop ($(date -u +%T))" | tee -a "$LOG"
    touch "$CONTROL_DIR/STOP" ) &
  TIMER=$!
fi
run
stop_timer
[ "$STOP_AFTER" -gt 0 ] || exit 0

# A requested stop leaves CONTROL_DIR/STOP.saved once its checkpoint is on disk; a run that reached STEPS leaves none.
for _ in $(seq 300); do
  if [ -e "$CONTROL_DIR/STOP.saved" ]; then break; fi
  sleep 1
done
if [ ! -e "$CONTROL_DIR/STOP.saved" ]; then echo "fsdp2-rehearsal: finished without a stop" | tee -a "$LOG"; exit 0; fi
echo "fsdp2-rehearsal: stopped and saved; resuming ($(date -u +%T))" | tee -a "$LOG"
if [ -n "${MASTER_PORT:-}" ]; then export MASTER_PORT=$((MASTER_PORT + 1)); fi  # a fresh rendezvous port
run
