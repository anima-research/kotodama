#!/bin/bash
# One arm of the FSDP2 parity gate (scripts/utils/fsdp2_parity.py compares the arms): the kotodama-tsumugi shape,
# eager, at a test batch of 24 sequences x 4096 tokens (divisible by 2, 4 and 6 ranks), seed 17 / data seed 42.
# Weights-only saves after the first step, at STEPS/4 and at the end; the log is written to OUT/train.log.
#
# Usage: scripts/utils/fsdp2_gate.sh ARM WORLD OUT DATA_PATH [STEPS] [extra train flags...]
#   ARM  ddp      the reference: DDP, micro-batch 2
#        ddp-mb1  the reference's own noise: DDP, micro-batch 1 (same data per rank, twice the accumulation)
#        fsdp2    the arm under test: --fsdp2, micro-batch 2
#        fsdp2-bf16rs  --fsdp2 with gradients reduced in bf16 (DDP's reduction precision), micro-batch 2
set -euo pipefail
ARM="$1"; WORLD="$2"; OUT="$3"; DATA="$4"; STEPS="${5:-200}"; shift 5 || shift $#
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
case "$ARM" in
  ddp) MB=2; EXTRA=() ;;
  ddp-mb1) MB=1; EXTRA=() ;;
  fsdp2) MB=2; EXTRA=(--fsdp2) ;;
  fsdp2-bf16rs) MB=2; EXTRA=(--fsdp2 --fsdp2_reduce_dtype bf16) ;;
  *) echo "unknown arm $ARM (ddp | ddp-mb1 | fsdp2 | fsdp2-bf16rs)" >&2; exit 2 ;;
esac
SEQ=4096; SEQS=24; T=$((SEQ * SEQS)); Q=$((STEPS / 4))
[ "$Q" -ge 2 ] || { echo "STEPS must be >= 8" >&2; exit 2; }
SCHEDULE="$T:$T,$(( (Q - 1) * T )):$(( Q * T )),$(( (STEPS - Q) * T )):$(( STEPS * T ))"
mkdir -p "$OUT"
export KOTODAMA_NPROC="$WORLD"
export KOTODAMA_DEBUG_BATCH_HASH="${KOTODAMA_DEBUG_BATCH_HASH:-20}"
echo "fsdp2-gate arm=$ARM world=$WORLD steps=$STEPS micro_batch=$MB schedule=$SCHEDULE"
"$ROOT/tools/run_train.sh" --config "$ROOT/configs/tsumugi-1b.yaml" "${EXTRA[@]}" \
  --no_compile --no_geo_monitor --inloop_evals '' --mix_fraction 0 --no_attnres_eps_fp64_compat \
  --data_path "$DATA" --sequence_length "$SEQ" --global_batch_tokens "$T" --micro_batch_size "$MB" \
  --total_steps "$STEPS" --warmup_steps 20 --log_every 1 --norm_gain_every 10 \
  --save_every 0 --save_schedule "$SCHEDULE" --anchor_tokens '' \
  --checkpoint_dir "$OUT" --checkpoint_shm_dir "/dev/shm/kotodama-fsdp2-gate-$ARM-w$WORLD" \
  "$@" 2>&1 | tee "$OUT/train.log"
