#!/bin/bash
# G9: the FSDP2 parity gate across nodes, under HSDP. Run under a multi-node launch (the scheduler sets NNODES,
# NODE_RANK, MASTER_ADDR, MASTER_PORT, NPROC_PER_NODE on every node); OUT must be one directory every node sees.
# Arms, in order, each a scripts/utils/fsdp2_gate.sh run at the same world size and data:
#   ddp            the reference (DDP, micro-batch 2)
#   ddp-mb1        its own noise (DDP, micro-batch 1)
#   fsdp2          --fsdp2 with --fsdp2_replicas NNODES (shard within a node, replicate across nodes)
#   fsdp2-resume   the same, stopped at STEPS/2 with a sharded checkpoint, then relaunched to STEPS (resume across
#                  nodes); judged against the reference like the fsdp2 arm
# Interconnect settings (e.g. NCCL_IB_HCA) come from the environment. Compare afterwards with
# scripts/utils/fsdp2_parity.py (--test OUT/fsdp2 and --test OUT/fsdp2-resume against OUT/ddp, OUT/ddp-mb1).
#
# Usage: scripts/utils/fsdp2_g9.sh OUT DATA_PATH [STEPS]
set -uo pipefail
OUT="$1"; DATA="$2"; STEPS="${3:-200}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
GATE="$ROOT/scripts/utils/fsdp2_gate.sh"
NODES="${NNODES:-1}"; WORLD=$(( NODES * ${NPROC_PER_NODE:-${KOTODAMA_NPROC:-8}} ))
BASE_PORT="${MASTER_PORT:-29500}"
HALF=$(( STEPS / 2 ))
run() {  # run PORT_OFFSET ARM DIR [extra flags]
  local off="$1"; shift
  echo "=== g9 $2 node ${NODE_RANK:-0} $(date -u +%H:%M:%S) ==="
  MASTER_PORT=$(( BASE_PORT + off )) "$GATE" "$1" "$WORLD" "$2" "$DATA" "$STEPS" "${@:3}"
  echo "=== g9 $2 exit $? $(date -u +%H:%M:%S) ==="
}
run 1 ddp "$OUT/ddp"
run 2 ddp-mb1 "$OUT/ddp-mb1"
run 3 fsdp2 "$OUT/fsdp2" --fsdp2_replicas "$NODES"
run 4 fsdp2 "$OUT/fsdp2-resume" --fsdp2_replicas "$NODES" --save_every "$STEPS" --keep_checkpoints 1 \
  --stop_at_step "$HALF"
run 5 fsdp2 "$OUT/fsdp2-resume" --fsdp2_replicas "$NODES" --save_every "$STEPS" --keep_checkpoints 1
