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
# Point-to-point rails (one NIC on each node per cable) only pair up when GPU i sits beside NIC i on both nodes, i.e.
# when the nodes run the same GPU indices. If they differ, use only the first listed NIC: one rail every GPU reaches.
if [ "$NODES" -gt 1 ] && [ -n "${NCCL_IB_HCA:-}" ]; then
  SETS="$OUT/.gpu-sets-${HEIMDALL_JOB_ID:-run}"
  mkdir -p "$SETS"
  echo "${CUDA_VISIBLE_DEVICES:-all}" > "$SETS/node${NODE_RANK:-0}"
  for _ in $(seq 180); do
    [ "$(find "$SETS" -name 'node*' | wc -l)" -ge "$NODES" ] && break
    sleep 1
  done
  if [ "$(cat "$SETS"/node* | sort -u | wc -l)" -gt 1 ]; then
    FIRST="${NCCL_IB_HCA#=}"; FIRST="${FIRST#^}"; FIRST="${FIRST%%,*}"
    export NCCL_IB_HCA="=$FIRST"
    echo "g9: GPU sets differ across nodes ($(cat "$SETS"/node* | tr '\n' ' ')); NCCL on one rail: $NCCL_IB_HCA"
  fi
fi
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
