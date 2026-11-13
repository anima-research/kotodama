#!/bin/bash
# Distributed training (NCA pre-pretraining, language pretraining, CPT): torchrun
# over the one trainer. GPUs per node: KOTODAMA_NPROC (default 8).
# Usage: tools/run_train.sh --config configs/<run>.yaml [overrides...]
# Several nodes: the scheduler's NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT / NPROC_PER_NODE (see below).
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/_env.sh"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1  # persist FA4 JIT kernels to disk
# Multi-node (the scheduler sets NNODES > 1 with NODE_RANK, MASTER_ADDR, MASTER_PORT and NPROC_PER_NODE on every
# node): one torchrun per node, rendezvous at the master. Otherwise a standalone single-node launch.
if [ "${NNODES:-1}" -gt 1 ]; then
  exec torchrun --nnodes="$NNODES" --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" --nproc_per_node="${NPROC_PER_NODE:-${KOTODAMA_NPROC:-8}}" \
    -m kotodama.training.train "$@"
fi
exec torchrun --standalone --nproc_per_node="${KOTODAMA_NPROC:-8}" -m kotodama.training.train "$@"
