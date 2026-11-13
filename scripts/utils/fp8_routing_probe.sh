#!/bin/bash
# Does fp8 freeze the AttnRes routing (BUG-LEDGER B1: with DDP + torch.compile + fp8 together, the pre-MLP routing
# queries got exactly zero gradient and stayed at their zero init)? One arm per call, the 1B ladder-A0 recipe
# compiled, STEPS steps at ~2M tokens, one weights-only save at the end; then counts the routing queries still
# exactly zero, by site kind (pre-attn / pre-MLP / final). The first boundary layer's pre-attn query is frozen by
# design and is not counted. Telemetry (router/dead_sites/{attn,mlp,final} every 10 steps) goes to the scheduler.
#
# Usage: scripts/utils/fp8_routing_probe.sh ARM WORLD OUT DATA_PATH [STEPS]
#   ARM  ddp-mxfp8          DDP + compile + --fp8_recipe mxfp8 (B1's conditions with the mxfp8 recipe)
#        fsdp2-bf16         FSDP2 + compile, no fp8 (control)
#        fsdp2-mxfp8        FSDP2 + compile + mxfp8
#        fsdp2-mxfp8-eager  FSDP2 + mxfp8, no compile
set -euo pipefail
ARM="$1"; WORLD="$2"; OUT="$3"; DATA="$4"; STEPS="${5:-150}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
case "$ARM" in
  ddp-mxfp8) EXTRA=(--fp8 --fp8_recipe mxfp8) ;;
  fsdp2-bf16) EXTRA=(--fsdp2) ;;
  fsdp2-mxfp8) EXTRA=(--fsdp2 --fp8 --fp8_recipe mxfp8) ;;
  fsdp2-mxfp8-eager) EXTRA=(--fsdp2 --fp8 --fp8_recipe mxfp8 --no_compile) ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
MB=8; SEQ=4096; MICRO=$((WORLD * MB * SEQ)); BATCH=$(( (2097152 / MICRO) * MICRO )); END=$((STEPS * BATCH))
mkdir -p "$OUT"
export KOTODAMA_NPROC="$WORLD"
echo "fp8-routing-probe arm=$ARM world=$WORLD steps=$STEPS batch=$BATCH"
"$ROOT/tools/run_train.sh" --config "$ROOT/configs/ladder-a0.yaml" "${EXTRA[@]}" \
  --no_geo_monitor --inloop_evals '' --no_attnres_eps_fp64_compat --varlen_dynamic \
  --data_path "$DATA" --micro_batch_size "$MB" --global_batch_tokens "$BATCH" --total_steps "$STEPS" \
  --log_every 10 --norm_gain_every 10 --save_every 0 --save_schedule "$END:$END" --anchor_tokens '' \
  --checkpoint_dir "$OUT" --checkpoint_shm_dir "/dev/shm/kotodama-fp8-probe-$ARM" 2>&1 | tee "$OUT/train.log"
source "$ROOT/tools/_env.sh"
python - "$OUT" "$ARM" <<'EOF'
import sys
from pathlib import Path
import torch
from kotodama import presets
out, arm = Path(sys.argv[1]), sys.argv[2]
save = sorted((out / "weights").glob("step_*.pt"))[-1]
sd = torch.load(save, map_location="cpu", weights_only=False)["model"]
first = min(presets.DEFAULT_BOUNDARIES["1b"])
zero = {"attn": [], "mlp": [], "final": []}
total = {"attn": 0, "mlp": 0, "final": 0}
for k, v in sd.items():
    if not k.endswith("res_query"):
        continue
    kind = "final" if k.startswith("final") else ("attn" if k.endswith("attn_res_query") else "mlp")
    if kind == "attn" and k == f"layers.{first}.attn_res_query":
        continue  # frozen by design (never used)
    total[kind] += 1
    if torch.count_nonzero(v) == 0:
        zero[kind].append(k)
print(f"{arm} ({save.name}): routing queries still exactly zero -> "
      + ", ".join(f"{k} {len(zero[k])}/{total[k]}" for k in zero))
print("B1 signature (frozen routing):", "YES" if any(zero.values()) else "no")
EOF
