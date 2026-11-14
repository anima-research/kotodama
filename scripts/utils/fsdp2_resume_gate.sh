#!/bin/bash
# G6 resume gate for --fsdp2 on GPUs: a 16-step run that saves a sharded checkpoint at step 8 (OUT/a), a run resumed
# from a copy of that checkpoint (OUT/b), and a fresh rerun of the first (OUT/c, the determinism ruler). Prints the
# step 9-15 losses and grad norms of each and whether the final weights are bitwise equal: b == a is the gate
# wherever c == a. Extra train flags go to every run (e.g. --fsdp2_replicas 2 --fsdp2_node_local_checkpoints under a
# multi-node launch, with OUT a node-local path: each node then copies its own checkpoint for b). Only node 0 checks.
#
# Usage: scripts/utils/fsdp2_resume_gate.sh WORLD OUT DATA_PATH [extra train flags...]
set -euo pipefail
WORLD="$1"; OUT="$2"; DATA="$3"; shift 3
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
GATE="$ROOT/scripts/utils/fsdp2_gate.sh"
[ ! -e "$OUT/a" ] && [ ! -e "$OUT/b" ] && [ ! -e "$OUT/c" ] || { echo "refusing: $OUT/{a,b,c} already exist" >&2; exit 2; }
"$GATE" fsdp2 "$WORLD" "$OUT/a" "$DATA" 16 --save_every 8 --keep_checkpoints 2 "$@"
mkdir -p "$OUT/b"
cp -r "$OUT/a/step_00000008.dcp" "$OUT/b/"
"$GATE" fsdp2 "$WORLD" "$OUT/b" "$DATA" 16 --save_every 8 --keep_checkpoints 1 "$@"
"$GATE" fsdp2 "$WORLD" "$OUT/c" "$DATA" 16 --save_every 0 "$@"
[ "${NODE_RANK:-0}" = 0 ] || exit 0
source "$ROOT/tools/_env.sh"
python - "$OUT" <<'EOF'
import re, sys
from pathlib import Path
import torch
out = Path(sys.argv[1])
pat = re.compile(r"step=(\d+)/\d+ \| loss=([0-9.]+) .*?grad_norm=([0-9.]+)")
def steps(d):
    return {int(m.group(1)): (m.group(2), m.group(3)) for m in pat.finditer((out / d / "train.log").read_text())}
a, b, c = steps("a"), steps("b"), steps("c")
for s in range(9, 16):
    print(f"step {s}: a {a.get(s)}  b {b.get(s)}  c {c.get(s)}")
w = {d: torch.load(out / d / "weights" / "step_00000015.pt", map_location="cpu", weights_only=False)["model"]
     for d in "abc"}
eq = lambda x, y: all(torch.equal(w[x][k], w[y][k]) for k in w[x])
ca, ba = eq("c", "a"), eq("b", "a")
print(f"final weights: fresh rerun == run (determinism): {ca}; resumed == run: {ba}")
print("G6:", "PASS (bitwise)" if ba else ("FAIL" if ca else "not bitwise; the reference itself is not deterministic, "
                                         "compare the step 9-15 curves above against the a/c spread"))
EOF
