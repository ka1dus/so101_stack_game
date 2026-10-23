#!/bin/bash
# Phase 6: compare inference-time re-planning settings on one checkpoint (60 sim episodes each, seed 777, in parallel).
#   scripts/screen_replanning.sh <step>     e.g. 100000
cd "$(dirname "$0")/.." || exit 1
STEP=${1:-100000}; CK=data/checkpoints/act_v1/$STEP/pretrained_model
K=(-i "$HOME/.ssh/so101_pc_ed25519" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
if [ ! -f "$CK/model.safetensors" ]; then
  mkdir -p "data/checkpoints/act_v1/$STEP"
  scp -q -r "${K[@]}" "user@so101-gpu:C:/Users/User/so101_runs/act_v1/checkpoints/$STEP/pretrained_model" "data/checkpoints/act_v1/$STEP/" || exit 1
fi
run() { MUJOCO_GL=cgl HF_HUB_OFFLINE=1 ~/.venvs/so101_eval/bin/python scripts/sim_eval.py --policy "$CK" --n 60 --videos 60 \
          --out "data/eval/replan_${STEP}_$1" "${@:2}" > "data/eval/replan_${STEP}_$1.log" 2>&1; }
run te001 --temporal-ensemble 0.01 &
run k10 --n-action-steps 10 &
run k25 --n-action-steps 25 &
run k100 --n-action-steps 100 &
wait
for v in k100 k25 k10 te001; do printf "%-6s " $v; grep -E "^success" "data/eval/replan_${STEP}_$v/summary.txt"; done
