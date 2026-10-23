#!/bin/bash
# Phase 5 large generation (approved 2026-10-21): 4 workers x (500 ACT + 250 VLA) = 2,000 ACT + 1,000 VLA episodes.
# Re-running resumes each run where it stopped (generate.py counts the existing ep_* folders).
cd "$(dirname "$0")/.." || exit 1
PY=~/.venvs/so101_stack_sim/bin/python
for w in 0 1 2 3; do
  (
    MUJOCO_GL=cgl $PY scripts/generate.py --n 500 --mode act --seed $((1000 + w)) --run act_main_w$w
    MUJOCO_GL=cgl $PY scripts/generate.py --n 250 --mode vla --seed $((2000 + w)) --run vla_main_w$w
  ) 2>&1 | grep --line-buffered -v "^       {\|ik_place:" >> data/raw/logs/worker_$w.log &
done
wait
