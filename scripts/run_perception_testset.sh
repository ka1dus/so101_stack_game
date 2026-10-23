#!/bin/zsh
# Phase 7: (re)build the 400-sample sim perception test set and refit the cameras (Mac, ~15 min).
cd ~/so101_stack_game
rm -rf data/perception/sim_v1
for s in 7001 7002 7003 7004; do
  MUJOCO_GL=cgl ~/.venvs/so101_stack_sim/bin/python scripts/perception_testset.py --n 100 --seed $s --out data/perception/sim_v1 > logs/perc_$s.log 2>&1 &
done
wait
MUJOCO_GL=cgl ~/.venvs/so101_stack_sim/bin/python scripts/perception_eval.py --check supply --out notes/phase7_supply_sim.txt > logs/perc_supply.log 2>&1
echo done > logs/perc_done.txt
