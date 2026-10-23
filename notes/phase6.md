# Phase 6: Train and evaluate in sim (ACT v1)

Overnight log with all the details: `notes/phase6_overnight.md`. Per-checkpoint results: `notes/phase6_eval_log.md`.

## 1. Training
- GPU PC (RTX 3060 Ti 8 GB, native Windows), LeRobot ca69a206, ACT defaults (chunk 100, ResNet-18, lr 1e-5,
  KL weight 10) + AMP, batch 8, **100k steps = 1.14 epochs** of the 2,000-episode ACT dataset (699,945 frames).
  6 h 17 min, 0.23 s/step, 5.5 GB VRAM. Final loss 0.048.
  Checkpoints: PC `C:\Users\User\so101_runs\act_v1\checkpoints\`; Mac copies in `data/checkpoints/act_v1/`.
- The first dataset was unusable (the streaming export dropped frames); it was rebuilt with no-drop settings, a frame
  check and a pixel check. See the overnight log 03:39–04:56.

## 2. Evaluation method (`scripts/sim_eval.py`, Mac, eval venv `~/.venvs/so101_eval`)
- Closed loop at 30 Hz from the rendered `front`/`wrist` frames + joint state, the real arm's 10-unit
  max_relative_target, home ± 3°, nominal (calibrated) visuals. Success = CLAUDE.md criterion (box on top, stack
  standing 3 s after release), judged with privileged state. The judge was checked against the expert (10/12 agree; the 2
  differences explained: backdrop flag, resting-stack creep → such states are now skipped).
- Eval states from `game_states.sample_state` (uniform heights 1–7, hard share 0.5) with seeds never used for training.

## 3. Results
Checkpoints at default inference (60 states, seed 777): 10k 3 %, 20k 12 %, 30k 12 %, 40k 33 %, 50k 35 %, 60k 28 %,
70k 40 %, 80k 45 %, 90k 38 %, 100k 42 %: a plateau at ~40 % from 40k on (± 6 % noise at n = 60).

Inference re-planning, 100k checkpoint, the same 60 states:
| setting | success |
|---|---|
| execute 100 steps per prediction (trained default) | 25/60 = 41.7 % |
| re-plan every 25 steps | 22/60 = 36.7 % |
| re-plan every 10 steps | 1/17 = 6 % (stopped) |
| **temporal ensembling 0.01** (re-plan every step, averaged) | **34/60 = 56.7 %** |

**Temporal ensembling on 200 new states (seeds 1001–1004): 98/200 = 49.0 %** (`data/eval/te200_summary.txt`).
| bucket | result | target (CLAUDE.md §1) |
|---|---|---|
| stack height 0–3 | 31/83 = 37 % | ≥ 90 % |
| height 4+ (all) | 67/117 = 57 % | ≥ 70 % on crooked 4+ |
| height 4+ and crooked ≥ 10 mm | 18/33 = 55 % | ≥ 70 % |
| large onto small (no3_on_no10) | 5/18 = 28 % | ≥ 70 % |
| picked no3 / no35 / no10 | 54 % / 44 % / 58 % | ≥ 90 % (0–3) each |
- Failures: missed grasp 92, drop 3, knocked on approach 3, knocked on release 1, misplaced 3. Wrong box 20/200
  (10 %, the nearest-box rule). Arm touched the backdrop 2/200.
- By supply size (boxes left to pick from): 1: 54 %, 2: 48 %, 3: 58 %, 4: 64 %, **5: 38 %, 6: 24 % (73 % missed
  grasps)**, 7: 75 %. Dense, partly emptied supplies are the hardest. That's why "height 0–3" scores lower than 4+,
  the opposite of the expected difficulty.
- **No Section 1 sim criterion is met.** The placement itself is good once a box is held: of the episodes where a box
  was grasped, ~90 % ended in a correct placement (98 of the 108 non-missed-grasp episodes).

## 4. Why: missed-grasp review (videos, both cameras)
1. Aim error of ~1–2 cm, about half a box width (descends into gaps, straddles two packed boxes), not millimetres.
2. With 100-step chunks the descent is planned from high above and runs open loop. Temporal ensembling lets the
   wrist view correct it (+15 points), but the error is still too large in dense supplies.
3. No retry after a miss: the demos never contain a failed grasp, so the policy "places" an empty gripper.
4. Sparse supplies: the wrist view is mostly black mat; the box is ~50 px wide in `front`.
5. The expert's grasp is very tight (opening = width + ~6 mm → ~3 mm per side), which doesn't forgive learned
   positioning errors.
6. Only ~1 epoch of training; the curve flattened at 40k–100k at default inference.

## 5. Decisions (to confirm)
- Inference for ACT v1: **temporal ensembling, coefficient 0.01** (n_action_steps = 1). The real-arm runner
  (Phase 8/9) must use the same, and run inference every control tick (~20 ms on the Mac's GPU).

## 6. Next steps (proposed, not started)
1. **Data v2 (expert):**
   - wider pre-grasp opening (width + 15–20 mm; the clearance check allows ~10 mm/side), slow final descent and a
     pause before closing;
   - **recovery data** (DART-style: noise on the expert's commands, the expert's corrections as labels);
   - **retry-after-miss demos**;
   - a pick rule without near-ties (wrong box 10 %); a stricter resting-stack check.
   - Over-sample dense supplies (5–6 boxes left).
2. **Training v2:** 300–500k steps (~19–30 h on the 3060 Ti), or a smaller image size for more epochs per hour (check
   size discrimination first).
3. Later: DAgger in sim (the expert relabels states the policy visits; needs the expert to plan from any state).
4. Evaluate every new version on the same 200 states (seeds 1001–1004) with temporal ensembling, so results are
   comparable.

## 7. Compute: second GPU machine `botmind` (2026-10-22)
- Windows 11 Home, admin user `botmi`, Tailscale 100.101.192.30 (SSH with the project key). **RTX 4070 Ti SUPER
  16 GB** (the user called it a 4080), i7-14700KF 20 cores / 28 threads, 32 GB RAM. Corporate background software
  installed (work machine).
- Isolated sim env: `C:\Users\botmi\so101_stack_game\.venv-sim` (Python 3.12.10, MuJoCo 3.16.0, the Mac's pins).
- **Offscreen rendering works over SSH** on the NVIDIA driver (OpenGL 4.6), 1.3 ms per front+wrist pair (Mac ~9 ms);
  image identical to the Mac's (mean diff 0.72/255).
- Generation: 1 process ~11 s/episode; **16 workers → 2.2 s/episode overall** (Mac: ~6.5 s with 4 workers).
- Supply layout (`scripts/supply_layout.py`): 2 cm gaps in every direction don't fit the reachable mat (297 cm²
  vs ~373 needed). With ≥ 1 cm in every direction: **≥ 20.5 mm beside every long side**
  (`data/layout_search/gap1cm_s5.yaml`, `notes/img/phase6_gap1cm_s5.png`); pick-order margin only 5.3 mm (to improve).
