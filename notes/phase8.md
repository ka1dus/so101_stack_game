# Phase 8: real placement trials (no game yet)

Status 2026-10-26: tools built and checked without motion (§5); **no real motion yet** (checkpoint, CLAUDE.md working rule 8). ACT v4
(trimmed-backdrop fine-tune) is training on botmind; the first real trials use it once its sim check is in.

## 1. Tools

**`skill/runner.py`, `run_place_skill()`:** the ACT policy in our own 30 Hz loop, not the `lerobot-rollout` CLI (Section 4). Per tick it:
- reads the joints and both cameras through LeRobot's `SO101Follower` and OpenCV cameras (640×480, RGB);
- undistorts `front` with the calibrated f/k1 (cached remap maps, identical to `cv2.undistort`, 1.2 ms);
- resizes both images to 320×240 and runs the policy with temporal ensembling 0.01;
- clips the target to ±10 units of the reading (the training clip and `max_relative_target`) and to the calibrated ranges;
- runs the **safety check**: the commanded jaw tips must stay above −12 mm (mapping error ~1 cm), below 35 cm, and within 5 cm of the table outline; a failing target makes the arm hold and ends the skill;
- sends the target.

The skill ends as in `sim_game.py`:
- `released`: drop_check sees the release in the zone above the stack, then the tips rise 3.5 cm or 1.5 s passes;
- `drop`;
- `guard`;
- `timeout` after 25 s.

After that comes a scripted move home at 30 units/s, raising the tips 6 cm straight up first unless the box was released (twin kinematics, damped least squares).

**`scripts/phase8_trial.py`:** the trial session.
- Reference frame (template full, stack area empty), then camera refit and the supply-slot monitor.
- Per trial you type the stack and the type to pick. The planner (ACT pick rule, `game_states.choose_pick`) lists the template slots that form the stack and the boxes to set aside.
  Every planned supply is a template with boxes taken out, which is in the training distribution (layout progression).
- Then:
  1. a check that the supply matches the plan;
  2. the stack top from perception;
  3. a 3 s countdown and the skill;
  4. home, then a 3 s settle;
  5. the perception call (picked slot and collapse check, with the arm frames at release and at the end);
  6. **your call** (ground truth) with a failure type.
- Logs go to `data/real_trials/<session>/trials.jsonl`, with front/wrist mp4 and the joint trace.
- `--report` gives results by stage, picked type, pairing and height, plus how often perception agreed with you.
- **Ctrl-C:** the arm holds where it is, then you choose: return home slowly, torque off, or quit.

**Environment:** `~/.venvs/so101_eval`.
- It has LeRobot 0.6.2 from `~/lerobot`, MuJoCo 3.16, torch 2.11 with MPS, and `feetech-servo-sdk` 1.0.0 and `deepdiff` 8.6.2, which with pyserial make up LeRobot's `feetech` extra (added 2026-10-26, the same versions as the conda env).
- The conda `lerobot` env's MuJoCo 3.8.1 cannot load `sim/scene.xml`, and that env is left unchanged.

## 2. Timing (Mac, MPS)

- ACT step on MPS: 26 ms median (CPU: 285 ms).
- Full loop with a fake arm serving recorded frames: **29.3 Hz**. The real serial reads and writes add a few ms; the no-motion run measures it (`loop_hz` in the log).
- Training ran at 30 fps. A loop a little slower than 30 Hz executes the same per-tick targets more slowly, which is on the cautious side.

## 3. Trial plan (stages from CLAUDE.md, about 10 trials per bucket)

Heights start at 1: training never placed onto the empty base, because the human always starts.

| stage | buckets | example specs |
|---|---|---|
| 1 | same type, heights 1–3: no3 on no3 (h1–2), no35 on no35 (h1–3) | `no3 > no3`, `no3 no3 > no3`, `no35 no35 > no35` |
| 2 | small on large, then large on small | `no3 > no10`, `no10 > no3`, `no35 no10 > no3` |
| 3 | mixed stacks of height 3–4, some slightly crooked | `no3 no35 no10 > no3`, `no35 no3 no35 no10 > no35` |

no10 on no10 is impossible: there is only one small box.

## 4. Checkpoint before the first real motion

1. No-motion run: everything (cameras, refit, policy, perception, logging) with the arm holding home. Check `loop_hz` and the saved front/wrist videos.
2. The user approves real motion.
3. First motion trial: stage 1, `no35 > no35`, with a hand near the e-stop (Ctrl-C).
EOF
git status --short
## 5. No-motion run, 2026-10-26 (`data/real_trials/20261026_201850`, ACT v3)

The trial was `no35 > no35`: stack from slot 5, pick slot 6.

**Everything worked end to end:**
- connect, cameras, reference frame, camera refit, supply check (no warning), stack-top estimate, 25 s of policy inference, 3 s settle, logging and both videos.

**Loop:** 29.2 Hz. Compute per tick was 30.3 ms median and 35.1 ms p95, including the serial read and write. The first tick took 483 ms (warm-up).
- The joint readings stayed fixed. The policy's targets saturated the ±10-unit clip, because it was trying to move.

**Camera:** the new reference matches the Phase 7 photo within 0.1 px (phase correlation), so the tripod has not moved.

**The refit is loosely constrained.** On the two identical images it lands 9 mm vs 27 mm and 0.4° vs 1.5° from the nominal camera, at the same mask cost (0.1163 vs 0.1166).
- The two fits still project the scene almost identically: stack column ≤ 2.2 px, table corners ≤ 5.4 px. So the image-space checks are unaffected.
- **Open:** add a weak pull toward the nominal calibration, so the fitted camera pose itself is repeatable.

**Wrist:** at the exact home pose the real view matches the sim's layout: medium box at the left, mat corner, yellow area top right, jaws at the bottom.
- The real image is softer (focus), and the real jaws look a little larger. Training randomizes focal length ±5 % and pose.
- See `notes/img/phase8_wrist_home_sim_vs_real.jpg`. (A sim episode's first frame looked different only because of the ±3° home jitter.)
- A missing `deepdiff` (LeRobot's Feetech bus needs it) was added to the eval venv before this run.

## 6. Camera-refit prior and ACT v4, 2026-10-26

**Refit prior** (`perception/register.py` `PRIOR_W` 5e-4 per deg² and per cm²): a weak pull toward the nominal calibration.
- Real: the two identical reference images now refit 4.8 mm apart (projections within 1.1 px), against 9 vs 27 mm before. The real clips are still 11/11.
- Sim (400 samples, true cameras known), worst projected point error:

  | fit | median | p95 | max |
  |---|---|---|---|
  | nominal | 11.3 px | 21.7 px | 29.6 px |
  | no prior | 3.0 px | 10.7 px | 41.4 px |
  | **prior** | **2.3 px** | **4.4 px** | **7.2 px** |

- Supply slots: empty slots read as full 9 → 0. Sure and correct turns 363 → 373, "ask the player" 37 → 27. Turn level 394/400; every miss is in the ask band except one occupied slot at dark 0.85, which was there before.
- Collapse unchanged: 112/126, 0/274.
- Old fits are kept as `cam_refined_noprior.json`. `notes/phase7_supply_sim.txt` and `notes/phase7_collapse_sim.txt` are re-scored.

**ACT v4** (`scripts/b4_v4.ps1`, botmind):
- Data: 1,500 new episodes with the trimmed backdrop (1,000 game distribution plus 500 crooked height 4–7 no3/no35), merged with v2 and v3 into 5,500 episodes.
- Training: 40k steps from v3 60k, which took 46 min.
- On the same 460 states it gets **416/460**: v3 on the trimmed board 409, v3 on the old board 417.
  - Against v3 on the trimmed board: 23 states fixed, 16 broken.
  - Height 0–3: 98 / 96 / 100 %. Crooked 4+: 76 / 85 / 94 %. Large on small: 86 %. 0 backdrop touches.
  - Wrong-box picks 49 → 56; any placed box counts in the game.
- **v4 is the policy for the real trials** (`data/policies/act_v4_ft_040000`, MD5 checked against botmind; the trial script's default).

## 7. First real-motion trial, 2026-10-27 (`data/real_trials/20261027_115530`, ACT v4)

Before it: a no-motion check (`20261027_114400`). The camera refit agrees with 26 Oct within 1.7 px.
- Fix: `top_xy()` now keeps only the largest blob. A patch of auto-exposure change on the white table had pulled the detected stack top 3.6 cm off the real box; after the fix both sessions give 8–9 mm from the X.
- Unchanged by the fix: sim collapse (112/126, 0/274) and the real clips.

**Trial 1, `no35 > no35`: fail.**
1. Picked the wrong box: a large box from slot 2, where the medium in slot 6 was expected. The grasp held.
2. Carried it low (tips peaked at 5.5 cm).
3. Stalled 9 cm short of the stack from 12 s, the held box apparently resting on the table.
4. Timed out at 25 s and went home still holding the box.
5. No unsafe motion.

**Replay of the same commands in sim** (`scripts/phase8_replay.py`, `notes/phase8_replay_20261027_115530_t001.txt`, `notes/img/phase8_replay_20261027_115530_t001.png`).

Mean gap between target and reading (units ≈ degrees):

| phase | real elbow | sim elbow | other |
|---|---|---|---|
| reach, unloaded (0–5 s) | −2.4 | −0.2 | real wrist roll 3.4, sim 0.6 |
| carry, moving (9–12 s) | −7.3 | 0.0 | |
| carry, stalled (12–25 s) | −10 | 0.0 | real wrist flex −3 to −4; −10 is the per-tick clip, so the joint isn't moving at all |

Real tips sit about 1.5–2 cm lower than sim during the carry, with the xy gap growing to 3.9 cm.

What this shows:
- The sim servos are near-perfect position trackers (gain 998 N·m/rad, capped at 3.35 N·m). The real STS3215s lag a few degrees while moving, more under load.
- The policy was trained on that perfect tracking. On the real arm the carry comes out lower and shorter, so the box drags on the table and the elbow stalls.

**Next:** fit the sim servo model (gain, damping, force cap) to the real response, using this trace and a scripted tracking test with and without a box. Then re-check the policy in sim with the fitted servos, and fine-tune if it drops. The wrong-box pick is a separate issue: a vision gap in the policy's pick rule.

## 8. Servo tracking test and sim servo fit, 2026-10-27

**Real test** (`scripts/phase8_tracking.py`, 46 s high over the table; logs `data/real_tracking/20261027_121433_empty.npz` and `..._121605_box.npz`):
- **Moving:** the real servos lag up to ~9 units in fast moves.
- **Holding still:** they sit 2–4 units off target (elbow; worse against gravity). LeRobot runs them at P 16, I 0, D 32 (`so_follower/config_so_follower.py:45-47`).
- **The No.3 box (38 g)** barely changes this, so the 10-unit elbow stall in trial 1 was contact (the held box on the table), not weight.
- **Loop rate:** 27.6 Hz in this joint-only loop.

**Fit** (`scripts/phase8_servo_fit.py`; both logs plus trial 1's first 12 s; `sim/assets/servo_fit.yaml`, proposal, not applied to `scene.xml`):

| joint | kp (N·m/rad) | kv (N·m·s/rad) |
|---|---|---|
| shoulder_pan | 34.3 | 2.66 |
| shoulder_lift | 34.3 | 2.66 |
| elbow_flex | 21.2 | 2.66 |
| wrist_flex | 13.1 | 0.71 |
| wrist_roll | 21.2 | 2.66 |

- **How much softer:** the sim currently uses kp 998 and kv 2.73, so the fitted servos are 30–75× softer. The force cap stays at 3.35 N·m.
- **Mean |sim − real| per joint:** 1.02 → 0.62 units (elbow 2.17 → 1.30). Plot: `notes/img/phase8_servo_fit_empty.png`.
- **What's left:** the real elbow's larger offset when holding still (about 4 vs 2.5 units), likely servo deadband or friction; not modelled yet.

**Next:** run the policy in sim with the fitted servos and see whether it reproduces the low, short carry. If it does, fine-tune on data generated with the fitted servos plus servo randomization.

## 9. Soft-servo data and fine-tune, 2026-10-27

**v4 with the fitted servos** (same 460 sim states as the v4 check): 276/460 (60 %), down from 416/460 with the old stiff servos. Most new failures were drops and knocks while carrying, which matches the real trial.

**Expert fix** (`scripts/grasp_test.py`, `Sim.set_comp`): with soft servos the scripted expert itself knocked stacks (12 of 44 attempts). It now sends command = plan + (gravity torque + kv · plan velocity) / kp, clipped to ±10 units of the reading (as `max_relative_target=10` on the real arm). The action label is that command. With the fitted servos: 30/34 ok, 0 knocks.

**Data** (`scripts/generate.py --servo sim/assets/servo_fit.yaml --servo-jitter 0.3`; kp and kv × exp(U(±0.3)) per joint per episode): 3,009 episodes. 1,000 targeted ones (height 4+, no3/no35, crooked) plus 2,009 game-rule ones, made on botmind (16 workers) and the 3060 Ti PC (new sim venv, 5 workers). Exported 320×240 as `so101_stack_sim_act_soft_half`.

**Fine-tunes from v4 40k** (batch 8, AMP): A = 40k steps (botmind), B = 80k steps (PC), checkpoints at 40k and 80k.

**Results with the fitted servos** (460 states; `scripts/b4_soft_eval.ps1`; botmind `data\eval\<label>_*`):

| policy | all | stacking with the right box held | missed grasp | wrong box |
|---|---|---|---|---|
| v4, stiff servos (reference) | 416/460 | 364/396 (92 %) | 8 | 56 |
| v4, fitted servos | 276/460 | 234/318 (74 %) | 56 | 86 |
| A (40k) | 324/460 | 295/339 (87 %) | 53 | 68 |
| B40 | 341/460 | 309/345 (90 %) | 52 | 63 |
| **B80** | **354/460** | **309/344 (90 %)** | **27** | 89 |

- **Stacking is back to the stiff-servo level** (90 vs 92 %). Drops while carrying fell from 26 to 7, knocks on approach from 23 to 9.
- **More steps helped grasping, not stacking:** B40 → B80 halved missed grasps; stacking stayed at 90 %.
- **Section 1 criteria (B80, `confirmation_softB80.md`):** large-on-small 86.4 % (confirmed), crooked 4+ no35 83.3 % (confirmed), no3 71.1 % (met). The small-box (no10) buckets are low (45 % / 54 %) only through wrong-box picks and missed grasps: when B80 held the small box it stacked it almost every time. The wrong-box pick is set aside for now (user).
- **B80 is the policy for the next real trial** (`data/policies/act_softB_080000`).

**Robustness of B80 to servo fit error** (2026-10-28; `sim_eval.py --servo-scale`, kp and kv of every joint scaled; `scripts/b4_robust_eval.ps1`; botmind `data\eval\softB80_x070_*`, `softB80_x140_*`):

| servos | all | stacking with the right box held | height 6+ | large on small | missed grasp |
|---|---|---|---|---|---|
| fit × 0.7 (softer) | 367/460 | 315/349 (90 %) | 65/84 | 89/98 | 26 |
| fit | 354/460 | 309/344 (90 %) | 70/86 | 90/100 | 27 |
| fit × 1.4 (stiffer) | 369/460 | 329/375 (88 %) | 65/82 | 82/100 | 22 |

B80 holds 88–90 % stacking across 0.7–1.4× the fitted gains (differences are within run-to-run noise). This range is about the training jitter (×exp(±0.3) = 0.74–1.35), so values further out are untested. For the real trial: a failure there is unlikely to come from the servo gains being somewhat off; check the unmodelled parts first (servo deadband or friction at rest, cameras, grasp).

## 10. Real trials with B80, 2026-10-28: release too high

**Sessions:** `20261028_104808` (no-motion), `20261028_105023` (trial lost to a camera crash; Photo Booth was open), `20261028_105812` (trial 1).
- **Both real trials** (`no35 > no35`): right box picked, carried high (tips up to 8.8 cm; the low-carry failure is gone), but released about 2 cm too high and 2–3 cm off the stack. The box fell beside it.
- **The servos tracked:** command − reading < 1 unit at release, so the policy chose to open there.

**Sim twin of trial 1** (`scripts/phase8_sim_twin.py`; same stack, start pose, B80, fitted servos): 4/4 stacked.

| | jaw tips at release (x, y, z) | from the top |
|---|---|---|
| real | (18.3, −10.0, 4.3) cm | 3.3 cm |
| sim (mean of 4) | (19.6, −8.8, 2.7) cm | 1.5–2.2 cm |

Frames: `notes/img/phase8_twin_20261028_105812_t001_{real,sim0}_{pre,rel}.png`.

**Wrist camera gap** (`scripts/phase8_wrist_gap.py`; the sim wrist camera rendered at the real joint readings before the grasp, yellow/black masks compared):
- **At home:** the jaws line up in real and sim (the fit used them), but the scene beyond does not.
- **Hovering over the supply box (2–4.4 s):** the real view is shifted about +27 px right and −51 px up (≈ 3° and 5.6°) and rolled ~4°. The boxes look larger in the real view, i.e. closer than sim says. Mask IoU 0.2 → 0.62 after the best shift.
- **Focus:** the real wrist image is strongly defocused at close range (Laplacian variance 9 vs 845 in sim; ≈ Gaussian σ 2 px at 640×480). Training blur is σ ≤ 1.5, and 40 % of frames have none.
- **Overlays:** `notes/img/phase8_wristgap_20261028_105812_t001_{000,066,132}.png`.

**Reading:** the policy decides when to open from the images and joint readings (there is no separate release rule). Its wrist view of the stack top differs from anything it saw in training, so it judges the box to be just above the top while it is still ~2 cm up. The wrist camera mount was fitted only from the jaw tips and two table points (Phase 3, checkerboard skipped), and the fit disagreed with the ruler by ~4 cm out of the jaw plane.

**Next:** calibrate the wrist camera properly (a printed target on the table, the arm at several poses), check overlays, then regenerate data with the corrected mount (and stronger close-range blur) and fine-tune.

## 11. Wrist camera calibration, 2026-10-28

**Captures** (`scripts/wrist_calib_capture.py`; board `notes/wrist_calib/charuco_board_A4.pdf`, ChArUco 5×7, 28 mm):
- **Part A, handheld at R_raise:** 41 frames, `data/wrist_calib/20261028_115416_handheld`.
- **Part B, tour:** 14 sim-checked poses over the board lying on the yellow area (`scripts/wrist_calib_poses.py`, `notes/img/wrist_calib_poses.png`); `data/wrist_calib/20261028_115746_poses`. Real frames: `notes/img/wrist_calib_real_poses.png`.

**Intrinsics** (`scripts/wrist_calib_fit.py intrinsics`; 37 frames, board 18–49 cm): f 508.0 px (datasheet 518), centre (324.7, 243.2), k1 0.014, k2 −0.090, rms 0.29 px. Close to what the sim uses; not the cause.

**Mount** (`wrist_calib_fit.py mount`; corners of 92 detected markers, 368 points; joint readings through the validated FK; hand-eye start, then a joint least-squares fit of mount + board pose):

| mount | tour frames, median rms | today's trial frames (`phase8_wrist_gap.py`) |
|---|---|---|
| current sim | 19.7 px | shift +24 / −45 px, IoU 0.36 |
| fitted, board flat on the table | 9.1 px | shift +15 / −90 px, IoU 0.30 (does not transfer) |
| **fitted, board pose free** | **5.1 px** | **shift +9 / 0 px, IoU 0.68** |

- **Change from the current sim mount:** rotated 8.6° (tilt +6.3°, pan +5.7°, roll −0.8°) and moved 21 mm.
- **The free fit puts the board 2.5 cm above the table and tilted 3°.** That is either real (board not lying flat) or arm-angle error at those poses. Fitting shoulder_lift / elbow / wrist_flex offsets instead gives +6° on shoulder_lift, which the Phase 1/3 touch tests rule out.
- **The free-board mount transfers to independent frames** (the trial's approach over the supply), so it is the one to use: `data/wrist_calib/20261028_115746_poses/mount_free.json`. Overlay: `notes/img/phase8_wristgap_20261028_105812_t001_132_newmount.png`.

**Next (pending the user's OK):** put the fitted mount and intrinsics in `sim/assets/cameras.yaml`, rebuild `scene.xml`, raise the wrist blur randomization to the measured close-range level (σ up to ~2.5 px at 640×480), regenerate the soft-servo data, fine-tune from B80, evaluate in sim, then stage 1 real trials again.

## 12. Retrain with the calibrated wrist camera, 2026-10-28

**Data:** 3,080 soft-servo episodes with the calibrated wrist mount + intrinsics and wrist blur σ 0.6–2.5 px (1,000 targeted `act_wctall_*`, 2,080 game `act_wcgame_*` / `act_pcwcal_*`; botmind + PC), exported 320×240 as `so101_stack_sim_act_wcal_half`. **Fine-tune** from B80, 60k steps (`C:\Users\botmi\so101_runs\act_wcal`). **Eval** on the same 460 states with the fitted servos and the new scene (`scripts/b4_wcal_chain.ps1`; botmind `data\eval\wcal30_*`, `wcal60_*`, `b80wcal_*`; summaries `data/eval/confirmation_{wcal30,wcal60,b80wcal}.md`).

| policy (scene) | all | stacking with the right box held | knocked on release (all states) | missed grasp | wrong box |
|---|---|---|---|---|---|
| B80 (old wrist camera) | 354/460 | 309/344 (90 %) | 11 | 27 | 89 |
| B80 (calibrated camera, no retrain) | 323/460 | 276/324 (85 %) | 28 | 37 | 99 |
| wcal30 | 340/460 | 304/344 (88 %) | 16 | 36 | 80 |
| **wcal60** | **372/460** | **317/344 (92 %)** | **5** | 36 | 80 |

- **wcal60 is the best soft-servo policy so far** (372/460; stiff-servo v4 reference 416). Knocks on release fell from 11 to 5.
- B80 loses 31 states on the calibrated scene, so the wrist view matters to it, as the real trials suggested.
- **Section 1 (wcal60):** large-on-small 93.6 % and crooked 4+ no3 84.3 % / no35 81.8 % confirmed; height 0–3 no3 93.0 %, no35 93.9 % met. The small-box (no10) buckets stay low (60 % / 54 %) through wrong-box picks (41 of 109 small-box states), as before (deferred, user).
- **wcal60 = policy for the next real trial**, copied to `data/policies/act_wcal_060000`.
- **Sim twin of the 28 Oct trial 1 scene** (`phase8_sim_twin.py ... --policy data/policies/act_wcal_060000 --runs 4`, Mac, MPS): 4/4 stacked, release tip ~2 cm above the stack top (real B80 released 4.3 cm up there). Loads and runs on the Mac.

**Next:** stage 1 real trials with wcal60 (needs the user's OK).

## 13. Real trial with wcal60, 2026-10-28 17:19 (`data/real_trials/20261028_171710`)

(A first attempt, `20261028_171409`, stopped before motion: the arm's USB serial port dropped out briefly, `termios.error 6 Device not configured`; reconnected by itself.)

**Trial 1, no35 onto no35 (height 1): fail, timeout after 25 s, stack standing.** Right box picked and carried. Over the stack the arm hovered with the box held: the jaw tips came down to 3.6 cm (stack top face 1.6 cm), 3–4 cm short of the stack centre toward the arm, rose to ~5 cm, came back, three times, and never opened (gripper command stayed below the held reading).
- Not the safety limits: the command and the reading agree to ~1 unit while hovering.
- **The joint mapping and the wrist camera agree with reality:** the sim wrist camera rendered at the real readings shows the stack box where the real wrist frame shows it.
- **Sim twin of this scene** (`phase8_sim_twin.py`, 2 runs): both released at tip height 2.0–2.2 cm and stacked, one of them at the same xy as the real hover (18.9, −9.5 cm). The difference is the height the policy goes down to, and that it doesn't open.
- **The grasp differs:** gripper reading while holding real 15.2 vs sim 17.8–20.9 (the real jaws squeeze the cardboard ~3–5 units further), and the held box looks larger and further right in the real wrist view than in sim.

**Open:** one trial only. More stage-1 trials will show whether the hover repeats; if it does, the next suspect is the held-box appearance / gripper reading (the cardboard squeeze) rather than the camera.

**Three more stage-1 trials (wcal60), 17:22–17:29** (`20261028_172237`, `_172625`, `_172915`):

| stack > pick | result | lowest tip over the stack | gripper while holding | release point |
|---|---|---|---|---|
| no35 > no35 | fail, hovered, timeout 25 s, went home still holding the box | 3.4 cm (top face 1.6) | 15.5 | none |
| no3 > no3 | **success** (box on top, overhanging toward the camera) | 4.0 cm (top 1.7) | 17.5 | 4.0 cm up, 4.0 cm short toward the arm |
| no35 no35 > no35 | **success** (top box offset) | 4.2 cm (top 3.2) | 15.4 | 4.2 cm up, 2.7 cm short |

- **wcal60 so far: 2/4 real stage-1 trials.** It releases now (B80 also did), but both no35-on-one-no35 trials hovered without releasing.
- **Same pattern in all four:** the policy stops 2–3 cm short of the stack centre toward the arm, and stays 1–2.5 cm higher than in sim (sim twin: tip 2.0–2.2 cm at release).

**Sim test: is the gripper reading the cause?** (`phase8_sim_twin.py ... 20261028_172237 1 --grip-offset`; the offset is added to the gripper reading the policy sees once it holds the box, ~300 of 440 ticks; `skill/runner.py run_place_skill(state_fn=...)`, sim only.)

| offset | runs | released at tip height | stacked |
|---|---|---|---|
| 0 | 3 | 2.0–2.2 cm | 3/3 |
| −3 | 3 | 2.0–2.2 cm | 3/3 |
| −5 | 3 | 2.1 cm | 3/3 |
| −8 | 1 | 2.1 cm | 1/1 |

**Ruled out:** the policy barely uses the gripper reading here; even −8 units changes the release point by < 1 mm. The hover comes from what it sees: the real held box in the wrist view (sits differently in the jaws) or the real front view.

**Image-swap test: which camera?** (`scripts/phase8_image_swap.py data/real_trials/20261028_172237 1 --regions`): at real ticks with the tips over the stack, the policy gets the real reading plus different images, and its action chunk (100 steps, no ensembling) is read. Sim images are rendered at the same reading with the held box in the jaws. "Opens" = the chunk opens the gripper past the held reading + 3.

| front image | wrist image | chunks that open | lowest tip in 1 s |
|---|---|---|---|
| real | real (what it saw) | 0/18 | 4.3 cm |
| real | sim | 0/18 | 4.8 cm |
| sim | real | 16/18 | 3.5 cm |
| sim | sim | 17/18 | 3.6 cm |

**The front camera decides; the wrist image hardly matters.** Pasting sim pixels into the real front image one region at a time (9 ticks; regions from a sim segmentation render; `notes/img/phase8_swap_regions_*.png`):

| real front with ... | opens |
|---|---|
| nothing changed | 0/9 |
| sim background (wall, banner, room) | **8/9** |
| sim arm | 6/9 |
| sim table and mats | 3/9 |
| sim boxes | 0/9 |
| banner area painted plain black (logo covered; `notes/img/phase8_swap_banner_*.png`) | 6/9 |
| banner area painted plain dark blue | 7/9 |
| background blurred | 7/9 |

- **Main cause: the background behind the arm**, above all the dark banner with the gold "BotMind" logo. Training only saw a plain wall (randomized colour). Covering the logo alone brings most releases back. The arm's look (red/white servo cables, blue parts) is a second, smaller factor.
- This matches the earlier lesson on this arm (CLAUDE.md §2): a cluttered background broke a working policy; a plain one fixed it without retraining.

**Banner covered (user; a crumpled black plastic bag over the banner, not matte), 18:06–18:11** (`20261028_180637`, `_180923`, `_181144`): no35 > no35 fail twice (hovered, timeout), no3 > no3 success. **The cover changed nothing:** over all 7 wcal60 trials, no35 onto one no35 is 0/4 (2 with the logo, 2 covered), no3 onto no3 2/2, no35 onto two no35 1/1. The image-swap result (sim wall pasted in → chunks open) does not carry over to the real arm; the background is not the main cause. (`_180923`: perception put the stack top 7 cm off, (13.9, −5.9) cm; only the logged top/analysis used it, not the policy.)

**The failure is specific to no35 onto a single no35:**
- Not the held no35 in general: it released onto two no35 at the same gripper reading (15.4).
- Not the height reached: the tips went as low as in the no3 successes (3.3–3.8 cm vs 3.9–4.0).
- Not the gripper reading (sim test above).
- In every real trial the tips hover 2.5–3.7 cm short of the stack top toward the arm (sim twin: beside the top in y, not short in x). With no3 (7 cm long) the held box still covers the stack box in the wrist view and the policy opens (the box lands overhanging). With no35 (6 cm) the stack box stays visible beyond the held box (`notes/img` hover frames) and the policy keeps adjusting without moving further forward.
- Working explanation: the common problem is the ~3 cm shortfall toward the arm; the box size decides whether the policy still accepts the position.

**Check: is the ~3 cm shortfall real or a reading error?** (`notes/img/phase8_fk_check_180637_181144.png`: the twin arm outline at the real joint reading, drawn with the session's refit front camera on the real hover frame.)
- **The joint readings are right:** the outline sits on the real arm and jaws in both trials.
- **The "3 cm short" was measured at the jaw tip, not the box.** Taking the held box centre (with the sim grasp's box-in-jaw pose) gives 1.3–2.6 cm from the stack top, the same in successes (no3 −1.3/−0.7, two-no35 −1.8/−1.2 cm) and failures (no35 −1.6/−1.2, −2.2/−2.6 cm). **So the xy position does not explain the no35 failures; the explanation above is withdrawn.** The sim twin of `_180637` also released at (18.8, −9.9) cm, about where the real arm hovers, and stacked.
- What remains: on the real arm the policy opens over no3 (at a 3.9–4.0 cm tip height) but not over a single no35 (it goes to 3.3–3.8 cm and holds), while in sim it releases at 2.0–2.2 cm in both. The offline chunk test reproduces this (real images: no35 0/12 chunks open, no3 3/4), so it can be used to find the image cue.

**Faithful offline replay** (`phase8_image_swap.py <session> 1 --ensemble --variants ...`): the whole trial is replayed through `select_action` (temporal ensembling, as on the arm) at the real joint readings, with the images changed only while the tips are over the stack. **Check: with the real images it reproduces the logged commands** (gripper command median |diff| 0.04 / 0.00 units). The earlier single-chunk "opens" counts were not reliable (the covered no35 trial "opened" in 5/14 chunks but never on the arm).

No35 onto one no35, banner covered with the bag (`20261028_180637`); "lowest" = lowest commanded tip height over the stack:

| images while over the stack | median / lowest commanded tip | opens? |
|---|---|---|
| real front + real wrist (what happened) | 5.2 / 3.7 cm | never |
| sim front + real wrist | 3.8 / 2.8 cm | never* |
| real front + sim wrist | 5.6 / 4.2 cm | never |
| sim front + sim wrist | 4.0 / 2.9 cm | at 23.7 s* |
| real front, sim room pasted in (wall, bag/banner, panel) | 3.8 / 2.8 cm | never* |
| ... only above the banner's bottom row | 4.1 / 2.9 cm | 12.4 s |
| ... only below it (white panel, room) | 4.1 / 3.0 cm | never* |
| ... banner area painted plain dark blue | 4.2 / 2.9 cm | 12.5 s |
| real front, sim arm pasted in | 4.4 / 3.1 cm | 12.5 s |
| real front, sim table | 4.6 / 3.2 cm | never |
| real front, sim boxes / only the held box | 5.7 / 3.9, 5.4 / 3.7 cm | never |

(*open-loop: the arm in the replay stays at the real heights, ~1 cm above where these images would have sent it, so "never" here does not mean it would not open in closed loop.) No3 onto no3 (`_181144`): ~4.2–4.7 cm lowest and opens in every variant.

- **The front image sets how far the policy descends; the release follows the height.** With the real front image (bag cover) it stops ~1 cm higher than with sim's. The room behind the arm accounts for all of it: the bag/banner area and the white panel below it each give most of the effect; painting the banner area plain dark blue alone gives the sim height. The wrist image, the boxes and the gripper reading don't.
- **The bag cover was not a fair test of "plain background":** crumpled, glossy, with bright highlights; the sim wall is flat and matte.
- No3 releases higher anyway (it is longer and covers the stack in the wrist view), so it doesn't need the last centimetre; no35 does.

**Real trials with the software paint (`phase8_trial.py --plain-wall`, `skill/plain_wall.py`), 19:39** (`data/real_trials/20261028_193906`, bag still up; preview `ref_plain_wall.png`):

| stack > pick | result | release tip height |
|---|---|---|
| no35 > no35 | **success** | 2.8 cm (offline replay predicted 2.9) |
| no35 > no35 | **success** | 2.8 cm |
| no3 > no3 | **success** (perception called a collapse: "new box moved 24 mm while the arm left"; false call) | 4.0 cm |
| no3 no35 > no10 | **success** | 4.0 cm |

- **4/4; no35 onto a single no35 0/4 → 2/2.** The arm now goes down to the sim release height and opens, as the replay predicted. Confirms the cause: the room behind the arm in the front image (bag or banner) made the policy stop ~1 cm high.
- **Open: one false collapse call** by the perception check (trial 3) — to look at before games (false calls ruin the game).
- Painting is a workaround. The lasting fix is training on varied backgrounds (pictures, text, textures behind the arm) so the policy ignores them.

## 14. False collapse call on a high release, 2026-10-28

**Trial `20261028_193906` t3 (no3 > no3, success) was called "collapse: new box moved 24 mm while the arm left".** That cue tracks the new box from the release frame, through the arm leaving, to the end of the settle window (`perception/stack.py features`, `arm_frames`). The policy released the no3 with the tips 4.0 cm up, 2.3 cm above the stack top. In the release frame the box is still in the jaws, and its fall onto the stack (~0.1 s) was counted as movement. Its "changed" share was 0.46 against the 0.45 threshold. In sim the release is ~3 mm above the top, so the sim-tuned thresholds never saw a fall like this.

**Fix:** `landed_tick()` (perception/stack.py): start tracking at the last tick up to 0.5 s after the release with the tips still within 5 mm of the release point (the box has landed, the arm hasn't moved). Used by `scripts/phase8_trial.py` and `game/controller.py` (`scripts/sim_game.py` keeps its own release frame; sim releases are low). Offline on t3: 8–24 mm → 0 mm; t1, t2, t4 unchanged.

**Caveat, a missed failure:** B80 trial `20261028_105812` t1 (real failure: the base box was pushed forward and the new box ended up beside it) was called **standing** live. Offline, the old cue (from the release frame) flags it ("moved 16 mm") and the fix doesn't. But that offline re-run used the after frame in place of the unsaved settle window, so it doesn't match what ran live. The collapse check on high real releases is not trustworthy in either direction yet. **Now logged per trial:** the settle window (`tNNN_settle.mp4`) and the collapse features (`perception_features` in trials.jsonl), so it can be checked and tuned on real data. In games, the "is the new box on the stack?" question (Phase 9) still backs it up when the stack top doesn't rise.

## 15. Stage 1 batch with wcal60 + `--plain-wall`, 2026-10-28 20:00 (`data/real_trials/20261028_195923`)

| # | asked | robot picked | result | release (tips above the top face) |
|---|---|---|---|---|
| 1, 2 | no35 > no35 | no35 | success, success | +1.3, +1.2 cm |
| 3, 4 | no35 no35 > no35 | no35 | success, success | +0.7, +0.8 cm |
| 5 | no35 ×3 > no35 | no35 | success | +0.6 cm |
| 6 | no3 > no3 | **no35 (wrong box)** | box placed on top | +1.2 cm |
| 7 | no3 > no3 | no3 | success | +2.3 cm |
| 8, 9 | no3 no3 > no3 | no3 | success, success | +2.3, +2.8 cm |
| 10 | no35 > no35 | **no3 (wrong box)** | fail: held it over the stack, timeout | none |

**With the right box picked: 8/8 this batch; 12/12 over both `--plain-wall` sessions** (no35 onto 1/2/3 no35: 4/4, 2/2, 1/1; no3 onto 1/2 no3: 3/3, 2/2; no3 no35 > no10 1/1). **Wrong box picked: 2/10** (deferred earlier by the user); one of them placed anyway, the other (no3 onto a single no35) hovered.
- **No3 is always released high** (+2.3 to +2.8 cm; no35 +0.6 to +1.3 cm). It lands fine so far, but this is the setup that gave the false collapse call (§14) and is a risk on crooked/taller stacks.
- **Perception agreed with the user's call 10/10** (with the `landed_tick` fix; trial 10 was called "not lying on the stack", the box was never released). Settle videos + features are logged for all 10.

## 16. Stage 2 (mixed pairings) with wcal60 + `--plain-wall`, 2026-10-28 20:17–20:40 (`20261028_201746` t1–7, `20261028_203311` t1–5; the first session stopped on a wrist-camera error: frames came in at 640×360 instead of 640×480)

| asked (stack > pick) | robot | result |
|---|---|---|
| no3 > no10 (×2) | picked a no3 instead (both) | fail ×2 (wrong box; placed, high) |
| no35 > no10 | picked a no35 instead | fail (wrong box) |
| no3 no3 > no10 | missed the grasp | fail |
| **no3 > no35 (×2)** | right box | **success ×2** |
| no10 > no3 (×2) | missed the no3, took a no35 and held it; missed grasp | fail ×2 |
| no35 > no3 (×2) | wrong box / no pick; picked a no35 instead | fail ×2 |
| no10 > no35 | missed the grasp | fail |
| **no3 no10 > no3** | right box | **success** (perception: false collapse call) |

- **3/12. Every failure is in the pick: 6 wrong box, 3 missed grasps.** Placement with the right box held: 3/3. Placement went fine even with a wrong box (released 2.1–2.2 cm high).
- **The small box (no10) was never picked (0/4):** 3× the robot took the nearer large/medium box, 1× it missed. This matches sim (wcal60: wrong box in 41 of 109 small-box states).
- **No3 picks failed when the planner had set a box aside** (no10 > no3, no35 > no3: 0/4), unlike stage 1 (no3 from slot 2 with the full supply: 5/6).
- **Perception:** one more false collapse call on a high release (no3 placed 2.5 cm above a no10: "changed below/beside the new box, stack top 34 px lower"). Settle videos logged.

## 17. Sim twin of the 22 real trials: which box does wcal60 pick in sim? (botmind `scripts\b4_stage2_twin.ps1`, 4 runs per trial, fitted servos; logs `logs\twin_<session>_t<n>.log`)

| case | real | sim |
|---|---|---|
| **small box (no10, slot 8) wanted:** no3 > no10 ×2, no35 > no10 | wrong box 3/3 | **wrong box 12/12** |
| no3 no3 > no10 | missed grasp | right box 2/4 |
| **no3 from slot 2 onto a no10 / no35:** no10 > no3 ×2, no35 > no3 ×2 | **right box 0/4** (wrong box ×2, no pick ×2) | **right box 16/16, stacked 16/16** |
| no3 from slot 2 onto no3 (stage 1) | right box 3/4 | right box 16/16 |
| no35 from slot 6 (stage 1 and no3 > no35) | right box 7/8 | right box 22/28 |
| no10 > no35 (slot 6) | missed grasp | right box 2/4 |
| no3 no10 > no3 (slot 1) | success | right box 2/4 |

**Two different problems:**
1. **Choosing the small box fails in sim too** (0/12). The policy never learned to prefer the no10 when it is nearest the stack. That needs training data.
2. **Picking the no3 from slot 2 works in sim (32/32) but fails on the real arm (3/8)**, mostly in stage 2 (0/4). That is a sim-to-real gap at the pick, like the background was for the release; to be found with the replay test (`phase8_image_swap.py --ensemble` on the pick phase).

**Why the real no3 picks from slot 2 fail (`scripts/phase8_pick_replay.py`):**
- **The policy aims for the right box.** In single-chunk predictions from home to the grasp attempt, every image pair (real, sim, mixed) aims at slot 2 until ~6 s. Only after the failed grasp does it switch to other boxes (a retry), so the logged "wrong box" was a miss plus a retry.
- **The gripper doesn't open wide enough.** Before the grasp the real gripper opened to only 22.8–24.4 units in all 4 failures (command = reading). In the stage-1 successes from the same slot it widened to 33.9–35.1 just before the grasp. The jaws landed on the no3 (tips stopped 2.5–3.0 cm up instead of 0.6), closed on nothing, and the policy retried elsewhere. The xy position was the same (±3 mm).
- **Sim twin of the same scene opens to 27.4–28.3** and the no3 fits (tips 0.4 cm, picked 16/16). Stage-1 scenes: 36.6 in sim.
- **Not the images:** replaying the approach with ensembling (`--ensemble`; RR matches the log within 0.03–0.25 units), sim images give the same opening (24–25) at the real joint states.
- **So:** in these stage-2 scenes the policy opens only just wide enough for a no3 in sim (~27), and on the real arm's slightly different trajectory it opens ~3–4 units less: too narrow. The widening it does in stage 1 near the box doesn't happen.

### 18. Gripper-opening floor (runtime, no retraining), 29 Oct

`skill/grip_floor.py` (`GripFloor`, runner hook `act_fn`, `phase8_trial.py --grip-floor N`): once the policy's gripper command first exceeds 15, command at least N until the policy starts closing (3 units below its peak); then pass-through (retries and release untouched).

Jaw gap vs gripper reading (sim, `scripts/analysis/phase1_gripper_fit.py` geometry): 24 → 44.6 mm at the tips / 41.8 mm 1 cm back (no3 is 40 mm: ~1 mm per side); 28 → ~52 mm; 30 → 55 mm; 33 → 60 mm; 35 → 64 mm. Neighbours beside a template no3 are ~11 mm away on each side (~62 mm between them).

Sim twin of the 22 stage-1/2 trials × 4 runs (`scripts/b4_gripfloor_twin.ps1`, summary `scripts/phase8_gripfloor_summary.py`), wcal60:

| condition | no3 slot 2: right box | no3 slot 2: neighbour moved > 5 mm | no35 slot 6: right / bumped | no10 slot 8: right |
|---|---|---|---|---|
| none | 32/32 | 4 (max 10 mm) | 27/36 / 1 | 2/16 |
| cap 24 (real-like) | 32/32 | 0 | 27/36 / 1 | 2/16 |
| floor 28 | 32/32 | 9 (max 14 mm) | 27/36 / 6 | 3/16 |
| floor 30 | 30/32 | 16 (max 20 mm) | 27/36 / 9 | 2/16 |
| floor 33 | 21/32 | 15 (max 27 mm) | 27/36 / 12 | 0/16 |

Reading: the sim grasps even at 24 (its aim is exact), so the sim cannot show the benefit, only the cost: wider floors push neighbours and, at 33, change the pick. **Floor 28** keeps every pick and adds small nudges (≤ 14 mm). Next: real stage-2 trials with `--plain-wall --grip-floor 28`. Lasting fix: the expert's pre-grasp opening in the next data run sized to ~6 mm per side of the box, not more.

**Forced release (29 Oct, user request):** `skill/runner.py release_if_holding()`, called by `phase8_trial.py` when the skill ends without a release (timeout or guard): if the gripper reading is > 8 and ≥ 5 units above the policy's last gripper command (jaws stopped on a box, the `perception/drop.py` rule), the gripper opens 12 units where the arm is, waits 1 s, then the scripted home with lift. Logged as `forced_release` (the reading it opened from). Sim check: skill cut at 8 s while carrying a no35 → opened from 17.6, box left the jaws, arm home; after a normal release nothing happens.

**Real stage 2 with `--grip-floor 28`, 29 Oct 11:17** (`data/real_trials/20261029_111738`): t1 `no10 > no3`: opened to 27.7 (floor), took the **right no3 from slot 2** and lifted it, but grasped near one end (held reading 16.9 vs ~24 for a centred no3; the box hung tilted, `notes/img/phase8_gf28_20261029_111738_t001.png`). Over the stack the policy opened to 21.5 (≈ held + 5, as it learned) and the box stayed wedged; the runner counted the opening as the release, so the forced release didn't fire. Fix: `open_before_home()` (replaces `release_if_holding`) also opens to ≥ 32 (OPEN_MIN, ~58 mm) after a seen release before the scripted home. t2 crashed after the skill: wrist camera (OpenCV index 1) stopped delivering frames (`latest frame is too old: 4021 ms`).

**Real stage 2 with `--grip-floor 28` (29 Oct, sessions 111738, 114535, 121913):**

| trials | pick (right no3 from slot 2) | placed | failures |
|---|---|---|---|
| `no10 > no3` × 4 | 4/4 | 0/4 | box stayed in the jaws at release (1, before `open_before_home`), knocked on release (3) |
| `no35 > no3` × 4 | 4/4 | 2/4 (one only via the scripted opening after a 25 s timeout) | knocked on release (2) |
| `no3 > no35` × 2 (control) | 2/2 | 2/2 | – |

Picks fixed (no3 from slot 2: 3/8 before → 8/8). Every no3 was held at reading ~17 (as in the stage-1 successes). New failure point = **the release**: over a no10 / no35 the policy opens only to 21–25 (stage-1 no3-onto-no3 successes: 32–36), which barely frees a 40 mm box held at 17; the box pivots out of one jaw (`notes/img/phase8_gf28_release_20261029_114535_t001.png`) or stays wedged until the scripted opening drops it from ~4 cm, off the stack (`..._121913_t003.png`). Two no35-base trials timed out with the gripper closed (policy never opened; cmd 11). Release tips 3.4–3.8 cm from the perceived top centre (box grasped off its centre along its length).
Proposed runtime fix (no retraining): a release floor: once the runner sees the release begin over the stack, command at least 32.
**Release floor** (`run_place_skill(release_floor=…)`, `phase8_trial.py --release-floor 32`): once the runner has seen the release begin over the stack, the gripper command is at least the floor. Sim twin of the 10 trials above × 4 runs (wcal60, `--grip-floor 28`): 40/40 placed with and without `--release-floor 32` (right box 40/40 both). The sim does not reproduce the real narrow release, so this only shows it does no harm; the real trial decides.

**Real stage 2 with `--grip-floor 28 --release-floor 32`, 29 Oct 12:54** (`data/real_trials/20261029_125445`):

| trials | pick | placed | what happened |
|---|---|---|---|
| `no10 > no3` × 4 | right 2/4 (2 missed the no3, then took a no35) | 0/4 | both right picks: release floor fired (opened to 31.6), box still fell off, tilted: released 2.9–3.5 cm off the top centre, 2.3–2.5 cm up (`notes/img/phase8_relfloor_20261029_125445_t001.png`) |
| `no35 > no3` × 4 | 4/4 | 0/4 | **hovered**: over the stack 1.3–2.9 cm up, 3.2–3.8 cm off, up and down until the 25 s timeout; the policy opened only 17 → 20 (+3), under the release detection (+4), so the floor never fired; the box was carried home (`..._t006.png`: the held no3 hides the stack from the wrist camera) |
| `no3 > no35` × 2 | 2/2 | 2/2 | – |

Conclusion: the opening is not the limit any more; the **release position** is (no3 released ~3.5 cm off the top centre, sim twin 1.7 cm) and, over a no35, the policy does not commit to a release. Bug fixed: `open_before_home` after a timeout now opens whenever the gripper reads > 8 (the "blocked" test missed the loosely held no3s, which went home in the jaws).

### 19. Scripted place (29 Oct, sim only so far)

**Where the real no3 really was** (`scripts/phase8_inhand.py`: box pose in the gripper from the template slot at the grasp + FK): held within 3 mm of its centre along its length (the earlier "held near one end" was wrong). At the policy's release the box centre was 11–27 mm from the perceived top centre (mostly about (−15, −13) mm) and its bottom 3–22 mm above the face. Sim twin of the same trials: 3–15 mm off, 0–11 mm up.

**`skill/scripted_place.py`** (`phase8_trial.py --scripted-place`; runner `handover_lift`): the policy picks; once it has lifted the box 4 cm the script takes over:
1. Box in the gripper: along the length and height from the template slot at the last tick before the lift; across the jaws = fixed-jaw inner face + half the jaw gap at the held reading (the closing jaw pushes the box ~8 mm off its slot).
2. Target: box centre over the perceived top centre, yaw = top box yaw (`fit_box` on the front image; the nearer of the two flat directions), lowest box corner 3 mm above the face.
3. Motion: up, across at hover + 3 cm, down to 5 cm above; reading-based sag correction (command += plan − reading, up to 3×); straight down, slow for the last 2 cm, **stopping early if the gripper reading drops 1.5 units** (the box met the stack); loosen 4 units, lift 8 mm with the jaws loose, open to 32, straight up 5 cm. Every command ±10 per tick and through the safety envelope.

Sim twin of the 20 real stage-2 trials × 4 runs (`scripts/phase8_twin_scripted.py`, `scripts/b4_twin_scripted.ps1`):

| version | placed (of handovers) | landed offset (mean / max) | stack moved > 2 mm |
|---|---|---|---|
| v1 (box from the slot only) | 70/78 | – | 8 |
| v2 (+ jaw-push across the jaws) | 69/78 | 2.8 / 12.9 mm | 10 |
| **v3 (+ stop on touch, loosen-lift-open)** | **78/78** | **2.9 / 7.2 mm** | **0** |

(2 runs of 80: the policy never picked, timeout.) The v1/v2 failures: a box tilted ~6° in the grasp touched the stack early, was pushed out of the jaws, and the moving jaw closed over its top and dragged box and stack ~3 cm when opening.
Not yet known: whether the front camera's top centre is accurate enough on the real arm (the real policy releases were biased about (−15, −13) mm; part of that may be perception). The first real trials will show it.

**First real scripted-place trials, 29 Oct 15:29** (`data/real_trials/20261029_152918`): `no10 > no3` **2/2 success** (policy pick from slot 2, scripted place: release 0.4/−1.7 and −0.6/−0.3 mm from the top centre by FK, 2.5–3.0 mm above the face; perception "standing"). Before: 0/8 with the policy's release. t3 crashed after its place: the wrist camera stalled (`latest frame is too old`) while `go_home` read the joints through `get_observation()` (which also reads both cameras).
Fixes: `RealArm.joints()` reads the motor bus only and `RealArm.front()` only the front camera (homing, the scripted place and the settle video no longer depend on the wrist camera); the scripted place keeps every command within ±10 of the reading (wrist_roll lagged up to 21 behind the command and LeRobot clamped it, 24 ticks) and ramps at ≤ 5 per tick; a camera failure while recording the place repeats the last frame instead of stopping. Sim re-check 4/4.

**Real scripted-place trials, 29 Oct 15:37** (`data/real_trials/20261029_153706`): 3/8. `no10 > no3` 0/2 (misplaced), `no35 > no3` 2/4 (+1 "barely standing", 1 misplaced), `no3 > no35` 1/2 (1 knocked on release). Every pick was right; by FK every release was within 2 mm of the target, 1.7–2.8 mm above the face.
Where the boxes really landed (`scripts/phase8_landed.py`: box fits to the before/after front images): relative to the target (the perceived top centre) every placed box, in both sessions (10 trials), landed **≈ (+0, +14) mm** (toward the arm's left), spread ±7 mm. So there is a consistent execution bias between FK and the real arm/camera at the stack (the box-in-gripper estimate is FK at the supply, the release FK at the stack).
Second error in this session: the stack-top estimate. The camera did not move (phase correlation < 0.6 px) but the light changed (refs 3 % darker, large box shadows): the change mask included the shadows and the base-box fit (IoU 0.33–0.60, vs 0.8 in the 15:29 session) and `top_xy` sat on the far side of the real box (`notes/img/phase8_base_estimates_zoom.png`).
Next: correct the execution bias (target − (0, 14) mm, measured on 10 trials) and make the stack-top estimate shadow-proof (or steady lighting), then re-run.
**Correction to the 15:37 analysis:** that session's reference image had a 2-box stack on the stack area (left from the previous session; `notes/img/phase8_refs_152918_153706.png`), so every stack-top estimate in it compared against a non-empty reference: the targets were wrong by 1–2 cm, not because of shadows. The landing offset (+14 mm in y vs the commanded target) is measured from after-vs-before and is unaffected.
Fixes in `phase8_trial.py`: the reference is retaken until all 8 supply slots read full (override with y); the scripted-place target is the flat-box fit to the shadow-filtered stack silhouette (`change(..., shadows=True)`, biggest blob; clean session 15:29: IoU 0.86 / 0.74, outline on the box, `notes/img/phase8_base_fit_shadowfree.png`), falling back to `top_xy` below IoU 0.6; `--place-bias-mm` (default 0 14) removes the measured landing offset. Logged per trial: `top_fit`, `place_bias_mm`.

**Real stage 2, scripted place with the fixes, 29 Oct 15:58** (`data/real_trials/20261029_155805`): **10/10**. `no10 > no3` 4/4, `no35 > no3` 4/4, `no3 > no35` 2/2. Top box fit IoU 0.90–0.96 (clean reference). Landed (front-image fits, `scripts/phase8_landed.py`): new box centre 1–7 mm from the base box centre (mean ≈ (−2, −3) mm), yaw within 6°. Release by FK 1.4–5.0 mm above the face. Perception: 9/10 agreed; t6 was a **false collapse call** ("changed below/beside the new box", new-box fit IoU 0.41) on a successful placement: to tune in the collapse check.
Compared with the policy's own release on the same trials: 0/8 (no3 onto no10/no35). Stage 2 for the no3 meets the ≥ 80 % target; the no10 pick (small-on-large) still fails at the box choice and needs new data.

### 20. Stage 3 preparation (29 Oct)

**Top box fit on multi-level stacks:** the whole-silhouette fit put the top box between levels (IoU 0.6–0.78, still "trusted"). `fit_box(..., top_only=True)` scores only the mask rows down to the candidate outline's lowest row (lower levels ignored): recorded 2–3 box stacks now fit the top box (IoU 0.77–0.87, `notes/img/phase8_base_fit_shadowfree.png`); one badly rotated top box 0.72 → `FIT_MIN_IOU` raised to 0.75 (falls back to `top_xy`). Single boxes unchanged (0.91–0.98).
**Sim check** (`scripts/phase8_stage3_sim.py`, `scripts/b4_stage3_sim.ps1`; stacks planned as in `phase8_trial.py`, each level ±6 mm / ±8°, from home): 8 setups × 4 runs: placed **31/32** (1 policy pick timeout), landed 1–4 mm from the top centre, stack moved ≤ 1 mm, **no arm–backdrop contact**. Wrong box picked: `no35 no10 no35 > no3` 4/4 (policy takes slot 7, a no35), `no3 no35 no10 > no3` 1/4 (the scripted place still put the wrong box on the stack). The first is left out of the real list.

**Real stage 3, scripted place, 29 Oct 16:30–16:50** (sessions 163025, 164135, 164839; the wrist camera stalled twice, see below): **4/9**. 3-box stacks 4/6 (`no3 no35 no35 > no3` 1/2, `no35 no35 no3 > no35` 1/1, `no3 no3 no35 > no35` 1/1, `no3 no35 no10 > no3` 1/2), 4-box stacks 0/3. Picks right 8/9.
Frames (`notes/img/phase8_stage3_fail_a.png`, `_b.png`): at release the held box was visibly 2–3 cm off the top box on the 3-box failures though FK put it within 2 mm of the target; on 4-box stacks one box slid off after release, one crooked stack swayed and fell at release, one landed off the top. Causes:
1. The top-box fit on taller real stacks is weak (IoU 0.48–0.83; three fell back to `top_xy`, which is 7–10 mm off even on single boxes).
2. The landing correction (0, +14) mm was measured at height 1; the FK/real error depends on the arm pose, so it does not transfer to 3–4 box heights.
Proposed: measure the stack top with the **wrist camera from a look pose over the stack before the pick** (calibrated camera, close range, and the arm in nearly the release pose, so the FK error mostly cancels), instead of the front-camera fit + a fixed correction.

**Camera stalls:** `RealArm` now reads the joints from the bus only and each camera through `camera()`, which reconnects a stalled camera (disconnect, 1.5 s, connect; 4 attempts) and sets `recovered` so the policy restarts from fresh frames; the arm holds its last command meanwhile. `phase8_trial.py`: both cameras are checked (and reconnected) before every trial; if a camera cannot be reconnected (`CameraLost`) the arm holds, the user is asked to re-plug the cable, and the trial continues home instead of the session crashing. Unit-tested with a fake camera; the real reconnect is not tested yet.

### 21. Wrist-camera look at the stack (29 Oct, sim only so far)

`skill/stack_look.py` + `phase8_trial.py --look` (with `--scripted-place`): before the pick the arm goes home → straight up 10 cm → a look pose with the wrist camera 12 cm above the top face, looking down (tilt ≤ 3° on most poses), the top centre at image (320, 160), gripper closed; it takes a wrist frame, finds the top box, and goes back the same way to home; the policy then picks as usual and the scripted place aims at the wrist result (no landing correction).
- Look pose: IK on the calibrated wrist camera (`cameras.yaml`, `mj_camlight` for the camera pose), with a soft ≥ 30 mm clearance between the wrist/gripper and the backdrop (an open gripper swung the moving jaw 22 mm into it). Route clearance (sim, real box bodies as 1–5 box stacks, guess ±1.5 cm): ≥ 29 mm from the stack, ≥ 27 mm from the backdrop.
- Detector: rectangle of the top box's size at the known height, scored by the image gradient across its four edges (visible samples only; arm pixels masked from a segmentation render of the twin), times the share of its inside that is not yellow mat / dark backdrop / arm; grid ±3 cm, 5°, then Nelder-Mead. Trusted if score ≥ 20 and within 3 cm of the front estimate, else the front target is kept.
- Sim (`scripts/phase8_look_sim.py`, 40 random crooked 1–4 box stacks, guess off by up to 1.5 cm, the game's wrist randomization incl. f ±5 %, blur): position error median 4.3 mm, 90 % 8.3 mm, 3/40 > 12 mm (max 31 mm, a pose at the wrist-flex limit); yaw median 1.5°. Saved: `notes/img/phase8_look_sim.png`, per trial `tNNN_look.png` (magenta = front guess, green = wrist detection).
Not yet tested on the real arm or real wrist frames.

### 22. Camera stalls: root cause and a crash-proof camera layer (29 Oct)

**Root cause (hardware):** the macOS kernel log shows the Orbbec Gemini 2 wrist camera dropping off the USB bus and re-enumerating ("AppleUSBHostPort::terminateDevice: destroying 0x2bc5/0670 (Orbbec)", then "enumerated", with dozens of "pipe stalled" errors) at 16:47–16:48, 17:20 (the failure at trial 3 of session 171424) and 17:23. Its port offers 900 mA (it asks for 896); Orbbec lists the Gemini 2 at 5 V ≥ 1.5 A, peaks ≈ 6.5 W. So it browns out and resets. Only power fixes the resets: a powered USB 3 hub, or a USB-C port/cable that negotiates 1.5 A.
**Why it broke the sessions (software):** LeRobot's `OpenCVCamera` kills its read thread for good after 11 failed reads; reopening a device mid-reset runs OpenCV/AVFoundation calls in the main process that can block indefinitely; and after a re-enumeration the camera order changes (the macOS list read 0 = Orbbec, 1 = C922 at 17:30, while the trial opened front = 0, wrist = 1), which could silently swap the views.
**Fix (`skill/cameras.py`, used by `RealArm`):** each camera runs in its own child process, opened by its macOS device name through AVFoundation (pyobjc, added to `~/.venvs/so101_eval`), never by index; frames go through shared memory with a lock-free seqlock; the child sends a heartbeat and the supervisor SIGKILLs and restarts a hung or dead child; the main process never makes a camera driver call, and reads never wait (`CameraStale` at once). The policy loop holds the arm at one pose while a camera is stale and restarts the policy from fresh frames; the trial waits (with status) for cameras before each trial and the look; the scripted place records the last good frames. LeRobot's robot is created with no cameras.
**Proof:** `scripts/test_cameras.py` (no hardware, fake camera source): 9/9: driver call that never returns, child crash, child frozen (SIGSTOP), device absent 5 s, wrong frame size, 30 SIGKILLs mid-write with no torn or swapped frame, the other camera unaffected, longest main-process read 6.6 ms. `scripts/test_camera_stall_sim.py`: a 3 s camera outage during the reach and at the grasp: 90 identical hold commands, arm at rest (≤ 0.14 units), skill resumed, box placed (2/2). On hardware: `scripts/camera_soak.py` (both real cameras, injected SIGKILL/SIGSTOP every 15 s, identity check against the session reference, optional unplug/re-plug of the wrist camera): to be run by the user (this shell has no camera permission; the AVFoundation path opened both devices here but gets no frames without it).
**Hardware soak, 29 Oct 17:38 and 17:45** (`data/camera_soak/20261029_173825`, `_174500`): identity PASS both runs (front frame vs the session reference 0.95–0.96, wrist −0.30/−0.37); 19/19 injected faults recovered in run 1 (SIGKILL 0.6–0.8 s, SIGSTOP 3.6–3.8 s); run 2 with a real wrist unplug/re-plug: the camera came back by itself after the re-plug (the one "not back" was a fault injected while the cable was out). Longest main-loop read 27 / 22 ms: the restart (kill + spawn) ran inside a read. Fixed: restarts run in a watchdog thread; reads are a shared-memory copy (fake-source test: longest 4.5 ms, the bound being Python's 5 ms thread switch interval; limit set to 10 ms, control tick 33 ms). The soak test now logs natural stalls and injects no faults for 45 s around the unplug.
**Soak re-run 17:51** (`data/camera_soak/20261029_175136`): identity, recovery 12/12, non-block (5.8 ms) and coverage PASS, but the log showed a real gap: after the wrist unplug/re-plug at 17:53:38 the same child process got no frames until an injected SIGKILL at 17:54:38 forced a new one. A long-running process does not see the re-enumerated camera (macOS does not refresh its device list there) and its heartbeat kept the watchdog quiet. Fix: a child with no frames for 4 s (`CHILD_EXIT_NO_FRAME_S`) exits and the supervisor starts a fresh one. New test T7 (a child whose device list stays stale after a re-plug): fresh again 4.2 s after the 'unplug' via a new child; `test_cameras.py` 10/10, longest read 1.2 ms. The soak test now checks the re-plug explicitly ("PASS re-plug") and keeps injected faults away for 60 s around it.
**Soak 17:58, final** (`data/camera_soak/20261029_175831`): all PASS. Identity (front = C922 vs the session reference), recovery 11/11 injected faults (SIGKILL 0.6–0.9 s, SIGSTOP 3.6–4.8 s), longest main-loop read 3.5 ms, coverage front 96 % / wrist 90 % including the unplug, and **re-plug: the wrist camera came back by itself 13.2 s after the unplug** (three fresh child processes while it was out, then frames), with nothing restarted by hand. The camera layer is done; the remaining risk is hardware (the Orbbec still resets on 900 mA; each reset now costs a few seconds of holding still instead of a session).

**Real stage 3 with the wrist look, 29 Oct 18:04** (`data/real_trials/20261029_180449`): **4/6** (with 171424 t1–2: 6/8). 4-box stacks 2/3 (before the look 0/3). Failures, both crooked: t2 the look saw the box almost outside the image, was rejected (score 6) and the front fallback (IoU 0.58) missed; t5 the top box was half out of the image, the visible part matched (score 28) and was trusted. The real boxes appeared 80–150 px higher than the aimed row (front estimate + arm error; `notes/img/phase8_look_real_180449.png`).
Changes: look from 15 cm aiming at row 200; a look is good only with the whole top-box outline inside the image (≥ 98 %) and ≥ 75 % clear of the jaws; trusted only when two consecutive good looks from different aims agree (≤ 6 mm, ≤ 8°), up to 6 looks (re-aim at what was seen, or +4 cm higher); if none is trusted the trial asks (front estimate / last good look / skip). Sim (40 crooked 1–4 box stacks, guess off ≤ 3 cm, randomized wrist intrinsics): 28/40 trusted, all within 9.6 mm (median 4.2); without the agreement rule 36/40 trusted but 4 were 10–41 mm off (lock on a lower box of similar size). Route clearance at 15–23 cm looks, aims ±3 cm, look-to-look moves: ≥ 60 mm from 1–5 box stacks, ≥ 24 mm from the backdrop.

**Real stage 3, full re-run with the agreeing-look rule, 29 Oct 18:37** (`data/real_trials/20261029_183746`): **8/8 placed** (3-box straight/crooked 5/5, 4-box straight/crooked 3/3 incl. a no35 onto a small no10 top 4 high), right box picked 8/8. Every look trusted after 2–3 looks (score 35–94, outline 97–100 % in the image, two looks within 6 mm); front estimate 4–24 mm off the look. Release by FK 2.3–4.5 mm above the face.
Stage 3 total with the look: 8/8 here (14/16 incl. the 18:04 session before the look changes). Open: the **collapse check gave 3 false calls in 8 successful placements** (t2 "changed below/beside", t3 "stack top 64 px lower", t7 "stack top 49 px lower"), all on 3–4 box stacks; with the 15:58 t6 that is 4 false calls on real successes. This must be fixed before games (Section 1: no false collapse calls).

### 23. Turn check v2: false collapse calls (29 Oct, in progress)

Replay tool `scripts/phase8_collapse_replay.py` re-runs the check on every recorded real trial (82 with logged features, 3 skipped for a hand in the after image) against labels: OK = the new box lies on top and the stack stands (also a wrong box), NOT = anything else. Labels = the user's call, with overrides checked by eye (`notes/img/phase8_fail_after_*.png`, `phase8_turn_odd.png`, `phase8_turn_overrides.png`; 203311 t5 was a user "success" whose box slid off onto the table on the retract).
**Old cue-based `collapse_check`:** 18/56 false collapse calls, 11/26 missed. Causes: front-camera auto exposure (after images brighter: big "changes" on the mat/table, speckles that broke the stack-top-row cue), a fallen box beside the stack lying inside the new box's allowed column, boxes landing outside the watched area.
**New `perception/stack.py turn_check()`** (before + after the settle window): photometric match of after→before and ref→before (per-channel gain/offset fitted on the static scene: backdrop, wall, banner, floor, table front; `static_region`); change mask over the whole table (`table_region`, supply mat and arm excluded); new box = flat-box fit on the old top at its height, only from changes near the old top (`TURN_NEAR_R`); old top from a top-only fit in `before` (trusted ≥ 0.6, else the trial's top_xy); cues: fit ≥ 0.45, centre ≤ 32 mm from the old top, old top face ≥ 30 % covered, and no other place with > 300 new/vanished edge pixels (edges are insensitive to exposure drift). Thresholds from the labelled real data (margins noted in the code).
**Result on the 82 real trials: 0/56 false collapse calls, 1/26 missed** (201746 t2: a box leaning on the base with one end on the table, fit 0.64, cover 0.6, edges 140).
**Turn check v2, continued (29 Oct evening):**
- Fit bias on tall stacks: the old-top (top-only) fit slid down onto lower levels of 5–7 box stacks; the outline's top row is now tied to the silhouette's top row (`TOP_ROW_W`). In the game the old top is known (`top_known`: the previous turn's new-box position, or the robot's wrist look), the image fit is only a fallback.
- Cover rule (old top face covered) only for single-box stacks (`TURN_COVER_ZMAX`): there a box beside the stack fits nearly as well as one on top; on taller stacks a box beside it is on the table and fails the fit.
- New cue: old-stack change (share of the before-silhouette outside the new box that changed, `TURN_STACK_CHANGE` 0.15): a stack pushed as a whole.
- Edge limit 450 (sim's hard shadows give good placements up to 432; real ≤ 220, real falls ≥ 1044).
- A no-new-box mode (new_type=None) for judging a scene that must not change.
**Results:** real trials (82) 0/56 false calls, 1/26 missed (old: 18, 11); sim set (`scripts/turn_check_sim_eval.py`, 294 placing turns, old top known when the old stack did not move) 5/218 false, 5/76 missed with truth = on top, nothing on the table, old stack ≤ 15 mm (the sim truth that also counts the new box's own slide > 15 mm: 9/197, 24/97, the extra misses are 'nudge' turns where the box stays on top); real clips (`phase7_real_clips.py`) **11/11** (old 10/11), 0 false calls over 51 sliding windows on static stacks.
**Wired in:** `phase8_trial.py` (perception call = turn_check, old call logged next to it; the old top is known when the wrist look was trusted) and `game/controller.py` (JUDGE uses turn_check; the next turn's old top = this turn's new-box position; the "is the new box on the stack?" question only when turn_check passes but is borderline: fit < 0.55 or > 26 mm off). Sim games running (10, seed 14001, compare with Phase 9: 75/75).
Rule note (for the user): the sim truth's "top moved > 15 mm" also counts a new box that slides on the top but stays there; turn_check calls that standing (it judges the end state), as the real labels do.
**Sim games, 10 games seed 14001 (first run with turn_check, `data/games/sim_20261029_232951`):** 53 turns, calls correct 51/53, losses caught 7/7, **2 false loss calls** (Phase 9's cue check + question: 0/66): (1) a no3 squarely on a no35 shaded/hid the base's sides → "old stack moved 29 %"; (2) a rotated, overhanging no3 on top, fit 0.37. Fixes: the old-stack change ignores the whole column under the new box's footprint; a turn where only a borderline fit (0.30–0.45) or offset (32–40 mm) fails and nothing else moved is "unsure" and the game asks the player (as it asked before for borderline passes). After: real trials 0/56 false, 1/26 missed, 2 asked; sim set 3/218, 6/76, 5 asked; clips 11/11. Second sim-game run started.
**Sim games, second run with the fixes** (`data/games/sim_20261030_000937`, 10 games seed 14001): **62/62 turn calls correct, false loss calls 0/53, losses caught 9/9, 10/10 games called right**; asked the player on-stack 5 times (borderline/unsure), box type 14 times (slot check unsure), no watchdogs. (Phase 9 cue check + question: 75/75, 0/66.)
Status of the turn check: real trials 0/56 false, 1/26 missed (a box leaning on the base); sim set 3/218, 6/76; real clips 11/11; sim games 62/62. Next: the controller's robot turn still runs the plain policy; the real-game robot needs the Phase 8 recipe (grip floor 28, plain wall, wrist look, scripted place, camera layer).

### 24. The Phase 8 robot recipe in the game controller (30 Oct)

`game/controller.py` robot turn (`recipe=True`, default): wrist look at the stack (`skill/stack_look.look_at_stack`, moved out of the trial script; trusted when a good look agrees with ANY earlier good look: sim had A,B,A,B alternations), the policy picks with `GripFloor(28)` and the plain-wall front on the real arm (`skill/plain_wall.PlainWall`, built at setup), handover at 4 cm, `ScriptedPlace` onto the look's top (or the known top), `open_before_home`, home. The scripted place now holds still while `paused()` (hand check) and lets the e-stop through. The sim arm has `camera()` / `wait_cameras()` so the look runs in sim (pinhole projection). `play_game.py` default policy = `act_wcal_060000` (was v4).
Sim games with the recipe (10, seed 14001, policy v4 by the old default; `data/games/sim_20261030_*`): looks trusted 27/27, robot placements 23/27 (2 no pick, 1 box on the table at h7), calls 57/58, 1 false loss call: a no3 squarely on a look-alike box at h5 (fit 0.29: the image barely changes) plus a few-mm base shift (edges 523). turn_check: the edge cue ignores the column under the new box's footprint; "unsure" (ask the player) now covers fit 0.20–0.45 / edges 450–1000 / ≤ 40 mm off with the stack unmoved (real failures with such fits are 56–61 mm off). After: real 0/56 false, 1/26 missed, 2 asked; sim set 3/218, 8/76, 14 asked; clips 11/11; that turn is asked. Final 10-game run with act_wcal_060000 running.
**Sim games with soft servos (30 Oct night):** the game's `SimBackend` never applied the fitted soft-servo model (`servo_fit.yaml`), so act_wcal_060000 (trained with it) played on stiff servos: robot 14/22. Now applied by default (`servos="fit"`). That exposed three issues, fixed: (1) the look's wait in sim called `arm.hold()`, which re-commands the sagged reading so a soft arm sinks further (target drifted from image row 101 to −111); the wait now re-sends the last command (`SimArm.last_cmd`); on the real arm the wait is a sleep (not affected); (2) the look pose gets the reading-based sag correction the scripted place has (within ~1 unit, target within ~25 px of the aim); (3) the scripted place's touch stop only counts in the last 12 mm (a soft arm's gripper reading wobbled as it slowed and stopped a place 3 cm high). Also: the first box onto the empty base has no off-centre rule (a 36 mm human placement was called a loss); SimBackend retries a supply layout that would overlap.
**Sim games, soft servos + all fixes** (`data/games/sim_20261030_033329`, 10 games seed 14001, act_wcal_060000, full recipe): looks trusted **31/31**, robot placements **24/31** (h1 10/10, h3 9/10, h5 3–4/8, h7 1/3; failures: knocks of tall stacks, a box on the table, one no pick), turn calls **60/63**: 1 false loss call (h5, "old stack moved 18 %", fit 0.49), 2 misses: one is a sim truth bookkeeping error (the robot placed a different box than the truth tracked: the after image shows it on top), one is a box resting tilted on its end on top (fit 0.75). A "sticks up" cue (new edges above a flat box at the old top) separates it only weakly (24 px vs ≤ 20 on real good placements), so it is kept at 60 px as an ask-trigger only and the tilted box is a known limit (the scripted place releases flat 3 mm above the top, so it should be rare on the real arm).
Final regression with all changes: real trials 0/56 false, 1/26 missed, 2 asked; sim set 3/218, 8/76, 14 asked; real clips 11/11.
