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
