# Phase 6: data v2 and training v2 plan (for review; nothing built yet)

Status 2026-10-22. ACT v1 scores **49 % on the fixed 200 sim states** (temporal ensembling). 92 of the 102 failures are
missed grasps; once a box is held, ~90 % are placed correctly (`notes/phase6.md`). So v2 targets **the grasp**: a
grasp that tolerates the policy's aim error, data that teaches correction, and data that teaches recovery after a miss.

## 0. What the failures tell us (evidence)
| finding | evidence | consequence for v2 |
|---|---|---|
| aim error ~1–2 cm (half a box width) | missed-grasp videos (episodes 2–5): jaws in the gap or straddling two boxes | the policy must learn to **correct** the aim near the box; tolerance alone can't cover 2 cm |
| the fixed jaw is aimed 2 mm from the box face; the moving jaw opens ~6 mm wider than the box | `expert.py`: `p_grip = xin + width/2 + 2 mm`, `PREGRASP = contact + 6.5` units (~0.93 mm/unit) | an error > 2 mm toward the fixed jaw puts that jaw **on top of the box** |
| room beside a box is limited | supply sampler: 10 mm free per side (`game_states.side_free`); jaw tips ~6 mm thick (`notes/phase4_finger_geometry.txt`) | in dense supplies at most **~4 mm per side** of margin fits |
| dense supplies are the worst | 5–6 boxes left: 38 % / 24 % success, 56–73 % missed grasps | over-sample them; most of the "height 0–3" target sits here |
| the descent runs open loop at default inference | 100-step chunks: 42 % → 57 % with temporal ensembling | demos should **slow down and pause** where correction happens |
| no retry after a miss | videos 3–5: carries an empty gripper to the stack | add demos that **start from a missed grasp** |
| 10 % wrong box | 20/200 | measure whether near-ties cause missed grasps (averaging between two boxes) |
| training plateaued at 1 epoch | 40k–100k flat at ~40 % (default inference) | more steps on better data |

## 1. Expert changes (`scripts/expert.py`)
1. **Centred, adaptive grasp.** Aim the jaw gap's centre at the box centre, not the fixed jaw 2 mm from the face.
   Opening = box width + 2 × m, where the per-side margin m = min(8 mm, free space on the tighter side − jaw
   thickness − 1 mm), from the privileged neighbour positions. Dense layout ≈ 3–4 mm/side; open layout up to
   8 mm/side. (The policy sees the neighbours, so this is learnable, not random.) Check: the Phase 4 grasp test
   (100 per type) must stay ≥ 95 % with the centred grasp.
2. **Hover, then slow final descent.** Stop 3 cm above the box top for 0.3 s (open jaws, the wrist camera looking
   straight down). Descend the last 3 cm at ~1.5 cm/s (now ~5 cm/s), pause 0.2 s before closing. This gives temporal
   ensembling and the wrist view time to line up, and makes "close now" a clear visual cue.
3. **Check the grasp, as the policy will.** After lifting 2 cm, compare the gripper reading with the box's
   contact reading (held: no3 ~20.7 / no35 ~17.6 / no10 ~13.4; empty: ~0). It's always "held" in clean demos;
   the retry demos (section 3) use it. The same check goes into the real skill runner (`drop_check`).
4. **Tie-free pick rule (only if the diagnostic shows it matters).** First count, in the v1 200-state eval, the
   missed grasps where the second-nearest box was within 1 cm of the nearest. If that's a real share, rank by
   distance in 2 cm bands and break ties by a fixed visible order (nearest to the arm base).

## 2. Recovery data: DART-style noise injection
- In ~40 % of episodes, the commanded joint targets get smooth random noise (Ornstein–Uhlenbeck, σ ≈ 1.5° per body
  joint ≈ 6–10 mm at the jaw tip, a 0.5 s time constant), during the approach and the descent down to 1 cm above
  the box, then fade to zero before closing (and during the carry, away from the stack).
- The **recorded action label is the clean expert target** (the plan), not the noisy command. So the dataset holds
  states slightly off the expert path, each labelled with "go back to the plan". This is exactly what the
  policy is missing when its aim drifts. (Ref: Laskey et al. 2017, "DART: Noise Injection for Robust Imitation
  Learning".)
- No noise near the stack top, so placements stay gentle (the release is already ~90 % fine).
- Episodes that fail are discarded as now (noise can make the expert miss).

## 3. Retry demos: starting from a missed grasp
- ~15 % of episodes start from a **natural missed grasp**: the expert runs a grasp with its target offset 10–25 mm
  sideways (unrecorded), closes on air or on the box edge, lifts 2–3 cm. **Recording starts there**: gripper empty
  (reading ~0), the box possibly nudged. The recorded expert then opens, rises to the hover pose above the box's
  *current* position, does the centred grasp and places as normal.
- The failed attempt itself is **never recorded**, so the policy never learns to miss. It learns "closed and empty
  → reopen and go back". The cue is visible in `observation.state` (gripper ~0 vs ≥ 13 when holding) and in the
  wrist view.
- The eval gains a per-episode "retries" count; the skill runner gets a retry budget (e.g. 2) before giving up.

## 4. Game-state sampling (`scripts/game_states.py`)
- **Rebalance toward dense supplies:** stack heights 2–3 (5–6 boxes left) get 35 % of states (v1: ~17 %).
  Heights 4–7 keep ~45 %, height 1 ~20 %. CLAUDE.md asked for "more weight on taller stacks"; the failures say the
  dense supply is the harder part. **Your call.**
- Stricter stability check before an episode (`stands()`): rest 2 s, < 0.5 mm motion (removes the rare slow creep).
- Keep v1's hard-case share (0.6) and the per-level type mix.

## 5. Dataset v2
- **Fresh dataset, not mixed with v1**: v1's tight, lopsided grasp would conflict with the centred adaptive one
  (ACT would average the two styles).
- 3,000 ACT episodes: ~45 % clean, ~40 % DART-noised, ~15 % retry starts; the type of each is stored in `meta.json`.
- Same raw format and verified export (`--streaming --queue 600` + video check). The VLA dataset is rebuilt from the
  same v2 episodes later (the v1 VLA export now on the PC becomes obsolete).
- Generation on the Mac: ~6–7 s per episode with 4 workers (retry episodes are longer) → **~6 h**, overnight.
  Export on the PC: ~1.3 h.

## 6. Small A/B pilot before the full run (recommended)
Before spending ~6 h of generation and ~19 h of training:
1. Expert-only checks: grasp test (centred grasp, 100 per type) and `expert.py --n 100` (≥ 95 % expected).
2. A 20-episode pilot: inspect videos and joint plots (CLAUDE.md checkpoint before large generation).
3. **600 v2 episodes** (~1 h) → train **30k steps** (~2 h) → 60-state eval with temporal ensembling. v1 at 30k scored
   12 % (default inference); re-evaluate v1@30k with temporal ensembling for a fair comparison. A clear gain means
   go; no gain means rethink before the big run.

## 7. Training v2
- ACT, the same LeRobot commit, AMP, batch 8 (5.5 GB of 8 GB; batch 16 won't fit), **300k steps ≈ 19 h** on the
  3060 Ti (0.226 s/step) ≈ 2 epochs of ~1.2M frames. Checkpoint every 25k.
- Option (test in the pilot, not assumed): train at **half resolution (320×240)**, an estimated ~3× faster per step,
  so ~6 h for 300k or more epochs. Costs: the policy, `sim_eval` and the real runner all resize the same way; the
  wrist size cue halves (No.3 vs No.35: 41×17 px → ~20×8 px, still separable); front-view detail halves. Decide
  from the pilot.
- Inference: temporal ensembling 0.01 (v1 decision), re-checked on v2.

## 8. Evaluation protocol (unchanged, so results compare)
- Fixed 200 states (seeds 1001–1004), nominal visuals, temporal ensembling, all Section 1 buckets + failure types +
  retries + backdrop flag. Checkpoint tracking on the 60-state set (seed 777).
- **v2 target:** ≥ 75 % overall and missed grasps ≤ 10 %. The Section 1 criteria (≥ 90 % at heights 0–3 for every
  type, ≥ 70 % crooked 4+, ≥ 70 % large-on-small) may need a v3 (DAgger, section 10).

## 9. Time budget (estimates)
| step | time |
|---|---|
| expert changes + DART + retry starts + tests | ~0.5 day |
| A/B pilot (600 episodes, 30k steps, eval) | ~4 h |
| full generation (3,000) | ~6 h (overnight) |
| export + verify | ~1.5 h |
| training 300k | ~19 h (or ~6 h at half resolution) |
| 200-state eval | ~2 h |
| **total** | **~2–2.5 days** |

## 10. Later (v3), if v2 isn't enough
- **DAgger in sim:** run the policy, let the expert label the states it actually visits (the expert must plan from
  any state, a moderate rewrite of `expert.py`), retrain on the union.
- Chunk size 50 instead of 100 (shorter horizon; pairs well with temporal ensembling).
- Diffusion Policy as an alternative learner if ACT plateaus.

## 11. Decisions needed from you
1. Fresh v2 dataset (v1 not reused): **yes / no**.
2. Height rebalance toward dense supplies (heights 2–3 = 35 %), against CLAUDE.md's "more weight on taller": **ok?**
3. Run the A/B pilot first (~4 h): **yes / skip**.
4. Half-resolution training: **test in the pilot / don't**.
5. Optional **game rule** with a large effect: ask the human to keep ≥ 2 cm gaps between supply boxes, giving room
   for an ~8 mm/side margin. Needs a check that 8 boxes still fit the mat (the Phase 5 supply analysis used 10 mm);
   not part of this plan unless you want it.

## 12. Implementation status (2026-10-22)
Built and tested on the Mac (not committed):
- `expert.py` v2 (`kind` = clean / dart / retry; v1 kept as `kind="v1"`):
  - **centred, adaptive grasp**: per-side margin = free space − jaw thickness − 1 mm, clipped to 2–8 mm;
  - **deep grip**: jaw tips planned ~2.5 mm above the mat;
  - hover 3 cm / 0.3 s, last 3 cm at 1.5 cm/s, 0.2 s pause, grasp check after a 2 cm lift;
  - DART noise (OU, σ 2.0°/joint, τ 0.5 s) on the approach until the grip point is 25 mm above the box, and during
    the carry; clean final descents. The recorded action is the clean plan (`Sim.clean_ctrl()`);
  - retry starts from an unrecorded offset miss; the pick is re-chosen after it.
- Findings while building it:
  1. The jaw opening is ~1.0 mm per gripper unit (measured), so the v1 grasp was width + 6.5 mm (2 mm fixed /
     4.5 mm moving side).
  2. Bug fix: jaw-side feasibility must use the v2 grip point (one IK failure, 24° tilt).
  3. The arm sags 3–5 mm at the grasp under position control, so the v1 grip only held the top ~5 mm of a box.
- Grasp variants on 120 v2 states (seed 1): centred 119/120, centred+deep 119/120, lopsided 3/8 mm 120/120 (all
  failures are backdrop touches) → **centred + deep chosen** (most aim tolerance, sag margin).
  DART 37/40, retry 29/40 (9 "not missed" discards; offsets then widened).
- DART noise size at σ 1.5°: gripper offset median 5.5 mm, 90th percentile 9.6 mm → raised to 2.0°.
- `game_states.py`: `HEIGHT_P_V2`, `rest=2.0` (< 0.5 mm), `side_clear_width()`.
- `generate.py --data-version v2` (default): kind mix 45/40/15, v2 states, `kind` and `pairing_final` in meta.
- **botmind**: `.venv-lerobot` (LeRobot ca69a206 `[training]`, torch 2.11.0+cu128, MuJoCo 3.16.0, numpy 2.2.6).
  ACT training **0.119 s/step** (3060 Ti: 0.226) → 300k steps ≈ 10 h. torchcodec can't load (FFmpeg DLLs); LeRobot
  falls back to PyAV decoding (fine at this speed).

Still to do before the pilot: half-resolution option (export + sim_eval resize), sync code to botmind and run the
20-episode inspection there, re-evaluate v1@30k with temporal ensembling for the fair comparison, finalize the
supply layout (pick-order margin) and add it as a supply mode.

## 13. Pilot run (2026-10-22)
- 20-episode v2 inspection on botmind: units OK, |action − state| ≤ 9.5° (DART; under the 10° limit), wrist_flex
  overshoot 0.5° under noise (commanded ≤ 101.9°), episodes 398–456 frames.
- 600 pilot episodes generated on botmind (16 workers, ~21 min); exported to `pilot_full` (640×480) and
  `pilot_half` (320×240): 600 episodes / 252,034 frames each, video checks OK (also re-checked after copying to the PC).
- Training (30k steps, AMP, batch 8): full-res on botmind 0.119 s/step; half-res on the 3060 Ti PC 0.084 s/step.
- Baseline for the comparison: v1@30k with temporal ensembling on the 60 states (seed 777), running on the Mac.
- Baseline v1@30k + temporal ensembling (60 states, seed 777): **5/60 = 8.3 %** (v1@30k default inference: 7/60).
- 18:20 Self-starting evaluations (`scripts/eval_pilot_after_training.ps1`): botmind evaluates pilot_full and the PC
  evaluates pilot_half (MuJoCo 3.16.0 added to the PC's `lerobot` env; it renders on the 3060 Ti over SSH, numpy
  unchanged). Each evaluates checkpoints 10k/20k/30k on the 60 states after its training's 30k checkpoint appears.
  Both evaluators were tested on Windows with 1 episode. Results: `data\eval\pilot_{full,half}_0X0000_te\summary.txt`
  on each machine; progress in `logs\eval_pilot_{full,half}.log`. ETA ~19:30.
- **Eval bug found and fixed (2026-10-22, late):** after the DART change, `Sim.wait()` re-applies the stored plan
  every tick. `sim_eval.resting_drift()` called `wait()` with the *previous episode's* plan, so the arm swung back
  toward that pose during the 2 s rest before each episode, disturbing boxes and changing which states were
  sampled. The first pilot evaluations and the v1@30k baseline (all after ~15:00) were on different, partly
  disturbed states. v1@100k (11:16) and all training data are unaffected. Fix: `sync_plan()` before the rest. The
  fixed evaluator reproduces v1@100k's state sequence exactly. Re-running v1@30k, pilot_full@30k, pilot_half@30k.
- First (invalid) pilot numbers, for the record: full 10k/20k/30k = 10.0 / 31.7 / 33.3 %, half = 8.3 / 8.3 / 33.3 %,
  v1@30k + TE = 8.3 %.
- **Corrected pilot results (identical 60 states, verified equal to v1@100k's sequence on the Mac and on botmind):**
  v1@30k + TE 4/60 = 6.7 %; v2 full@30k 14/60 = 23.3 % (10 drops, 19 wrong box); **v2 half@30k 21/60 = 35.0 %**
  (3 drops, 4 wrong box); reference v1@100k 34/60 = 56.7 %. At equal steps v2 >> v1, but the pilot (600 episodes ≈ 0.95
  epoch at 30k) isn't a clean per-epoch data comparison. Half resolution is at least as good and 2.7× faster →
  chosen for v2. Next: the full v2 run (3,000 episodes, half resolution, 300k steps), evaluated on the 200 states.

## 14. Supply layout: correction (2026-10-23)
- The earlier "max-gap" layouts (22.7 / 20.5 / 15.5 mm beside long sides) were **invalid**. The layout search only
  kept boxes on the arm side of the reach tape's centre line, but the tape is 1.7 cm wide, so some boxes overlapped
  it by ~8 mm (visible in the earlier figures). Fixed: the search now uses `game_states.inside_supply` (mat edges +
  the tape's near edge) plus an edge margin.
- With the tape respected, 8 boxes leave much less room:
  | edge margin | free space beside long sides | min gap | pick-order margin |
  |---|---|---|---|
  | 5 mm (loose placement) | ~8 mm | ~7 mm | ~9 mm |
  | **2 mm (template placement)**, clear pick order | **11.2 mm** | **10.0 mm** | **13.1 mm** |
  | 2 mm, clearance only | 12.8 mm | 10.0 mm | 6.0 mm |
  | v1 loose grid | ~10 mm | 3–6 mm at the ends | — |
  So the "≥ 2 cm gaps" rule is not achievable with all 8 boxes on the reachable mat. Chosen layout:
  `sim/assets/supply_layout.yaml` (from `data/layout_search/m2order_s8.yaml`, figure
  `notes/img/phase6_supply_layout_template.png`), placed with a template (±2 mm, ±3°). Gaps then grow as boxes are
  used. More real spacing would need a larger reachable supply area (mat geometry), not a different layout.
- The `rule` sampler (≥ 15 mm beside long sides, ≥ 10 mm everywhere) still applies when fewer boxes remain.

## 15. Full v2 run (started 2026-10-23 02:06 on botmind, user approved)
- Pipeline `scripts/b4_full_v2.ps1` (detached; log `C:\Users\botmi\so101_stack_game\logs\full_v2.log`):
  generation (3,000 episodes, game-rule states `SUPPLY_P_GAME`, template layout `sim/assets/supply_layout.yaml`,
  kinds 45/40/15) → export 320×240 in 8 verified parts + verified merge (`so101_stack_sim_act_v2_half`) → ACT 300k
  steps, checkpoint every 25k (`C:\Users\botmi\so101_runs\act_v2_half`) → evaluation of 300k with temporal
  ensembling on the v1 200-state set (seeds 1001–1004) and the game-rule set (seeds 3001–3004), summaries in
  `data\eval\v2half_300k_{v1,game}.txt`.
- Pre-run checks: layout placement succeeds 50/50; expert 38/40 on game states; game-rule evaluation, aggregation
  and game-state generation tested on botmind.
- Generation speed ~33 episodes/min (16 workers) → ~90 min. Rough ETA: export ~04:30, training to ~09:30–10:30
  (half-resolution step time on botmind not yet measured), evaluation ~20 min after.

## 16. Full v2 results (2026-10-23, botmind; ACT v2, 3,000 episodes, 320×240, 300k steps, temporal ensembling 0.01)
Timeline: generation 02:06–03:31, export + merge 03:31–03:59 (1,260,293 frames, verified), training 03:59–09:43
(0.069 s/step), evaluation 09:43–10:05.

| test set (200 states each) | v1 policy | **v2 policy** |
|---|---|---|
| v1 set (seeds 1001–1004; dense random supplies) | 98/200 = 49.0 % | **144/200 = 72.0 %** |
| game-rule set (seeds 3001–3004; template-layout games + rule-spaced) | — | **186/200 = 93.0 %** |

Section 1 sim criteria on the **game-rule set** (95 % Wilson intervals):
| criterion | no3 | no35 | no10 |
|---|---|---|---|
| height 0–3 ≥ 90 % | 29/29 = 100 % (88–100) | 46/49 = 94 % (83–98) | 18/20 = 90 % (70–97) |
| crooked 4+ (≥ 10 mm) ≥ 70 % | 7/7 (65–100) | 5/6 = 83 % (44–97) | 3/3 (44–100) |
| large onto small ≥ 70 % | 12/12 = 100 % (76–100) | | |
All point estimates meet the criteria, but the crooked-4+ and large-on-small buckets are too small (3–12 episodes)
to establish them, and no10 at height 0–3 sits exactly at 90 %. Game-set failures: missed grasp 6, timeout holding 3,
knocked 3, misplaced 2; wrong box 36/200 (still placed); no backdrop touches.
On the v1 set (dense supplies, which the template layout avoids) v2 still misses the criteria: height 0–3 73 %,
large-on-small 8/18; 45 missed grasps.
Next: a confirmation evaluation with enough episodes in the small buckets (more game-rule states, plus targeted
crooked-tall and large-on-small states).

## 17. Confirmation evaluation (2026-10-23, botmind; ACT v2 300k, 320×240, temporal ensembling)
1,060 episodes pooled (`notes/phase6_confirmation.md`; runs `data\eval\confirm_*` + the first game set on botmind):
600 more game-rule states (seeds 3005–3016), targeted crooked (≥ 10 mm) height-4–7 states per picked type
(60 each, hard stacks, seeds 4001–4006), targeted large-on-small (80, seeds 5001–5002), plus the 200-state game set.
| criterion | no3 | no35 | no10 |
|---|---|---|---|
| height 0–3 ≥ 90 % | 155/159 = 97.5 % (94–99) **confirmed** | 196/206 = 95.1 % (91–97) **confirmed** | 92/94 = 97.9 % (93–99) **confirmed** |
| crooked height 4+ ≥ 70 % | 71/101 = 70.3 % (61–78) met, not confirmed | 74/103 = 71.8 % (62–80) met, not confirmed | 62/69 = 89.9 % (81–95) **confirmed** |
| large onto small ≥ 70 % | 128/152 = 84.2 % (78–89) **confirmed** | | |
- Weak spot: large/medium boxes onto **tall** crooked stacks (targeted runs: no3 at height 7 8/16, height 6 4/7;
  no35 height 7 4/9). Failures there are mostly knocks on approach/release, not missed grasps. Plausible causes:
  the release tilt at heights 6–7 (wrist_flex limit), and v2's data weighting away from tall stacks.
- Overall 937/1060; failures: missed grasp 50, misplaced 25, knocked on release 17, drop 13, timeout holding 10,
  knocked on approach 8. Wrong box 178 (17 %, still placed). **Backdrop touches 3/1060** (safety flag to watch).
- Operational: 20 parallel evaluations exceeded 16 GB GPU memory at startup (6 crashed with CUDA OOM); restarted
  staggered. Keep ≤ 14 parallel or stagger starts.

## 18. Targeted round, step 1: expert diagnosis on tall crooked states (2026-10-23)
`expert.py --kind clean --game-states --height-min 5 --hard-p 1 --min-crooked-mm 10 --want-types <t>`, 60 per type (botmind):
| expert picks | overall | h5 | h6 | h7 | failures |
|---|---|---|---|---|---|
| no3 | 52/60 = 87 % | 29/31 | 14/14 | 9/15 = 60 % | knocked_stack 6, hit_backdrop 2 |
| no35 | 48/60 = 80 % | 27/28 | 10/16 | 11/16 = 69 % | missed_grasp 9, knocked 1, hit_backdrop 2 |
The policy's tall-stack weakness (h7 ≈ 50 %) sits close to the **expert's own ceiling** there, so more data from the
same expert has limited upside. Proposed: fix the expert at heights 6–7 first (no3 tilted-release knocks; no35 missed
grasps, cause not yet known), then the targeted data + fine-tune round.

## 19. Expert fixes for tall stacks (2026-10-23) and the targeted v3 round
Diagnosis (`tall_diag` traces of the failing targeted seeds):
1. **No.35 "missed grasps" were backdrop collisions**: boxes near the arm base (r ≈ 14–15 cm, late-game template slots).
   From home, the joint-space move swung the upper arm back into the backdrop (12.5 mm away at home) before the base
   turned, and the blocked arm never reached the box. After fixing the approach, the same collision moved to the carry:
   rising vertically above a near-base box to the safe height over a tall stack folds the arm into the backdrop.
   (A real collision risk on the real arm too, not only a sim score issue.)
2. **No.3 knocks at height 7**: the release is tilted 9–10° (wrist_flex limit), so a 70 mm box's low end sat ~6 mm below
   its centre; with the 3 mm clearance set for the centre, the corner pressed into the top box.
3. One case: a precarious stack tipping on its own (not the arm).
Fixes in `expert.py`: `safe_move()` checks every joint-space move against the backdrop (`path_clear`, 15 samples) and
reorders it (pan first, or the rest first) when needed; the carry rises no closer than `RISE_MIN_R` = 19 cm to the pan
axis; the release height is raised until the held box's **lowest corner** (`lowest_corner()`) clears the top face by
3 mm.
Expert on the same targeted states (120 episodes): no3 52 → **56/60**, no35 48 → **60/60**; height 7: no3 9/15 → 12/15,
no35 11/16 → 16/16; no backdrop touches. Regression on normal game states: clean 59/60, DART 57/60, retry as before.
v3 round (`scripts/b4_v3.ps1`, started 12:35 on botmind): 1,000 targeted episodes (heights 5–7, hard stacks, picks
no3/no35, 70 % crooked ≥ 10 mm) → merged with the 3,000 v2 episodes → fine-tune v2 300k for 60k steps (LeRobot
recomputes the normalisation stats from the merged data, `lerobot_train.py:641-667`) → re-evaluation on the same
targeted seeds + game sets.

## 20. v3 results (2026-10-23; v2 300k + 60k fine-tune on 4,000 episodes incl. 1,000 targeted, fixed expert)
Re-evaluation on the same seeds (760 episodes; `notes/phase6_confirmation_v3.md`):
| criterion | v2 | **v3** | 95 % interval | confirmed |
|---|---|---|---|---|
| height 0–3 no3 / no35 / no10 | 97.5 / 95.1 / 97.9 % | **97.1 / 100 / 98.4 %** | 92–99 / 97–100 / 91–100 | yes / yes / yes |
| crooked 4+ no3 | 70.3 % | **74.5 %** | 65–82 | **no** (met) |
| crooked 4+ no35 | 71.8 % | **88.8 %** | 80–94 | yes |
| crooked 4+ no10 | 89.9 % | **92.4 %** | 83–97 | yes |
| large onto small | 84.2 % | **87.1 %** | 80–92 | yes |
Game-rule 200-state set: 186 → **194/200 = 97 %**; backdrop touches 3 → **0**; overall 701/760.
Remaining gap: no3 on crooked tall stacks (targeted runs 43/60; height 6–7: 14/23; failures spread over drop,
knocked on release, missed grasp, knocked on approach). The fixed expert gets 93 % on these states, so this is a
learning gap, not an expert ceiling.
