# Phase 6: Train and evaluate in sim (report)

**Result:** ACT v3 meets every CLAUDE.md Section 1 sim criterion, and all but one are confirmed by their 95 %
confidence interval. Known weak spot: **No.3 (large) boxes onto crooked stacks of height 4+**, 74.5 % (65–82 %).
Policy: `C:\Users\botmi\so101_runs\act_v3_ft\checkpoints\060000\pretrained_model` (botmind), 320×240 input,
inference with **temporal ensembling 0.01** (re-plan every tick).

Detailed working notes: `notes/phase6_overnight.md` (v1 pipeline night), `notes/phase6_data_v2_plan.md` (v2/v3 plan,
implementation, pilot, layout, results §12–20), `notes/phase6_confirmation.md` (v2), `notes/phase6_confirmation_v3.md`
(v3), `notes/phase6_eval_log.md` (v1 checkpoint curve).

## 1. What was done
| version | data | training | where |
|---|---|---|---|
| v1 | 2,000 episodes, v1 expert (tight grasp), 640×480 | 100k steps (1.1 epochs) | 3060 Ti PC |
| pilot | 600 v2 episodes, full vs half resolution | 30k steps each | botmind / PC |
| v2 | 3,000 v2 episodes (game-rule supplies, template layout), 320×240 | 300k steps (1.6 epochs) | botmind |
| v3 | v2 + 1,000 targeted tall-crooked episodes from the fixed expert | v2 + 60k fine-tune | botmind |
All ACT, LeRobot ca69a206, AMP, batch 8, chunk 100.

Main changes between versions:
- **Expert v2:** centred grasp with a margin set by the free space (2–8 mm per side) and a deep grip; hover and slow
  final descent; DART recovery demos (execution noise, clean-plan labels); retry demos starting after a miss.
- **Game spec:** a fixed **template starting layout** (`sim/assets/supply_layout.yaml`, ≥ 11.2 mm beside long sides,
  ≥ 10 mm all round, 13.1 mm pick-order margin). The "≥ 2 cm gaps" rule turned out to be impossible with 8 boxes.
- **Expert fixes for tall stacks:**
  - backdrop-safe move ordering, after the upper arm hit the backdrop on near-base picks (a real-arm safety issue too);
  - the carry rises ≥ 19 cm from the pan axis;
  - the release is raised until the box's lowest corner clears the top face.
- **Inference:** temporal ensembling (+15 points over 100-step chunks on v1).

## 2. Evaluation method (`scripts/sim_eval.py`)
- **Episode:** closed loop at 30 Hz from the rendered `front`/`wrist` frames and the joint state, in LeRobot units,
  with the real arm's 10-unit `max_relative_target` and home ± 3°.
- **Success:** judged from the sim's exact state by the Section 1 rule (box on top, stack standing 3 s after release).
- **Failure types:** missed grasp, drop, knocked on approach/release, misplaced, timeout. The wrong-box and
  backdrop-contact flags are reported separately.
- **Test sets** (seeds never used in training):
  - v1 set: seeds 1001–1004;
  - game-rule set: seeds 3001+, template-layout games and rule-spaced supplies;
  - targeted crooked-tall sets per picked type: seeds 4001–4006;
  - large-on-small: seeds 5001–5002.
  States that creep at rest are skipped.
- **Bugs found and fixed along the way:**
  - a stale-plan replay in the pre-episode rest check;
  - LeRobot's streaming encoder dropping frames under load, which would have misaligned images and states. The
    export now uses `--queue 600` plus a frame check, and pixel alignment was verified.

## 3. Results
| policy | v1 set (dense supplies) | game-rule set |
|---|---|---|
| v1 (100k) | 98/200 = 49.0 % | — |
| v2 (300k) | 144/200 = 72.0 % | 186/200 = 93.0 % |
| **v3** | — | **194/200 = 97.0 %** |

Section 1 criteria, v3 (760 episodes on the same seeds as the v2 confirmation; 95 % Wilson intervals):
| criterion | No.3 | No.35 | No.10 |
|---|---|---|---|
| height 0–3 ≥ 90 % | 97.1 % (92–99) ✓ | 100 % (97–100) ✓ | 98.4 % (91–100) ✓ |
| crooked 4+ ≥ 70 % | **74.5 % (65–82) met, not confirmed** | 88.8 % (80–94) ✓ | 92.4 % (83–97) ✓ |
| large onto small ≥ 70 % | 87.1 % (80–92) ✓ | | |
✓ = the lower bound of the 95 % interval clears the target. Overall 701/760; failures: drop 13, missed grasp 14,
knocked on release 14, misplaced 11, knocked on approach 4, timeout 3. Wrong box 106 (14 %, still placed).
**0 backdrop touches.**

## 4. Known weak spot and open issues
- **No.3 onto crooked tall stacks** (targeted runs 43/60; height 6–7: 14/23). The fixed expert reaches 93 % there, so
  this is a learning gap. Options later: more No.3-only targeted data and a longer fine-tune; Phase 11's model-based
  final placement would also help.
- **Wrong box 14 %:** the policy doesn't always follow the nearest-box rule (harmless for ACT in the game, but it matters
  for SmolVLA instruction accuracy later).
- **Dense supplies** (v1 set) remain weaker (v2: 72 %), relevant only if the template layout isn't used.
- **The physics gap to the real arm** is untested: Phase 8.
- **Not done in Phase 6:**
  - the SmolVLA comparison (its v2 VLA data wasn't generated, and the v1 VLA dataset is obsolete);
  - `sim_game.py` (full games in sim against a scripted human);
  - pushing the datasets to the Hub (needs `hf auth login`).
- **The real-arm runner must reproduce the policy input exactly:**
  - undistort the `front` frames and apply the corner mask (Phase 3);
  - resize both cameras to 320×240 with INTER_AREA;
  - run temporal ensembling every 30 Hz tick.

## 5. Compute
- **Mac:** code, orchestration, small tests.
- **3060 Ti PC** (`so101-gpu`): v1 training, the half-resolution pilot.
- **botmind** (RTX 4070 Ti SUPER, 20 cores): generation (16 workers, ~2 s/episode), export, training
  (0.069 s/step at 320×240), evaluation (≤ 14 parallel; 20 at once exhausted GPU memory).
- Access: SSH over Tailscale with the project key. Both PCs run detached pipelines (WMI-started) that survive disconnects.

## 6. Next step
Phase 7 (classical perception): `stack_state()` (box types per level), `collapse_check()`, `drop_check()` and
`hand_in_workspace()`, developed on sim renders first, then validated on the real frames and the recorded clips.
