# Phase 5: Scripted expert, game states and data generation

Date: 2026-10-21. Status: **in progress, blocked on a supply-layout decision (section 2).**

## 1. `scripts/game_states.py`

- **Stack:** heights 1–7. The human starts, so the robot never places onto the empty
  base, and in a strict game its turns come onto stacks of 1, 3, 5, 7.
- **Box types per level** from the real mix (3 / 4 / 1).
- **Placement error:** per-level offsets (σ 4 mm, or 9 mm for "precarious" stacks, clipped
  at ± 20 mm) and yaw (σ 7°, or 15° on the top level of hard stacks).
- **Hard cases** over-sampled: a small top box (large onto small next), precarious stacks.
- **Standing check:** every stack is settled 0.4 s, then must not move > 1 mm or tilt > 2°
  over 0.6 s.
- **Supply:** non-overlapping, entirely on the arm side of the reach-line tape, mostly
  square to the mat (a loose grid like the real one), 2 mm minimum gaps.
- **Pick rule** (Section 3c): ACT = the nearest box to the stacking spot with room for the
  jaws; VLA = the nearest such box of the requested type. "Room for the jaws" = 10 mm free
  on the fixed-jaw side and 22 mm on the moving-jaw side (the expert puts the moving jaw on
  the roomier side).

## 2. Blocking finding: the supply can't hold enough boxes

(`notes/phase5_supply_layout_check.txt`)

With the reach-line tape, only ~61–66 % of the black mat is usable. Random layouts that
also leave one pickable box:

| boxes in the supply | mat as now | mat moved 4 cm toward the arm | 5 cm | 6 cm |
|---|---|---|---|---|
| 7 (robot's first turn) | **0 / 60** | 40 / 60 | 51 / 60 | 55 / 60 |
| 6 | **0 / 60** | 58 / 60 | 60 / 60 | 59 / 60 |
| 5 | 21 / 60 | 58 / 60 | 59 / 60 | 60 / 60 |

So with the mat where it is, the robot's first turns (7 or 6 boxes in the supply) usually
have no box it can pick. A straight 5 cm shift toward the pan axis puts two mat corners off
the table near the arm's corner, and grasps near the inner corner then make the arm touch
the table. A new position must be optimized: slid along the table's left edge, kept on the
table, clear of the arm base and the yellow area, with every part pickable.

Options put to the user:
A. Re-position the black mat (I compute the exact placement and verify it; you re-tape it
   and re-touch its 4 corners, ~5 min).
B. Keep the mat; at most 5 boxes on it at a time, the rest held in reserve off the mat for
   the human to refill.
C. Fewer boxes per game (6), which contradicts the decided 8.

## 3. Re-evaluation after user feedback (2026-10-21; `notes/phase5_supply_reevaluation.txt`)

User: all boxes fit on the mat, and every placement frees space for the next pick. Section
2 was **too pessimistic**:
1. **It tested random dense layouts.** A real game starts from a loose grid, and each
   removal frees the neighbours.
2. **Jaw clearance was over-strict** (22 mm along the whole long side). The jaws only touch
   the middle of the box (±1 cm along its length) and are 6–7 mm thick, so **10 mm free on
   each side near the middle** is enough with a tighter pre-grasp opening (width + 6 mm).
   This must be confirmed by the grasp test with neighbours present.

Still true: vertical picks only reach ~29 cm, so boxes beyond the tape can only be taken
by the human. Results with a 1 cm-gap grid of all 8 boxes, simulating games (human takes
a box, robot takes the nearest pickable one, alternating):

| set-up | fit | robot stuck |
|---|---|---|
| whole mat, human picks at random | 8/8 (2 beyond reach) | 167/300 games |
| **whole mat, human takes the beyond-tape boxes first** | 8/8 | **0/300** |
| **mat expanded 2 cm toward the arm** (strip along the corner-1/corner-4 edge) | 8/8 on the arm side | **0/300** (random human) |

Section 2's "move the mat" option is withdrawn.

## 4. Decision: expand the black mat 2 cm toward the arm (user, 2026-10-21)

- **What:** a 2 cm wide, 19.2 cm long black matte strip along the mat's edge between
  corner 1 (nearest the arm base) and corner 4 (nearest the yellow area), on the outside,
  toward the arm. The new mat is 22.9 × 19.2 cm (`notes/img/phase5_mat_expansion.png`).
- **Unchanged:** corners 2 and 3 and the reach-line tape. The tape's ends stay put; the end
  on edge 1→2 is now 16.7 cm from the new corner 1.
- **Re-touch:** the 4 supply corners with `so101_record_points.py --points supply`
  → `inputs/supply_points_<date>.json`. `scene_layout.py` uses the newest such file and
  the expanded size.

## 5. Scripted expert (`scripts/expert.py`)
Privileged state + IK (`grasp_test.Sim.ik`), one episode = home (±3° jitter) → pick → carry → place → release → retract.
- **Pick:** nearest pickable box to the stacking spot (Section 3c rule; requested type for VLA). Jaws open to
  width + ~6 mm, straight-down Cartesian descent (Phase 4), close, lift 6 cm.
- **Place target:** 9×9 grid over the top face; choose the xy that maximizes the worst centre-of-mass support
  margin over **all** levels (matters for large-on-small). Yaw aligned to the top box (mod 180°).
- **Release:** box bottom 3 mm above the top face, last 2 cm of the descent slow, gripper opens over 0.6 s,
  then straight up 5 cm. The episode ends there (no return home). Judged 3 s later (not recorded).

Fixes while bringing it to 20/20 (seed 0):
| Symptom | Cause | Fix |
|---|---|---|
| 5/20 "misplaced" (16–29 mm), IK error 17–27 mm | release IK stuck in one wrist direction | try both wrist directions (box is 180°-symmetric) + fallback seeds |
| 2/20 still misplaced with ~0 IK error | joints stalled 5–18° short: the **wrist-camera housing hit the backdrop** at roll ≈ −180° | `backdrop_gap()` (`mj_geomDistance`): ≥ 10 mm for wrist/gripper/camera, ≥ 3 mm for shoulder/upper/lower arm; any arm–backdrop contact during an episode = failure `hit_backdrop` |
| 3/20 "ik_place" at h = 6–7 | IK traded 7 mm of position for jaw verticality | `ik_pos_first()`: lower the orientation weight 100 → 30 → 10 → 3 until the position error ≤ 2 mm |

Note: the upper arm sits 5–10 mm from the backdrop at every place pose, whatever the wrist does. That's a
property of the layout. The real hover spot-checks (Phase 3) touched nothing, but the margin is small: if the
backdrop is moved, re-check it.

Held-out check, `expert.py --n 100 --seed 1` (`notes/phase5_expert_check100.txt`): **98/100**.
| bucket | result |
|---|---|
| height 1 / 2 / 3 / 4 / 5 / 6 / 7 | 12/12, 23/23, 20/20, 15/15, 11/11, 6/7, 11/12 |
| picked no3 / no35 / no10 | 29/30, 57/58, 12/12 |
| large-on-small (no3_on_no10) | 10/10 |
Both failures are `hit_backdrop` during tall-stack carries (the placement itself succeeded); the generator discards them.
Placement offset ≤ 4 mm; stack motion ≤ 2 mm; release tilt 1° up to h = 5, 5–10° at h = 6–7 (Phase 3 predicted ≤ 10–11°).

## 6. Visual randomization (`scripts/randomize.py`) and scene changes
Rendering-only (physics untouched), drawn per episode and stored in `meta.json`:
- front camera ±1 cm per axis, ≤ 1.5° rotation; wrist ±3 mm, ≤ 1°, f 518 px ±5 %, k1 ∈ [−0.05, 0.05]
  (applied as an image remap; no checkerboard calibration)
- lights: ceiling ×0.75–1.25 (position ±20 cm), fill ×0.5–1.5, headlight ×0.6–1.4; hard shadows on in 50 %
- colours around the real values: white table, black mat/backdrop, yellow tape, reach-line tape, carpet,
  room wall (light lower panel, dark blue upper wall; distance and panel height shifted); arm grey 0.03–0.15
- boxes: per type, brightness ±15 % and per-channel tint ±6 %; plain colour (no print) in 5 % of episodes.
  Texture rotation by 180° comes from the box yaw; mirror flips are not done (the print would read backwards)
- image space, per camera: saturation 0.6–1.05, contrast 0.8–1.1, brightness ±0.08, gamma 0.85–1.15,
  Gaussian blur σ ∈ {0, 0.6, 1, 1.5}, noise σ ≤ 0.02
- front frames: real front frames are undistorted before the policy (Phase 3), which leaves black corners;
  the same mask is applied to sim front frames.
- start pose ±3° per joint (expert).
- Not done: distractor objects outside the play area (optional in the spec).

Scene changes found by comparing the first renders with the real frames (`build_scene.py`, visual only):
- **the arm is black** in reality (both cameras see the jaws in every frame); the vendor model is yellow.
  Arm materials now dark grey (randomized).
- a room wall behind the arm as seen from `front`: a light lower panel and a dark blue wall
  (`front_cam_stack3.png`). Distance (0.45 m) and panel height (5 cm above the table top) are APPROX, randomized.

## 7. Episode generator (`scripts/generate.py`) and export (`scripts/export_lerobot.py`)
- Raw episode `data/raw/<run>/ep_NNNNNN/`: `front.mp4`, `wrist.mp4` (640×480, 30 fps, H.264 crf 18),
  `data.npz` (state, action in LeRobot units via `JointMapping.sim_to_real`), `meta.json` (task string,
  mode, requested/picked type, stack composition, offsets, expert result, randomization draw).
- Frame t = observation after control tick t; action[t] = joint targets of tick t+1.
- Only successes are written. ~18 s per episode on the Mac (render ~9 ms + post-processing ~9 ms per frame pair).
- `export_lerobot.py` (WSL lerobot env): features from LeRobot's `hw_to_dataset_features` with the
  SO-101 `{joint}.pos` names and the `front`/`wrist` cameras; `--dataset act` uses only ACT-mode episodes;
  `--dataset vla` uses all, with the size-naming task string. Per-episode game metadata goes to
  `meta/so101_stack_episodes.json`, for evaluation split by type and pairing.

## 8. 20-episode pilot (`data/raw/act_pilot`, seed 100) — evidence for the checkpoint
`generate.py --n 20 --mode act --seed 100` → 20/20 attempts succeeded. `inspect_episodes.py act_pilot`
(`notes/phase5_act_pilot_inspect.txt`):
- Figures: `notes/img/phase5_act_pilot_joints.png` (state/action per joint), `..._dist.png`, `..._frames.png`
  (front + wrist at 4 moments, 6 episodes).
- Units: every joint inside the calibrated LeRobot range; gripper 0–27 (0 = commanded closed on the box).
  wrist_flex reaches its +101.9° servo limit on tall stacks (as Phase 3 found).
- Smoothness: max |action − state| ≤ 2.6° for body joints (the real arm's max_relative_target is 10);
  max per-frame action step 3.1°. Episodes 329–398 frames (11.7 s mean).
- Distribution (20 episodes, too few to judge balance): heights 1–6 (no 7 in this draw), picks no3 10 / no35 8 /
  no10 2, 6 pairings including large-on-small ×3, max stack offset median 6.5 mm (max 22 mm),
  supply 10 game-progression / 10 random, hard shadows in 11.

Two problems found by looking at the pilot, both fixed before the final pilot run:
1. **elbow_flex start beyond its limit (99.5° vs 97.0°).** The ±3° home jitter wasn't clipped and home sits
   0.5° from the limit. The jittered start is now clipped to the calibrated range.
2. **wrist_roll multimodality.** The moving-jaw side was always +1 (both jaw clearances are 10 mm, so the
   side is never forced), so the pick roll landed anywhere in ±180°. Similar-looking boxes (yaw 89° vs 91°)
   got opposite 300° roll swings, which ACT would average. New rule: of the feasible sides, take the one whose
   roll is nearest the place roll (−12° sim). Pick roll is now −10…176° LeRobot (no wrap); episodes are 1.5 s shorter.

VLA mode (`data/raw/vla_smoke`, 6 episodes): picked type = requested type 6/6; task strings name the size.
Export check (Mac LeRobot 0.6.2 env, 2 episodes, local only): reads back as `so101_follower`, 30 fps,
`observation.state`/`action` float32[6] with `{joint}.pos` names, both videos 480×640×3, task string; state
identical to the raw file. **The real export must run in the WSL training env**, whose LeRobot version may differ.

Cost: ~16 s per episode in one process, ~2.1 MB per raw episode. 3,000 episodes ≈ 6.3 GB; ≈ 13 h in one process,
or ~3–5 h with 4 parallel processes (8 cores; 4 performance).

## 9. Open issues
- `hit_backdrop` in ~2 % of attempts (tall-stack carries). Discarded, but it shows how small the real
  upper-arm/camera-to-backdrop margin is at h ≥ 6. Don't move the backdrop closer.
- Room wall distance and panel height are APPROX (randomized); not measured.
- Mirror flips of the box print and distractor objects aren't used.
- Tap-test order mismatch from Phase 4 is still open (sim: the tapped box slides out first).

## 10. Proposal for the large generation (needs the user's OK)
- ACT: 2,000 episodes, `--mode act`; VLA: 1,000 more with `--mode vla` (the VLA dataset uses all 3,000).
- More weight on hard cases: hard-state share 0.5 → 0.6; height weights 1–3 : 4–7 = 40 : 60
  (now uniform 1–7); keep the game-progression supply share at 60 %.
- 4 parallel processes with distinct seeds (`--seed 1000..1003`, separate `--run` folders), then
  `inspect_episodes.py` on each run before copying `data/raw/` to WSL and running `export_lerobot.py` there.
