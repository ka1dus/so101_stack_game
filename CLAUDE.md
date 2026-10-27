# SO-101 Stacking Game (MuJoCo, sim-trained): Project Brief for Claude Code

Read this whole file before doing anything. It is the source of truth for this
project. Record every decision in the **Decisions log** at the bottom so later
sessions don't re-derive or contradict it.

---

## 1. Goal

Build a turn-based **stacking game** played between a human and my SO-101 arm.

- Players take turns placing one small cardboard box on top of a shared stack.
- **A player loses** if they drop a box while carrying it, or if the stack
  collapses during their turn or within the settle window after their placement.

**Hard constraint:** all policy training happens in **MuJoCo simulation**. No real
demonstrations are used for training. The real setup is used only for
calibration (measurements, camera frames, stability checks) and evaluation.

**Core design principle:** don't learn the game end to end.
- **Learned (in sim):** one physical skill, *"pick the next box from the supply
  and place it gently on top of the current stack"*.
- **Classical code:** game rules, turn-taking, stack/collapse/drop perception,
  safety, the user interface.

**Mixed box sizes from the start:** three MAX staple-box types (No.3 large,
No.35 medium, No.10 small) are used together. Decided by me; don't propose a
single-type version. Instead, keep every evaluation **broken down by box type and
box-pair combination**, so failures can be traced to calibration vs size handling.

### Success criteria
1. **Sim:** placement success ≥ 90% on stacks of height 0–3 and ≥ 70% on crooked
   stacks of height 4+ (success = box on top and stack standing 3 s after release),
   **for every box type**, and ≥ 70% on the hardest pairing (large box onto small).
2. **Real placement trials:** ≥ 80% on fixed real stacks of height 0–3, reported per
   box type and per pairing.
3. **Real games:** complete games against a human with correct win/loss calls,
   no false collapse calls, and no unsafe motion.

---

## 2. What already exists (do not redo)

### Hardware
- **Robot:** SO-101 follower arm, 6 Feetech STS3215 servos, clamped to a small table.
  **No leader arm.**
- **Cameras:**
  - `front`: on a tripod, viewing the setup at about **45°**
  - `wrist`: mounted on the gripper
  - Keep these exact names; existing tools and the SmolVLA camera mapping depend on them.
- **Mac (Apple Silicon):** the arm and cameras are plugged in here. All real-arm runs,
  and the final game, run here. **You never move the real robot**; give me commands.
- **GPU machine:** Windows with an **NVIDIA RTX 3080** (check 10 vs 12 GB with
  `nvidia-smi`). A WSL2 Ubuntu install has a working `lerobot` conda env (LeRobot
  `main`, from source, CUDA verified) for training.

### Prior work to reuse
- A previous project (`so101_sim/`) planned a MuJoCo twin of a pick-and-drop task.
  **If it exists, reuse its validated pieces**, especially the SO-101 MJCF model and
  `joint_mapping.py`. Check its `notes/` and Decisions log first. If a component was
  never validated, treat it as not done.
- Helper scripts on the Mac:
  - `so101_go_home.py`: drives the arm to the saved home pose and holds it
  - `so101_rollout_loop.py`: homes before and after each trial; success tally
  - `so101_start_poses.py`, `so101_grasp_poses.py`: dataset analysis
  - home pose file: `~/.cache/so101_home/my_follower_arm.json`

### Lessons already learned on the real arm (design around them)
- **For a vision policy, the camera setup is part of the model.** A cluttered
  background and slight camera drift once dropped a working policy to 1 of 5
  positions; a plain background and re-tuned camera fixed it with no retraining.
- Lock camera pose (tape marks, rigid tripod) and, where possible, autofocus,
  exposure and white balance.
- Every policy run must start from the same saved home pose.
- High contrast between objects and background makes perception far more reliable.

### Units and format conventions (must match LeRobot)
- Joint order: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`
- LeRobot `use_degrees=True`: body joints in **degrees** from calibrated mid-range;
  gripper **0–100** (`MotorNormMode.RANGE_0_100`). Sim must convert via the validated
  joint mapping. Verify in LeRobot source (`src/lerobot/robots/so_follower/`,
  `src/lerobot/motors/`) and my calibration file, not from memory.
- Dataset (sim-generated) format:

| Key | Shape / type |
|---|---|
| `observation.state` | float32[6] |
| `action` | float32[6] (absolute joint targets) |
| `observation.images.front` | video 480×640×3 |
| `observation.images.wrist` | video 480×640×3 |
| fps | 30 |
| robot_type | `so101_follower` |
| task | ACT: `"Pick up the next box and place it on top of the stack"`; SmolVLA: `"Pick up the {large\|medium\|small} box and place it on top of the stack"` (see Section 3c) |

---

## 3. Game specification

Items marked **TODO** must be confirmed with me before Phase 2 ends.

| Item | Default / proposal | Status |
|---|---|---|
| Boxes | **mixed MAX stapler-staple boxes**: No.3 (large, 7.0×4.0×1.7 cm, 38 g), No.35 (medium, 6.0×3.5×1.6 cm, 31 g), No.10 (small, 5.2×2.9×1.3 cm, 21 g); printed designs. Grasp across the width, which must be **< gripper max opening − ~1.5 cm for every type** | dimensions + mass done (`measurements.yaml`); gripper max opening 150 mm at tips (ruler), 154 mm in sim; every type fits with ≥ 95 mm to spare after the 1.5 cm clearance (`notes/phase2.md`): done |
| Box mix | 3 × No.3, 4 × No.35, 1 × No.10 (8 boxes) | done |
| Human's box choice | the human may pick any box from the supply | default |
| Robot's box choice | see Section 3c | default |
| Box contents | all **full** or all **empty**, never mixed (mass and stability differ a lot; loose staples shift inside) | done: **all full** |
| Box orientation | lying flat (largest face down) for both supply and stack | done: always flat |
| Box count | 8 (max stack height ≤ 8 levels; also limited by reach/view) | done |
| Supply | **black mat**, arm's left; ruler size 20.9 × 19.2 cm; centre/heading in `measurements.yaml` | done |
| Stack base | **yellow area**, arm's right; ruler size 21.4 × 14.6 cm; stacking spot in `measurements.yaml` | done |
| Contrast | boxes contrast well with black and yellow tape but **poorly with the white table** | see Section 3a |
| Turn order | coin flip or the human starts | done: **human starts** every game |
| Human turn end | **keypress** (vision-based detection is a later upgrade) | default |
| Robot turn end | box released + gripper raised clear of the stack (or timeout) | default |
| Settle window | **3 s** after each placement before judging stability | default |
| Lose: drop | box leaves the gripper before reaching the stack | default |
| Lose: collapse | any stack box ends on the table, or the stack top moves > threshold, during a turn or its settle window | default |
| Max height | limited by arm reach and camera view; compute in Phase 3 | open until Phase 3 (computed there, then confirmed with me) |

### 3a. Visual contrast (fix in the real setup before calibrating)

The boxes blend into the white table. That hurts both the learned policy (it has to
find box edges in pixels) and classical stack detection. Proposed fixes, cheapest first:
1. **Dark matte mat or card** inside the yellow-tape stack area (and ideally under the
   whole play area), larger than the box footprint. Keep the yellow tape visible
   around it. Matte, to avoid glare.
2. **Dark matte backdrop** behind the stack in the `front` view, so upper boxes are
   seen against dark rather than white.
3. If the boxes still don't stand out: a single strip of coloured tape on each box's
   edges (same on every box).

Decide with me in Phase 2, set up the real scene, **then** take the calibration photos.
The sim must model whatever is chosen.

### 3b. Box appearance in sim

The printed design matters for vision. Use photos of each box face (`inputs/photos/box_faces/`)
as **MuJoCo textures** on the box geometry. In data generation, randomize slightly:
hue/brightness shift, texture flip/rotation, and occasionally a plain colour, so the
policy keys on box shape and edges rather than one exact print.

### 3c. Which box the robot picks (mixed sizes)

Mixed sizes mean the robot must choose a box, and the choice affects stability.
- **ACT has no instruction input.** If the scripted expert chose boxes randomly,
  identical-looking scenes would map to different actions and ACT would average
  them (hesitation, reaching between boxes). So for ACT the expert uses a
  **deterministic rule** visible from the image: pick the reachable box **closest to
  the stack area**; ties broken by a fixed order. The policy learns that rule.
- **SmolVLA takes the size as an instruction:** the expert picks the nearest box of
  the requested type, and the task string names it (`large` / `medium` / `small`).
  The game controller then decides the size each turn, via the rule or later strategy
  (Phase 11). This is the main reason to train SmolVLA in this project.
- Every generated episode stores the picked box's type in metadata, so evaluation can
  be split by type even for ACT.

### 3d. Visual resolution and size discrimination

No.3 and No.35 boxes look similar. SmolVLA resizes camera images to 256×256 (check
the installed config), so a ~1 cm size difference may be only a few pixels in the
`front` view. The **wrist camera** is the main source of size detail near the grasp.
In Phase 3, report the pixel width of each box type in both cameras at native and at
policy input resolution, and flag it if types are hard to separate.

---

## 4. System architecture

```
game/controller.py   state machine: SETUP → ROBOT_TURN / HUMAN_TURN → SETTLE → JUDGE → GAME_OVER
  ├── perception/    CV on the `front` camera (+ gripper state); see Phase 7 for method
  │     stack_state()      -> list of box poses, stack height, top-box pose
  │     collapse_check()   -> compare before/after; box on table outside stack area
  │     drop_check()       -> gripper closes past "holding a box" width mid-carry
  │     hand_in_workspace()-> motion/skin detection inside a workspace mask
  ├── skill/          the learned policy (ACT first; SmolVLA optional)
  │     run_place_skill()  -> runs the policy until termination condition or timeout
  ├── safety/         speed caps, e-stop key (torque off), pause on hand detection
  └── ui/             turn indicator, live camera view, result screen, sounds
```

- The policy runs in **our own control loop** using LeRobot's policy inference API
  (load policy, build observations, `select_action`, send to robot), **not** the
  `lerobot-rollout` CLI, so the game controller decides when a turn ends.
- Skill termination is detected classically: gripper opened above the stack top
  plus wrist raised by a margin, or a timeout. Then return home.
- The robot holds still with torque on during the human's turn.

---

## 5. Inputs I will provide (in `inputs/`)

```
inputs/
  measurements.yaml         # filled-in measurements; read its measurement_quality block first
  table_points_20261019_181950.json   # raw joint angles of gripper-tip touches (table edges,
                            # mat corners, stacking spot); positions in measurements.yaml came
                            # from these via FK with so101_new_calib.urdf (deg -> rad). Redo the
                            # conversion with the validated joint mapping in Phase 1.
  photos/
    front_cam_frame.png     # FROM the front camera: empty stack base, arm at home
    front_cam_stack3.png    # FROM the front camera: a mixed 3-box stack (one of each type) on the base
    wrist_cam_frame.png     # FROM the wrist camera, arm at home
    topdown.jpg             # whole setup from above, ruler / A4 visible
    side.jpg                # side view: tripod height and angle
    box_closeup.jpg         # a box next to a ruler
    box_faces/<type>/       # per type (no3, no35, no10): each of the 6 faces, flat-on, even light (textures)
    table_tape_areas.jpg    # top-down of the black-tape supply area and yellow-tape stack area
    gripper_open.jpg        # gripper fully open next to a ruler
  videos/
    stack_collapse.mp4      # a real stack being pushed over (for perception tests)
    human_turn.mp4          # me placing a box (hand-detection test)
  stability/
    overhang_tests.yaml     # real topple tests (Phase 4 protocol)
  calibration/
    my_follower_arm.json    # LeRobot calibration file from the Mac
```

If something you need is missing or ambiguous, **stop and ask**. Don't guess
dimensions or masses.

---

## 6. Working rules

1. **Verify, don't recall.** Check installed source or docs for MuJoCo, LeRobot and
   robot-model APIs before using them. Cite the file/line or URL in your notes.
2. **Separate environments:** the sim env (MuJoCo + numpy + scripts) generates raw
   episodes; the LeRobot env (WSL `main`) exports datasets and trains. Write datasets
   with the **same LeRobot version used for training**. Don't modify that env without asking.
3. **Rendering backend:** confirm offscreen rendering works (`MUJOCO_GL`) on the
   machine before building on it. WSL GPU OpenGL may not; use native Windows or the Mac.
4. **Small steps, evidence each time:** renders, numbers, plots, log excerpts in `notes/img/`.
5. **Look at the images** you render and compare them with the real frames.
6. **Never move the real robot.** Produce exact commands for me to run.
7. **No secrets in files.** Use `hf auth login`.
8. **Checkpoints:** stop for my OK at the end of Phases 0, 2, 3 and 4, before large
   data generation in Phase 5, and before first real-arm motion in Phase 8.
9. **One report per phase** in `notes/phaseN.md`: what was done, evidence, open issues, next step.
10. **Test classical components against recorded real clips** before live use.

---

## 7. Project layout

```
so101_stack_game/
  CLAUDE.md
  inputs/                 # read-only for you
  sim/
    assets/               # SO-101 model + meshes, box/tray/base models
    scene.xml             # calibrated twin of the game setup
  scripts/
    joint_mapping.py      # reused or rebuilt + tests
    render_check.py       # sim vs real camera overlays
    stability_check.py    # sim topple tests mirroring the real protocol
    expert.py             # scripted pick-and-place on arbitrary stacks
    game_states.py        # randomized stack-state generator
    generate.py           # episodes -> raw files
    export_lerobot.py     # raw -> LeRobotDataset (run in the lerobot env)
    sim_eval.py           # policy evaluation in sim, by height/crookedness
    sim_game.py           # full game in sim: policy vs scripted "human"
  perception/             # stack state, collapse, drop, hand detection
  skill/                  # policy runner with termination
  safety/
  game/                   # controller + ui
  notes/
    img/
  data/                   # gitignored
```

---

## 8. Phases

### Phase 0: Environment check
- For each machine: OS, CPU cores, RAM, GPU/VRAM.
- Fresh sim env; `pip install mujoco`; load a model; **render an offscreen frame to PNG**;
  report the working `MUJOCO_GL` backend.
- Look for prior `so101_sim/` work and list what can be reused (validated or not).
- **Checkpoint.**

### Phase 1: Robot model and joint mapping
- Reuse the SO-101 MJCF and `joint_mapping.py` if validated; otherwise get the official
  SO-101 simulation model (TheRobotStudio SO-ARM100 repo; MuJoCo Menagerie as reference)
  and build `real_to_sim()` / `sim_to_real()` with round-trip tests from LeRobot
  normalization code plus `inputs/calibration/my_follower_arm.json`.
- **Recompute the touch points** in `inputs/table_points_20261019_181950.json` with your
  validated mapping and compare with the positions in `measurements.yaml`. Expected
  findings: touch heights should be equal (they spread ~1.5 cm with the plain deg→rad
  mapping), table edges should meet at 90° (measured 96°), and mat sides should match the
  ruler sizes. Report whether any calibration offset explains the residuals.
- Measure the **gripper's maximum opening** in sim and compare with `gripper_open.jpg`.
  Check the grasp clearance for **each** box type's narrowest dimension.
- Visual check: sim arm at my home pose versus `front_cam_frame.png`; plus 2–3 poses I
  reproduce on the real arm.

### Phase 2: Lock the game spec
- With the gripper opening and box measurements, confirm the box size fits (margin
  ≥ ~1.5 cm). If it doesn't, propose box dimensions for me to make.
- Fill every **TODO** in Section 3 with me. Record the answers in the Decisions log.
- **Checkpoint.**

### Phase 3: Scene twin and camera calibration
- Build `sim/scene.xml` from `measurements.yaml`: table (oversized, white, visible area
  matched), black supply mat, yellow stack area, any contrast fix chosen in Section 3a,
  lights, and **one box model per type** (measured dimensions and mass, photo textures
  per type from `inputs/photos/box_faces/<type>/`, Section 3b).
- Cameras: `front` (tripod, ~45°) at measured position and look-at point, 640×480, FOV
  fitted to `front_cam_frame.png`; `wrist` attached to the gripper, fitted to
  `wrist_cam_frame.png`.
- `render_check.py`: side-by-side and 50% blend overlays for the empty base and the
  mixed 3-box stack (`front_cam_stack3.png`). Iterate until edges align.
- Report the size-discrimination check from Section 3d.
- Compute **max reachable stack height** (IK with top-down approach clearance) and
  max height visible in `front`. Report the playable max height.
- **Checkpoint:** I approve the overlays and max height.

### Phase 4: Physics and stack-stability calibration
- **Real protocol (I run it, results in `inputs/stability/overhang_tests.yaml`):**
  overhang tests for each **pairing** listed in that file (same-type stacks for each
  type, large-on-small, small-on-large, a mixed 3-box stack), plus tap and slide tests.
- `stability_check.py` reproduces the same tests in sim. Tune **per-type** mass,
  friction (box–box, box–table, gripper–box), `condim`, solref/solimp and timestep
  until sim topple thresholds match real within ~2–3 mm **for every pairing**. If one
  pairing can't be matched, report it; that pairing becomes a known weak spot.
- Grasp test: 100 randomized picks **per box type** from the supply; ≥ 95% hold for
  2 s at height for each type.
- **Checkpoint:** I approve the stability match. If sim stacks are much more or less
  stable than real ones, the policy learns the wrong level of care.

### Phase 5: Scripted expert, game states and data generation
- `game_states.py` produces randomized starting states:
  - stack height 0..H_max, with **box types drawn per level** from the real box mix
  - **each box offset (±x/y mm) and rotated (± yaw deg)** to mimic imperfect human
    placement, including a share of deliberately precarious stacks that are still
    standing at the start
  - **over-sample hard combinations:** large on small, tall mixed stacks, small top
    surfaces, and stacks whose top box is rotated relative to the one below
  - remaining boxes at randomized poses inside the black-tape supply area (the next box
    to pick is any reachable one; the expert picks the nearest/least obstructed)
- `expert.py` (privileged state + IK):
  1. start at home (± ~3° jitter per joint)
  2. choose the box (Section 3c: deterministic nearest-to-stack rule for ACT data;
     requested type for SmolVLA data); move above it, open gripper, descend, close, lift
  3. move above the **current stack-top pose**, align yaw to the top box. Centre the
     new box's centre of mass over the stack's support, not just over the top box's
     centre; this matters most for large-on-small
  4. **descend slowly**; stop at a small clearance above the top face
  5. open the gripper gently; **retract straight up** before any lateral motion
  6. end the episode once clear of the stack (no return home)
- Motion must be smooth (real arm runs `max_relative_target=10`); slow final approach
  and release are mandatory.
- Randomization per episode: camera pose jitter (front ±1 cm/±1–2°, wrist ± a few mm/±1°),
  lighting, table/background colour around the real values, box texture variation
  (Section 3b), tape colour shades, start pose, optional distractor objects outside the
  play area.
- Keep only successes: box on top **and** stack standing 3 s after release.
- `generate.py` writes raw episodes (both cameras 640×480, state and action in LeRobot
  units via `sim_to_real()`, 30 fps); `export_lerobot.py` converts them to
  `Maxctl352/so101_stack_sim` in the Section 2 format.
- Store per-episode metadata: picked box type, stack composition (types per level),
  offsets, success.
- Generate **20 episodes**, inspect videos and joint plots, and check the distribution
  of stack heights, box types, pairings and offsets. **Checkpoint**, then generate
  **~2,000–3,000** (more weight on hard combinations, taller and crooked stacks).
  Produce two datasets from the same generator:
  `Maxctl352/so101_stack_sim_act` (rule-based choice, single task string) and
  `Maxctl352/so101_stack_sim_vla` (requested type, size named in the task string).

### Phase 6: Train and evaluate in sim
- Train **ACT** in the WSL `lerobot` env on the 3080. Optional later: SmolVLA
  (`lerobot/smolvla_base`, `--rename_map` front→camera1, wrist→camera2,
  `--policy.empty_cameras=1`, `--batch_size=8` or lower).
- `sim_eval.py`: success rate **broken down by stack height, crookedness (max box
  offset), picked box type, and pairing (picked type onto top type)**, plus
  failure-type counts: missed grasp, wrong box picked, drop, knocked stack on approach,
  knocked on release/retract, misplaced. For SmolVLA also report **instruction
  accuracy** (did it pick the requested size?).
- Compare ACT and SmolVLA on the same evaluation set.
- `sim_game.py`: full games in sim, policy vs a scripted "human" that places boxes with
  realistic random error. Exercises the game controller, perception-free (privileged state).
- Iterate data generation toward the weakest buckets until the Section 1 sim criteria are met.

### Phase 7: Classical perception
- Develop on sim renders first, then validate on real frames and the recorded clips.
  - `stack_state()`: must return the **type of each box in the stack**, not just the
    height, because heights differ by type. **Not** plain colour segmentation (the
    boxes are printed and blend into white). Candidate methods, pick by evidence:
    a. difference against an empty-scene reference image, plus the dark mat/backdrop
       (Section 3a) for clean silhouettes; stack height from the silhouette top edge
       in the calibrated `front` camera
    b. a small box detector/segmenter **trained on MuJoCo renders** with randomized
       textures, lighting and backgrounds (keeps everything sim-trained)
    c. fiducial markers (ArUco/AprilTag) on box faces, only if a and b aren't reliable
    Output: box positions, stack height, top-box pose in table coordinates.
  - `collapse_check()`: before/after comparison after the settle window
  - `drop_check()`: gripper position below the "holding a box" width during carry
  - `hand_in_workspace()`: motion or skin detection inside a workspace polygon in `front`
- Exit: on `stack_collapse.mp4` and `human_turn.mp4`, zero false collapse calls on
  stable stacks, collapse detected within the settle window, hand detected whenever it
  enters the workspace.

### Phase 8: Real placement trials (no game yet)
- **Checkpoint before any real motion.** Then give me a trial script: build a fixed
  stack, run the skill from home, judge after 3 s, log results (stack composition,
  picked type, outcome, failure type).
- **Staged order** (isolates calibration problems from size-handling problems without
  a separate single-type version):
  1. same-type stacks of each type, heights 0–3
  2. mixed pairings: small-on-large, then large-on-small
  3. mixed stacks of height 3–4, some slightly crooked
- Ask for ~10 trials per stage bucket. Analyse by bucket and compare with the same
  buckets in sim. If stage 1 already fails, it's calibration (camera, mapping,
  physics), not sizes.
- If real success is far below sim: check camera overlay drift first, then the joint
  mapping, then the physics (stability calibration), then add randomization.
  Real demos are **not** allowed (sim-only constraint); flag to me if it can't close
  without them.

### Phase 9: Game controller, safety and UI
- Implement `game/controller.py`, `skill/` runner with termination plus home before
  and after, `safety/` (e-stop key → torque off; pause skill on hand detection; robot
  frozen during the human turn), `ui/` (whose turn, live front view with detected stack
  overlay, result screen, sounds).
- Dry-run the controller with recorded clips (no robot), then with the robot in a
  "simulate only" mode where the skill runs in MuJoCo but perception runs on the real camera.

### Phase 10: Play and log
- Real games against people. Log every turn: stack height before and after, placement
  offset, outcome, failure type, timings. Report win rate, false-call rate and the
  robot's failure modes. Feed the weak buckets back into Phase 5 generation.

### Phase 11 (optional): Strategic robot
- Before each robot turn, simulate candidate moves (**which box size**, centred versus
  offset placement) on a MuJoCo replica of the **perceived** stack, and choose the one
  that keeps the stack stable now while leaving it harder for the next placement
  (e.g. finish with a small top surface). With SmolVLA the size choice is passed as
  the instruction.
- Needs either a **goal-conditioned** policy (target offset as an extra input) or a
  model-based final placement (IK to the chosen pose) after the learned pick. Propose
  the design before building.

---

## 9. Known pitfalls

- **Joint mapping errors look like a bad policy.** Validate Phase 1 first.
- **Camera mismatch is the biggest real-world failure mode** (proven on this arm). The
  tripod must not move between calibration and play; keep a reference frame and check
  it before each session.
- **Stability mismatch:** if sim boxes are too sticky or too heavy, the policy learns
  careless placements that topple real stacks. Phase 4 is not optional.
- **Release and retract** are where stackers knock stacks over. Over-sample tall stacks
  and slow the expert near the top.
- **Mixed sizes:** stack-top height is no longer a multiple of one box height, so the
  policy must perceive the top (wrist camera). Large-on-small is the least stable
  pairing; make sure sim matches real there (Phase 4).
- **ACT multimodality:** random box choice in demos makes ACT average between boxes.
  Keep the expert's choice deterministic for ACT data (Section 3c).
- **Size discrimination at low resolution** (Section 3d): No.3 vs No.35 may be
  ambiguous in the `front` view at policy resolution.
- **Gripper vs box size:** too tight a fit means pushing the box on release; too loose
  means rotation in the grasp.
- **Sim-only visual gap:** MuJoCo rendering isn't photoreal; lean on randomization and
  careful camera/colour matching.
- **Staple boxes:** if partly full, staples shift inside and change the centre of mass;
  keep every box consistently full or empty. Cardboard flexes under the gripper; sim
  boxes are rigid, so don't grip harder than needed.
- **Low contrast with the white table** (Section 3a) is the first thing to fix. Don't
  calibrate cameras or generate data before the real scene's contrast setup is final.
- **False collapse calls** ruin the game. Prefer conservative thresholds plus the settle window.

---

## 10. Decisions log

Append entries as `YYYY-MM-DD: decision, reason, evidence/link`.

- 2026-10-19: Simulator = MuJoCo; all policy training in sim (user constraint). Real setup used only for calibration and evaluation.
- 2026-10-19: Architecture = learned place skill + classical game logic and perception; no end-to-end game learning.
- 2026-10-19: Camera names fixed as `front` (tripod ~45°) and `wrist` for compatibility with existing tools and the SmolVLA camera mapping.
- 2026-10-19: Human turn ends on keypress (initial); robot frozen with torque on during the human turn.
- 2026-10-19: Arm is mounted at **one corner** of the 38×38 cm table; its +x axis runs ~7° off the table diagonal. Measurement frame: origin on the shoulder_pan axis at table height, +x = arm direction at shoulder_pan = 0, +y = arm's left. Positions from gripper-tip touch points + FK (±1.5 cm); mat sizes from the ruler. Supply = black mat (arm's left), stack = yellow area (arm's right), confirmed.
- 2026-10-19: **Mixed box sizes from the start** (No.3 / No.35 / No.10), no single-type version (user decision). Mitigation: every evaluation split by type and pairing; real trials staged same-type → mixed pairs → mixed stacks. ACT uses a deterministic box-choice rule; SmolVLA gets the size as an instruction.
- 2026-10-19: Boxes are printed stapler-staple boxes. Supply = black-tape area; stack area = yellow-tape area (to confirm). Boxes contrast poorly with the white table, so add a dark mat/backdrop (Section 3a) and use photo textures plus randomization in sim (3b). Perception won't rely on plain colour segmentation.
- 2026-10-20: Sim env = separate uv venv `~/.venvs/so101_stack_sim` (CPython 3.12.6, MuJoCo 3.16.0; pins in `scripts/requirements-sim.txt`), kept apart from the `lerobot` conda env. Evidence: `notes/phase0.md`.
- 2026-10-20: Mac offscreen rendering uses `MUJOCO_GL=cgl` (valid Darwin backends cgl/glfw per `mujoco/rendering/classic/gl_context.py:24-53`; 640×480 at ~1.9 ms/frame). Data generation and rendering run on the Mac; dataset export and training run in WSL. Evidence: `notes/img/phase0_render_cgl.png`.
- 2026-10-20: No prior `so101_sim/` found on the Mac, so nothing is reused as validated; Phase 1 builds the model and joint mapping from the official SO-101 sources unless the user finds it on the GPU machine.
- 2026-10-20: Project repo = `~/so101_stack_game` (own git repo; the stray home-dir repo is left alone, user cleans it up). Remote = private `ka1dus/so101_stack_game` (user choice). First commit waits until the user adds `so101_start_poses.py` and `so101_grasp_poses.py`.
- 2026-10-20: Robot model = official TheRobotStudio SO-ARM100 `Simulation/SO101` "new_calib" MJCF, commit `a758567`, copied unmodified to `sim/assets/so101/`. No prior `so101_sim/` exists on either machine (user confirmed). Sim limits are narrower than the real calibrated ranges; widen them in our scene, never edit the vendor file. Evidence: `notes/phase1.md` §1–2.
- 2026-10-20: Gripper mapping `q_sim = radians(-10.75 + 1.3029·g)`: slope from calibration (jaw on the servo horn), closed angle fitted to the closed + 3 box-width ruler readings (rms 1.7 mm; open tip gap 154 vs 150 mm). Official jaws match the real ones. Evidence: `notes/phase1_logs/phase1_gripper_fit.txt`.
- 2026-10-20: Body joint mapping = LeRobot formula (verified against LeRobot's own `_normalize` on every tick) with per-joint offsets, **provisionally 0** (`sim/assets/joint_mapping.yaml`, status provisional). Touch data suggests a ~1.4° base tilt and lift ≈ −7°, elbow ≈ +5° offsets, plus a ±90° wrist_roll offset from the home-pose image. Not applied until the user's M1–M4 measurements (`notes/phase1.md` §7).
- 2026-10-20: Phase 1 M3 fit: shoulder_lift −5.6 ± 2.5°, elbow_flex +6.8 ± 2.2° offsets, base tilt −1.1° / +0.5°; confirmed on held-out data (home tip 25.8 vs 24 mm measured; old touches give a 89.8° table corner and ruler-matching mat sizes). wrist_roll (+95 or +5) and wrist_flex (+4.6 / +8.0) wait for the M4 photos. Touch contact = lowest vertex in the last 9 mm of a jaw. All `measurements.yaml` positions (plain mapping) are ~3 cm too close to the arm and must be recomputed in Phase 3. Evidence: `notes/phase1.md` §9.
- 2026-10-20: Phase 1 closed: joint mapping validated (`sim/assets/joint_mapping.yaml`). wrist_roll offset +95.4° from the M4 front photos (+5° rejected); roll direction matches sim (+45° command → ~45° CCW from above; no sign flip); wrist_flex +4.61°. Accuracy ≈ ±2.5° per joint (~1 cm at the tip). Evidence: `notes/phase1.md` §10.
- 2026-10-21: Phase 2 game spec: boxes **all full**, **always lying flat** (supply and stack), **human starts** every game; box fit confirmed (≥ 95 mm spare for every type). Max height stays open until Phase 3. Evidence: `notes/phase2.md`.
- 2026-10-21: Contrast fix (Section 3a) = **matte black backdrop** standing on the table's right edge (arm corner → yellow-side corner), about 10 cm from the arm's corner to the corner plus a ~5 cm wrap, ≥ 20 cm tall. Reason: from 4 boxes up, the front view sees the top box against the white panel (and the human). Sizing: `notes/phase2_background_check.txt`. The sim models it; real reference frames are retaken after it's installed.
- 2026-10-21: Phase 3 (in progress): layout recomputed with the validated mapping (mats moved ~3 cm outward); table placed from front-image corner picks (edge touches sit inside the edge). Front camera calibrated: 2.73 px rms, f 656 px, k1 −0.139 (k2 0); proposal: undistort real front frames before the policy. Wrist camera NOT calibrated (silhouette fits failed; needs checkerboard intrinsics; measured lens offset looks wrong). SmolVLA in the installed LeRobot resizes to 512×512 padded (not 256). Evidence: `notes/phase3.md`.
- 2026-10-21 (CORRECTED same day): wrist_flex hits its servo-enforced calibrated max (+101.9°, written to the motor's position limits, `feetech.py:272-273`), so jaw tilt over the stacking spot grows with stack height. The first claim, "only onto ≤ 3-box stacks", was WRONG: it combined a 3 cm vertical-approach requirement with worst-case ±2 cm offsets. Verified by an exhaustive lift×elbow sweep (`notes/phase3_reach_bruteforce.txt`), no riser, place pose only: onto 6 boxes at ≤ 10° tilt, 7 at ≤ 15°, 8 at ≤ 20°; one box more with the spot moved to 23 cm. A riser is optional margin, not required. ~14 % of the supply mat (r > 31.6 cm) isn't vertically pickable.
- 2026-10-21: Riser recommendation (8 cm) WITHDRAWN after the correction above; riser heights 6–9 cm showed no arm–table/backdrop contact (`notes/phase3_riser_check.txt`) if margin is ever wanted. Decision on riser / spot / release-tilt tolerance pending with the user.
- 2026-10-21: Stacking spot moved to **23 cm from the pan axis**, (20.9, −9.7) cm, 2.5 cm toward the arm from the original X (user decision). No riser. Last robot placement in a full game needs ≈ 10–11° release tilt; lower ones ≤ 6°. Backdrop still covers the stack sightlines. The real X is re-marked by ruler and checked with the hover poses in `scripts/phase3_spot_check.sh`. Evidence: `notes/phase3.md` §11.
- 2026-10-21: Layout = per-feature average of the 19 Oct and 21 Oct touch sessions (base confirmed unmoved: 0.36° / 2 mm rigid fit); the 21 Oct `stack_corner_2` is excluded (17 mm below the table). Stacking spot = the touched new X, (20.6, −10.1) cm, r 22.9 cm. Front camera re-fit 2.60 px. Evidence: `notes/phase3.md` §13.
- 2026-10-21: Wrist camera = Orbbec Gemini 2 (user): housing 90×25×30 mm, 98 g; RGB 4:3 FOV 63×50° → f ≈ 518 px at 640×480 (datasheet). The measured lens offset (3.6 cm from the fixed tip) contradicts the wrist image (tip size implies 10–14 cm); re-measurement requested. Backdrop measured 18.5 cm tall × 28 cm, from 10 cm along the right edge to the corner, no wrap; mat thickness negligible. Evidence: `notes/phase3.md` §15–16.
- 2026-10-21: Wrist camera (provisional): f 518 px (datasheet), mount fitted from jaw tips + P1/P3 table features with the measured lens position as prior (`scripts/wrist_fit_scene.py`); jaw IoU ≈ 0.7 in home / P1 / P3. The fit puts the lens ~5–6 cm out of the jaw plane vs the user's 2 cm (asked to re-check). Vendor camera mount removed; Gemini 2 housing body (98 g) added. Checkerboard to be captured with the arm at `R_raise`. Evidence: `notes/phase3.md` §17–18.
- 2026-10-21: Wrist checkerboard calibration **skipped** (user decision; rated low priority). Wrist intrinsics = Gemini 2 datasheet f 518 px, k1 0. Mitigation: Phase 5 randomizes wrist f ± 5 % and |k1| ≤ 0.05 in addition to the pose jitter.
- 2026-10-21: Wrist lens position = fitted (≈ 9 cm from the jaw tips, ≈ 6 cm out of the jaw plane), not the ruler's 2–3 cm: with the lens fixed at the measurement no orientation reproduces the home jaw tips (18–21 px) or the P1/P3 table features (140–240 px); the fit gives 3–4 px / 16–32 px and jaw IoU ≈ 0.7. Wrist size check: No.3 vs No.35 differ by 41 × 17 px at pre-grasp (front only 4–6 px). Max game height = full 8 boxes, no riser. Evidence: `notes/phase3.md` §20–21 and checkpoint summary.
- 2026-10-21: Phase 3 approved by the user (overlays, full 8-box max height). Next: Phase 4.
- 2026-10-21: Phase 4 physics (`sim/assets/physics.yaml`): measured per-type box friction (max rule), tape/mat/table with priority 1, elliptic cone, impratio 10 (stops in-grasp creep), MuJoCo default softness. Overhang within ≤ 1.5 mm of real for every pairing (large-on-small +1.3); slide within 0.2°. Tap order mismatched (sim: tapped box slides out first) — open. Gripper: finger box colliders along the real inner faces (mesh hulls fill the jaw gap), torque cap 0.75 N·m (LeRobot 25 % overload limit). Grasp test 100/100 per type with a straight-down Cartesian descent (the Phase 5 expert must use it). Vertical picks reach only ≤ 29 cm → 71 % of the supply mat. Evidence: `notes/phase4.md`.
- 2026-10-21: Tap test reconciled: with sustained (≥ 300 ms) pushes, sim stack B tips over its base (0.35 N) and slides (0.40 N) close together, as the user observed; sim A (0.20 N) remains a little easier than real (recorded, cautious side). Supply reach: a white tape line across the black mat at the 29 cm reach limit (ends 14.7 cm from corner 1 on edge 1→2 and 13.0 cm from corner 4 on edge 4→3), no re-touch (user decision); boxes stay on the arm side (~66 % of the mat); modelled visually in sim; Phase 5 supply sampling respects it.
- 2026-10-21: Phase 4 approved by the user (stability match, grasp test). Next: Phase 5.
- 2026-10-21: Supply re-evaluated after user feedback: the earlier "supply can't hold the boxes" finding was wrong (random dense layouts, 22 mm full-length jaw clearance). Jaw clearance = 10 mm each side within ±1 cm of the box middle, pre-grasp opening width + 6 mm (to confirm in the expert grasp test). Decision: expand the black mat by a 2 cm strip along the corner-1/corner-4 edge toward the arm (new size 22.9 × 19.2 cm); simulated games then never leave the robot without a pickable box (0/300). Supply corners re-touched (`--points supply`). Evidence: `notes/phase5.md` §3–4.
- 2026-10-21: Phase 5 expert (`scripts/expert.py`): place xy maximizes the worst centre-of-mass support margin over all levels; release 3 mm above the top, slow last 2 cm, straight-up retract. The place IK tries both wrist directions, rejects poses < 10 mm (wrist/gripper/camera) or < 3 mm (shoulder/upper/lower arm) from the backdrop, and puts position first (orientation weight 100 → 3). Any arm–backdrop contact fails the episode. Held-out: 98/100 (2 × backdrop touch at h ≥ 6). Evidence: `notes/phase5.md` §5.
- 2026-10-21: Moving-jaw side = the feasible side whose wrist roll is nearest the place roll (−12° sim), so pick rolls never wrap at ±180° (ACT multimodality); jittered home clipped to the calibrated ranges. Evidence: `notes/phase5.md` §8.
- 2026-10-21: Sim arm recoloured black (the real arm is black; the vendor model is yellow). Room wall added behind the arm in the `front` view (light lower panel, dark blue wall; APPROX, randomized). Visual randomization in `scripts/randomize.py` (Phase 5 ranges plus image-space colour/blur/noise; real front frames are undistorted, so sim front frames get the same corner mask). Evidence: `notes/phase5.md` §6.
- 2026-10-21: Raw episode format (`scripts/generate.py`): front/wrist mp4 640×480 at 30 fps, `data.npz` state/action in LeRobot units, `meta.json`. action[t] = targets of the tick after observation t. The VLA dataset reuses ACT episodes (their size is known). Export (`scripts/export_lerobot.py`) uses LeRobot's `hw_to_dataset_features`; checked with the Mac LeRobot 0.6.2, must be re-run in the WSL training env. Pilot 20/20: `notes/phase5.md` §8.
- 2026-10-21: Large generation (approved): weighting hard 0.6, heights 4–7 at 60 % (`generate.py --hard-p/--tall-p`). Started on the Mac (4 workers, ~26 s/episode each); stopped after 326 ACT episodes (user: too slow) and moved to native Windows on the GPU machine with `scripts/run_generation.py`. The work is CPU-bound, so speed scales with cores, not the GPU. Instructions: `notes/phase5_gpu_generation.md`.
- 2026-10-22: **GPU machine corrected** (checked over SSH): AMD Ryzen 5 5600X (6 cores), 16 GB RAM, **RTX 3060 Ti 8 GB** (not a 3080), Windows 11 Home, **no WSL** (the "working lerobot env in WSL" did not exist). Training runs in **native Windows**: Miniforge env `lerobot` (Python 3.12.15, torch 2.11.0+cu128, LeRobot ca69a206 `[training]`, ffmpeg 7.1). SSH from the Mac with a dedicated key; Tailscale installed (unattended) for remote access. Evidence: `notes/phase6_overnight.md`.
- 2026-10-22: Dataset generation complete: 2,000 ACT + 1,000 VLA raw episodes (~5 % of attempts discarded, mostly backdrop touches). Export with LeRobot streaming encoding (`export_lerobot.py --streaming`, 3.7 s/episode vs 20 s default), 4 parallel parts merged with LeRobot `merge_datasets` (`scripts/merge_lerobot.py`). Datasets local on the PC, not pushed yet (needs `hf auth login`).
- 2026-10-22: ACT training v1: LeRobot defaults + AMP, batch 8, 100k steps, checkpoint every 10k (`C:\Users\User\so101_runs\act_v1`). Sim evaluation runs on the Mac in a separate venv `~/.venvs/so101_eval` (MuJoCo + LeRobot; `scripts/sim_eval.py`); eval states that drift at rest (> 0.5 mm in 2 s) are skipped.
- 2026-10-22: Found: rare slow creep of a resting stack's top box in sim (one state drifted 10 mm in 12 s with nothing touching it; 0/40 fresh states > 5 mm, 1/40 > 1 mm). Not seen in Phase 4's short tests. The expert's "stack moved" check starts at release, so a few training episodes may contain drift. Open: tighten `stands()` (longer rest check) for the next generation.
- 2026-10-22: ACT v1 (100k steps, 1.14 epochs) in sim: ~40 % at default inference (plateau from 40k on); **temporal ensembling 0.01 chosen for inference** (56.7 % vs 41.7 % on the same 60 states; re-planning every 10 or 25 steps without ensembling was worse). On 200 new states (seeds 1001–1004, the fixed comparison set from now on): **49.0 %** (0–3: 37 %, 4+: 57 %, large-on-small 28 %); 92 of 102 failures are missed grasps (aim error ~1–2 cm, worst with 5–6 boxes left in the supply); once a box is held, ~90 % are placed correctly. No sim criterion met yet. Proposed v2: forgiving grasp, recovery (DART) and retry demos, longer training. Evidence: `notes/phase6.md`.
- 2026-10-22: Data/training v2 decisions (user, on `notes/phase6_data_v2_plan.md`): fresh v2 dataset (v1 not reused); state weighting shifted toward dense supplies (stack heights 2–3 = 35 %; supersedes "more weight on taller" for v2); run the A/B pilot (600 episodes, 30k steps) first; test half-resolution training in the pilot. **New game rule accepted: the human keeps ≥ 2 cm gaps between supply boxes.** The user will also set up a fixed max-gap supply layout if one exists (investigated next).
- 2026-10-22: Work split (user approved): **`botmind`** (RTX 4070 Ti SUPER 16 GB, i7-14700KF 20 cores, Windows, Tailscale 100.101.192.30; offscreen MuJoCo rendering works over SSH, 1.3 ms per frame pair; 16 workers → 2.2 s/episode) does v2 data generation, sim evaluation and the main training. The 3060 Ti PC trains the second pilot variant. The Mac builds code and orchestrates. Evidence: `notes/phase6.md` §7.
- 2026-10-22: Expert v2 built (`expert.py` kinds clean/dart/retry; centred adaptive grasp + deep grip chosen from a 3-variant test, 119/120; hover + slow final descent; DART OU noise σ 2°/joint with clean-plan labels via `Sim.clean_ctrl()`; retry starts after an unrecorded miss). Pilot: 600 v2 episodes (botmind), 30k steps full-res (botmind, 0.119 s/step) and half-res 320×240 (3060 Ti, 0.084 s/step). An evaluator bug (stale plan replayed during the pre-episode rest check, introduced with DART) invalidated the first pilot comparison; fixed, re-evaluating. Evidence: `notes/phase6_data_v2_plan.md` §12–13.
- 2026-10-23: CORRECTION to the supply-layout results of 2026-10-22: those layouts let boxes overlap the 1.7 cm reach tape (the search used the tape's centre line). With the tape respected, 8 boxes fit with at most ~11–13 mm beside long sides; **the "≥ 2 cm gaps" rule is not achievable with all 8 boxes**. Chosen starting layout (pending user OK): `sim/assets/supply_layout.yaml`, ≥ 11.2 mm beside long sides, ≥ 10 mm all round, 13.1 mm pick-order margin, placed with a template (±2 mm). Evidence: `notes/phase6_data_v2_plan.md` §14.
- 2026-10-23: **Game starting layout = the template layout** (`sim/assets/supply_layout.yaml`; user approved): ≥ 11.2 mm beside long sides, ≥ 10 mm all round, 13.1 mm pick-order margin; boxes placed with a printed template (±2 mm, ±3°). Replaces the "≥ 2 cm gaps" rule. Full v2 run approved: 3,000 episodes with game-rule supplies (`SUPPLY_P_GAME`: 50 % layout progression, 30 % rule-spaced random, 20 % dense), half resolution 320×240, 300k steps on botmind, evaluated on the v1 200-state set (seeds 1001–1004) and a new game-rule 200-state set (`--states game`, seeds 3001–3004).
- 2026-10-23: **ACT v2 results** (3,000 v2 episodes, 320×240, 300k steps on botmind, temporal ensembling): v1 200-state set **72.0 %** (v1 policy 49.0 %); game-rule 200-state set **93.0 %** (height 0–3: no3 100 %, no35 94 %, no10 90 %; crooked 4+ 15/16; large-on-small 12/12). The sim criteria are met at point estimate on the game-rule set but the crooked-4+ and large-on-small buckets are too small (3–12 episodes) to confirm; a larger targeted evaluation is next. Evidence: `notes/phase6_data_v2_plan.md` §16.
- 2026-10-23: Confirmation evaluation of ACT v2 (1,060 game-rule episodes incl. targeted buckets): height 0–3 **confirmed** for every type (97.5 / 95.1 / 97.9 %, lower 95 % bounds ≥ 91 %); large-on-small **confirmed** (84.2 %, 78–89); crooked 4+ no10 confirmed (89.9 %), no3 70.3 % and no35 71.8 % meet 70 % only at point estimate (lower bounds 61–62 %). Weak spot: large/medium boxes onto tall (6–7) crooked stacks, mostly knocks on approach/release. 3 backdrop touches in 1,060. Evidence: `notes/phase6_confirmation.md`, `notes/phase6_data_v2_plan.md` §17.
- 2026-10-23: Expert fixed for tall stacks (backdrop-safe move ordering, carry rises ≥ 19 cm from the pan axis, release raised by the held box's lowest corner): targeted expert success no3 87 → 93 %, no35 80 → 100 %. **ACT v3** (v2 300k + 60k fine-tune on 3,000 v2 + 1,000 targeted episodes, `C:\Users\botmi\so101_runs\act_v3_ft`): all Section 1 sim criteria **met**, all **confirmed** (lower 95 % bound ≥ target) except crooked 4+ for no3 (74.5 %, 65–82 %); game-rule set 97 %; 0 backdrop touches. Evidence: `notes/phase6_confirmation_v3.md`, `notes/phase6_data_v2_plan.md` §19–20.
- 2026-10-23: Phase 7 perception decisions (user): **D1** box types per stack level come from game history plus a supply-slot occupancy check in `front` (every game starts in the template layout; boxes stay in their slots until taken; a bumped box is re-seated), not from the stack's appearance (type heights differ by only 1–4 px). **D2** stack height / collapse use method a (difference against an empty-scene reference + backdrop) first; a sim-trained segmenter (b) only if (a) fails its tests. **D3** hand detection during the robot turn masks the arm with the sim twin rendered at the current joint state from the calibrated `front` camera (dilated margin, tuned on the recorded clips). Evidence: `notes/phase7.md`.
- 2026-10-23: Drop rule (user): a box that slips out of the gripper **right over the stack** (jaw tips within 4 cm of the stack top centre and < 7 cm above its face) is **not** called a drop; the collapse check judges the result (box on the stack = turn continues; box on the table = collapse). Anywhere else, the box leaving the jaws after a 3 cm lift is a drop. Sim: `drop_check()` 0 false calls / 367 normal turns, 223/233 drops called (the other 10 slid out over the stack and go to the collapse check), median 0.13 s. Evidence: `notes/phase7.md` §9.
- 2026-10-26: **Backdrop trimmed** (user, option A): its arm end now starts ~16 cm (was 10 cm) along the right edge from the arm's corner, because the shoulder bracket hit it when panning toward the supply (sim contact from 50° pan). Sim updated (`build_scene.py`, `scene.xml`): ≥ 39 mm arm clearance at every pan. ACT v3 on the same 460 states: 417 → 409 (noise between identical runs: 4 flips; trim: 18), all sim criteria still met; fine-tune on trimmed-board renders before Phase 8 if wanted. Evidence: `notes/phase7.md` §13.
- 2026-10-26: Phase 7 perception on the real recordings (current setup): camera refit agrees within 3.1 px between the two reference photos; supply slots 8/8 on both; collapse 11/11 clips (0 false calls on 3 static stacks with a person walking by, 4 human placements standing, 4 losses called). Fix: `stack_region` now keeps only the table top below 4 cm plus an 8 cm column at the spot (it looked past the table edge into the room); sim collapse set unchanged (112/126, 0/274). Hand: 8/8 caught; nuisance flags from a person walking by and from the camera's auto exposure (lock it if possible). Evidence: `notes/phase7.md` §14.
- 2026-10-26: Phase 8 tools: `skill/runner.py` (ACT in our own 30 Hz loop on MPS, 26 ms/step; ±10-unit clip, calibrated ranges, jaw-tip safety envelope, sim_game termination, scripted home with lift) and `scripts/phase8_trial.py` (planner for the ACT pick rule, perception call plus the user's call, JSONL log, `--report`). Real-arm runs use `~/.venvs/so101_eval` (feetech-servo-sdk 1.0.0 added); the conda `lerobot` env's MuJoCo 3.8.1 cannot load the scene and is left alone. Trials start at height 1 (training never placed onto the empty base). ACT v4 (v3 + 40k steps on 5,500 episodes incl. 1,500 with the trimmed backdrop) training on botmind. Evidence: `notes/phase8.md`.
- 2026-10-26: Camera refit gets a weak prior toward the nominal calibration (`PRIOR_W`): repeatable on real images (4.8 mm apart vs 27 mm), sim worst point error max 41 → 7 px, supply false "full" 9 → 0, collapse unchanged. **ACT v4** (trimmed-backdrop fine-tune) 416/460 on the v3 wave-1 states (v3: 409 on the trimmed board, 417 on the old one); all sim criteria met; v4 is the policy for the Phase 8 real trials. No-motion real run passed (29.2 Hz loop). Evidence: `notes/phase8.md` §5–6.
- 2026-10-27: Phase 9 built (`game/controller.py`, `game/backends.py`, `safety/monitor.py`, `ui/display.py`, `scripts/play_game.py`, `scripts/test_safety.py`): hand pause with 1 s resume, e-stop (hold + torque off), settle restarts on a hand, box type from the slot check (asked when unsure). Added after the first sim batch: (1) the game asks "is the new box on the stack?" (y/n) when the stack top does not rise by half a box height (the `front` view cannot tell a box on the top from one on the table in front of the stack; a camera-only rule gave false calls); (2) a 20 s watchdog on every wait for a clear workspace (a sim game had hung for 9 h). 10 sim games with both: 75/75 turn calls, 0/66 false loss calls, 10/10 games right; the robot succeeded onto 7-box stacks only 1/7 (weak spot). Evidence: `notes/phase9.md`.
