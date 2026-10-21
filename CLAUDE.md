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
