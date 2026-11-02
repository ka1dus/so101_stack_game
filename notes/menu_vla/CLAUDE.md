# SO-101 Menu VLA (MuJoCo, sim-trained): Project Brief for Claude Code

Read this whole file before doing anything. It is the source of truth for this
project. Record every decision in the **Decisions log** at the bottom so later
sessions don't re-derive or contradict it. When resuming, read the newest
`notes/phaseN_status.md` handoff file first.

---

## 1. Goal

Tell my SO-101 arm what to do in plain language and have it do it on the same
table as the stacking game:

> "Rebuild that stack in reverse order on spot B."
> "Take the marker off the large box, then put the large box on stack A."
> "Sort the boxes by size into one stack, largest at the bottom."

The system has three parts:
1. **Planner:** a language model turns the command plus a scene map into a short
   sequence of menu actions (Section 3), and re-plans when a step fails.
2. **Menu VLA:** one language-conditioned policy (SmolVLA, fine-tuned in sim) that
   performs the learned part of each action.
3. **Classical layer:** perception (scene map), IK placement, per-step checking,
   safety. Reused from `so101_stack_game` wherever possible.

**Hard constraint (kept from the stacking game):** all robot training data is
generated in **MuJoCo simulation**. No real demonstrations. Pretrained models are
allowed (SmolVLA base, open-vocabulary detectors, segmenters, grasp networks,
language models), since they are not demonstrations of this robot. The real setup
is used for calibration and evaluation only.

**Core design principle:** learn only what must be learned. Precise placement stays
computed (IK, as in the stacking game's `ScriptedPlace`); the VLA picks, carries and
pushes. Composite tasks come from the planner, never from extra training.

### Objects
- **The 8 boxes** from the stacking game (3 × No.3, 4 × No.35, 1 × No.10), lying flat.
- **Foreign objects, including never-seen ones**, only in this accepted class (user
  decision): **small, rigid, light, with a clear narrow side** (tape roll, marker,
  small toy, eraser, cup grasped from above). Out of scope, the robot says "can't":
  flat (phone, card), transparent or shiny, heavy, needing a side grasp, out of reach.

### Success criteria
1. **Sim, per action type** (Section 3), on held-out scenes and held-out object meshes:
   box actions ≥ 90 % per step; object pick ≥ 80 %; object onto box/stack ≥ 60 %;
   instruction accuracy (right object chosen) ≥ 95 %.
2. **Real, per action type:** box actions ≥ 80 % per step; accepted-class objects
   (10 real objects never used in sim) ≥ 60 % per pick.
3. **Real composite tasks:** reverse a 4-box stack, copy a stack, sort 4 boxes,
   clear 2 objects then stack: each finished in ≥ 4 of 5 runs, with retries allowed,
   no unsafe motion, and every failed step detected by the checker.
4. Every evaluation is **broken down by action type, object type (box type or
   object category) and seen vs never-seen object**.

---

## 2. What already exists (reuse, do not redo)

Everything below lives in `~/so101_stack_game` (read its `CLAUDE.md` Decisions log
and `notes/phase*.md`). Treat it as a read-only dependency: copy or import, never
edit it from this project.

### Machine names (strict; use exactly these everywhere)
- **`botmind`**: RTX 4070 Ti SUPER PC.
- **`siix-pc`**: RTX 4000 Ada PC (added 2026-11-02).
- **`home pc`**: RTX 3060 Ti PC (its Tailscale address stays `so101-gpu`).
Never write "second PC", "new PC", "3060 Ti PC" or "GPU PC" for these.

### Hardware (same setup, unchanged)
- SO-101 follower arm at one corner of a 38 × 38 cm white table; black supply mat
  (arm's left), yellow stack area (arm's right), matte black backdrop on the right
  edge, white reach-limit tape on the mat. No leader arm.
- Cameras `front` (tripod, ~45°) and `wrist` (Orbbec Gemini 2, **RGB-D**; depth is
  unused so far). Keep the names.
- Mac (Apple Silicon): arm and cameras; all real runs. **Claude never moves the real
  robot**; it gives the user exact commands.
- `botmind`: Windows 11, RTX 4070 Ti SUPER 16 GB, i7-14700KF 20 cores, 32 GB RAM,
  Tailscale 100.101.192.30, SSH key `~/.ssh/so101_pc_ed25519`, user `botmi`.
  Data generation (16 workers), sim evaluation, pilots. Start long jobs detached
  (processes started inside an SSH session die when it closes).
- siix-pc (added 2026-11-02): Windows, RTX 4000 Ada 20 GB, Core Ultra 7 265K 20 cores,
  64 GB RAM. Data generation (16 workers), pilots, evaluation, targeted fine-tunes.
  Not set up yet: `lerobot` env, SSH/Tailscale, headless MuJoCo check, speed test.
- home pc (RTX 3060 Ti 8 GB, 6 cores, Tailscale `so101-gpu`): extra data generation / eval only.
- No cloud GPU (2026-11-02): main training runs on siix-pc (Section 7).

### Validated pieces to reuse
| Piece | Where (`so101_stack_game/`) | Status |
|---|---|---|
| SO-101 MJCF + joint mapping (offsets, gripper fit) | `sim/assets/`, `scripts/joint_mapping.py`, `sim/assets/joint_mapping.yaml` | validated ±2.5°/joint |
| Calibrated scene twin (table, mats, backdrop, wall) | `scripts/build_scene.py`, `sim/scene.xml` | validated |
| Cameras (front fit 2.6 px; wrist ChArUco, f 508 px) | `sim/assets/cameras.yaml`, `perception/camera.py` | validated |
| Box physics (overhang within 1.5 mm per pairing) | `sim/assets/physics.yaml` | validated |
| Soft-servo model + sag compensation | `sim/assets/servo_fit.yaml`, `scripts/expert.py` | validated |
| Scripted expert (pick, stability-aware place, DART/retry) | `scripts/expert.py`, `scripts/game_states.py` | validated for supply→stack |
| Episode writer, LeRobot export, merge | `scripts/generate.py`, `export_lerobot.py`, `merge_lerobot.py` | validated (LeRobot ca69a206) |
| Visual randomization | `scripts/randomize.py` | validated |
| Scripted place (box-in-gripper, IK, sag, touch stop) | `skill/scripted_place.py` | validated real 18/18 |
| Wrist look at a stack (two looks agree ≤ 6 mm) | `skill/stack_look.py` | validated real |
| Turn judge (`turn_check`), supply slots, hand detection | `perception/stack.py`, `supply.py`, `hand.py`, `twin.py` | validated real |
| Supervised camera processes | `skill/cameras.py` | validated (unplug test) |
| Policy loop, safety envelope, e-stop | `skill/runner.py`, `safety/monitor.py` | validated |
| ACT wcal60 (box pick, fallback baseline) | `data/policies/act_wcal_060000` | real 30/30 with the recipe |

### Lessons to design around (proven on this arm)
- The camera setup is part of the model: background clutter behind the arm made a
  policy hover 1 cm short. Randomize the background (images/text) in all new data.
- Real servos are soft (sim servos were 30–75× too stiff); always use the fitted
  servo model and compensation.
- The learned policy is accurate to ~1–3 cm at release; keep precise placement computed.
- The `front` view can't tell box types apart by height (1–4 px); the wrist camera
  can (No.3 vs No.35: 41 × 17 px at pre-grasp).
- The Orbbec resets on an underpowered USB port; use a powered hub before relying on depth.
- False "failure" calls ruin the experience; the checker asks the user when unsure.

### Units and format (must match LeRobot)
Same as the stacking game: joint order `shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll, gripper`; body joints in degrees, gripper 0–100; 30 fps;
`observation.images.front`, `observation.images.wrist`; `robot_type so101_follower`.
New: **one task string per episode naming the action and object**, generated from
templates with paraphrases (Section 4).

---

## 3. Action menu

How: **VLA** = learned in sim; **VLA + IK** = the VLA picks/carries, computed IK does
the final placement; **Classical** = planner/perception/checks.

### Box actions
| # | Action | Example | How |
|---|---|---|---|
| A1 | Pick a box by size or position | "pick up the large box", "the box nearest the stack", "the leftmost box" | VLA |
| A2 | Place on top of a stack | "put it on stack A" | VLA + IK |
| A3 | Place at a named spot | spot A / B, set-aside zone, back in the supply | VLA + IK |
| A4 | Unstack the top box | "take the top box off stack A" | VLA |
| A5 | Move a box (pick + place) | "move the small box to spot B" | VLA + IK |
| A6 | Relative place | "next to / left of / right of / in front of the X" | VLA + IK |
| A7 | Rotate / align (box stays flat) | "turn it 90°", "line it up with the box below" | VLA + IK |

### Object actions (accepted class only)
| # | Action | Example | How |
|---|---|---|---|
| O1 | Pick an object by description | "the marker", "the red toy", "the thing on the box" | VLA; depth grasp planner as fallback |
| O2 | Place an object at a spot / next to something | "put the tape next to stack A" | VLA + IK |
| O3 | Clear a box (lift object off, set aside) | "take the eraser off the medium box" | VLA + IK |
| O4 | Put a flat-bottomed object on a box/stack | "put the eraser on the stack" | VLA + IK (lower target) |
| O5 | Push aside (anything not graspable) | "push that out of the way" | VLA |

### Support actions (classical)
| # | Action | How |
|---|---|---|
| S1 | Look / scan (wrist look at a stack or object, front-view scene map) | Classical |
| S2 | Check the step, re-plan on failure, say "can't" (unreachable, ungraspable, out of class), go home, stop | Classical |

### Composite tasks (planner only, no training)
Reverse a stack, copy a stack (same order or reversed, with per-level offsets/yaw),
sort by size, clear the table, swap two stacks, build a stack in a given order,
"put all the toys in the set-aside zone".

### Named locations (fixed in sim and on the real table)
Stack spot A (the stacking game's spot), stack spot B (in the yellow area, ≥ 10 cm
from A; choose by reach/height map in Phase 2), set-aside zone (mark with tape, Phase
2), supply mat. Report max reachable stack height at each spot.

---

## 4. System architecture

```
command ─► planner (language model, small/fast model first)
              │  steps: [A4(stack A) , A2(spot B) , ...]  with preconditions/expected result
              ▼
 scene map ◄─ perception: front + wrist RGB + wrist depth
 (boxes: type, pose; stacks: levels; objects: mask, 3D box, class guess)
              │
              ▼
 skill executor ─► menu VLA (SmolVLA, task string per step) ─► handover ─► IK place
              │                                         └─► depth grasp planner (O1 fallback)
              ▼
 step checker (extended turn_check + wrist look) ─► ok → next step / fail → re-plan / ask user
 safety: runner envelope, hand pause, e-stop (reused)
```

- The planner sees only the scene map (structured JSON), never raw images, and may
  only emit menu actions with arguments that refer to scene-map IDs.
- The VLA's task string is produced by the executor from the step, with the same
  templates used in data generation (paraphrased), so train and test strings match.
- Policy inference runs in our own loop (as in `skill/runner.py`), on the Mac (MPS)
  or on botmind over Tailscale if the Mac is too slow (measure in Phase 6).
- Box types come from the wrist camera at pick time and from task history, not from
  the front view.

---

## 5. Simulation additions

- **Random box poses** anywhere in reach on the mats and table (not only template
  slots), multiple stacks at A and B, boxes on boxes in any order.
- **Object library:** YCB, Google Scanned Objects and a filtered Objaverse subset,
  filtered to the accepted class (bounding box, a narrow side < ~9 cm, mass ≤ ~150 g,
  rigid). Convex collision shapes (CoACD), mass from volume × density range,
  friction randomized. Split meshes into **train / held-out** sets; held-out meshes
  are never used in training. Target ≥ 300 train meshes.
- **Depth rendering** for the wrist camera (Orbbec intrinsics), with noise and holes
  similar to the real sensor (measure in Phase 1).
- **Background randomization** behind the arm (images, text, posters), not only colour.
- **Experts:** one scripted expert per VLA action (A1–A7, O1–O5), privileged state +
  IK, reusing the stacking-game expert, soft-servo compensation, DART noise and
  retry demos. Object grasp in the expert = top-down antipodal grasp across the
  narrow side from the mesh (the same rule the depth grasp planner uses on real data).
- **Task strings:** templates per action with paraphrases and referring expressions
  (size, colour, position, relation). Store action, target IDs, object category,
  seen/held-out, success in per-episode metadata.
- Keep only successes; the step's end condition is checked with privileged state.

---

## 6. Working rules

1. **Verify, don't recall.** Check installed source/docs (MuJoCo, LeRobot, SmolVLA
   config, Orbbec SDK) before using an API. Cite file/line or URL in notes.
2. **Same LeRobot version for export and training** on every machine (pin the commit;
   record it in the Decisions log).
3. **Small steps, evidence each time** in `notes/img/`; look at every render.
4. **Never move the real robot.** Exact commands for the user.
5. **No secrets in files.** `hf auth login`; cloud API keys only in the user's shell env.
6. **Pilot before scale:** every data or training change is tested on a small pilot
   (≈ 600 episodes / short run) before a main data run or main training run.
7. **Machine discipline:** one booking per GPU in `notes/machines.md`; never leave a
   GPU idle when a job is ready; report machine use at each checkpoint.
8. **Checkpoints (stop for the user's OK):** end of Phases 0, 2, 4; before each
   main training run; before first real-arm motion.
9. **One report per phase** in `notes/phaseN.md`, plus a live `notes/phaseN_status.md`.
10. **Parallel agents** (Section 9) work only in their own area/worktree; shared
    files (this brief, `sim/assets/`) are changed by the coordinator only.

---

## 7. Compute plan and budget

| Job | Where | Notes |
|---|---|---|
| Data generation | botmind (owner) + siix-pc when not training + home pc | free; ~3–4 s/episode with objects (measure) |
| Main SmolVLA training (~40k steps) and fine-tune round 1 | **siix-pc** (20 GB) | ~30–45 h for the main run (estimate; measure in week 1); checkpoint every 2k steps |
| Pilots, sim evaluation, fine-tune round 2 | botmind | |

No cloud GPU (user, 2026-11-02). Which job runs on which machine is booked in
`notes/machines.md`, edited by the coordinator only. Timeline, gates and the
workstream split: `notes/execution_plan.md`. The RunPod notes in
`notes/compute_plan.md` are kept only as a fallback if the schedule slips.

---

## 8. Phases

### Phase 0: Setup and checks
- New repo `~/so101_menu_vla`; link/copy the reused pieces (Section 2) and pin versions.
- **Orbbec depth on macOS**: does the depth stream work (pyorbbecsdk or equivalent)?
  Frame rate, minimum range, noise, holes on boxes and on the white table. If it
  fails on macOS, decide with the user (Linux host for the wrist camera, or RGB only).
- Powered USB 3 hub installed for the Orbbec; repeat the camera soak test.
- SmolVLA installed in the training env on botmind (same LeRobot commit); confirm
  its image resize, frozen parts and memory at batch 16/32 on 16 GB.
- **Checkpoint.**

### Phase 1: Perception for the scene map
- Box pose and type anywhere on the mats and table (front + wrist RGB + wrist depth);
  stacks and their levels (wrist look per stack); open-vocabulary object detection
  (OWLv2 / Grounding DINO) + segmentation (SAM 2) + depth → 3D box per object.
- Depth grasp planner for O1 (top-down antipodal across the narrow side).
- Develop on sim renders, validate on real frames (photos of ≥ 20 layouts, 10 real
  accepted-class objects). Report detection/pose errors per object type.

### Phase 2: Scene, locations and object library
- Spot B, set-aside zone, reach/height map per spot; mark them on the real table.
- Object library import + filtering + collision shapes + train/held-out split;
  physics sanity tests (drop, rest, grasp hold) per mesh, discard failures.
- Background randomization. Depth rendering + noise model.
- **Checkpoint:** spots, zone, library stats, sample renders vs real photos.

### Phase 3: Experts and task strings
- One expert per VLA action (A1–A7, O1–O5); expert success ≥ 95 % per action in sim.
- Task-string templates and paraphrases; referring-expression generator.
- 20 episodes per action: inspect videos, joint plots, strings.

### Phase 4: Data pilot and training pilot (botmind)
- ~600 episodes across all actions; SmolVLA short run on botmind; sim eval per action.
- Measure throughput (samples/s) on botmind and siix-pc and decide the main run length.
- **Checkpoint** (before the main data run): pilot results, main run length and timing.

### Phase 5: Main data and training
- ~30–60k episodes (weights decided from the pilot; more on weak actions).
- Export at 320×240 (proven in the stacking game), copy to siix-pc.
- Main run on siix-pc; evaluate checkpoints on botmind in sim.
- Up to two targeted rounds (generate only what fails, fine-tune ~20k steps).

### Phase 6: Planner, executor and checker
- Language-model planner (scene map in, action list out, JSON schema), executor,
  extended step checker for every action, re-planning, "can't" handling.
- Full composite tasks in sim (privileged and perception-based scene map).

### Phase 7: Real trials
- **Checkpoint before real motion.** Staged: box actions → object actions (seen
  categories, then never-seen real objects) → composite tasks.
- Log every step: command, plan, action, target, outcome, failure type, timings.
- If real ≪ sim: camera overlay drift, then mapping, servos, physics, then randomization
  (same order as the stacking game). Real demos stay off limits; flag if blocked.

### Phase 8: Demo and write-up
- A recorded demo of the composite tasks, per-action results table, honest limits.

---

## 9. Parallel agents (suggested split)

| Agent | Area (owns these folders) | Phases |
|---|---|---|
| A: sim and data | `sim/`, `experts/`, `scripts/generate*` | 2, 3, 5 (data) |
| B: perception | `perception/` | 1, 7 support |
| C: planner and checks | `planner/`, `executor/`, `checks/` | 6 |
| D: training and eval | `train/`, `eval/`, remote training scripts | 0 (SmolVLA), 4, 5 (training) |

The coordinator session owns this brief, `sim/assets/` and the Decisions log, and
merges branches. Agents never edit another agent's folders.

---

## 10. Known pitfalls

- **Train/test task-string mismatch** silently ruins a VLA; generate both from one module.
- **Instruction ignored:** check instruction accuracy separately from success.
- **Object meshes with bad physics** (non-watertight, wrong mass) teach wrong grasps;
  test every mesh before use.
- **Depth holes** on dark mat and box edges; never depend on one depth pixel.
- **Compounding errors** in long tasks: every step is checked; the planner retries.
- **Background, lighting, auto exposure:** lock the camera settings; randomize in sim.
- **Idle GPUs:** a finished job with nothing queued wastes days; keep the booking file current.
- **Scope creep:** anything outside the accepted object class gets "can't", not a new skill.

---

## 11. Decisions log

Append entries as `YYYY-MM-DD: decision, reason, evidence/link`.

- 2026-10-30: Project = menu VLA + language-model planner + per-step camera checks on the unchanged stacking-game setup. Reason: "any action on any object" is not achievable on this arm with sim-only data; a fixed menu with planner composition is. Evidence: planning discussion in the stacking-game session.
- 2026-10-30: Accepted foreign-object class = small, rigid, light, with a clear narrow side (user decision). Everything else gets "can't".
- 2026-10-30: Sim-only training data kept; pretrained models (SmolVLA base, detectors, segmenters, grasp nets, language models) allowed.
- 2026-10-30: Placement stays computed (IK), as in the stacking game; the VLA picks, carries and pushes.
- 2026-10-30: Compute: data and pilots on botmind; main training on a rented RunPod A100 80 GB (Community Cloud) with prepaid credit as the budget cap ($150, pending sponsor approval); botmind-only fallback.
- 2026-11-02: siix-pc added (RTX 4000 Ada 20 GB, Core Ultra 7 265K, 64 GB): data generation, pilots, evaluation and fine-tunes run on botmind + siix-pc in parallel; only the main run is rented. Cloud cap lowered to $50 (expected ~$30–35). Speed of siix-pc to be measured. Evidence: `notes/compute_plan.md`.
- 2026-11-02: **No cloud GPU** (user: siix-pc is the only addition). Main training and fine-tune round 1 on siix-pc, data generation/eval/pilots on botmind. Deadline 11 Dec 2026, target 4 Dec. One coordinator session + workstream sessions A–D + Codex for self-contained coding tasks. Plan, gates and dates: `notes/execution_plan.md`.
