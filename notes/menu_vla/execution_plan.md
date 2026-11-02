# Execution plan: menu VLA by 11 Dec 2026 (target 4 Dec)

Written 2026-11-02. Companion to `CLAUDE.md` (the brief) and `compute_plan.md`.
Hardware: the current setup plus siix-pc. **No cloud GPU.** Extra AI capacity:
Claude Code (several parallel sessions) and a ChatGPT Codex Plus plan.

## 1. Strategy in one paragraph

Learn one thing well, compute the rest. The VLA learns **"pick the thing I name"**
(boxes by size/position/stack level, objects by description) plus **push aside**.
Every placement variant (on a stack, at a spot, next to X, rotated) is an IK target,
so most of the menu comes from code that already works on the real arm
(`ScriptedPlace`, 18/18 real). Long tasks come from the planner. This keeps
training to one model and one main run, which fits the two GPUs we own.

## 2. Timeline

Today is Mon 2 Nov. Five working weeks to the target, one buffer week.

| Week | Dates | Goal | Gate at the end (stop for your OK) |
|---|---|---|---|
| 1 | 2–8 Nov | Setup and foundations: siix-pc, depth test, spot B and set-aside zone, object library, SmolVLA running, first baseline | Depth works on the Mac or not; SmolVLA speed measured on botmind and siix-pc; object library stats |
| 2 | 9–15 Nov | Experts for every learned action, task strings, 600-episode pilot, pilot training; planner + checker running in sim with the scripted expert | Pilot success per action → decide whether objects are learned or classical |
| 3 | 16–22 Nov | Main data (~25–30k episodes on botmind and siix-pc), main training on siix-pc; planner on the perception-based scene map in sim | Main checkpoint evaluated per action |
| 4 | 23–29 Nov | Targeted fine-tune round 1; real trials of box actions, then objects | Real per-action results |
| 5 | 30 Nov–6 Dec | Real composite tasks, fine-tune round 2 only if needed, demo recording, write-up. **Target finish Fri 4 Dec** | Final results |
| Buffer | 7–11 Dec | Only for slips. Nothing new is started this week | — |

### Decision gates that protect the deadline
- **Fri 6 Nov, depth:** if the Orbbec depth stream doesn't work on macOS, objects use
  RGB only (detector + mask + known table plane + front-camera height). No Linux host.
- **Sun 15 Nov, objects:** if the pilot picks objects < 50 % in sim, object picking
  becomes classical (detector + SAM 2 + depth grasp + IK). The VLA keeps boxes and
  push. This is still the full menu, just with an honest "how" column.
- **Sun 29 Nov, real:** freeze scope at what works on the real arm. Stretch actions
  (A7 rotate, O4 object onto stack, A6 relative place) go only if they're already reliable.
- Any week that slips more than 2 days drops the lowest stretch item, never the buffer.

## 3. Compute (two GPUs, no cloud)

| Machine | Owns | Notes |
|---|---|---|
| siix-pc (RTX 4000 Ada 20 GB, 20 cores, 64 GB) | **Main training** (larger batch fits in 20 GB); data generation when not training | Set up in week 1 |
| botmind (4070 Ti SUPER 16 GB, 20 cores) | **Data generation**, sim evaluation of checkpoints, pilots, the second fine-tune | Busy all project |
| home pc | Extra evaluation workers | Optional |
| Mac | Coordinator sessions, real robot, cameras, perception on real frames, inference | You run every real-arm command |

Hours (estimates until the week-1 speed test):

| Job | Machine | Time |
|---|---|---|
| Baseline SmolVLA on the existing 1,000-episode size-instruction dataset | siix-pc | ~10–15 h (week 1) |
| Pilot: 600 episodes + short training | botmind gen (~30 min) + siix-pc (~6–8 h) | week 2 |
| Main data: ~25–30k episodes | botmind and siix-pc, ~1.5–2 s/episode combined | ~12–16 h |
| Main training: ~40k steps | siix-pc | ~30–45 h (2 days) |
| Fine-tune round 1 (failing actions only, ~15k steps) | siix-pc | ~12–18 h |
| Fine-tune round 2 (optional) | botmind | ~12–18 h |

Main training runs Wed 18 → Fri 20 Nov. If the speed test shows siix-pc is
too slow, the main run drops to 30k steps at a larger batch; a run never blocks
data generation, because they're on different machines.

A **machine booking file** (`notes/machines.md`) lists what runs where and until
when. Only the coordinator edits it, so two sessions never start jobs on the same GPU.

## 4. Who does what

### How the sessions work (answer to "do I send an md file around?")
No copying files between sessions by hand:
1. I set up `~/so101_menu_vla` with this brief as its `CLAUDE.md` (it loads
   automatically in any session opened there), plus this plan and `compute_plan.md`.
2. You open **one new Claude Code session there: the coordinator.** It reads the
   brief, keeps the plan, the Decisions log and the machine bookings, reviews and
   merges work, and talks to you at each gate.
3. The coordinator **starts the workstream sessions itself** (the desktop app shows
   them in your sidebar, each in its own git worktree). It writes each one's brief.
   You can open any of them to watch or steer. Small jobs inside a workstream use
   subagents and need nothing from you.
4. **Codex** gets self-contained coding tasks. The coordinator writes each one as a
   GitHub issue (or `notes/codex_tasks/NN.md`) with the files, the interface and the
   tests that define "done". You paste it into Codex; Codex opens a PR on its own
   branch; the coordinator reviews and merges it. An `AGENTS.md` at the repo root
   points Codex to the brief and its rules.

### Workstreams

| Session | Owns | Main work | Machines |
|---|---|---|---|
| Coordinator | brief, `sim/assets/`, Decisions log, `notes/machines.md`, merges | plan, gates, reviews, your real-arm commands | Mac |
| A: sim and data | `sim/`, `experts/`, `scripts/generate*` | spots, object library, depth rendering, background randomization, experts, data generation | botmind, siix-pc (CPU) |
| B: perception | `perception/` | depth test, scene map (boxes anywhere, stacks, objects), depth grasp planner, extended step checker images | Mac |
| C: planner and executor | `planner/`, `executor/`, `checks/` | language-model planner, executor, step checks, re-planning, composite tasks in sim | Mac |
| D: training and eval | `train/`, `eval/` | SmolVLA env on botmind and siix-pc, speed test, baseline, pilot, main run, fine-tunes, per-action evaluation | siix-pc, botmind (GPU) |

Four parallel Claude sessions at once is heavy use: plan on a Claude **Max 20×**
plan ($200/month) for these five weeks. If limits are hit, B and C pause first,
since A and D keep the GPUs busy.

### Codex tasks (pure code with tests; no robot, no remote machines)
1. Object-library pipeline: download YCB / Google Scanned Objects, filter to the
   accepted class, CoACD collision shapes, mass and friction, train/held-out split,
   per-mesh drop/rest tests (A reviews).
2. Task-string module: templates, paraphrases, referring expressions, one module
   used by both data generation and the executor, with unit tests (A and C review).
3. Planner JSON schema, validator and a library of 30 test commands with expected
   plans (C reviews).
4. Evaluation report: per-action / per-object / seen-vs-held-out tables and plots
   from evaluation logs (D reviews).
5. Second reviewer: Codex reviews Claude's large PRs for bugs before merging.

Codex Plus has usage limits per time window, so it works on 1–2 of these at a time.

## 5. Your part

About 20–30 minutes a day of approvals and gate decisions, plus these sessions at the
arm (the coordinator sends the exact commands each time):

| When | What | Time |
|---|---|---|
| Mon 2–Tue 3 Nov | siix-pc: install Tailscale, OpenSSH, Miniforge (commands provided); move the Orbbec to another Mac port and run the depth test | ~1 h |
| Wed 4–Thu 5 Nov | Mark spot B and the set-aside zone with tape; touch points; photos of 20 box layouts; collect 10 real accepted-class objects never used in sim (+ 2 out-of-class ones, e.g. a phone) and photograph them | ~1.5 h |
| ~Thu 12 Nov | First real motion: baseline SmolVLA picking a box by size | ~1 h |
| Mon 23–Wed 25 Nov | Real trials: box actions | ~2 × 1.5 h |
| Thu 26–Sat 28 Nov | Real trials: objects (seen categories, then never-seen) | ~2 × 1.5 h |
| Mon 30 Nov–Thu 3 Dec | Real composite tasks (5 runs each) and demo recording | ~3 × 1.5 h |

Buy the powered USB hub (~$40) only if the other Mac port still resets in the depth test.

## 6. What makes the result impressive

The demo video (recorded 1–3 Dec):
1. "Rebuild that stack in reverse order on spot B." (4 boxes, mixed sizes)
2. "Take the marker off the large box, then put the large box on stack A." The marker
   is a real object never seen in simulation.
3. "Sort the boxes by size into one stack, largest at the bottom."
4. A person moves a box mid-task; the checker notices and the planner re-plans.
5. "Pick up the phone": the robot says it can't (out of class) instead of trying.

Backed by numbers: per-action success in sim and on the real arm, never-seen vs seen
objects, composite tasks out of 5 runs, every step's check logged, and the line
"no real demonstrations: all robot training data came from simulation".

## 7. Week 1 in detail (starts now)

| Day | Coordinator | A | B | C | D | Codex |
|---|---|---|---|---|---|---|
| Mon 2 | set up repo, `AGENTS.md`, `notes/machines.md`; siix-pc setup commands for you | — | Orbbec depth test script for the Mac | — | — | — |
| Tue 3 | start sessions A–D | spot B / zone candidates from the reach map | depth test results with you | scene-map JSON schema, planner prompt design | siix-pc: `lerobot` env, MuJoCo headless check | task 1 (object library) |
| Wed 4 | — | background randomization, depth rendering | box pose/type anywhere (sim renders) | executor skeleton on privileged sim state | SmolVLA speed test on botmind and siix-pc | task 2 (task strings) |
| Thu 5 | your tape and photo session | review object library PR | real layout photos → box detection check | planner on 30 test commands | baseline training on the existing size-instruction dataset (siix-pc) | task 3 (planner schema) |
| Fri 6 | **depth gate** | object physics tests | object detection + SAM 2 on real object photos | re-planning on simulated failures | baseline evaluation in sim | — |
| Sun 8 | **week-1 gate** with you | — | — | — | — | — |
