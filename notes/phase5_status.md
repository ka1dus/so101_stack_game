# Phase 5 working status (handoff file)

Read this first when resuming Phase 5 in a new session or after a context reset. Update it
whenever a step finishes. The full report is `notes/phase5.md`; this file is the short "where are we".

Last updated: 2026-10-21

## Done (not committed yet; all Phase 5 work is uncommitted on `main`)
- Supply mat expanded 2 cm toward the arm, to 22.9 × 19.2 cm, re-touched (`inputs/supply_points_20261021.json`);
  layout, scene and front calibration updated (front calibration uses `supply_area_pre_expansion`,
  because the reference photos predate the expansion). Reported in `notes/phase5.md` §1–4.
- `scripts/game_states.py`: randomized states, stack heights 1–7, per-level types, hard-case
  over-sampling, game-progression supply (`grid_supply`, `progress_game`), `choose_pick` rule.
- `scripts/expert.py`: pick (straight-down descent, signed jaw side), carry, place-target search
  (centre of mass over support), slow descent, release, retract, 3 s judgement.

## Current step: expert placement accuracy
`expert.py --n 20` (seed 0) gives 15/20. The stack is never knocked (moved ≤ 0.4 mm).
- Fixed: IK at the place pose now tries both wrist directions (the box is 180°-symmetric) plus
  extra seeds. Worst IK error is now ≤ 2.9 mm (it was 17–27 mm).
- Still failing:
  - ep 6 and ep 18 "misplaced" (16 and 21 mm offset) with a small IK error. So the box
    shifts during release or retract, or `b_in_g` (the held box measured in the gripper frame)
    is stale. Next: log the box pose before release, after the gripper opens, and after retract.
  - ep 0, 12 and 15 "ik_place" at h = 6–7: tall stacks hit the wrist_flex limit. Next: allow
    jaw tilt up to about 12–15° at release (Phase 3: 7 boxes need ≤ 15°), and check the threshold
    `t3 > 30` / `max(e3, e_rel) > 3` in `run()`.
- Goal: ≥ 19/20 on `--n 20`, then a 100-episode check broken down by height and pairing.

## Remaining Phase 5 steps
1. Randomization: front camera ±1 cm/±1–2°; wrist camera ±few mm/±1°, f ±5 %, |k1| ≤ 0.05; lighting;
   table/background colour; box texture tint/flip/plain colour; tape shades; start pose ±3°.
2. `scripts/generate.py`: front and wrist at 640×480, 30 fps; state/action in LeRobot units via
   `sim_to_real`; raw episode = mp4 per camera + npz + metadata json; keep successes only.
3. Generate 20 episodes; inspect videos and joint plots; distribution of heights, types, pairings, offsets.
4. Finish `notes/phase5.md`, add Decisions-log entries, then **CHECKPOINT** (user OK before 2,000–3,000 episodes).
5. Later: `export_lerobot.py` (WSL lerobot env) → `Maxctl352/so101_stack_sim_act` and `_vla`.

## Commands
    MUJOCO_GL=cgl ~/.venvs/so101_stack_sim/bin/python scripts/expert.py --n 20 --seed 0   # ~2.7 min
