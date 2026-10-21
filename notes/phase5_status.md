# Phase 5 working status (handoff file)

Read this first when resuming Phase 5 in a new session or after a context reset. Update it
whenever a step finishes. The full report is `notes/phase5.md`; this file is the short "where are we".

Last updated: 2026-10-21 (evening)

## Done (not committed yet; all Phase 5 work is uncommitted on `main`)
- Supply mat expanded 2 cm toward the arm, to 22.9 × 19.2 cm, re-touched (`inputs/supply_points_20261021.json`);
  layout, scene and front calibration updated (front calibration uses `supply_area_pre_expansion`,
  because the reference photos predate the expansion). Reported in `notes/phase5.md` §1–4.
- `scripts/game_states.py`: randomized states, stack heights 1–7, per-level types, hard-case
  over-sampling, game-progression supply (`grid_supply`, `progress_game`), `choose_pick` rule.
- `scripts/expert.py`: pick (straight-down descent, signed jaw side), carry, place-target search
  (centre of mass over support), slow descent, release, retract, 3 s judgement.

## Current step: expert placement accuracy (fixed; 100-episode check running)
`expert.py --n 20` (seed 0): **20/20** (was 15/20). Offsets ≤ 4 mm; stack moved ≤ 1.9 mm; release tilt 5–10° at h = 6–7.
Fixes, in order:
1. Place IK tries both wrist directions (the box is 180°-symmetric).
2. "Misplaced" root cause: at roll ≈ −180° the **wrist camera housing hit the backdrop**, so shoulder_pan/lift
   stalled 5–18° short. Fix: `backdrop_gap()` (mj_geomDistance) rejects place poses with < 10 mm for the
   wrist/gripper/camera (< 3 mm for the shoulder/upper/lower arm, which sit 5–10 mm from the backdrop at every
   place pose by layout). Any arm–backdrop contact during an episode → failure "hit_backdrop".
3. `ik_pos_first()`: lower the orientation weight 100 → 30 → 10 → 3 until the position error is ≤ 2 mm
   (fixed the tall-stack "ik_place" failures).
Held-out check: `expert.py --n 100 --seed 1` → `notes/phase5_expert_check100.txt`.
Debug prints in `run()` (`self.dbg`, ik_place line) can be removed once stable.

## Done since (uncommitted)
- 100-episode held-out check: 98/100 (2 × hit_backdrop on tall carries; discarded by the generator).
- `scripts/randomize.py` (cameras, lights, colours, box tints, image augmentation, wrist k1, front corner mask).
- Scene: arm recoloured black (real arm is black), room wall added (APPROX, randomized). Rebuilt `sim/scene.xml`.
- `scripts/generate.py` (raw episodes), `scripts/inspect_episodes.py` (joint plots, distributions, frames),
  `scripts/export_lerobot.py` (WSL; untested against the WSL version yet).
- `notes/phase5.md` §5–7 written.

## Pilot done → WAITING AT THE PHASE 5 CHECKPOINT (user OK needed before large generation)
- Pilot `data/raw/act_pilot` 20/20; fixes: clipped home jitter, roll-consistent jaw side. VLA smoke 6/6.
- Export checked locally (Mac LeRobot 0.6.2). The report is complete through §10 (`notes/phase5.md`), and the
  Decisions log is updated. Not committed since db49dbd.

## After the OK
1. Apply the §10 proposal (hard share 0.6, heights 1–3 : 4–7 = 40 : 60; needs a `sample_state` weight option),
   run 4 parallel `generate.py` processes (ACT 2,000 + VLA 1,000), inspect each run.
2. Copy `data/raw/` to WSL; run `export_lerobot.py` there (check the WSL LeRobot version first); push with
   `hf auth login` → `Maxctl352/so101_stack_sim_act` and `_vla`.

## Commands
    MUJOCO_GL=cgl ~/.venvs/so101_stack_sim/bin/python scripts/expert.py --n 20 --seed 0   # ~2.7 min
