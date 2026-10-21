# Phase 4: Physics and stack-stability calibration

Date: 2026-10-21. Status: **at the checkpoint.** Overhang and slide match real within
target, and the grasp test passes for every type. One open question about the tap test.

Parameters: `sim/assets/physics.yaml` (also used by `build_scene.py`). Logs:
`notes/phase4_stability_final.txt`, `notes/phase4_grasp_test.txt`.

## 1. What the real tests imply

For a rigid, uniform box, the top box tips when the overhang reaches half its own length.
Every real pairing tips **1–2 mm before** that point, so the boxes behave almost like
rigid uniform blocks; the 1–2 mm comes from slightly rounded cardboard edges or a slightly
off-centre load. Friction μ = tan(slide angle).

## 2. Sim vs real (`scripts/stability_check.py`)

**Overhang** (mm at topple; target within 2–3 mm for every pairing):

| pairing (bottom → top) | real mean (runs) | sim | sim − real |
|---|---|---|---|
| No.3 → No.3 | 33.3 (35, 33, 32) | 34.8 | +1.5 |
| No.35 → No.35 | 29.0 (28, 29, 30) | 29.6 | +0.6 |
| No.35 → No.10 | 25.0 (24, 26, 25) | 25.9 | +0.9 |
| **No.10 → No.3 (large on small)** | 33.3 (30, 36, 34) | 34.7 | **+1.3** |
| No.3 → No.10 | 25.0 (25, 25, 25) | 25.9 | +0.9 |
| No.3 → No.35 → No.10 | 25.7 (25, 26, 26) | 25.7 | 0.0 |

**Worst error 1.5 mm, so every pairing is within target.** Sim boxes are very slightly
(≈ 1 mm) more forgiving than real, which is the safe side for careful placement.

**Slide** (deg at slide onset): No.3 on No.3 26.0 → 26.1; No.35 on No.35 19.7 → 19.8;
No.10 on No.35 23.3 → 23.3; box on yellow tape 16.3 → 16.3; on black mat 19.3 → 19.3.
**All within 0.2°.**

**Tap (open):** real stack B (No.10 base, deliberately top-heavy) toppled 3/5 and stack A
1/5. In sim, A always needs *less* push than B: true for all six tap models tried (fixed
impulse, fixed speed and 50 ms force pulse, each across and along the long side). The sim
failure mode is that the tapped level-3 box **slides out**; A's tapped box is a light
No.35 under the No.10, and B's is the heavier, higher-friction No.3. Real B more likely
**tipped over its small No.10 base**. Real taps aren't calibrated (strength, contact
point), and the two quantitative tests match, so this is recorded as a known discrepancy.
Question for the user below.

## 3. Physics settings (`sim/assets/physics.yaml`)

- **Box friction by type:** No.3 0.49, No.35 0.36, No.10 0.43. Box–box contact uses the
  larger value, verified in MuJoCo 3.16, which reproduces all three measured pairs.
- **Surfaces:** tape 0.29, mat 0.35, table 0.30 (APPROX, not measured), with **priority 1**
  so the surface value wins. Verified: a higher-priority geom's friction is used.
- **Contact model:** elliptic friction cone, `impratio` 10, timestep 2 ms, MuJoCo default
  contact softness (solref/solimp), condim 3, box masses as measured (38 / 31 / 21 g).
  - `impratio` was raised from 1: at 1 a held No.3 slowly crept out of a ~10 N grip
    (MuJoCo's guidance for grasping is > 1).
  - Overhang and slide re-checked after the change (above).

## 4. Gripper contact model (changed in `build_scene.py`)

- **Problem:** MuJoCo collides meshes as convex hulls. The gripper part's hull (fixed jaw
  plus the body near the hinge) fills the gap between the jaws, so a box placed there was
  thrown out.
- **Fix:** two **finger box colliders per jaw** along the real inner faces, measured from
  the vendor meshes (`notes/phase4_finger_geometry.txt`). The gripper hulls now collide only
  with the table and backdrop, not with boxes (collision bitmasks documented in the code).
  Finger friction 0.8 (APPROX: jaw pads not measured).
- **Gripper torque** capped at **0.75 N·m**. LeRobot sets the gripper to 50 % max torque and
  25 % when overloaded (`so_follower.py:192-195`), about 0.74 N·m sustained for an STS3215;
  the vendor file allowed 3.35.

## 5. Grasp test (`scripts/grasp_test.py`; target ≥ 95 % hold for 2 s per type)

Scripted pick: IK with vertical jaws, jaws across the width, box side 2 mm from the fixed
jaw. Pre-grasp 5 cm above with the jaws open to width + 15 mm, then a **straight-down
Cartesian descent** (waypoints every 5 mm), tips 1 cm below the box top. Close
(torque-limited), lift 10 cm, hold 2 s. Commands at 30 Hz, ≤ 60°/s. 100 random positions
and yaws per type.

| box | held 2 s | closed at reading (sim / real contact) |
|---|---|---|
| No.3 | **100 / 100** | 20.7 / 24.0 |
| No.35 | **100 / 100** | 17.6 / 18.6 |
| No.10 | **100 / 100** | 13.4 / 15.5 |

**PASS.** On the way:
- Joint-space interpolation from a tilted pre-grasp pose swung the jaws into the box
  (No.3 91 %). The straight-down descent fixed it; **the Phase 5 expert uses it too.**
- The sim closes 1–3 reading units tighter than the real contact readings (jaw-pad
  thickness, cardboard flex). The expert's squeeze command should be set relative to the
  sim contact reading.

## 6. Reach limit found (game constraint)

Vertical grasp IK fails from about **29.5 cm** from the pan axis (the arm is almost
straight). Only **71 % of the black mat** is within 29 cm (`notes/phase4_supply_reach.txt`);
its two outer corners are 35–36 cm out. The grasp test samples within 29 cm. Options:
mark a reach line on the mat and keep boxes inside it, or move the mat about 5 cm toward
the arm (then re-touch its corners).

## 7. Your answers (2026-10-21)

**Tap test.** Real stack B fell both ways: sometimes the whole stack tipped over its
small No.10 base, sometimes the tapped box slid out. In sim
(`notes/phase4_tap_modes.txt`, push on level 3 across the long side):
- **Short taps:** a 50–150 ms push only ever slides boxes out; B never tips, even at 3 N.
- **Sustained pushes (≥ 300 ms):** B shows **both modes close together**, tipping over the
  base from **0.35 N** and the tapped box sliding out from **0.40 N**, matching your
  description. So the tipping physics is fine; the short-tap model was wrong.
- **Remaining difference:** A gives way at 0.20 N, lower than B, while your A toppled less
  (1/5 vs 3/5). The sim lets a light mid-stack box slide out a little too easily. That's
  the cautious side for placement, so it's recorded, not tuned.

**Reach line.** A straight strip of matte white tape (~1.7 cm) across the black mat at
the reach limit, no re-touching (`sim/assets/reach_line.yaml`, `notes/phase4_reach_line.txt`):
- **End 1:** on edge 1→2, 14.7 cm from corner 1 (the one nearest the arm base).
- **End 2:** on edge 4→3, 13.0 cm from corner 4 (the one nearest the yellow area).
- **Length and position:** 19.3 cm long; its middle is 27.4 cm from the pan axis, 1.6 cm
  inside the 29 cm limit.
- **Usable area:** boxes stay entirely on the arm side, about **66 % of the mat**, enough
  for all 8 boxes in a loose grid.
- **In sim:** a visual-only white strip in `build_scene.py`. Phase 5 game states place
  supply boxes only on the arm side.

## Open items
1. Tap test: the sim's light mid-stack box slides out slightly too easily (recorded, cautious side).
3. APPROX: finger friction 0.8, white-table friction 0.30, gripper torque 0.75 N·m.

## Next step
After your OK: Phase 5, the scripted expert, game states and the first 20 episodes.
