# Phase 1: Robot model and joint mapping

Date: 2026-10-20. Status: **mapping code done and tested; arm offsets (and base tilt)
not yet validated. They need three short real-arm measurements from you (section 7).**

Logs for every number below are in `notes/phase1_logs/`. The analysis scripts are in
`scripts/analysis/`.

## 1. Robot model

- No prior `so101_sim/` exists on either machine (you confirmed), so I used the official
  model: TheRobotStudio **SO-ARM100 `Simulation/SO101`, commit `a758567`** (2026-10-06),
  Apache-2.0. It's copied unmodified into `sim/assets/so101/` (STL meshes plus
  `so101_new_calib.xml`, `so101_new_calib_camera.xml` and `so101_new_calib.urdf`; the CAD
  `.part` files are left out). `SOURCE.txt` records where it came from.
- **"New calibration" variant:** its joint zeros sit at the middle of each CAD range. The
  MJCF and URDF have identical kinematics, and `gripperframe` = `gripper_frame_link`.
- The arm in `side.jpg` is a standard SO-101. The jaws also match the CAD; the gripper
  fit in section 3 shows it.
- **The wrist camera does not match the official camera mount.** The real one is a large
  black box on the other side of the gripper (`phase1_home_compare.png`). Phase 3 has to
  model your camera housing.

## 2. Joint mapping: `scripts/joint_mapping.py`

Taken from LeRobot source (main `ca69a206`), not from memory:

| | formula | source |
|---|---|---|
| body joints | deg = (raw − (min+max)/2) · 360/4095 | `motors/motors_bus.py:874-877`, `feetech/tables.py:190` |
| gripper | g = (clip(raw) − min)/(max − min) · 100, 0 = closed | `motors_bus.py:871-873`, `so_follower.py:80` |
| homing | the pose held during calibration reads raw **2047** | `feetech/feetech.py:278-288`, `so_follower.py:155-167` |

So LeRobot's 0° is the **midpoint of the range you swept** during calibration. It is
neither your homing pose (that pose reads pan −2.0°, lift +3.0°, elbow +7.4°, wrist +4.2°)
nor necessarily the CAD zero.

Mapping, with parameters in `sim/assets/joint_mapping.yaml`:
- body: `q_sim = radians(q_lerobot − offset_deg)`, offsets **0 for now (provisional)**
- gripper: `q_sim = radians(−10.75° + g · 1.3029°)`. The slope comes straight from the
  calibration file, because the jaw sits on the servo horn. The −10.75° was fitted
  (section 3).

Tests (`scripts/test_joint_mapping.py`, 7 pass):
- real→sim→real and sim→real→sim round trips (1e-9)
- raw↔LeRobot round trip, spot values, gripper slope
- home pose inside the sim joint limits

`scripts/check_mapping_vs_lerobot.py` runs in the `lerobot` env and calls LeRobot's own
`FeetechMotorsBus._normalize/_unnormalize` on every raw tick of your calibration. The
difference is exactly 0, with no hardware connected.

**Sim joint limits are narrower than your real calibrated ranges:**

| joint | real (LeRobot deg) | sim limit |
|---|---|---|
| shoulder_pan | ±119.4 | ±110 |
| shoulder_lift | ±105.3 | ±100 |
| elbow_flex | ±97.0 | ±96.8 (home is 96.79, at the limit) |
| wrist_flex | ±101.9 | ±95 |
| wrist_roll | ±180 | −157..+163 |
| gripper | −10.8..119.5° | −10..100° |

Proposal: widen the sim limits in `scene.xml` (Phase 3) to the real calibrated ranges
mapped into sim, without editing the vendor file.

## 3. Gripper: max opening and grasp clearance

Fitted only the closed angle; the slope is fixed by the calibration
(`phase1_logs/phase1_gripper_fit.txt`):

| reading | measured (ruler) | sim with −10.75° + 1.303·g |
|---|---|---|
| 0.13 closed | tips touch | 1.6 mm nearest jaw gap (3.8 mm between the picked tip vertices) |
| 24.0 on No.3 | 40 mm | 41.8 mm |
| 18.6 on No.35 | 35 mm | 32.9 mm |
| 15.5 on No.10 | 29 mm | 27.8 mm |
| 99.66 open, tips | **150 mm** | **153.6 mm** |
| 99.66 open, 1 cm back | 122 mm | 137.6 mm (your weakest measurement: where is "1 cm depth"?) |

The box rows fit with 1.7 mm RMS. The official jaws match yours, and the closed position
coincides with the model's −10° lower limit.

**Max opening 150 mm (real) / 154 mm (sim, at 119°, past the vendor's 100° limit).**

| box | width | contact reading (sim / your ruler) | pre-grasp (width + 15 mm) | margin to full opening |
|---|---|---|---|---|
| No.3 | 40 mm | 22.9 / 24.0 | 32.2 | 110 mm |
| No.35 | 35 mm | 19.8 / 18.6 | 29.1 | 115 mm |
| No.10 | 29 mm | 16.2 / 15.5 | 25.3 | 121 mm |

Every box type clears the "< max opening − 1.5 cm" rule by a wide margin, so the
Section 3 gripper TODO can close in Phase 2.

## 4. Touch points recomputed (`phase1_logs/phase1_touch_fit.txt`)

Touch point = the lowest vertex of either jaw mesh, not the `gripperframe` site.
Frame: origin on the pan axis (38.8 mm in front of the base origin).

| hypothesis | heights range / std | table-edge angle | supply sides (ruler 20.9 × 19.2) | stack sides (ruler 21.4 × 14.6) |
|---|---|---|---|---|
| H0 plain deg→rad (reproduces `measurements.yaml`) | 15.0 / 4.1 mm | 83.5° (= 96.5°) | 21.5/20.6 × 17.2/14.3 | 17.5/18.9 × 14.4/15.1 |
| H1 zero = your homing pose | 43.8 / 13.9 | 88.3° | 20.8/20.4 × 18.9/19.2 | 21.7/21.0 × 14.1/14.6 |
| **H0 + base tilt** (1.35° about x) | **6.2 / 1.6** | n/a | n/a | n/a |
| tilt + fitted lift/elbow/wrist | 7.7 / 2.4 | 89.9° | 21.0/20.3 × 18.5/18.1 | 20.8/20.5 × 14.0/14.7 |

What the data says:
1. **The mount looks tilted about 1.4–1.6°** relative to the table. The base's left side
   (black mat) sits lower; seen from the arm, the table rises toward its left by about
   2.5 mm per 10 cm. Tilt alone, with
   no joint offsets, flattens the touch heights from 15 mm to 6 mm. Every variant I tried
   agrees (tilt_x −1.35 to −1.6°).
2. **Joint offsets:** the best fit is lift −6.9 ± 1.1°, elbow +4.5 ± 1.2°, wrist +9 to +13°.
   The lift value is stable across variants. The wrist value is suspect, because
   wrist_flex was 62.8° in every touch, so it's poorly separated from the tilt.
3. **Your homing pose is not the CAD zero** (H1 makes heights much worse). The LeRobot zero
   is your swept-range midpoint, so offsets of a few degrees are expected.
4. **Pan offset can't be identified** from table touches, and it doesn't need to be: the
   scene is laid out with the same mapping, so it cancels.
5. **H0 contradicts itself at home:** with the arm at your home pose, H0 puts the jaw tips
   0.9 cm *below* the table that the touches define. The fitted offsets put them 1.6–1.8 cm
   above it. A ruler reading at home will settle this (section 7, M1).

I haven't applied any offset or tilt yet. The data can't separate elbow and wrist offsets
from tilt well enough, and a wrong offset in the mapping looks like a bad policy later
(Section 9 pitfall). Section 7 gives the measurement that pins them down.

## 5. Visual check: home pose vs `front_cam_frame.png`

`scripts/render_pose.py`, figures `notes/img/phase1_home_H0.png`, `phase1_home_fit.png`
and `phase1_home_compare.png`.

- **Rough camera:** fitted to 10 hand-picked table and tape corners, with focal length,
  bounded distortion, and position held near your tape value. Result: 14 px RMS, f ≈ 783 px,
  position 5 cm from the tape value. The corners themselves come from H0 FK and are off by
  up to 3 cm (FK put the stack's long side at 18.3 cm against 21.4 cm by ruler), so a
  precise overlay isn't possible yet. This is a Phase 3 job, after better positions.
- **Joint signs are right:** the sim home pose has the same shape as the real one (folded
  arm, upper arm up, jaws pointing down at about the same image spot).
- **wrist_roll is probably about 90° off.** In the real frame both jaws are side by side
  facing the camera. In the sim (offset 0) they're edge-on. A +90° or −90° offset fixes
  the look (scratch comparison), and +90° also puts the camera box in front, as in the real
  image. The touch data can't tell +90° from −90°. This likely means the gripper was rolled
  90° from the CAD pose when you calibrated; LeRobot's wrist_roll zero is the homing roll.
- From this view, a wrist_flex offset mostly moves the jaws toward or away from the lens,
  so it can't be judged here.

## 6. Grasp/pose safety tool: `scripts/phase1_test_poses.py`

Writes the diagnostic poses `notes/phase1_poses/{R_raise,P1_over_stack,P2_side_raised,P3_over_stack_roll45}.json`
in the home-file format, so `so101_go_home.py --home-file` can drive to them. It checks
every straight joint-space leg (home → R_raise → P → R_raise → home) under **all five open
hypotheses** (H0, tilt, fitted with roll 0/+90/−90). The lowest arm point stays ≥ 10.6 cm
above the table away from home, and the first leg only rises (`phase1_logs/phase1_test_poses.txt`).
Not checked: self-collision and cables.

## 7. What I need from you (real arm; you run everything)

> **Update 2026-10-20:** M1 is done (**2.4 cm**). It rules out the plain mapping and
> matches the fitted offsets. The detailed, step-by-step instructions for M2–M4 (with
> mark diagrams, the corrected safety check that follows how `so101_go_home.py` really
> moves, and new mark positions) are in **`notes/phase1_measurement_guide.md`**. That
> guide replaces the short version below.

**M1. Ruler at home (1 min).** Run `so101_go_home.py`, then measure the height of the jaw
tips above the table. Predictions: H0 ≈ touching or pressing; fitted offsets ≈ 1.6–1.8 cm.

**M2. Level check (2 min).** With a phone level app, measure the table top and the top face
of the arm's base plate (or the extrusion it sits on), each in two directions: along the
arm's forward axis, and left to right. Prediction: the base is rolled about 1.4° relative
to the table, its left (black-mat) side lower.

**M3. Multi-pose touches (about 15 min).** Mark three Xs:
- A = the existing X on the stacking spot
- B = an X on a small piece of white tape near the far side of the black mat, about 25 cm
  in front of the arm and 15 cm to its left
- C = a pen X on the white table about 13 cm in front of the arm, slightly left

Measure **A–B, B–C and A–C with a ruler (mm)**. Then run your touch script in its new mode:
```
conda activate lerobot
python scripts/so101_record_points.py --port /dev/tty.usbmodem5B910311481 --id my_follower_arm --points multipose --out inputs/multipose_points.json
```
Touch each mark three times with the jaws closed, using the wrist key (r/f) to change the
tilt between touches: **steep** (jaws near vertical), **mid**, **flat** (as tilted as you can
while the tips still touch the X). The exact angles don't matter, only that they differ.
At C the arm can't tilt far; do what it allows and skip (k) any touch it can't reach.
Same rules as before: barely touch, don't press.

Simulated with 2 mm placement and 0.2° encoder noise, this protocol recovers lift, elbow
and wrist offsets to **±0.8–1.0°**, against ±1–9° from the existing data
(`phase1_logs/phase1_protocol_sim.txt`).

**M4. Diagnostic poses with photos (about 10 min, first real motion of this project).**
Clear the boxes from under the arm. Keep Ctrl-C ready (it stops motion) and use a slow
speed. For each P in P1, P2, P3:
```
python scripts/so101_go_home.py --port /dev/tty.usbmodem5B910311481 --id my_follower_arm --speed 15 --home-file notes/phase1_poses/R_raise.json
python scripts/so101_go_home.py --port /dev/tty.usbmodem5B910311481 --id my_follower_arm --speed 15 --home-file notes/phase1_poses/P1_over_stack.json
python scripts/so101_capture_frames.py --cameras front=0 wrist=1 --out inputs/photos/phase1 --suffix P1
python scripts/so101_go_home.py --port /dev/tty.usbmodem5B910311481 --id my_follower_arm --speed 15 --home-file notes/phase1_poses/R_raise.json
```
(Replace P1 with P2 and P3 in the second and third lines.) Then go home:
```
python scripts/so101_go_home.py --port /dev/tty.usbmodem5B910311481 --id my_follower_arm --speed 15
```
Also tell me for P3: seen from the front camera's side, did the jaws turn **clockwise or
counter-clockwise** going from P1 to P3? That fixes the sign of the wrist_roll offset.

## 8. Open issues
1. Arm offsets (lift, elbow, wrist) and base tilt: pending M1–M3.
2. wrist_roll offset ±90°: pending M4 (P1 vs P3).
3. Wrist camera housing differs from the official mount; model it in Phase 3 from photos
   plus a few ruler measurements.
4. Sim joint ranges are narrower than the real ones; widen them in the Phase 3 scene.
5. Touch-point corner positions are only good to ±3 cm. Once offsets are fixed, recompute
   them; if they're still poor, take a ruler-based layout from the top-down photo.

## Next step
Once M1–M4 are in: fit the offsets and tilt (`scripts/fit_joint_offsets.py`, to be written
on that data), set `joint_mapping.yaml` to `status: validated`, recompute the table and mat
layout, and redo the home and P1–P3 overlays. Then Phase 2.

## 9. M1–M3 results (2026-10-20)

Inputs: `notes/phase1_measurements.yaml` (M1 2.4 cm; M2 every phone-level reading 0°;
M3 mark distances 295/185/140 mm) and `inputs/multipose_points.json` (9 touches, none
skipped, wrist_flex −89° … +95°). Fit: `scripts/fit_joint_offsets.py`, log
`phase1_logs/fit_joint_offsets.txt`.

**Model fix found on the way.** At steep tilts (e.g. B_flat, 71°), the sim's lowest point
was the gripper body 6 cm behind the tips. Taking contact as "lowest vertex anywhere on the
jaws" gave 5–56 mm residuals. The contact is now the lowest vertex within the last 9 mm of
either jaw, which is what you touch with. Residuals dropped to 3–15 mm in xy and 0–5 mm in z.

| parameter | fit (± 1σ, inflated by the residual size) |
|---|---|
| shoulder_lift offset | **−5.6 ± 2.5°** (same in every wrist_roll variant) |
| elbow_flex offset | **+6.8 ± 2.2°** |
| wrist_flex offset | +4.6 ± 1.3° (if wrist_roll ≈ +95°) or +8.0° (if wrist_roll ≈ +5°) |
| wrist_roll offset | two near-equal fits: **+95.4°** (cost 83.5) or **+4.7°** (85.9); −90° fits worse (110.6) → **M4 decides** |
| base tilt | −1.1 ± 1.2° side to side, +0.5 ± 1.5° front-back |
| mat + tape under B | +2.7 mm (fitted; please confirm the thickness) |

**Checks on data the fit didn't use:**

| check | plain mapping (H0) | fitted |
|---|---|---|
| M1 home tip height (measured 24 mm) | −2.5 to −9 mm | **25.8 mm** |
| old touches: table corner (90°) | 83.5° | **89.8°** |
| old touches: stack area sides (ruler 21.4 × 14.6) | 17.5/18.9 × 14.4/15.1 | **20.9/20.6 × 14.3/14.5** |
| old touches: supply sides (ruler 20.9 × 19.2) | 21.5/20.6 × 17.2/14.3 | **21.2/20.0 × 18.8/18.6** |
| old touch on the stacking-spot X vs fitted mark A (same X) | 3.5 cm apart | **0.5 cm apart** |
| old touch heights | 15 mm range | 10 mm range, but on average **+7 mm above** the M3 table plane (open) |

**Consequences:**
- `sim/assets/joint_mapping.yaml` now holds these values, with `status: fitted_pending_M4`.
- Every position in `measurements.yaml` came from the plain mapping and is about 3 cm too
  close to the arm. Phase 3 must recompute the table, mat and stack-spot layout from the
  old touches with the fitted mapping (the side lengths above show it works).
- The residuals are about 3× the noise I assumed (touching at steep tilts is harder than
  2 mm), so the offsets are good to about ±2.5°, not the ±1° the simulation promised.
  Over this workspace that's about ±1 cm of tip position; the Phase 3 camera overlays and
  the M4 photos will tighten it.
- Open: the old touches sit about 7 mm higher than the new ones. Possible causes: lighter
  touching in the old session, or something that changed between sessions (base seated
  differently?). Not explained yet.
- M4 safety was re-checked with both fitted candidates: ≥ 10.8 cm clearance.

## 10. M4 results and Phase 1 close-out (2026-10-20)

Photos: `inputs/photos/phase1/{front,wrist}_cam_P{1,2,3}.png`, taken with
`scripts/phase1_m4_run.sh`. All three poses ran normally.
Comparison: `notes/img/phase1_m4_roll_compare.png` (real | sim +95 | sim +5) and
`phase1_m4_P*_final.png`.

- **Absolute wrist_roll: +95°.** In the front photos at P1 and P3, both jaws show as two
  separate prongs with the camera box above them, as in the +95° render. The +5° render
  shows the jaws edge-on in both poses. P2 agrees (one pointed jaw silhouette with the
  box on top).
- **Roll direction (sign check).** Between P1 and P3 (+45° command), the wrist image turns
  about 42–47° **clockwise**. Measured on the table corner, which has carpet on both sides
  (upper-right → right of the X), and on the tape edges (−11° → 122–135°). So the gripper
  turned counter-clockwise seen from above. The sim gives +43° counter-clockwise for the
  same command, so **the sign is the same; no flip**. My first eyeball read of these
  photos said the opposite; the measurement corrects it.
- **The wrist camera works:** the X is centred in the P1 wrist view, so the arm really is
  above the stacking spot.

**Final mapping** (`sim/assets/joint_mapping.yaml`, `status: validated`):

| joint | offset (q_sim = q_lerobot − offset) |
|---|---|
| shoulder_pan | 0 (defines the scene's +x) |
| shoulder_lift | −5.66 ± 2.5° |
| elbow_flex | +6.82 ± 2.2° |
| wrist_flex | +4.61 ± 1.3° |
| wrist_roll | +95.4° |
| gripper | q = −10.75° + 1.3029° · g |
| base tilt (scene) | −1.10° about x, +0.47° about y |

**Carried into Phase 3:**
1. Recompute the table, mat and stacking-spot layout from the old touches with this
   mapping. `measurements.yaml` positions are about 3 cm too close to the arm.
2. Model the real wrist-camera housing; the official mount doesn't match.
3. Widen the sim joint limits to the real calibrated ranges.
4. The front camera needs a proper calibration; the Phase 1 one is rough (14 px RMS).
5. Unexplained: the old touch session sits about 7 mm higher than M3; the mat under B
   fitted at +2.7 mm (thickness not yet confirmed).
