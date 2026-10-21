# Phase 3: Scene twin and camera calibration

Date: 2026-10-21. Status: **at the checkpoint, with one decision needed (max stack height)
and the wrist camera still to calibrate.** Front camera, layout, scene and box models are
done.

## 1. Reference frames (2026-10-21, backdrop installed)

| frame | verdict |
|---|---|
| `front_cam_stack3.png` | good: 3-box stack in order; tripod unmoved (black-mat corners on identical pixels in both frames) |
| `front_cam_frame.png` | **retake**: scissors on the black mat |
| `wrist_cam_frame.png` | **retake**: M3 mark C (blue pen X) still on the table; geometry is fine for calibration |

## 2. Layout recomputed with the validated mapping (`scripts/scene_layout.py` → `sim/assets/layout.yaml`)

- **Mats:** ruler-size rectangles fitted to the old corner touches; corner residuals
  1–8 mm. Supply centre moved 3.0 cm and stack centre 3.1 cm from `measurements.yaml`
  (stacking spot 3.6 cm), as Phase 1 predicted. The stacking spot is now 2 mm from the
  yellow area's centre.
- **Table:** placed from the image, not the touches. The edge touches sit inside the real
  edge, and with the touch-based table the yellow area overhung it by up to 2 cm. Two
  table corners picked in `front_cam_frame.png` and back-projected with the calibrated
  camera come out 39.4 cm apart (true 38.0). With the table placed this way, the black mat
  sits flush with its edge to 3–6 mm, as in reality (`notes/phase3_layout.txt`, `calibrate_cameras.py`).

## 3. Scene (`scripts/build_scene.py` → `sim/scene.xml`)

- **Arm:** official SO-101 (camera variant) with the validated base tilt, the pan axis at
  the origin, and joint ranges = **your calibrated ranges** mapped into sim (all wider
  than the vendor's). There's a `home` keyframe from your home pose, and the arm touches
  nothing at home.
- **Table and play area:** table 38 × 38 cm (white), black supply mat, yellow stack area,
  stacking-spot site, and the **backdrop** on the right edge (Phase 2 spec, collidable).
- **Boxes:** 8 free boxes (3 No.3, 4 No.35, 1 No.10) with measured sizes and masses and
  **photo textures**. Faces are cropped from `inputs/photos/box_faces/` by
  `scripts/make_box_textures.py` (`notes/img/phase3_box_textures.png`). The MuJoCo cube-face
  mapping and orientation were verified with a marker test, not assumed
  (+x right, −x left, +y up, −y down, +z front, −z back, with per-face pre-rotation).
  Settled after 1 s, every box rests at exactly mat thickness + half its height.
- **Physics:** friction values are placeholders until Phase 4.
- **APPROX items** (visual or placeholder, listed in `build_scene.py`): mat thickness
  2 mm, backdrop size (20 cm tall, starting 10 cm from the arm's corner, 5 cm wrap), table
  legs and shelf, carpet floor, no wall or panel yet, the official wrist-camera mount
  (yours is larger), lights.

## 4. Front camera: calibrated

`scripts/calibrate_cameras.py` → `sim/assets/cameras.yaml`.
- **Points:** 8 mat corners (detected by colour, sub-pixel line intersections), the jaw
  tips at home (3D, so not all points are on the table plane), and a 3 cm prior on your
  tape position.
- **Result:** **2.73 px RMS**, f = 656 px (fovy 40.2°), **k1 = −0.139**. k2 was dropped:
  freeing it didn't improve the fit and it ran to its bound. The camera sits 2.4 cm from
  the tape-measured position. Phase 1's rough fit was 14 px.
- **Overlays** (`notes/img/phase3_front_empty.png`, `phase3_front_stack3.png`;
  real | sim | 50 % blend | sim edges on real): mats, table edges and the 3-box stack line
  up; the arm lands within ~10–15 px. The backdrop's top edge renders a little high, and
  the wrist camera housing differs (section 7).
- **Distortion:** MuJoCo renders pinhole only. **Proposal:** undistort the real `front`
  frames with (f, k1) before they reach the policy, and train on pinhole sim renders. This
  is one `cv2.undistort` per frame at runtime. The alternative is distorting every sim
  frame during generation.

## 5. Wrist camera: NOT calibrated yet

- **Point fit:** fitting to table features fails (30 px). At home the lens is about 6 cm
  above the table, so the arm's remaining ±1 cm position uncertainty moves table features
  by up to 100 px.
- **Silhouette fit:** fitting to the jaw silhouettes (camera-to-jaw geometry only, no arm
  error) didn't converge either: overlap 0.66–0.70, then 0.33 at home with other starting
  points. With focal length, distortion and 6-DoF pose all free, there are too many
  unknowns for silhouettes alone.
- **The measurement is probably off:** the real frame looks like the lens sits near the
  jaw **roots** (about 8–10 cm back from the tips; the free fits land at 8.7–10.3 cm),
  not 3.6 cm back as in `measurements.yaml`.
- **Fix (standard):** get the lens intrinsics from a **printed checkerboard** (OpenCV),
  then fit only the 6-DoF mount from the jaw tips and silhouettes.
  `scripts/capture_checkerboard.sh` captures the frames.

## 6. Size discrimination (Section 3d; `notes/phase3_size_check.txt`)

- **Policy input sizes:** installed LeRobot **SmolVLA resizes to 512 × 512 with padding**
  (`policies/smolvla/configuration_smolvla.py:45`), not 256 as the brief assumed; a
  640 × 480 frame scales by 0.8. ACT has no resize setting (native 640 × 480).
- **Front view:** No.3 vs No.35 differ by **5.7–6.0 px in length and 3.8–3.9 px in width**
  (4.6 / 3.0 px at SmolVLA resolution). **Flagged: hard to separate by size alone.**
  The prints differ clearly (No.35 has the orange MAX/Thai band), so appearance will
  carry more than size; keep texture randomization moderate (hue shifts, not full
  recolouring) so the print stays informative.
- **Wrist view:** pending the wrist calibration (expected several times larger near the grasp).

## 7. Max stack height — SUPERSEDED by section 10 (the numbers below were too pessimistic) (`scripts/max_height.py`, `notes/phase3_*height*.txt`, `phase3_vertical_reach_map.txt`)

**Main finding: the arm can't keep the jaws vertical high above the stacking spot.**
wrist_flex runs into the top of its range (97° in sim = your calibrated +101.9°). Over the
spot (25.5 cm from the pan axis), the jaw tilt needed rises about 4° per box: 1–7° for
0–2 boxes, 10.6° for 3, 14° for 4, 31° for 8.
- A **tilted grip isn't a workaround.** The box would be held tilted by the same angle,
  unless it was also picked up tilted, which forces every supply box to point at the arm.
- Your M4 photos agree: at P1, ~12 cm over the stack, the jaws were tilted 25°.
- **The camera view isn't the limit:** stacks up to 19 boxes fit in the front image.

Highest stack the robot can place onto, allowing ≤ 10° jaw tilt, a 3 cm straight-down
approach and ±2 cm stack offsets:

| option | place onto stacks of up to | physical change |
|---|---|---|
| as now (spot 25.5 cm out) | **3 boxes** (4-box stacks) | none |
| move the stacking spot to 23 cm (5 cm inside the yellow) | **4** | none; new X on the yellow |
| riser 3 cm under the arm base | **5** | riser, then redo base pose (touches) + reference frames |
| riser 6 cm | **6** | same |
| riser 9 cm | **8** | same |

(The riser rows don't yet check the forearm against the table edge.)

Also: about **14 % of the black mat** (the part more than ~31.6 cm from the arm, near its
far corner) can't be picked from with vertical jaws, whatever the riser height. Either
keep boxes out of that region or move the mat about 4 cm closer.

Section 1 asks for ≥ 70 % success on crooked stacks of height 4+, so the current geometry
(4-box stacks max) barely touches that criterion. A riser of 6–9 cm makes the game and
the criteria meaningful.

## 8. What I need from you

1. **Decision: max height / riser** (section 7). My recommendation: a **rigid 6–9 cm
   riser** under the arm base. Before that, I check the forearm against the table edge
   for the riser height you can build.
2. Retake `front_cam_frame.png` and `wrist_cam_frame.png` (scissors off, mark C wiped):
   `python scripts/so101_capture_frames.py --cameras front=0 wrist=1 --out inputs/photos --suffix frame`.
   If you choose a riser, do this after installing it.
3. **Wrist intrinsics:** print a checkerboard (9 × 7 squares of 20 mm, 8 × 6 inner
   corners, at 100 % on A4, glued flat), then run `bash scripts/capture_checkerboard.sh wrist 1`.
   Take 15–25 shots at different distances (5–25 cm), tilts and image positions.
4. **Measurements:**
   - wrist camera: distance from the lens to the fixed jaw tip, plus a side photo of the
     gripper and camera housing with a ruler (the 3.6 cm figure looks wrong);
   - housing size (L × W × H);
   - backdrop: length, height, and where it starts along the edge from the arm's corner;
   - black mat thickness.

## Next step
After the decision and the inputs: (re)measure the base pose if there's a riser, fit the
wrist camera, redo the overlays and the wrist size check, then Phase 4.

## 9. Riser decision — WITHDRAWN, see section 10 (2026-10-21; `scripts/riser_check.py`, `notes/phase3_riser_check.txt`)

You chose a riser. Each height was checked in sim with full collision checking of every
arm link against the table, mats and backdrop:
- **place:** all 3 box types onto stacks of 0–8, offsets 0 / ±2 cm, 3 cm approach, jaws within 10° of vertical;
- **pick:** a 7 × 7 grid across the black mat;
- **home pose.**

| riser | place onto (worst type) | max game stack | arm links touching anything | supply pickable | front view at the highest pose (gripper / whole arm, top pixel row) |
|---|---|---|---|---|---|
| 6.0 cm | 6 | 7 | none | 86 % | 9 / 9 |
| 7.5 cm | 7 | 8 (no margin) | none | 86 % | 9 / 2 |
| **8.0 cm** | **8** | **8, with one box of margin** | none | 86 % | **9 / −1** |
| 9.0 cm | 8 | 8, with one box of margin | none | 86 % | 10 / −5 |

**Recommendation: 8.0 cm (± 3 mm).** It's the lowest riser that keeps one box of reach margin
for the ±2.5° mapping uncertainty and crooked stacks. The gripper stays well inside the
`front` view; only the top of the upper arm touches the frame edge at the single highest
pose. 9 cm buys nothing extra and pushes the arm further out of view.

Unaffected by the riser: about 14 % of the black mat (more than ~31.6 cm from the arm)
stays unpickable. Keep boxes out of the mat's far corner region, or move the mat ~4 cm
toward the arm while it's re-taped anyway.

## 10. Correction: max stack height reassessed (2026-10-21)

You challenged "only about 4 boxes", and it was wrong. Re-checked as follows
(`notes/phase3_reach_bruteforce.txt`):
1. **Exhaustive sweep:** every shoulder_lift × elbow_flex combination in 1° steps (with
   the arm pointed at the spot), with the wrist set to make the jaws vertical, verified by
   the gripper's full orientation. A first version of the sweep had its own bug (it
   counted jaws pointing straight **up** as vertical, giving a false 42 cm) and was fixed.
2. **The physical limit is real but smaller than claimed:** the wrist reaches its
   calibrated maximum, +101.9° in LeRobot units, which LeRobot writes into the servo's
   position limits (`motors/feetech/feetech.py:272-273`). Beyond about 7 cm grip height
   over the spot, the jaws therefore tilt.
3. **What made it look like 3–4 boxes:** requiring the jaws to be vertical already 3 cm
   above the stack (the approach can come in tilted and straighten while descending), and
   requiring the worst ±2 cm stack offset to pass at the same time.

Highest stack the robot can place onto (place pose, no riser; stack box height 1.53 cm on average):

| release tilt allowed | spot as now (25.5 cm) | spot at 23 cm | spot at 20 cm |
|---|---|---|---|
| ≤ 5° | onto 5 | onto 5 | onto 6 |
| ≤ 10° | **onto 6** (7-box stack) | **onto 7** (full 8-box game) | onto 7 |
| ≤ 15° | **onto 7** (full game) | onto 8 | onto 8 |
| ≤ 20° | onto 8 | onto 9 | onto 9 |

**Conclusion: no riser needed.** The full 8-box game works without one if either
(a) the stacking spot moves about 2.5 cm toward the arm (to 23 cm, still 5 cm inside the
yellow area) with ≤ 10° release tilt, or (b) the spot stays put and ≤ 15° release tilt
is acceptable. Whether a 10–15° tilted release keeps real stacks standing is a Phase 4
question: sim drop tests on the calibrated physics, then a few real checks.

## 11. New stacking spot at 23 cm (user decision 2026-10-21)

- **Position:** the game's stacking spot is moved 2.5 cm from the touched X toward the pan
  axis, to (20.9, −9.7) cm in the arm frame (was (23.2, −10.7)). It's 5.0 cm from the
  nearest yellow edge (was 7.2). `scene_layout.py` keeps both (`stacking_spot`,
  `stacking_spot_original`); photo-based checks taken before the move (the wrist
  calibration and the 3-box-stack overlay) use the original.
- **Reach** (`notes/phase3_new_spot_check.txt`, no riser, full IK with collisions): release
  tilt is 1–2° up to 5-box stacks and 6° onto 6. The robot's last placement in a full
  game (onto 7) needs 10.7° for a No.3, 10.4° for a No.35 and 9.6° for the No.10.
  22 cm would bring that to 9.4–9.7°; that's within the model's uncertainty, so 23 cm is kept.
- **Backdrop:** sightlines past 3–8-box stacks cross the right edge 14–35 cm from the arm's
  corner, up to 13.2 cm high, so the existing backdrop still covers them.
- **Check poses:** `notes/phase3_poses/S1_above_new_spot.json` (7 cm) and
  `S2_hover_new_spot.json` (closed tips 3 cm above the new X). Safety follows how
  `so101_go_home.py` actually moves: the arm stays ≥ 2.1 cm above the table on the
  validated mapping, ≥ 1.6 cm under every hypothesis. Guided run:
  `scripts/phase3_spot_check.sh`.

## 12. Spot check result (2026-10-21, `inputs/photos/phase3/*_spotcheck.png`) — partly WITHDRAWN, see 12b

- **The new X is where the sim expects it:** in the front photo it lands within 2 px of the
  projection (~3–4 mm), and the old X is covered.
- **The hovering jaw tips miss it by 0.8 cm** (you estimated ~1 cm): **+0.6 cm sideways
  toward the arm's left** (reliable) and −0.6 cm toward the arm (±0.5 cm, since it depends
  on the assumed 3 cm hover height). The sideways part is about 1.5° of pan.
- **Likely cause:** the layout comes from the 19 Oct touches and the joint offsets from the
  20 Oct M3 touches, and those two sessions already disagreed by 7 mm in height. A small
  shift or turn of the arm base between sessions explains both.
- **Fix:** redo the 13 table touches in the current setup (stacking_spot = the new X), then
  recompute the layout and the front-camera table placement, and repeat the spot check.
- **Also seen:** the backdrop is missing from this photo; asked the user.

### 12b. Correction (same day)
- **Base moved?** No: you confirm the arm base was never moved, so the "arm re-seated
  between sessions" explanation is withdrawn.
- **Front photo:** you say it wasn't updated. It does show the arm in today's hover pose
  and no backdrop, so the camera index or the backdrop at capture time needs checking.
  The front-photo measurement (0.8 cm, +0.6 sideways / −0.6 radial) is withdrawn. What
  stands is your ~1 cm estimate, direction pending.
- **Working explanation:** the ordinary residual error. That's the Phase 1 mapping
  (±2.5° ≈ ±1 cm at the tip) plus the imprecise 19 Oct layout touches (inside the edges,
  ~7 mm high).
- **Fix:** a fresh 13-point touch session, with the stacking-spot point taken on the new
  X so the spot is exact in the arm's own kinematics.

## 13. 21 Oct touch session (`inputs/table_points_20261021.json`, `notes/phase3_touches_20261021.txt`)

- **All 13 points recorded.** Point 4 (right edge, far) was taken next to the yellow-side
  corner, beside point 12.
- **The arm base didn't move** (confirmed by data): the best rigid fit between the 19 Oct
  and 21 Oct touches of the same features is a 0.36° turn and a 2 mm shift. 10 of the 11
  mat corners repeat to within 2–5.5 mm.
- **Outlier excluded:** the new `stack_corner_2` touch is 17 mm *below* the table and
  16 mm from its 19 Oct position (pressed or deflected). The layout uses the 19 Oct touch
  for that corner.
- **Layout:** per-feature average of both sessions, with the **stacking spot = the touched
  new X**, (20.6, −10.1) cm, r = 22.9 cm. Mats moved ≤ 2 mm; corner fits 1.5–7.4 mm. The
  layout's earlier prediction of the new X was 5.4 mm off, almost all sideways (toward
  the yellow side). That's about half of the ~1 cm spot-check miss; the rest depends on
  the arm pose (the touch used wrist ≈ 49°, the hover ≈ 80°).
- **Front camera re-fit** on the new layout: 2.60 px RMS, f 654 px, k1 −0.122. Table re-placed
  from the image; overlays still line up.
- **Hover poses regenerated** for the touched X (`scripts/make_spot_check_poses.py`): the
  arm stays ≥ 2.1 cm above the table (validated mapping), ≥ 1.6 cm under any hypothesis.
- **Next:** rerun `scripts/phase3_spot_check.sh`. Expect the miss to drop to about 5 mm
  or less; report the direction this time.
- **Spot check rerun (2026-10-21):** you measured the hovering tips about **5 mm** from the
  new X, down from ~1 cm, as predicted once the spot came from the touch. The remainder is
  the pose-dependent mapping residual (touch at wrist ≈ 49° vs hover ≈ 80°), within the
  ±2.5°/±1 cm Phase 1 accuracy. Backdrop confirmed present.

## 14. Reference frames retaken (2026-10-21 13:09)

| frame | verdict |
|---|---|
| `front_cam_frame.png` | **good**: backdrop present, table empty, nothing stray |
| `wrist_cam_frame.png` | **retake**: M3 mark C (blue pen X) still on the white table, centre of the view |
| `front_cam_stack3.png` | **retake**: the top box looks like a 2nd No.35 (≈ 95 % of the middle box's length, green top) rather than the No.10 (≈ 85 %, blue top); the stack sits ≈ 1.5 cm off the new X toward the camera-left |

Front camera re-fit on the new frames: **2.47 px RMS**, f 657 px, k1 −0.108, 2.2 cm from
the tape position.
- **Retake 13:16:** `wrist_cam_frame.png` is now clean (mark C gone). `front_cam_stack3.png`
  has the right order (No.3 / No.35 / No.10). The stack still sits ≈ 1–1.5 cm off the sim
  stack in the image and looks slightly rotated (bottom box ≈ 11 % longer in the image
  than the aligned sim No.3). The X itself, seen in `front_cam_frame.png` at
  (207.5, 295.4) px, is **2.3 px (≈ 4 mm) from the projection of the touched X**, so camera
  and arm kinematics agree. The stack offset is hand placement; accepted, since this frame
  is a visual reference for box size and print, not position.
- **Front camera** on the final frames: 2.48 px RMS, f 654 px, k1 −0.086, 2.3 cm from the tape.

**Reference frames are final** (empty front, empty wrist, 3-box stack), with the backdrop
in place and the new X marked.

## 15. Wrist camera identified: Orbbec Gemini 2 (user, 2026-10-21)

From Orbbec's product page and datasheet (orbbec.com/products/stereo-vision-camera/gemini-2,
ORBBEC_Datasheet_Gemini-2-1.pdf):
- **Housing:** 90 × 25 × 30 mm, 98 g. This is the large black box on the gripper; it
  replaces the official mount in the sim, and its mass belongs on the gripper.
- **RGB view:** 86 × 55° in 16:9 mode, **63 × 50° in 4:3 mode**. The 640 × 480 stream is 4:3,
  so **f ≈ 518 px** (522 / 515). The earlier failed fits had f = 328 / 421 px.
- **Depth:** range 0.15–10 m, so it's no use near the grasp (the lens is ~6 cm from the table at home).
- **Focus and lens distortion:** not published, so the checkerboard is still needed to
  confirm f and measure distortion.
- **Wrist silhouette fit re-running** with f held to 518 ± 26 px
  (`notes/phase3_calibration_wrist_fspec.txt`).
- **Checkerboard distance corrected:** the board must be 15–35 cm from the lens (fully in
  view at ~63° horizontal), not 5–25 cm.
- **Result of the re-run:** f = 516 px (held by the datasheet prior), k1 0.03, but jaw
  overlap is still poor (home 0.31, P1 0.68, P3 0.67), with the lens 8.6 cm from the tip and
  the optical axis 45° off the jaws. Silhouettes alone don't pin the mount pose, even
  with f fixed. **Stopping blind fits:** the mount will be fitted from the measured lens
  position (side photo + ruler) with checkerboard intrinsics, and the silhouettes used
  only as a check. The wrist camera stays marked uncalibrated.

## 16. Measurements received (2026-10-21; `notes/phase3_measurements.yaml`)

- **Backdrop:** 18.5 cm tall, 28 cm long, from 10 cm along the right edge to the corner,
  no wrap. Now in `build_scene.py`. The overlay top edge matches (it was a bit high at
  20 cm). The tallest stack sightlines cross it at up to 13.2 cm, so it's still covered.
- **Black mat:** thickness negligible, modelled as 0.5 mm.
- **Wrist lens offset:** `measurements.yaml` (3.6 cm along the jaws from the fixed tip,
  1.3 cm sideways, angled in) is **not consistent with the images**:
  - the fixed jaw is ~36 px wide just below its tip in `wrist_cam_frame.png`; with the
    CAD section (7–10 mm) and the datasheet f (518 px), the lens is **~10–14 cm** from the
    tips (3.6 cm would give 100–145 px);
  - `gripper_open.jpg` shows the housing near the jaw roots, with the jaws extending
    ~8–10 cm beyond it;
  - the free fits gave 8.6–10.3 cm.
  Asked the user to re-measure (method in the chat reply).

## 17. Wrist camera from the re-measured lens position (2026-10-21; `scripts/wrist_from_measurement.py`)

- **Measured** (relative to the fixed jaw tip): 7.5–8 cm back along the jaws, 2.7 cm in the
  jaw plane away from the moving jaw, ~2 cm out of the jaw plane. f = 518 px (datasheet),
  k1 = 0 for now.
- **Fit:** orientation from the jaw-tip pixels, lens held to the measurement ± 1 cm. The
  out-of-plane side is **+y**, the side the vendor mount is on (jaw IoU 0.65 at home vs
  0.04 for −y). Tip pixels fit to 0.5–3.7 px, the lens moved 1.2 cm from the measurement,
  and the optical axis is 18° off the jaws ("angled in", as described).
- **Scene changes:** the vendor camera mount (official camera variant) is removed; it
  doesn't match and sat in front of the lens. An Orbbec Gemini 2 housing (90 × 25 × 30 mm,
  98 g, own body on the gripper) is added behind the lens. **APPROX:** the RGB lens is
  assumed at the centre of the housing's front face.
- **Still wrong:** near the lens, the gripper body renders much larger and rounder in sim
  than the slim black prongs in the real frame. The lens is only ~2.5 cm from where the
  jaws attach, so small direction errors change the view a lot. P1/P3 also differ because
  those photos predate the backdrop, and the moving jaw stays in the sim view when open
  (the real one leaves the frame).
- **Status: PROVISIONAL.** Waiting for the two side photos (lens position and facing
  relative to the gripper body) and the checkerboard (f, k1). Overlays in
  `notes/img/phase3_wrist_overlays.png` (`scripts/render_wrist_check.py`).

## 18. Wrist camera: side photos and scene-constrained fit (2026-10-21)

- **Side photos** (`notes/img/phase3_wrist_side_A_IMG1222.jpg`, `_B_IMG1224.jpg`): the Gemini 2
  lies across the top of the gripper, lens face toward the tips, its lower face ~8 cm
  from the jaw tip (confirms the 7.5–8 cm; the old 3.6 cm was wrong). Its long side is
  ~69° to the jaw line in the photo (~21° off square). The lens window itself isn't
  visible.
- **Silhouette refine from the measured start** (`scripts/wrist_refine.py`): jaw IoU
  0.75 / 0.41 / 0.39, but the P1/P3 view looked out toward the carpet, so the pitch was
  wrong. Jaw tips alone can't fix the pitch.
- **Scene-constrained fit** (`scripts/wrist_fit_scene.py`, saved): jaw tips plus the P1/P3
  X and yellow-corner pixels, with a 5 mm prior on the measured lens position.
  - Tips 2.8–4.4 px; scene points 17–32 px (≈ the arm's ±1 cm FK error at 15 cm).
  - **Jaw IoU 0.71 / 0.73 / 0.69**, consistent across all three frames for the first time.
    The P1/P3 views now look down at the yellow area as in the real frames
    (`notes/img/phase3_wrist_overlays.png`).
- **Lens position, fit vs measurement:** the fit wants back 6.8 cm (measured 7.5–8),
  in-plane +1.1 cm (measured −2.7, other side), out-of-plane **+5.9 cm** (measured +2.0).
  A 2 mm prior gives 6.3 / +0.8 / +5.0. Asked the user to re-check the out-of-plane
  distance; a housing on top of the servo would be ~5–6 cm out.
- **Open:** k1 (checkerboard), RGB lens position along the housing face, the hard
  housing shadow (single point light), and the open moving jaw still partly in the sim view.
- **Checkerboard pose:** at home the wrist camera sees only ~6 cm of table. Use `R_raise`
  (lens 33 cm up, looking 32° down toward the far corner, sightline reaches the table at
  61 cm), which leaves free air over the table for the board.

## 19. Checkerboard skipped (user decision, 2026-10-21)

- **Wrist intrinsics stay** f = 518 px (Gemini 2 datasheet, 4:3 mode) with k1 = 0. The
  scene-constrained fit is consistent with that (tip errors ≤ 4.4 px).
- **Mitigation in Phase 5:** randomize the wrist intrinsics per episode (f ± 5 %, small
  radial distortion |k1| ≤ 0.05) on top of the planned pose jitter (± a few mm / ± 1°),
  so the policy doesn't depend on the exact lens model.
- `scripts/capture_checkerboard.sh` stays in the repo, unused, in case it's needed later.
- **Still open for the wrist camera:** the lens out-of-plane offset (fit ~5–6 cm vs measured 2 cm).

## 20. Wrist lens position: measurement vs images (2026-10-21; `notes/phase3_wrist_hypotheses.txt`)

The user re-confirmed the lens is 2–3 cm out of the jaw-opening plane. With the lens
**fixed** at each hypothesis (only the orientation fitted):

| lens position from the fixed tip | jaw tips | table features (P1/P3) | jaw IoU |
|---|---|---|---|
| as measured (−2.7 in-plane, +2.5 out, 7.75 back) | 18–21 px | 140–240 px | 0.66 / 0.40 / 0.38 |
| swapped (−2.5, +2.7) | 16–21 px | 137–224 px | 0.66 / 0.44 / 0.41 |
| other side (−2.5 out) | 10–62 px | 166–431 px | ≈ 0.1 |
| free fit (+1.1, **+6.0 out**, 6.9 back) | **3–4 px** | **16–32 px** | **0.71 / 0.73 / 0.69** |

The home jaw tips depend only on camera-to-jaw geometry (no arm error), and at 2–3 cm no
orientation reproduces them. **Decision: keep the fitted position** (lens ≈ 9 cm from the
tips, ≈ 6 cm out of the jaw plane). The camera model must reproduce what the camera sees.
The difference from the ruler is probably a reference difference (jaw surface vs jaw
mid-plane, housing edge vs RGB window). Phase 5 jitter (± a few mm / ± 1°) covers the
remaining uncertainty.

## 21. Wrist size check (Section 3d; `notes/phase3_size_check.txt`)

Pre-grasp (jaws vertical, tips 3 cm above the box top, ~11 cm from the lens):

| box | length × width (native) | at SmolVLA 512 (× 0.8) |
|---|---|---|
| No.3 | 276 × 157 px | 221 × 126 px |
| No.35 | 235 × 140 px | 188 × 112 px |
| No.10 | 203 × 118 px | 162 × 95 px |

No.3 vs No.35: **41 px in length, 17 px in width**, against 5–6 / 4 px in the front
camera. The wrist view separates the types easily. All boxes are fully in view over the
supply and over a 3-box stack.

## Phase 3 checkpoint summary

| item | status |
|---|---|
| layout | both touch sessions; stacking spot = touched new X, 22.9 cm from the pan axis |
| scene | `sim/scene.xml`: measured backdrop, textured boxes, Gemini 2 housing, real joint ranges |
| front camera | **2.48 px RMS**, overlays aligned (`phase3_front_empty.png`, `phase3_front_stack3.png`) |
| wrist camera | f from the datasheet (no checkerboard, user decision), mount fitted; jaw IoU ≈ 0.7 in 3 frames (`phase3_wrist_overlays.png`) |
| size check | front: No.3/No.35 hard to separate (4–6 px); wrist: easy (17–41 px) |
| max height | **full 8-box game, no riser** (spot at 23 cm; releases ≤ 6° tilt up to a 6-box stack, ≈ 10–11° for the last placement); front view fits 19 boxes |
| reach caveat | ~14 % of the supply mat (> 31.6 cm from the arm) isn't vertically pickable; keep boxes out of its far corner |
| spot check | hovering tips ≈ 5 mm from the new X |
| open, low priority | RGB lens spot on the housing face, hard housing shadow, open moving jaw partly in the sim view |

**Next:** Phase 4 (physics and stack stability) after your OK.
