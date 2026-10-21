# Phase 2: Lock the game spec

Date: 2026-10-21. Status: **done, waiting at the checkpoint for your OK.** All Section 3
TODOs are settled except max stack height, which Phase 3 computes.

## 1. Box fit vs gripper (Phase 1 numbers)

Rule: box width < max opening − ~1.5 cm, for every type.

| box | width | opening needed (+1.5 cm) | gripper reading at that opening (sim) | max opening | spare |
|---|---|---|---|---|---|
| No.3 | 40 mm | 55 mm | 32.2 | 150 mm (ruler) / 154 mm (sim) | 95 mm |
| No.35 | 35 mm | 50 mm | 29.1 | 150 mm | 100 mm |
| No.10 | 29 mm | 44 mm | 25.3 | 150 mm | 106 mm |

**Every type fits easily, so no new box dimensions are needed.** The gripper only needs
readings of roughly 25–33 for the pre-grasp, far from fully open, so the jaws never swing
wide near the stack.

## 2. Decisions (recorded in CLAUDE.md Section 3 and the Decisions log)

| item | decision | why |
|---|---|---|
| box contents | **all full** | matches the measured masses (38/31/21 g) and your Phase 4 stability tests, so nothing is redone |
| box orientation | **always lying flat**, largest face down, in the supply and on the stack | one grasp style (top-down, across the width); matches the photos and tests |
| turn order | **human starts** every game | simplest; the robot never places the first box onto the empty base |
| contrast fix (3a) | **matte black backdrop** on the table's right edge | see section 3 |
| max height | **open, computed in Phase 3** | needs the reach and camera-view calculation |

Defaults kept unchanged: human turn ends on a keypress; robot turn ends when the box is
released and the gripper is raised (or timeout); 3 s settle window; the lose rules; the
human may take any box; robot box choice as in Section 3c.

## 3. Contrast: why a backdrop, and where it goes

`notes/phase2_background_check.txt` uses the corrected Phase 1 layout and the
tape-measured front-camera position. It traces the line of sight past the top of each
stack height:

| stack height | what's behind the top box in the `front` view |
|---|---|
| 1–3 boxes | yellow stack area (good contrast) |
| 4–8 boxes | **off the table: the white panel**, and you as the human player, since you stand on that side |

White boxes on a white panel is the low-contrast case Section 3a warned about, and a
person in the background makes the stack and collapse perception harder too. A backdrop
fixes both. From the camera's view it hides the panel and your body, and your hand
crossing it becomes an easy hand-in-workspace signal (Phase 7).

**Backdrop spec:**
- **Material:** matte black (foam board, card or fabric on a frame), not glossy.
- **Where:** standing upright on or against the table's **right edge**, the edge running
  from the arm's corner to the yellow-side corner. Cover from about **10 cm from the
  arm's corner** to the **yellow-side corner**, and wrap **~5 cm around that corner**.
  The sightlines cross between 13 and 37 cm along the 38 cm edge; the extra margin covers
  the uncertainty in the camera position.
- **Height:** at least **20 cm** above the table top (16 cm needed, plus margin).
- **Fixed:** taped or clamped so it sits in the same place every session. It's part of the
  calibrated scene from now on, like the tripod.
- **You reach over it** to place boxes. If that's awkward, tell me; a lower board plus
  black cloth on the panel behind is the fallback.
- **Arm clearance:** the board is at the table edge, outside the arm's path to the stack
  and the supply. Phase 3 puts it in the sim scene and checks every motion against it.

## 4. Before Phase 3 (you)

1. Install the backdrop as above.
2. Retake the reference frames with the **final** scene: backdrop in place, boxes off the
   table, arm at home, tripod untouched.
   ```
   conda activate lerobot && cd ~/so101_stack_game
   python scripts/so101_go_home.py --port /dev/tty.usbmodem5B910311481 --id my_follower_arm --speed 15
   python scripts/so101_capture_frames.py --cameras front=0 wrist=1 --out inputs/photos --suffix frame
   ```
   Then build the mixed 3-box stack (No.3 bottom, No.35, No.10 top) on the stacking spot:
   ```
   python scripts/so101_capture_frames.py --cameras front=0 --out inputs/photos --suffix stack3
   ```
   This overwrites `front_cam_frame.png`, `wrist_cam_frame.png` and `front_cam_stack3.png`;
   the old ones are in git if needed.
3. Optional: a phone photo of the backdrop from above with the ruler visible, so I can
   place it in the sim (`inputs/photos/backdrop_topdown.jpg`).
4. Open from Phase 1: the thickness of the black mat (B fitted at +2.7 mm), and whether the
   arm was re-seated between the 19 Oct and 20 Oct touch sessions (they differ by ~7 mm).

## Next step
Phase 3: scene twin and camera calibration, after your OK and the new frames.
