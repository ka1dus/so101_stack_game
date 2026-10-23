# Phase 7: Classical perception (plan, in progress)

Status 2026-10-23: D1–D3 approved. Sim test set built (§6); camera refit and supply-slot check work in sim (§7).
`collapse_check()` (§8), `drop_check()` (§9) and `hand_in_workspace()` (§10) built and scored in sim. Waiting on the user's recording session (§4).

## 1. What the inputs allow
- **`inputs/videos/stack_collapse.mp4`** (301 frames) and **`human_turn.mp4`** (451 frames): 640×480, 30 fps, recorded
  2026-10-20. That was **before** the backdrop, the mat expansion, the reach tape and the stacking-spot move, and the
  camera framing differs from the current `front_cam_frame.png`.
  - **Usable for:** timing logic (when a collapse starts and ends, how a hand enters, motion levels).
  - **Not usable for:** anything tied to the calibrated camera (workspace polygons, stack ROI, box-on-table regions).
  - **Consequence:** the Phase 7 exit test needs new clips of the current setup (§4).
- **Front camera calibration** (`sim/assets/cameras.yaml`): pose, f 654 px, k1 −0.086, 2.5 px rms. It lets us
  project any table-frame point (stack area, supply slots, mat corners, a box at level n) into the image, and
  render a matching sim view.
- **Template supply layout** (`sim/assets/supply_layout.yaml`): 8 fixed slots, each with a known type.
  - Printable stencil: `notes/supply_template.pdf`, from `scripts/make_supply_template.py`. Print it at 100 % and
    check the 100 mm bar with a ruler.
- **Scale at the stack:** about 1 px per mm in `front` (camera ~0.65 m away).
  - Box heights (17 / 16 / 13 mm) differ by only 1–4 px, so **height alone can't tell the types apart**.
  - Box lengths (70 / 60 / 52 mm) differ by 8–18 px. That is enough to check the top box, but not enough to rely on.

## 2. Proposed methods

### `stack_state()`: types per level from game history (D1)
- **Types:** the stack is only ever built one box at a time, so the controller knows each level's type if it knows
  which box was taken each turn.
  - **How to tell which box was taken:** a **slot-occupancy check** on the supply mat. Each template slot is a
    projected polygon in `front`. A printed box on the black mat has very high contrast, so "slot occupied" is a
    simple brightness/difference test against an empty-mat reference.
  - **Works for both players:** the human may take any box; the robot's policy may take a non-nearest box (14 % in
    sim).
  - **Cross-check:** the new top box's visible length must match the type (8–18 px apart).
- **Geometry (method a in CLAUDE.md):** difference against an empty-scene reference inside the projected stack ROI.
  - The backdrop gives a dark field behind levels 4+.
  - Silhouette top edge → stack height in mm; compared with the expected height (sum of the types' heights).
  - Top-box centre and yaw from the silhouette's top face.
  - **Output:** levels (type, nominal height), measured top height, top-box pose in table coordinates.
  - Only if (a) misses its targets: method b, a small segmenter trained on MuJoCo renders (keeps everything
    sim-trained).
- **What the game needs from this:** the controller judges on change, not on absolute accuracy.
  - The policy perceives the stack itself and doesn't take its input from here.
  - So the bar is: count and top pose stable over a settle window, and robust to the hand and the arm.

### `collapse_check()`: before/after plus the settle window
- **Before snapshot:** after the turn ends, with the hand or arm out of the workspace; compared over the 3 s settle
  window.
- **Collapse** if any of these holds:
  1. the silhouette top edge drops by more than about half the thinnest box (6–7 mm, about 6 px);
  2. the top box moves sideways by more than a threshold (start at 15 mm);
  3. a new foreground blob appears on the table outside the stack area and the supply. The boxes contrast poorly with
     white, so this uses a difference against the reference image, not colour.
- **Avoiding false calls (more important than speed):** judge only on frames with no motion in the workspace, require
  the change to persist for ≥ 0.5 s, and never decide from a single frame.

### `drop_check()`: gripper reading during the carry
- **Rule:** while carrying, the gripper reading should stay near the holding width for the picked type (about 1 unit
  per mm; widths 40 / 35 / 29 mm). A drop shows as the reading closing past `width − margin`, or a sudden step.
- **Tests:**
  - first on sim eval logs, which already contain the drop failures and their gripper traces;
  - then in Phase 8, from real carry readings (the arm moved by the user's commands).
- No vision needed. Optional extra check later: the wrist camera sees no box.

### `hand_in_workspace()`: motion inside a projected workspace polygon
- **Region:** the workspace polygon is the table area around the stack and supply, projected through the calibrated
  camera.
- **Motion signal:** frame differencing against a slowly updated background, plus a minimum blob area.
- **Robot turn:** the arm moves, so its pixels are masked by **rendering the sim twin at the current joint state** from
  the calibrated `front` camera (the Phase 3 overlays show the twin lines up to a few px), dilated by a margin.
- **Human turn:** the robot is frozen, so any motion counts.
- Skin colour is an optional extra cue, not the main one (lighting and gloves vary).

## 3. Development order
1. **Sim test set (no real data needed):** render `front` frames of game states with known ground truth:
   - stable stacks of height 0–8 with offsets;
   - supplies with k slots emptied;
   - collapsed states (a box pushed off, made with `stability_check.py`-style pushes);
   - randomized lighting and texture as in Phase 5.
   - Measure: slot-occupancy accuracy, height and top-pose error, collapse true/false call rates.
2. **Real frames:** run the same functions on the new reference frames and static stacks (§4). Compare the projected
   polygons and silhouettes with the images.
3. **Clips:** the timing logic on the old clips, then the full exit test on the new clips.
   - **Exit (CLAUDE.md):** zero false collapse calls on stable stacks; every collapse detected within the settle
     window; the hand detected whenever it enters.

## 4. Recordings needed (user runs these on the Mac; arm at home, torque on or off, **no robot motion**)
Camera setup exactly as in play: tripod on its tape marks, backdrop in place. Lock exposure, focus and white balance
if the camera allows; if not, note it. All commands run from `~/so101_stack_game` with the `lerobot` env active.
Clips go to a new folder `inputs/videos/phase7/`.

- **Reference frames:**
  - (a) template supply full, empty stack area;
  - (b) mat empty, stack area empty.
  - The commands save `front_cam_<suffix>.png` in `inputs/photos/phase7/`.
- **Static stacks** of height 2, 4 and 6 (mixed types; write down the order bottom to top). Each is about 20 s of
  nobody touching. Have someone walk behind the backdrop once during the clip; these are the false-call tests.
- **Human turns:** 4 clips of 15 s each. Take a box from the supply, place it, withdraw the hand. One of them slow and
  hesitant, one with the hand resting near the stack for a few seconds.
- **Collapses:**
  1. stack of 3, knocked over;
  2. stack of 5, knocked over;
  3. stack of 4, only the top box pushed off;
  4. stack of 4, the top box nudged about 2 cm without falling (a "moved" case).
- **Write-up:** a short note per clip in `inputs/videos/phase7/clips.yaml`: what happened, the stack composition, and
  roughly at which second.

Commands (front camera = index 0, as in Phase 3; check it if the camera order changed):
```
python scripts/so101_capture_frames.py --cameras front=0 --out inputs/photos/phase7 --suffix ref_supply_full
python scripts/so101_capture_frames.py --cameras front=0 --out inputs/photos/phase7 --suffix ref_empty
python scripts/so101_record_video.py --camera 0 --seconds 20 --out inputs/videos/phase7/static_h2.mp4
python scripts/so101_record_video.py --camera 0 --seconds 20 --out inputs/videos/phase7/static_h4.mp4
python scripts/so101_record_video.py --camera 0 --seconds 20 --out inputs/videos/phase7/static_h6.mp4
python scripts/so101_record_video.py --camera 0 --seconds 15 --out inputs/videos/phase7/human_turn_1.mp4   # ..._4
python scripts/so101_record_video.py --camera 0 --seconds 10 --out inputs/videos/phase7/collapse_h3.mp4
python scripts/so101_record_video.py --camera 0 --seconds 10 --out inputs/videos/phase7/collapse_h5.mp4
python scripts/so101_record_video.py --camera 0 --seconds 10 --out inputs/videos/phase7/top_off_h4.mp4
python scripts/so101_record_video.py --camera 0 --seconds 10 --out inputs/videos/phase7/top_nudged_h4.mp4
```

## 5. Decisions for the user
- **D1 (approved 2026-10-23):** Box types per level come from **game history plus supply-slot occupancy**, not from the stack's appearance.
  - This requires the supply to start in the template layout, with boxes left in their slots until taken (a bumped
    box is re-seated by the human).
  - Fallback if a slot check is ambiguous: ask on screen (keypress) which type was taken.
- **D2 (approved 2026-10-23):** Start with method a (difference + backdrop). Train a sim segmenter (b) only if (a) fails its tests. Markers
  (c) stay last.
- **D3 (approved 2026-10-23):** Mask the arm with the rendered sim twin for hand detection during the robot's turn.

## 6. Sim test set (`scripts/perception_testset.py`, built 2026-10-23)
- **400 samples**, seeds 7001–7004, in `data/perception/sim_v1/` (gitignored).
  - About 1.5 s per sample on the Mac; 4 workers took about 4 min.
- **One sample = one turn**, arm at home, with the full Phase 5 visual randomization. That includes front-camera
  jitter of ±1 cm / ±1.5°, which stands in for tripod drift.
- **Files per sample:**
  - `ref.png`: the game start (template layout, empty stack area);
  - `before.png`, then `after.mp4` (3 s from the drop) and `after.png`;
  - segmentation labels;
  - `truth.json`: slot types, stack levels as slot ids, the event, final poses, and labels in the `sim_eval.Judge` rule.
- **Events:**
  - place: 227, of which 10 collapse;
  - push: 71, of which 44 collapse (a finger-like push of 0.15–0.6 N that stops after the box moves 1.5–4 cm);
  - nudge: 67 (top box slid 5–25 mm; 5 collapse);
  - none: 35 (false-call tests).
- **Bugs fixed while building it:**
  - a 0.3 s push of up to 1.5 N threw boxes off the table and made the sim unstable, so the push is now gentle and
    unstable samples are flagged;
  - samples with an empty stack rendered a black `before` frame (missing `mj_forward`).

## 7. Results so far
**Camera refit** (`perception/register.py`, `perception/twin.py`):
- **Method:**
  - render the twin's class masks (dark = mat, backdrop and black arm; yellow = stack area) with the template boxes in
    place;
  - adjust the 6-DOF camera pose until those masks overlap the reference frame's dark and yellow pixels (soft Dice,
    Nelder-Mead from the nominal calibration).
- **Speed:** about 5 s, once per game, on the setup frame.
- **Accuracy:** on 20 references, the worst projected slot/stack point error drops from a median 8.3 px (max 22) with
  the nominal calibration to a **median 1.3 px (max 4.1)**.
- **Why it's needed:** a first attempt with fixed polygons failed on drift. A corner-picking refit also failed: with
  all 8 boxes on the mat, only thin gaps of mat show near the arm.

**Supply-slot check** (`perception/supply.py`, decision D1), evaluated with `scripts/perception_eval.py --check supply`
(`notes/phase7_supply_sim.txt`):
- **How each slot is tested:**
  - the test patch is the slot's footprint at mat level, minus anything a neighbouring box could hide. A patch at
    box-top height looked past the far-edge slots (2, 4, 7) onto the white table;
  - "mat-like" means darker than the mat's own level + 35. That level is measured each frame on the strip beyond the
    reach tape, where boxes never go.
- **Per slot:** 6,378 / 6,400 slot readings correct.
- **Per turn (which slot was taken):** 394 / 400 correct.
- **With an "ask the player" band** (dark share 0.3–0.7) the check never answers wrongly:
  - printed boxes: 350 decided correctly, 35 (9 %) asked;
  - plain-colour boxes (a randomization the real game doesn't have): 7 decided, 8 asked.
- **Open:** cut the 9 % asked. Candidates: a second cue (difference to the reference frame), and tuning on real frames.


## 8. Collapse check (`perception/stack.py`, decision D2), 2026-10-23
Scored with `scripts/perception_eval.py --check collapse` (`notes/phase7_collapse_sim.txt`).
- **Test-set fix first:** nudges fired only when the time matched to the microsecond, so most nudge samples never
  nudged. Fixed and the set rebuilt (400 samples again).
- **Game-rule truth:** a collapse is a stack box (the new one included) ending on the table, or the stack top moving
  > 15 mm. This is not the stricter "any box moved > 5 mm" rule used in policy evaluation. Result: 126 collapses and 274
  standing stacks.

**Cues, all frame differences within one game:**
- **Stack region:** the yellow area grown by 4 cm, up to 16 cm high, minus the supply mat with its boxes and minus the
  arm. The arm mask comes from the twin rendered at home (D3).
- **Top move:** a template match of the top box's own pixels between the start and the end of the 3 s settle window,
  converted to mm on the box's plane. Error on nudges: median 1.2 mm.
- **Outside:** changed pixels between the frames before and after the turn, outside the new box's column above the old
  top.
  - Shadow-like changes (darker or lighter with the same colour balance) are ignored.
  - The old top is located from the "before" silhouette (median 14 mm error), which is enough for a 6.5 cm column.
- **On-top fit:** IoU of the projected outline of a box of the known type, lying on the old top (best xy and yaw), with
  the turn's changed pixels.

- **Release slide:** the new box tracked from the release frame to the start of the settle window. It must have moved
  > 11 mm, and ≥ 45 % of its release-time pixels must have changed. The second condition rejects a false template match
  when a drop only rocks the stack.

**Rule** (thresholds set with a margin over the largest value seen on standing stacks): a collapse is called if any of
these holds:
- top moved > 18 mm in the settle window (the game's 15 mm plus 3 mm for estimate error), or the top box no longer found;
- more than 600 changed pixels outside the column (largest on standing stacks: 491);
- an on-top IoU below 0.3 (smallest on standing stacks: 0.38);
- the new box slid > 11 mm after release (largest on standing stacks: 8 mm; smallest on fall-offs: 13 mm).

**Results:** **0 false collapse calls out of 274** standing stacks, including all 35 "nothing happened" turns.
**112 of 126** collapses detected (89 %):
- all 16 placement fall-offs;
- 70 of 71 push collapses;
- 26 of 39 nudges over 15 mm.

The 14 misses:
- **12 tops moved 15–20 mm.** They sit inside the 3 mm safety margin, so they're missed on purpose.
- **1 nudge of 21–22 mm** read as 16–18 mm.
- **1 push.**

Nothing that moved more than 22 mm went undetected.

**Thin margin to watch:** the release-slide check's changed-pixel share. One standing stack whose drop rocked it 4° gave
0.35, against at least 0.50 for every fall-off. Re-check this on the real clips.

**Not built yet:**
- **Stack height measured from the image:** the type of each level comes from game history (D1), so the expected height
  is exact. Only the top position is estimated from the image.

**For the real game:**
- **Release frame:** the release-slide cue needs the frame at release. For a human turn that's the moment the hand
  leaves; for a robot turn it's the gripper opening. This is the start of the 3 s window, so the controller must buffer
  frames from then on.
- **Fallback:** if the real clips break method a, the next step is the sim-trained segmenter (method b).

## 9. Drop check (`perception/drop.py`), 2026-10-23
**Test set:**
- **Source:** `scripts/drop_testset.py`, 600 scripted pick-and-place turns on game-rule states, with no rendering, so it
  runs on the Mac CPU.
- **Injection:** each turn gets one injected event at a random carry tick:
  - none: 239;
  - slip (box and jaw friction cut): 125;
  - knock (a 4–14 N shove for 0.1 s): 127;
  - early open (gripper commanded open away from the stack): 109.
- **Box types:** No.3 179, No.35 321, No.10 100.
- **Recorded per 30 Hz tick:** what the real arm reports (joint readings and the gripper command, in LeRobot units), plus
  the true box–gripper contact.
- **Truth:** a drop is the box leaving the jaws (no contact for 3 ticks) after it was lifted 3 cm, anywhere but just
  above the stack. There are 233 drops; only 3 knocks dislodged the box.
- **Note:** the Phase 6 sim evaluations saved no gripper traces, so this scripted set replaces them. Policy traces would
  have needed a GPU re-run.

**Method** (joint readings only, plus the stack top from perception; jaw tips from the twin's kinematics):
- **Holding:** the jaws are stopped by a box, i.e. the reading stays ≥ 5 units above a "close" command and steady over
  5 ticks. That reading is the hold level.
- **Carrying:** starts once the tips have risen 3 cm.
- **Drop calls:**
  - **slip:** the reading falls > 3 units below hold for 3 ticks (the jaws closed on nothing);
  - **open:** the reading rises > 4 units above hold while the tips are outside the release zone (within 4 cm of the
    stack top centre, −1 to +4.5 cm above its face).
- **Release:** an opening inside the zone is the release.
- **Slip over the stack:** a slip with the tips right over the stack (< 7 cm up) is also treated as a release, and the
  collapse check judges the result.

**Results** (`scripts/perception_eval.py --check drop`, `notes/phase7_drop_sim.txt`):
- **0 false drop calls out of 367 normal turns**, including 124 knocks that didn't dislodge the box.
- **223 of 233 drops detected**, median 0.13 s (max 0.2 s) after the box left the jaws.
- **The 10 "misses":** boxes that slid out right over the stack, so the monitor hands them to the collapse check. Its
  "box on the table" cue caught 16 of 16 fall-offs in §8.
- **The trade-off:** calling these drops instead would also flag a box that slips out over the stack and lands on it
  (seen once), which would make a player lose wrongly.

**Bugs fixed while building it:**
- the first rule armed on the closing gripper at home (No.10) and on empty grasps;
- traces stopped as soon as the expert aborted, cutting off the drop.

**Open:**
- Check the real gripper's hold readings per type; the sim says No.3 ≈ 21, No.35 ≈ 18, No.10 ≈ 13.4 units. Cardboard
  flexes, so the real reading may creep while carrying. The 3-unit slip margin needs checking on the real arm in Phase 8.
- Decided with the user (2026-10-23): a box slipping out right over the stack is judged by the collapse check, not called a drop.

## 10. Hand check (`perception/hand.py`, decision D3), 2026-10-23
**Test set:**
- **Source:** `scripts/hand_testset.py`, 120 episodes of `front` video at 30 Hz with the Phase 5 randomization; 35 of
  them add a ±1.2 % per-frame brightness flicker.
- **The hand stand-in:** added at run time (MjSpec; `sim/scene.xml` is untouched). It's a forearm, palm and fingers on
  a motion-driven body with no collisions, with random skin tones (6, light to dark) and sleeve colours.
- **Episodes:**
  - robot turn: 50, the expert does a full pick-and-place; in 25 of them a hand reaches in meanwhile;
  - human turn: 52, the arm at home while a hand reaches to the stack or the supply and back;
  - near miss: 18, the hand stays 4–12 cm outside a table edge.
- **Hand motion:** 0.25–0.7 m/s, held still for 0.3–2 s.
- **Truth:** at least 150 hand pixels inside the workspace image region (from segmentation).
- **Camera:** the episode's true camera here. In the game it's the per-game refit (1.3 px median).

**Method:**
- **Workspace:** the table top (+2 cm) up to 30 cm, projected through the camera.
- **Mask:** the twin's arm at the current joint readings, dilated 8 px (D3), plus a 7 cm zone around the grip point
  (the carried box, and supply boxes the closing jaws nudge).
- **Foreground:** changed versus a background frame. Shadow-like changes are ignored (same colour balance, chroma
  ≤ 0.05).
- **A hand is called** when a foreground blob of > 400 px inside the workspace is also moving (versus 2 frames back,
  counted only where both frames were unmasked), for 2 frames in a row. The blob must also either:
  - reach out across one of the two human-side table edges (a hand comes with a forearm), or
  - be lighter than the background somewhere (the arm's shadow is darker everywhere).
- **Latch:** the call is held while foreground stays where the hand was, and released when that region is back to the
  background, or after 4 s with no motion.
- **Background upkeep:** frozen while a hand is suspected. Otherwise, changed-but-still pixels are taken in after 1 s (a
  placed box, a settled stack).

**Results** (`scripts/perception_eval.py --check hand`, `notes/phase7_hand_sim.txt`):
- **Detection:** **95 of 95 reaches** into the workspace caught, median 0.03 s after the hand entered (max 0.57 s).
- **Coverage:** 98 % of hand-in-workspace frames flagged (median per reach; lowest 77 %).
- **False triggers:** in **2 of 120 episodes** (3 after the speed-up, see below), none in human turns or near misses. Both are robot turns where a hand had
  just reached in: the call lingers about 3 s after the hand has left. That keeps the robot paused longer, which is the
  safe side.
- **How it got here:**

  | version | reaches caught | episodes with false triggers |
  |---|---|---|
  | first version | 95/95 | 62/120 (the gripper leaving a placed box looked like motion; hand "ghosts" in the background) |
  | after those fixes | 95/95 | 15/120 (the moving arm's hard shadow in sim) |
  | entry rule only | 87/95 | 3/120 |
  | entry rule or "not shadow-like" (final) | 95/95 | 2/120 |

**Open:**
- Real hands, sleeves and room light; the sim's hard shadows are harsher than the real diffuse light.
- Tune the arm-mask margin on the recorded clips (D3).
- **Speed (2026-10-23):** first version 91 ms per frame on the Mac; after the speed-up, **robot turns 18 ms median
  (p95 26, max 32.5 ms)** and **human turns 11.5 ms (max 21 ms)**, inside the 33 ms budget at 30 Hz. The changes:
  - the background blend uses `cv2.addWeighted` plus a masked copy instead of boolean indexing (27 → ~1 ms);
  - the twin's arm mask is rendered once per frame at half resolution (`Twin(scale=2)`), and reused while no joint has
    moved more than 0.1 units (the arm is frozen during the human turn);
  - the shadow test runs only on pixels that already changed. Shared with the collapse check, whose results are
    identical after the change.
- **Rescore after the speed-up:** 95 of 95 reaches still caught. One extra short false trigger (0.6 s, robot turn), from
  the blend's different rounding: 3 of 120 episodes, all robot turns.
