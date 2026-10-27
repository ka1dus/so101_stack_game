# Phase 9: game controller, safety and UI

Status 2026-10-27: built and dry-run on the recorded clips and in sim. Not yet run with the real arm (that waits for
the Phase 8 trials and the real-motion checkpoint).

## 1. What was built

| file | role |
|---|---|
| `game/controller.py` | the state machine of CLAUDE.md Section 4; logs per game |
| `game/backends.py` | the worlds: `RealBackend` (SO-101 + cameras; `--no-motion` stand-in mode), `SimBackend` (MuJoCo: the policy on the sim arm, sim_game.py's scripted human), `ClipBackend` (a recorded clip) |
| `safety/monitor.py` | `EStop`; `HandWatch`: hand_in_workspace() (D3) in its own thread, `clear(1 s)`, `rebase()` |
| `ui/display.py` | OpenCV game window (banner, live `front` view with the stack column, stack top and slot states, red frame on a hand), keys, macOS sounds; `NullUI` for scripted runs |
| `skill/runner.py` | now also takes `on_tick` (frames to the hand watch and the window) and `paused` (hold while a hand is in), drives a sim stand-in arm, and pauses scripted moves too |
| `scripts/play_game.py` | entry point: `real` / `sim` / `clip` |
| `scripts/test_safety.py` | pause and e-stop tests in sim |

**Flow:**
1. **SETUP:**
   - press SPACE with the template full, the stack area empty and the arm at home;
   - the game refuses to start if the arm is more than 2 units from home;
   - then camera refit (with the prior), supply monitor and hand watch.
2. **Human's turn:**
   - the arm holds with torque on;
   - you place a box and press SPACE;
   - SPACE is refused while a hand is in the workspace, unless nothing there has moved for 1 s (the hand check can stay latched up to 4 s after the hand left, from auto exposure); then your SPACE counts and the hand check re-baselines.
3. **Robot's turn:**
   - starts only after 1 s with no hand;
   - the skill pauses (holds where it is) whenever a hand enters, and resumes 1 s after it left, with a fresh policy plan;
   - the scripted move home pauses the same way.
4. **SETTLE:** 3 s with nobody in the workspace (restarted otherwise).
5. **JUDGE:**
   - **slot check (D1):** the emptied slot gives the new level's type;
   - **asking:** when the slot check is unsure or doesn't see exactly one slot emptied, the game asks on screen, `l`/`m`/`s`/`n`. "Not seen" is never called a loss by itself; that avoids false calls.
   - **robot drop:** `drop_check`;
   - **collapse (D2):** `collapse_check`, with the robot's arm frames at release and at the end.
6. A loss ends the game; so does the 8th box (full stack, nobody lost).
7. **E-stop:** `e` or ESC in the window, or Ctrl-C. The arm holds and torque goes off.

## 2. Dry runs

**Recorded clips** (`play_game.py clip`, the 4 human turns of 2026-10-26, SPACE pressed shortly after the hand left):
- 4/4 called ok, with the right slot and type each time and no question asked.
- human_turn_1: SPACE at 9.0 s was refused while the hand was still in view, then accepted.

**Sim games:** see §3.

**Safety, in sim** (`test_safety.py`):
- **Pause:** a hand from 3 to 5 s into the robot's turn gave 1 pause of 3.0 s (2 s hand plus the 1 s resume delay).
  The joints changed by 0.00 units per frame while held, against 0.62 while moving just before. The turn then finished (released, call ok, truth ok).
- **E-stop:** `e` about 1 s into the robot's turn ended the game as "stopped: e-stop key". The arm moved 0.01 units in the next second.
- **Bug found and fixed:** the sim stand-in arm only advances time when it is sent a command, so a pause never ended.
  Paused ticks now re-send the held position, and scripted moves hold at their current target rather than the reading (no sag on the real arm).
- **Sim artifact fixed:** the scripted human drops its box into place in one frame, which the hand check read as a hand; that restarted the settle window 3 times in one game.
  The scripted human now "presses SPACE" (hand check re-baselined) once its box is down.

## 3. Results

**First batch** (10 sim games, seed 14001, before the fixes below), 9 games finished:
- 2 wrong calls. In 2 games the scripted human's box landed on the table just in front of the stack, and the game called it "ok".
  From the `front` camera, a box there sits almost exactly where a box on the top would appear.
  The game still ended at the next turn's truth check, but the call itself was wrong.
- Game 10 hung for 9 hours: the wait for a clear workspace before the robot's turn had no time limit, and the hand check never cleared.

**Fixes:**
1. **"Is the new box on the stack?" (y/n):** asked when the stack top in the image rises by less than half a box height. Answering n ends the game as a loss for the player whose turn it was.
   - A camera-only rule was tried first: fit the new box on the table versus on the top. It was dropped because it gave 2 false loss calls and caught only 1 of the 2 bad turns on the saved frames.
   - On the saved frames, the question would have caught both bad turns and asked on 5 of about 45 good ones. On the 4 real recorded human turns it would not have asked.
   - In sim, the scripted player answers from the exact state.
2. **Watchdog** (`WATCHDOG_S` 20 s): if the workspace stays "busy" for that long in any wait (robot start, skill pause, settle), the game saves `watchdog_NN_<where>.png`.
   In a real game it then asks for SPACE once the workspace is clear; in sim the scripted player presses it at once. Either way the hand check takes a new background.
   `test_safety.py` still passes (pause 3.0 s with 0.00 units of motion; e-stop ok).

**Second batch** (10 sim games, seed 14001, with both fixes; `data/games/sim_20261027_104658`):

| | |
|---|---|
| turn calls correct | **75/75** |
| losses caught | 9/9 |
| false loss calls | **0/66** |
| games called right | **10/10** |
| asked box type (slot check unsure) | 6 |
| asked "on the stack?" | 1 |
| watchdogs | 0 |
| robot pauses / settle restarts | 0 / 0 |
| game time | 2.6–6.9 min |

Outcomes:
- 1 full stack;
- 8 robot losses: 6 at the 8th box (height 7), 2 at height 5;
- 1 human loss.

The robot placed onto a 7-box stack successfully in only 1 of 7 tries. That is the known hard case (about 10–11° release tilt at full height, Phase 3 §11) and the main weak spot for Phase 10.

## 4. Open issues and next step
- Asking the box type 6 times in 75 turns means the slot check is often unsure in sim. Look at it before real games.
- Robot placements at height 7 are weak (1/7 here). Either target them in the next data generation, or cap the robot's turn at height 6.
- Not yet run: the real `--no-motion` controller mode (needs the setup), then real games after the Phase 8 trials and the real-motion checkpoint.
