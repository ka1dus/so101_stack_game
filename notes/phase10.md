# Phase 10: real games (log and analysis)

## 1. First real games, 30 Oct 2026 (wcal60 + full recipe, `play_game.py real`)

Folders: `data/games/real_20261030_104017` (1 turn), `real_20261030_104456` (7 turns),
`real_20261030_111530` (6 turns).

| Game | Turns | Result called | Right? |
|---|---|---|---|
| 104017 | 1 | human loses at turn 0 ("no box lying flat on the top") | **no**: judged before any box was placed |
| 104456 | 7 | human loses at turn 6, stack 6 high | **no**: judged before the box was placed (hand comes in 13 s later, box then placed fine) |
| 111530 | 6 | robot loses at turn 5 (its no3 ended on the table) | yes |

### Robot turns (6)
| Game / turn | Box onto top (height) | Look | Result |
|---|---|---|---|
| 104456 t1 | no3 onto no35 (1) | trusted | on the stack |
| 104456 t3 | no3 onto no10 (3) | trusted | on the stack (game asked y/n) |
| 104456 t5 | no35 onto no35 (5) | trusted | on the stack (game asked y/n) |
| 111530 t1 | no35 onto no10 (1) | trusted | on the stack |
| 111530 t3 | no35 onto no35 (3) | trusted | on the stack |
| 111530 t5 | no3 onto no3 (5) | **not trusted** (6 looks) | slid off the top as the gripper opened |

Robot 5/6 placed; every pick was the right box; no pauses, e-stops or watchdogs; loop 23–25 Hz.

### Findings
1. **Turn ended before the box was placed (both wrong results).** Game 104017: the human turn's before-frame is
   video frame 44 and the after-frame 152 (3.6 s), with no hand ever in view. So SPACE came within ~0.6 s of the turn
   starting: most likely the setup SPACE counted twice (key repeat or a double press). Game 104456 t6: SPACE ~3 s into
   the turn, hand in view only 13 s later (`notes/img/phase10_g1_late_place.png`). In both, nothing changed and no
   supply slot emptied, yet the game called a loss. Fix proposed: (a) ignore keys pressed before a wait starts and
   in its first second; (b) "nothing changed and no slot emptied" = no move yet: say so and keep waiting, not a loss.
2. **Robot's failed place (111530 t5).** The box was on the stack and held (frames 4838–4846), then slid toward the
   camera as the jaws opened (4848–4852, `notes/img/phase10_g2_release_slide.png`). The top it sat on was the human's
   no3 lying crosswise (139°) and 9 mm off the no35 below. Release clearance was 0.1 mm (pressed onto the top). Likely
   cause: a tilted top surface; a jaw dragging the box can't be ruled out from the front video alone.
3. **The look's safety rule worked.** The 6 looks agreed with each other (±0.5 mm) but each was only 73–74 % clear of
   the jaws (limit 75 %), so none was trusted. They were in fact 13 mm off; the front estimate the game fell back to
   matches the image (`notes/img/phase10_g2_look_vs_front.png`, red = front estimate used, green = look).
4. **Extra y/n questions.** Two good robot placements at heights 3 and 5 were asked about because the "sticks up"
   cue read 167/176 px (limit 60; the real trials never went above 0). Correct, but it nags the player at heights ≥ 3.
5. No false collapse calls on boxes actually placed (9 placed turns judged right).

### Fix for finding 1 (`game/controller.py`)
- `KEY_DEAD_S` = 1 s: SPACE is ignored for the first second of every SPACE wait (catches a doubled or early press).
- `_nothing_placed()`: after a human turn, if no supply slot emptied (and the slot check is sure) and fewer than
  `NOTHING_PX` = 150 table/stack pixels changed, the game shows "No new box seen" and the human turn goes on (no loss).
  A knocked or moved box changes far more pixels, so it is still judged. Logged as `early_space` per turn.
- Checks: replayed on all 14 turns of the 30 Oct games: both early-SPACE turns caught, the 12 real placements
  untouched. Recorded clip `human_turn_1` with SPACE held from 0 s: 1 "no new box seen", then the real placement
  judged ok; SPACE at 2 s and 9 s: ok, no change in behaviour.

## 2. Game 4 with the early-SPACE fix, 30 Oct (`data/games/real_20261030_115623`, 8 turns, 16 min)

Full game to a 7-box stack; the robot lost on turn 7 (it knocked the 7-box stack over). Call correct.

- Human turns 4/4 judged right, no early SPACE, no extra question except one y/n on a no10 at height 2 (yes).
  Turn 4 had a big exposure change and a person at the image edge (21,655 changed px), yet the edge cues gave 25 px:
  correct "ok".
- Robot turns 3/4 placed (no3 onto no35 at h1, no3 onto no10 at h3, no35 onto no35 at h5); every pick the right box.
- **Turn 7 failure: wrong held-box model.** The robot picked the no3 from slot 3, but the scripted place assumed the
  no35 from slot 4, the template slot nearest the grasp, which the human had already taken on turn 6. A no3 held near one end was
  placed as if centred, and it pressed one side of the 7-box stack, which tipped (`notes/img/phase10_g3_knock.png`).
  The wrist look again wasn't trusted (6 looks agreed to 1 mm but 67–69 % jaw-clear), so the front estimate was used.
- **Fix** (`skill/scripted_place.py`, `game/controller.py`): the held-box slot is chosen only among slots the game's
  slot check still sees full; the distance from the gripper estimate to the chosen slot is logged (`guess_mm`). On the 47
  real trials the choice is unchanged (median 7 mm, max 12 mm from the slot centre).

- Sim check (1 game, seed 14003, `data/games/sim_20261030_121800`): the game code runs and the slot choice is right
  (slot 5, 5 mm). That game had a **false loss call** at h5 ("old stack moved", 40 % changed; sim truth: ok). It isn't
  from this fix: it's the same cue as the 30 Oct sim false call at h5 (18 %). Real good turns so far peak at 12 %.

### Next
- Re-run real games with both fixes.

## 3. The "old stack moved" false call (sim game 14003, turn 5)

- Cause: the new no35 landed on a no3 and nudged the top two old boxes ~4 px (~3–4 mm) sideways; the lower boxes
  didn't move (image match per band). The sim rule allows the top 15 mm. But the cue counts the share of the old
  stack's pixels that changed, and on printed boxes a few-mm shift changes most of them: 40 % (limit 15 %).
  `notes/img/phase10_sim_nudge_false_call.png` (red = old stack, yellow = changed on it, magenta = new box).
- Fix (`perception/stack.py nudged_change`, `NUDGE_MM` 10, `NUDGE_BAND_PX` 12): only when the plain share is over the
  limit, the old stack is cut into 12-row bands, each may shift up to the pixel size of 10 mm at the stack top, and
  the share is taken at each band's best shift. A fallen or removed box still changes its band at any shift. Raw value
  logged as `stack_change_raw`. Cost ≤ 0.4 s, only on turns that would otherwise fail this cue.
- Checks:

  | Set | Before | After |
  |---|---|---|
  | The sim false call | 40 % → loss | 0 % → ok |
  | Real trials (82) | 0/56 false, 1/26 missed | same |
  | Real clips (11) | 11/11 | 11/11 (incl. the top pushed ~2 cm: still a loss) |
  | Real games (22 turns) | — | every call unchanged |
  | Sim set, game rule (294) | 2/218 false, 8/76 missed, 15 asked | **1/218**, 8/76, 17 asked |
  | Sim set, strict truth | 0/197 false, 27/97 missed | 0/197, 28/97 |

  The strict-truth extra miss is a turn the game rule counts as ok anyway (the new box's own slide, decided 30 Oct).
  The cue is weaker on real game 4's collapse (24 % → 10 %), but other cues still call that turn
  (47 mm off, 837 edge px).
- Look into the stick cue at heights ≥ 3 (finding 4), the release onto tilted tops (finding 2), and the look's
  jaw-clear limit (both untrusted looks agreed with themselves; one was 13 mm off, so keep the limit for now).
