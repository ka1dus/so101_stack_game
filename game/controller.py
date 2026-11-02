"""The game controller: SETUP -> HUMAN_TURN / ROBOT_TURN -> SETTLE -> JUDGE -> ... -> GAME_OVER (CLAUDE.md Section 4).

SETUP       reference frame (all 8 boxes in the template, stack area empty, arm at home), per-game camera refit,
            supply-slot monitor, hand watch.
HUMAN_TURN  the robot holds still (torque on). You place one box and press SPACE; SPACE is refused while a hand is
            still in the workspace.
ROBOT_TURN  starts only once no hand has been seen for 1 s; the place skill runs (skill/runner.py) and pauses (arm
            holds) whenever a hand enters the workspace, resuming 1 s after it left; then the scripted move home,
            which pauses the same way.
SETTLE      3 s of frames with nobody in the workspace (restarted if a hand shows up).
JUDGE       which slot was emptied -> the new level's type (D1; unsure or not seen -> asked on screen), drop_check
            (robot), collapse_check (D2). A loss ends the game; so does the 8th box (full stack: nobody lost).
E-STOP      e / ESC in the window or Ctrl-C: the arm holds, torque goes off, the game ends as "stopped".
Logs: <out>/turns.jsonl, game.json, front.mp4 (the whole game), per-turn before/after frames.
"""
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from perception.drop import Kinematics
from perception.register import refine
from perception.stack import COLUMN_R, SPOT, STACK_TOP_Z, change, fit_box, landed_tick, prism_mask, stack_region, top_xy, turn_check
from perception.supply import BOX, LAYOUT, SupplyMonitor
from perception.twin import Twin, arm_mask
from safety.monitor import RESUME_S, EStop, EStopped, HandWatch
from skill.grip_floor import GripFloor
from skill.runner import FPS, go_home, open_before_home, run_place_skill, safety_check
from skill.scripted_place import ScriptedPlace
from skill.stack_look import look_at_stack
from ui.display import sound

import sys
from perception.camera import ROOT
sys.path.insert(0, str(ROOT / "scripts"))
from perception_eval import template_poses  # noqa: E402

SETTLE_S = 3.0
UNSURE = (0.3, 0.7)
N_BOXES = 8
TYPE_KEYS = {"l": "no3", "m": "no35", "s": "no10", "n": None}
NAMES = {"no3": "large", "no35": "medium", "no10": "small"}
SLOT_TYPE = {s["id"]: s["type"] for s in LAYOUT["boxes"]}
HOME_TOL = 2.0
WATCHDOG_S = 20.0         # the workspace "busy" this long (game time): a stuck hand call (sim_game 2026-10-27 waited 9 h)
QUIET_S = 1.0             # SPACE accepted with the hand check still latched if nothing has moved for this long
KEY_DEAD_S = 1.0          # SPACE ignored this long after a wait starts (real games 30 Oct: an early or doubled SPACE ended
                          # the human turn before any box was placed, and the game called a loss)
NOTHING_PX = 150          # human turn with no supply slot emptied and fewer changed table pixels than this: no move yet
                          # (real 30 Oct: 0 px with nothing placed; placed boxes 707-1775 px)
GRIP_FLOOR = 28.0          # robot recipe: min gripper opening before the first grasp (notes/phase8.md section 18)
HANDOVER_LIFT = 0.04       # ... the scripted place takes over once the box is 4 cm up
ASK_FIT, ASK_OFF_MM = 0.55, 26.0   # turn_check passed but borderline (fit 0.45-0.55, or 26-32 mm off): the player confirms


class Tail:
    """Keeps the last n skill frames (the release frame is at most 1.5 s before the end), with the running count."""

    def __init__(self, n=60):
        self.buf = deque(maxlen=n); self.count = 0

    def append(self, item):
        self.buf.append(item); self.count += 1

    def get(self, i):
        j = i - (self.count - len(self.buf))
        return self.buf[j] if 0 <= j < len(self.buf) else None


class Controller:
    def __init__(self, backend, policy, ui, out_dir, human_first=True, initial_stack=(), threaded_hand=True, no_motion=False,
                 max_turns=None, recipe=True):
        self.b, self.policy, self.ui = backend, policy, ui
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.human_first, self.stack = human_first, list(initial_stack)
        self.kin, self.twin, self.twin2 = Kinematics(), Twin(), Twin(scale=2)
        self.estop = EStop(); self.quit = False; self.threaded_hand = threaded_hand; self.no_motion = no_motion
        self.vw = None; self.turns = []; self.max_turns = max_turns
        self.last_frame = None; self.busy_since = None; self.watchdogs = []; self.top_known = None
        self.recipe = recipe; self.front_fn = None         # the Phase 8 robot recipe (see _robot_turn)

    # ------------------------------------------------------------ frames, keys, video
    def _key(self, k):
        if k in ("e", "esc"):
            self.estop.trigger("e-stop key")
        elif k == "q":
            self.quit = True
        return k

    def _tick(self, frame, q=None, show=True):
        """One frame through the hand watch, the video and the window. Returns the key pressed (or None)."""
        self.hand.feed(frame, self.b.home if q is None else q); self.last_frame = frame
        if self.vw is not None:
            self.vw.write(frame)
        k = self._key(self.ui.show(frame, self.hand.present)) if show else None
        if self.estop:
            raise EStopped(self.estop.reason)
        return k

    def _next(self, q=None):
        f = self.b.frame(); return f, self._tick(f, q)

    def _blocked(self, where, wait=RESUME_S):
        """True while the workspace is not clear (a hand in it, or seen in the last `wait` s). Watchdog: busy for
        WATCHDOG_S means a stuck call more likely than a hand (something keeps moving, e.g. a wobbling box): a debug
        image is saved, then the player confirms the workspace is clear with SPACE (scripted worlds: at once) and the
        hand check takes a new background."""
        busy = self.hand.present if wait == 0 else not self.hand.clear(wait)
        if not busy:
            self.busy_since = None; return False
        t = self.b.clock(); self.busy_since = t if self.busy_since is None else self.busy_since
        if t - self.busy_since < WATCHDOG_S:
            return True
        n = len(self.watchdogs); f = self.last_frame
        if f is not None:
            img = f.copy(); tr = getattr(self.hand.mon, "track", None) if hasattr(self.hand, "mon") else None
            if tr is not None:
                img[tr] = (0.5 * img[tr] + (0, 0, 127)).astype(np.uint8)
            cv2.imwrite(str(self.out / f"watchdog_{n:02d}_{where}.png"), img)
        self.watchdogs.append(dict(turn=len(self.turns), where=where, t=round(t, 1)))
        print(f"  watchdog: workspace busy {WATCHDOG_S:.0f} s ({where}); hand check re-baselined", flush=True)
        if not self.b.scripted_human:
            title, sub, mode = self.ui.banner
            self.ui.set("Workspace still looks busy", "if nothing is in it, press SPACE (e = E-STOP)", "alert"); sound("alert")
            while True:
                f, k = self._next()
                if k == "space":
                    break
            self.ui.set(title, sub, mode)
        self.hand.rebase(f if f is not None else self.b.frame()); self.busy_since = None
        return False

    def _settle(self):
        """SETTLE_S of frames with no hand in the workspace (restarted when one shows up)."""
        frames = []; restarts = 0
        while len(frames) < int(SETTLE_S * FPS):
            f, _ = self._next()
            if self._blocked("settle", wait=0):
                if frames:
                    restarts += 1
                frames = []; self.ui.set(self.ui.banner[0], "hand in the workspace: waiting until it is clear", self.ui.banner[2]); continue
            frames.append(f)
        return frames, restarts

    def _wait_space(self, title, sub, mode, allow_hand=False):
        """Wait for SPACE. Refused while a hand is in the workspace, unless nothing there has moved for QUIET_S (the
        hand check can stay latched on the placed box or an exposure change for up to 4 s): then the player's word
        counts and the hand check takes the current frame as its new background."""
        self.ui.set(title, sub, mode); t0 = self.b.clock()
        while True:
            f, k = self._next()
            if k == "space" and self.b.clock() - t0 < KEY_DEAD_S:
                continue                                     # pressed before (or just as) the wait began
            if k == "space":
                if self.hand.present and not allow_hand:
                    if self.hand.quiet_s() < QUIET_S:
                        self.ui.set(title, "take your hand out of the workspace first, then press SPACE", mode); continue
                self.hand.rebase(f)
                return f

    def _ask_type(self, player):
        if hasattr(self.b, "answer"):                       # scripted worlds answer for the player
            return self.b.answer(), True
        self.ui.set("Which box was placed?", "l = large   m = medium   s = small   n = none", "alert"); sound("alert")
        while True:
            f, k = self._next()
            if k in TYPE_KEYS:
                return TYPE_KEYS[k], True

    def _ask_on_stack(self):
        """The front view cannot tell a box on the top from one on the table just in front of the stack (same image
        place); asked when the stack top did not rise as a box landing on it would (sim_game 2026-10-27)."""
        if hasattr(self.b, "on_stack"):
            return self.b.on_stack()
        self.ui.set("Is the new box on the stack?", "y = yes   n = no (it is on the table)", "alert"); sound("alert")
        while True:
            f, k = self._next()
            if k in ("y", "n"):
                return k == "y"

    # ------------------------------------------------------------ setup
    def setup(self):
        if hasattr(self.b, "prepare_game"):
            self.b.prepare_game()
        if not self.b.scripted_human and self.b.arm is not None:
            q = self.b.joints()
            if np.abs(q - self.b.home).max() > HOME_TOL:
                raise RuntimeError(f"the arm is not at home ({np.round(q, 1).tolist()}); run scripts/so101_go_home.py first")
        self.hand = None
        f0 = self.b.frame()
        if not self.b.scripted_human and self.b.arm is not None:
            self.ui.set("Setup", "all 8 boxes in the template, stack area empty, hands out: press SPACE", "judge")
            while self._key(self.ui.show(f0)) != "space":
                if self.estop:
                    raise EStopped(self.estop.reason)
                f0 = self.b.frame()
        self.ref = f0
        self.cam, self.refit = refine(self.ref, self.twin, template_poses(self.twin), self.b.home.tolist())
        self.supply = SupplyMonitor(self.ref, self.cam); self.home_mask = arm_mask(self.twin, self.cam, self.b.home.tolist())
        if self.recipe and self.b.arm is not None and not getattr(self.b.arm, "virtual", False):
            from skill.plain_wall import PlainWall           # the room behind the arm painted plain for the policy
            self.front_fn = PlainWall(self.cam)
        self.hand = HandWatch(self.cam, self.twin2, self.kin, clock=self.b.clock, threaded=self.threaded_hand)
        col = prism_mask(self.cam, SPOT, COLUMN_R, 0.0, STACK_TOP_Z).astype(np.uint8)
        self.ui.overlay["column"] = max(cv2.findContours(col, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
        self.slot_uv = {k: p.mean(0) for k, p in self.supply.polys.items()}
        self._overlay_slots(self.ref)
        h, w = self.ref.shape[:2]
        self.vw = cv2.VideoWriter(str(self.out / "front.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        cv2.imwrite(str(self.out / "ref.png"), self.ref)

    def _overlay_slots(self, frame):
        occ = self.supply.occupied(frame); self.ui.overlay["slots"] = [(self.slot_uv[k], v) for k, v in occ.items()]
        self.ui.overlay["stack"] = [NAMES[t] for t in self.stack]

    def _stack_top(self, before):
        z_top = 0.0002 + sum(BOX[t]["height"] for t in self.stack)
        xy = None
        if self.stack and self.top_known is not None:      # from the previous turn's judge
            xy = self.top_known
        elif self.stack:
            fg = change(before, self.ref) & stack_region(self.cam, arm_mask=self.home_mask); xy = top_xy(self.cam, fg, z_top)
        xy = SPOT if xy is None else xy
        self.ui.overlay["top_uv"] = self.cam.project(np.array([[*xy, z_top]]))[0]
        return xy, z_top

    # ------------------------------------------------------------ turns
    def _human_turn(self, again=False):
        sound("alert" if again else "turn")
        if self.b.arm is not None and not self.b.scripted_human:
            self.b.arm.hold()                                # frozen, torque on
        if self.b.scripted_human:                           # the scripted human "presses SPACE" once its box is down
            self.ui.set("Your turn", "(scripted human)", "human"); self.b.human_turn(); self.hand.rebase(self.b.frame())
        else:
            self._wait_space("No new box seen" if again else "Your turn",
                             "place one box on the stack, take your hand away, press SPACE", "human")
        self.ui.set("Settling", "3 s: hands off", "judge")
        frames, restarts = self._settle()
        return dict(settle_restarts=restarts), frames, None

    def _robot_turn(self, xy, z_top, before=None):
        sound("robot"); self.ui.set("Robot's turn", "keep your hands out of the workspace", "robot")
        while self._blocked("robot_start"):                  # start only with the workspace clear
            self._next()
        if self.no_motion:                                   # real arm, no motion: policy runs, you stand in
            run_place_skill(self.b.arm, self.policy, self.kin, xy, z_top, move=False, t_max=5.0,
                            on_tick=lambda f, q: self._tick(f, q))
            self._wait_space("Robot's turn (no motion)", "stand-in: place the robot's box by hand, hand away, SPACE", "robot")
            res = dict(event="released (stand-in)", duration_s=None, pauses=0, paused_s=0, release_tick=None, trace=np.zeros((0, 13)))
            arm_frames = None
        else:
            tail = Tail(); n = [0]

            def on_tick(f, q):
                n[0] += 1; self._tick(f, q, show=n[0] % 2 == 0)

            def paused():
                if self.estop:
                    raise EStopped(self.estop.reason)
                clear = not self._blocked("skill")
                if not clear:
                    self.ui.set("Robot paused", "hand in the workspace: the arm holds until it is clear", "alert")
                else:
                    self.ui.set("Robot's turn", "keep your hands out of the workspace", "robot")
                return not clear

            def paused_moving():                             # scripted moves: no per-tick hook, so look here
                self._tick(self.b.frame()); return paused()
            placed = looked = None
            if not self.recipe:
                res = run_place_skill(self.b.arm, self.policy, self.kin, xy, z_top, record=tail, on_tick=on_tick, paused=paused)
            else:
                res, placed, looked = self._recipe_skill(xy, z_top, before, tail, on_tick, paused, paused_moving)
            if self.recipe and (o := open_before_home(self.b.arm, self.kin, res["trace"], res["event"] == "released")) is not None:
                print(f"  opened the gripper before going home (it read {o:.1f})", flush=True)
            go_home(self.b.arm, self.kin, self.b.home, lift=res["event"] != "released", paused=paused_moving)
            arm_frames = None
            if res.get("release_tick") is not None:
                tr = res["trace"]; k0 = landed_tick(self.kin, tr, res["release_tick"]); first, last = tail.get(k0), tail.get(tail.count - 1)
                if first is not None and last is not None:
                    arm_frames = [(first[0], arm_mask(self.twin, self.cam, tr[k0, :6].tolist())),
                                  (last[0], arm_mask(self.twin, self.cam, tr[-1, :6].tolist()))]
        if hasattr(self.b, "robot_done"):
            self.b.robot_done()
        self.ui.set("Settling", "3 s: hands off", "judge")
        frames, restarts = self._settle()
        info = dict(skill_event=res["event"], skill_s=res["duration_s"], pauses=res["pauses"], paused_s=res["paused_s"],
                    loop_hz=res.get("loop_hz"), settle_restarts=restarts)
        if self.recipe and not self.no_motion:
            info.update(scripted_place=placed, look=looked)
        return info, frames, arm_frames

    def _recipe_skill(self, xy, z_top, before, tail, on_tick, paused, paused_moving):
        """The robot turn as in the Phase 8 trials (notes/phase8.md sections 18-22): wrist look at the stack, the policy
        picks with the gripper-opening floor (the plain-wall front on the real arm), the scripted place puts the box on
        what the look saw. Returns (skill result, scripted place info, look info)."""
        arm = self.b.arm; real = not getattr(arm, "virtual", False)
        top_t = self.stack[-1] if self.stack else None; yaw = 0.0; target = np.asarray(xy, float); looked = None
        if top_t is not None and before is not None:         # front estimate: yaw (and xy unless known)
            fg = change(before, self.ref, shadows=True) & stack_region(self.cam, arm_mask=self.home_mask)
            if fg.sum() > 50:
                sc, fxy, yaw = fit_box(self.cam, fg, z_top - BOX[top_t]["height"], BOX[top_t], target, top_only=len(self.stack) > 1)
                if self.top_known is None and sc >= 0.75:
                    target = fxy
            self.ui.set("Robot's turn", "looking at the stack", "robot")
            looked = look_at_stack(arm, self.kin, self.b.home, target, yaw, z_top, top_t, real=real, paused=paused_moving,
                                   log=lambda m: print(m, flush=True))
            if looked["ok"]:
                target = np.array(looked["xy"]); yaw = np.radians(looked["yaw_deg"])
        self.ui.set("Robot's turn", "keep your hands out of the workspace", "robot")
        res = run_place_skill(arm, self.policy, self.kin, xy, z_top, record=tail, on_tick=on_tick, paused=paused,
                              front_fn=self.front_fn, act_fn=GripFloor(GRIP_FLOOR), handover_lift=HANDOVER_LIFT)
        placed = None
        if res["event"] == "handover":
            extra = []

            def place_tick(c):
                # the sim backend's frame() also advances time; the sim arm's camera() only renders
                f = self.b.frame() if real else cv2.cvtColor(arm.camera("front"), cv2.COLOR_RGB2BGR)
                q = arm.joints(); self._tick(f, q); tail.append((f,)); extra.append(np.r_[q, c, 0.0])
            k_hand = len(res["trace"])
            full = set(self.supply.features(self.ref)) - {k for t in self.turns for k in t.get("emptied", [])}
            placed = ScriptedPlace(self.kin).run(arm, res["trace"], target, z_top, yaw, safety=lambda c: safety_check(self.kin, c),
                                                 on_tick=place_tick, paused=paused, log=lambda m: print(m, flush=True), full=full)
            if extra:
                res["trace"] = np.vstack([res["trace"], np.array(extra, np.float32)])
            res["event"] = "released" if placed["event"] == "released" else "scripted " + placed["event"]
            res["release_tick"] = k_hand + placed["release_index"] if placed.get("release_index") is not None else None
        return res, placed, looked

    def _nothing_placed(self, before, after, xy, z_top):
        """SPACE without a move: no supply slot emptied (and the slot check is sure) and almost nothing changed on the
        table or the stack. Not a loss; the turn goes on. A box knocked off or moved changes far more than NOTHING_PX."""
        fb, fa = self.supply.features(before), self.supply.features(after)
        if any(fb[k]["dark"] < 0.5 <= fa[k]["dark"] for k in fb):
            return False
        if any(UNSURE[0] < v["dark"] < UNSURE[1] for v in list(fb.values()) + list(fa.values())):
            return False
        _, _, f = turn_check(self.cam, self.ref, before, after, z_top, self.stack[-1] if self.stack else None, None,
                             arm_mask=self.home_mask, top_xy0=xy, top_known=self.top_known is not None)
        return f["changed"] < NOTHING_PX and f["blob"] < NOTHING_PX

    def _judge(self, player, before, frames, arm_frames, info, xy, z_top):
        after = frames[-1]
        fb, fa = self.supply.features(before), self.supply.features(after)
        emptied = [k for k in fb if fb[k]["dark"] < 0.5 <= fa[k]["dark"]]
        unsure = any(UNSURE[0] < v["dark"] < UNSURE[1] for v in list(fb.values()) + list(fa.values()))
        if unsure or len(emptied) != 1:
            ptype, asked = self._ask_type(player)
        else:
            ptype, asked = SLOT_TYPE[emptied[0]], False
        # turn check v2 (perception/stack.py turn_check, notes/phase8.md section 23): photometric match, the new box as a
        # flat box on the old top (known from the previous turn), old-top coverage, edges elsewhere, old stack change
        ok2, why, f = turn_check(self.cam, self.ref, before, after, z_top, self.stack[-1] if self.stack else None, ptype,
                                 arm_mask=self.home_mask, top_xy0=xy, top_known=self.top_known is not None)
        col = not ok2
        asked_on_stack = None
        if ptype and z_top > 0.005 and (f.get("unsure") or (not col and (f["fit"] < ASK_FIT or f["off_mm"] > ASK_OFF_MM))):
            asked_on_stack = self._ask_on_stack()          # borderline either way: the player says where the box is
            col = not asked_on_stack
            why = ["new box not on the stack (player's answer)"] if col else []
        if not col and ptype:
            self.top_known = np.array(f["xy_new"])          # the next turn's old top
        drop = info.get("skill_event", "").startswith("drop")
        loss = bool(col or drop or ptype is None)
        call = ("loss: " + (info["skill_event"] if drop else "no box placed" if ptype is None else ", ".join(why))) if loss else "ok"
        wd = [w for w in self.watchdogs if w["turn"] == len(self.turns)]
        info.update(watchdogs=len(wd), emptied=emptied, asked=asked, asked_on_stack=asked_on_stack, placed_type=ptype, call=call, top_xy=np.round(xy, 4).tolist(), z_top=round(z_top, 4),
                    features={k: v for k, v in f.items() if k != "top_xy"})
        return loss, call, ptype

    # ------------------------------------------------------------ game
    def play(self):
        t_start = time.time(); result = None
        try:
            self.setup()
            player = "human" if self.human_first else "robot"
            while len(self.stack) < N_BOXES and not self.quit and (self.max_turns is None or len(self.turns) < self.max_turns):
                if hasattr(self.b, "begin_turn"):
                    self.b.begin_turn()
                before, _ = self._next(); self._overlay_slots(before)
                xy, z_top = self._stack_top(before)
                if player == "human":
                    info, frames, arm_frames = self._human_turn(); early = 0
                    while not self.b.scripted_human and self._nothing_placed(before, frames[-1], xy, z_top):
                        early += 1; print("  no new box seen and no supply slot emptied: the human turn goes on", flush=True)
                        info, frames, arm_frames = self._human_turn(again=True)
                    info.update(early_space=early)
                else:
                    info, frames, arm_frames = self._robot_turn(xy, z_top, before)
                self.ui.set("Judging", "", "judge"); self._tick(frames[-1])
                loss, call, ptype = self._judge(player, before, frames, arm_frames, info, xy, z_top)
                info.update(turn=len(self.turns), player=player, height=len(self.stack), t=round(time.time() - t_start, 1))
                if hasattr(self.b, "truth"):
                    t_loss, t_why = self.b.truth(); info.update(truth="loss: " + t_why if t_loss else "ok", call_ok=t_loss == loss)
                    loss_end = loss or t_loss
                else:
                    loss_end = loss
                i = len(self.turns)
                cv2.imwrite(str(self.out / f"t{i:02d}_before.png"), before); cv2.imwrite(str(self.out / f"t{i:02d}_after.png"), frames[-1])
                self.turns.append(info)
                with open(self.out / "turns.jsonl", "a") as fh:
                    fh.write(json.dumps(info, default=str) + "\n")
                print(f"  turn {i} {player:5s} h{info['height']}: {call}" + (f"   truth {info['truth']}" if "truth" in info else ""), flush=True)
                if loss_end:
                    result = dict(loser=player, why=call if loss else info.get("truth")); break
                self.stack.append(ptype); self._overlay_slots(frames[-1])
                player = "robot" if player == "human" else "human"
            if result is None:
                result = dict(loser=None, why="full stack" if len(self.stack) >= N_BOXES else "quit")
        except (EStopped, KeyboardInterrupt) as e:
            self.b.estop(); result = dict(loser=None, why=f"stopped: {e or 'Ctrl-C'}"); print("  E-STOP: arm held, torque off", flush=True)
        finally:
            if self.vw is not None:
                self.vw.release()
            if getattr(self, "hand", None) is not None:
                self.hand.close()
        result.update(turns=len(self.turns), stack=self.stack, refit=getattr(self, "refit", None), minutes=round((time.time() - t_start) / 60, 1))
        (self.out / "game.json").write_text(json.dumps(dict(result=result, turns=self.turns), indent=1, default=str))
        self._game_over(result)
        return result

    def _game_over(self, r):
        if r["loser"] == "robot":
            title, s = "You win!", "win"
        elif r["loser"] == "human":
            title, s = "Robot wins", "loss"
        else:
            title, s = "Game over", "turn"
        sound(s); self.ui.set(title, f"{r['why']}  -  press SPACE to close", "over")
        if not self.b.scripted_human and self.ui.last is not None and self.b.arm is not None:
            f = self.b.frame()
            while self._key(self.ui.show(f, False, 30)) != "space":
                pass
