"""What the controller needs from the world, three ways.

A backend provides: `frame()` (the next undistorted BGR `front` frame; in sim this advances the physics one tick),
`joints()`, `home`, `arm` (the object run_place_skill / go_home drive), `estop()`, `clock()`, and for scripted worlds
`human_turn()` (the "human" places a box), `answer()` (replies to "which box?") and `truth()` (the game rule from
the exact state, to score the controller's calls).

RealBackend  the SO-101 and its cameras (skill/runner.RealArm). no_motion=True: the robot's turn runs the policy
             with the arm holding home, then you place the robot's box by hand as a stand-in (exercises the
             perception and the controller on real frames without motion).
SimBackend   MuJoCo: the robot's turn runs the policy on the simulated arm (same runner, a sim stand-in arm); the
             human's turn is sim_game.py's scripted placement with human-like error.
ClipBackend  one recorded clip (inputs/videos/phase7/): frames in order, the arm frozen at home; a human turn only.
"""
import json
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml

from perception.camera import ROOT
from skill.runner import FPS, HOME_SPEED, RealArm, safety_check, undistort_front

import sys
sys.path.insert(0, str(ROOT / "scripts"))
from joint_mapping import JOINT_NAMES, JointMapping  # noqa: E402

HOME_FILE = Path.home() / ".cache/so101_home/my_follower_arm.json"
MEAS = yaml.safe_load((ROOT / "inputs/measurements.yaml").read_text())


def saved_home():
    if HOME_FILE.exists():
        h = json.loads(HOME_FILE.read_text())
    else:
        h = MEAS["arm"]["home_pose"]
    return np.array([h[j] for j in JOINT_NAMES], float)


class RealBackend:
    scripted_human = False

    def __init__(self, port, robot_id="my_follower_arm", front=0, wrist=1, no_motion=False):
        self.arm = RealArm(port, robot_id, front, wrist); self.arm.connect(); self.home = saved_home()
        self.no_motion = no_motion

    def frame(self):
        return cv2.cvtColor(undistort_front(self.arm.read()[1]), cv2.COLOR_RGB2BGR)

    def joints(self):
        return self.arm.joints()

    def clock(self):
        return time.monotonic()

    def estop(self):
        try:
            self.arm.hold()
        finally:
            self.arm.torque_off()

    def close(self):
        self.arm.disconnect()


class SimArm:
    """The simulated SO-101 behind run_place_skill's arm interface (LeRobot units in and out)."""
    pinhole_front = True
    virtual = True

    def __init__(self, game):
        self.g = game; s = game.sim; self.jm = game.jm
        self.act = [s.act[j] for j in JOINT_NAMES]
        self.clo, self.chi = s.m.actuator_ctrlrange[self.act, 0], s.m.actuator_ctrlrange[self.act, 1]
        self.lo, self.hi = self.jm.lerobot_limits().T; self.ticks = 0

    def read(self):
        f = cv2.cvtColor(self.g.front(), cv2.COLOR_BGR2RGB)
        return self.g.q_real(), f, self.g.wrist()

    def joints(self):
        return self.g.q_real()

    def send(self, q):
        s = self.g.sim; s.d.ctrl[self.act] = np.clip(self.jm.real_to_sim(np.asarray(q, float)), self.clo, self.chi)
        s.sync_plan()                                       # later sim.wait() calls hold this command
        for _ in range(s.steps_per_ctrl):
            mujoco.mj_step(s.m, s.d)
        self.ticks += 1

    def hold(self):
        self.send(self.joints())


class SimBackend:
    scripted_human = True

    def __init__(self, policy_path, device="mps", seed=12001, video=None):
        import game_states as G
        import grasp_test as GT
        from perception.drop import Kinematics
        from perception.twin import Twin
        from randomize import H, W, Randomizer
        from sim_game import Game
        self.G = G; sim = GT.Sim(); rand = Randomizer(sim.m); ren = mujoco.Renderer(sim.m, H, W)
        self.game = Game(sim, rand, ren, Twin(), Kinematics(), None, np.random.default_rng(seed), video=video)
        self.arm = SimArm(self.game); self.home = self.game.jm.sim_to_real(sim.home); self.t = 0.0

    # world bookkeeping (mirrors sim_game.Game.play)
    def prepare_game(self):
        g = self.game; G = self.G; sim, m, d, rng = g.sim, g.sim.m, g.sim.d, g.rng
        g.rand.apply(rng); start = G.layout_supply(rng)
        for i, b in enumerate(start):
            b.slot = i + 1
        st = G.State(supply=start); mujoco.mj_resetData(m, d); G.assign_bodies(st, m); G.apply(st, m, d)
        d.qpos[sim.qadr] = sim.home[:5]; d.qpos[sim.gadr] = sim.home[5]
        for j, v in zip(JOINT_NAMES, sim.home):
            d.ctrl[sim.act[j]] = v
        mujoco.mj_forward(m, d); sim.sync_plan(); sim.wait(0.5)
        g.type_of = {b.body: b.type for b in start}; self.slot_of = {b.body: b.slot for b in start}
        self.body_of = {b.slot: b.body for b in start}; self.stack = []; self.pose0 = {b.body: g.pose(b.body)[0] for b in start}
        self.placed = None

    def frame(self):
        g = self.game; g.sim.wait(1 / FPS); self.t += 1 / FPS; return g.front()

    def joints(self):
        return self.game.q_real()

    def clock(self):
        return self.t + self.arm.ticks / FPS

    def left(self):
        g = self.game
        return [b for b in self.body_of.values() if b not in self.stack and np.linalg.norm(g.pose(b)[0] - self.pose0[b]) < 0.01]

    def begin_turn(self):
        self.old = {b: self.game.pose(b) for b in self.stack}; self.left0 = self.left(); self.placed = None

    def human_turn(self):
        g = self.game; body = self.left0[int(g.rng.integers(len(self.left0)))]
        g.human_place(body, self.stack); self.placed = body

    def robot_done(self):
        g = self.game; moved = [b for b in self.left0 if np.linalg.norm(g.pose(b)[0] - self.pose0[b]) > 0.02]
        self.placed = max(moved, key=lambda b: g.pose(b)[0][2]) if moved else None

    def answer(self):
        return self.game.type_of[self.placed] if self.placed else None

    def on_stack(self):
        """Replies to "is the new box on the stack?" from the exact state."""
        loss, why = self.game.truth_turn(self.stack, self.placed, self.old)
        return self.placed is not None and why != "new box not on the stack"

    def truth(self):
        loss, why = self.game.truth_turn(self.stack, self.placed, self.old)
        if self.placed is None:
            loss, why = True, "no placement"
        if not loss:
            self.stack.append(self.placed)
        return loss, why

    def estop(self):
        self.arm.hold()

    def close(self):
        pass


class ClipBackend:
    """A recorded clip as the world: frames in order (already undistorted here), arm frozen at home."""
    scripted_human = False

    def __init__(self, clip_path, ref_path, placed_type=None):
        sys.path.insert(0, str(ROOT / "scripts"))
        from phase7_real_refit import undistort_img
        c = cv2.VideoCapture(str(clip_path)); self.frames = []
        while True:
            ok, im = c.read()
            if not ok:
                break
            self.frames.append(undistort_img(im))
        self.ref = undistort_img(cv2.imread(str(ref_path))); self.k = 0; self.home = saved_home(); self.arm = None
        self.first = True; self.placed_type = placed_type

    def answer(self):
        return self.placed_type

    def on_stack(self):
        return self.placed_type is not None              # the recorded placements all stayed on the stack

    def frame(self):
        if self.first:                                      # the game-start reference photo
            self.first = False; return self.ref
        f = self.frames[min(self.k, len(self.frames) - 1)]; self.k += 1; return f

    def joints(self):
        return self.home

    def clock(self):
        return self.k / FPS

    def estop(self):
        pass

    def close(self):
        pass
