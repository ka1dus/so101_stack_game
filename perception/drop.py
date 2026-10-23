"""drop_check(): did the carried box leave the gripper before the stack? From the joint readings only (LeRobot
units, what the real arm reports at 30 Hz) plus the stack top from perception.

A closed gripper holding a box stops at the box's width: the reading stays well above the "close" command and stops
changing (BLOCKED_UNITS, STEADY_TICKS). That level is `hold`; carrying starts once the tips have also risen LIFT_M.
While carrying:
  slip   the reading falls more than SLIP_UNITS below hold for SLIP_TICKS ticks: the jaws closed on nothing
  open   the reading rises more than OPEN_UNITS above hold while the jaw tips are not in the release zone above the
         stack: the box was let go (or pulled out) away from the stack
An opening (or a slip) inside the release zone counts as the release and ends the watch; the collapse check judges
the placement.
Jaw-tip position from the twin's kinematics (sim/scene.xml, validated joint mapping).
"""
import sys

import mujoco
import numpy as np

from .camera import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from joint_mapping import JointMapping  # noqa: E402

SLIP_UNITS, SLIP_TICKS = 3.0, 3
OPEN_UNITS = 4.0
LIFT_M = 0.03                 # carrying starts once the tips are this far above where the jaws closed on the box
BLOCKED_UNITS = 5.0           # reading this far above a "close" command = jaws stopped by something
STEADY_TICKS, STEADY_UNITS = 5, 0.5   # ... and steady: no more change than this over these ticks
ZONE_R = 0.04                 # release zone: tips within this of the stack top centre (xy) ...
ZONE_Z = (-0.01, 0.045)       # ... and this far above/below the top face (tips sit near the box's bottom)
SLIP_ZONE_TOP = 0.07          # a box slipping out right over the stack (tips < 7 cm up) falls onto it: hand over to
                              # the collapse check instead of calling a drop
CLOSED_CMD = 5.0              # gripper command at or below this = "close"


class Kinematics:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/scene.xml")); self.d = mujoco.MjData(self.m)
        self.jm = JointMapping.from_files()
        self.adr = [self.m.jnt_qposadr[self.m.joint(j).id] for j in
                    ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]
        self.gb = self.m.body("gripper").id
        fg = self.m.geom("finger_gripper_0").id
        self.tip = np.array([self.m.geom_pos[fg][0] + self.m.geom_size[fg][0], 0.0, self.m.geom_pos[fg][2] - self.m.geom_size[fg][2]])

    def tip_pos(self, q_real):
        self.d.qpos[self.adr] = self.jm.real_to_sim(np.asarray(q_real, float))
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.xpos[self.gb] + self.d.xmat[self.gb].reshape(3, 3) @ self.tip


class DropMonitor:
    """Feed one tick at a time; .event is None, "drop: slip", "drop: open" or "released"."""

    def __init__(self, kin, top_xy, top_face):
        self.kin, self.top_xy, self.top_face = kin, np.asarray(top_xy, float), float(top_face)
        self.close_z = None; self.carrying = False; self.hist = []; self.hold = None; self.low = 0
        self.event = None; self.tick_n = 0; self.event_tick = None

    def in_zone(self, tip, top=ZONE_Z[1]):
        return np.linalg.norm(tip[:2] - self.top_xy) < ZONE_R and ZONE_Z[0] < tip[2] - self.top_face < top

    def tick(self, q_real, g_cmd):
        k = self.tick_n; self.tick_n += 1
        if self.event is not None:
            return self.event
        g = float(q_real[5]); tip = self.kin.tip_pos(q_real)
        if not self.carrying:
            self.hist.append(g)
            blocked = g_cmd <= CLOSED_CMD and g > g_cmd + BLOCKED_UNITS and len(self.hist) >= STEADY_TICKS \
                and np.ptp(self.hist[-STEADY_TICKS:]) < STEADY_UNITS
            if blocked:
                if self.close_z is None:
                    self.close_z, self.hold = tip[2], float(np.median(self.hist[-STEADY_TICKS:]))
                if tip[2] > self.close_z + LIFT_M:
                    self.carrying = True
            elif g_cmd > CLOSED_CMD or g < g_cmd + BLOCKED_UNITS:
                self.close_z = None; self.hold = None
            return None
        if g < self.hold - SLIP_UNITS:
            self.low += 1
            if self.low >= SLIP_TICKS:
                self.event, self.event_tick = ("released" if self.in_zone(tip, SLIP_ZONE_TOP) else "drop: slip"), k
        else:
            self.low = 0
        if g > self.hold + OPEN_UNITS:
            self.event, self.event_tick = ("released" if self.in_zone(tip) else "drop: open"), k
        return self.event


def drop_check(trace_q, trace_cmd, top_xy, top_face, kin=None):
    """Offline helper: run the monitor over a whole trace. Returns (event, tick)."""
    mon = DropMonitor(kin or Kinematics(), top_xy, top_face)
    for q, c in zip(trace_q, trace_cmd):
        if mon.tick(q, c):
            break
    return mon.event, mon.event_tick
