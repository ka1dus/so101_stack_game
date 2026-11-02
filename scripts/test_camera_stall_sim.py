"""Proof in sim that a camera stall during the policy skill holds the arm still and the skill then completes.
The sim arm raises CameraStale from read() for STALL ticks starting at tick START (as RealArm does when a camera
resets). Checks: during the stall every command is the same hold pose and the arm is at rest after 0.5 s (< 0.5 units; it
was moving when the camera stopped), the skill resumes and the
scripted place puts the box on the stack.

    MUJOCO_GL=cgl ~/.venvs/so101_eval/bin/python scripts/test_camera_stall_sim.py
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import servo_model  # noqa: E402
from game.backends import SimBackend  # noqa: E402
from joint_mapping import JOINT_NAMES  # noqa: E402
from perception.drop import Kinematics  # noqa: E402
from perception.stack import SPOT  # noqa: E402
from perception.supply import BOX  # noqa: E402
from phase8_replay import set_box  # noqa: E402
from skill.cameras import CameraStale  # noqa: E402
from skill.grip_floor import GripFloor  # noqa: E402
from skill.runner import load_policy, run_place_skill, safety_check  # noqa: E402
from skill.scripted_place import ScriptedPlace  # noqa: E402


class StallingArm:
    """SimArm whose cameras 'reset' for `stall` read() calls from call `start` on."""
    def __init__(self, arm, start, stall):
        self.a, self.start, self.stall, self.calls, self.sent_during = arm, start, stall, 0, []
        self.virtual, self.pinhole_front, self.lo, self.hi = arm.virtual, arm.pinhole_front, arm.lo, arm.hi

    def read(self):
        self.calls += 1
        if self.start <= self.calls < self.start + self.stall:
            raise CameraStale("test: wrist camera resetting")
        return self.a.read()

    def stalled(self):
        return self.start <= self.calls < self.start + self.stall

    def send(self, q):
        if self.stalled():
            self.sent_during.append((np.array(q), self.a.joints()))
        self.a.send(q)

    def joints(self):
        return self.a.joints()

    def hold(self):
        self.a.hold()


def main():
    pol = str(ROOT / "data/policies/act_wcal_060000"); kin = Kinematics(); policy = load_policy(pol, "mps")
    params = servo_model.load(str(ROOT / "sim/assets/servo_fit.yaml"))
    home = np.array([json.loads((Path.home() / ".cache/so101_home/my_follower_arm.json").read_text())[j] for j in JOINT_NAMES])
    ok_all = True
    for start, label in ((60, "during the reach"), (150, "at the grasp")):
        b = SimBackend(pol, "cpu", seed=30000); b.prepare_game(); g = b.game; s = g.sim; m, d = s.m, s.d; servo_model.apply_params(m, params)
        set_box(m, d, b.body_of[6], SPOT, 0.0005 + BOX["no35"]["height"] / 2, np.radians(55)); z = 0.0005 + BOX["no35"]["height"]
        q0 = g.jm.real_to_sim(home); d.qpos[s.qadr] = q0[:5]; d.qpos[s.gadr] = q0[5]
        for j, v in zip(JOINT_NAMES, q0):
            d.ctrl[s.act[j]] = v
        mujoco.mj_forward(m, d); s.sync_plan(); s.wait(1.5)
        arm = StallingArm(b.arm, start, 90)                    # 90 ticks = 3 s without camera frames
        res = run_place_skill(arm, policy, kin, np.array(SPOT), z, act_fn=GripFloor(28), handover_lift=0.04)
        cmds = np.array([q for q, _ in arm.sent_during]); reads = np.array([r for _, r in arm.sent_during])
        cmd_change = float(np.abs(cmds - cmds[0]).max()) if len(cmds) else None       # the hold command never changes
        drift = float(np.ptp(reads[15:, :5], axis=0).max()) if len(reads) > 15 else None  # arm at rest after 0.5 s
        info = {}
        if res["event"] == "handover":
            info = ScriptedPlace(kin).run(b.arm, res["trace"], np.array(SPOT), z, np.radians(55), log=lambda *a: None,
                                          safety=lambda c: safety_check(kin, c))
        s.wait(3.0)
        pid = b.body_of.get(info.get("slot")) if info.get("slot") else None
        on = pid is not None and np.linalg.norm(g.pose(pid)[0][:2] - SPOT) < 0.03 and g.pose(pid)[0][2] > z
        ok = res["pauses"] >= 1 and cmd_change is not None and cmd_change < 1e-6 and drift < 0.5 and on
        ok_all &= ok
        print(f"{'PASS' if ok else 'FAIL'}  stall {label}: skill {res['event']}, pauses {res['pauses']} ({res['paused_s']} s), "
              f"{len(arm.sent_during)} hold commands, all identical (change {cmd_change}), arm motion after 0.5 s {drift and round(drift, 2)} units, "
              f"place {info.get('event')}, box on the stack {on}", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
