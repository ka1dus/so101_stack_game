"""Phase 8: servo tracking test (real arm, or the same motions in sim). Data for fitting the sim servo model.

From home, the arm follows a fixed joint-space script high above the white table (tips 10-28 cm up, pan -15..+5 deg,
away from both mats and the backdrop): smooth moves between 4 poses at a slow (20 units/s) and a fast (60 units/s)
speed with 2 s holds (steady-state sag), +-8-unit steps on shoulder_lift / elbow_flex / wrist_flex, and an elbow
sine. Every target is clipped to +-10 units of the reading (as in the skill) and checked by the safety envelope.
Run twice: without a box, and holding a large box (No.3) in the jaws.

    # real arm (no cameras needed), arm at home first
    ~/.venvs/so101_eval/bin/python scripts/phase8_tracking.py --port /dev/tty.usbmodem5B910311481 --label empty
    ~/.venvs/so101_eval/bin/python scripts/phase8_tracking.py --port /dev/tty.usbmodem5B910311481 --label box
    # the same script in sim (contact check + the sim response)
    MUJOCO_GL=cgl ~/.venvs/so101_eval/bin/python scripts/phase8_tracking.py --sim --label empty
Ctrl-C: the arm holds where it is, then you choose: return home or torque off.
Log: data/real_tracking/<time>_<label>.npz (reading[6], target sent[6], time per tick), or ..._sim_<label>.npz.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from game.backends import saved_home  # noqa: E402
from perception.drop import Kinematics  # noqa: E402
from servo_model import apply_params  # noqa: E402
from skill.runner import FPS, MAX_REL, safety_check  # noqa: E402

POSES = {"A": [0, -60, 50, 55], "B": [-15, -60, 30, 40], "C": [5, -40, 40, 60], "D": [-15, -45, 55, 70]}
SLOW, FAST, HOLD = 20.0, 60.0, 2.0
GRIP_BOX = 0.0          # gripper target while holding the box (the box stops the jaws; torque-limited)
GRIP_OPEN = 40.0        # for handing the box in and out


def minjerk(a, b, speed):
    a, b = np.asarray(a, float), np.asarray(b, float); T = max(np.abs(b[:5] - a[:5]).max() / speed, 0.3)
    s = np.linspace(0, 1, int(round(T * FPS)) + 1)[1:]; s = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    return [a + (b - a) * v for v in s]


def script(home, grip):
    """The target sequence (LeRobot units, 30 Hz) and a label per tick."""
    full = lambda p: np.r_[p, home[4], grip]
    out, lab = [], []

    def seg(path, name):
        out.extend(path); lab.extend([name] * len(path))

    def hold(name, s=HOLD):
        seg([out[-1].copy()] * int(s * FPS), name)
    start = np.r_[home[:5], grip]
    seg(minjerk(start, full(POSES["A"]), SLOW), "home>A slow"); hold("hold A")
    for a, b, v in (("A", "B", FAST), ("B", "C", SLOW), ("C", "D", FAST), ("D", "A", FAST), ("A", "C", FAST), ("C", "A", SLOW)):
        seg(minjerk(full(POSES[a]), full(POSES[b]), v), f"{a}>{b} {'fast' if v == FAST else 'slow'}"); hold(f"hold {b}")
    for j, name in ((1, "lift"), (2, "elbow"), (3, "wflex")):
        for d in (8, -8):
            q = out[-1].copy(); q[j] = POSES["A"][j] + d; seg([q] * int(1.5 * FPS), f"step {name} {d:+d}")
            q = q.copy(); q[j] = POSES["A"][j]; seg([q] * int(1.5 * FPS), f"step {name} back")
    t = np.arange(int(4 * FPS)) / FPS
    for v in t:
        q = full(POSES["A"]); q[2] += 10 * np.sin(2 * np.pi * 0.5 * v); out.append(q); lab.append("elbow sine")
    hold("hold A", 1.0)
    seg(minjerk(full(POSES["A"]), np.r_[home[:5], grip], SLOW), "A>home slow"); hold("hold home", 1.0)
    return np.array(out), lab


def check(kin, cmds):
    bad = [(i, why) for i, q in enumerate(cmds) if (why := safety_check(kin, q))]
    tips = np.array([kin.tip_pos(q.tolist()) for q in cmds])
    return bad, tips


class SimArm:
    """The MuJoCo twin behind the same read/send interface; optionally a large box held in the jaws."""
    virtual = True

    def __init__(self, box=False, params=None):
        import mujoco
        import grasp_test as GT
        from perception.supply import BOX
        self.mj = mujoco; self.s = GT.Sim(); s = self.s; m, d = s.m, s.d
        if params is not None:
            apply_params(m, params)
        self.jm = s.jm; mujoco.mj_resetData(m, d); self.box = None
        for b in range(m.nbody):                                   # park every box away from the table
            if m.body(b).name.startswith("box"):
                a = m.jnt_qposadr[m.body_jntadr[b]]; d.qpos[a:a + 3] = [2 + b * 0.1, 2, 0.01]
        self.dim = BOX["no3"]; self.loaded = box; self.contacts = []

    def start(self, q_real):
        s, m, d, mj = self.s, self.s.m, self.s.d, self.mj
        q = self.jm.real_to_sim(np.asarray(q_real, float)); d.qpos[s.qadr] = q[:5]; d.qpos[s.gadr] = q[5]
        for j, v in zip(s.arm + ["gripper"], q):
            d.ctrl[s.act[j]] = v
        mj.mj_forward(m, d)
        if self.loaded:                                            # the held box as its weight at the grip point (a floating
            g = s.gb; mb = 0.038; mg = m.body_mass[g]       # box gets squeezed out by the closing jaws in sim)
            pg = np.array([s.xin + self.dim["width"] / 2 + 0.002, 0, s.ztip + 0.010])
            m.body_ipos[g] = (mg * m.body_ipos[g] + mb * pg) / (mg + mb); m.body_mass[g] = mg + mb
            mj.mj_setConst(m, d); mj.mj_forward(m, d)
        for _ in range(int(0.5 / m.opt.timestep)):
            mj.mj_step(m, d)

    def joints(self):
        s = self.s; return self.jm.sim_to_real(np.r_[s.d.qpos[s.qadr], s.d.qpos[s.gadr]])

    def send(self, q):
        s, m, d = self.s, self.s.m, self.s.d; c = self.jm.real_to_sim(np.asarray(q, float))
        for j, v in zip(s.arm + ["gripper"], c):
            d.ctrl[s.act[j]] = np.clip(v, *m.actuator_ctrlrange[s.act[j]])
        for _ in range(s.steps_per_ctrl):
            self.mj.mj_step(m, d)
        for k in range(d.ncon):                                   # arm touching anything but the held box
            b1, b2 = m.geom_bodyid[d.contact[k].geom1], m.geom_bodyid[d.contact[k].geom2]
            names = {m.body(b1).name, m.body(b2).name}
            if any(n in ("gripper", "moving_jaw_so101_v1", "wrist", "lower_arm", "upper_arm", "shoulder", "wrist_camera_housing") for n in names):
                self.contacts.append(sorted(names))



def run(arm, cmds, real):
    log = []; t0 = time.perf_counter()
    for c in cmds:
        t1 = time.perf_counter(); q = arm.joints()
        a = np.clip(c, q - MAX_REL, q + MAX_REL); arm.send(a); log.append(np.r_[q, a, t1 - t0])
        if real:
            time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - t1)))
    return np.array(log, np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port"); ap.add_argument("--id", default="my_follower_arm")
    ap.add_argument("--label", choices=["empty", "box"], required=True); ap.add_argument("--sim", action="store_true")
    a = ap.parse_args(); kin = Kinematics(); home = saved_home(); grip = GRIP_BOX if a.label == "box" else home[5]
    cmds, lab = script(home, grip); bad, tips = check(kin, cmds)
    print(f"script: {len(cmds)} ticks ({len(cmds) / FPS:.0f} s); tips z {tips[:, 2].min() * 100:.1f}-{tips[:, 2].max() * 100:.1f} cm "
          f"(away from home: >= {tips[45:-45, 2].min() * 100:.1f}); safety-check failures: {len(bad)}")
    if bad:
        print("  first:", bad[0]); return
    out = ROOT / "data/real_tracking"; out.mkdir(parents=True, exist_ok=True); stamp = time.strftime("%Y%m%d_%H%M%S")
    if a.sim:
        arm = SimArm(box=a.label == "box"); arm.start(home); log = run(arm, cmds, False)
        print(f"sim: arm contacts {len(arm.contacts)} {sorted(set(map(tuple, arm.contacts)))[:5]}")
        np.savez(out / f"{stamp}_sim_{a.label}.npz", log=log, labels=np.array(lab)); return
    from skill.runner import RealArm, move_to
    arm = RealArm(a.port, a.id, front=None); arm.connect()
    try:
        q = arm.joints()
        if np.abs(q[:5] - home[:5]).max() > 3:
            print(f"the arm is not at home ({np.round(q, 1).tolist()}); run scripts/so101_go_home.py first"); return
        if a.label == "box":
            move_to(arm, np.r_[q[:5], GRIP_OPEN], kin, speed=20)
            input("Hold a LARGE box (No.3) flat between the jaws, width across the jaws, then press Enter to close them: ")
            move_to(arm, np.r_[q[:5], GRIP_BOX], None, speed=20); time.sleep(1.0)
            input("Let go of the box. Hands out of the arm's way; press Enter to start (Ctrl-C holds the arm): ")
        else:
            input("Hands out of the arm's way; press Enter to start (Ctrl-C holds the arm): ")
        try:
            log = run(arm, cmds, True)
        except KeyboardInterrupt:
            arm.hold(); print("\nheld.")
            if input("h = return home slowly, anything else = torque off: ").strip() == "h":
                move_to(arm, np.r_[home[:5], arm.joints()[5]], kin, speed=15)
            else:
                arm.torque_off()
            return
        np.savez(out / f"{stamp}_{a.label}.npz", log=log, labels=np.array(lab))
        print(f"saved {out / f'{stamp}_{a.label}.npz'}; worst gap {np.abs(log[:, 6:11] - log[:, :5]).max():.1f} units")
        if a.label == "box":
            input("Hold the box, then press Enter to open the jaws: "); move_to(arm, np.r_[arm.joints()[:5], GRIP_OPEN], None, speed=20)
            input("Take the box out, then press Enter to close the jaws to home: "); move_to(arm, home, kin, speed=20)
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
