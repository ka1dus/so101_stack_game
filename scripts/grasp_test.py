"""Phase 4 grasp test: N randomized picks per box type from the supply mat (target >= 95 % hold for 2 s).

Per trial: one box of the tested type at a random position (>= 3 cm inside the mat edges,
<= 29 cm from the pan axis (REACH_MAX, the vertically pickable region)) and random yaw; the rest are
parked off the table. Scripted pick with IK (jaws vertical, jaw-opening axis across the
box width, box side ~2 mm from the fixed jaw):
  home -> pre-grasp (5 cm above, gripper open to width + 15 mm) -> straight-down descent through
  Cartesian waypoints every 5 mm (tips end 1 cm below the box top) -> close (torque-limited) -> lift 10 cm -> hold 2 s.
Commands go out at 30 Hz as smooth joint-space ramps (<= 60 deg/s), like the real arm.
Success: box raised >= 8 cm, still between the jaws (< 15 mm from the grip point), tilt < 20 deg.

    MUJOCO_GL=cgl ~/.venvs/so101_stack_sim/bin/python scripts/grasp_test.py [--n 100] [--seed 0] [--video]
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
from joint_mapping import JOINT_NAMES, JointMapping  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LAY = yaml.safe_load((ROOT / "sim/assets/layout.yaml").read_text())
MEAS = yaml.safe_load((ROOT / "inputs/measurements.yaml").read_text())
BOX = {v["sim_name"]: v["dimensions"] for v in MEAS["boxes"]["variants"]}
PREGRASP_READING = {"no3": 32.2, "no35": 29.1, "no10": 25.3}    # width + 15 mm (Phase 1/2)
CTRL_HZ = 30
REACH_MAX = 0.29          # m from the pan axis: vertical grasp IK fails from ~29.5 cm (Phase 4 grasp test log)
_sweep = np.load(ROOT / "notes/phase3_reach_bruteforce.npy")      # r, z, tilt, lift, elbow, wrist (sim rad)
VERT = _sweep[_sweep[:, 2] < 2.0]
MAX_SPEED = np.radians(60)
MAX_REL_RAD = np.radians(10.0)   # real arm: max_relative_target 10 (degrees for body joints)


class Sim:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/scene.xml"))
        self.d = mujoco.MjData(self.m)
        self.jm = JointMapping.from_files()
        self.arm = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(j).id] for j in self.arm])
        self.gadr = self.m.jnt_qposadr[self.m.joint("gripper").id]
        self.act = {j: self.m.actuator(j).id for j in JOINT_NAMES}
        self.gb = self.m.body("gripper").id
        self.rng_lim = np.array([self.m.jnt_range[self.m.joint(j).id] for j in self.arm])
        fg = self.m.geom("finger_gripper_0").id
        self.xin = self.m.geom_pos[fg][0] + self.m.geom_size[fg][0]        # fixed-jaw inner face (gripper x)
        self.ztip = self.m.geom_pos[fg][2] - self.m.geom_size[fg][2]       # jaw tip (gripper z)
        self.home = self.jm.real_to_sim(np.array([MEAS["arm"]["home_pose"][j] for j in JOINT_NAMES]))
        self.steps_per_ctrl = int(round(1 / CTRL_HZ / self.m.opt.timestep))
        self.kin = mujoco.MjData(self.m)
        self.on_tick = None            # called after every 30 Hz control tick (recording hook)
        # DART-style execution noise (data v2): the arm executes plan + OU noise, recorders store the clean plan
        self.plan_q = None; self.plan_g = None
        self.noise_sigma = np.radians(2.0); self.noise_tau = 0.5; self.noise_fade = 0.3
        self._nx = np.zeros(5); self._gain = 0.0; self._gain_target = 0.0; self.noise_rng = np.random.default_rng(0)
        # soft-servo compensation (Phase 8 servo fit): command = plan + (gravity + kv * plan velocity) / kp, at most
        # MAX_REL_RAD from the present reading (the real arm's max_relative_target). None = stiff servos, command = plan.
        self.comp = None; self._prev_plan = None; self._cmd = None

    def sync_plan(self):
        """Take the current actuator targets as the plan (call after writing d.ctrl directly)."""
        self.plan_q = np.array([self.d.ctrl[self.act[j]] for j in self.arm]); self.plan_g = float(self.d.ctrl[self.act["gripper"]])
        self._prev_plan = None; self._cmd = None

    def set_noise(self, on):
        """Fade the execution noise in/out over noise_fade seconds (the OU state keeps evolving)."""
        self._gain_target = 1.0 if on else 0.0

    def set_comp(self, params):
        """params: {joint: (kp, kv, force)} of the servo model the expert assumes (None: off)."""
        self.comp = None if params is None else (np.array([params[j][0] for j in self.arm]), np.array([params[j][1] for j in self.arm]))
        self._prev_plan = None; self._cmd = None

    def _compensate(self, q_plan):
        """The arm command for a planned pose: gravity + damping feed-forward of the assumed servos, clipped to the reading."""
        k = self.kin; k.qpos[:] = self.d.qpos; k.qpos[self.qadr] = q_plan; k.qvel[:] = 0
        mujoco.mj_forward(self.m, k)
        dof = [self.m.jnt_dofadr[self.m.joint(j).id] for j in self.arm]
        qd = np.zeros(5) if self._prev_plan is None else (q_plan - self._prev_plan) * CTRL_HZ
        kp, kv = self.comp; q = self.d.qpos[self.qadr]
        return np.clip(q_plan + (k.qfrc_bias[dof] + kv * qd) / kp, q - MAX_REL_RAD, q + MAX_REL_RAD)

    def clean_ctrl(self):
        """The commanded targets without DART noise in JOINT_NAMES order (the action label); equals d.ctrl when noise is off."""
        return np.r_[self._cmd if self._cmd is not None else self.plan_q, self.plan_g]

    def _apply(self):
        dt = 1.0 / CTRL_HZ
        self._nx += -self._nx / self.noise_tau * dt + self.noise_sigma * np.sqrt(2 / self.noise_tau * dt) * self.noise_rng.standard_normal(5)
        self._gain += np.clip(self._gain_target - self._gain, -dt / self.noise_fade, dt / self.noise_fade)
        q = np.clip(self.plan_q + self._gain * self._nx, self.rng_lim[:, 0], self.rng_lim[:, 1])
        if self.comp is not None:
            self._cmd = np.clip(self._compensate(self.plan_q), self.rng_lim[:, 0], self.rng_lim[:, 1])
            q = np.clip(self._cmd + self._gain * self._nx, self.rng_lim[:, 0], self.rng_lim[:, 1]); self._prev_plan = self.plan_q.copy()
        for j, v in zip(self.arm, q):
            self.d.ctrl[self.act[j]] = v
        self.d.ctrl[self.act["gripper"]] = self.plan_g

    def gripper_sim(self, reading):
        return self.jm.real_to_sim(np.r_[np.zeros(5), reading])[5]

    # ---- IK: grip point (gripper frame) to a world point, jaws vertical, gripper x along a world direction ----
    def ik(self, p_grip, target, xdir, seeds, w_orient=100.0, signed=False):
        k = self.kin
        def r(q):
            k.qpos[self.qadr] = q; mujoco.mj_kinematics(self.m, k)
            R = k.xmat[self.gb].reshape(3, 3)
            pos = k.xpos[self.gb] + R @ p_grip
            r = np.r_[(pos - target) * 1000, (R[:, 2] - [0, 0, 1]) * w_orient, (R[:, 0] @ np.r_[-xdir[1], xdir[0], 0]) * w_orient]
            if signed:                                   # gripper x must point along +xdir (moving jaw side)
                r = np.r_[r, min(0.0, R[:, 0] @ np.r_[xdir[0], xdir[1], 0]) * w_orient]
            return r
        best = None
        for s in seeds:
            sol = least_squares(r, np.clip(s, self.rng_lim[:, 0] + 1e-3, self.rng_lim[:, 1] - 1e-3),
                                bounds=(self.rng_lim[:, 0], self.rng_lim[:, 1]))
            if best is None or sol.cost < best.cost:
                best = sol
        res = r(best.x)
        k.qpos[self.qadr] = best.x; mujoco.mj_kinematics(self.m, k)
        R = k.xmat[self.gb].reshape(3, 3)
        return best.x, np.linalg.norm(res[:3]), float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))

    # ---- motion: 30 Hz targets along a joint-space ramp ----
    def move(self, q_to, g_to, min_time=0.3, video=None):
        if self.plan_q is None:
            self.sync_plan()
        q_from, g_from = self.plan_q.copy(), self.plan_g
        T = max(min_time, np.max(np.abs(q_to - q_from)) / MAX_SPEED)
        n = max(1, int(T * CTRL_HZ))
        for i in range(1, n + 1):
            s = i / n; s = s * s * (3 - 2 * s)                   # smoothstep
            self.plan_q = q_from + s * (q_to - q_from); self.plan_g = g_from + s * (g_to - g_from)
            self._apply()
            for _ in range(self.steps_per_ctrl):
                mujoco.mj_step(self.m, self.d)
            if video is not None:
                video.append(self.d.qpos.copy())
            if self.on_tick is not None:
                self.on_tick()

    def wait(self, t, video=None):
        if self.plan_q is None:
            self.sync_plan()
        for _ in range(int(t * CTRL_HZ)):
            self._apply()
            for _ in range(self.steps_per_ctrl):
                mujoco.mj_step(self.m, self.d)
            if video is not None:
                video.append(self.d.qpos.copy())
            if self.on_tick is not None:
                self.on_tick()


def random_supply_pose(rng):
    S = np.array(LAY["supply_area"]["corners"]); e1 = S[1] - S[0]; e2 = S[3] - S[0]
    for _ in range(1000):
        a, b = rng.uniform(0.03 / np.linalg.norm(e1), 1 - 0.03 / np.linalg.norm(e1)), rng.uniform(0.03 / np.linalg.norm(e2), 1 - 0.03 / np.linalg.norm(e2))
        p = S[0] + a * e1 + b * e2
        if np.linalg.norm(p) <= REACH_MAX:
            return p, rng.uniform(0, np.pi)
    raise RuntimeError("no reachable supply position")


def trial(sim, t, rng, video=None):
    m, d = sim.m, sim.d
    mujoco.mj_resetData(m, d)
    d.qpos[sim.qadr] = sim.home[:5]; d.qpos[sim.gadr] = sim.home[5]
    for j, v in zip(JOINT_NAMES, sim.home):
        d.ctrl[sim.act[j]] = v
    sim.sync_plan()
    target_body = None
    for b in range(m.nbody):
        n = m.body(b).name
        if n.startswith("box"):
            adr = m.jnt_qposadr[m.body_jntadr[b]]
            if target_body is None and n.endswith(t):
                target_body = b
            else:
                d.qpos[adr:adr + 3] = [2 + b * 0.1, 2, 0.01]       # parked far away on the floor plane level
    adr = m.jnt_qposadr[m.body_jntadr[target_body]]
    dim = BOX[t]; p, yaw = random_supply_pose(rng)
    z = 0.0005 + dim["height"] / 2 + 0.0005                         # mat 0.5 mm
    d.qpos[adr:adr + 3] = [p[0], p[1], z]; d.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(m, d)
    for _ in range(200):
        mujoco.mj_step(m, d)                                         # settle
    centre = d.xpos[target_body].copy(); Rb = d.xmat[target_body].reshape(3, 3)
    width_dir = Rb[:, 1]                                             # box local y = width
    top = centre[2] + dim["height"] / 2
    # grip point in the gripper frame: box side 2 mm from the fixed jaw, at the box's mid-width, 1 cm above the tips
    p_grip = np.array([sim.xin + dim["width"] / 2 + 0.002, 0.0, sim.ztip + 0.010])
    pan = -np.arctan2(centre[1], centre[0])
    # seeds: the nearest exactly-vertical lift/elbow/wrist poses from the Phase 3 sweep (planar, so valid at any pan)
    rr = np.linalg.norm(centre[:2]); zz = top + p_grip[2] * 0     # sweep z is the grip point height
    near = VERT[np.argsort((VERT[:, 0] - rr) ** 2 + (VERT[:, 1] - top) ** 2)[:3]]
    seeds = [np.r_[pan, cfg[3:6], roll] for cfg in near for roll in (-4.3, -2.7, -1.2, 0.4)]
    seeds += [np.r_[pan, a_, roll] for a_ in ([0.3, 0.3, 1.0], [0.6, -0.3, 1.2]) for roll in (-4.3, -2.7, -1.2, 0.4)]
    q_grasp, e1, tilt1 = sim.ik(p_grip, np.r_[centre[:2], top], width_dir, seeds)
    # above the box, position matters more than verticality (Phase 3 reach limit); jaws vertical only at the grasp
    q_pre, e2, tilt2 = sim.ik(p_grip, np.r_[centre[:2], top + 0.05], width_dir, [q_grasp] + seeds, w_orient=10.0)
    q_lift, e3, tilt3 = sim.ik(p_grip, np.r_[centre[:2], top + 0.10], width_dir, [q_grasp] + seeds, w_orient=10.0)
    if e1 > 3 or tilt1 > 10:
        return dict(ok=False, reason=f"ik grasp (err {e1:.1f} mm, tilt {tilt1:.0f} deg)", r=float(np.linalg.norm(p)))
    if max(e2, e3) > 5 or max(tilt2, tilt3) > 30:
        return dict(ok=False, reason=f"ik above (err {max(e2,e3):.1f} mm, tilt {max(tilt2,tilt3):.0f} deg)", r=float(np.linalg.norm(p)))
    g_open = sim.gripper_sim(PREGRASP_READING[t]); g_close = sim.gripper_sim(0.0)
    # straight-down descent: Cartesian waypoints every 5 mm, each as vertical as reachable, seeded from the last
    path, q_prev = [], q_grasp
    for dz in np.arange(0.05, 0.0, -0.005)[::-1]:          # solve bottom-up so each seed is close
        q_w, e_w, _ = sim.ik(p_grip, np.r_[centre[:2], top + dz], width_dir, [q_prev], w_orient=100.0)
        if e_w > 3:
            q_w, e_w, _ = sim.ik(p_grip, np.r_[centre[:2], top + dz], width_dir, [q_prev], w_orient=10.0)
        path.append(q_w); q_prev = q_w
    path = path[::-1]                                       # top -> bottom
    sim.move(path[0], g_open, 1.0, video)
    for q_w in path[1:] + [q_grasp]:
        sim.move(q_w, g_open, 0.1, video)
    sim.move(q_grasp, g_close, 0.6, video)
    sim.wait(0.3, video)
    z_before = d.xpos[target_body][2]
    sim.move(q_lift, g_close, 1.0, video)
    sim.wait(2.0, video)
    R = d.xmat[sim.gb].reshape(3, 3); grip_w = d.xpos[sim.gb] + R @ p_grip
    rise = d.xpos[target_body][2] - z_before
    sep = np.linalg.norm((d.xpos[target_body] - grip_w)[:2])
    tilt = np.degrees(np.arccos(np.clip(abs(d.xmat[target_body].reshape(3, 3)[2, 2]), -1, 1)))
    ok = rise > 0.08 and sep < 0.015 and tilt < 20
    reason = "ok" if ok else ("dropped/not lifted" if rise <= 0.08 else ("slipped in grasp" if sep >= 0.015 else "tilted in grasp"))
    return dict(ok=ok, reason=reason, rise=float(rise), sep=float(sep), tilt=float(tilt), r=float(np.linalg.norm(p)),
                reading_closed=float(sim.jm.sim_to_real(np.r_[np.zeros(5), d.qpos[sim.gadr]])[5]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--types", default="no3,no35,no10"); ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    sim = Sim(); rng = np.random.default_rng(a.seed)
    summary = {}
    for t in a.types.split(","):
        res = [trial(sim, t, rng) for _ in range(a.n)]
        ok = sum(r["ok"] for r in res)
        reasons = {}
        for r in res:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        closed = [r["reading_closed"] for r in res if "reading_closed" in r]
        summary[t] = ok / a.n
        print(f"{t:5s}: {ok}/{a.n} held 2 s ({ok/a.n*100:.0f} %)  outcomes {reasons}  "
              f"closed reading {np.mean(closed):.1f} +/- {np.std(closed):.1f}" if closed else f"{t}: no completed trials {reasons}")
    print("PASS" if all(v >= 0.95 for v in summary.values()) else "FAIL", "(target >= 95 % per type)")
    if a.video:
        import imageio  # optional
        print("video not implemented in batch mode; use --n 1 with a viewer")


if __name__ == "__main__":
    main()
