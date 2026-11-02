"""Scripted place: the policy picks and lifts the box, then this module puts it on the stack (no retraining).

Real stage 2 (29 Oct, notes/phase8.md §18-19): the policy picks the no3 reliably but releases it 11-27 mm off the
stack top centre (one direction) and 3-22 mm above the face, so it falls off small tops; the sim twin of the same
trials releases 3-15 mm off and 0-11 mm up. Here the release pose is computed instead, like the data expert
(scripts/expert.py): box centre over the top centre, box yaw = top yaw (the nearer of the two flat directions),
lowest box point RELEASE_CLEARANCE above the top face, slow last 2 cm, open wide, straight up.

Box in the gripper: the box lies in its template slot (sim/assets/supply_layout.yaml, +-2 mm) until the jaws lift it,
so box-in-gripper = slot pose seen from the gripper (FK of the reading) at the last tick before the lift.
Soft servos: every waypoint is corrected from the reading (command += plan - reading, CORR_ITERS times while
holding), so gravity sag and lag do not move the release point.
All motion in LeRobot units, +-MAX_REL per tick, through runner.safety_check.
"""
import sys
import time

import numpy as np
import yaml
from scipy.optimize import least_squares

from perception.camera import ROOT
from perception.supply import BOX

sys.path.insert(0, str(ROOT / "scripts"))
import game_states as G  # noqa: E402

RELEASE_CLEARANCE = 0.003      # lowest box point above the top face at release (expert)
HOVER = 0.05                   # above the release pose before the descent
SLOW_LAST = 0.02               # last part of the descent at slow speed
SPEED_FAST, SPEED_SLOW = 0.05, 0.01   # m/s along the descent
OPEN_TO = 32.0                 # gripper reading at release (~58 mm at the tips)
RETRACT = 0.05
TOUCH_UNITS = 1.5              # gripper reading drop that means the box met the stack during the descent
TOUCH_ZONE = 0.012             # ...counted only in the last 12 mm (sim soft servos: the reading wobbled as the arm
                               # slowed, stopping a place 3 cm high)
LOOSEN = 4.0                   # gripper units: first release step (stops the clamp, jaw gap +6 mm)
OPEN_LIFT = 0.008              # lift with the jaws loose before opening fully
CARRY_EXTRA = 0.03             # transit height: this much above the higher of the box now and the hover pose
CORR_ITERS = 3                 # reading-based corrections at the hover and release poses
W_FLAT, W_YAW = 30.0, 10.0     # IK residual weights (mm per unit vector): position first, then a flat box, then yaw
LAYOUT = yaml.safe_load((ROOT / "sim/assets/supply_layout.yaml").read_text())["boxes"]


def slot_poses():
    """{slot id: (type, centre xyz, R box->world)} of the template layout."""
    S = G.SUPPLY_RECT; head = np.arctan2(*(S[1] - S[0])[::-1]); out = {}
    for b in LAYOUT:
        yaw = head + (0 if b["long_axis"] == "along edge 1->2" else np.pi / 2)
        c, s = np.cos(yaw), np.sin(yaw)
        out[b["id"]] = (b["type"], np.r_[b["centre_world_m"], G.MAT_Z + BOX[b["type"]]["height"] / 2],
                        np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))
    return out


def jaw_gap(reading):
    """Jaw gap 1 cm back from the tips (m) at a gripper reading: linear fit to the Phase 1 geometry
    (scripts/analysis/phase1_gripper_fit.py: 20 -> 35.2 mm, 24 -> 41.8, 27 -> 46.7, 30 -> 51.5)."""
    return (35.2 + (reading - 20.0) * 1.63) / 1000


class ScriptedPlace:
    def __init__(self, kin):
        self.kin = kin; self.lo, self.hi = kin.jm.lerobot_limits().T; self.slots = slot_poses()
        fg = kin.m.geom("finger_gripper_0").id
        self.xin = kin.m.geom_pos[fg][0] + kin.m.geom_size[fg][0]          # fixed-jaw inner face (gripper x)

    def pose(self, q):
        """Gripper body pose (p, R) for a LeRobot reading."""
        self.kin.tip_pos(q); return self.kin.d.xpos[self.kin.gb].copy(), self.kin.d.xmat[self.kin.gb].reshape(3, 3).copy()

    def box_in_gripper(self, trace, full=None):
        """From the skill trace up to the handover: (slot, type, b_p, b_R) of the held box in the gripper frame.
        full: the slots that still hold a box (the game's slot check); others can't be the picked one (real game
        30 Oct: the nearest slot was already empty, so a no3 picked near its end was placed as a no35)."""
        q = trace[:, :6].astype(float); z = np.array([self.kin.tip_pos(x.tolist())[2] for x in q])
        k0 = int(np.argmin(z)); k = k0
        while k + 1 < len(z) and z[k + 1] < z[k0] + 0.002:       # last tick before the lift: box still in its slot
            k += 1
        p, R = self.pose(q[k].tolist())
        guess = p + R @ np.array([0.013, 0.0, -0.102])          # box centre in the gripper on the real arm (29 Oct)
        cand = [s for s in self.slots if full is None or s in full] or list(self.slots)
        sid = min(cand, key=lambda s: np.linalg.norm(self.slots[s][1][:2] - guess[:2]))
        self.guess_mm = float(np.linalg.norm(self.slots[sid][1][:2] - guess[:2]) * 1000)
        typ, c, Rb = self.slots[sid]
        b_p = R.T @ (c - p)
        # across the jaws the closing moving jaw pushes the box against the fixed jaw (sim: 8 mm off the slot), so the
        # box centre is at the fixed jaw's inner face + half the jaw gap at the held reading
        b_p[0] = self.xin + jaw_gap(float(trace[-1][5])) / 2
        return sid, typ, b_p, R.T @ Rb

    def ik(self, q_seed, b_p, b_R, target, yaw):
        """Body joints (LeRobot) putting the box centre at `target`, flat, long axis along yaw or yaw + pi."""
        g = q_seed[5]; ax = np.array([np.cos(yaw), np.sin(yaw), 0.0])

        def r(x):
            p, R = self.pose(np.r_[x, g]); Rb = R @ b_R
            return np.r_[(p + R @ b_p - target) * 1000, (Rb[:, 2] - [0, 0, 1]) * W_FLAT, np.cross(Rb[:, 0], ax)[2] * W_YAW]
        sol = least_squares(r, np.clip(q_seed[:5], self.lo[:5] + 0.1, self.hi[:5] - 0.1), bounds=(self.lo[:5], self.hi[:5]))
        res = r(sol.x); tilt = np.degrees(np.arccos(np.clip(res[5] / W_FLAT + 1, -1, 1)))
        return np.r_[sol.x, g], float(np.linalg.norm(res[:3])), float(tilt)

    def lowest_point(self, q, b_p, b_R, typ):
        p, R = self.pose(q); Rb = R @ b_R; c = p + R @ b_p; dm = BOX[typ]
        corners = [c + Rb @ np.array([a * dm["length"] / 2, b * dm["width"] / 2, -dm["height"] / 2]) for a in (-1, 1) for b in (-1, 1)]
        return min(x[2] for x in corners)

    def run(self, arm, trace, top_xy, top_face, top_yaw, log=print, safety=None, max_rel=10.0, fps=30, on_tick=None, paused=None,
            full=None):
        """Place the held box. on_tick(command): called after every command sent (recording). paused(): while True the
        arm holds (the game's hand check). Returns dict(event, slot,
        type, b_in_g_mm, ik_err_mm, release_err_mm, release_clear_mm, ticks, release_index (tick the opening starts))."""
        sent = [0]
        virtual = getattr(arm, "virtual", False)
        sid, typ, b_p, b_R = self.box_in_gripper(trace, full)
        q = arm.joints(); _, R0 = self.pose(q.tolist()); ax0 = (R0 @ b_R)[:2, 0]
        yaw = top_yaw if np.cos(np.arctan2(ax0[1], ax0[0]) - top_yaw) >= 0 else top_yaw + np.pi   # nearer flat direction
        h = BOX[typ]["height"]; z_rel = top_face + h / 2 + RELEASE_CLEARANCE
        tgt = np.r_[top_xy, z_rel]
        q_rel, e_rel, _ = self.ik(q, b_p, b_R, tgt, yaw)
        for _ in range(3):                                       # a tilted box: raise until its lowest corner clears
            deficit = top_face + RELEASE_CLEARANCE - self.lowest_point(q_rel.tolist(), b_p, b_R, typ)
            if deficit <= 0.0005:
                break
            tgt[2] += deficit; q_rel, e_rel, _ = self.ik(q_rel, b_p, b_R, tgt, yaw)
        info = dict(slot=sid, guess_mm=round(self.guess_mm, 1), type=typ, b_in_g_mm=np.round(b_p * 1000, 1).tolist(), ik_err_mm=round(e_rel, 1))
        log(f"  scripted place: {typ} from slot {sid}, box in gripper {info['b_in_g_mm']} mm; target ({tgt[0]*100:.1f}, "
            f"{tgt[1]*100:.1f}, {tgt[2]*100:.1f}) cm, yaw {np.degrees(yaw):.0f} deg, IK error {e_rel:.1f} mm")
        if e_rel > 5:
            info["event"] = f"place IK error {e_rel:.0f} mm"; return info

        g_hold = float(trace[-1][11]); g_carry = float(arm.joints()[5])                    # keep the policy's grip command (the reading is the box width)
        cmd = np.r_[q[:5], g_hold]

        def send_toward(goal, ticks_min=1):
            """Ramp the command to goal at <= max_rel per tick (all joints), at least ticks_min ticks."""
            nonlocal cmd
            start = cmd.copy(); n = max(ticks_min, int(np.ceil(np.abs(goal - start).max() / (max_rel / 2))))
            for i in range(1, n + 1):
                while paused is not None and paused():         # hand in the workspace: hold the current command
                    arm.send(cmd)                              # (an e-stop raises out of paused())
                    if not virtual:
                        time.sleep(1 / fps)
                t0 = time.perf_counter(); c = np.clip(start + (goal - start) * i / n, self.lo, self.hi)
                rd = arm.joints()                            # stay within max_rel of the reading, as the runner does
                c = np.clip(c, rd - max_rel, rd + max_rel)   # (soft servos lag; LeRobot would clamp it anyway)
                if safety is not None and (why := safety(c)):
                    raise RuntimeError(why)
                arm.send(c); cmd = c; sent[0] += 1
                if on_tick is not None:
                    on_tick(c)
                if not virtual:
                    time.sleep(max(0.0, 1 / fps - (time.perf_counter() - t0)))

        def settle_at(q_plan, off, hold_s=0.3):
            """Hold q_plan + off; correct off by the reading's error (soft servos sag). Returns the new offset."""
            off = off.copy()
            for _ in range(CORR_ITERS):
                send_toward(q_plan + off, ticks_min=int(hold_s * fps))
                err = q_plan[:5] - arm.joints()[:5]
                if np.abs(err).max() < 0.2:
                    break
                off[:5] += err
            return off

        def line(z_from, z_to, speed, q_s, off, g, stop_on_touch=False):
            """Straight vertical line of the box centre, IK every tick, command = plan + offset. stop_on_touch: stop
            when the gripper reading falls TOUCH_UNITS below the carry reading (the box met the stack and is being
            pushed out of the jaws; sim: a box tilted 6 deg in the grasp touched early, slipped out, and the moving
            jaw closed over its top and dragged it off when opening)."""
            n = max(1, int(abs(z_from - z_to) / speed * fps)); qq = q_s
            for i in range(1, n + 1):
                tz = tgt.copy(); tz[2] = z_from + (z_to - z_from) * i / n
                qq, _, _ = self.ik(qq, b_p, b_R, tz, yaw); c = qq + off; c[5] = g
                send_toward(c)
                if stop_on_touch and tz[2] - tgt[2] < TOUCH_ZONE and arm.joints()[5] < g_carry - TOUCH_UNITS:
                    info["touch_stop_mm"] = round((tz[2] - tgt[2]) * 1000, 1); break
            return qq

        try:
            q_rel[5] = g_hold
            q_hov, _, _ = self.ik(q_rel, b_p, b_R, tgt + [0, 0, HOVER], yaw)
            p0, R0 = self.pose(q.tolist()); b0 = p0 + R0 @ b_p; z_carry = max(b0[2], tgt[2] + HOVER) + CARRY_EXTRA
            q_up, _, _ = self.ik(np.r_[q[:5], g_hold], b_p, b_R, np.r_[b0[:2], z_carry], yaw)    # up where it is,
            q_over, _, _ = self.ik(q_hov, b_p, b_R, np.r_[tgt[:2], z_carry], yaw)              # across, then down
            for qq, tmin in ((q_up, 0.8), (q_over, 1.5), (q_hov, 0.8)):
                qq = qq.copy(); qq[5] = g_hold; send_toward(qq, ticks_min=int(tmin * fps))
            off = settle_at(q_hov, np.zeros(6))
            q_mid = line(tgt[2] + HOVER, tgt[2] + SLOW_LAST, SPEED_FAST, q_hov, off, g_hold)
            q_end = line(tgt[2] + SLOW_LAST, tgt[2], SPEED_SLOW, q_mid, off, g_hold, stop_on_touch=True)
            if "touch_stop_mm" not in info:
                off = settle_at(q_end, off)                                    # fresh correction at the release pose
            qr = arm.joints().tolist(); p_now, R_now = self.pose(qr); bc = p_now + R_now @ b_p
            info["release_err_mm"] = np.round((bc[:2] - top_xy) * 1000, 1).tolist()
            info["release_clear_mm"] = round((self.lowest_point(qr, b_p, b_R, typ) - top_face) * 1000, 1)
            log(f"  release: box centre {info['release_err_mm']} mm from the top centre, lowest point {info['release_clear_mm']} mm above the face")
            g_open = max(OPEN_TO, qr[5] + 12); info["release_index"] = sent[0]
            # release in three steps: loosen (the box drops onto the stack), lift OPEN_LIFT with the jaws loose (a moving
            # jaw resting on the box top in a bad grasp, reading ~10, comes off instead of dragging the box and the stack
            # sideways when it swings open: sim, 3 cm), then open fully
            g0 = cmd[5]; c = cmd.copy(); c[5] = g0 + LOOSEN; send_toward(c, ticks_min=int(0.3 * fps))
            for _ in range(int(0.2 * fps)):
                arm.send(cmd); sent[0] += 1
                if on_tick is not None:
                    on_tick(cmd)
                if not virtual:
                    time.sleep(1 / fps)
            q_end = line(tgt[2], tgt[2] + OPEN_LIFT, SPEED_SLOW, q_end, off, g0 + LOOSEN); tgt[2] += OPEN_LIFT
            c = cmd.copy(); c[5] = g_open; send_toward(c, ticks_min=int(0.4 * fps))
            for _ in range(int(0.3 * fps)):
                arm.send(cmd); sent[0] += 1
                if on_tick is not None:
                    on_tick(cmd)
                if not virtual:
                    time.sleep(1 / fps)
            line(tgt[2], tgt[2] + RETRACT, SPEED_FAST, q_end, off, g_open)
            info["event"] = "released"
        except RuntimeError as e:
            arm.hold(); info["event"] = f"guard: {e}"
        info["ticks"] = sent[0]
        return info
