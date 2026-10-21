"""Phase 5: scripted expert (privileged state + IK) for the place skill, CLAUDE.md Phase 5.

  1. start at home (+/- 3 deg jitter per joint)
  2. pick the chosen box (game_states pick rule): moving jaw on the roomier side, pre-grasp
     opening = box width + ~6 mm, straight-down descent (Cartesian waypoints), close (torque-limited),
     straight-up lift
  3. carry above the stack top; yaw aligned with the top box; target xy chosen inside the top
     face so the combined centre of mass of everything above each level stays deepest inside that
     level's support (matters most for large-on-small)
  4. descend straight down, slowing near the top, to a 3 mm release clearance
  5. open gently, retract straight up 5 cm; the episode ends there (no return home)
Success: the new box is on top and nothing in the stack moved/tilted, 3 s after release.

    MUJOCO_GL=cgl ~/.venvs/so101_stack_sim/bin/python scripts/expert.py --n 20 [--seed 0]
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import game_states as G  # noqa: E402
import grasp_test as GT  # noqa: E402
from joint_mapping import JOINT_NAMES  # noqa: E402

# pre-grasp opening = width + ~6 mm: sim contact readings (Phase 4 close test) + 6.5 units (~0.93 mm/unit)
CONTACT_READING = {"no3": 20.7, "no35": 17.6, "no10": 13.4}
PREGRASP = {t: r + 6.5 for t, r in CONTACT_READING.items()}
RELEASE_CLEARANCE = 0.003
APPROACH_ABOVE = 0.04          # box bottom this far above the stack top before the straight descent
LIFT = 0.06
RETRACT = 0.05
SLOW_ZONE = 0.02               # last 2 cm of the place descent at reduced speed


class Expert:
    def __init__(self, sim: GT.Sim, rng):
        self.sim, self.rng = sim, rng

    # ---------------------------------------------------------------- helpers
    def seeds(self, target):
        sim = self.sim
        pan = -np.arctan2(target[1], target[0]); r = np.linalg.norm(target[:2])
        near = GT.VERT[np.argsort((GT.VERT[:, 0] - r) ** 2 + (GT.VERT[:, 1] - target[2]) ** 2)[:3]]
        out = [np.r_[pan, c[3:6], roll] for c in near for roll in (-4.3, -2.7, -1.2, 0.4)]
        out += [np.r_[pan, a, roll] for a in ([0.3, 0.3, 1.0], [0.6, -0.3, 1.2]) for roll in (-4.3, -2.7, -1.2, 0.4)]
        return out

    def straight(self, p_grip, xy, z_from, z_to, xdir, q_start, g, step=0.005, speed_time=0.1, slow=None):
        """Move the grip point straight up/down through Cartesian waypoints (IK seeded from the last)."""
        sim = self.sim; q_prev = q_start
        zs = np.arange(z_from, z_to, step if z_to > z_from else -step)[1:].tolist() + [z_to]
        for z in zs:
            q, e, t = sim.ik(p_grip, np.r_[xy, z], xdir, [q_prev], w_orient=100.0, signed=True)
            if e > 2:
                q, e, t = sim.ik(p_grip, np.r_[xy, z], xdir, [q_prev] + self.seeds(np.r_[xy, z]), w_orient=100.0, signed=True)
            if e > 3:
                q, e, t = sim.ik(p_grip, np.r_[xy, z], xdir, [q_prev] + self.seeds(np.r_[xy, z]), w_orient=10.0, signed=True)
            self.worst_err = max(self.worst_err, e)
            dt = speed_time if slow is None or abs(z - z_to) > SLOW_ZONE else slow
            sim.move(q, g, dt); q_prev = q
        return q_prev

    def place_target(self, stack, carried):
        """xy for the carried box on the top face maximizing the worst support margin over all levels."""
        top = stack[-1]; dt = G.BOX[top.type]; dc = G.BOX[carried]
        c, s = np.cos(top.yaw), np.sin(top.yaw); u, v = np.array([c, s]), np.array([-s, c])
        masses = {"no3": 0.038, "no35": 0.031, "no10": 0.021}
        best = None
        for a in np.linspace(-0.4, 0.4, 9) * dt["length"]:
            for b in np.linspace(-0.4, 0.4, 9) * dt["width"]:
                p = top.xy + a * u + b * v
                worst = np.inf
                for k in range(len(stack)):                        # support level k carries everything above it
                    above = [(stack[j].xy, masses[stack[j].type]) for j in range(k + 1, len(stack))] + [(p, masses[carried])]
                    com = sum(x * m for x, m in above) / sum(m for _, m in above)
                    sk = stack[k]; dk = G.BOX[sk.type]; ck, sk_ = np.cos(sk.yaw), np.sin(sk.yaw)
                    rel = com - sk.xy; lu = rel @ np.array([ck, sk_]); lv = rel @ np.array([-sk_, ck])
                    worst = min(worst, dk["length"] / 2 - abs(lu), dk["width"] / 2 - abs(lv))
                # the carried box itself on the top face
                worst = min(worst, dt["length"] / 2 - abs(a) + 0 * dc["length"])
                if best is None or worst > best[0]:
                    best = (worst, p)
        return best[1], best[0]

    # ---------------------------------------------------------------- episode
    def run(self, st, record=None):
        sim, m, d, rng = self.sim, self.sim.m, self.sim.d, self.rng
        self.worst_err = 0.0
        # home with jitter
        q0 = sim.home.copy(); q0[:5] += np.radians(rng.uniform(-3, 3, 5))
        d.qpos[sim.qadr] = q0[:5]; d.qpos[sim.gadr] = q0[5]
        for j, v in zip(JOINT_NAMES, q0):
            d.ctrl[sim.act[j]] = v
        d.qvel[:] = 0; mujoco.mj_forward(m, d)
        sim.on_tick = record
        box = st.supply[st.pick]; dim = G.BOX[box.type]; bid = m.body(box.body).id
        centre = d.xpos[bid].copy(); Rb = d.xmat[bid].reshape(3, 3); top_z = centre[2] + dim["height"] / 2
        side = st.meta.get("moving_jaw_side", 1) or 1
        xdir = side * Rb[:2, 1] / np.linalg.norm(Rb[:2, 1])        # gripper +x -> moving jaw side
        p_grip = np.array([sim.xin + dim["width"] / 2 + 0.002, 0.0, sim.ztip + 0.010])
        q_g, e, tilt = sim.ik(p_grip, np.r_[centre[:2], top_z], xdir, self.seeds(np.r_[centre[:2], top_z]), signed=True)
        if e > 3 or tilt > 10:
            return dict(ok=False, reason="ik_pick")
        g_open = sim.gripper_sim(PREGRASP[box.type]); g_close = sim.gripper_sim(0.0)
        # pre-grasp above, then straight down
        q_pre, e2, _ = sim.ik(p_grip, np.r_[centre[:2], top_z + 0.05], xdir, [q_g] + self.seeds(np.r_[centre[:2], top_z + 0.05]),
                              w_orient=10.0, signed=True)
        sim.move(q_pre, g_open, 1.0)
        q = self.straight(p_grip, centre[:2], top_z + 0.05, top_z, xdir, q_pre, g_open)
        sim.move(q, g_close, 0.6); sim.wait(0.3)
        q = self.straight(p_grip, centre[:2], top_z, top_z + LIFT, xdir, q, g_close)
        # held-box pose in the gripper frame (privileged)
        Rg = d.xmat[sim.gb].reshape(3, 3); xg = d.xpos[sim.gb]
        b_in_g = Rg.T @ (d.xpos[bid] - xg)
        if d.xpos[bid][2] < top_z + LIFT * 0.6 - dim["height"]:
            return dict(ok=False, reason="missed_grasp")
        Rbg = Rg.T @ d.xmat[bid].reshape(3, 3)                       # box orientation in the gripper frame
        # place target
        top = st.stack[-1]; ht = G.BOX[top.type]["height"]
        tid = m.body(top.body).id; top_c = d.xpos[tid].copy(); Rt = d.xmat[tid].reshape(3, 3)
        top.xy = top_c[:2]; top.yaw = float(np.arctan2(Rt[1, 0], Rt[0, 0]))
        for s_ in st.stack:
            sid = m.body(s_.body).id; s_.xy = d.xpos[sid][:2].copy()
            Rs = d.xmat[sid].reshape(3, 3); s_.yaw = float(np.arctan2(Rs[1, 0], Rs[0, 0]))
        xy_t, margin = self.place_target(st.stack, box.type)
        z_rel = top_c[2] + ht / 2 + dim["height"] / 2 + RELEASE_CLEARANCE            # carried box centre at release
        # gripper x in world so that the carried box ends with yaw = top yaw (mod pi, nearest)
        box_x_in_g = Rbg[:, 0]                                       # carried box length axis, gripper frame
        cur_yaw = np.arctan2((Rg @ box_x_in_g)[1], (Rg @ box_x_in_g)[0])
        dyaw = ((top.yaw - cur_yaw + np.pi / 2) % np.pi) - np.pi / 2
        gx = Rg[:2, 0]; ca, sa = np.cos(dyaw), np.sin(dyaw)
        xdir_p = np.array([ca * gx[0] - sa * gx[1], sa * gx[0] + ca * gx[1]]); xdir_p /= np.linalg.norm(xdir_p)
        tgt_above = np.r_[xy_t, z_rel + APPROACH_ABOVE]; tgt_rel = np.r_[xy_t, z_rel]
        # the box is symmetric under a 180 deg turn: try both wrist directions, keep the one that reaches
        # the release pose best (position first, then jaw tilt)
        best = None
        for sgn in (1, -1):
            xd = sgn * xdir_p
            q_r, e_r, t_r = sim.ik(b_in_g, tgt_rel, xd, [q] + self.seeds(tgt_rel), w_orient=100.0, signed=True)
            q_a, e_a, t_a = sim.ik(b_in_g, tgt_above, xd, [q_r] + self.seeds(tgt_above), w_orient=100.0, signed=True)
            if e_a > 3:
                q_a, e_a, t_a = sim.ik(b_in_g, tgt_above, xd, [q_r] + self.seeds(tgt_above), w_orient=10.0, signed=True)
            score = max(e_r, e_a) + 0.2 * max(t_r, t_a)
            if best is None or score < best[0]:
                best = (score, xd, q_a, e_a, t_a, e_r, t_r)
        _, xdir_p, q_above, e3, t3, e_rel, t_rel = best
        if max(e3, e_rel) > 3 or t3 > 30:
            return dict(ok=False, reason="ik_place")
        # carry: first rise to a safe height above both the pick and the stack, then over
        safe = max(tgt_above[2], d.xpos[bid][2]) + 0.02
        q_up, _, _ = sim.ik(b_in_g, np.r_[d.xpos[bid][:2], safe], xdir, [q], w_orient=10.0, signed=True)
        sim.move(q_up, g_close, 0.6)
        q_over, _, _ = sim.ik(b_in_g, np.r_[xy_t, safe], xdir_p, [q_above], w_orient=10.0, signed=True)
        sim.move(q_over, g_close, 1.0)
        sim.move(q_above, g_close, 0.6)
        # straight down to the release height, slow near the top
        q = self.straight(b_in_g, xy_t, tgt_above[2], z_rel, xdir_p, q_above, g_close, step=0.003, speed_time=0.1, slow=0.25)
        sim.wait(0.2)
        pre = {s_.body: d.xpos[m.body(s_.body).id].copy() for s_ in st.stack}
        # gentle release, then straight up
        sim.move(q, g_open, 0.6); sim.wait(0.3)
        q = self.straight(b_in_g, xy_t, z_rel, z_rel + RETRACT, xdir_p, q, g_open)
        sim.on_tick = None
        # judge 3 s later (not recorded)
        sim.wait(3.0)
        moved = max(np.linalg.norm(d.xpos[m.body(b).id] - p) for b, p in pre.items())
        tilts = [np.degrees(np.arccos(np.clip(d.xmat[m.body(s_.body).id].reshape(3, 3)[2, 2], -1, 1))) for s_ in st.stack]
        nb = d.xpos[bid]; nb_tilt = np.degrees(np.arccos(np.clip(d.xmat[bid].reshape(3, 3)[2, 2], -1, 1)))
        on_top = abs(nb[2] - (top_c[2] + ht / 2 + dim["height"] / 2)) < 0.004 and np.linalg.norm(nb[:2] - xy_t) < 0.015
        ok = on_top and moved < 0.005 and max(tilts) < 5 and nb_tilt < 5
        reason = "ok" if ok else ("knocked_stack" if (moved >= 0.005 or max(tilts) >= 5) else "misplaced")
        return dict(ok=bool(ok), reason=reason, stack_moved_mm=float(moved * 1000), new_box_tilt=float(nb_tilt),
                    worst_ik_err_mm=float(self.worst_err), release_tilt_deg=float(t_rel),
                    place_offset_mm=float(np.linalg.norm(nb[:2] - xy_t) * 1000), support_margin_mm=float(margin * 1000))


def home_fn(sim):
    def f(m, d):
        d.qpos[sim.qadr] = sim.home[:5]; d.qpos[sim.gadr] = sim.home[5]
        for j, v in zip(JOINT_NAMES, sim.home):
            d.ctrl[sim.act[j]] = v
    return f


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sim = GT.Sim(); rng = np.random.default_rng(a.seed); ex = Expert(sim, rng)
    res = []
    for i in range(a.n):
        st, _ = G.sample_state(rng, sim.m, sim.d, home=home_fn(sim))
        r = ex.run(st); r.update(height=st.meta["stack_height"], pairing=st.meta["pairing"]); res.append(r)
        print(f"{i:3d} h={r['height']} {r['pairing']:14s} -> {r['reason']:14s}" +
              (f" moved {r['stack_moved_mm']:.1f} mm, offset {r['place_offset_mm']:.1f} mm, margin {r['support_margin_mm']:.1f} mm,"
               f" worst IK {r['worst_ik_err_mm']:.1f} mm, release tilt {r['release_tilt_deg']:.0f} deg" if 'stack_moved_mm' in r else ""))
    ok = sum(r["ok"] for r in res); print(f"success {ok}/{a.n}")


if __name__ == "__main__":
    main()
