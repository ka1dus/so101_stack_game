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
DEBUG_GAP = False
BACKDROP_CLEAR = 0.010         # wrist/gripper/camera to backdrop at the place poses (real backdrop position ~±5 mm)
PROXIMAL_CLEAR = 0.003         # shoulder/upper/lower arm to backdrop (fixed by layout; real hover checks: no contact)
ROLL_REF = np.radians(-12.0)   # sim wrist_roll at the place pose (backdrop-safe direction; pilot: -45..+20 deg)
SLOW_ZONE = 0.02               # last 2 cm of the place descent at reduced speed
# ---- data v2 grasp (notes/phase6_data_v2_plan.md section 1) ----
GAP_PER_UNIT = 0.0010          # jaw opening per gripper unit near the box widths (measured in sim, 2026-10-22)
JAW_T_FIXED, JAW_T_MOVING = 0.0062, 0.0068   # jaw tip thickness (notes/phase4_finger_geometry.txt)
MARGIN_MIN, MARGIN_MAX = 0.002, 0.008        # per-side gap between jaw and box at the pre-grasp opening
HOVER = 0.03                   # stop this far above the box top (grip point) before the final descent
HOVER_T, PRECLOSE_T = 0.3, 0.2
RISE_MIN_R = 0.19              # carry: rise to the safe height no closer than this to the pan axis
NOISE_OFF_ABOVE = 0.025        # DART noise fades out once the grip point is this close to the box top (tips ~15 mm above)
GRIP_DEPTH = "deep"            # "v1": jaw tips 10 mm below the box top; "deep" (v2 default): tips planned ~2.5 mm above the mat
FIXED_MARGIN_MAX = MARGIN_MAX  # cap on the fixed-jaw-side margin (the moving jaw pushes the box this far on closing)


class Expert:
    def __init__(self, sim: GT.Sim, rng):
        self.sim, self.rng = sim, rng
        m = sim.m
        self.backdrop = [g for g in range(m.ngeom) if m.geom(g).name.startswith("backdrop")]
        self.arm_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and not m.body(m.geom_bodyid[g]).name.startswith("box")
                          and m.body(m.geom_bodyid[g]).name not in ("base", "base_link")]

    def backdrop_gap(self, q):
        """Backdrop clearance margin (m) at joint config q: min over arm geoms of distance minus the required gap
        (BACKDROP_CLEAR for the wrist/gripper/camera, whose position depends on the wrist choice; PROXIMAL_CLEAR for
        shoulder/upper/lower arm, which sit ~5-10 mm from the backdrop at every place pose by layout). >= 0 is OK."""
        sim = self.sim; k = sim.kin
        k.qpos[sim.qadr] = q; mujoco.mj_forward(sim.m, k)
        out = np.inf
        for a in self.arm_geoms:
            need = PROXIMAL_CLEAR if sim.m.body(sim.m.geom_bodyid[a]).name in ("shoulder", "upper_arm", "lower_arm") else BACKDROP_CLEAR
            out = min(out, min(mujoco.mj_geomDistance(sim.m, k, a, b, 0.05, None) for b in self.backdrop) - need)
        return out

    def lowest_corner(self, q, b_in_g, Rbg, dim):
        """World z of the held box's lowest corner with the arm at q (box pose in the gripper frame from the grasp)."""
        sim = self.sim; k = sim.kin
        k.qpos[sim.qadr] = q; mujoco.mj_kinematics(sim.m, k)
        Rg = k.xmat[sim.gb].reshape(3, 3); xg = k.xpos[sim.gb]
        L, W, H = dim["length"] / 2, dim["width"] / 2, dim["height"] / 2
        return min((xg + Rg @ (b_in_g + Rbg @ np.array([a * L, b * W, c * H])))[2] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1))

    def path_clear(self, qa, qb, n=15):
        """Every joint-interpolated pose from qa to qb keeps the backdrop clearances (backdrop_gap >= 0)."""
        return all(self.backdrop_gap(qa + s * (qb - qa)) >= 0 for s in np.linspace(0, 1, n))

    def safe_move(self, q_to, g, T):
        """sim.move, but if the straight joint-space path brings the arm/camera too close to the backdrop, reorder it:
        shoulder_pan first (turn away, then reach), or everything but the pan first. Found 2026-10-23: from home to boxes
        near the arm base the upper arm swung back into the backdrop (12.5 mm away at home) before the base turned."""
        sim = self.sim; q_from = sim.plan_q.copy()
        if self.path_clear(q_from, q_to):
            sim.move(q_to, g, T); return
        for first in ("pan", "rest"):
            q_mid = q_from.copy() if first == "pan" else q_to.copy()
            q_mid[0] = q_to[0] if first == "pan" else q_from[0]
            if self.path_clear(q_from, q_mid) and self.path_clear(q_mid, q_to):
                self.reordered += 1
                sim.move(q_mid, g, 0.6 * T); sim.move(q_to, g, 0.6 * T); return
        self.unsafe_paths += 1
        sim.move(q_to, g, T)

    # ---------------------------------------------------------------- helpers
    def seeds(self, target):
        sim = self.sim
        pan = -np.arctan2(target[1], target[0]); r = np.linalg.norm(target[:2])
        near = GT.VERT[np.argsort((GT.VERT[:, 0] - r) ** 2 + (GT.VERT[:, 1] - target[2]) ** 2)[:3]]
        out = [np.r_[pan, c[3:6], roll] for c in near for roll in (-4.3, -2.7, -1.2, 0.4)]
        out += [np.r_[pan, a, roll] for a in ([0.3, 0.3, 1.0], [0.6, -0.3, 1.2]) for roll in (-4.3, -2.7, -1.2, 0.4)]
        return out

    def ik_pos_first(self, p_grip, target, xdir, seeds, tol=2.0):
        """IK that keeps the jaws as vertical as possible but never at the cost of position: lower the orientation
        weight (100 -> 30 -> 10 -> 3) until the position error is <= tol mm. Returns (q, err_mm, tilt_deg)."""
        best = None
        for w in (100.0, 30.0, 10.0, 3.0):
            q, e, t = self.sim.ik(p_grip, target, xdir, seeds, w_orient=w, signed=True)
            if best is None or e < best[1]:
                best = (q, e, t)
            if e <= tol:
                return q, e, t
            seeds = [q] + list(seeds)
        return best

    def straight(self, p_grip, xy, z_from, z_to, xdir, q_start, g, step=0.005, speed_time=0.1, slow=None):
        """Move the grip point straight up/down through Cartesian waypoints (IK seeded from the last)."""
        sim = self.sim; q_prev = q_start
        zs = np.arange(z_from, z_to, step if z_to > z_from else -step)[1:].tolist() + [z_to]
        for z in zs:
            q, e, t = self.ik_pos_first(p_grip, np.r_[xy, z], xdir, [q_prev])
            if e > 2:
                q, e, t = self.ik_pos_first(p_grip, np.r_[xy, z], xdir, [q_prev] + self.seeds(np.r_[xy, z]))
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
    def grasp_margin(self, box, others, side):
        """Jaw margins (fixed side, moving side) for the v2 grasp from the free space beside the box (privileged).
        Centred when FIXED_MARGIN_MAX = MARGIN_MAX; a smaller cap shortens how far the moving jaw pushes the box."""
        c_mov = G.side_clear_width(box, others, side); c_fix = G.side_clear_width(box, others, -side)
        m_fix = float(np.clip(c_fix - JAW_T_FIXED - 0.001, MARGIN_MIN, FIXED_MARGIN_MAX))
        m_mov = float(np.clip(c_mov - JAW_T_MOVING - 0.001, MARGIN_MIN, MARGIN_MAX))
        if FIXED_MARGIN_MAX >= MARGIN_MAX:                   # centred: equal margins, the smaller of the two
            m_fix = m_mov = min(m_fix, m_mov)
        return m_fix, m_mov

    def grip_point(self, dim, m_fix):
        depth = 0.010 if GRIP_DEPTH == "v1" else min(dim["height"] - 0.0025, 0.014)
        return np.array([self.sim.xin + m_fix + dim["width"] / 2, 0.0, self.sim.ztip + depth])

    def refresh(self, st):
        """Re-read every box pose from the sim into the State (after an unrecorded missed grasp)."""
        m, d = self.sim.m, self.sim.d
        for b in st.stack + st.supply:
            bid = m.body(b.body).id; b.xy = d.xpos[bid][:2].copy(); b.z = float(d.xpos[bid][2])
            R = d.xmat[bid].reshape(3, 3); b.yaw = float(np.arctan2(R[1, 0], R[0, 0]))

    def reading(self):
        return float(self.sim.jm.sim_to_real(np.r_[np.zeros(5), self.sim.d.qpos[self.sim.gadr]])[5])

    def miss(self, st, xdir, p_grip, box, dim, margin):
        """Unrecorded missed grasp for a retry start: grasp with the target offset sideways (past the jaw margin) or
        along the box (past its end), close, lift 2-3 cm. Returns True if the gripper ends up empty."""
        sim, d, rng = self.sim, self.sim.d, self.rng
        bid = sim.m.body(box.body).id; c = d.xpos[bid].copy(); top_z = c[2] + dim["height"] / 2
        R = d.xmat[bid].reshape(3, 3); u, v = R[:2, 0] / np.linalg.norm(R[:2, 0]), R[:2, 1] / np.linalg.norm(R[:2, 1])
        if rng.random() < 0.5:
            off = v * rng.choice([-1, 1]) * (max(margin) + rng.uniform(0.005, 0.012))
        else:
            off = u * rng.choice([-1, 1]) * (dim["length"] / 2 + rng.uniform(0.005, 0.013))
        tgt = c[:2] + off
        q_pre, _, _ = sim.ik(p_grip, np.r_[tgt, top_z + 0.05], xdir, self.seeds(np.r_[tgt, top_z + 0.05]), w_orient=10.0, signed=True)
        g_open = sim.gripper_sim(CONTACT_READING[box.type] + sum(margin) / GAP_PER_UNIT)
        self.safe_move(q_pre, g_open, 1.0)
        q = self.straight(p_grip, tgt, top_z + 0.05, top_z + 0.002, xdir, q_pre, g_open)
        sim.move(q, sim.gripper_sim(0.0), 0.6); sim.wait(0.3)
        self.straight(p_grip, tgt, top_z + 0.002, top_z + rng.uniform(0.02, 0.03), xdir, q, sim.gripper_sim(0.0))
        sim.wait(0.2)
        return self.reading() < CONTACT_READING[box.type] - 4

    def run(self, st, record=None, kind="clean", mode="act", requested=None):
        """kind: "clean" (v2 grasp), "dart" (v2 + execution noise, clean labels), "retry" (recording starts after an
        unrecorded missed grasp), "v1" (the v1 expert: lopsided tight grasp, no hover/slow descent)."""
        sim, m, d, rng = self.sim, self.sim.m, self.sim.d, self.rng
        self.worst_err = 0.0; v2 = kind != "v1"; dart = kind == "dart"; self.reordered = 0; self.unsafe_paths = 0
        sim.noise_rng = np.random.default_rng(rng.integers(1 << 31)); sim._nx[:] = 0; sim._gain = 0.0; sim.set_noise(False)
        # home with jitter
        q0 = sim.home.copy(); q0[:5] += np.radians(rng.uniform(-3, 3, 5))
        q0[:5] = np.clip(q0[:5], sim.rng_lim[:, 0] + np.radians(0.5), sim.rng_lim[:, 1] - np.radians(0.5))   # home elbow sits ~0.5 deg from its limit
        d.qpos[sim.qadr] = q0[:5]; d.qpos[sim.gadr] = q0[5]
        for j, v in zip(JOINT_NAMES, q0):
            d.ctrl[sim.act[j]] = v
        d.qvel[:] = 0; mujoco.mj_forward(m, d); sim.sync_plan()
        self.hit = False; bd = set(self.backdrop); self.recording = kind != "retry"
        def tick():
            for c in d.contact[:d.ncon]:
                if (c.geom1 in bd) != (c.geom2 in bd) and m.geom_bodyid[c.geom2 if c.geom1 in bd else c.geom1] != 0 \
                        and not m.body(m.geom_bodyid[c.geom2 if c.geom1 in bd else c.geom1]).name.startswith("box"):
                    self.hit = True
            if record is not None and self.recording:
                record()
        sim.on_tick = tick
        info = dict(kind=kind)
        if kind == "retry":                                 # unrecorded missed grasp, then re-plan from the current scene
            box = st.supply[st.pick]; dim = G.BOX[box.type]; others = [o for k, o in enumerate(st.supply) if k != st.pick]
            side = (G.jaw_sides(box, others) or [1])[0]
            Rb = d.xmat[m.body(box.body).id].reshape(3, 3); xd = side * Rb[:2, 1] / np.linalg.norm(Rb[:2, 1])
            mf, mm = self.grasp_margin(box, others, side)
            p_g = self.grip_point(dim, mf)
            if not self.miss(st, xd, p_g, box, dim, (mf, mm)):
                return dict(ok=False, reason="retry_not_missed")
            if self.hit:
                return dict(ok=False, reason="hit_backdrop")
            self.refresh(st)
            st.pick = G.choose_pick(st, mode, requested)
            if st.pick < 0:
                return dict(ok=False, reason="retry_no_pick")
            info["retry_reading"] = self.reading()
            self.recording = True
        box = st.supply[st.pick]; dim = G.BOX[box.type]; bid = m.body(box.body).id
        centre = d.xpos[bid].copy(); Rb = d.xmat[bid].reshape(3, 3); top_z = centre[2] + dim["height"] / 2
        # moving-jaw side: of the feasible sides, the one whose wrist roll is nearest the place roll. Keeps the pick roll
        # within +-90 deg of it (no +-180 deg wrap: similar-looking scenes get similar motions, ACT multimodality) and
        # the carry rotation short.
        others = [o for k, o in enumerate(st.supply) if k != st.pick]
        sides = G.jaw_sides(box, others) or [st.meta.get("moving_jaw_side", 1) or 1]
        p_grip = np.array([sim.xin + dim["width"] / 2 + 0.002, 0.0, sim.ztip + 0.010])     # v1: fixed jaw 2 mm off the box
        cands = []
        for side in sides:
            xd = side * Rb[:2, 1] / np.linalg.norm(Rb[:2, 1])        # gripper +x -> moving jaw side
            mg = self.grasp_margin(box, others, side) if v2 else None
            pg = self.grip_point(dim, mg[0]) if v2 else p_grip    # feasibility with the grip point actually used
            q_c, e_c, t_c = sim.ik(pg, np.r_[centre[:2], top_z], xd, self.seeds(np.r_[centre[:2], top_z]), signed=True)
            if e_c <= 3 and t_c <= 10:
                cands.append((abs(q_c[4] - ROLL_REF), side, xd, q_c, e_c, t_c, mg, pg))
        if not cands:
            return dict(ok=False, reason="ik_pick", **info)
        _, side, xdir, q_g, e, tilt, mg, pg = min(cands, key=lambda c: c[0])
        self.pick_side = side
        g_close = sim.gripper_sim(0.0)
        if v2:                                              # v2 grasp, margins from the free space beside the box
            p_grip = pg; info["grasp_margin_mm"] = [round(mg[0] * 1000, 1), round(mg[1] * 1000, 1)]
            g_open = sim.gripper_sim(CONTACT_READING[box.type] + (mg[0] + mg[1]) / GAP_PER_UNIT)
        else:
            g_open = sim.gripper_sim(PREGRASP[box.type])
        # pre-grasp above, then straight down
        q_pre, e2, _ = sim.ik(p_grip, np.r_[centre[:2], top_z + 0.05], xdir, [q_g] + self.seeds(np.r_[centre[:2], top_z + 0.05]),
                              w_orient=10.0, signed=True)
        sim.set_noise(dart)
        self.safe_move(q_pre, g_open, 1.0)
        if v2:
            q = self.straight(p_grip, centre[:2], top_z + 0.05, top_z + HOVER, xdir, q_pre, g_open)
            sim.wait(HOVER_T)                               # hover: the wrist camera looks straight down at the box
            zs = [top_z + HOVER, top_z + NOISE_OFF_ABOVE]
            q = self.straight(p_grip, centre[:2], zs[0], zs[1], xdir, q, g_open, step=0.003, speed_time=0.2)
            sim.set_noise(False)                            # clean final 2.5 cm (fade 0.3 s)
            q = self.straight(p_grip, centre[:2], zs[1], top_z, xdir, q, g_open, step=0.003, speed_time=0.2)
            sim.wait(PRECLOSE_T)
        else:
            q = self.straight(p_grip, centre[:2], top_z + 0.05, top_z, xdir, q_pre, g_open)
        sim.move(q, g_close, 0.6); sim.wait(0.3)
        if v2:                                              # lift 2 cm, then check the grasp like the real skill will
            q = self.straight(p_grip, centre[:2], top_z, top_z + 0.02, xdir, q, g_close)
            info["grasp_reading"] = round(self.reading(), 1)
            if self.reading() < CONTACT_READING[box.type] - 4:
                return dict(ok=False, reason="missed_grasp", **info)
            q = self.straight(p_grip, centre[:2], top_z + 0.02, top_z + LIFT, xdir, q, g_close)
        else:
            q = self.straight(p_grip, centre[:2], top_z, top_z + LIFT, xdir, q, g_close)
        # held-box pose in the gripper frame (privileged)
        Rg = d.xmat[sim.gb].reshape(3, 3); xg = d.xpos[sim.gb]
        b_in_g = Rg.T @ (d.xpos[bid] - xg)
        if d.xpos[bid][2] < top_z + LIFT * 0.6 - dim["height"]:
            return dict(ok=False, reason="missed_grasp", **info)
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
            q_r, e_r, t_r = self.ik_pos_first(b_in_g, tgt_rel, xd, [q] + self.seeds(tgt_rel))
            q_a, e_a, t_a = self.ik_pos_first(b_in_g, tgt_above, xd, [q_r] + self.seeds(tgt_above))
            gap = min(self.backdrop_gap(q_r), self.backdrop_gap(q_a))
            if DEBUG_GAP:
                k = sim.kin
                for qq, nm in ((q_r, "rel"), (q_a, "above")):
                    k.qpos[sim.qadr] = qq; mujoco.mj_forward(m, k)
                    gs = sorted((mujoco.mj_geomDistance(m, k, g, self.backdrop[0], 0.05, None), m.geom(g).name or f"#{g}",
                                 m.body(m.geom_bodyid[g]).name, int(m.geom_contype[g])) for g in self.arm_geoms)[:3]
                    print(f"    sgn {sgn:+d} {nm}: roll {np.degrees(qq[4]):.0f} ", [(round(x*1000, 1), n, b_, c) for x, n, b_, c in gs])
            score = max(e_r, e_a) + 0.2 * max(t_r, t_a) + (1000 if gap < 0 else 0)
            if best is None or score < best[0]:
                best = (score, xd, q_a, e_a, t_a, e_r, t_r)
        score, xdir_p, q_above, e3, t3, e_rel, t_rel = best
        if score >= 1000:
            return dict(ok=False, reason="backdrop")
        if max(e3, e_rel) > 3 or t3 > 30:
            print(f"      ik_place: release err {e_rel:.1f} mm tilt {t_rel:.0f} | above err {e3:.1f} mm tilt {t3:.0f} | "
                  f"target r {np.linalg.norm(xy_t)*100:.1f} cm z {z_rel*100:.1f} cm")
            return dict(ok=False, reason="ik_place")
        # a tilted release (wrist_flex limit on tall stacks) puts the held box's low end below its centre: raise the
        # release until the LOWEST CORNER, not the centre, clears the top face by RELEASE_CLEARANCE (2026-10-23 fix:
        # a No.3 at 10 deg tilt pressed ~3 mm into the top box and knocked 7-high stacks)
        if v2:
            top_face = top_c[2] + ht / 2; info["release_raise_mm"] = 0.0
            for _ in range(3):
                q_r2, _, _ = self.ik_pos_first(b_in_g, np.r_[xy_t, z_rel], xdir_p, [q_above] + self.seeds(np.r_[xy_t, z_rel]))
                deficit = top_face + RELEASE_CLEARANCE - self.lowest_corner(q_r2, b_in_g, Rbg, dim)
                if deficit < 0.0003:
                    break
                z_rel += deficit; info["release_raise_mm"] += round(deficit * 1000, 1)
            tgt_above = np.r_[xy_t, z_rel + APPROACH_ABOVE]
            q_above, e3, t3 = self.ik_pos_first(b_in_g, tgt_above, xdir_p, [q_above] + self.seeds(tgt_above))
        # carry: first rise to a safe height above both the pick and the stack, then over
        safe = max(tgt_above[2], d.xpos[bid][2]) + 0.02
        xy_up = d.xpos[bid][:2].copy(); r_up = np.linalg.norm(xy_up)
        if r_up < RISE_MIN_R:                               # near the base: rise further out, or the folded arm's upper arm
            xy_up *= RISE_MIN_R / r_up                      # swings back into the backdrop (2026-10-23 diagnosis)
        q_up, _, _ = sim.ik(b_in_g, np.r_[xy_up, safe], xdir, [q] + self.seeds(np.r_[xy_up, safe]), w_orient=10.0, signed=True)
        sim.set_noise(dart)                                 # noise during the carry, away from boxes
        self.safe_move(q_up, g_close, 0.6)
        q_over, _, _ = sim.ik(b_in_g, np.r_[xy_t, safe], xdir_p, [q_above], w_orient=10.0, signed=True)
        self.safe_move(q_over, g_close, 1.0)
        sim.set_noise(False)                                # clean from here: approach, descent, release, retract
        sim.wait(0.3 if dart else 0.0)                      # let the noise fade out above the stack
        self.safe_move(q_above, g_close, 0.6)
        # straight down to the release height, slow near the top
        q = self.straight(b_in_g, xy_t, tgt_above[2], z_rel, xdir_p, q_above, g_close, step=0.003, speed_time=0.1, slow=0.25)
        sim.wait(0.2)
        Rg2 = d.xmat[sim.gb].reshape(3, 3)
        self.dbg = dict(grasp_drift_mm=float(np.linalg.norm(Rg2.T @ (d.xpos[bid] - d.xpos[sim.gb]) - b_in_g) * 1000),
                        before_release=(d.xpos[bid][:2] - xy_t) * 1000,
                        track_err_deg=np.degrees(d.qpos[sim.qadr] - q),
                        contacts=sorted({f"{m.geom(c.geom1).name}|{m.geom(c.geom2).name}" for c in d.contact[:d.ncon] if "wrist_camera" in m.body(m.geom_bodyid[c.geom2]).name or "wrist_camera" in m.body(m.geom_bodyid[c.geom1]).name}),
                        q_cmd_deg=np.degrees(q), jnt_range_deg=np.degrees(m.jnt_range[[m.joint(j).id for j in JOINT_NAMES[:5]]]),
                        ctrl_minus_q_deg=np.degrees(np.array([d.ctrl[sim.act[j]] for j in JOINT_NAMES[:5]]) - q))
        pre = {s_.body: d.xpos[m.body(s_.body).id].copy() for s_ in st.stack}
        # gentle release, then straight up
        sim.move(q, g_open, 0.6); sim.wait(0.3)
        self.dbg["after_open"] = (d.xpos[bid][:2] - xy_t) * 1000
        q = self.straight(b_in_g, xy_t, z_rel, z_rel + RETRACT, xdir_p, q, g_open)
        self.dbg["after_retract"] = (d.xpos[bid][:2] - xy_t) * 1000
        sim.on_tick = None
        # judge 3 s later (not recorded)
        sim.wait(3.0)
        moved = max(np.linalg.norm(d.xpos[m.body(b).id] - p) for b, p in pre.items())
        tilts = [np.degrees(np.arccos(np.clip(d.xmat[m.body(s_.body).id].reshape(3, 3)[2, 2], -1, 1))) for s_ in st.stack]
        nb = d.xpos[bid]; nb_tilt = np.degrees(np.arccos(np.clip(d.xmat[bid].reshape(3, 3)[2, 2], -1, 1)))
        on_top = abs(nb[2] - (top_c[2] + ht / 2 + dim["height"] / 2)) < 0.004 and np.linalg.norm(nb[:2] - xy_t) < 0.015
        ok = on_top and moved < 0.005 and max(tilts) < 5 and nb_tilt < 5
        ok = ok and not self.hit
        reason = "ok" if ok else "hit_backdrop" if self.hit else ("knocked_stack" if (moved >= 0.005 or max(tilts) >= 5) else "misplaced")
        info.update(reordered_moves=self.reordered, unsafe_paths=self.unsafe_paths)
        return dict(ok=bool(ok), reason=reason, **info, stack_moved_mm=float(moved * 1000), new_box_tilt=float(nb_tilt),
                    worst_ik_err_mm=float(self.worst_err), moving_jaw_side=int(self.pick_side), release_tilt_deg=float(t_rel),
                    place_offset_mm=float(np.linalg.norm(nb[:2] - xy_t) * 1000), support_margin_mm=float(margin * 1000))


def home_fn(sim):
    def f(m, d):
        d.qpos[sim.qadr] = sim.home[:5]; d.qpos[sim.gadr] = sim.home[5]
        for j, v in zip(JOINT_NAMES, sim.home):
            d.ctrl[sim.act[j]] = v
    return f


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", default="clean", choices=("clean", "dart", "retry", "v1"))
    ap.add_argument("--v2-states", action="store_true", help="v2 state distribution (dense supplies) + strict rest check")
    ap.add_argument("--game-states", action="store_true", help="full-v2 states: game-rule supplies + fixed layout (SUPPLY_P_GAME)")
    ap.add_argument("--height-min", type=int, default=None); ap.add_argument("--height-max", type=int, default=None)
    ap.add_argument("--hard-p", type=float, default=None); ap.add_argument("--min-crooked-mm", type=float, default=0.0)
    ap.add_argument("--want-types", default=None, help="comma list: only states where the rule picks one of these types")
    ap.add_argument("--grip-depth", default="deep", choices=("v1", "deep")); ap.add_argument("--fixed-margin-max", type=float, default=None)
    a = ap.parse_args()
    global GRIP_DEPTH, FIXED_MARGIN_MAX
    GRIP_DEPTH = a.grip_depth
    if a.fixed_margin_max is not None:
        FIXED_MARGIN_MAX = a.fixed_margin_max / 1000
    sim = GT.Sim(); rng = np.random.default_rng(a.seed); ex = Expert(sim, rng)
    res = []
    for i in range(a.n):
        kw = dict(height_p=G.HEIGHT_P_V2, rest=2.0, hard_p=0.6) if (a.v2_states or a.game_states) else {}
        if a.game_states:
            kw["supply_p"] = G.SUPPLY_P_GAME
        if a.height_min is not None:
            kw["height_p"] = {h: 1.0 for h in range(a.height_min, (a.height_max or G.MAX_STACK) + 1)}
        if a.hard_p is not None:
            kw["hard_p"] = a.hard_p
        for _ in range(3000):
            st, _ = G.sample_state(rng, sim.m, sim.d, home=home_fn(sim), **kw)
            off = max((np.linalg.norm(o) for o in st.meta.get("offsets_mm", [])), default=0.0)
            if off >= a.min_crooked_mm and (not a.want_types or st.supply[st.pick].type in a.want_types.split(",")):
                break
        r = ex.run(st, kind=a.kind); r.update(height=st.meta["stack_height"], pairing=st.meta["pairing"]); res.append(r)
        print(f"{i:3d} h={r['height']} {r['pairing']:14s} -> {r['reason']:16s} margin/side {r.get('grasp_margin_mm', '-')} mm," +
              (f" moved {r['stack_moved_mm']:.1f} mm, offset {r['place_offset_mm']:.1f} mm, margin {r['support_margin_mm']:.1f} mm,"
               f" worst IK {r['worst_ik_err_mm']:.1f} mm, release tilt {r['release_tilt_deg']:.0f} deg" if 'stack_moved_mm' in r else ""))
        if not r["ok"] and getattr(ex, "dbg", None) and r["reason"] == "misplaced":
            print("      ", {k: (np.round(v, 1) if not isinstance(v, list) else v) for k, v in ex.dbg.items()})
        ex.dbg = None
    ok = sum(r["ok"] for r in res); print(f"success {ok}/{a.n}")
    from collections import Counter, defaultdict
    print("failures:", dict(Counter(r["reason"] for r in res if not r["ok"])))
    for key, f in (("height", lambda r: r["height"]), ("picked type", lambda r: r["pairing"].split("_on_")[0]),
                   ("pairing", lambda r: r["pairing"])):
        b_ = defaultdict(list)
        for r in res:
            b_[f(r)].append(r["ok"])
        print(f"by {key}: " + ", ".join(f"{k} {sum(v)}/{len(v)}" for k, v in sorted(b_.items())))


if __name__ == "__main__":
    main()
