"""Phase 5: where can the black supply mat go so 7 boxes fit and every box is pickable?

Step 1 (--map): pickability map on a 1.5 cm grid over the table's supply side. A point is
pickable if a No.3 resting there can be grasped with vertical jaws (IK error <= 2 mm, tilt
<= 5 deg) at ALL of 4 box yaws (0/45/90/135 deg) without any arm link touching the table,
mats or backdrop. Saved to notes/phase5_pick_map.npz.
Step 2 (default): slide the mat along the table's left edge (same heading, flush or with an
offset from the edge) and score each pose: mat on the table, >= 2 cm from the yellow area
and from the arm base, and the share of box-centre positions (>= 2.5 cm inside the mat)
that are pickable.

    MUJOCO_GL=cgl ~/.venvs/so101_stack_sim/bin/python scripts/supply_placement.py --map
    MUJOCO_GL=cgl ~/.venvs/so101_stack_sim/bin/python scripts/supply_placement.py
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import grasp_test as GT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LAY = yaml.safe_load((ROOT / "sim/assets/layout.yaml").read_text())
MAP = ROOT / "notes/phase5_pick_map.npz"
T = LAY["table"]; TC = np.array(T["centre"]); HD = np.radians(T["heading_deg"])
TU, TV = np.array([np.cos(HD), np.sin(HD)]), np.array([-np.sin(HD), np.cos(HD)])
S0 = np.array(LAY["supply_area"]["corners"]); K = np.array(LAY["stack_area"]["corners"])


def pickable(sim, p, top=0.0005 + 0.017):
    pg = np.array([sim.xin + 0.020 + 0.002, 0, sim.ztip + 0.010])     # No.3, box side 2 mm from the fixed jaw
    pan = -np.arctan2(p[1], p[0]); r = np.linalg.norm(p)
    near = GT.VERT[np.argsort((GT.VERT[:, 0] - r) ** 2 + (GT.VERT[:, 1] - top) ** 2)[:3]]
    seeds = [np.r_[pan, c[3:6], roll] for c in near for roll in (-4.3, -2.7, -1.2, 0.4)]
    seeds += [np.r_[pan, a, roll] for a in ([0.3, 0.3, 1.0], [0.6, -0.3, 1.2], [-0.5, 1.2, 0.6]) for roll in (-2.7, 0.4)]
    for yaw in np.radians([0, 45, 90, 135]):
        wd = np.array([-np.sin(yaw), np.cos(yaw)])
        q, e, t = sim.ik(pg, np.r_[p, top], wd, seeds)
        if e > 2 or t > 5:
            return False
        sim.kin.qpos[sim.qadr] = q; mujoco.mj_forward(sim.m, sim.kin)
        for i in range(sim.kin.ncon):
            b1, b2 = sim.m.geom_bodyid[sim.kin.contact[i].geom1], sim.m.geom_bodyid[sim.kin.contact[i].geom2]
            if (b1 == 0) != (b2 == 0) and not any(sim.m.body(b).name.startswith(("box", "wrist_camera")) for b in (b1, b2)):
                return False
    return True


def build_map():
    sim = GT.Sim()
    for b in range(sim.m.nbody):                         # park the boxes so they don't count as obstacles
        if sim.m.body(b).name.startswith("box"):
            adr = sim.m.jnt_qposadr[sim.m.body_jntadr[b]]; sim.kin.qpos[adr:adr + 3] = [5 + b, 5, -5]
    pts, ok = [], []
    for a in np.arange(-0.19, 0.19 + 1e-9, 0.015):
        for b in np.arange(-0.19, 0.19 + 1e-9, 0.015):
            p = TC + a * TU + b * TV
            if p[1] < -0.02 or np.linalg.norm(p) < 0.06:     # supply side only (arm's left), not on the base
                continue
            pts.append(p); ok.append(pickable(sim, p))
            print(f"\r{len(pts)} points, {sum(ok)} pickable", end="", flush=True)
    np.savez(MAP, pts=np.array(pts), ok=np.array(ok))
    print("\nsaved", MAP)


def score(S, pts, ok):
    """Fraction of interior box-centre positions near a pickable grid point (nearest-neighbour lookup)."""
    e1, e2 = S[1] - S[0], S[3] - S[0]
    L1, L2 = np.linalg.norm(e1), np.linalg.norm(e2)
    g1 = np.linspace(0.025 / L1, 1 - 0.025 / L1, 15); g2 = np.linspace(0.025 / L2, 1 - 0.025 / L2, 15)
    C = np.array([S[0] + a * e1 + b * e2 for a in g1 for b in g2])
    idx = np.argmin(((C[:, None, :] - pts[None]) ** 2).sum(-1), axis=1)
    near = np.sqrt(((C - pts[idx]) ** 2).sum(-1)) < 0.012
    return float((ok[idx] & near).mean())


def main():
    if "--map" in sys.argv:
        build_map(); return
    z = np.load(MAP); pts, ok = z["pts"], z["ok"]
    base_r = 0.075                                      # arm base footprint + margin around the pan axis (APPROX)
    left_dir = (np.array(T["corners"]["left"]) - np.array(T["corners"]["mount"])); left_dir /= np.linalg.norm(left_dir)
    inward = TU if TU @ (TC - np.array(T["corners"]["left"])) > 0 else -TU
    if abs(inward @ left_dir) > 0.5:
        inward = TV if TV @ (TC - np.array(T["corners"]["left"])) > 0 else -TV
    rows = []
    for slide in np.arange(0.0, 0.12, 0.005):          # toward the arm's corner along the left edge
        for off in np.arange(0.0, 0.06, 0.005):        # away from the left edge, into the table
            S = S0 - left_dir * slide + inward * off
            on_table = all(max(abs((p - TC) @ TU), abs((p - TC) @ TV)) <= 0.19 + 1e-6 for p in S)
            # distance from the yellow area (polygon-polygon, sampled)
            edge = lambda P: np.array([P[i] + t * (P[(i + 1) % 4] - P[i]) for i in range(4) for t in np.linspace(0, 1, 30)])
            d_yel = np.min(np.linalg.norm(edge(S)[:, None] - edge(K)[None], axis=-1))
            d_base = np.min(np.linalg.norm(edge(S), axis=1))
            if not on_table or d_yel < 0.02 or d_base < base_r:
                continue
            rows.append((score(S, pts, ok), slide, off, d_yel, d_base, S))
    rows.sort(key=lambda r: (-r[0], r[1] + r[2]))
    print("best mat positions (slide toward the arm along the left edge, offset from the edge):")
    for sc, sl, of, dy, db, S in rows[:8]:
        print(f"  slide {sl*100:4.1f} cm, offset {of*100:3.1f} cm: pickable {sc*100:5.1f} % | {dy*100:.1f} cm from yellow, "
              f"{db*100:.1f} cm from the pan axis | corner r = {np.round(np.linalg.norm(S, axis=1)*100,1)}")
    print(f"current mat: pickable {score(S0, pts, ok)*100:.1f} %")
    best = rows[0]
    np.save(ROOT / "notes/phase5_best_mat.npy", best[5])


if __name__ == "__main__":
    main()
