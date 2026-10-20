"""Fit SO-101 joint offsets + base tilt from the Phase 1 multi-pose touches (M3).

Model (notes/phase1.md section 4, notes/phase1_measurement_guide.md):
  q_sim = q_lerobot - offset   (shoulder_lift, elbow_flex, wrist_flex; wrist_roll from a
                                discrete set or free; shoulder_pan fixed at 0 = defines +x)
  touch point = lowest vertex of the last 9 mm of either closed jaw, after rotating the arm base by tilt (x, y)
Unknowns: offsets, tilt_x, tilt_y, table height, the xy of marks A/B/C, height of B
(it sits on the black mat + tape) relative to A/C.
Data: inputs/multipose_points.json, ruler distances and M2 level prior from
notes/phase1_measurements.yaml. M1 (home tip height) and the 13 old table touches are
held out and used as checks.

    ~/.venvs/so101_stack_sim/bin/python scripts/fit_joint_offsets.py
"""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
from joint_mapping import BODY_JOINTS, JOINT_NAMES, JointMapping, from_dict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = mujoco.MjModel.from_xml_path(str(ROOT / "sim/assets/so101/so101_new_calib.xml"))
DATA = mujoco.MjData(MODEL)
ADR = [MODEL.jnt_qposadr[MODEL.joint(j).id] for j in JOINT_NAMES]
JAWS = [g for g in range(MODEL.ngeom) if MODEL.geom_group[g] == 2
        and MODEL.mesh(MODEL.geom_dataid[g]).name in ("wrist_roll_follower_so101_v1", "moving_jaw_so101_v1")]
mujoco.mj_forward(MODEL, DATA)
PAN = np.r_[DATA.xanchor[MODEL.joint("shoulder_pan").id][:2], 0.0]
TIP_REGION_M = 0.009   # contact = lowest vertex within 9 mm of either jaw's end (you touch with the tips)


def _tip_vertex_mask(g):
    """Vertices of geom g that lie in the last TIP_REGION_M of the jaw (measured in the body frame)."""
    mid = MODEL.geom_dataid[g]
    v = MODEL.mesh_vert[MODEL.mesh_vertadr[mid]:MODEL.mesh_vertadr[mid] + MODEL.mesh_vertnum[mid]]
    loc = MODEL.geom_pos[g] + v @ _quat2mat(MODEL.geom_quat[g]).T      # in the owning body's frame
    body = MODEL.body(MODEL.geom_bodyid[g]).name
    if body == "gripper":          # fixed jaw points along body -z
        return loc[:, 2] < loc[:, 2].min() + TIP_REGION_M
    r = np.linalg.norm(loc[:, :2], axis=1)   # moving jaw: distance from its hinge axis (body z)
    return r > r.max() - TIP_REGION_M


def _quat2mat(q):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, q); return R.reshape(3, 3)
JM = JointMapping.from_files()

TIP_MASK = {g: _tip_vertex_mask(g) for g in JAWS}
MEAS = yaml.safe_load((ROOT / "notes/phase1_measurements.yaml").read_text())
TOUCH = json.loads((ROOT / "inputs/multipose_points.json").read_text())["points"]
OLD = json.loads((ROOT / "inputs/table_points_20261019_181950.json").read_text())["points"]
HOME = dict(shoulder_pan=-3.16, shoulder_lift=-87.30, elbow_flex=96.79, wrist_flex=48.35, wrist_roll=8.66, gripper=0.74)
FREE = ["shoulder_lift", "elbow_flex", "wrist_flex"]

SIG_XY, SIG_Z, SIG_DIST, SIG_LEVEL, SIG_BZ = 2e-3, 1.5e-3, 2e-3, 0.5, 2e-3   # m, m, m, deg, m


def rot(tx, ty):
    rx, ry = np.radians(tx), np.radians(ty)
    return (np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
            @ np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]))


def touch_point(joints: dict, offsets: dict, tilt) -> np.ndarray:
    JM.offset_deg = {j: offsets.get(j, 0.0) for j in BODY_JOINTS}
    for a, v in zip(ADR, JM.real_to_sim(from_dict(joints))):
        DATA.qpos[a] = v
    mujoco.mj_forward(MODEL, DATA)
    V = []
    for g in JAWS:
        mid = MODEL.geom_dataid[g]
        v = MODEL.mesh_vert[MODEL.mesh_vertadr[mid]:MODEL.mesh_vertadr[mid] + MODEL.mesh_vertnum[mid]]
        V.append((DATA.geom_xpos[g] + v @ DATA.geom_xmat[g].reshape(3, 3).T)[TIP_MASK[g]])
    V = (np.vstack(V) - PAN) @ rot(*tilt).T
    return V[np.argmin(V[:, 2])]


def mark_of(label):
    return label.split("_")[1]                     # mark_A_steep -> A


def unpack(x, free_roll):
    k = len(FREE)
    offs = dict(zip(FREE, x[:k]))
    i = k
    if free_roll:
        offs["wrist_roll"] = x[i]; i += 1
    tilt = x[i:i + 2]; zt = x[i + 2]; bz = x[i + 3]
    marks = {m: x[i + 4 + 2 * n:i + 6 + 2 * n] for n, m in enumerate("ABC")}
    return offs, tilt, zt, bz, marks


def residuals(x, roll, free_roll):
    offs, tilt, zt, bz, marks = unpack(x, free_roll)
    if not free_roll:
        offs["wrist_roll"] = roll
    r = []
    for p in TOUCH:
        m = mark_of(p["label"])
        t = touch_point(p["joints"], offs, tilt)
        r += list((t[:2] - marks[m]) / SIG_XY)
        r.append((t[2] - zt - (bz if m == "B" else 0.0)) / SIG_Z)
    for pair, d in MEAS["M3_mark_distances_m"].items():
        a, b = pair.split("_")
        r.append((np.linalg.norm(marks[a] - marks[b]) - d) / SIG_DIST)
    r.append(tilt[1] / SIG_LEVEL)                  # M2: front-back within ~0.5 deg (whole-degree phone)
    r.append(bz / SIG_BZ)                          # mat + tape under B: 0 +/- 2 mm (thickness not given yet)
    return np.array(r)


def fit(roll=0.0, free_roll=False):
    x0 = np.r_[np.zeros(len(FREE)), [roll] if free_roll else [], 0, 0, 0, 0,
               [0.196, -0.101, 0.246, 0.190, 0.150, 0.031]]
    sol = least_squares(residuals, x0, args=(roll, free_roll), diff_step=1e-4)
    J = sol.jac
    dof = max(1, len(sol.fun) - len(sol.x))
    s2 = max(1.0, 2 * sol.cost / dof)              # inflate if residuals exceed the assumed noise
    sd = np.sqrt(np.diag(np.linalg.pinv(J.T @ J) * s2))
    return sol, sd


def checks(offs, tilt, zt):
    home_h = touch_point(HOME, offs, tilt)[2] - zt
    P = {p["label"]: touch_point(p["joints"], offs, tilt) for p in OLD}
    z = np.array([v[2] for v in P.values()]) - zt
    def side(a, b): return np.linalg.norm(P[a][:2] - P[b][:2]) * 100
    sides = {k: round(side(*v), 1) for k, v in dict(
        sup12=("supply_corner_1", "supply_corner_2"), sup23=("supply_corner_2", "supply_corner_3"),
        sup34=("supply_corner_3", "supply_corner_4"), sup41=("supply_corner_4", "supply_corner_1"),
        stk12=("stack_corner_1", "stack_corner_2"), stk23=("stack_corner_2", "stack_corner_3"),
        stk34=("stack_corner_3", "stack_corner_4"), stk41=("stack_corner_4", "stack_corner_1")).items()}
    u = P["table_left_edge_far"][:2] - P["table_left_edge_near"][:2]
    v = P["table_right_edge_far"][:2] - P["table_right_edge_near"][:2]
    ang = np.degrees(np.arccos(abs(u @ v) / np.linalg.norm(u) / np.linalg.norm(v)))
    return home_h, z, sides, ang


def report(name, sol, sd, roll, free_roll):
    offs, tilt, zt, bz, marks = unpack(sol.x, free_roll)
    if not free_roll:
        offs["wrist_roll"] = roll
    names = FREE + (["wrist_roll"] if free_roll else []) + ["tilt_x", "tilt_y"]
    vals = list(sol.x[:len(names)])
    r = residuals(sol.x, roll, free_roll)
    n = len(TOUCH)
    rxy = r[:3 * n].reshape(n, 3)[:, :2] * SIG_XY * 1e3
    rz = r[:3 * n].reshape(n, 3)[:, 2] * SIG_Z * 1e3
    print(f"\n=== {name}: cost {sol.cost:.1f}")
    print("   " + "  ".join(f"{k} {v:+.2f}+/-{s:.2f}" for k, v, s in zip(names, vals, sd)))
    print(f"   B height above A/C plane {bz*1e3:+.1f} mm; marks (cm): "
          + ", ".join(f"{m} {np.round(v*100,1)}" for m, v in marks.items()))
    for p, e, ez in zip(TOUCH, rxy, rz):
        print(f"     {p['label']:13s} xy err {np.linalg.norm(e):4.1f} mm  z err {ez:+5.1f} mm")
    dist = {k: round(np.linalg.norm(marks[k[0]] - marks[k[2]]) * 1000, 1) for k in MEAS["M3_mark_distances_m"]}
    print("   fitted mark distances mm", dist, "measured", {k: v * 1000 for k, v in MEAS["M3_mark_distances_m"].items()})
    home_h, z, sides, ang = checks(offs, tilt, zt)
    print(f"   CHECK M1 home tip height: {home_h*1e3:.1f} mm (measured {MEAS['M1_home_jaw_tip_height_above_table_m']*1e3:.0f})")
    print(f"   CHECK old 13 touches: height range {np.ptp(z)*1e3:.1f} mm, mean {z.mean()*1e3:+.1f}, std {z.std()*1e3:.1f}; "
          f"table edge angle {ang:.1f} deg (90)")
    print(f"         sides cm {sides} (ruler supply 20.9 x 19.2, stack 21.4 x 14.6)")
    return dict(offsets={k: float(v) for k, v in offs.items()}, tilt=[float(t) for t in tilt], table_z=float(zt),
                cost=float(sol.cost), sd=dict(zip(names, map(float, sd))), home_tip_mm=float(home_h * 1e3))


def main():
    out = {}
    for roll in (0.0, 90.0, -90.0):
        sol, sd = fit(roll)
        out[f"roll{roll:+.0f}"] = report(f"wrist_roll offset fixed {roll:+.0f}", sol, sd, roll, False)
    sol, sd = fit(90.0, free_roll=True)
    out["roll_free_from+90"] = report("wrist_roll offset free (start +90)", sol, sd, 0, True)
    sol, sd = fit(-90.0, free_roll=True)
    out["roll_free_from-90"] = report("wrist_roll offset free (start -90)", sol, sd, 0, True)
    (ROOT / "notes/phase1_logs/fit_joint_offsets.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
