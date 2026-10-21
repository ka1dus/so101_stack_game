"""Phase 3: playable max stack height = min(arm reach, front-camera view).

Reach: for each stack-top height z, IK on sim/scene.xml (validated mapping, real joint
ranges) for a top-down placement over the stacking spot:
  - place pose: jaws vertical (within 10 deg), jaw tips 1 cm below the carried box's top,
    carried box bottom 3 mm above the stack top (release height)
  - approach pose: the same, 5 cm higher (straight-down descent)
for each box type (heights differ), and with +/-2 cm stack offset (crooked stacks).
Feasible = both poses solved to < 2 mm / 3 deg inside the joint ranges, no arm-table contact.
View: the top of an n-box stack (and the carried box above it) must project inside the
calibrated front image with a 10 px margin.

    MUJOCO_GL=cgl ~/.venvs/so101_stack_sim/bin/python scripts/max_height.py
"""

from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
M = mujoco.MjModel.from_xml_path(str(ROOT / "sim/scene.xml"))
D = mujoco.MjData(M)
LAY = yaml.safe_load((ROOT / "sim/assets/layout.yaml").read_text())
MEAS = yaml.safe_load((ROOT / "inputs/measurements.yaml").read_text())
CAM = yaml.safe_load((ROOT / "sim/assets/cameras.yaml").read_text())["front"]
BOX = {v["sim_name"]: v["dimensions"] for v in MEAS["boxes"]["variants"]}
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ADR = [M.jnt_qposadr[M.joint(j).id] for j in ARM]
RNG = np.array([M.jnt_range[M.joint(j).id] for j in ARM])
GB = M.body("gripper").id
GADR = M.jnt_qposadr[M.joint("gripper").id]
MEAN_H = (3 * 0.017 + 4 * 0.016 + 0.013) / 8
SEEDS = ([25, 0, 30, 70, -90], [25, 30, -10, 80, -90], [25, -20, 60, 40, -90], [25, 10, 10, 90, 0], [25, 40, -40, 90, 0])


def tip_frame():
    """Fixed-jaw tip in the gripper frame (lowest point along -z)."""
    g = [i for i in range(M.ngeom) if M.geom_group[i] == 2 and M.mesh(M.geom_dataid[i]).name == "wrist_roll_follower_so101_v1"][0]
    mujoco.mj_resetData(M, D); mujoco.mj_forward(M, D)
    mid = M.geom_dataid[g]
    v = M.mesh_vert[M.mesh_vertadr[mid]:M.mesh_vertadr[mid] + M.mesh_vertnum[mid]]
    w = D.geom_xpos[g] + v @ D.geom_xmat[g].reshape(3, 3).T
    loc = (w - D.xpos[GB]) @ D.xmat[GB].reshape(3, 3)
    return loc[np.argmin(loc[:, 2])]


TIP = tip_frame()
GRIP_CENTRE = TIP + np.array([0.0099, 0, 0])   # between the jaws (~half the closed-jaw gap toward the moving jaw)


def ik(target, q0):
    def r(q):
        D.qpos[ADR] = q; mujoco.mj_kinematics(M, D)
        R = D.xmat[GB].reshape(3, 3)
        p = D.xpos[GB] + R @ GRIP_CENTRE
        tilt = np.degrees(np.arccos(np.clip(R[:, 2] @ [0, 0, 1.0], -1, 1)))   # jaws along -z -> z axis up
        return np.r_[(p - target) * 1000, tilt / 3]
    sol = least_squares(r, q0, bounds=(RNG[:, 0], RNG[:, 1]))
    D.qpos[ADR] = sol.x; mujoco.mj_forward(M, D)
    res = r(sol.x)
    return sol.x, np.linalg.norm(res[:3]), res[3] * 3


def arm_hits_table(q):
    D.qpos[ADR] = q; mujoco.mj_forward(M, D)
    for i in range(D.ncon):
        b1, b2 = M.geom_bodyid[D.contact[i].geom1], M.geom_bodyid[D.contact[i].geom2]
        n1, n2 = M.body(b1).name, M.body(b2).name
        if (b1 == 0) != (b2 == 0) and not (n1.startswith("box") or n2.startswith("box")):
            return True
    return False


def reach(top_z, box, offset):
    sp = np.array(LAY["stack_area"]["stacking_spot"]) + offset
    h = BOX[box]["height"]
    grasp_below_top = 0.010
    place_centre = np.r_[sp, top_z + 0.003 + h - grasp_below_top]
    ok = True; worst = 0
    for dz in (0.05, 0.0):          # approach, then place
        best = None
        for seed in SEEDS:          # several IK starts; keep the best solution
            q, perr, tilt = ik(place_centre + [0, 0, dz], np.radians(seed))
            if best is None or perr + tilt < best[1] + best[2]:
                best = (q, perr, tilt)
        q, perr, tilt = best
        good = perr < 2 and tilt < 10 and not arm_hits_table(q)
        ok &= good; worst = max(worst, tilt)
    return ok, worst


def project(P):
    R, _ = cv2.Rodrigues(np.array(CAM["rvec"]))
    pc = (P - np.array(CAM["pos"])) @ R.T
    xy = pc[:, :2] / pc[:, 2:3]; r2 = np.sum(xy ** 2, 1, keepdims=True)
    return CAM["f_px"] * xy * (1 + CAM["k1"] * r2) + [320, 240]


def main():
    D.qpos[GADR] = np.radians(10)
    print("REACH (top-down place + 5 cm straight-down approach over the stacking spot)")
    max_reach = {}
    for box in BOX:
        last = 0
        for n in range(0, 16):
            top = n * MEAN_H
            ok = all(reach(top, box, np.array(o))[0] for o in ((0, 0), (0.02, 0), (-0.02, 0), (0, 0.02), (0, -0.02)))
            if not ok:
                break
            last = n
        max_reach[box] = last
        print(f"  carrying {box:5s}: can place onto stacks up to {last} boxes (top {last*MEAN_H*100:.1f} cm) incl. +/-2 cm offsets")
    print("VIEW (front camera, calibrated; stack top + carried box inside the image, 10 px margin)")
    sp = np.array(LAY["stack_area"]["stacking_spot"])
    max_view = 0
    for n in range(1, 30):
        top = n * MEAN_H
        corners = np.array([[sp[0] + dx, sp[1] + dy, top + z] for dx in (-0.035, 0.035) for dy in (-0.035, 0.035) for z in (0, 0.017)])
        uv = project(corners)
        inside = np.all((uv[:, 0] > 10) & (uv[:, 0] < 630) & (uv[:, 1] > 10) & (uv[:, 1] < 470))
        if not inside:
            break
        max_view = n
    print(f"  stacks up to {max_view} boxes fit in the front view (top {max_view*MEAN_H*100:.1f} cm)")
    playable = min(min(max_reach.values()), max_view, MEAS["boxes"]["count"])
    print(f"PLAYABLE MAX HEIGHT = {playable} boxes (reach {min(max_reach.values())}, view {max_view}, box count {MEAS['boxes']['count']})")


if __name__ == "__main__":
    main()
