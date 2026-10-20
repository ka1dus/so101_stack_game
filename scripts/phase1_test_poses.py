"""Phase 1: diagnostic poses for the real arm + safety check in sim.

Writes notes/phase1_poses/<name>.json (LeRobot units, same format as the home-pose file,
so `so101_go_home.py --home-file` can drive to them) and checks, under every mapping
hypothesis still open, that the gripper and wrist stay well above the table along the
straight joint-space path home -> pose -> home.

    MUJOCO_GL=cgl python scripts/phase1_test_poses.py
"""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from joint_mapping import BODY_JOINTS, JOINT_NAMES, JointMapping, as_dict, from_dict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notes" / "phase1_poses"
HOME = dict(shoulder_pan=-3.16, shoulder_lift=-87.30, elbow_flex=96.79, wrist_flex=48.35, wrist_roll=8.66, gripper=0.74)

# Open hypotheses (notes/phase1.md section 4): body offsets (deg) + base tilt (deg about x, y).
HYPS = {
    "H0": ({}, (0.0, 0.0)),
    "tilt": ({}, (-1.35, 0.19)),
    "fit_roll+90": (dict(shoulder_lift=-6.93, elbow_flex=4.22, wrist_flex=9.95, wrist_roll=90), (-1.56, -0.51)),
    "fit_roll-90": (dict(shoulder_lift=-6.74, elbow_flex=4.43, wrist_flex=8.93, wrist_roll=-90), (-1.60, -0.34)),
    "fit_roll0": (dict(shoulder_lift=-6.94, elbow_flex=4.75, wrist_flex=12.97), (-1.59, -0.41)),
    "M3_roll+95": (dict(shoulder_lift=-5.66, elbow_flex=6.82, wrist_flex=4.61, wrist_roll=95.4), (-1.10, 0.47)),
    "M3_roll+5": (dict(shoulder_lift=-5.64, elbow_flex=6.89, wrist_flex=8.03, wrist_roll=4.68), (-1.15, 0.44)),
}

# Poses (LeRobot readings). Chosen to be diagnostic, not to touch anything:
#   P1 above the stacking spot, gripper ~vertical: wrist camera looks straight at the yellow X
#   P2 turned toward the supply side and raised: front camera sees the arm side-on (link angles)
#   P3 = P1 with wrist_roll +45: which way the jaws turn fixes the sign of the roll offset
RAISE = dict(HOME, elbow_flex=45.0)   # waypoint: unfolding the elbow lifts the gripper straight up from home
POSES = {
    "P1_over_stack": dict(shoulder_pan=27.0, shoulder_lift=-20.0, elbow_flex=10.0, wrist_flex=75.0, wrist_roll=8.66, gripper=30.0),
    "P2_side_raised": dict(shoulder_pan=-55.0, shoulder_lift=-45.0, elbow_flex=20.0, wrist_flex=40.0, wrist_roll=8.66, gripper=0.74),
    "P3_over_stack_roll45": dict(shoulder_pan=27.0, shoulder_lift=-20.0, elbow_flex=10.0, wrist_flex=75.0, wrist_roll=53.66, gripper=30.0),
}
MIN_CLEARANCE_M = 0.06   # lowest arm point above the table once away from home; boxes are <= 1.7 cm tall
NEAR_HOME_S = 0.25       # first quarter of the path: may be low (home is low) but must not dip below home

MODEL = mujoco.MjModel.from_xml_path(str(ROOT / "sim/assets/so101/so101_new_calib_camera.xml"))
DATA = mujoco.MjData(MODEL)
ADR = [MODEL.jnt_qposadr[MODEL.joint(j).id] for j in JOINT_NAMES]
ARM_GEOMS = [g for g in range(MODEL.ngeom) if MODEL.geom_group[g] == 2 and MODEL.body(MODEL.geom_bodyid[g]).name
             not in ("base", "shoulder")]
BASE_Z = 0.005  # base origin above the table (touch fit)


def lowest_point(q_lerobot, offsets, tilt):
    jm = JointMapping.from_files()
    jm.offset_deg = {j: offsets.get(j, 0.0) for j in BODY_JOINTS}
    q = jm.real_to_sim(q_lerobot)
    for a, v in zip(ADR, q):
        DATA.qpos[a] = v
    mujoco.mj_forward(MODEL, DATA)
    rx, ry = np.radians(tilt)
    R = (np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
         @ np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]))
    zmin, where = np.inf, None
    for g in ARM_GEOMS:
        mid = MODEL.geom_dataid[g]
        v = MODEL.mesh_vert[MODEL.mesh_vertadr[mid]:MODEL.mesh_vertadr[mid] + MODEL.mesh_vertnum[mid]]
        w = (DATA.geom_xpos[g] + v @ DATA.geom_xmat[g].reshape(3, 3).T) @ R.T
        i = np.argmin(w[:, 2])
        if w[i, 2] < zmin:
            zmin, where = w[i, 2], (MODEL.body(MODEL.geom_bodyid[g]).name, w[i])
    return zmin + BASE_Z, where


def go_home_path(a, b, speed=15.0, fps=30):
    """Waypoints of so101_go_home.py driving from a to b: every joint (gripper too, in its
    0-100 units) moves toward its goal by at most speed/fps per tick, independently, so
    joints with short moves finish early (NOT a straight joint-space line)."""
    step = speed / fps
    q, path = a.astype(float).copy(), [a.astype(float).copy()]
    while np.any(np.abs(b - q) > 1e-9):
        q = q + np.clip(b - q, -step, step)
        path.append(q.copy())
    return path


def check_leg(a, b, offs, tilt, first_leg):
    """Lowest arm point along the go_home trajectory a -> b, sampled every ~4 ticks."""
    path = go_home_path(a, b)
    path = path[::4] + [path[-1]]
    zs = np.array([lowest_point(p, offs, tilt)[0] for p in path])
    if first_leg:   # leaving home: may start low, must never dip below home, then clear the table
        return zs, bool(zs.min() >= zs[0] - 1e-4 and zs[len(zs) // 2:].min() >= MIN_CLEARANCE_M)
    return zs, bool(zs.min() >= MIN_CLEARANCE_M)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lim = JointMapping.from_files().lerobot_limits()
    home, raise_ = from_dict(HOME), from_dict(RAISE)
    ok_all = True
    (OUT / "R_raise.json").write_text(json.dumps(as_dict(raise_), indent=2))
    print("leg home -> R_raise (and back):")
    for hyp, (offs, tilt) in HYPS.items():
        zs, ok = check_leg(home, raise_, offs, tilt, True)
        zr, okr = check_leg(raise_, home, offs, tilt, False)
        okr = bool(zr.min() >= min(zr[-1], MIN_CLEARANCE_M) - 1e-4)   # returning: may end low only at home itself
        ok &= okr
        ok_all &= ok
        print(f"  {hyp:12s} home {zs[0]*100:5.1f} cm, min {zs.min()*100:5.1f}, at R {zs[-1]*100:5.1f} cm  {'OK' if ok else 'NOT OK'}")
    for name, pose in POSES.items():
        q = from_dict(pose)
        assert np.all(q >= lim[:, 0]) and np.all(q <= lim[:, 1]), (name, "outside calibrated range")
        print(f"\nleg R_raise -> {name} (and back): {pose}")
        for hyp, (offs, tilt) in HYPS.items():
            zs, ok = check_leg(raise_, q, offs, tilt, False)
            zb, okb = check_leg(q, raise_, offs, tilt, False)
            zs, ok = np.r_[zs, zb], ok and okb
            ok_all &= ok
            z_end, (body, pt) = lowest_point(q, offs, tilt)
            print(f"  {hyp:12s} min clearance {zs.min()*100:5.1f} cm | final {z_end*100:5.1f} cm ({body})  {'OK' if ok else 'NOT OK'}")
        (OUT / f"{name}.json").write_text(json.dumps(as_dict(q), indent=2))
    print("\nALL LEGS SAFE" if ok_all else "\nSOME LEGS TOO LOW - do not use")


if __name__ == "__main__":
    main()
