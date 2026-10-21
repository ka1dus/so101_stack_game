"""Phase 3: hover poses above the stacking spot for scripts/phase3_spot_check.sh.

S1: closed jaw tips 7 cm straight above the spot; S2: 3 cm above. Jaws vertical.
Safety: every leg (R_raise <-> S1 <-> S2) simulated the way so101_go_home.py drives it
(per-joint rate limit), under the validated mapping and every older hypothesis.

    MUJOCO_GL=cgl ~/.venvs/so101_stack_sim/bin/python scripts/make_spot_check_poses.py
"""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
import phase1_test_poses as T  # noqa: E402
from joint_mapping import JointMapping, from_dict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notes/phase3_poses"
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
LAY = yaml.safe_load((ROOT / "sim/assets/layout.yaml").read_text())
M = mujoco.MjModel.from_xml_path(str(ROOT / "sim/scene.xml"))
D = mujoco.MjData(M)
JM = JointMapping.from_files()
ADR = [M.jnt_qposadr[M.joint(j).id] for j in J]


def tips():
    pts = []
    for mesh in ("wrist_roll_follower_so101_v1", "moving_jaw_so101_v1"):
        g = [i for i in range(M.ngeom) if M.geom_group[i] == 2 and M.mesh(M.geom_dataid[i]).name == mesh][0]
        mid = M.geom_dataid[g]
        v = M.mesh_vert[M.mesh_vertadr[mid]:M.mesh_vertadr[mid] + M.mesh_vertnum[mid]]
        w = D.geom_xpos[g] + v @ D.geom_xmat[g].reshape(3, 3).T
        pts.append(w[np.argmin(w[:, 2])])
    return np.mean(pts, 0)


def solve(target, seeds):
    def r(x):
        D.qpos[ADR] = JM.real_to_sim(from_dict(dict(zip(J, [*x, 0.74]))))
        mujoco.mj_forward(M, D)
        R = D.xmat[M.body("gripper").id].reshape(3, 3)
        return np.r_[(tips() - target) * 1000, np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))) / 2]
    best = min((least_squares(r, s, diff_step=1e-3) for s in seeds), key=lambda s: s.cost)
    return best.x, r(best.x)


def main():
    sp = np.array(LAY["stack_area"]["stacking_spot"])
    qa, ra = solve(np.r_[sp, 0.07], [[22, 20, 0, 80, 8.66], [22, 40, -30, 95, 8.66], [22, 10, 20, 70, 8.66]])
    qb, rb = solve(np.r_[sp, 0.03], [qa])
    print(f"target spot {np.round(sp*100,1)} cm | S1 (7 cm) tilt {ra[3]*2:.1f} deg, tip err {np.linalg.norm(ra[:3]):.1f} mm"
          f" | S2 (3 cm) tilt {rb[3]*2:.1f} deg, tip err {np.linalg.norm(rb[:3]):.1f} mm")
    OUT.mkdir(parents=True, exist_ok=True)
    poses = {"S1_above_new_spot": [*np.round(qa, 2), 0.74], "S2_hover_new_spot": [*np.round(qb, 2), 0.74]}
    for k, v in poses.items():
        (OUT / f"{k}.json").write_text(json.dumps(dict(zip(J, map(float, v))), indent=2))
        print(k, {a: round(float(b), 2) for a, b in zip(J, v)})
    R = from_dict(T.RAISE); A = np.array(poses["S1_above_new_spot"]); B = np.array(poses["S2_hover_new_spot"])
    for group, hyps in (("validated mapping", ["M3_roll+95"]), ("all hypotheses", list(T.HYPS))):
        worst = min(min(T.lowest_point(p, *T.HYPS[h])[0] for p in T.go_home_path(a, b)[::4] + [b])
                    for h in hyps for a, b in ((R, A), (A, B), (B, A), (A, R)))
        print(f"  lowest arm point on R_raise<->S1<->S2 legs, {group}: {worst*100:.1f} cm above the table")


if __name__ == "__main__":
    main()
