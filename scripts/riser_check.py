"""Phase 3: choose the riser height under the arm base.

For each riser height, the arm base is raised in sim/scene.xml (table, mats, backdrop stay)
and every pose the game needs is solved with IK and checked for contact between any arm
link and the table, mats or backdrop (all world geoms):
  - place:  top-down (jaws within 10 deg of vertical) over the stacking spot, onto stacks of
            0..8 boxes, the carried box released 3 mm above the stack top, the straight-down
            approach starting 3 cm higher; stack offsets 0 / +-2 cm both ways
  - pick:   top-down at mat level over a 7 x 7 grid on the black mat (centres >= 2.5 cm inside)
  - home:   your home pose (joint angles unchanged) must not touch anything
All three box heights are checked for place; grip point = 1 cm below the carried box's top.

    MUJOCO_GL=cgl ~/.venvs/so101_stack_sim/bin/python scripts/riser_check.py
"""

from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
LAY = yaml.safe_load((ROOT / "sim/assets/layout.yaml").read_text())
MEAS = yaml.safe_load((ROOT / "inputs/measurements.yaml").read_text())
BOX = {v["sim_name"]: v["dimensions"] for v in MEAS["boxes"]["variants"]}
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
SEEDS = ([25, 0, 30, 70, -90], [25, 30, -10, 80, -90], [25, -20, 60, 40, -90], [25, 10, 10, 90, 0],
         [25, 40, -40, 90, 0], [-40, 0, 30, 70, -90], [-40, 30, -10, 80, 0], [-40, 50, -40, 90, 0])
MEAN_H = (3 * 0.017 + 4 * 0.016 + 0.013) / 8


class Arm:
    def __init__(self, riser):
        self.m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/scene.xml"))
        self.d = mujoco.MjData(self.m)
        self.m.body_pos[self.m.body("base").id][2] += riser
        for b in range(self.m.nbody):                      # park the loose boxes out of the way
            if self.m.body(b).name.startswith("box"):
                adr = self.m.jnt_qposadr[self.m.body_jntadr[b]]
                self.d.qpos[adr:adr + 3] = [5 + b, 5, 0.5]
        self.adr = [self.m.jnt_qposadr[self.m.joint(j).id] for j in ARM]
        self.rng = np.array([self.m.jnt_range[self.m.joint(j).id] for j in ARM])
        self.gb = self.m.body("gripper").id
        self.d.qpos[self.m.jnt_qposadr[self.m.joint("gripper").id]] = np.radians(10)
        mujoco.mj_forward(self.m, self.d)
        g = [i for i in range(self.m.ngeom) if self.m.geom_group[i] == 2
             and self.m.mesh(self.m.geom_dataid[i]).name == "wrist_roll_follower_so101_v1"][0]
        mid = self.m.geom_dataid[g]
        v = self.m.mesh_vert[self.m.mesh_vertadr[mid]:self.m.mesh_vertadr[mid] + self.m.mesh_vertnum[mid]]
        w = self.d.geom_xpos[g] + v @ self.d.geom_xmat[g].reshape(3, 3).T
        loc = (w - self.d.xpos[self.gb]) @ self.d.xmat[self.gb].reshape(3, 3)
        self.grip = loc[np.argmin(loc[:, 2])] + np.array([0.0099, 0, 0])

    def ik(self, target):
        m, d = self.m, self.d
        def r(q):
            d.qpos[self.adr] = q; mujoco.mj_kinematics(m, d)
            R = d.xmat[self.gb].reshape(3, 3)
            tilt = np.degrees(np.arccos(np.clip(R[:, 2] @ [0, 0, 1.0], -1, 1)))
            return np.r_[(d.xpos[self.gb] + R @ self.grip - target) * 1000, tilt / 3]
        best = None
        for s in SEEDS:
            sol = least_squares(r, np.radians(s), bounds=(self.rng[:, 0], self.rng[:, 1]))
            res = r(sol.x); perr, tilt = np.linalg.norm(res[:3]), res[3] * 3
            hit = self.hits(sol.x)
            score = perr + tilt + (1000 if hit else 0)
            if best is None or score < best[0]:
                best = (score, sol.x, perr, tilt, hit)
        return best[1:]

    def hits(self, q):
        m, d = self.m, self.d
        d.qpos[self.adr] = q; mujoco.mj_forward(m, d)
        for i in range(d.ncon):
            b1, b2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
            if (b1 == 0) != (b2 == 0):
                other = m.body(b2 if b1 == 0 else b1).name
                if not other.startswith("box"):
                    return True
        return False

    def ok(self, target, tol=10.0):
        q, perr, tilt, hit = self.ik(target)
        return perr < 2 and tilt <= tol and not hit, tilt, hit


def main():
    sp = np.array(LAY["stack_area"]["stacking_spot"])
    S = np.array(LAY["supply_area"]["corners"])
    e1 = (S[1] - S[0]) / np.linalg.norm(S[1] - S[0]); e2 = (S[3] - S[0]) / np.linalg.norm(S[3] - S[0])
    L1, L2 = np.linalg.norm(S[1] - S[0]), np.linalg.norm(S[3] - S[0])
    supply = [S[0] + a * e1 + b * e2 for a in np.linspace(0.025, L1 - 0.025, 7) for b in np.linspace(0.025, L2 - 0.025, 7)]
    offs = [(0, 0), (0.02, 0), (-0.02, 0), (0, 0.02), (0, -0.02)]
    home = MEAS["arm"]["home_pose"]
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from joint_mapping import JOINT_NAMES, JointMapping, from_dict
    q_home = JointMapping.from_files().real_to_sim(from_dict(home))[:5]
    for riser in [float(x) for x in (sys.argv[1:] or (0.06, 0.075, 0.09))]:
        A = Arm(riser)
        print(f"\n=== riser {riser*100:.1f} cm")
        print(f"  home pose touches something: {A.hits(q_home)}")
        per_box = {}
        for box, dm in BOX.items():
            best = -1; why = ""
            for n in range(0, 9):
                top = n * MEAN_H
                centre_z = top + 0.003 + dm["height"] - 0.010
                fails = []
                for o in offs:
                    for dz in (0.0, 0.03):
                        good, tilt, hit = A.ok(np.r_[sp + o, centre_z + dz])
                        if not good:
                            fails.append(f"offset {o} {'approach' if dz else 'place'}: tilt {tilt:.0f}{' CONTACT' if hit else ''}")
                if fails:
                    why = fails[0]; break
                best = n
            per_box[box] = best
            print(f"  carrying {box:5s}: onto stacks up to {best} boxes" + (f"   (first failure at {best+1}: {why})" if why else ""))
        pick = []
        for p in supply:
            good, tilt, hit = A.ok(np.r_[p, 0.002 + 0.007])
            pick.append(good)
        far = [np.linalg.norm(p) * 100 for p, g in zip(supply, pick) if not g]
        print(f"  supply picks (vertical, mat level): {np.mean(pick)*100:.0f}% of the mat"
              + (f"; failing points from r = {min(far):.1f} cm" if far else ""))
        print(f"  => playable: place onto {min(per_box.values())} boxes -> max stack {min(per_box.values())+1} boxes (of 8)")


if __name__ == "__main__":
    main()
