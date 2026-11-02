"""Phase 8: where does the box sit in the gripper on the real arm, and where was it really released?

The box lies in its template slot (sim/assets/supply_layout.yaml, placed with a template to +-2 mm) until the jaws
close on it, so its pose in the gripper frame = the slot pose seen from the gripper (FK of the joint reading) at the
tick the jaws close. Carried rigidly, the box centre at release = gripper pose at release x that offset; compared with
the perceived stack top centre. Only trials where the robot took the planned slot.

    ~/.venvs/so101_eval/bin/python scripts/phase8_inhand.py data/real_trials/20261029_125445 [more sessions]
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from perception.drop import Kinematics  # noqa: E402
from perception.supply import BOX  # noqa: E402
from phase8_sim_twin import release_tick  # noqa: E402

MAT_Z = 0.0005
SLOTS = {b["id"]: b for b in yaml.safe_load((ROOT / "sim/assets/supply_layout.yaml").read_text())["boxes"]}


def slot_pose(sid):
    """Centre (m) and long-axis unit vector (xy) of a template slot."""
    b = SLOTS[sid]; c = np.r_[b["centre_world_m"], MAT_Z + BOX[b["type"]]["height"] / 2]
    yaw = np.radians(b["yaw_world_deg"]) if "yaw_world_deg" in b else None
    return c, yaw


def gripper_pose(kin, q):
    kin.tip_pos(q); return kin.d.xpos[kin.gb].copy(), kin.d.xmat[kin.gb].reshape(3, 3).copy()


def main():
    kin = Kinematics()
    print("session      t  stack       > pick | box centre in gripper (mm: x, y, z) | release: box centre vs top centre (mm), "
          "box bottom above the face | outcome")
    for sess in map(Path, sys.argv[1:]):
        for line in (sess / "trials.jsonl").read_text().splitlines():
            t = json.loads(line); n = t["trial"]
            if t["picked_slot"] != t["expected_pick_slot"] or t["picked_slot"] is None:
                continue
            tr = np.load(sess / f"t{n:03d}_trace.npy"); q, c = tr[:, :6].astype(float), tr[:, 6:12].astype(float)
            tz = np.array([kin.tip_pos(x.tolist())[2] for x in q]); kg = 60 + int(np.argmin(tz[60:240]))
            held = np.median(q[kg + 30:kg + 90, 5]); ko = int(np.argmax(q[:kg + 30, 5]))
            kc = ko + int(np.argmax(q[ko:, 5] < held + 3))              # jaws closed on the box
            cs, _ = slot_pose(t["picked_slot"]); p, R = gripper_pose(kin, q[kc].tolist()); b_in_g = R.T @ (cs - p)
            top = np.array(t["top_xy"]); k = release_tick(kin, q, c, top); p2, R2 = gripper_pose(kin, q[k].tolist())
            bc = p2 + R2 @ b_in_g; typ = t["picked_type"]
            face = 0.0002 + sum(BOX[x]["height"] for x in t["stack_types"])
            err = (bc[:2] - top) * 1000; up = (bc[2] - BOX[typ]["height"] / 2 - face) * 1000
            print(f"{sess.name} {n:2d} {' '.join(t['stack_types']):11s} > {typ:4s} | {b_in_g[0]*1000:6.1f} {b_in_g[1]*1000:6.1f} {b_in_g[2]*1000:6.1f} | "
                  f"({err[0]:+5.1f}, {err[1]:+5.1f}) = {np.linalg.norm(err):4.1f} mm, {up:+5.1f} mm | {t['outcome']} {t['failure'] or ''}")


if __name__ == "__main__":
    main()
