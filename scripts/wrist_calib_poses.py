"""Phase 8: arm poses for the wrist camera calibration (ChArUco board lying flat on the yellow stack area).

Board: 5 x 7 squares of 28 mm (ChArUco, DICT_5X5_100, markers 21 mm), 140 x 196 mm, long side along the yellow
area's long side, centred on it (`board_world()`). Poses are sampled around S1 (jaws vertical ~7 cm above the X) and
kept if, in the sim twin with the current wrist model:
  - at least 16 of the 24 inner board corners project inside the image (30 px margin), board 12-35 cm from the lens;
  - the jaw tips pass the skill's safety check and stay >= 5 cm above the table;
  - every arm collision geom stays >= 25 mm from the table top and the backdrop.
Then 14 diverse poses are chosen (lens position and viewing direction), ordered, and every leg between them is
simulated the way `skill.runner.move_to` drives it (same clearances); a leg that fails goes via S1.
Writes notes/wrist_calib/poses.json (sequence, LeRobot units) and a contact sheet of the predicted wrist views.

    MUJOCO_GL=cgl ~/.venvs/so101_eval/bin/python scripts/wrist_calib_poses.py
"""
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import grasp_test as GT  # noqa: E402
from joint_mapping import JOINT_NAMES  # noqa: E402
from perception.drop import Kinematics  # noqa: E402
from perception.stack import AREA  # noqa: E402
from skill.runner import FPS, HOME_SPEED, safety_check  # noqa: E402

SQ, NX, NY = 0.028, 5, 7                 # square size, squares across (short side), along (long side)
CLEAR_M, TIP_MIN_M = 0.025, 0.05
OUT = ROOT / "notes/wrist_calib"
W, H = 640, 480


def board_world():
    """Board centre, long axis (along yellow corner 1->2), short axis (corner 1->4), and the 24 inner corners (world)."""
    c = AREA.mean(0); u = (AREA[1] - AREA[0]) / np.linalg.norm(AREA[1] - AREA[0]); v = (AREA[3] - AREA[0]) / np.linalg.norm(AREA[3] - AREA[0])
    pts = [np.r_[c + (i - (NX - 2) / 2) * SQ * v + (j - (NY - 2) / 2) * SQ * u, 0.002] for j in range(NY - 1) for i in range(NX - 1)]
    return c, u, v, np.array(pts)


class Twin:
    def __init__(self):
        self.s = GT.Sim(); m = self.s.m; self.m, self.d = m, mujoco.MjData(m); self.jm = self.s.jm; self.kin = Kinematics()
        self.cam = m.camera("wrist").id; self.f = 0.5 * H / np.tan(np.radians(m.cam_fovy[self.cam]) / 2)
        arm_bodies = set()
        for b in range(m.nbody):
            n = m.body(b).name; p = b
            while p > 0 and m.body(p).name != "shoulder":
                p = m.body_parentid[p]
            if p > 0:
                arm_bodies.add(b)
        self.arm_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in arm_bodies and m.geom_contype[g] | m.geom_conaffinity[g]]
        self.static = [m.geom("table_top").id, m.geom("backdrop").id]
        self.ren = None

    def set(self, q_real):
        q = self.jm.real_to_sim(np.asarray(q_real, float)); d = self.d
        d.qpos[self.s.qadr] = q[:5]; d.qpos[self.s.gadr] = q[5]; mujoco.mj_forward(self.m, d)

    def clearance(self):
        ft = np.zeros(6)
        return min(mujoco.mj_geomDistance(self.m, self.d, a, b, 0.2, ft) for a in self.arm_geoms for b in self.static)

    def project(self, P):
        R = self.d.cam_xmat[self.cam].reshape(3, 3); x = (P - self.d.cam_xpos[self.cam]) @ R   # camera axes: x right, y up, -z forward
        z = -x[:, 2]; u = W / 2 + self.f * x[:, 0] / z; v = H / 2 - self.f * x[:, 1] / z
        return np.c_[u, v], z

    def ok(self, q, P):
        if safety_check(self.kin, q) or self.kin.tip_pos(list(q))[2] < TIP_MIN_M:
            return None
        self.set(q)
        if self.clearance() < CLEAR_M:
            return None
        uv, z = self.project(P)
        inside = (z > 0.12) & (z < 0.35) & (uv[:, 0] > 30) & (uv[:, 0] < W - 30) & (uv[:, 1] > 30) & (uv[:, 1] < H - 30)
        if inside.sum() < 16:
            return None
        R = self.d.cam_xmat[self.cam].reshape(3, 3)
        return np.r_[self.d.cam_xpos[self.cam] / 0.05, -R[:, 2] / 0.25], int(inside.sum())

    def render(self, q, P):
        if self.ren is None:
            self.ren = mujoco.Renderer(self.m, H, W)
        self.set(q); self.ren.update_scene(self.d, "wrist"); img = cv2.cvtColor(self.ren.render(), cv2.COLOR_RGB2BGR)
        uv, z = self.project(P)
        for p in uv[z > 0]:
            cv2.circle(img, tuple(int(c) for c in p), 4, (255, 0, 255), -1)
        return img


def leg(tw, a, b, speed=HOME_SPEED):
    """Simulate move_to's stepping from a to b (LeRobot units); worst clearance and tip check along the way."""
    q = np.asarray(a, float).copy(); b = np.asarray(b, float); step = speed / FPS; worst = 1.0; k = 0
    while np.abs(b - q).max() > 0.3:
        q = q + np.clip(b - q, -step, step); k += 1
        if k % 3 == 0:
            if safety_check(tw.kin, q):
                return -1.0
            tw.set(q); worst = min(worst, tw.clearance())
    return worst


def main():
    tw = Twin(); c, u, v, P = board_world(); rng = np.random.default_rng(0)
    s1 = np.array([json.loads((ROOT / "notes/phase3_poses/S1_above_new_spot.json").read_text())[j] for j in JOINT_NAMES])
    raise_ = np.array([json.loads((ROOT / "notes/phase1_poses/R_raise.json").read_text())[j] for j in JOINT_NAMES])
    lo, hi = tw.jm.lerobot_limits().T; lo, hi = lo + 2, hi - 2
    span = np.array([15, 25, 30, 25, 60, 0])
    cands = []
    for _ in range(6000):
        q = np.clip(s1 + rng.uniform(-1, 1, 6) * span, lo, hi); q[5] = s1[5]
        r = tw.ok(q, P)
        if r is not None:
            cands.append((q, r[0], r[1]))
    print(f"valid poses: {len(cands)} / 6000", flush=True)
    cands = [c_ for c_ in cands if min(leg(tw, s1, c_[0]), leg(tw, c_[0], s1)) >= CLEAR_M]
    print(f"  with safe legs to and from S1: {len(cands)}", flush=True)
    F = np.array([f for _, f, _ in cands]); pick = [int(np.argmax([n for *_, n in cands]))]
    while len(pick) < 14:
        dmin = np.min(np.linalg.norm(F[:, None] - F[pick][None], axis=2), axis=1); pick.append(int(np.argmax(dmin)))
    poses = [cands[i][0] for i in pick]
    order = [];  cur = s1; left = list(range(len(poses)))           # nearest neighbour from S1
    while left:
        j = min(left, key=lambda i: np.abs(poses[i] - cur).max()); order.append(j); left.remove(j); cur = poses[j]
    seq = [("home", None), ("R_raise", raise_), ("S1", s1)]; prev = s1; via = 0
    for n, j in enumerate(order):
        q = poses[j]
        if leg(tw, prev, q) < CLEAR_M:
            seq.append(("S1", s1)); via += 1
        seq.append((f"C{len(seq)}", q)); prev = q
    seq += [("S1", s1), ("R_raise", raise_), ("home", None)]
    home = np.array(list(tw.jm.sim_to_real(tw.s.home)))
    w = leg(tw, home, raise_); w2 = leg(tw, raise_, s1)
    print(f"legs home->R_raise {w * 1000:.0f} mm, R_raise->S1 {w2 * 1000:.0f} mm min clearance; {via} detours via S1")
    OUT.mkdir(parents=True, exist_ok=True); calib = [(n, q) for n, q in seq if n.startswith("C")]
    (OUT / "poses.json").write_text(json.dumps({
        "board": {"squares": [NX, NY], "square_m": SQ, "marker_m": 0.021, "dict": "DICT_5X5_100",
                  "centre_xy": c.tolist(), "long_axis": u.tolist(), "short_axis": v.tolist()},
        "sequence": [{"name": n, "capture": n.startswith("C"), "q": None if q is None else [round(float(x), 2) for x in q]} for n, q in seq]},
        indent=1))
    tiles = []
    for n, q in calib:
        im = tw.render(q, P); tw.set(q); tip = tw.kin.tip_pos(list(q))
        cv2.putText(im, f"{n} tips {tip[2] * 100:.0f} cm up, clear {tw.clearance() * 1000:.0f} mm", (10, 25), 0, 0.7, (0, 0, 255), 2)
        tiles.append(cv2.resize(im, (320, 240)))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(ROOT / "notes/img/wrist_calib_poses.png"), sheet)
    print(f"{len(calib)} capture poses, {len(seq)} moves -> {OUT / 'poses.json'}; views: notes/img/wrist_calib_poses.png")


if __name__ == "__main__":
    main()
