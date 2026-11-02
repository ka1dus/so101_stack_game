"""Phase 8: test skill/stack_look.py on sim wrist renders: random crooked stacks (1-4 boxes, real mix, each level
+-OFF / +-YAW), arm set to the look pose computed from a guess that is off by up to GUESS_ERR, wrist frame rendered
(with the game's visual randomization), top box detected; error vs the true top box pose.

    MUJOCO_GL=cgl python scripts/phase8_look_sim.py [n_cases] [--save K]
"""
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from game.backends import SimBackend  # noqa: E402
from perception.drop import Kinematics  # noqa: E402
from perception.stack import SPOT  # noqa: E402
from perception.supply import BOX  # noqa: E402
from phase8_replay import set_box  # noqa: E402
import skill.stack_look as SL  # noqa: E402
from skill.stack_look import StackLook  # noqa: E402

OFF, YAW, GUESS_ERR = 0.008, 10.0, 0.03
HOME = np.array([0.0, -100.0, 98.0, 75.0, 0.0, 1.0])     # replaced by the saved home below when available


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else 20
    save = int(sys.argv[sys.argv.index("--save") + 1]) if "--save" in sys.argv else 4
    import json
    hp = Path.home() / ".cache/so101_home/my_follower_arm.json"
    home = np.array([json.loads(hp.read_text())[j] for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]) if hp.exists() else HOME
    kin = Kinematics(); look = StackLook(kin); rng = np.random.default_rng(7); errs = []; tiles = []
    b = SimBackend(str(ROOT / "data/policies/act_wcal_060000"), "cpu", seed=1); b.prepare_game(); g = b.game; s = g.sim; m, d = s.m, s.d
    slots = {"no3": [1, 2, 3], "no35": [4, 5, 6, 7], "no10": [8]}
    for i in range(n):
        b.prepare_game(); g = b.game; s = g.sim; m, d = s.m, s.d
        h = int(rng.integers(1, 5)); pool = ["no3"] * 3 + ["no35"] * 4 + ["no10"]; rng.shuffle(pool); types = pool[:h]
        used = {t: 0 for t in slots}; z = 0.0005; yaw0 = rng.uniform(0, np.pi)
        for typ in types:
            sl = slots[typ][used[typ]]; used[typ] += 1
            xy = SPOT + rng.uniform(-OFF, OFF, 2); yaw = yaw0 + np.radians(rng.uniform(-YAW, YAW))
            set_box(m, d, b.body_of[sl], xy, z + BOX[typ]["height"] / 2, yaw); z += BOX[typ]["height"]; top_b = b.body_of[sl]
        mujoco.mj_forward(m, d)
        for _ in range(200):
            mujoco.mj_step(m, d)
        tp = d.xpos[m.body(top_b).id].copy(); Rt = d.xmat[m.body(top_b).id].reshape(3, 3); tyaw = np.arctan2(Rt[1, 0], Rt[0, 0]) % np.pi
        z_top = 0.0002 + sum(BOX[t]["height"] for t in types)
        guess = tp[:2] + rng.uniform(-GUESS_ERR, GUESS_ERR, 2)
        aim = guess.copy(); h = SL.LOOK_H; n_looks = 0; trusted = False; cand = None
        for k in range(6):                                # the trial's look loop (phase8_trial.look_at_stack)
            q, info = look.look_pose(home, aim, z_top, h=h); n_looks += 1
            qs = g.jm.real_to_sim(q); d.qpos[s.qadr] = qs[:5]; d.qpos[s.gadr] = qs[5]; mujoco.mj_forward(m, d)
            img = g.wrist()
            xy, yaw, sc = look.detect(img, q, types[-1], z_top, aim, real=False)
            vis, clr = look.visible(q, types[-1], z_top, xy, yaw, real=False); shift = np.linalg.norm(xy - guess)
            good = sc >= 20 and vis >= 0.98 and clr >= 0.75 and shift <= 0.05
            if good and cand is not None and np.linalg.norm(xy - cand[0]) < 0.006 and abs((np.degrees(yaw - cand[1]) + 90) % 180 - 90) < 8:
                trusted = True; break                     # two looks from different aims agree
            if good:
                cand = (xy, yaw); aim = xy; continue      # look again, aimed at it
            cand = None
            if sc >= 12 and shift <= 0.05:
                aim = xy
            else:
                h += 0.04
        e = np.linalg.norm(xy - tp[:2]) * 1000; ey = abs((np.degrees(yaw - tyaw) + 90) % 180 - 90)
        errs.append((e, ey, trusted, n_looks))
        print(f"case {i:2d} {' '.join(types):22s} look tilt {info["tilt_deg"]:4.1f} deg px {info["target_px"]} h {info['height_mm']:5.1f} | "
              f"guess off {np.linalg.norm(guess - tp[:2]) * 1000:4.1f} mm -> {n_looks} look(s), trusted {trusted}, detected off {e:4.1f} mm, yaw {ey:4.1f} deg, score {sc:5.1f}", flush=True)
        if i < save:
            o = look.outline(img, q, types[-1], z_top, tp[:2], tyaw, real=False, colour=(0, 0, 255))
            o = look.outline(o, q, types[-1], z_top, xy, yaw, real=False, colour=(0, 255, 0))
            tiles.append(cv2.resize(cv2.cvtColor(o, cv2.COLOR_RGB2BGR), (320, 240)))
    e = np.array([r[:2] for r in errs]); tr = np.array([r[2] for r in errs]); nl = np.array([r[3] for r in errs])
    et = e[tr]
    print(f"trusted {tr.sum()}/{len(tr)} (looks: {np.bincount(nl)[1:].tolist()} with 1/2/3); trusted error median {np.median(et[:, 0]):.1f} mm, "
          f"max {et[:, 0].max():.1f} mm, > 10 mm: {(et[:, 0] > 10).sum()}; yaw max {et[:, 1].max():.1f} deg")
    print(f"all: position error median {np.median(e[:, 0]):.1f} mm, 90 % {np.percentile(e[:, 0], 90):.1f}, max {e[:, 0].max():.1f}; "
          f"yaw median {np.median(e[:, 1]):.1f} deg, max {e[:, 1].max():.1f}; > 5 mm: {(e[:, 0] > 5).sum()}/{len(e)}")
    if tiles:
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(ROOT / "notes/img/phase8_look_sim.png"), np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]))


if __name__ == "__main__":
    main()
