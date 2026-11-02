"""Phase 8: run the policy closed-loop in sim on a real trial's scene and compare where it releases with the real arm.

The trial's stack is built at the perceived stack top (trials.jsonl top_xy), the supply as in the template with the
trial's set-aside boxes removed, the arm starts at the real first reading, and the sim servos use the fit
(sim/assets/servo_fit.yaml). Several runs with different visual randomization seeds. Prints the jaw-tip position at
release (real vs each sim run) and the outcome in sim.

    MUJOCO_GL=cgl ~/.venvs/so101_eval/bin/python scripts/phase8_sim_twin.py data/real_trials/20261028_105812 1 \
        --policy data/policies/act_softB_080000 --runs 6
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import mujoco  # noqa: E402
import servo_model  # noqa: E402
from game.backends import SimBackend  # noqa: E402
from joint_mapping import JOINT_NAMES  # noqa: E402
from perception.drop import Kinematics  # noqa: E402
from perception.supply import BOX  # noqa: E402
from phase8_replay import set_box  # noqa: E402
from skill.grip_floor import GripFloor  # noqa: E402
from skill.runner import load_policy, run_place_skill  # noqa: E402


def release_tick(kin, q, cmd, top):
    """First tick with the jaw tips within 8 cm of the stack top where the gripper command opens > 3 units past the
    gripper reading on arrival there (the held box's width)."""
    tips = np.array([kin.tip_pos(x.tolist()) for x in q]); near = np.flatnonzero(np.linalg.norm(tips[:, :2] - top, axis=1) < 0.08)
    if len(near) == 0:
        return len(q) - 1
    k0 = near[0]; held = q[k0, 5]; op = np.flatnonzero((cmd[k0:, 5] > held + 3))
    return k0 + int(op[0]) if len(op) else len(q) - 1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("session"); ap.add_argument("trial", type=int)
    ap.add_argument("--policy", default=str(ROOT / "data/policies/act_softB_080000")); ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--servo", default=str(ROOT / "sim/assets/servo_fit.yaml")); ap.add_argument("--device", default="mps")
    ap.add_argument("--grip-offset", type=float, default=0.0,
                    help="added to the gripper reading the policy sees while it holds a box, e.g. -4 to mimic the real cardboard squeeze")
    ap.add_argument("--grip-floor", type=float, default=None, help="skill.grip_floor.GripFloor: min opening before the first grasp")
    ap.add_argument("--grip-cap", type=float, default=None, help="GripFloor cap: max opening before the first grasp (mimic the real ~24)")
    ap.add_argument("--release-floor", type=float, default=None, help="run_place_skill release_floor")
    a = ap.parse_args(); sess = Path(a.session); kin = Kinematics()
    t = json.loads((sess / "trials.jsonl").read_text().splitlines()[a.trial - 1])
    tr = np.load(sess / f"t{a.trial:03d}_trace.npy"); rq, rc = tr[:, :6].astype(float), tr[:, 6:12].astype(float)
    top = np.array(t["top_xy"]); kr = release_tick(kin, rq, rc, top); tip_r = kin.tip_pos(rq[kr].tolist())
    print(f"stack top (perceived) ({top[0]*100:.1f}, {top[1]*100:.1f}) cm")
    print(f"real: release at {kr/30:.1f} s, tip ({tip_r[0]*100:.1f}, {tip_r[1]*100:.1f}, {tip_r[2]*100:.1f}) cm, "
          f"{np.linalg.norm(tip_r[:2]-top)*100:.1f} cm from the top; outcome {t['outcome']} {t['failure']}")
    import cv2
    for kk, tag in ((max(kr - 15, 0), "pre"), (kr, "rel")):                     # real front + wrist at release (and 0.5 s before)
        fr = []
        for cam in ("front", "wrist"):
            c = cv2.VideoCapture(str(sess / f"t{a.trial:03d}_{cam}.mp4")); c.set(cv2.CAP_PROP_POS_FRAMES, kk); ok, im = c.read(); fr.append(im)
        cv2.imwrite(str(ROOT / f"notes/img/phase8_twin_{sess.name}_t{a.trial:03d}_real_{tag}.png"), np.hstack(fr))
    params = servo_model.load(a.servo); out = []; policy = load_policy(a.policy, a.device)
    for r in range(a.runs):
        b = SimBackend(a.policy, a.device, seed=20000 + r); b.prepare_game(); g = b.game; s = g.sim; m, d = s.m, s.d
        servo_model.apply_params(m, params)
        z = 0.0005
        for slot, typ in zip(t["stack_slots"], t["stack_types"]):
            set_box(m, d, b.body_of[slot], top, z + BOX[typ]["height"] / 2, np.radians(-30)); z += BOX[typ]["height"]
        for slot in t["aside_slots"]:
            set_box(m, d, b.body_of[slot], (0.0, 0.30), 0.02, 0.0)
        q0 = g.jm.real_to_sim(rq[0]); d.qpos[s.qadr] = q0[:5]; d.qpos[s.gadr] = q0[5]
        for j, v in zip(JOINT_NAMES, q0):
            d.ctrl[s.act[j]] = v
        mujoco.mj_forward(m, d); s.sync_plan(); s.wait(1.0)
        start = {sl: g.pose(bd)[0].copy() for sl, bd in b.body_of.items()}
        rec = []; off = a.grip_offset
        sf = None
        if off:                                                    # only once the jaws have opened and closed on a box
            mx = [0.0]
            def sf(q, mx=mx):
                mx[0] = max(mx[0], q[5])
                if mx[0] > 20 and q[5] < mx[0] - 3 and q[5] > 8:
                    mx.append(1); return np.r_[q[:5], q[5] + off]
                return q
        gf = GripFloor(a.grip_floor, a.grip_cap) if (a.grip_floor or a.grip_cap) else None
        res = run_place_skill(b.arm, policy, kin, top, z, record=rec, state_fn=sf, act_fn=gf, release_floor=a.release_floor)
        if gf:
            print(f"  grip floor {a.grip_floor} / cap {a.grip_cap}: active {gf.ticks} ticks, policy peak {gf.peak:.1f}")
        if off:
            print(f"  gripper offset {off:+.1f} applied on {len(mx) - 1} of {len(res['trace'])} ticks (max opening read {mx[0]:.1f})")
        st = res["trace"]; k = release_tick(kin, st[:, :6].astype(float), st[:, 6:12].astype(float), top); tip = kin.tip_pos(st[k, :6].astype(float).tolist())
        import cv2
        for kk, tag in ((max(k - 15, 0), 'pre'), (k, 'rel')):
            cv2.imwrite(str(ROOT / f"notes/img/phase8_twin_{sess.name}_t{a.trial:03d}_sim{r}_{tag}.png"), np.hstack(rec[min(kk, len(rec) - 1)]))
        s.wait(3.0)
        picked = [bd for bd, p in ((bd, g.pose(bd)[0]) for bd in b.body_of.values()) if np.linalg.norm(p[:2] - top) < 0.06 and p[2] > z]
        print(f"sim {r}: {res['event']} at {k/30:.1f} s, tip ({tip[0]*100:.1f}, {tip[1]*100:.1f}, {tip[2]*100:.1f}) cm, "
              f"{np.linalg.norm(tip[:2]-top)*100:.1f} cm from the top; gripper reading 0.5 s before {st[max(k - 15, 0), 5]:.1f}; "
              f"box on the stack after 3 s: {bool(picked)}", flush=True)
        moved = {sl: np.linalg.norm(g.pose(bd)[0][:2] - start[sl][:2]) for sl, bd in b.body_of.items() if sl not in t["stack_slots"]}
        took = max(moved, key=moved.get) if max(moved.values()) > 0.03 else None
        bumped = sorted(f"{sl}:{v * 1000:.0f}mm" for sl, v in moved.items() if sl != took and sl not in t["aside_slots"] and v > 0.005)
        sq = st[:, :6].astype(float); stz = np.array([kin.tip_pos(x.tolist())[2] for x in sq]); kl = 60 + int(np.argmin(stz[60:240]))
        print(f"  GRASP sim {r}: lowest tip in 2-8 s {stz[kl] * 100:.1f} cm at {kl / 30:.1f} s, gripper max before {sq[:kl, 5].max():.1f}", flush=True)
        exp = t["expected_pick_slot"]; on = bool(picked)
        print(f"  PICK sim {r}: expected slot {exp}, took slot {took} -> {'right' if took == exp else 'WRONG' if took is not None else 'none'}; "
              f"placed on the stack {on}; neighbours moved > 5 mm: {bumped or 'none'}", flush=True)
        out.append(tip)
    out = np.array(out); print(f"sim mean tip ({out[:,0].mean()*100:.1f}, {out[:,1].mean()*100:.1f}, {out[:,2].mean()*100:.1f}) cm; "
                               f"real - sim: {np.round((tip_r - out.mean(0))*100, 1)} cm")


if __name__ == "__main__":
    main()
