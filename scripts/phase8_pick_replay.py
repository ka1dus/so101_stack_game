"""Phase 8: which box does the policy aim for on the real arm, and which input decides it? Pick-phase replay.

From the start of a recorded real trial to the grasp, every `--every` ticks the policy gets the real joint reading
plus an image pair and predicts its 100-step action chunk (no ensembling); the slot it aims for = the supply slot
nearest (xy) to the jaw tips at the chunk's lowest point. Image pairs (front, wrist):
  RR real (painted like the trial if it ran with --plain-wall), SS sim, SR sim front + real wrist, RS real front + sim
  wrist, RW real front with the sim room pasted in.
Sim images: the sim twin at the real reading with the trial's stack (perceived top) and supply (template minus stack
and set-aside boxes), nominal visuals.

    MUJOCO_GL=cgl ~/.venvs/so101_eval/bin/python scripts/phase8_pick_replay.py data/real_trials/20261028_203311 3
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from game.backends import SimBackend  # noqa: E402
from perception.drop import Kinematics  # noqa: E402
from perception.register import refine  # noqa: E402
from perception.supply import BOX  # noqa: E402
from perception.twin import Twin  # noqa: E402
from perception_eval import template_poses  # noqa: E402
from phase8_replay import set_box  # noqa: E402
from skill.plain_wall import PlainWall  # noqa: E402
from skill.runner import RESIZE, TASK, load_policy  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("session"); ap.add_argument("trial", type=int)
    ap.add_argument("--policy", default=str(ROOT / "data/policies/act_wcal_060000")); ap.add_argument("--device", default="mps")
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--ensemble", action="store_true",
                    help="instead: replay ticks 0..grasp through select_action (ensembling, as on the arm) per image pair; "
                         "report the largest gripper opening commanded before the grasp (RR checked against the log)")
    a = ap.parse_args(); sess = Path(a.session); kin = Kinematics()
    t = json.loads((sess / "trials.jsonl").read_text().splitlines()[a.trial - 1]); top = np.array(t["top_xy"])
    tr = np.load(sess / f"t{a.trial:03d}_trace.npy"); rq = tr[:, :6].astype(float)
    pol, pre, post = load_policy(a.policy, a.device)

    b = SimBackend(a.policy, a.device, seed=1); b.prepare_game(); g = b.game; s = g.sim; m, d = s.m, s.d; g.rand.reset()
    z = 0.0005
    for slot, typ in zip(t["stack_slots"], t["stack_types"]):
        set_box(m, d, b.body_of[slot], top, z + BOX[typ]["height"] / 2, np.radians(-30)); z += BOX[typ]["height"]
    for slot in t["aside_slots"]:
        set_box(m, d, b.body_of[slot], (0.0, 0.30), 0.02, 0.0)
    mujoco.mj_forward(m, d)
    free = [sl for sl in b.body_of if sl not in t["stack_slots"] + t["aside_slots"]]
    slot_xy = {sl: g.pose(b.body_of[sl])[0][:2].copy() for sl in free}

    vids = {}
    for cam in ("front", "wrist"):
        c = cv2.VideoCapture(str(sess / f"t{a.trial:03d}_{cam}.mp4")); fr = []
        while True:
            ok, im = c.read()
            if not ok:
                break
            fr.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        vids[cam] = fr
    painted = bool(t.get("plain_wall"))
    pw = None
    if painted:
        tw = Twin(); cam, _ = refine(cv2.imread(str(sess / "ref.png")), tw, template_poses(tw), rq[0].tolist()); pw = PlainWall(cam, every=1)

    seg = mujoco.Renderer(m, 480, 640); seg.enable_segmentation_rendering(); names = [m.geom(i).name for i in range(m.ngeom)]
    room = np.array([m.geom_bodyid[i] == 0 and not (names[i].startswith("table") or names[i] in ("stack_area", "supply_mat", "reach_line_tape"))
                     for i in range(m.ngeom)] + [True])

    def sim_frames(q):
        qs = g.jm.real_to_sim(q); d.qpos[s.qadr] = qs[:5]; d.qpos[s.gadr] = qs[5]; mujoco.mj_forward(m, d)
        _, f, w = b.arm.read()
        seg.update_scene(d, "front"); idx = seg.render()[..., 0]
        return f, w, room[np.where(idx < 0, m.ngeom, idx)]

    def aim(q, f, w):
        f = cv2.resize(f, RESIZE, interpolation=cv2.INTER_AREA); w = cv2.resize(w, RESIZE, interpolation=cv2.INTER_AREA)
        obs = {"observation.state": torch.from_numpy(q.astype(np.float32))[None],
               "observation.images.front": torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1)[None].float() / 255,
               "observation.images.wrist": torch.from_numpy(np.ascontiguousarray(w)).permute(2, 0, 1)[None].float() / 255, "task": [TASK]}
        pre.reset(); post.reset()
        with torch.inference_mode():
            c = pol.predict_action_chunk(pre(obs))
            acts = np.stack([post(c[:, i]).squeeze(0).float().cpu().numpy() for i in range(c.shape[1])])
        tips = np.array([kin.tip_pos(x.tolist()) for x in acts]); lo = tips[np.argmin(tips[:, 2])]
        return min(slot_xy, key=lambda sl: np.linalg.norm(slot_xy[sl] - lo[:2])), lo[2]

    tz = np.array([kin.tip_pos(x.tolist())[2] for x in rq]); kg = 60 + int(np.argmin(tz[60:240]))
    if a.ensemble:
        cmd = tr[:, 6:12].astype(float)
        print(f"{sess.name} t{a.trial}: grasp (lowest tip in 2-8 s) at {kg / 30:.1f} s; logged max opening before it {cmd[:kg, 5].max():.1f}")
        for p in ("RR", "SS", "SR", "RS", "RW"):
            pol.reset(); pre.reset(); post.reset(); gc = []
            for k in range(kg):
                q = rq[k]; fr, wr = vids["front"][k], vids["wrist"][k]
                fp = pw(fr, q) if pw else fr
                if p != "RR":
                    fs, ws, rm = sim_frames(q)
                    rmk = cv2.dilate(rm.astype(np.uint8), np.ones((5, 5), np.uint8))[..., None].astype(bool)
                    f, w = {"SS": (fs, ws), "SR": (fs, wr), "RS": (fp, ws), "RW": (np.where(rmk, fs, fr), wr)}[p]
                else:
                    f, w = fp, wr
                f = cv2.resize(f, RESIZE, interpolation=cv2.INTER_AREA); w = cv2.resize(w, RESIZE, interpolation=cv2.INTER_AREA)
                obs = {"observation.state": torch.from_numpy(q.astype(np.float32))[None],
                       "observation.images.front": torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1)[None].float() / 255,
                       "observation.images.wrist": torch.from_numpy(np.ascontiguousarray(w)).permute(2, 0, 1)[None].float() / 255, "task": [TASK]}
                with torch.inference_mode():
                    act = post(pol.select_action(pre(obs))).squeeze(0).float().cpu().numpy()
                gc.append(np.clip(act[5], q[5] - 10, q[5] + 10))
            gc = np.array(gc); extra = f" (vs logged: median |diff| {np.median(np.abs(gc - cmd[:kg, 5])):.2f})" if p == "RR" else ""
            print(f"  {p}: max opening commanded before the grasp {gc.max():.1f} at {np.argmax(gc) / 30:.1f} s; in the last 1 s {gc[-30:].max():.1f}{extra}", flush=True)
        return
    g_ = rq[:, 5]; grasp = int(np.argmax((g_ < g_.max() - 6) & (np.arange(len(g_)) > np.argmax(g_))))
    ticks = list(range(0, max(grasp - 15, 1), a.every))
    print(f"{sess.name} t{a.trial}: stack {t['stack_types']}, set aside {t['aside_slots']}, expected slot {t['expected_pick_slot']}, "
          f"real took {t.get('picked_slot')} ({t['outcome']}); front painted: {painted}; grasp at {grasp / 30:.1f} s; {len(ticks)} ticks")
    res = {p: [] for p in ("RR", "SS", "SR", "RS", "RW")}
    for k in ticks:
        q = rq[k]; fr, wr = vids["front"][k], vids["wrist"][k]; fs, ws, rm = sim_frames(q)
        fp = pw(fr, q) if pw else fr
        rmk = cv2.dilate(rm.astype(np.uint8), np.ones((5, 5), np.uint8))[..., None].astype(bool)
        for p, (f, w) in (("RR", (fp, wr)), ("SS", (fs, ws)), ("SR", (fs, wr)), ("RS", (fp, ws)), ("RW", (np.where(rmk, fs, fr), wr))):
            res[p].append(aim(q, f, w)[0])
        print(f"  {k / 30:4.1f} s: " + "  ".join(f"{p} {res[p][-1]}" for p in res), flush=True)
    for p, v in res.items():
        c = Counter(v)
        print(f"{p}: aims at {dict(c)} -> expected slot {t['expected_pick_slot']} in {c.get(t['expected_pick_slot'], 0)}/{len(v)}")


if __name__ == "__main__":
    main()
