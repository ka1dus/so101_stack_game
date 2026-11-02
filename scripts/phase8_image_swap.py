"""Phase 8: which camera makes the policy hover over the stack on the real arm? Image-swap test on a recorded trial.

At ticks of a real trial where the jaw tips are over the stack, the policy is given the real joint reading plus one of
four image pairs and its predicted action chunk is read (no motion, no ensembling):
  RR real front + real wrist (what it saw), SS sim front + sim wrist, RS real front + sim wrist, SR sim front + real wrist.
The sim images are rendered at the same real reading, with the trial's stack at the perceived top and the held box in
the jaws at the pose it has in a closed-loop sim run of the same scene (first tick with the tips within 6 cm of the
top), through the same randomizer post-processing as in training (nominal settings).
Per tick and pair: the lowest jaw-tip height over the first second of the chunk and whether the chunk opens the
gripper (command > held reading + 3 units) within its 100 steps (3.3 s).

    MUJOCO_GL=cgl ~/.venvs/so101_eval/bin/python scripts/phase8_image_swap.py data/real_trials/20261028_172237 1 \
        --policy data/policies/act_wcal_060000
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from game.backends import SimBackend  # noqa: E402
from joint_mapping import JOINT_NAMES  # noqa: E402
from perception.drop import Kinematics  # noqa: E402
from perception.supply import BOX  # noqa: E402
from phase8_replay import set_box  # noqa: E402
from skill.runner import RESIZE, TASK, load_policy, run_place_skill  # noqa: E402


def build(b, t, top):
    m, d = b.game.sim.m, b.game.sim.d; z = 0.0005
    for slot, typ in zip(t["stack_slots"], t["stack_types"]):
        set_box(m, d, b.body_of[slot], top, z + BOX[typ]["height"] / 2, np.radians(-30)); z += BOX[typ]["height"]
    for slot in t["aside_slots"]:
        set_box(m, d, b.body_of[slot], (0.0, 0.30), 0.02, 0.0)
    return z


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("session"); ap.add_argument("trial", type=int)
    ap.add_argument("--policy", default=str(ROOT / "data/policies/act_wcal_060000")); ap.add_argument("--device", default="mps")
    ap.add_argument("--every", type=int, default=30)
    ap.add_argument("--regions", action="store_true",
                    help="also: real front with the sim pixels pasted in one region at a time (wall, arm, boxes, table), and the banner area painted plain")
    ap.add_argument("--plain", action="store_true", help="also: real front through skill.plain_wall.PlainWall (session camera refit from ref.png)")
    ap.add_argument("--ensemble", action="store_true",
                    help="replay the whole trial open-loop through select_action (temporal ensembling, as on the arm), "
                         "with the front image changed only while the tips are over the stack; first checks RR against the logged commands")
    ap.add_argument("--variants", default="RR,SR,RS,SS,R+wall,R+banner_blue,R+arm,R+boxes")
    ap.add_argument("--banner-row", type=int, default=150, help="image row where the banner ends (real front)")
    a = ap.parse_args(); sess = Path(a.session); kin = Kinematics()
    t = json.loads((sess / "trials.jsonl").read_text().splitlines()[a.trial - 1]); top = np.array(t["top_xy"])
    tr = np.load(sess / f"t{a.trial:03d}_trace.npy"); rq = tr[:, :6].astype(float)
    policy = load_policy(a.policy, a.device); pol, pre, post = policy

    # 1. held-box pose in the gripper frame, from a closed-loop sim run of the same scene
    b = SimBackend(a.policy, a.device, seed=20000); b.prepare_game(); g = b.game; s = g.sim; m, d = s.m, s.d
    g.rand.reset(); z_top = build(b, t, top)
    q0 = g.jm.real_to_sim(rq[0]); d.qpos[s.qadr] = q0[:5]; d.qpos[s.gadr] = q0[5]
    for j, v in zip(JOINT_NAMES, q0):
        d.ctrl[s.act[j]] = v
    mujoco.mj_forward(m, d); s.sync_plan(); s.wait(1.0)
    box = b.body_of[t["picked_slot"] if t.get("picked_slot") is not None else t["expected_pick_slot"]]; grip = m.body("gripper").id; rel = []

    def tick(_f, q):
        if not rel and np.linalg.norm(kin.tip_pos(list(q))[:2] - top) < 0.06:
            pb, Rb = g.pose(box); pg, Rg = d.xpos[grip].copy(), d.xmat[grip].reshape(3, 3).copy()
            rel.append((Rg.T @ (pb - pg), Rg.T @ Rb, q[5]))
    run_place_skill(b.arm, policy, kin, top, z_top, on_tick=tick)
    if not rel:
        print("the sim run never reached the stack"); return
    p_rel, R_rel, g_sim = rel[0]
    print(f"sim held box in the gripper frame: {np.round(p_rel * 100, 1)} cm; sim gripper reading {g_sim:.1f}")

    # 2. real frames
    vids = {}
    for cam in ("front", "wrist"):
        c = cv2.VideoCapture(str(sess / f"t{a.trial:03d}_{cam}.mp4")); fr = []
        while True:
            ok, im = c.read()
            if not ok:
                break
            fr.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        vids[cam] = fr
    tips = np.array([kin.tip_pos(x.tolist()) for x in rq]); near = np.flatnonzero(np.linalg.norm(tips[:, :2] - top, axis=1) < 0.06)
    ticks = [k for k in near[::a.every] if k < len(vids["wrist"])]
    print(f"real: {len(near)} ticks with the tips over the stack; testing {len(ticks)}; stack top face {z_top * 100:.1f} cm")

    # 3. sim frames at the real readings, held box placed in the jaws
    a_box = m.jnt_qposadr[m.body(box).jntadr[0]]
    def sim_frames(q):
        qs = g.jm.real_to_sim(q); d.qpos[s.qadr] = qs[:5]; d.qpos[s.gadr] = qs[5]; mujoco.mj_forward(m, d)
        pg, Rg = d.xpos[grip], d.xmat[grip].reshape(3, 3); R = Rg @ R_rel; qu = np.zeros(4); mujoco.mju_mat2Quat(qu, R.flatten())
        d.qpos[a_box:a_box + 3] = pg + Rg @ p_rel; d.qpos[a_box + 3:a_box + 7] = qu; mujoco.mj_forward(m, d)
        _, f, w = b.arm.read(); return f, w

    seg = mujoco.Renderer(m, 480, 640); seg.enable_segmentation_rendering()
    gb = m.geom_bodyid; names = [m.geom(i).name for i in range(m.ngeom)]
    boxes = {m.body(n).id for n in b.body_of.values()}
    cat = np.full(m.ngeom + 1, 0)                                   # 0 wall/room/backdrop (and no geom)
    for i in range(m.ngeom):
        if gb[i] in boxes:
            cat[i] = 3
        elif gb[i] != 0:
            cat[i] = 2                                             # arm (every robot body)
        elif names[i].startswith("table") or names[i] in ("stack_area", "supply_mat", "reach_line_tape"):
            cat[i] = 4
    REG = {"wall": 0, "arm": 2, "boxes": 3, "table": 4}

    def held_mask():
        seg.update_scene(d, "front"); idx = seg.render()[..., 0]
        return np.isin(idx, [g for g in range(m.ngeom) if gb[g] == m.body(box).id])

    def regions():
        seg.update_scene(d, "front"); idx = seg.render()[..., 0]
        return cat[np.where(idx < 0, m.ngeom, idx)]

    def chunk(q, f, w):
        f = cv2.resize(f, RESIZE, interpolation=cv2.INTER_AREA); w = cv2.resize(w, RESIZE, interpolation=cv2.INTER_AREA)
        obs = {"observation.state": torch.from_numpy(q.astype(np.float32))[None],
               "observation.images.front": torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1)[None].float() / 255,
               "observation.images.wrist": torch.from_numpy(np.ascontiguousarray(w)).permute(2, 0, 1)[None].float() / 255, "task": [TASK]}
        pre.reset(); post.reset()
        with torch.inference_mode():
            c = pol.predict_action_chunk(pre(obs))
            acts = np.stack([post(c[:, i]).squeeze(0).float().cpu().numpy() for i in range(c.shape[1])])
        tz = np.array([kin.tip_pos(x.tolist())[2] for x in acts[:30]])
        txy = kin.tip_pos(acts[29].tolist())[:2]
        op = np.flatnonzero(acts[:, 5] > q[5] + 3)
        return tz.min(), np.linalg.norm(txy - top), (op[0] / 30 if len(op) else None)

    if a.ensemble:
        tr_cmd = tr[:, 6:12].astype(float); hov = set(int(x) for x in near); n_t = min(len(rq), len(vids["front"]))
        held = rq[near[0], 5]

        pwall = None
        if "R+plainwall" in a.variants:                              # the module the trial script uses (--plain-wall)
            from perception.register import refine
            from perception.twin import Twin
            from perception_eval import template_poses
            from skill.plain_wall import PlainWall
            tw = Twin(); cam_s, _ = refine(cv2.imread(str(sess / "ref.png")), tw, template_poses(tw), rq[0].tolist())
            pwall = PlainWall(cam_s); print(f"PlainWall colour (RGB) {pwall.colour.tolist()}")

        def variant(name, k, fr, wr):
            if name == "R+plainwall":                                # applied on every tick, as on the arm
                return pwall(fr, rq[k]), wr
            if name == "RR" or k not in hov:
                return fr, wr
            q = rq[k]; fs, ws = sim_frames(q)
            if name == "SR":
                return fs, wr
            if name == "RS":
                return fr, ws
            if name == "SS":
                return fs, ws
            lab = regions(); c = REG[name[2:]] if name[2:] in REG else None
            if c is not None:
                mk = cv2.dilate((lab == c).astype(np.uint8), np.ones((5, 5), np.uint8))[..., None].astype(bool)
                return np.where(mk, fs, fr), wr
            if name in ("R+wall_top", "R+wall_low"):                 # sim room above / below the banner's bottom row
                rows = np.arange(480)[:, None] < a.banner_row
                mk = (lab == 0) & (rows if name == "R+wall_top" else ~rows)
                mk = cv2.dilate(mk.astype(np.uint8), np.ones((5, 5), np.uint8))[..., None].astype(bool)
                return np.where(mk, fs, fr), wr
            if name == "R+arm+boxes":
                mk = cv2.dilate(((lab == 2) | (lab == 3)).astype(np.uint8), np.ones((5, 5), np.uint8))[..., None].astype(bool)
                return np.where(mk, fs, fr), wr
            if name == "R+heldbox":                                 # only the held box (sim box pixels above the stack top)
                hb = (lab == 3) & held_mask()
                mk = cv2.dilate(hb.astype(np.uint8), np.ones((5, 5), np.uint8))[..., None].astype(bool)
                return np.where(mk, fs, fr), wr
            f2 = fr.copy(); f2[(lab == 0) & (np.arange(480)[:, None] < a.banner_row)] = (25, 30, 70)   # R+banner_blue
            return f2, wr

        for name in a.variants.split(","):
            pol.reset(); pre.reset(); post.reset(); g_cmd = []; z_cmd = []
            for k in range(n_t):
                f, w = variant(name, k, vids["front"][k], vids["wrist"][k])
                f = cv2.resize(f, RESIZE, interpolation=cv2.INTER_AREA); w = cv2.resize(w, RESIZE, interpolation=cv2.INTER_AREA)
                obs = {"observation.state": torch.from_numpy(rq[k].astype(np.float32))[None],
                       "observation.images.front": torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1)[None].float() / 255,
                       "observation.images.wrist": torch.from_numpy(np.ascontiguousarray(w)).permute(2, 0, 1)[None].float() / 255, "task": [TASK]}
                with torch.inference_mode():
                    act = post(pol.select_action(pre(obs))).squeeze(0).float().cpu().numpy()
                g_cmd.append(np.clip(act[5], rq[k, 5] - 10, rq[k, 5] + 10)); z_cmd.append(kin.tip_pos(np.r_[act[:5], rq[k, 5]].tolist())[2])
            g_cmd = np.array(g_cmd); hv = np.array(sorted(h for h in hov if h < n_t))
            opened = np.flatnonzero(g_cmd[hv] > held + 3)
            z_cmd = np.array(z_cmd)
            msg = f"{name:14s} commanded tip height over the stack: median {np.median(z_cmd[hv]) * 100:.1f} cm, lowest {z_cmd[hv].min() * 100:.1f} cm | gripper command: max {g_cmd[hv].max():5.1f} (held {held:.1f}); "
            msg += f"opens at {hv[opened[0]] / 30:.1f} s" if len(opened) else "never opens"
            if name == "RR":
                msg += f" | vs logged commands: median |diff| {np.median(np.abs(g_cmd - tr_cmd[:n_t, 5])):.2f}, logged max over stack {tr_cmd[hv, 5].max():.1f}"
            print(msg, flush=True)
        return

    pw = None
    if a.plain:
        import time
        from perception.register import refine
        from perception.twin import Twin
        from perception_eval import template_poses
        from skill.plain_wall import PlainWall
        tw = Twin(); home = rq[0]
        cam, _ = refine(cv2.imread(str(sess / "ref.png")), tw, template_poses(tw), home.tolist())
        pw = PlainWall(cam, banner_row=a.banner_row); dts = []
    rows = []; out = ROOT / "notes/img"
    print(f"{'t (s)':>6} {'tip z':>6} | " + " | ".join(f"{p}: low z, off, opens" for p in ("RR", "SS", "RS", "SR")))
    for n, k in enumerate(ticks):
        q = rq[k]; fr, wr = vids["front"][k], vids["wrist"][k]; fs, ws = sim_frames(q)
        r = {p: chunk(q, *im) for p, im in (("RR", (fr, wr)), ("SS", (fs, ws)), ("RS", (fr, ws)), ("SR", (fs, wr)))}
        if pw is not None:
            t0 = time.perf_counter(); fp = pw(fr, q); dts.append(time.perf_counter() - t0)
            r["R+plain"] = chunk(q, fp, wr)
            if n == len(ticks) // 2:
                cv2.imwrite(str(out / f"phase8_plainwall_{sess.name}_t{a.trial:03d}_{k:03d}.png"), cv2.cvtColor(fp, cv2.COLOR_RGB2BGR))
        if a.regions:
            lab = regions()
            for name, c in REG.items():
                mk = cv2.dilate((lab == c).astype(np.uint8), np.ones((5, 5), np.uint8))[..., None].astype(bool)
                r["R+" + name] = chunk(q, np.where(mk, fs, fr), wr)
            armz = cv2.dilate((lab == 2).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
            hsv = cv2.cvtColor(fr, cv2.COLOR_RGB2HSV)
            red = ((hsv[..., 0] < 10) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 60)
            white = (hsv[..., 1] < 50) & (hsv[..., 2] > 150)
            cab = cv2.dilate(((red | white) & armz).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            f2 = fr.copy(); f2[cab] = (20, 20, 22); r["R-cables"] = chunk(q, f2, wr)  # red/white servo cables painted black
            if n == len(ticks) // 2:
                cv2.imwrite(str(out / f"phase8_swap_nocables_{sess.name}_t{a.trial:03d}_{k:03d}.png"), cv2.cvtColor(f2, cv2.COLOR_RGB2BGR))
            wall = (lab == 0)
            ys = np.arange(480)[:, None] < a.banner_row
            for name, col in (("banner_black", (22, 22, 24)), ("banner_blue", (25, 30, 70))):
                f2 = fr.copy(); f2[wall & ys] = col                  # banner area painted plain (the logo covered)
                r["R+" + name] = chunk(q, f2, wr)
            f2 = fr.copy(); f2[wall] = cv2.GaussianBlur(fr, (0, 0), 25)[wall]
            r["R+wall_blur"] = chunk(q, f2, wr)
            if n == len(ticks) // 2:
                f2 = fr.copy(); f2[wall & ys] = (22, 22, 24)
                cv2.imwrite(str(out / f"phase8_swap_banner_{sess.name}_t{a.trial:03d}_{k:03d}.png"), cv2.cvtColor(f2, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(out / f"phase8_swap_regions_{sess.name}_t{a.trial:03d}_{k:03d}.png"),
                            cv2.applyColorMap((lab * 60).astype(np.uint8), cv2.COLORMAP_JET))
        rows.append(r)
        cell = lambda v: f"{v[0] * 100:4.1f} {v[1] * 100:4.1f} {('%.1fs' % v[2]) if v[2] is not None else ' no '}"
        print(f"{k / 30:6.1f} {tips[k, 2] * 100:6.1f} | " + " | ".join(f"{p}: {cell(r[p])}" for p in r), flush=True)
        if n in (0, len(ticks) // 2):
            cv2.imwrite(str(out / f"phase8_swap_{sess.name}_t{a.trial:03d}_{k:03d}.png"),
                        cv2.cvtColor(np.vstack([np.hstack([fr, wr]), np.hstack([fs, ws])]), cv2.COLOR_RGB2BGR))
    if pw is not None:
        print(f"PlainWall: {np.median(dts) * 1000:.1f} ms per frame (median)")
    for p in rows[0]:
        v = [r[p] for r in rows]
        print(f"{p}: median lowest tip {np.median([x[0] for x in v]) * 100:.1f} cm, median xy off {np.median([x[1] for x in v]) * 100:.1f} cm, "
              f"opens in {sum(x[2] is not None for x in v)}/{len(v)} chunks")


if __name__ == "__main__":
    main()
