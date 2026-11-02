"""Phase 8: how far the sim wrist camera is from the real one, measured on a real trial's wrist video.

For ticks of the trial (the wrist video frame k = trace tick k), the sim twin is set to the real joint reading, with
the template supply, the trial's stack at the perceived top and the nominal visuals, and the wrist camera is rendered
raw. The yellow stack area and the black supply mat are segmented in both images (HSV); per frame, the shift (and
roll) of the sim masks that best matches the real ones is searched. Reports IoU before/after and the shift in px and
degrees (f 518 px). Frames after the grasp are skipped (the held box is not in the sim scene).

    MUJOCO_GL=cgl ~/.venvs/so101_eval/bin/python scripts/phase8_wrist_gap.py data/real_trials/20261028_105812 1 [mount.json]
(optional mount.json from scripts/wrist_calib_fit.py: render with that wrist mount instead of the scene's)
"""
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from game.backends import SimBackend  # noqa: E402
from perception.supply import BOX  # noqa: E402
from phase8_replay import set_box  # noqa: E402

F_PX = 518.0


def masks(bgr):
    h = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    yellow = ((h[..., 0] >= 15) & (h[..., 0] <= 40) & (h[..., 1] > 90) & (h[..., 2] > 90))
    dark = h[..., 2] < 55
    return yellow.astype(np.uint8), dark.astype(np.uint8)


def iou(a, b):
    u = np.logical_or(a, b).sum(); return np.logical_and(a, b).sum() / u if u else np.nan


def best_shift(sim, real, rng=90, step=3, rolls=(-8, -4, 0, 4, 8)):
    H, W = real.shape; best = (iou(sim, real), 0, 0, 0)
    for r in rolls:
        R = cv2.getRotationMatrix2D((W / 2, H / 2), r, 1.0); sr = cv2.warpAffine(sim, R, (W, H), flags=cv2.INTER_NEAREST)
        for dy in range(-rng, rng + 1, step):
            for dx in range(-rng, rng + 1, step):
                M = np.float32([[1, 0, dx], [0, 1, dy]]); s = cv2.warpAffine(sr, M, (W, H), flags=cv2.INTER_NEAREST)
                v = iou(s, real)
                if v > best[0]:
                    best = (v, dx, dy, r)
    return best


def main():
    sess, n = Path(sys.argv[1]), int(sys.argv[2]); mount = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else None
    t = json.loads((sess / "trials.jsonl").read_text().splitlines()[n - 1]); tr = np.load(sess / f"t{n:03d}_trace.npy")
    rq = tr[:, :6].astype(float); g = rq[:, 5]; grasp = int(np.argmax((g < g.max() - 6) & (np.arange(len(g)) > np.argmax(g))))
    b = SimBackend(str(ROOT / "data/policies/act_softB_080000"), "cpu", seed=1); b.prepare_game(); G = b.game; s = G.sim; m, d = s.m, s.d
    G.rand.reset(); top = np.array(t["top_xy"]); z = 0.0005
    if mount:                                                                   # a wrist mount from scripts/wrist_calib_fit.py
        c = m.camera("wrist").id; m.cam_pos[c] = mount["cam_pos"]; m.cam_quat[c] = mount["cam_quat"]; m.cam_fovy[c] = mount["fovy_deg"]
    for slot, typ in zip(t["stack_slots"], t["stack_types"]):
        set_box(m, d, b.body_of[slot], top, z + BOX[typ]["height"] / 2, np.radians(-30)); z += BOX[typ]["height"]
    for slot in t["aside_slots"]:
        set_box(m, d, b.body_of[slot], (0.0, 0.30), 0.02, 0.0)
    ren = mujoco.Renderer(m, 480, 640); cap = cv2.VideoCapture(str(sess / f"t{n:03d}_wrist.mp4")); frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    ticks = list(range(0, max(grasp - 10, 1), 6)); rows = []; out = ROOT / "notes/img"
    print(f"trial {n}: {len(frames)} wrist frames, grasp at {grasp / 30:.1f} s; checking {len(ticks)} frames before it")
    for k in ticks:
        q = G.jm.real_to_sim(rq[k]); d.qpos[s.qadr] = q[:5]; d.qpos[s.gadr] = q[5]; mujoco.mj_forward(m, d)
        ren.update_scene(d, "wrist"); sim = cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR); real = frames[k]
        ys, ds = masks(sim); yr, dr = masks(real)
        both = ((ys | ds) > 0).astype(np.uint8) * 1 + 0; bothr = ((yr | dr) > 0).astype(np.uint8)
        lab_s = ys * 1 + ds * 2; lab_r = yr * 1 + dr * 2
        v0y, v0d = iou(ys, yr), iou(ds, dr)
        bv, dx, dy, r = best_shift((lab_s == 1).astype(np.uint8) + 2 * (lab_s == 2).astype(np.uint8), lab_r) if False else best_shift(ys | ds, yr | dr)
        rows.append((k, v0y, v0d, bv, dx, dy, r))
        print(f"  t {k / 30:4.1f} s: IoU yellow {v0y:.2f} dark {v0d:.2f} | best shift dx {dx:+d} dy {dy:+d} px, roll {r:+d} deg -> IoU {bv:.2f}", flush=True)
        if k in (ticks[0], ticks[len(ticks) // 2], ticks[-1]):
            cv2.imwrite(str(out / f"phase8_wristgap_{sess.name}_t{n:03d}_{k:03d}{'_newmount' if mount else ''}.png"), np.hstack([real, sim, cv2.addWeighted(real, 0.5, sim, 0.5, 0)]))
    a = np.array(rows, float)
    print(f"median: shift dx {np.median(a[:, 4]):+.0f} dy {np.median(a[:, 5]):+.0f} px "
          f"(~{np.median(a[:, 4]) / F_PX * 57.3:+.1f} / {np.median(a[:, 5]) / F_PX * 57.3:+.1f} deg), roll {np.median(a[:, 6]):+.0f} deg; "
          f"IoU {np.nanmedian(a[:, 1:3].mean(1)):.2f} -> {np.median(a[:, 3]):.2f}")


if __name__ == "__main__":
    main()
