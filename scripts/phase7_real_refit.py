"""Phase 7: the per-game camera refit (perception/register.py) on the real `front` frames from 2026-10-21.

Those frames predate the 2 cm mat expansion and the reach tape, so the twin used here gets the pre-expansion mat
outline and no tape (in memory only; sim/scene.xml is untouched). Checks:
  1. agreement: refit vs the hand-picked calibration points (undistorted), compared with the nominal calibration
  2. convergence: start the refit from cameras knocked +-1 cm / +-1.5 deg off and see whether it comes back
Overlays: notes/img/phase7_real_refit_<frame>.png

    MUJOCO_GL=cgl ~/.venvs/so101_stack_sim/bin/python scripts/phase7_real_refit.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from joint_mapping import JOINT_NAMES  # noqa: E402
from perception.camera import CAMS, Camera, H, W  # noqa: E402
from perception.register import _perturb, refine  # noqa: E402
from perception.twin import Twin  # noqa: E402

LAY = yaml.safe_load((ROOT / "sim/assets/layout.yaml").read_text())
MEAS = yaml.safe_load((ROOT / "inputs/measurements.yaml").read_text())
HOME = [MEAS["arm"]["home_pose"][j] for j in JOINT_NAMES]
FRAMES = ["front_cam_frame.png", "front_cam_stack3.png"]


def undistort_img(img):
    c = CAMS["front"]; K = np.array([[c["f_px"], 0, W / 2], [0, c["f_px"], H / 2], [0, 0, 1]])
    return cv2.undistort(img, K, np.array([c["k1"], c.get("k2", 0.0), 0, 0]))


def undistort_pts(uv):
    c = CAMS["front"]; K = np.array([[c["f_px"], 0, W / 2], [0, c["f_px"], H / 2], [0, 0, 1]])
    return cv2.undistortPoints(np.asarray(uv, np.float64).reshape(-1, 1, 2), K, np.array([c["k1"], c.get("k2", 0.0), 0, 0]), P=K).reshape(-1, 2)


def pre_expansion_twin():
    tw = Twin(); m = tw.m
    S = np.array(LAY["supply_area_pre_expansion"]["corners"]); c = S.mean(0)
    g = m.geom("supply_mat").id
    e1 = S[1] - S[0]; e2 = S[3] - S[0]
    m.geom_pos[g][:2] = c; m.geom_size[g][:2] = [np.linalg.norm(e1) / 2, np.linalg.norm(e2) / 2]
    yaw = np.arctan2(e1[1], e1[0]); m.geom_quat[g] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    m.geom_pos[m.geom("reach_line_tape").id] = [5, 5, -1]                     # not there yet on 21 Oct
    return tw


def main():
    tw = pre_expansion_twin(); nom = Camera()
    P = CAMS["front"]["points"]
    K3 = np.array(LAY["stack_area"]["corners"]); S3 = np.array(LAY["supply_area_pre_expansion"]["corners"])
    world = {f"stack_corner_{i+1}": np.r_[K3[i], 0.0002] for i in range(4)}
    world.update({f"supply_corner_{i+1}": np.r_[S3[i], 0.002] for i in range(4)})
    names = list(world); picked = undistort_pts([P[n] for n in names]); Wp = np.array([world[n] for n in names])
    rng = np.random.default_rng(0); lines = []
    for fr in FRAMES:
        img = undistort_img(cv2.imread(str(ROOT / "inputs/photos" / fr)))
        cam, info = refine(img, tw, {}, HOME, cam=nom)
        e_nom = np.linalg.norm(nom.project(Wp) - picked, axis=1); e_ref = np.linalg.norm(cam.project(Wp) - picked, axis=1)
        lines.append(f"{fr}: refit moved {info['delta_mm']} mm / {info['delta_deg']} deg; mask cost {info['cost_before']} -> "
                     f"{info['cost_after']}")
        lines.append(f"  picked corners: nominal rms {np.sqrt((e_nom ** 2).mean()):.1f} px (max {e_nom.max():.1f}); "
                     f"refit rms {np.sqrt((e_ref ** 2).mean()):.1f} px (max {e_ref.max():.1f})")
        diffs = []
        for k in range(4):                               # convergence from knocked-off starting cameras
            p = np.r_[rng.uniform(-1.5, 1.5, 3), rng.uniform(-10, 10, 3)]
            c2, _ = refine(img, tw, {}, HOME, cam=_perturb(nom, p))
            diffs.append(np.linalg.norm(c2.project(Wp) - cam.project(Wp), axis=1).max())
        lines.append(f"  from 4 knocked-off starts (+-1.5 deg, +-1 cm): worst corner differs from the refit by "
                     f"{', '.join(f'{d:.1f}' for d in diffs)} px")
        ov = img.copy()
        for n, uv in zip(names, picked):
            cv2.drawMarker(ov, tuple(np.round(uv).astype(int)), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
        for camx, col in ((nom, (255, 0, 0)), (cam, (0, 0, 255))):
            for Q in (K3, S3):
                pts = camx.project(np.c_[Q, np.zeros(4)]).round().astype(np.int32)
                cv2.polylines(ov, [pts], True, col, 1)
        cv2.putText(ov, "green: picked  blue: nominal  red: refit", (10, 20), 0, 0.5, (255, 255, 255), 1)
        out = ROOT / f"notes/img/phase7_real_refit_{Path(fr).stem}.png"; cv2.imwrite(str(out), ov)
        lines.append(f"  overlay: {out.relative_to(ROOT)}")
    txt = "\n".join(lines); print(txt); (ROOT / "notes/phase7_real_refit.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
