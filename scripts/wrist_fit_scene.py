"""Phase 3: wrist camera orientation + position from jaw tips AND table features in P1/P3.

The jaw tips alone leave the pitch ambiguous (they sit near the image bottom). In P1/P3 the
lens is ~15 cm above the yellow area, so the original stacking X and the yellow area's
table-corner vertex constrain the pitch (arm FK error ~1 cm -> a few degrees). Lens held to
the measured position (+/- 1 cm), f = 518 px (datasheet), k1 = 0.

    MUJOCO_GL=cgl ~/.venvs/so101_stack_sim/bin/python scripts/wrist_fit_scene.py [--save]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
import calibrate_cameras as C  # noqa: E402
import wrist_from_measurement as WM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRIOR_M = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--prior=")), "0.005"))   # lens-position sigma [m]
# --measured: per-component sigmas from the user's measurements (in-plane x, out-of-plane y, along-jaws z):
# in-plane 2.7 cm (+/- 1 cm), out-of-plane 2-3 cm (+/- 0.5 cm, re-confirmed 2026-10-21), along 7.5-8 cm (+/- 0.5 cm)
PRIOR_SIGMA = np.array([0.010, 0.005, 0.005]) if "--measured" in sys.argv else np.full(3, PRIOR_M)


def main():
    G = C.gripper_poses(); P = C.wrist_world_points()
    tipf = G["home"][2]["tip_fixed"]
    c_meas = tipf + np.array([WM.MEASURED["side"], WM.MEASURED["out"], WM.MEASURED["back"]])
    obs = [("home", "tip_fixed", (318, 322), 1.0), ("home", "tip_moving", (352, 333), 1.0),
           ("P1", "tip_fixed", (318, 322), 0.5), ("P3", "tip_fixed", (320, 325), 0.5),
           ("P1", "stacking_spot", (322, 237), 0.25), ("P1", "stack_corner_4", (612, 97), 0.25),
           ("P3", "stacking_spot", (262, 193), 0.25), ("P3", "stack_corner_4", (528, 276), 0.25)]
    pts = []
    for frame, key, uv, w in obs:
        Rg, xg, tips = G[frame]
        pg = tips[key] if key.startswith("tip_") else (P[key] - xg) @ Rg
        pts.append((frame, key, pg, np.array(uv, float), w))

    def res(p):
        R, _ = cv2.Rodrigues(p[:3]); r = []
        for _, _, pg, uv, w in pts:
            pc = R.T @ (pg - p[3:6])
            r += list((WM.F_PX * pc[:2] / pc[2] + [C.W / 2, C.H / 2] - uv) * w) if pc[2] > 0 else [1e3, 1e3]
        r += list((p[3:6] - c_meas) / PRIOR_SIGMA)
        return np.array(r)

    best = None
    cams = yaml.safe_load(C.OUT.read_text())
    starts = [np.r_[cams["wrist"]["rvec_in_gripper"], cams["wrist"]["pos"]]]
    to_tip = tipf - c_meas
    for roll0 in np.radians(np.arange(0, 360, 30)):
        for pitch in np.radians((-40, -20, 0, 20, 40)):
            z = to_tip / np.linalg.norm(to_tip)
            a = np.cross(z, [0, 0, 1.0]); a /= np.linalg.norm(a); b = np.cross(z, a)
            x = np.cos(roll0) * a + np.sin(roll0) * b; y = np.cross(z, x)
            Rp = cv2.Rodrigues(x * pitch)[0]                     # tilt the axis about image x
            starts.append(np.r_[cv2.Rodrigues(Rp @ np.c_[x, y, z])[0].ravel(), c_meas])
    for s in starts:
        sol = least_squares(res, s)
        if best is None or sol.cost < best.cost:
            best = sol
    p = best.x; R, _ = cv2.Rodrigues(p[:3])
    err = {f"{f}:{k}": round(float(np.linalg.norm(rr)), 1) for (f, k, *_), rr in zip(pts, (res(p)[:2 * len(pts)] /
           np.repeat([w for *_, w in pts], 2)).reshape(-1, 2))}
    print("pixel errors:", err)
    dd = p[3:6] - tipf
    print(f"lens from the fixed tip: back {dd[2]*100:.1f} cm (measured 7.5-8), in-plane {dd[0]*100:+.1f} cm (measured -2.7), "
          f"out-of-plane {dd[1]*100:+.1f} cm (measured +2.0)")
    print(f"lens {np.linalg.norm(p[3:6]-c_meas)*100:.1f} cm from the measurement; optical axis "
          f"{np.degrees(np.arccos(np.clip(R[:,2]@[0,0,-1.0],-1,1))):.1f} deg off the jaws")
    print("jaw IoU (check):", WM.silhouette_iou(np.r_[p, WM.F_PX, 0.0]))
    if "--save" in sys.argv:
        w = cams["wrist"]
        w.update(pos=p[3:6].round(5).tolist(), xyaxes=np.r_[R[:, 0], -R[:, 1]].round(5).tolist(), rvec_in_gripper=p[:3].tolist(),
                 tip_and_scene_err_px=err, method="measured lens + datasheet f; orientation from jaw tips + P1/P3 table features "
                 "(scripts/wrist_fit_scene.py)", status="PROVISIONAL: k1 pending checkerboard")
        C.OUT.write_text(yaml.safe_dump(cams, sort_keys=False)); print("saved")


if __name__ == "__main__":
    main()
