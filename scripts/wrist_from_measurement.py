"""Phase 3: wrist camera from the measured lens position (user, 2026-10-21).

Lens centre relative to the FIXED jaw tip, in the gripper frame (jaws along -z, moving jaw
toward +x, jaw-opening plane = x-z):
  7.5-8 cm back along the jaws (+z), 2.7 cm in-plane away from the moving jaw (-x),
  ~2 cm out of the jaw plane (+y or -y; both tried, the official mount is on +y).
Intrinsics: f = 518 px (Orbbec Gemini 2 datasheet, RGB 4:3 FOV 63 x 50 deg at 640x480),
k1 = 0 until the checkerboard calibration. Only the orientation is fitted, from the jaw-tip
pixels (home: both tips; P1/P3: fixed tip), with a +/- 1 cm prior on the measured position.
The jaw silhouettes (IoU) are the independent check.

    MUJOCO_GL=cgl ~/.venvs/so101_stack_sim/bin/python scripts/wrist_from_measurement.py [--save]
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
import calibrate_cameras as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
F_PX = 518.0
MEASURED = dict(back=0.0775, side=-0.027, out=0.025)    # out-of-plane re-confirmed 2-3 cm (user, 2026-10-21)    # metres, relative to the fixed jaw tip


def main():
    G = C.gripper_poses()
    tipf = G["home"][2]["tip_fixed"]; tipm = G["home"][2]["tip_moving"]
    obs = [(tipf, (318, 322), 1.0), (tipm, (352, 333), 1.0),
           (G["P1"][2]["tip_fixed"], (318, 322), 0.5), (G["P3"][2]["tip_fixed"], (320, 325), 0.5)]

    def proj(rv, c, pg):
        R, _ = cv2.Rodrigues(rv)
        pc = R.T @ (pg - c)
        return F_PX * pc[:2] / pc[2] + np.array([C.W / 2, C.H / 2]), pc[2]

    results = []
    for out_sign in (+1, -1):
        c_meas = tipf + np.array([MEASURED["side"], out_sign * MEASURED["out"], MEASURED["back"]])
        to_tip = tipf - c_meas
        best = None
        for roll0 in np.radians(np.arange(0, 360, 45)):
            z = to_tip / np.linalg.norm(to_tip)                 # start looking at the fixed tip
            a = np.cross(z, [0, 0, 1.0]); a /= np.linalg.norm(a); b = np.cross(z, a)
            x = np.cos(roll0) * a + np.sin(roll0) * b; y = np.cross(z, x)
            rv0 = cv2.Rodrigues(np.c_[x, y, z])[0].ravel()
            def res(p):
                r = []
                for pg, uv, w in obs:
                    pix, depth = proj(p[:3], p[3:6], pg)
                    r += list((pix - uv) * w) if depth > 0 else [1e3, 1e3]
                r += list((p[3:6] - c_meas) / 0.01 * 2.0)       # +/- 1 cm on the measured lens position
                return np.array(r)
            sol = least_squares(res, np.r_[rv0, c_meas])
            if best is None or sol.cost < best.cost:
                best = sol
        p = best.x
        # silhouette check (render with f fixed, k1 = 0)
        cam_p = np.r_[p[:3], p[3:6], F_PX, 0.0]
        ious = silhouette_iou(cam_p)
        R, _ = cv2.Rodrigues(p[:3])
        tip_err = [float(np.linalg.norm(proj(p[:3], p[3:6], pg)[0] - uv)) for pg, uv, _ in obs]
        results.append(dict(out_sign=out_sign, p=p, ious=ious, tip_err_px=np.round(tip_err, 1).tolist(),
                            moved_cm=float(np.linalg.norm(p[3:6] - c_meas) * 100),
                            axis_vs_jaw_deg=float(np.degrees(np.arccos(np.clip(R[:, 2] @ [0, 0, -1.0], -1, 1))))))
        r = results[-1]
        print(f"out-of-plane {'+y' if out_sign > 0 else '-y'}: tip errors {r['tip_err_px']} px, lens moved {r['moved_cm']:.1f} cm "
              f"from the measurement, optical axis {r['axis_vs_jaw_deg']:.0f} deg off the jaws, jaw IoU {r['ious']}")
    best = max(results, key=lambda r: np.mean(list(r["ious"].values())))
    if "--save" in sys.argv:
        p = best["p"]; R, _ = cv2.Rodrigues(p[:3])
        cams = yaml.safe_load(C.OUT.read_text())
        cams["wrist"] = dict(body="gripper", pos=p[3:6].round(5).tolist(), xyaxes=np.r_[R[:, 0], -R[:, 1]].round(5).tolist(),
                             fovy=float(np.degrees(2 * np.arctan(C.H / 2 / F_PX))), f_px=F_PX, k1=0.0, k2=0.0,
                             rvec_in_gripper=p[:3].tolist(), jaw_silhouette_iou=best["ious"], tip_err_px=best["tip_err_px"],
                             method="measured lens position (user 2026-10-21) + datasheet f; orientation from jaw-tip pixels",
                             status="PROVISIONAL: intrinsics from datasheet; k1 pending checkerboard")
        C.OUT.write_text(yaml.safe_dump(cams, sort_keys=False))
        print("saved wrist camera (out-of-plane", "+y" if best["out_sign"] > 0 else "-y", ") to", C.OUT)


def silhouette_iou(p):
    """IoU of rendered vs real jaw masks for the three wrist frames (half resolution)."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/scene.xml")); d = mujoco.MjData(m)
    from joint_mapping import JointMapping
    jm = JointMapping.from_files()
    cid = m.camera("wrist").id
    Rc, _ = cv2.Rodrigues(p[:3]); Rm = np.c_[Rc[:, 0], -Rc[:, 1], -Rc[:, 2]]
    q = np.zeros(4); mujoco.mju_mat2Quat(q, Rm.ravel())
    m.cam_pos[cid] = p[3:6]; m.cam_quat[cid] = q; m.cam_fovy[cid] = np.degrees(2 * np.arctan(C.H / 2 / p[6]))
    jaw = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in ("gripper", "moving_jaw_so101_v1")}
    w2, h2 = C.W // 2, C.H // 2
    out = {}
    with mujoco.Renderer(m, h2, w2) as r:
        r.enable_segmentation_rendering()
        for name, fr in C.WRIST_FRAMES.items():
            d.qpos[:] = 0; d.qpos[m.jnt_qposadr[m.joint("gripper").id]] = jm.real_to_sim(np.r_[np.zeros(5), fr["gripper"]])[5]
            mujoco.mj_forward(m, d); r.update_scene(d, "wrist"); seg = r.render()
            sm = np.isin(seg[:, :, 0], list(jaw)) & (seg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
            real = cv2.resize(C.real_jaw_mask(fr["image"]), (w2, h2), interpolation=cv2.INTER_AREA) > 127
            out[name] = round(float((sm & real).sum() / max(1, (sm | real).sum())), 3)
    return out


if __name__ == "__main__":
    main()
