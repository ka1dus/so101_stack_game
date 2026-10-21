"""Phase 3: refine the wrist camera mount from the measured start (wrist_from_measurement.py).

Free: orientation (3) + lens position (3), lens held to the measured position by a +/- 1 cm
penalty; f fixed at the datasheet value (518 px), k1 = 0. Loss = (1 - jaw IoU) summed over
home / P1 / P3, plus jaw-tip pixel residuals. Quarter resolution for speed.

    MUJOCO_GL=cgl ~/.venvs/so101_stack_sim/bin/python scripts/wrist_refine.py [--save]
"""
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
import calibrate_cameras as C  # noqa: E402
import wrist_from_measurement as WM  # noqa: E402
from joint_mapping import JointMapping  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    cams = yaml.safe_load(C.OUT.read_text()); w = cams["wrist"]
    p0 = np.r_[w["rvec_in_gripper"], w["pos"]]
    G = C.gripper_poses(); tipf = G["home"][2]["tip_fixed"]
    c_meas = tipf + np.array([WM.MEASURED["side"], WM.MEASURED["out"], WM.MEASURED["back"]])
    tips = [(tipf, (318, 322)), (G["home"][2]["tip_moving"], (352, 333))]
    m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/scene.xml")); d = mujoco.MjData(m)
    jm = JointMapping.from_files(); cid = m.camera("wrist").id
    hb = m.body("wrist_camera_housing").id
    jaw = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in ("gripper", "moving_jaw_so101_v1")}
    w4, h4 = C.W // 4, C.H // 4
    r = mujoco.Renderer(m, h4, w4); r.enable_segmentation_rendering()
    frames = [(jm.real_to_sim(np.r_[np.zeros(5), fr["gripper"]])[5],
               cv2.resize(C.real_jaw_mask(fr["image"]), (w4, h4), interpolation=cv2.INTER_AREA) > 127)
              for fr in C.WRIST_FRAMES.values()]

    def set_cam(p):
        R, _ = cv2.Rodrigues(p[:3]); Rm = np.c_[R[:, 0], -R[:, 1], -R[:, 2]]
        q = np.zeros(4); mujoco.mju_mat2Quat(q, Rm.ravel())
        m.cam_pos[cid] = p[3:6]; m.cam_quat[cid] = q; m.cam_fovy[cid] = np.degrees(2 * np.arctan(C.H / 2 / WM.F_PX))
        m.body_pos[hb] = p[3:6] + Rm[:, 2] * 0.017; m.body_quat[hb] = q     # housing follows the lens

    def ious(p):
        set_cam(p); out = []
        for g, real in frames:
            d.qpos[:] = 0; d.qpos[m.jnt_qposadr[m.joint("gripper").id]] = g
            mujoco.mj_forward(m, d); r.update_scene(d, "wrist"); seg = r.render()
            sm = np.isin(seg[:, :, 0], list(jaw)) & (seg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
            out.append((sm & real).sum() / max(1, (sm | real).sum()))
        return out

    def loss(p):
        R, _ = cv2.Rodrigues(p[:3]); tip = 0.0
        for pg, uv in tips:
            pc = R.T @ (pg - p[3:6])
            if pc[2] <= 0:
                return 10.0
            tip += np.sum((WM.F_PX * pc[:2] / pc[2] + [C.W / 2, C.H / 2] - uv) ** 2)
        pos_pen = max(0.0, np.linalg.norm(p[3:6] - c_meas) - 0.01) * 100
        return float(sum(1 - np.array(ious(p))) + tip / 400 + pos_pen)

    print("start: IoU", np.round(ious(p0), 3), "loss", round(loss(p0), 3))
    sol = minimize(loss, p0, method="Powell", options=dict(maxiter=600, xtol=1e-4, ftol=1e-4))
    p = sol.x
    R, _ = cv2.Rodrigues(p[:3])
    res = dict(iou=np.round(ious(p), 3).tolist(), lens_from_measured_cm=round(float(np.linalg.norm(p[3:6] - c_meas) * 100), 2),
               axis_vs_jaw_deg=round(float(np.degrees(np.arccos(np.clip(R[:, 2] @ [0, 0, -1.0], -1, 1)))), 1))
    print("refined:", res)
    if "--save" in sys.argv:
        w.update(pos=p[3:6].round(5).tolist(), xyaxes=np.r_[R[:, 0], -R[:, 1]].round(5).tolist(), rvec_in_gripper=p[:3].tolist(),
                 jaw_silhouette_iou=dict(zip(C.WRIST_FRAMES, res["iou"])),
                 method="measured lens position + datasheet f; orientation+position refined on jaw silhouettes (scripts/wrist_refine.py)")
        C.OUT.write_text(yaml.safe_dump(cams, sort_keys=False)); print("saved")
    r.close()


if __name__ == "__main__":
    main()
