"""Render the sim SO-101 at LeRobot joint readings from a rough `front` camera, next to a real frame.

Phase 1 visual check only: the camera comes from a quick PnP fit to table / tape corners
picked by eye in front_cam_frame.png (H0 positions), not the Phase 3 calibration.

    MUJOCO_GL=cgl python scripts/render_pose.py --pose home --hyp H0 --out notes/img/phase1_home_H0.png
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from joint_mapping import BODY_JOINTS, JOINT_NAMES, JointMapping, from_dict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ROBOT_XML = ROOT / "sim/assets/so101/so101_new_calib_camera.xml"
W, H = 640, 480

# (table-frame xyz from measurements.yaml [fk], pixel in front_cam_frame.png picked on a 2x grid, +/- ~3 px)
CORRESPONDENCES = [
    ((0.244, -0.233, 0), (92, 300)),   # table right corner (yellow side)
    ((0.479, 0.066, 0), (320, 468)),   # table far corner
    ((0.180, 0.300, 0), (562, 299)),   # table left corner (black-mat side)
    ((0.081, -0.122, 0), (200, 257)),  # stack corner 1
    ((0.202, 0.008, 0), (300, 316)),   # stack corner 2
    ((0.318, -0.074, 0), (192, 372)),  # stack corner 3
    ((0.197, -0.220, 0), (95, 299)),   # stack corner 4
    ((0.023, 0.107, 0), (430, 242)),   # supply corner 1
    ((0.281, 0.176, 0), (465, 362)),   # supply corner 3
    ((0.142, 0.024, 0), (340, 280)),   # supply corner 4
]

# Hypotheses from notes/phase1.md section 4: joint offsets (deg) and base tilt (deg about x, y)
HYPOTHESES = {
    "H0": dict(offsets={}, tilt=(0.0, 0.0)),
    "tilt": dict(offsets={}, tilt=(-1.35, 0.19)),
    "fit": dict(offsets=dict(shoulder_lift=-6.94, elbow_flex=4.75, wrist_flex=12.97), tilt=(-1.59, -0.41)),
    "M3_roll+95": dict(offsets=dict(shoulder_lift=-5.66, elbow_flex=6.82, wrist_flex=4.61, wrist_roll=95.4), tilt=(-1.10, 0.47)),
    "M3_roll+5": dict(offsets=dict(shoulder_lift=-5.64, elbow_flex=6.89, wrist_flex=8.03, wrist_roll=4.68), tilt=(-1.15, 0.44)),
}
POSES = {
    "home": dict(shoulder_pan=-3.16, shoulder_lift=-87.30, elbow_flex=96.79, wrist_flex=48.35, wrist_roll=8.66, gripper=0.74),
}


TAPE_CAMERA_POS = np.array([0.865, 0.156, 0.38])  # measurements.yaml cameras.front.position (trilaterated, < 1 cm)


def _project(obj, rvec, pos, f, k1):
    R, _ = cv2.Rodrigues(rvec)
    pc = (obj - pos) @ R.T                      # OpenCV camera frame
    xy = pc[:, :2] / pc[:, 2:3]
    r2 = np.sum(xy ** 2, axis=1, keepdims=True)
    return f * xy * (1 + k1 * r2) + np.array([W / 2, H / 2])


def fit_camera():
    """Rotation + focal + bounded k1, position tied to the tape measurement by a prior
    (planar points alone leave focal vs distance degenerate)."""
    from scipy.optimize import least_squares
    obj = np.array([c[0] for c in CORRESPONDENCES], dtype=np.float64)
    img = np.array([c[1] for c in CORRESPONDENCES], dtype=np.float64)
    look = np.array([0.03, 0.03, 0.0]) - TAPE_CAMERA_POS   # measurements.yaml looks_at
    z = look / np.linalg.norm(look); x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x); y = np.cross(z, x)
    rvec0, _ = cv2.Rodrigues(np.stack([x, y, z]))
    def res(v):
        pix = (_project(obj, v[:3], v[5:8], v[3], v[4]) - img).ravel()
        prior = (v[5:8] - TAPE_CAMERA_POS) / 0.02 * 3.0      # tape position, ~2 cm sigma (3 px per sigma)
        return np.r_[pix, prior]
    x0 = np.r_[rvec0.ravel(), 500.0, -0.1, TAPE_CAMERA_POS]
    lo = np.r_[-np.inf * np.ones(3), 300, -0.5, TAPE_CAMERA_POS - 0.06]
    hi = np.r_[np.inf * np.ones(3), 1200, 0.1, TAPE_CAMERA_POS + 0.06]
    sol = least_squares(res, x0, bounds=(lo, hi))
    rvec, f, k1, pos = sol.x[:3], sol.x[3], sol.x[4], sol.x[5:8]
    pix = res(sol.x)[:-3].reshape(-1, 2)
    err = np.sqrt(np.mean(np.sum(pix ** 2, axis=1)))
    R, _ = cv2.Rodrigues(rvec)
    x_mj, y_mj = R[0], -R[1]                    # OpenCV (x right, y down, +z fwd) -> MuJoCo (y up, -z fwd)
    fovy = float(np.degrees(2 * np.arctan(H / 2 / f)))
    return dict(pos=pos, xyaxes=np.r_[x_mj, y_mj], fovy=fovy, f=f, k1=k1, rms_px=err, residuals=pix)


def undistort(real, cam):
    K = np.array([[cam["f"], 0, W / 2], [0, cam["f"], H / 2], [0, 0, 1]])
    return cv2.undistort(real, K, np.array([cam["k1"], 0, 0, 0]))


def tilt_quat(rx, ry):
    q = np.zeros(4)
    qx, qy = np.zeros(4), np.zeros(4)
    mujoco.mju_axisAngle2Quat(qx, np.array([1.0, 0, 0]), np.radians(rx))
    mujoco.mju_axisAngle2Quat(qy, np.array([0, 1.0, 0]), np.radians(ry))
    mujoco.mju_mulQuat(q, qy, qx)
    return q


def build(cam, tilt):
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
    wb = spec.worldbody
    base = spec.body("base")
    base.pos = [-0.0388, 0.0, 0.005]            # pan axis at origin; base origin ~5 mm above table (touch fit)
    base.quat = tilt_quat(*tilt)
    wb.add_light(pos=[0.2, 0, 1.5], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7], ambient=[0.3, 0.3, 0.3])
    # table (38 x 38 cm) from fitted corners; heading of the mount->right edge = -38.1 deg
    c = np.array([(-0.055, 0.002), (0.180, 0.300), (0.244, -0.233), (0.479, 0.066)]).mean(0)
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.19, 0.19, 0.01], pos=[c[0], c[1], -0.01],
                euler=[0, 0, np.radians(-38.1)], rgba=[0.92, 0.93, 0.95, 1])
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.107, 0.073, 0.0005], pos=[0.199, -0.102, 0.0005],
                euler=[0, 0, np.radians(47.2)], rgba=[0.95, 0.75, 0.1, 1])
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1045, 0.096, 0.0005], pos=[0.148, 0.148, 0.0005],
                euler=[0, 0, np.radians(55.3)], rgba=[0.03, 0.03, 0.03, 1])
    wb.add_camera(name="front", pos=cam["pos"], xyaxes=cam["xyaxes"], fovy=cam["fovy"])
    return spec.compile()


def render(q_lerobot, hyp, real_path):
    cam = fit_camera()
    h = HYPOTHESES[hyp]
    model = build(cam, h["tilt"])
    data = mujoco.MjData(model)
    jm = JointMapping.from_files()
    jm.offset_deg = {j: h["offsets"].get(j, 0.0) for j in BODY_JOINTS}
    q = jm.real_to_sim(q_lerobot)
    for i, j in enumerate(JOINT_NAMES):
        data.qpos[model.jnt_qposadr[model.joint(j).id]] = q[i]
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, H, W) as r:
        r.update_scene(data, camera="front")
        sim = r.render()
    real = undistort(np.asarray(Image.open(real_path).convert("RGB")), cam)
    blend = (0.5 * sim + 0.5 * real).astype(np.uint8)
    return cam, np.concatenate([real, sim, blend], axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pose", default="home", help=f"one of {list(POSES)} or a JSON dict of LeRobot readings")
    p.add_argument("--hyp", default="H0", choices=list(HYPOTHESES))
    p.add_argument("--offsets", default=None, help='JSON dict of extra joint offsets (deg), e.g. {"wrist_roll": 90}')
    p.add_argument("--real", default=str(ROOT / "inputs/photos/front_cam_frame.png"))
    p.add_argument("--out", required=True)
    a = p.parse_args()
    pose = POSES[a.pose] if a.pose in POSES else json.loads(a.pose)
    if a.offsets:
        HYPOTHESES[a.hyp] = dict(HYPOTHESES[a.hyp], offsets={**HYPOTHESES[a.hyp]["offsets"], **json.loads(a.offsets)})
    cam, img = render(from_dict(pose), a.hyp, a.real)
    print(f"camera fit: f={cam['f']:.0f}px fovy={cam['fovy']:.1f} deg k1={cam['k1']:.3f} rms={cam['rms_px']:.1f}px pos={np.round(cam['pos'], 3)}")
    print("  residuals px:", np.round(np.linalg.norm(cam["residuals"], axis=1), 1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
