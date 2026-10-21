"""Phase 3 render_check: sim `front`/`wrist` views vs the real reference frames.

Real frames are undistorted with the calibrated (f, k1) so they match MuJoCo's pinhole
renders. Output per frame: real | sim | 50 % blend | sim edges on real, in notes/img/.

    MUJOCO_GL=cgl ~/.venvs/so101_stack_sim/bin/python scripts/render_check.py
"""

from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CAMS = yaml.safe_load((ROOT / "sim/assets/cameras.yaml").read_text())
W, H = 640, 480


def undistort(img, cam):
    K = np.array([[cam["f_px"], 0, W / 2], [0, cam["f_px"], H / 2], [0, 0, 1]])
    return cv2.undistort(img, K, np.array([cam["k1"], cam.get("k2", 0.0), 0, 0]))


def render(camera, setup=None):
    m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    if setup:
        setup(m, d)
    mujoco.mj_forward(m, d)
    with mujoco.Renderer(m, H, W) as r:
        r.update_scene(d, camera)
        return r.render()


def hide_boxes(m, d):
    for b in range(m.nbody):
        if m.body(b).name.startswith("box"):
            adr = m.jnt_qposadr[m.body_jntadr[b]]
            d.qpos[adr:adr + 3] = [5.0 + b, 5.0, -5.0]          # park far away


def stack3(m, d):
    """No.3 bottom, No.35, No.10 top, centred on the stacking spot, aligned with the yellow area."""
    hide_boxes(m, d)
    lay = yaml.safe_load((ROOT / "sim/assets/layout.yaml").read_text())
    sp = lay["stack_area"]["stacking_spot"]   # front_cam_stack3.png retaken on the new X (2026-10-21 13:09)
    yaw = np.radians(lay["stack_area"]["heading_1_2_deg"])
    q = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    z = 0.0002
    for name in ("box0_no3", "box3_no35", "box7_no10"):
        b = m.body(name).id; g = [i for i in range(m.ngeom) if m.geom_bodyid[i] == b][0]
        hz = m.geom_size[g][2]
        adr = m.jnt_qposadr[m.body_jntadr[b]]
        d.qpos[adr:adr + 7] = [sp[0], sp[1], z + hz, *q]
        z += 2 * hz


def panel(real, sim):
    blend = (0.5 * real + 0.5 * sim).astype(np.uint8)
    edges = cv2.Canny(cv2.cvtColor(sim, cv2.COLOR_RGB2GRAY), 40, 120)
    over = real.copy(); over[edges > 0] = [255, 0, 255]
    return np.concatenate([np.concatenate([real, sim], 1), np.concatenate([blend, over], 1)], 0)


def main():
    out = ROOT / "notes/img"
    f = CAMS["front"]
    for name, real_path, setup in (("empty", "front_cam_frame.png", hide_boxes), ("stack3", "front_cam_stack3.png", stack3)):
        real = undistort(np.asarray(Image.open(ROOT / "inputs/photos" / real_path).convert("RGB")), f)
        sim = render("front", setup)
        Image.fromarray(panel(real, sim)).save(out / f"phase3_front_{name}.png")
        print("saved", out / f"phase3_front_{name}.png")
    if CAMS.get("wrist"):
        w = CAMS["wrist"]
        real = undistort(np.asarray(Image.open(ROOT / "inputs/photos/wrist_cam_frame.png").convert("RGB")), w)
        sim = render("wrist", hide_boxes)
        Image.fromarray(panel(real, sim)).save(out / "phase3_wrist_empty.png")
        print("saved", out / "phase3_wrist_empty.png")


if __name__ == "__main__":
    main()
