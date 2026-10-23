"""The sim twin as a perception tool (decisions D3 and camera refinement): render class masks of the real scene from a
given `front` camera pose and arm joint state. Classes: dark (supply mat, backdrop, black arm), yellow (stack area),
box, other."""
import sys

import mujoco
import numpy as np

from .camera import ROOT, Camera, H, W

sys.path.insert(0, str(ROOT / "scripts"))
from joint_mapping import JOINT_NAMES, JointMapping  # noqa: E402

DARK, YELLOW, BOX, OTHER = 1, 2, 3, 0


class Twin:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/scene.xml"))
        self.d = mujoco.MjData(self.m)
        self.ren = mujoco.Renderer(self.m, H, W); self.ren.enable_segmentation_rendering()
        self.cam = self.m.camera("front").id
        self.jm = JointMapping.from_files()
        arm_root = self.m.body_rootid[self.m.body("base").id]
        lut = np.full(self.m.ngeom + 1, OTHER, np.uint8)
        for g in range(self.m.ngeom):
            b = self.m.geom_bodyid[g]; name = self.m.geom(g).name
            if self.m.body_rootid[b] == arm_root or name in ("supply_mat", "backdrop"):
                lut[g] = DARK
            elif name == "stack_area":
                lut[g] = YELLOW
            elif self.m.body(b).name.startswith("box"):
                lut[g] = BOX
        self.lut = lut
        self.boxes = [self.m.body(b).name for b in range(self.m.nbody) if self.m.body(b).name.startswith("box")]

    def set_arm(self, real_joints):
        """real_joints: LeRobot units in JOINT_NAMES order."""
        q = self.jm.real_to_sim(np.asarray(real_joints, float))
        for j, v in zip(JOINT_NAMES, q):
            self.d.qpos[self.m.jnt_qposadr[self.m.joint(j).id]] = v

    def set_boxes(self, poses):
        """poses: {body name: (x, y, z, yaw)}; bodies not given are hidden under the table."""
        for k, name in enumerate(self.boxes):
            b = self.m.body(name).id; adr = self.m.jnt_qposadr[self.m.body_jntadr[b]]
            x, y, z, yaw = poses.get(name, (5.0 + k, 5.0, -1.0, 0.0))
            self.d.qpos[adr:adr + 7] = [x, y, z, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    def classes(self, cam: Camera):
        R = np.stack([cam.R[0], cam.R[1], cam.R[2]], 1)          # columns = camera axes in world
        q = np.zeros(4); mujoco.mju_mat2Quat(q, R.ravel())
        self.m.cam_pos[self.cam] = cam.pos; self.m.cam_quat[self.cam] = q
        self.m.cam_fovy[self.cam] = np.degrees(2 * np.arctan(H / 2 / cam.f))
        mujoco.mj_forward(self.m, self.d)
        self.ren.update_scene(self.d, "front")
        gid = self.ren.render()[..., 0]
        out = np.zeros(gid.shape, np.uint8); ok = gid >= 0; out[ok] = self.lut[gid[ok]]
        return out


def arm_mask(twin, cam, arm_real, margin_px=6):
    """The arm's image region from the twin at the given joint state, dilated by margin_px (decision D3)."""
    import cv2
    twin.set_arm(arm_real); twin.set_boxes({})
    twin.classes(cam)                                    # sets the camera and renders the scene
    arm = np.zeros((H, W), np.uint8)
    gid = twin.ren.render()[..., 0]
    root = twin.m.body_rootid[twin.m.body("base").id]
    is_arm = np.zeros(twin.m.ngeom + 1, bool)
    for g in range(twin.m.ngeom):
        is_arm[g] = twin.m.body_rootid[twin.m.geom_bodyid[g]] == root
    ok = gid >= 0; arm[ok] = is_arm[gid[ok]]
    return cv2.dilate(arm, np.ones((2 * margin_px + 1,) * 2, np.uint8)).astype(bool)
