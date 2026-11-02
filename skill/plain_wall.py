"""Paint the room behind the arm plain in the real `front` frame before the policy sees it (no retraining).

The policies were trained on a plain sim wall (dark blue, colour randomized). The real wall has a dark banner with a
gold logo, and the image-swap test on the 28 Oct trials (notes/phase8.md §13) showed that this background stops the
release: real front 0/9 chunks open, sim wall pasted in 8/9, banner area painted plain 6-7/9.

Mask: pixels of the room / backdrop in the sim twin (world geoms other than the table, mats and tape; no geom = sky)
above image row `banner_row`, minus the arm at the joint reading (twin arm mask, dilated 8 px, refreshed every 3rd tick). Colour: the sim
wall's nominal rendered colour (median over its pixels). The camera is the session's refit `front` camera.
"""
import cv2
import mujoco
import numpy as np

from perception.camera import H, W
from perception.twin import Twin, arm_mask

TABLE_GEOMS = ("stack_area", "supply_mat", "reach_line_tape")


class PlainWall:
    def __init__(self, cam, banner_row=150, margin_px=8, every=3):
        self.cam, self.margin, self.every, self.k, self.arm = cam, margin_px, every, 0, None
        self.twin = Twin(scale=2)                              # per-tick arm mask at half resolution
        t = Twin(); m = t.m; t.set_boxes({}); t.classes(cam)  # sets the camera
        arm_root = m.body_rootid[m.body("base").id]
        hide = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == arm_root]
        m.geom_group[hide] = 5; opt = mujoco.MjvOption(); opt.geomgroup[5] = 0
        t.ren.update_scene(t.d, "front", scene_option=opt); gid = t.ren.render()[..., 0]
        room = np.ones(m.ngeom + 1, bool)
        for g in range(m.ngeom):
            n = m.geom(g).name
            room[g] = m.geom_bodyid[g] == 0 and not (n.startswith("table") or n in TABLE_GEOMS)
        rows = np.arange(H)[:, None] < banner_row
        self.room = room[np.where(gid >= 0, gid, m.ngeom)] & rows       # arm hidden: the wall behind it too
        ren = mujoco.Renderer(m, H, W); ren.update_scene(t.d, "front", scene_option=opt)
        wall = gid == m.geom("wall_upper").id
        self.colour = np.median(ren.render()[wall], axis=0).astype(np.uint8)   # RGB
        ren.close()

    def __call__(self, front_rgb, q):
        """front_rgb: undistorted real front frame (RGB, 640x480); q: joint reading (LeRobot units)."""
        if self.k % self.every == 0:                            # arm mask every `every` ticks (~10 ms each)
            self.arm = arm_mask(self.twin, self.cam, list(q), self.margin)
        self.k += 1
        paint = self.room & ~self.arm
        out = front_rgb.copy(); out[paint] = self.colour
        return out
