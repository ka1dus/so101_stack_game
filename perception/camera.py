"""Calibrated `front` camera (sim/assets/cameras.yaml): project table-frame points to undistorted pixels."""
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CAMS = yaml.safe_load((ROOT / "sim/assets/cameras.yaml").read_text())
W, H = 640, 480


class Camera:
    def __init__(self, pos=None, xyaxes=None, f=None):
        c = CAMS["front"]
        self.pos = np.array(pos if pos is not None else c["pos"], float)
        a = np.asarray(xyaxes if xyaxes is not None else c["xyaxes"], float); x, y = a[:3].copy(), a[3:].copy()
        x /= np.linalg.norm(x); y -= (y @ x) * x; y /= np.linalg.norm(y)
        self.R = np.stack([x, y, np.cross(x, y)])          # world -> MuJoCo camera axes (looks along -z, y up)
        self.f = float(f if f is not None else c["f_px"])

    def project(self, p):
        """(N,3) world points -> (N,2) pixels (u right, v down)."""
        q = (np.atleast_2d(p) - self.pos) @ self.R.T
        return np.c_[W / 2 + self.f * q[:, 0] / -q[:, 2], H / 2 - self.f * q[:, 1] / -q[:, 2]]

    def depth(self, p):
        return -((np.atleast_2d(p) - self.pos) @ self.R.T)[:, 2]


def from_mujoco(pos, quat, fovy_deg=None):
    """A Camera from a MuJoCo camera pose (world-attached camera: pos, quat w-x-y-z)."""
    import mujoco
    Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, np.asarray(quat, float)); Rm = Rm.reshape(3, 3)
    f = H / 2 / np.tan(np.radians(fovy_deg) / 2) if fovy_deg else None
    return Camera(pos=pos, xyaxes=np.r_[Rm[:, 0], Rm[:, 1]], f=f)
