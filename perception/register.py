"""Refine the `front` camera pose from a reference frame (game setup, arm at home), render-and-compare with the twin.

The supply mat, backdrop and black arm are the dark regions, the stack area the yellow one; their image masks are
compared with the twin's rendered class masks (soft Dice, inside the table region) and the 6-DOF camera pose is
adjusted from the nominal calibration (Nelder-Mead, coarse then fine blur). Catches the tripod drift the policy's
randomization tolerates but fixed image polygons do not. The boxes in the reference are rendered where the template
puts them, so they are not mistaken for mat.
"""
import cv2
import numpy as np
from scipy.optimize import minimize

from .camera import Camera, H, W
from .twin import BOX, DARK, YELLOW, Twin

DARK_V = 60


def image_classes(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    dark = hsv[..., 2] < DARK_V
    yellow = (hsv[..., 0] >= 15) & (hsv[..., 0] <= 40) & (hsv[..., 1] > 90) & (hsv[..., 2] > 90)
    return dark.astype(np.float32), yellow.astype(np.float32)


def _perturb(cam, p):
    """p = (rx, ry, rz) deg about the camera axes, (tx, ty, tz) mm in the world."""
    dR = cv2.Rodrigues(np.radians(np.asarray(p[:3], float)))[0]
    R = dR @ cam.R
    return Camera(pos=cam.pos + np.asarray(p[3:], float) / 1000, xyaxes=np.r_[R[0], R[1]], f=cam.f)


def _dice(a, b, w):
    return 2 * (a * b * w).sum() / ((a * w).sum() + (b * w).sum() + 1e-6)


def refine(ref_bgr, twin: Twin, box_poses, arm_real, cam=None):
    """Returns (Camera, info). box_poses: twin body -> (x, y, z, yaw) for the boxes in the reference."""
    cam = cam or Camera()
    twin.set_arm(arm_real); twin.set_boxes(box_poses)
    dark, yellow = image_classes(ref_bgr)
    roi0 = twin.classes(cam) > 0                              # anything but "other"; widened below
    roi = cv2.dilate(roi0.astype(np.uint8), np.ones((81, 81), np.uint8)).astype(np.float32)
    roi[: int(H * 0.25)] *= 0.0 if roi0[: int(H * 0.25)].mean() < 0.01 else 1.0
    def cost(p, s):
        c = twin.classes(_perturb(cam, p))
        bl = lambda x: cv2.GaussianBlur(x, (0, 0), s)
        rd, ry = bl((c == DARK).astype(np.float32)), bl((c == YELLOW).astype(np.float32))
        return 1 - 0.5 * (_dice(rd, bl(dark), roi) + _dice(ry, bl(yellow), roi))
    p = np.zeros(6); c0 = cost(p, 2.0)
    for s, step in ((6.0, (1.0, 1.0, 1.0, 10, 10, 10)), (2.0, (0.3, 0.3, 0.3, 3, 3, 3))):
        simplex = np.vstack([p, p + np.diag(step)])
        r = minimize(cost, p, args=(s,), method="Nelder-Mead", options=dict(initial_simplex=simplex, maxiter=400, xatol=1e-3, fatol=1e-5))
        p = r.x
    new = _perturb(cam, p)
    return new, dict(ok=True, cost_before=round(float(c0), 4), cost_after=round(float(cost(p, 2.0)), 4),
                     delta_deg=np.round(p[:3], 3).tolist(), delta_mm=np.round(p[3:], 1).tolist())
