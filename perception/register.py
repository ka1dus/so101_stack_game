"""Refine the `front` camera pose from a reference frame (game setup, arm at home), render-and-compare with the twin.

The supply mat is the dark region and the stack area the yellow one (the arm and backdrop are left out: their twin
outlines are approximate, 2026-10-24 real-frame test); their image masks are
compared with the twin's rendered class masks (soft Dice, only over the table top: the real room has dark walls,
banners and carpet that the twin does not model, 2026-10-24 real-frame test) and the 6-DOF camera pose is
adjusted from the nominal calibration (Nelder-Mead, coarse then fine blur). Catches the tripod drift the policy's
randomization tolerates but fixed image polygons do not. The boxes in the reference are rendered where the template
puts them, so they are not mistaken for mat.
"""
import cv2
import numpy as np
from scipy.optimize import minimize

import yaml

from .camera import ROOT, Camera, H, W
from .twin import DARK, FUZZY, YELLOW, Twin

DARK_V = 60
ROI_PAD_PX = 25
FUZZ_K = np.ones((31, 31), np.uint8)   # arm/backdrop excluded with a 15 px margin (their twin outlines are approximate)        # around the projected table top (covers ~1 cm / 2 deg of drift)
LAY = yaml.safe_load((ROOT / "sim/assets/layout.yaml").read_text())
PRIOR_W = 5e-4         # weak pull toward the nominal calibration per (deg)^2 and per (cm)^2: the mask cost is nearly
                       # flat along the viewing direction (2026-10-26: one real image refit to 9 mm / 0.4 deg and to
                       # 27 mm / 1.5 deg at the same cost, 0.1166 vs 0.1163); 27 mm / 1.5 deg now costs +0.005


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
    T = np.array(list(LAY["table"]["corners"].values()), float); c = T.mean(0); T = T[np.argsort(np.arctan2(*(T - c).T[::-1]))]
    roi = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(roi, np.round(cam.project(np.c_[T, np.zeros(4)])).astype(np.int32), 1)
    roi = cv2.dilate(roi, np.ones((2 * ROI_PAD_PX + 1,) * 2, np.uint8)).astype(np.float32)
    def cost(p, s):
        c = twin.classes(_perturb(cam, p))
        bl = lambda x: cv2.GaussianBlur(x, (0, 0), s)
        rd, ry = bl((c == DARK).astype(np.float32)), bl((c == YELLOW).astype(np.float32))
        w = roi * (1 - cv2.dilate((c == FUZZY).astype(np.uint8), FUZZ_K).astype(np.float32))   # skip arm and backdrop
        return 1 - 0.5 * (_dice(rd, bl(dark), w) + _dice(ry, bl(yellow), w)) + \
            PRIOR_W * (np.sum(np.square(p[:3])) + np.sum(np.square(np.asarray(p[3:]) / 10)))
    p = np.zeros(6); c0 = cost(p, 2.0)
    for s, step in ((6.0, (1.0, 1.0, 1.0, 10, 10, 10)), (2.0, (0.3, 0.3, 0.3, 3, 3, 3))):
        simplex = np.vstack([p, p + np.diag(step)])
        r = minimize(cost, p, args=(s,), method="Nelder-Mead", options=dict(initial_simplex=simplex, maxiter=400, xatol=1e-3, fatol=1e-5))
        p = r.x
    new = _perturb(cam, p)
    return new, dict(ok=True, cost_before=round(float(c0), 4), cost_after=round(float(cost(p, 2.0)), 4),
                     delta_deg=np.round(p[:3], 3).tolist(), delta_mm=np.round(p[3:], 1).tolist())
