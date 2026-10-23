"""Supply-slot occupancy (decision D1): which template slots still hold a box.

Each slot's box top face (sim/assets/supply_layout.yaml, measured box sizes) is projected into `front` and shrunk
so placement jitter (+/- 2 mm, +/- 3 deg) and a neighbour's edge don't leak in. A printed box over the black mat is
bright; an empty slot shows the mat. Two cues per slot:
  dark   share of mat-like pixels (darker than the mat level + DARK_MARGIN; the mat level is measured each frame on
         the strip beyond the reach tape, where boxes never are)
  diff   mean abs difference to the game-start reference frame (all slots full)
"""
import cv2
import numpy as np
import yaml

from .camera import ROOT, Camera

LAYOUT = yaml.safe_load((ROOT / "sim/assets/supply_layout.yaml").read_text())
SUPPLY = yaml.safe_load((ROOT / "sim/assets/layout.yaml").read_text())["supply_area"]
MEAS = yaml.safe_load((ROOT / "inputs/measurements.yaml").read_text())
BOX = {v["sim_name"]: v["dimensions"] for v in MEAS["boxes"]["variants"]}
REACH = yaml.safe_load((ROOT / "sim/assets/reach_line.yaml").read_text())["reach_line"]
DARK_MARGIN = 35       # grey levels above the mat's own level (measured beyond the reach tape, never covered)
SHRINK = 0.8           # keep the inner 80 % of the footprint in each direction
DARK_EMPTY = 0.5       # slot empty if more than half of the patch is mat-dark


def _box_corners(s, shrink=1.0, pad=0.0, z=None):
    """3D corners of a template slot's box (top and bottom, or one level z)."""
    C = np.array(SUPPLY["corners"]); head = np.arctan2(*(C[1] - C[0])[::-1])
    d = BOX[s["type"]]; c = np.array(s["centre_world_m"])
    yaw = head + (0 if s["long_axis"] == "along edge 1->2" else np.pi / 2)
    u = np.array([np.cos(yaw), np.sin(yaw)]); v = np.array([-np.sin(yaw), np.cos(yaw)])
    L, Wd = shrink * d["length"] / 2 + pad, shrink * d["width"] / 2 + pad
    zs = (0.0, d["height"] + pad) if z is None else (z,)
    return np.array([np.r_[c + a * L * u + b * Wd * v, zz] for zz in zs for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def slot_polygons(cam=None, shrink=SHRINK):
    """Each slot's footprint at mat level (a box standing there always hides it; an empty slot always shows mat
    there, whereas a patch at box-top height would look past a far-edge slot onto the table)."""
    cam = cam or Camera()
    return {s["id"]: cam.project(_box_corners(s, shrink, z=0.0)) for s in LAYOUT["boxes"]}


def neighbour_hulls(cam=None, pad=0.003):
    """Image hull of each slot's box (padded for the +/- 2 mm template jitter): where it can hide other slots."""
    cam = cam or Camera()
    return {s["id"]: cv2.convexHull(cam.project(_box_corners(s, pad=pad)).astype(np.float32)).reshape(-1, 2) for s in LAYOUT["boxes"]}


def far_mat_polygon(cam=None):
    """The mat beyond the reach tape (corner 2/3 side), shrunk 1.5 cm away from the tape and the mat edges."""
    cam = cam or Camera()
    C = np.array(SUPPLY["corners"]); e = np.array(REACH["ends"])
    a, b = (e if np.linalg.norm(e[0] - C[1]) < np.linalg.norm(e[1] - C[1]) else e[::-1])   # a nearer corner 2
    poly = np.array([a, C[1], C[2], b]); c = poly.mean(0)
    poly = c + (poly - c) * 0.7
    return cam.project(np.c_[poly, np.zeros(4)])


def _masks(shape, polys):
    ms = {}
    for k, p in polys.items():
        m = np.zeros(shape[:2], np.uint8); cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1); ms[k] = m.astype(bool)
    return ms


class SupplyMonitor:
    def __init__(self, ref_bgr=None, cam=None):
        self.polys = slot_polygons(cam); self.far = far_mat_polygon(cam); self.hulls = neighbour_hulls(cam)
        self.masks = None; self.ref = None
        if ref_bgr is not None:
            self.ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def features(self, bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.masks is None:
            self.masks = _masks(g.shape, self.polys); self.far_mask = _masks(g.shape, {0: self.far})[0]
            hull = _masks(g.shape, self.hulls)
            for k in self.masks:                                 # drop what a neighbouring box could cover
                for j, h in hull.items():
                    if j != k:
                        self.masks[k] &= ~h
        thr = float(np.median(g[self.far_mask])) + DARK_MARGIN
        out = {}
        for k, m in self.masks.items():
            dark = float((g[m] < thr).mean())
            diff = float(np.abs(g[m].astype(np.float32) - self.ref[m]).mean()) if self.ref is not None else None
            out[k] = dict(dark=dark, diff=diff)
        return out

    def occupied(self, bgr):
        """slot id -> True (box there) / False (empty)."""
        return {k: f["dark"] < DARK_EMPTY for k, f in self.features(bgr).items()}
