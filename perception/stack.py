"""Stack region, change masks and collapse features on the `front` camera (decision D2, method a).

Everything is a difference between frames of the same game (no colour segmentation): the game-start reference has
an empty stack area, so |frame - reference| inside the stack region is the stack's silhouette; |after - before| is
what the turn changed. A clean placement changes only the column just above the old top; anything changed elsewhere
(lower boxes moved, a box on the table) or anything still changing during the settle window is a collapse cue.
"""
import cv2
import numpy as np
import yaml

from .camera import ROOT, Camera, H, W

LAY = yaml.safe_load((ROOT / "sim/assets/layout.yaml").read_text())["stack_area"]
SPOT = np.array(LAY["stacking_spot"], float)
AREA = np.array(LAY["corners"], float)
DIFF_THR = 30          # max channel difference (0-255) counted as a change
SHADE_CHROMA = 0.03    # colour-balance change still counted as a shadow (stack/collapse checks)
STACK_TOP_Z = 0.16     # m; above 8 boxes (max 8 x 17 mm = 13.6 cm)


def _fill(shape, pts):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(pts).astype(np.int32)), 1)
    return m.astype(bool)


def prism_mask(cam: Camera, xy, r, z0, z1, n=24):
    """Image region of a vertical cylinder (centre xy, radius r) between heights z0 and z1."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.c_[xy[0] + r * np.cos(a), xy[1] + r * np.sin(a)]
    pts = np.vstack([np.c_[ring, np.full(n, z0)], np.c_[ring, np.full(n, z1)]])
    return _fill((H, W), cam.project(pts))


SUPPLY = np.array(yaml.safe_load((ROOT / "sim/assets/layout.yaml").read_text())["supply_area"]["corners"], float)
TABLE = np.array(list(yaml.safe_load((ROOT / "sim/assets/layout.yaml").read_text())["table"]["corners"].values()), float)
FALLEN_Z = 0.04        # m; a fallen box lies (or leans) below this
COLUMN_R = 0.08        # m around the stacking spot: crooked stacks (+-2 cm) plus half a large box's length


def _grow(P, pad):
    c = P.mean(0); return c + (P - c) * (1 + 2 * pad / np.linalg.norm(P - c, axis=1, keepdims=True))


def stack_region(cam: Camera, pad=0.04, arm_mask=None):
    """Where stack boxes (standing or fallen) can be seen: the yellow area grown by pad, up to STACK_TOP_Z, minus the
    supply mat with its boxes (they change every turn) and minus the arm (arm_mask: the twin's arm, dilated).
    Only lines of sight that end on the table near its top (fallen boxes) or pass through the stack's own column
    (standing boxes, the backdrop behind them): the rest of that prism looks past the table edge into the room, where
    people walk by (real clips 2026-10-26)."""
    P = _grow(AREA, pad)
    roi = _fill((H, W), cam.project(np.vstack([np.c_[P, np.zeros(4)], np.c_[P, np.full(4, STACK_TOP_Z)]])))
    T = TABLE[np.argsort(np.arctan2(*(TABLE - TABLE.mean(0)).T[::-1]))]
    roi &= _fill((H, W), cam.project(np.vstack([np.c_[T, np.zeros(4)], np.c_[T, np.full(4, FALLEN_Z)]]))) | \
        prism_mask(cam, SPOT, COLUMN_R, 0.0, STACK_TOP_Z)
    S = _grow(SUPPLY, 0.015)
    roi &= ~_fill((H, W), cam.project(np.vstack([np.c_[S, np.zeros(4)], np.c_[S, np.full(4, 0.025)]])))
    if arm_mask is not None:
        roi &= ~arm_mask
    return roi


def change(a, b, thr=DIFF_THR, shadows=False, chroma=None):
    """Pixels that differ by > thr in some channel. shadows=True: drop pure darkening/brightening that keeps the
    colour balance (a shadow cast or removed), since a moved box changes the colour as well."""
    dd = cv2.absdiff(a, b); d = cv2.max(cv2.max(dd[..., 0], dd[..., 1]), dd[..., 2]) > thr
    if shadows:                                          # only where something changed (same test, far fewer pixels)
        ys, xs = np.nonzero(d)
        if len(ys):
            fa, fb = a[ys, xs].astype(np.float32) + 8, b[ys, xs].astype(np.float32) + 8
            sa, sb = fa.sum(1, keepdims=True), fb.sum(1, keepdims=True)
            ratio = (sa / sb)[:, 0]
            shade = (np.abs(fa / sa - fb / sb).max(1) < (SHADE_CHROMA if chroma is None else chroma)) & \
                (((ratio > 0.3) & (ratio < 1.0)) | ((ratio > 1.0) & (ratio < 3.3)))
            d[ys[shade], xs[shade]] = False
    return cv2.morphologyEx(d.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)


def backproject(cam: Camera, uv, z):
    """Pixels -> points on the horizontal plane at height z."""
    uv = np.atleast_2d(uv).astype(float)
    rays = np.c_[(uv[:, 0] - W / 2) / cam.f, -(uv[:, 1] - H / 2) / cam.f, -np.ones(len(uv))] @ cam.R   # world dirs
    t = (z - cam.pos[2]) / rays[:, 2]
    return cam.pos[:2] + t[:, None] * rays[:, :2]


def top_xy(cam, fg, z_top, band=30):
    """Stack top centre from a silhouette: back-project the top band of the silhouette onto the top-face plane.
    The far edge of the top face is the silhouette's top row; the face spans about `band` px below it."""
    ys, xs = np.nonzero(fg)
    if len(ys) < 50:
        return None
    v0 = np.percentile(ys, 0.5); sel = ys < v0 + band
    return backproject(cam, np.c_[xs[sel], ys[sel]], z_top).mean(0)


def features(cam, ref, before, frames, z_top, new_h, land_idx=18, col_r=0.065, arm_mask=None, new_dims=None, arm_frames=None):
    """Collapse cues for one turn (counts in pixels). frames: the settle window after the release.
    z_top: old stack top height (from the types per level, decision D1); new_h: the new box's height (0: no box).
    arm_frames (robot turns): [(frame, arm mask), ...] at the release and at the end of the skill, the arm still in
    view and masked out; the new box is tracked through them to the settle window's last frame, since a box can slide
    off or be knocked while the arm retracts and returns home."""
    roi = stack_region(cam, arm_mask=arm_mask); after = frames[-1]
    fg_b, fg_a = change(before, ref) & roi, change(after, ref) & roi
    xy = top_xy(cam, fg_b, z_top) if z_top > 0.005 else None
    xy = SPOT if xy is None else xy
    allowed = prism_mask(cam, xy, col_r, z_top - 0.006, z_top + new_h + 0.006) if new_h else np.zeros_like(roi)
    ch = change(after, before, shadows=True) & roi
    land = frames[land_idx]
    if new_h:                                            # the new box: changed on landing, inside its column
        top_mask = change(land, before) & allowed & roi; z_mid = z_top + new_h / 2
    else:                                                # no new box: the old top band
        top_mask = fg_b & (np.arange(H)[:, None] < (np.nonzero(fg_b)[0].min() + 30 if fg_b.any() else 0)); z_mid = z_top
    shift_mm, score = top_shift(cam, land, after, top_mask, z_mid)
    rel_mm, rel_score = 0.0, 1.0
    if new_h:                                            # the new box from its release pose (frame 0) to the landing frame
        rel_mask = change(frames[0], before) & allowed & roi
        rel_mm, rel_score = top_shift(cam, frames[0], land, rel_mask, z_mid)
        gone = float((change(frames[0], land) & rel_mask).sum() / max(1, rel_mask.sum()))
        if gone < RELEASE_GONE:                          # its release pixels barely changed: it did not slide away
            rel_mm = 0.0
    # silhouette top edge before vs after (rows grow downward): a box landing on top raises it by ~its own height
    def top_row(fg):
        ys = np.nonzero(fg)[0]
        return float(np.percentile(ys, 0.5)) if len(ys) > 50 else None
    rb, ra = top_row(fg_b), top_row(fg_a)
    rise = (rb - ra) if (rb is not None and ra is not None) else (H - ra if ra is not None else 0.0)
    exp_rise = float(np.diff(cam.project(np.array([[*xy, z_top], [*xy, z_top + new_h]]))[:, 1])[0] * -1) if new_h else 0.0
    arm_shift, arm_score = 0.0, 1.0
    if new_h and arm_frames:
        seq = list(arm_frames) + [(after, np.zeros_like(roi))]
        for (f0, m0), (f1, m1) in zip(seq[:-1], seq[1:]):
            tm = change(f0, before) & allowed & roi & ~m0 & ~m1
            mm, sc = top_shift(cam, f0, f1, tm, z_top + new_h / 2)
            gone = float((change(f0, f1, shadows=True) & tm).sum() / max(1, tm.sum()))
            if gone < RELEASE_GONE:                      # its pixels barely changed: it did not move
                mm = 0.0
            arm_shift, arm_score = max(arm_shift, mm), min(arm_score, sc)
    on_top_iou = on_top_cover = None
    if new_dims is not None:                            # does the turn's change look like one box lying on the old top?
        on_top_iou, _, _ = fit_box(cam, ch, z_top, new_dims, xy)
        on_top_cover, _, _ = fit_box(cam, ch, z_top, new_dims, xy, score="cover")
    return dict(
        vanished=int((fg_b & ~fg_a).sum()),                 # was stack, now background: something moved away
        outside=int((ch & ~allowed).sum()),                  # changed where a clean placement changes nothing
        inside=int((ch & allowed).sum()),
        settle=int((change(frames[land_idx], after, shadows=True) & roi).sum()),   # still changing after the box landed
        on_top_iou=None if on_top_iou is None else round(on_top_iou, 3),
        on_top_cover=None if on_top_cover is None else round(on_top_cover, 3),
        top_rise_px=round(float(rise), 1), top_rise_expected_px=round(exp_rise, 1),
        top_xy=np.round(xy, 4).tolist(), shift_mm=round(shift_mm, 1), shift_score=round(score, 3),
        release_mm=round(rel_mm, 1), release_score=round(rel_score, 3),
        arm_shift_mm=round(arm_shift, 1), arm_shift_score=round(arm_score, 3),
        release_gone=round(gone, 3) if new_h else 0.0,
        fg_before=int(fg_b.sum()), fg_after=int(fg_a.sum()))


def box_hull(cam, xy, yaw, z_bottom, dims):
    """Image outline (convex hull of the 8 projected corners) of a box lying flat at xy/yaw on height z_bottom."""
    L, Wd, h = dims["length"], dims["width"], dims["height"]
    u = np.array([np.cos(yaw), np.sin(yaw)]); v = np.array([-np.sin(yaw), np.cos(yaw)])
    pts = [np.r_[xy + a * L / 2 * u + b * Wd / 2 * v, z] for z in (z_bottom, z_bottom + h) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return cv2.convexHull(cam.project(np.array(pts)).astype(np.float32)).reshape(-1, 2)


def fit_box(cam, mask, z_bottom, dims, xy0, r=0.05, score="iou"):
    """Best fit of a flat box of `dims` resting at height z_bottom (xy within r of xy0, any yaw) to `mask`.
    score "iou": intersection over union; "cover": share of the box outline that changed (insensitive to other
    changes, e.g. the robot nudging the stack a few mm). Coarse grid then Nelder-Mead. Returns (score, xy, yaw)."""
    from scipy.optimize import minimize
    ys, xs = np.nonzero(mask)
    if len(ys) < 30:
        return 0.0, np.asarray(xy0), 0.0
    pad = 60; y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad); x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad)
    m = mask[y0:y1, x0:x1]; area = m.sum(); off = np.array([x0, y0])
    canvas = np.zeros(m.shape, np.uint8)
    def iou(p):
        canvas[:] = 0; cv2.fillConvexPoly(canvas, np.round(box_hull(cam, p[:2], p[2], z_bottom, dims) - off).astype(np.int32), 1)
        c = canvas.astype(bool); inter = (c & m).sum()
        return inter / (c.sum() + 1e-6) if score == "cover" else inter / (area + c.sum() - inter + 1e-6)
    best = (-1, None)
    for dx in np.linspace(-r, r, 9):
        for dy in np.linspace(-r, r, 9):
            for yaw in np.radians(np.arange(0, 180, 15)):
                p = np.r_[xy0[0] + dx, xy0[1] + dy, yaw]; v = iou(p)
                if v > best[0]:
                    best = (v, p)
    res = minimize(lambda p: -iou(p), best[1], method="Nelder-Mead",
                   options=dict(initial_simplex=np.vstack([best[1], best[1] + np.diag([0.006, 0.006, np.radians(8)])]), maxiter=150))
    p = res.x if -res.fun > best[0] else best[1]
    return float(max(best[0], -res.fun)), p[:2], float(p[2])


def top_shift(cam, f0, f1, top_mask, z, search=50, masked=True):
    """How far the top box moved between frames f0 and f1, in mm on the plane at height z (template match of the
    top box's image patch). Returns (mm, match score); score < 0.5 means the box is no longer recognisable there."""
    ys, xs = np.nonzero(top_mask)
    if len(ys) < 30:
        return 0.0, 1.0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    tpl = f0[y0:y1, x0:x1]; msk = top_mask[y0:y1, x0:x1].astype(np.uint8) * 255   # only the top box's own pixels
    msk = cv2.merge([msk] * 3)
    Y0, Y1, X0, X1 = max(0, y0 - search), min(H, y1 + search), max(0, x0 - search), min(W, x1 + search)
    res = cv2.matchTemplate(f1[Y0:Y1, X0:X1], tpl, cv2.TM_CCORR_NORMED, mask=msk) if masked else \
        cv2.matchTemplate(f1[Y0:Y1, X0:X1], tpl, cv2.TM_CCOEFF_NORMED)
    res[~np.isfinite(res)] = 0
    _, score, _, loc = cv2.minMaxLoc(res)
    du, dv = X0 + loc[0] - x0, Y0 + loc[1] - y0
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    p = backproject(cam, np.vstack([c, c + [du, dv]]), z)
    return float(np.linalg.norm(p[1] - p[0]) * 1000), float(score)


# collapse rule (conservative: a false collapse call ruins a game, CLAUDE.md Section 9); thresholds in pixels,
# set from the sim test set with a margin over the largest value seen on standing stacks (notes/phase7.md)
TOP_MOVE_MM = 18.0     # game rule 15 mm + 3 mm for the estimate's error (sim: median 1.2 mm), on the no-false-call side
OUTSIDE_PX = 600       # changed pixels outside the new box's column (lower boxes moved, a box on the table); sim max on standing stacks 491
ON_TOP_MIN_COVER = 0.35   # the best box outline on the old top must be at least this much changed; sim min on standing
                          # stacks 0.43 (IoU was used first; a robot nudging the stack lowered it, sim_game 2026-10-25)
TOP_RISE_SHARE = 0.5      # ...unless the silhouette top edge rose by at least half a box height: a box landing squarely
                          # on an identical box changes few pixels (same print), sim_game 2026-10-26
TOP_DROP_PX = 5.0         # the silhouette top edge sinking this far: a box came off the top (sim: standing >= -1 px)
RELEASE_MM = 11.0      # the new box sliding this far between release and the start of the settle window
RELEASE_GONE = 0.45    # ...and this share of its release-time pixels changed (sim: falls >= 0.50, one wobble 0.35)


def collapse_check(f):
    """f: features(). Returns (collapse, reasons)."""
    reasons = []
    if f["shift_score"] < 0.5:
        reasons.append("top box no longer there")
    elif f["shift_mm"] > TOP_MOVE_MM:
        reasons.append(f"top moved {f['shift_mm']:.0f} mm")
    if f["outside"] > OUTSIDE_PX:
        reasons.append("changed below/beside the new box")
    if f.get("top_rise_px", 0.0) < -TOP_DROP_PX:
        reasons.append(f"stack top {-f['top_rise_px']:.0f} px lower than before")
    if f.get("arm_shift_mm", 0.0) > RELEASE_MM or f.get("arm_shift_score", 1.0) < 0.5:
        reasons.append(f"new box moved {f['arm_shift_mm']:.0f} mm while the arm left" if f.get("arm_shift_score", 1.0) >= 0.5
                       else "new box gone while the arm left")
    if f.get("release_mm", 0.0) > RELEASE_MM:
        reasons.append(f"new box slid {f['release_mm']:.0f} mm after release")
    if f.get("on_top_cover") is not None and f["on_top_cover"] < ON_TOP_MIN_COVER \
            and f.get("top_rise_px", 0.0) < TOP_RISE_SHARE * f.get("top_rise_expected_px", 0.0):
        reasons.append("new box not lying on the stack")
    return bool(reasons), reasons
