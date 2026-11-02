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
from .supply import BOX as BOX_DIMS

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
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8))
    if n > 2:                                             # the stack is one blob; drop specks and lighting patches
        fg = lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))   # elsewhere (real 2026-10-27: auto exposure on the table)
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


def fit_box(cam, mask, z_bottom, dims, xy0, r=0.05, score="iou", top_only=False):
    """Best fit of a flat box of `dims` resting at height z_bottom (xy within r of xy0, any yaw) to `mask`.
    score "iou": intersection over union; "cover": share of the box outline that changed (insensitive to other
    changes, e.g. the robot nudging the stack a few mm). top_only: the top box of a stack; mask rows below the box's
    own outline (the lower levels) are ignored (29 Oct: whole-silhouette fits put the top box between levels).
    Coarse grid then Nelder-Mead. Returns (score, xy, yaw)."""
    from scipy.optimize import minimize
    ys, xs = np.nonzero(mask)
    if len(ys) < 30:
        return 0.0, np.asarray(xy0), 0.0
    pad = 60; y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad); x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad)
    m = mask[y0:y1, x0:x1]; area = m.sum(); off = np.array([x0, y0])
    m_top = np.percentile(np.nonzero(m)[0], 1) if top_only else 0.0
    canvas = np.zeros(m.shape, np.uint8)
    def iou(p):
        canvas[:] = 0; cv2.fillConvexPoly(canvas, np.round(box_hull(cam, p[:2], p[2], z_bottom, dims) - off).astype(np.int32), 1)
        c = canvas.astype(bool)
        if top_only:                                   # only the rows down to the outline's lowest row, and the
            rows = np.nonzero(c.any(1))[0]             # outline's top row on the silhouette's top row (the far edge
            if not len(rows):                          # of the top face fixes the depth; without it tall stacks pulled
                return 0.0                             # the fit down onto lower levels: sim 29 Oct)
            mm = m[:rows.max() + 1]; c = c[:rows.max() + 1]; inter = (c & mm).sum()
            return inter / (mm.sum() + c.sum() - inter + 1e-6) - TOP_ROW_W * abs(rows.min() - m_top)
        inter = (c & m).sum()
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


def landed_tick(kin, trace, k0, max_s=0.5, fps=30, tol_m=0.005):
    """The tick to start tracking the new box from (arm_frames): the last tick up to max_s after the release tick k0
    with the jaw tips still within tol_m of where they released. A box released from a few cm up is still in the jaws
    at k0 and falls onto the stack within ~0.1 s; measured from k0 that fall counted as "moved while the arm left"
    (false collapse call, real trial 20261028_193906 t3, release 2.3 cm above the top; notes/phase8.md §14)."""
    p0 = kin.tip_pos(trace[k0, :6].tolist()); k = k0
    for j in range(k0 + 1, min(len(trace), k0 + int(max_s * fps) + 1)):
        if np.linalg.norm(kin.tip_pos(trace[j, :6].tolist()) - p0) > tol_m:
            break
        k = j
    return k


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


# ------------------------------------------------------------------------------------------------ turn check v2
# Judge a turn from the scene before it and after its settle window: is the new box lying on the old top, with
# nothing else changed? (29 Oct: the cue-based collapse_check gave 17 false calls in 53 good real placements and
# missed 14 of 32 failures: auto exposure made large "changes", fallen boxes lay inside the new box's allowed column
# or outside the watched area; notes/phase8.md section 23.)
PHOTO_GRID = 4            # px step for the photometric fit samples
TURN_FIT_SOFT, TURN_OFF_SOFT = 0.20, 0.040   # fit >= 0.20, centre <= 40 mm, edges <= TURN_BLOB_SOFT, stack not moved,
TURN_BLOB_SOFT = 1000                        # but a stricter cue failed: "unsure" (the game asks the player). Real
                                             # failures with fit 0.2-0.45 are all 56-61 mm off: still called
STICK_MARGIN_PX = 8       # changed pixels this far above the flat box's outline count as "sticking up"
STICK_COL_H = 0.08        # ...looked for up to this far above the old top
STICK_MAX_PX = 60         # ...more than this: not lying flat (on its end): ask (good placements, real and sim: 0)
TOP_ROW_W = 0.01          # top-only fit: IoU penalty per px between the outline's top row and the silhouette's
TURN_FIT_MIN = 0.45       # new box fit (IoU of a flat box on the old top vs the change mask) at least this
                          # (real 82 trials: good placements >= 0.50, failures caught by it alone <= 0.37)
TURN_OFF_MAX = 0.032      # m: new box centre within this of the old top centre (real: good <= 29 mm, half-off
                          # leaning boxes >= 35 mm)
TURN_TOP_FIT_MIN = 0.6    # the old-top fit in `before` is used only at least this good (else the given top_xy0)
TURN_EDGE_BLOB = 450      # px of new/vanished edges in one place outside the new box: another box moved or fell
                          # (real: good placements <= 220, falls caught by it alone >= 1044; sim's hard shadows: good
                          # placements up to 432)
TURN_STACK_CHANGE = 0.15  # share of the old stack's silhouette (outside the new box) changed: the stack moved
NUDGE_MM = 10.0           # ...after letting each band of it shift this far (a box nudged a few mm changes most of its
                          # printed pixels; sim game 30 Oct: 4 px / ~4 mm -> 40 %; sim truth allows the top 15 mm)
NUDGE_BAND_PX = 12        # band height (rows) that shifts on its own (about one box at mid stack)
                          # (real good <= 0.03, sim good <= 0.11)
TURN_COVER_MIN = 0.25     # share of the old top face (inner 80 %) that must have changed: covered by the new box
TURN_COVER_ZMAX = 0.025   # ...applied to single-box stacks only: there a box lying beside the stack (1.3-1.7 cm lower)
                          # fits nearly as well as one on top (real: falls 0.0-0.2, good >= 0.35)
TURN_NEAR_R = 0.07        # m: the new box fit only looks at changes in this column around the old top
EDGE_T = 40.0             # Sobel magnitude (blurred grey) counted as an edge
TURN_ARM_MARGIN = 10      # px around the arm at home (it may settle a little differently)


def table_region(cam: Camera, arm_mask=None):
    """Where a box can be after a turn: the whole table top (up to FALLEN_Z) and the stack column, minus the supply mat
    (its picked slot empties every turn) and the arm."""
    T = TABLE[np.argsort(np.arctan2(*(TABLE - TABLE.mean(0)).T[::-1]))]
    roi = _fill((H, W), cam.project(np.vstack([np.c_[T, np.zeros(4)], np.c_[T, np.full(4, FALLEN_Z)]]))) | \
        prism_mask(cam, SPOT, COLUMN_R, 0.0, STACK_TOP_Z)
    S = _grow(SUPPLY, 0.015)
    roi &= ~_fill((H, W), cam.project(np.vstack([np.c_[S, np.zeros(4)], np.c_[S, np.full(4, 0.025)]])))
    if arm_mask is not None:
        roi &= ~cv2.dilate(arm_mask.astype(np.uint8), np.ones((2 * TURN_ARM_MARGIN + 1,) * 2, np.uint8)).astype(bool)
    return roi


def static_region(cam: Camera, arm_mask=None):
    """Pixels that no turn changes, for the photometric fit: everything except the table top, the stack column, the
    supply mat and the arm (dilated): backdrop, wall, banner, floor, table front and legs, a wide range of colours.
    (Fitting on the table top alone, near-white only, extrapolated badly: 29 Oct.)"""
    T = TABLE[np.argsort(np.arctan2(*(TABLE - TABLE.mean(0)).T[::-1]))]
    busy = _fill((H, W), cam.project(np.vstack([np.c_[T, np.zeros(4)], np.c_[T, np.full(4, FALLEN_Z)]]))) | \
        prism_mask(cam, SPOT, COLUMN_R + 0.03, 0.0, STACK_TOP_Z)
    if arm_mask is not None:
        busy |= arm_mask
    return ~cv2.dilate(busy.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)


def photometric_match(src, dst, static):
    """src (BGR) mapped per channel (gain, offset) to look like dst on the static pixels: cancels the front camera's
    auto exposure / white balance drift between two frames. Robust: two passes, dropping samples off by > 20."""
    ys, xs = np.nonzero(static[::PHOTO_GRID, ::PHOTO_GRID]); ys, xs = ys * PHOTO_GRID, xs * PHOTO_GRID
    out = np.empty_like(src, np.float32)
    for c in range(3):
        a = src[ys, xs, c].astype(np.float32); b = dst[ys, xs, c].astype(np.float32)
        keep = (a > 8) & (a < 247) & (b > 8) & (b < 247)
        g, o = 1.0, 0.0
        for _ in range(2):
            A = np.c_[a[keep], np.ones(keep.sum())]
            if len(A) < 50:
                break
            g, o = np.linalg.lstsq(A, b[keep], rcond=None)[0]
            keep &= np.abs(g * a + o - b) < 20
        out[..., c] = src[..., c] * g + o
    return np.clip(out, 0, 255).astype(np.uint8)


def edges(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 1.0)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) > EDGE_T


def nudged_change(after_n, before, sil, r_px):
    """Share of `sil` changed when each NUDGE_BAND_PX band of it may shift by up to r_px (its best shift). A nudged
    stack lines up again; a fallen or removed box does not (its band changes whatever the shift)."""
    ys, xs = np.nonzero(sil)
    if not len(ys):
        return 0.0
    x0, x1 = xs.min(), xs.max() + 1; n = 0
    for r0 in range(ys.min(), ys.max() + 1, NUDGE_BAND_PX):
        r1 = min(r0 + NUDGE_BAND_PX, ys.max() + 1); m = sil[r0:r1, x0:x1]
        if not m.any():
            continue
        b = before[r0:r1, x0:x1]; best = int(m.sum())
        for dy in range(-r_px, r_px + 1):
            if r0 + dy < 0 or r1 + dy > H:
                continue
            for dx in range(-r_px, r_px + 1):
                if x0 + dx < 0 or x1 + dx > W:
                    continue
                c = int((change(after_n[r0 + dy:r1 + dy, x0 + dx:x1 + dx], b, shadows=True) & m).sum())
                best = min(best, c)
        n += best
    return n / int(sil.sum())


def turn_check(cam, ref, before, after, z_top, top_type, new_type, arm_mask=None, top_xy0=None, top_known=False):
    """(ok, reasons, f). ok: the new box lies flat on the old top (fit >= TURN_FIT_MIN, centre within TURN_OFF_MAX of
    the old top) and no other box-sized change anywhere on the table or in the stack. top_known: top_xy0 is the old
    top's centre from the game state (the previous turn's result or the robot's wrist look) and is used as is;
    otherwise it is a guess refined by a top-only fit in `before`."""
    roi = table_region(cam, arm_mask)
    static = static_region(cam, arm_mask)
    ref_n = photometric_match(ref, before, static)
    fg_b = change(before, ref_n, shadows=True) & stack_region(cam, arm_mask=arm_mask)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg_b.astype(np.uint8))
    if n > 2:
        fg_b = lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    xy0 = SPOT if top_xy0 is None else np.asarray(top_xy0, float); s0, yaw0 = 0.0, 0.0
    if top_type is not None and fg_b.sum() > 50:
        s0, xy_f, yaw0 = fit_box(cam, fg_b, z_top - BOX_DIMS[top_type]["height"], BOX_DIMS[top_type], xy0, top_only=True)
        if top_known:
            s0 = max(s0, TURN_TOP_FIT_MIN)                 # (the yaw is still from the fit, for the cover test)
        elif s0 >= TURN_TOP_FIT_MIN:                       # trust the old-top fit only when it is good
            xy0 = xy_f
    after_n = photometric_match(after, before, static)
    d = change(after_n, before, shadows=True) & roi
    if new_type is None:                                  # no box placed (e.g. a judged settle window): nothing may change
        s, xy, yaw = 1.0, xy0, 0.0; expl = np.zeros((H, W), bool); stick_rows = 0; tall = None
    else:
        dn = BOX_DIMS[new_type]
        near = prism_mask(cam, xy0, TURN_NEAR_R, z_top - 0.005, z_top + dn["height"] + 0.01)   # where the new box can be
        s, xy, yaw = fit_box(cam, d & near, z_top, dn, xy0, r=0.045)
        hull = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(hull, np.round(box_hull(cam, xy, yaw, z_top, dn)).astype(np.int32), 1)
        expl = cv2.dilate(hull, np.ones((13, 13), np.uint8)).astype(bool)
        h0 = np.zeros((H, W), np.uint8)                       # a box standing on its end sticks up above a flat one
        cv2.fillConvexPoly(h0, np.round(box_hull(cam, xy0, yaw, z_top, dn)).astype(np.int32), 1)   # at the OLD top's
        top_row = np.nonzero(h0.any(1))[0].min()              # centre (the fit can slide back to cover an upright box)
        tall = prism_mask(cam, xy0, TURN_NEAR_R, z_top, z_top + STICK_COL_H)    # (the fit's column ends 1 cm above)
        stick_rows = max(0, top_row - STICK_MARGIN_PX)
    # did the old stack move? share of its silhouette (before vs ref), outside the new box AND the column under its
    # footprint (an overhanging new box shades and hides the sides below it: sim game 29 Oct, no3 on a no35), that changed
    under = expl.copy()
    if new_type is not None:
        c_, s_ = np.cos(yaw), np.sin(yaw); L_, W_ = dn["length"] / 2, dn["width"] / 2
        foot = [xy + a * L_ * np.r_[c_, s_] + b * W_ * np.r_[-s_, c_] for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        col_pts = cam.project(np.array([np.r_[p_, z] for p_ in foot for z in (0.0, z_top + dn["height"])]))
        colm = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(colm, cv2.convexHull(np.round(col_pts).astype(np.int32)), 1)
        under |= cv2.dilate(colm, np.ones((9, 9), np.uint8)).astype(bool)
    # anything else changed? new or vanished EDGES (exposure drift leaves edges alone; a fallen or moved box adds them)
    ea, eb = edges(after_n), edges(before)
    k3 = np.ones((5, 5), np.uint8)
    ue = ((ea & ~cv2.dilate(eb.astype(np.uint8), k3).astype(bool)) | (eb & ~cv2.dilate(ea.astype(np.uint8), k3).astype(bool))) & roi & ~under
    newe = ea & ~cv2.dilate(eb.astype(np.uint8), k3).astype(bool)   # new edges above a flat box at the old top
    stick = int((newe & tall)[:stick_rows].sum()) if tall is not None else 0
    nb, lb, sb, _ = cv2.connectedComponentsWithStats(cv2.dilate(ue.astype(np.uint8), np.ones((9, 9), np.uint8)))
    blob = int(max((ue[lb == i].sum() for i in range(1, nb)), default=0))
    blob_max = TURN_EDGE_BLOB
    old_sil = fg_b & ~under
    stack_change = float(d[old_sil].mean()) if old_sil.sum() > 50 else 0.0
    stack_change_raw = stack_change
    if stack_change > TURN_STACK_CHANGE:                  # moved, or only nudged a few mm?
        uv = cam.project(np.array([np.r_[xy0, z_top], np.r_[xy0 + [NUDGE_MM / 1000, 0], z_top], np.r_[xy0 + [0, NUDGE_MM / 1000], z_top]]))
        r_px = int(np.ceil(max(np.linalg.norm(uv[1] - uv[0]), np.linalg.norm(uv[2] - uv[0]))))
        stack_change = nudged_change(after_n, before, old_sil, r_px)
    off = float(np.linalg.norm(np.asarray(xy) - xy0))
    cover = None                                          # old top face covered? (a box beside it leaves it visible)
    if new_type is not None and top_type is not None and s0 >= TURN_TOP_FIT_MIN:
        face = np.zeros((H, W), np.uint8); dm = BOX_DIMS[top_type]; c_, s_ = np.cos(yaw0), np.sin(yaw0)
        cr = [np.r_[xy0 + a * dm["length"] / 2 * np.r_[c_, s_] * 0.8 + b * dm["width"] / 2 * np.r_[-s_, c_] * 0.8, z_top]
              for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        cv2.fillConvexPoly(face, np.round(cam.project(np.array(cr))).astype(np.int32), 1); face = face.astype(bool) & roi
        cover = float(d[face].mean()) if face.sum() > 20 else None
    f = dict(fit=float(s), off_mm=off * 1000, blob=blob, blob_max=float(blob_max), changed=int(d.sum()),
             top_fit=float(s0), cover=cover, stack_change=stack_change, stack_change_raw=stack_change_raw, stick_px=stick,
             xy_new=np.round(xy, 4).tolist(), yaw_new=float(np.degrees(yaw)), xy_top=np.round(xy0, 4).tolist())
    hard = blob > TURN_BLOB_SOFT or stack_change > TURN_STACK_CHANGE or (cover is not None and cover < TURN_COVER_MIN and z_top < TURN_COVER_ZMAX)
    # unsure: the new box is near the old top (<= TURN_OFF_SOFT) and the stack did not move, but the fit is weak (a box
    # squarely on a look-alike box barely changes the image: sim game 30 Oct) or the edges changed moderately (a base
    # box shifted a few mm, allowed): the game asks the player instead of calling a loss
    soft_fail = s < TURN_FIT_MIN or off > TURN_OFF_MAX or blob > blob_max or stick > STICK_MAX_PX
    f["unsure"] = bool(soft_fail and not hard and s >= TURN_FIT_SOFT and off <= TURN_OFF_SOFT)
    reasons = []
    if top_type is None:                                 # the first box onto the empty base: anywhere on it is fine
        off = 0.0; f["unsure"] = bool(f["unsure"] and s < TURN_FIT_MIN)
    if s < TURN_FIT_MIN:
        reasons.append(f"no box lying flat on the top (fit {s:.2f})")
    elif off > TURN_OFF_MAX:
        reasons.append(f"new box {off * 1000:.0f} mm off the top centre")
    if blob > blob_max:
        reasons.append(f"other change on the table/stack ({blob} edge px)")
    if stick > STICK_MAX_PX:
        reasons.append(f"new box not lying flat ({stick} px above a flat box)")
    if stack_change > TURN_STACK_CHANGE:
        reasons.append(f"old stack moved ({stack_change:.0%} of it changed)")
    if cover is not None and cover < TURN_COVER_MIN and z_top < TURN_COVER_ZMAX:
        reasons.append(f"old top still uncovered ({cover:.0%} of it changed)")
    return not reasons, reasons, f
