"""Phase 3: calibrate the `front` (and `wrist`) cameras against the real reference frames.

Front camera model: pinhole + radial distortion k1 (k2 = 0), principal point at the image
centre. Fitted to image points whose world positions come from the Phase 3 layout:
  - corners of the yellow stack area and the black supply mat (detected by colour)
  - the stacking-spot X
  - the closed jaw tips at home (3D, from the validated mapping; breaks the planar degeneracy)
plus a weak prior on the tape-measured camera position (measurements.yaml, sigma 3 cm).

MuJoCo renders pinhole only, so sim images are undistorted by construction; real frames
are undistorted with the same (f, k1, k2) before comparison (and before the policy, see
notes/phase3.md).

    ~/.venvs/so101_stack_sim/bin/python scripts/calibrate_cameras.py
Order: scene_layout.py -> calibrate_cameras.py (also places the table from the image) -> build_scene.py
"""

import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
LAY = yaml.safe_load((ROOT / "sim/assets/layout.yaml").read_text())
MEAS = yaml.safe_load((ROOT / "inputs/measurements.yaml").read_text())
OUT = ROOT / "sim/assets/cameras.yaml"
W, H = 640, 480
FRONT_EMPTY = ROOT / "inputs/photos/front_cam_frame.png"
FRONT_STACK = ROOT / "inputs/photos/front_cam_stack3.png"

# hand-picked on a 4x zoom of front_cam_frame.png (+/- ~1.5 px); Phase 3, 2026-10-21
PICKED = {"jaw_tips_home": (309.0, 267.0), "stacking_spot_x": None}


def quad(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    i = 1 + np.argmax(st[1:, 4])
    m = (lab == i).astype(np.uint8) * 255
    cnt = max(cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    for eps in np.linspace(0.005, 0.05, 40):
        ap = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(ap) == 4:
            break
    return refine_corners(m, ap.reshape(-1, 2).astype(np.float64))


def refine_corners(mask, corners):
    """Intersect lines fitted to each side's boundary pixels (sub-pixel, robust to rounded corners)."""
    cnt = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea).reshape(-1, 2)
    lines = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        d = b - a; L = np.linalg.norm(d); d /= L; nrm = np.array([-d[1], d[0]])
        t = (cnt - a) @ d; off = np.abs((cnt - a) @ nrm)
        sel = cnt[(t > 0.15 * L) & (t < 0.85 * L) & (off < 6)]
        vx, vy, x0, y0 = cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    out = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        s = np.linalg.solve(np.c_[d1, -d2], p2 - p1)
        out.append(p1 + s[0] * d1)
    return np.array(out)


def match_by_distance(img_pts, world_xy, cam_xy):
    """Order image corners like world corners: nearest-to-camera = lowest in the image, etc.
    Uses the cyclic order of the image quad and picks the rotation/flip that best matches
    the ranking of world distances from the camera to image row (y)."""
    best = None
    dist = np.linalg.norm(world_xy - cam_xy, axis=1)
    for flip in (False, True):
        seq = img_pts[::-1] if flip else img_pts
        for r in range(4):
            cand = np.roll(seq, r, axis=0)
            score = np.corrcoef(-dist, cand[:, 1])[0, 1]       # nearer -> larger y (lower in image)
            # also require same handedness: +y world (left of arm) appears to the image right
            hand = np.corrcoef(world_xy[:, 1], cand[:, 0])[0, 1]
            sc = score + hand
            if best is None or sc > best[0]:
                best = (sc, cand)
    return best[1]


def jaw_tip_home_world():
    m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    pts = []
    for mesh, body in (("wrist_roll_follower_so101_v1", "gripper"), ("moving_jaw_so101_v1", "moving_jaw_so101_v1")):
        g = [i for i in range(m.ngeom) if m.geom_group[i] == 2 and m.geom_dataid[i] >= 0
             and m.mesh(m.geom_dataid[i]).name == mesh][0]
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
        pts.append(w[np.argmin(w[:, 2])])        # lowest point of each jaw = its tip at home (jaws ~vertical)
    return np.mean(pts, axis=0)


def project(P, rvec, pos, f, k1, k2):
    R, _ = cv2.Rodrigues(rvec)
    pc = (P - pos) @ R.T
    xy = pc[:, :2] / pc[:, 2:3]
    r2 = np.sum(xy ** 2, axis=1, keepdims=True)
    return f * xy * (1 + k1 * r2 + k2 * r2 ** 2) + np.array([W / 2, H / 2])


def calibrate_front():
    tape = MEAS["cameras"]["front"]["position"]; tape = np.array([tape["x"], tape["y"], tape["z"]])
    empty = cv2.cvtColor(cv2.imread(str(FRONT_EMPTY)), cv2.COLOR_BGR2HSV)
    stack = cv2.cvtColor(cv2.imread(str(FRONT_STACK)), cv2.COLOR_BGR2HSV)
    yel = quad(cv2.inRange(empty, (15, 90, 120), (35, 255, 255)))          # clean yellow in the empty frame
    dk = cv2.inRange(stack, (0, 0, 0), (180, 255, 60)); dk[:250] = 0; dk[:, :300] = 0   # mat in the stack frame
    blk = quad(dk)
    Kw = np.array(LAY["stack_area"]["corners"]); Sw = np.array(LAY["supply_area"]["corners"])
    yel = match_by_distance(yel, Kw, tape[:2]); blk = match_by_distance(blk, Sw, tape[:2])
    tip = jaw_tip_home_world()
    names = [f"stack_corner_{i+1}" for i in range(4)] + [f"supply_corner_{i+1}" for i in range(4)] + ["jaw_tips_home"]
    world = np.vstack([np.c_[Kw, np.full(4, 0.0002)], np.c_[Sw, np.full(4, 0.002)], tip[None]])
    img = np.vstack([yel, blk, np.array(PICKED["jaw_tips_home"])[None]])
    wts = np.r_[np.ones(8), 0.7]

    look = np.array([LAY["table"]["centre"][0], LAY["table"]["centre"][1], 0]) - tape
    z = look / np.linalg.norm(look); x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x); y = np.cross(z, x)
    rvec0 = cv2.Rodrigues(np.stack([x, y, z]))[0].ravel()

    def res(p):
        r = ((project(world, p[:3], p[3:6], p[6], p[7], p[8]) - img) * wts[:, None]).ravel()
        return np.r_[r, (p[3:6] - tape) / 0.03 * 2.0]
    # k2 held at 0: freeing it gave the same rms (2.73 px) with k2 stuck at its bound (overfit, Phase 3 log)
    lo = np.r_[[-np.inf] * 3, tape - 0.15, 250, -0.6, -1e-9]; hi = np.r_[[np.inf] * 3, tape + 0.15, 1500, 0.3, 1e-9]
    sol = least_squares(res, np.r_[rvec0, tape, 600, -0.1, 0.0], bounds=(lo, hi))
    p = sol.x
    pix = project(world, p[:3], p[3:6], p[6], p[7], p[8]) - img
    err = np.linalg.norm(pix, axis=1)
    R, _ = cv2.Rodrigues(p[:3])
    cam = dict(pos=p[3:6].round(4).tolist(), xyaxes=np.r_[R[0], -R[1]].round(5).tolist(),
               fovy=float(np.degrees(2 * np.arctan(H / 2 / p[6]))), f_px=float(p[6]), k1=float(p[7]), k2=float(p[8]),
               rvec=p[:3].tolist(), rms_px=float(np.sqrt(np.mean(err ** 2))),
               residual_px={n: round(float(e), 1) for n, e in zip(names, err)},
               tape_offset_m=float(np.linalg.norm(p[3:6] - tape)),
               points=dict(zip(names, img.round(1).tolist())))
    return cam


# table-top corners hand-picked on 4x zooms of front_cam_frame.png (the yellow-side corner is
# hidden by the backdrop). The table is placed from these because the edge touches sit
# inside the real edge (the jaw tips can't touch the very edge).
TABLE_PICKS = {"far": (321.2, 468.7), "left": (565.0, 296.0)}


def backproject(cam, u, v, z=0.0):
    f, k1 = cam["f_px"], cam["k1"]
    R, _ = cv2.Rodrigues(np.array(cam["rvec"])); pos = np.array(cam["pos"])
    xd = (np.array([u, v]) - [W / 2, H / 2]) / f; x = xd.copy()
    for _ in range(20):
        x = xd / (1 + k1 * np.sum(x ** 2))
    ray = R.T @ np.r_[x, 1.0]
    return pos + (z - pos[2]) / ray[2] * ray


def table_from_image(cam):
    far = backproject(cam, *TABLE_PICKS["far"])[:2]; left = backproject(cam, *TABLE_PICKS["left"])[:2]
    size = MEAS["table"]["size_x"]
    e = (left - far) / np.linalg.norm(left - far)
    n = np.array([-e[1], e[0]])
    if n @ (-far) < 0:          # inward normal points toward the arm (origin)
        n = -n
    mid = (far + left) / 2
    F, Lc = mid - e * size / 2, mid + e * size / 2
    corners = {"far": F, "left": Lc, "mount": Lc + n * size, "right": F + n * size}
    return {"size_m": size, "centre": (mid + n * size / 2).round(4).tolist(),
            "heading_deg": float(np.degrees(np.arctan2(e[1], e[0]))),
            "corners": {k: v.round(4).tolist() for k, v in corners.items()},
            "source": "front_cam_frame.png corner picks back-projected with the calibrated front camera",
            "picked_side_length_m": float(np.linalg.norm(left - far))}


# ---------------- wrist camera ----------------
# Picks (640x480) on 20-px grids; world points from the Phase 3 layout / Phase 1 M3 fit.
# The fixed jaw tip lands on the same pixel in all three frames (camera rigid on the gripper).
WRIST_FRAMES = {
    "home": dict(image="inputs/photos/wrist_cam_frame.png", pose="home", gripper=0.74, points=[
        ("supply_corner_4", (213, 268), 1.0), ("mark_C", (316, 240), 0.4),
        ("tip_fixed", (318, 322), 1.0), ("tip_moving", (352, 333), 0.7)]),
    "P1": dict(image="inputs/photos/phase1/wrist_cam_P1.png", pose="notes/phase1_poses/P1_over_stack.json", gripper=30.0, points=[
        ("stacking_spot", (322, 237), 1.0), ("stack_corner_4", (612, 97), 1.0), ("stack_corner_2", (4, 386), 0.4),
        ("tip_fixed", (318, 322), 1.0)]),
    "P3": dict(image="inputs/photos/phase1/wrist_cam_P3.png", pose="notes/phase1_poses/P3_over_stack_roll45.json", gripper=30.0, points=[
        ("stacking_spot", (262, 193), 1.0), ("stack_corner_4", (528, 276), 1.0), ("tip_fixed", (320, 325), 1.0)]),
}
MARK_C_WORLD = (0.178, 0.024, 0.0)     # Phase 1 M3 fit (roll +95), +/- ~5 mm
WRIST_F_SPEC = 518.0                   # px, Orbbec Gemini 2 datasheet RGB 4:3 FOV 63 x 50 deg at 640x480


def wrist_world_points():
    K = np.array(LAY["stack_area"]["corners"]); S = np.array(LAY["supply_area"]["corners"])
    pts = {f"stack_corner_{i+1}": np.r_[K[i], 0.0002] for i in range(4)}
    pts.update({f"supply_corner_{i+1}": np.r_[S[i], 0.002] for i in range(4)})
    # the wrist frames (M4, 2026-10-20) show the ORIGINAL X, not the moved game spot
    pts["stacking_spot"] = np.r_[LAY["stack_area"].get("stacking_spot_original", LAY["stack_area"]["stacking_spot"]), 0.0002]
    pts["mark_C"] = np.array(MARK_C_WORLD)
    return pts


def gripper_poses():
    """Gripper-body pose and the two jaw-tip points (in gripper frame) for each wrist frame."""
    import json
    sys.path.insert(0, str(ROOT / "scripts"))
    from joint_mapping import JOINT_NAMES, JointMapping, from_dict
    m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/scene.xml")); d = mujoco.MjData(m)
    jm = JointMapping.from_files()
    gb = m.body("gripper").id
    out = {}
    for name, fr in WRIST_FRAMES.items():
        q = MEAS["arm"]["home_pose"] if fr["pose"] == "home" else json.loads((ROOT / fr["pose"]).read_text())
        q = dict(q, gripper=fr["gripper"])
        mujoco.mj_resetData(m, d)
        for j, v in zip(JOINT_NAMES, jm.real_to_sim(from_dict(q))):
            d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
        mujoco.mj_forward(m, d)
        Rg = d.xmat[gb].reshape(3, 3).copy(); xg = d.xpos[gb].copy()
        tips = {}
        for key, mesh in (("tip_fixed", "wrist_roll_follower_so101_v1"), ("tip_moving", "moving_jaw_so101_v1")):
            g = [i for i in range(m.ngeom) if m.geom_group[i] == 2 and m.mesh(m.geom_dataid[i]).name == mesh][0]
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
            loc = (w - xg) @ Rg                    # gripper frame
            # jaw tip = the vertex farthest along the jaw direction (gripper -z)
            tips[key] = loc[np.argmin(loc[:, 2])]
        out[name] = (Rg, xg, tips)
    return out


def calibrate_wrist():
    P = wrist_world_points(); G = gripper_poses()
    obs = []
    for name, fr in WRIST_FRAMES.items():
        Rg, xg, tips = G[name]
        for key, uv, w in fr["points"]:
            pg = tips[key] if key.startswith("tip_") else (P[key] - xg) @ Rg       # point in gripper frame
            obs.append((name, key, pg, np.array(uv, float), w))

    def proj(p, pg):
        Rc, _ = cv2.Rodrigues(p[:3]); c = p[3:6]; f, k1 = p[6], p[7]
        pc = Rc.T @ (pg - c)                       # camera frame (OpenCV: x right, y down, z forward)
        if pc[2] <= 1e-4:
            return np.array([1e4, 1e4])
        xy = pc[:2] / pc[2]; r2 = xy @ xy
        return f * xy * (1 + k1 * r2) + np.array([W / 2, H / 2])

    def res(p):
        r = [(proj(p, pg) - uv) * w for _, _, pg, uv, w in obs]
        # prior: lens ~ measurements.yaml offset (3.6 cm back from the fixed tip, ~1.3 cm to the side), 1.5 cm sigma
        tipf = G["home"][2]["tip_fixed"]
        r.append([(np.linalg.norm(p[3:6] - tipf) - np.hypot(0.036, 0.013)) / 0.015 * 3])
        return np.concatenate(r)

    best = None
    tipf = G["home"][2]["tip_fixed"]
    for side in ([0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]):
        c0 = tipf + np.array([0, 0, 0.036]) + 0.013 * np.array(side)
        for down in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]):
            zc = np.array([0, 0, -1.0]); yc = np.array(down, float)
            if abs(yc @ zc) > 0.5:
                continue
            xc = np.cross(yc, zc)
            rv0 = cv2.Rodrigues(np.c_[xc, yc, zc])[0].ravel()
            for f0 in (300.0, 450.0, 600.0):
                try:
                    sol = least_squares(res, np.r_[rv0, c0, f0, -0.2],
                                        bounds=(np.r_[[-np.inf] * 6, 150, -0.8], np.r_[[np.inf] * 6, 1500, 0.3]))
                except ValueError:
                    continue
                if best is None or sol.cost < best.cost:
                    best = sol
    p = best.x
    Rc, _ = cv2.Rodrigues(p[:3])
    errs = {f"{n}:{k}": round(float(np.linalg.norm(proj(p, pg) - uv)), 1) for n, k, pg, uv, _ in obs}
    e = np.array(list(errs.values()))
    return dict(body="gripper", pos=p[3:6].round(5).tolist(), xyaxes=np.r_[Rc[:, 0], -Rc[:, 1]].round(5).tolist(),
                fovy=float(np.degrees(2 * np.arctan(H / 2 / p[6]))), f_px=float(p[6]), k1=float(p[7]), k2=0.0,
                rvec_in_gripper=p[:3].tolist(), rms_px=float(np.sqrt(np.mean(e ** 2))), residual_px=errs,
                lens_to_fixed_tip_m=float(np.linalg.norm(p[3:6] - tipf)),
                optical_axis_vs_jaw_deg=float(np.degrees(np.arccos(np.clip(Rc[:, 2] @ np.array([0, 0, -1.0]), -1, 1)))))


def real_jaw_mask(path):
    """Dark blob touching the bottom of the wrist image, near the centre = the jaws."""
    hsv = cv2.cvtColor(cv2.imread(str(ROOT / path)), cv2.COLOR_BGR2HSV)
    dark = (hsv[:, :, 2] < 75).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(dark)
    keep = np.zeros_like(dark)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if y + h >= H - 2 and area > 2000 and abs(cen[i][0] - 330) < 200 and cen[i][1] > 330:
            keep[lab == i] = 255
    return keep


def calibrate_wrist_silhouette(init):
    """Fit the wrist camera (pose in the gripper frame, f, k1) so the rendered jaws match the
    real jaw silhouettes. Only the gripper opening matters (camera and jaws ride on the gripper),
    so arm-pose errors don't enter. Runs at half resolution."""
    from scipy.optimize import minimize
    m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/scene.xml")); d = mujoco.MjData(m)
    sys.path.insert(0, str(ROOT / "scripts"))
    from joint_mapping import JointMapping
    jm = JointMapping.from_files()
    cid = m.camera("wrist").id
    jaw_geoms = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in ("gripper", "moving_jaw_so101_v1")}
    s = 0.25; w2, h2 = int(W * s), int(H * s)
    r = mujoco.Renderer(m, h2, w2); r.enable_segmentation_rendering()
    frames = []
    for name, fr in WRIST_FRAMES.items():
        g_sim = jm.real_to_sim(np.r_[np.zeros(5), fr["gripper"]])[5]
        frames.append((name, g_sim, real_jaw_mask(fr["image"])))

    def set_cam(p):
        Rc, _ = cv2.Rodrigues(p[:3])
        Rm = np.c_[Rc[:, 0], -Rc[:, 1], -Rc[:, 2]]          # MuJoCo camera: x right, y up, looks along -z
        q = np.zeros(4); mujoco.mju_mat2Quat(q, Rm.ravel())
        m.cam_pos[cid] = p[3:6]; m.cam_quat[cid] = q
        m.cam_fovy[cid] = np.degrees(2 * np.arctan(H / 2 / p[6]))

    def sim_mask(g_sim):
        d.qpos[:] = 0; d.qpos[m.jnt_qposadr[m.joint("gripper").id]] = g_sim
        mujoco.mj_forward(m, d)
        r.update_scene(d, "wrist")
        seg = r.render()
        ids = seg[:, :, 0]; typ = seg[:, :, 1]
        return np.isin(ids, list(jaw_geoms)) & (typ == mujoco.mjtObj.mjOBJ_GEOM)

    def loss(p, detail=False):
        set_cam(p)
        K = np.array([[p[6], 0, W / 2], [0, p[6], H / 2], [0, 0, 1]])
        tot, out = 0.0, {}
        for name, g_sim, real in frames:
            und = cv2.undistort(real, K, np.array([p[7], 0, 0, 0]))
            und = cv2.resize(und, (w2, h2), interpolation=cv2.INTER_AREA) > 127
            sm = sim_mask(g_sim)
            iou = (sm & und).sum() / max(1, (sm | und).sum())
            tot += 1 - iou; out[name] = round(float(iou), 3)
        return (tot, out) if detail else tot

    def bounded(p):                 # keep k1 realistic and f sane; penalise leaving the box
        # Orbbec Gemini 2 RGB, 4:3 mode: FOV 63 x 50 deg -> f ~ 522 / 515 px at 640x480 (datasheet); +/- 5 %
        pen = max(0, -0.45 - p[7]) + max(0, p[7] - 0.1) + max(0, abs(p[6] - WRIST_F_SPEC) - 26) / 10
        return loss(p) + 10 * pen
    best = None
    for x0 in init:
        sol = minimize(bounded, x0, method="Powell", options=dict(maxiter=1500, xtol=1e-4, ftol=1e-4))
        if best is None or sol.fun < best.fun:
            best = sol
    p = best.x
    _, ious = loss(p, detail=True)
    r.close()
    return p, ious


def wrist_inits():
    """Starting points: lens 6-11 cm back from the fixed tip (the measurements.yaml 3.6 cm gave a poor
    fit: the real frame shows the jaws from near their roots), 1.3 cm to the +/-y side, looking along the jaws."""
    G = gripper_poses(); tipf = G["home"][2]["tip_fixed"]
    inits = []
    for back, side in [(b_, s_) for b_ in (0.06, 0.085, 0.11) for s_ in ([0, 1, 0], [0, -1, 0])]:
        c0 = tipf + np.array([0, 0, back]) + 0.013 * np.array(side)
        to_tip = tipf - c0
        zc = np.array([0, 0, -1.0]); yc = to_tip - (to_tip @ zc) * zc; yc /= np.linalg.norm(yc)   # tips toward image bottom
        xc = np.cross(yc, zc)
        rv0 = cv2.Rodrigues(np.c_[xc, yc, zc])[0].ravel()
        inits.append(np.r_[rv0, c0, WRIST_F_SPEC, -0.05])
    return inits


def main():
    cams = yaml.safe_load(OUT.read_text()) if OUT.exists() else {}
    cams["front"] = calibrate_front()
    lay = yaml.safe_load((ROOT / "sim/assets/layout.yaml").read_text())
    lay.setdefault("table_from_touches", lay["table"])
    lay["table"] = table_from_image(cams["front"])
    (ROOT / "sim/assets/layout.yaml").write_text(yaml.safe_dump(lay, sort_keys=False))
    print(f"table placed from the image: side between picks {lay['table']['picked_side_length_m']*100:.1f} cm (38.0), "
          f"centre {np.round(np.array(lay['table']['centre'])*100,1)} cm (touches: "
          f"{np.round(np.array(lay['table_from_touches']['centre'])*100,1)})")
    OUT.write_text(yaml.safe_dump(cams, sort_keys=False))
    f = cams["front"]
    print(f"front: f={f['f_px']:.0f}px fovy={f['fovy']:.1f} k1={f['k1']:.3f} k2={f['k2']:.3f} rms={f['rms_px']:.2f}px "
          f"pos={f['pos']} ({f['tape_offset_m']*100:.1f} cm from tape)")
    print("  residuals px:", f["residual_px"])
    if "--front-only" in sys.argv:      # keep the existing wrist entry (the silhouette fit takes ~20 min)
        OUT.write_text(yaml.safe_dump(cams, sort_keys=False))
        return
    p, ious = calibrate_wrist_silhouette(wrist_inits())
    Rc, _ = cv2.Rodrigues(p[:3])
    tipf = gripper_poses()["home"][2]["tip_fixed"]
    w = cams["wrist"] = dict(body="gripper", pos=p[3:6].round(5).tolist(), xyaxes=np.r_[Rc[:, 0], -Rc[:, 1]].round(5).tolist(),
                             fovy=float(np.degrees(2 * np.arctan(H / 2 / p[6]))), f_px=float(p[6]), k1=float(p[7]), k2=0.0,
                             rvec_in_gripper=p[:3].tolist(), jaw_silhouette_iou=ious,
                             lens_to_fixed_tip_m=float(np.linalg.norm(p[3:6] - tipf)),
                             optical_axis_vs_jaw_deg=float(np.degrees(np.arccos(np.clip(Rc[:, 2] @ [0, 0, -1.0], -1, 1)))),
                             method="jaw-silhouette fit on wrist_cam_frame + M4 P1/P3 wrist frames")
    w["rms_px"] = None
    OUT.write_text(yaml.safe_dump(cams, sort_keys=False))
    print(f"wrist: f={w['f_px']:.0f}px fovy={w['fovy']:.1f} k1={w['k1']:.3f} jaw IoU {w['jaw_silhouette_iou']} "
          f"lens {w['lens_to_fixed_tip_m']*100:.1f} cm from the fixed tip, axis {w['optical_axis_vs_jaw_deg']:.1f} deg off the jaws")


if __name__ == "__main__":
    main()
