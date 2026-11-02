"""Look at the stack with the wrist camera before the pick: the top box's centre and yaw for the scripted place.

Why: on 3-4 box real stacks (29 Oct) the front-camera fit was weak (IoU 0.48-0.83) and the landing correction
measured at height 1 did not transfer: boxes were released 2-3 cm off although FK put them within 2 mm. The wrist
camera is calibrated (ChArUco, sim/assets/cameras.yaml), sees the top face from ~12 cm, and at the look pose the
arm is close to its release pose, so most of the FK error is shared by the measurement and the place.

Look pose: IK so the wrist camera's line of sight goes through the guessed top centre from LOOK_H above the face,
looking as straight down as the joints allow (the camera is tilted on the gripper).
Detector: the top face is a rectangle of known size at the known height z_top. Every candidate (x, y, yaw) is
projected into the wrist image with the calibrated model (real: principal point, k1 k2; sim renders: pinhole,
centred); score = mean gradient across the four edges (gradient . edge normal, sampled along each edge, inset from
the corners). Grid search around the guess, then Nelder-Mead. Lower boxes' edges sit at other heights and project
at other sizes, so they do not fit as well.
"""
import cv2
import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares, minimize

from perception.camera import ROOT
from perception.supply import BOX

LOOK_H = 0.15                  # wrist camera above the top face (m); 12 cm left real stacks cut off at the image top
LOOK_UV = np.array([320.0, 200.0])   # where the top centre should appear: above the jaws (real 29 Oct 18:04: boxes
                                     # appeared 80-150 px higher than aimed, front-estimate + arm error)
VIS_MARGIN_PX = 8
LOOK_GRIP = 0.0                # gripper closed during the look (open, the moving jaw swings into the backdrop: sim -22 mm
                               # at 40); the jaws in view are masked out (arm_mask)
BACKDROP_CLEAR = 0.03          # look pose: wrist/gripper geoms at least this far from the backdrop (soft constraint)
ARM_MARGIN_PX = 6
LOOK_SEED = np.array([27.7, -28.6, 50.0, 49.1, 5.4])   # look pose over the spot at height 1 (solved 29 Oct)
W_TILT = 40.0                  # IK: mm per unit of camera-axis error (soft: the camera is tilted on the gripper)
SEARCH_R, SEARCH_STEP = 0.03, 0.002
YAW_STEP = np.radians(5)
N_EDGE = 24                    # samples per edge
CAM = yaml.safe_load((ROOT / "sim/assets/cameras.yaml").read_text())["wrist"]


class StackLook:
    def __init__(self, kin):
        self.kin = kin; self.cid = kin.m.camera("wrist").id; self.lo, self.hi = kin.jm.lerobot_limits().T
        m = kin.m; self.bd = m.geom("backdrop").id; wb = m.body("wrist").id
        sub = {wb}
        for b in range(m.nbody):                                                # bodies below the wrist
            if m.body_parentid[b] in sub:
                sub.add(b)
        self.wrist_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in sub and (m.geom_contype[g] | m.geom_conaffinity[g])]

    def cam_pose(self, q):
        """World pose of the wrist camera (MuJoCo convention: looks along -z, y up) for a LeRobot reading."""
        self.kin.tip_pos(q); mujoco.mj_camlight(self.kin.m, self.kin.d)        # mj_kinematics leaves cameras stale
        return self.kin.d.cam_xpos[self.cid].copy(), self.kin.d.cam_xmat[self.cid].reshape(3, 3).copy()

    def look_pose(self, q_seed, xy, z_top, g=None, h=None):
        """Body joints putting the wrist camera h (default LOOK_H) above (xy, z_top), the point at LOOK_UV."""
        tgt = np.r_[xy, z_top]; g = LOOK_GRIP if g is None else g; h = LOOK_H if h is None else h

        def r(x):
            qq = np.r_[x, g]; p, R = self.cam_pose(qq); a = -R[:, 2]          # viewing direction
            depth = (tgt - p) @ a
            uv = self.project(tgt[None], qq, real=True)[0] if depth > 0.03 else LOOK_UV + 2000
            clear = min(mujoco.mj_geomDistance(self.kin.m, self.kin.d, gg, self.bd, 0.1, None) for gg in self.wrist_geoms)
            return np.r_[(uv - LOOK_UV) * 0.5, (p[2] - tgt[2] - h) * 1000, (a - [0, 0, -1]) * W_TILT,
                         (p[:2] - tgt[:2]) * 200,                              # camera roughly above (soft)
                         max(0.0, BACKDROP_CLEAR - clear) * 3000]              # keep off the backdrop
        best = None
        for seed in [q_seed[:5]] + ([self._last[:5]] if getattr(self, "_last", None) is not None else []) + [LOOK_SEED]:
            sol = least_squares(r, np.clip(seed, self.lo[:5] + 0.1, self.hi[:5] - 0.1), bounds=(self.lo[:5], self.hi[:5]))
            p_, R_ = self.cam_pose(np.r_[sol.x, g])
            if (tgt - p_) @ -R_[:, 2] <= 0.03:                                  # looking away from the stack
                continue
            if best is None or sol.cost < best.cost:
                best = sol
        self._last = np.r_[best.x, g]
        p, R = self.cam_pose(np.r_[best.x, g]); a = -R[:, 2]
        tilt = np.degrees(np.arccos(np.clip(-a[2], -1, 1))); uv = self.project(tgt[None], np.r_[best.x, g], real=True)[0]
        clear = min(mujoco.mj_geomDistance(self.kin.m, self.kin.d, gg, self.bd, 0.1, None) for gg in self.wrist_geoms)
        return np.r_[best.x, g], dict(tilt_deg=round(float(tilt), 1), target_px=np.round(uv, 0).tolist(),
                                      backdrop_mm=round(float(clear * 1000), 1),
                                      height_mm=round(float((p[2] - z_top) * 1000), 1))

    def arm_mask(self, q, real=True):
        """Pixels of the arm itself (jaws, camera housing) in the wrist view at reading q: segmentation render of the
        twin (pinhole; for real frames the distortion is small near the jaws), dilated by ARM_MARGIN_PX."""
        if not hasattr(self, "_ren"):
            m = self.kin.m; self._ren = mujoco.Renderer(m, 480, 640); self._ren.enable_segmentation_rendering()
            root = m.body_rootid[m.body("base").id]
            self._armgeom = np.array([m.body_rootid[m.geom_bodyid[i]] == root for i in range(m.ngeom)] + [False])
        self.kin.tip_pos(q); mujoco.mj_camlight(self.kin.m, self.kin.d)
        self._ren.update_scene(self.kin.d, "wrist"); seg = self._ren.render()[..., 0]
        mask = self._armgeom[np.where(seg < 0, len(self._armgeom) - 1, seg)].astype(np.uint8)
        return cv2.dilate(mask, np.ones((2 * ARM_MARGIN_PX + 1,) * 2, np.uint8)).astype(bool)

    def project(self, P, q, real):
        """World points (N, 3) -> wrist pixels (N, 2). real: calibrated principal point + k1 k2 (raw wrist frames);
        else the sim render's pinhole (centre, f from fovy)."""
        p, R = self.cam_pose(q); c = (np.asarray(P) - p) @ R                   # camera frame
        x, y = c[:, 0] / -c[:, 2], c[:, 1] / -c[:, 2]
        f = CAM["f_px"]
        if real:
            r2 = x * x + y * y; d = 1 + CAM["k1"] * r2 + CAM["k2"] * r2 * r2; x, y = x * d, y * d; cx, cy = CAM["cx"], CAM["cy"]
        else:
            f = 240 / np.tan(np.radians(CAM["fovy"]) / 2); cx, cy = 320.0, 240.0
        return np.c_[cx + f * x, cy - f * y]

    def detect(self, img_rgb, q, top_type, z_top, xy_guess, real=True, yaw_guess=None):
        """Top box (xy, yaw, score) in a wrist frame taken at reading q. score: mean edge gradient (grey levels)."""
        g = cv2.GaussianBlur(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (5, 5), 1.2)
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        H_, W_ = g.shape; dm = BOX[top_type]; L, Wd = dm["length"], dm["width"]
        vis = ~self.arm_mask(q, real)                                           # arm pixels: no evidence either way
        t = np.linspace(0.1, 0.9, N_EDGE)                                     # inset from the corners
        # unit-box edge samples (u along length, v along width) and outward normals
        uv = np.concatenate([np.c_[(t - 0.5) * L, np.full(N_EDGE, -Wd / 2)], np.c_[(t - 0.5) * L, np.full(N_EDGE, Wd / 2)],
                             np.c_[np.full(N_EDGE, -L / 2), (t - 0.5) * Wd], np.c_[np.full(N_EDGE, L / 2), (t - 0.5) * Wd]])
        p_cam, R_cam = self.cam_pose(q)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV); hh, ss, vv = hsv[..., 0].astype(np.int16), hsv[..., 1], hsv[..., 2]
        # not box: the yellow mat (hue ~ 20-35 in OpenCV's 0-180), the dark backdrop / jaws, or arm pixels
        notbox = ((hh >= 15) & (hh <= 38) & (ss > 90)) | (vv < 50) | ~vis
        gi = np.linspace(0.15, 0.85, 7); gj = np.linspace(0.15, 0.85, 5)
        inner = np.array([((a - 0.5) * L, (b - 0.5) * Wd) for a in gi for b in gj])

        def score_many(cands):
            """cands (K, 3): x, y, yaw -> mean |gradient . image normal| over the edge samples."""
            K = len(cands); c, s = np.cos(cands[:, 2]), np.sin(cands[:, 2])
            wx = cands[:, :1] + c[:, None] * uv[None, :, 0] - s[:, None] * uv[None, :, 1]
            wy = cands[:, 1:2] + s[:, None] * uv[None, :, 0] + c[:, None] * uv[None, :, 1]
            P = np.stack([wx, wy, np.full_like(wx, z_top)], -1).reshape(-1, 3)
            px = self.project(P, q, real).reshape(K, -1, 2)
            # image-space edge direction from neighbouring samples on the same edge -> normal
            e = px.reshape(K, 4, N_EDGE, 2); dvec = np.gradient(e, axis=2); nrm = np.stack([-dvec[..., 1], dvec[..., 0]], -1)
            nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
            ui, vi = np.clip(np.round(e[..., 0]).astype(int), 0, W_ - 1), np.clip(np.round(e[..., 1]).astype(int), 0, H_ - 1)
            inside = (e[..., 0] >= 2) & (e[..., 0] < W_ - 2) & (e[..., 1] >= 2) & (e[..., 1] < H_ - 2) & vis[vi, ui]
            val = np.abs(gx[vi, ui] * nrm[..., 0] + gy[vi, ui] * nrm[..., 1]) * inside
            n_in = inside.sum(2); per_edge = val.sum(2) / np.maximum(n_in, 1)    # (K, 4), visible samples only
            seen = n_in >= 0.3 * N_EDGE
            per_edge = np.where(seen, per_edge, np.nan)
            sc = 0.5 * np.nanmin(per_edge, 1) + 0.5 * np.nanmean(per_edge, 1)  # every visible edge must be there
            sc = np.where(seen.sum(1) >= 3, sc, 0.0)                           # at least 3 edges in view
            ix = cands[:, :1] + c[:, None] * inner[None, :, 0] - s[:, None] * inner[None, :, 1]
            iy = cands[:, 1:2] + s[:, None] * inner[None, :, 0] + c[:, None] * inner[None, :, 1]
            ip = self.project(np.stack([ix, iy, np.full_like(ix, z_top)], -1).reshape(-1, 3), q, real).reshape(K, -1, 2)
            iu, iv = np.clip(np.round(ip[..., 0]).astype(int), 0, W_ - 1), np.clip(np.round(ip[..., 1]).astype(int), 0, H_ - 1)
            bad = notbox[iv, iu].mean(1)                                        # share of the inside that is not box
            return sc * np.clip(1 - 2 * bad, 0, 1)
        xs = np.arange(-SEARCH_R, SEARCH_R + 1e-9, SEARCH_STEP)
        yaws = np.arange(0, np.pi, YAW_STEP) if yaw_guess is None else yaw_guess + np.radians(np.arange(-30, 31, 5))
        gx_, gy_, gyaw = np.meshgrid(xy_guess[0] + xs, xy_guess[1] + xs, yaws, indexing="ij")
        cands = np.c_[gx_.ravel(), gy_.ravel(), gyaw.ravel()]
        sc = np.concatenate([score_many(cands[i:i + 4000]) for i in range(0, len(cands), 4000)])
        top = cands[np.argsort(sc)[-5:]]
        best = None
        for c0 in top:
            res = minimize(lambda p: -score_many(p[None])[0], c0, method="Nelder-Mead",
                           options=dict(initial_simplex=np.vstack([c0, c0 + np.diag([0.002, 0.002, np.radians(3)])]), maxiter=120))
            if best is None or res.fun < best.fun:
                best = res
        return best.x[:2], float(best.x[2] % np.pi), float(-best.fun)

    def visible(self, q, top_type, z_top, xy, yaw, real=True):
        """(share of the top face's outline inside the image, VIS_MARGIN_PX from the border; share of that part not on
        the arm). A detection is only trusted with the whole outline inside the image (real 18:04 t5: a top box half
        out of the image was matched to its visible part) and most of it clear of the jaws."""
        dm = BOX[top_type]; c, s = np.cos(yaw), np.sin(yaw); t = np.linspace(0, 1, 30)
        L, Wd = dm["length"], dm["width"]
        uv = np.concatenate([np.c_[(t - 0.5) * L, np.full(30, sw * Wd / 2)] for sw in (-1, 1)] +
                            [np.c_[np.full(30, sl * L / 2), (t - 0.5) * Wd] for sl in (-1, 1)])
        P = np.c_[xy[0] + c * uv[:, 0] - s * uv[:, 1], xy[1] + s * uv[:, 0] + c * uv[:, 1], np.full(len(uv), z_top)]
        px = self.project(P, q, real); m = self.arm_mask(q, real)
        inside = (px[:, 0] >= VIS_MARGIN_PX) & (px[:, 0] < 640 - VIS_MARGIN_PX) & (px[:, 1] >= VIS_MARGIN_PX) & (px[:, 1] < 480 - VIS_MARGIN_PX)
        pi = px[inside].astype(int); clear = ~m[pi[:, 1], pi[:, 0]]
        return float(inside.mean()), float(clear.mean()) if len(clear) else 0.0

    def outline(self, img_rgb, q, top_type, z_top, xy, yaw, real=True, colour=(0, 255, 0)):
        """Draw the top face outline (for checking)."""
        dm = BOX[top_type]; c, s = np.cos(yaw), np.sin(yaw)
        cr = [np.r_[xy + a * dm["length"] / 2 * np.r_[c, s] + b * dm["width"] / 2 * np.r_[-s, c], z_top] for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        out = img_rgb.copy(); cv2.polylines(out, [self.project(np.array(cr), q, real).astype(np.int32)], True, colour, 2)
        return out


# ------------------------------------------------------------------------- the look routine (trial and game)
def _settle(arm, t):
    """Wait t s with the arm holding (real: sleep; sim: time passes only with commands)."""
    import time
    if getattr(arm, "virtual", False):                     # keep the last command (hold() would re-command the
        last = getattr(arm, "last_cmd", None)                 # sagged reading and let a soft arm sink further)
        for _ in range(int(t * 30)):
            arm.send(last) if last is not None else arm.hold()
    else:
        time.sleep(t)


LOOK_MIN_SCORE = 20.0       # wrist look: edge score below this is not trusted (sim: 26-150 on correct fits)
LOOK_MAX_SHIFT = 0.05       # ... nor a box more than this from the front-camera estimate (real tall stacks: 9-30 mm)
LOOK_MIN_VIS = 0.98         # ... and the whole top-box outline inside the image
LOOK_MIN_CLEAR = 0.75       # ... and most of it clear of the jaws
LOOK_REAIM_SCORE = 12.0     # a weaker detection is still good enough to re-aim the next look at
LOOK_TRIES = 6
LOOK_AGREE_M, LOOK_AGREE_DEG = 0.006, 8.0   # two good looks must agree this well (sim: no trusted look > 10 mm off)
LOOK_LIFT = 0.10            # route home -> straight up -> look pose (and back the same way)


def look_at_stack(arm, kin, home, xy_guess, yaw_guess, z_top, top_type, save_path=None, real=True, paused=None, log=print):
    """Wrist-camera look at the stack (up to LOOK_TRIES looks), then back home. Route: home -> straight up -> look
    pose; between looks a direct move from one look pose to the next (all above the stack). A look is good when the
    edge score >= LOOK_MIN_SCORE, the top-box outline is inside the image (>= LOOK_MIN_VIS) and mostly clear of the
    jaws, and it is within LOOK_MAX_SHIFT of the front estimate; the result is trusted only when a good look agrees
    with an earlier good look from a different aim (LOOK_AGREE_M / _DEG; a lock on a lower box rarely repeats). Otherwise
    re-aim at what was seen or look from higher (+4 cm). Returns dict(ok, xy, yaw_deg, score, looks, why)."""
    import time
    from skill.runner import go_home, lift_target, move_to
    sl = StackLook(kin); aim = np.asarray(xy_guess, float); h = LOOK_H; tiles = []; looks = []; best = None; cand = None; goods = []
    up = lift_target(kin, home, dz=LOOK_LIFT)            # straight up first: sim >= 29 mm from 1-5 box stacks,
    up = home.copy() if up is None else up               # >= 27 mm from the backdrop
    for k in range(LOOK_TRIES):
        q_look, info = sl.look_pose(home, aim, z_top, h=h); up[5] = q_look[5]
        why = (move_to(arm, up, kin, paused=paused) or move_to(arm, q_look, kin, paused=paused)) if k == 0 else move_to(arm, q_look, kin, paused=paused)
        if why:
            break
        for _ in range(2):                               # soft servos sag: correct the command by the reading's error
            err = q_look[:5] - arm.joints()[:5]
            if np.abs(err).max() < 0.3:
                break
            c = q_look.copy(); c[:5] += err; why = move_to(arm, c, kin, paused=paused)
            if why:
                break
        if why:
            break
        _settle(arm, 0.6); arm.wait_cameras(("wrist",)); _settle(arm, 0.15)   # a fresh frame after the arm settled
        arm.wait_cameras(("wrist",)); q = arm.joints(); img = arm.camera("wrist")
        xy, yaw, score = sl.detect(img, q, top_type, z_top, aim, real=real)
        vis, clr = sl.visible(q, top_type, z_top, xy, yaw, real=real); shift = float(np.linalg.norm(xy - np.asarray(xy_guess)))
        good = score >= LOOK_MIN_SCORE and vis >= LOOK_MIN_VIS and clr >= LOOK_MIN_CLEAR and shift <= LOOK_MAX_SHIFT
        match = next((c_ for c_ in goods if np.linalg.norm(xy - c_[0]) < LOOK_AGREE_M and
                      abs((np.degrees(yaw - c_[1]) + 90) % 180 - 90) < LOOK_AGREE_DEG), None) if good else None
        ok = match is not None; cand = match                # agrees with ANY earlier good look (sim: A,B,A,B alternation)
        looks.append(dict(aim=np.round(aim, 4).tolist(), h_mm=round(h * 1000), xy=np.round(xy, 4).tolist(), yaw_deg=round(float(np.degrees(yaw)), 1),
                          score=round(score, 1), vis=round(vis, 2), clear=round(clr, 2), shift_mm=round(shift * 1000, 1), ok=bool(ok), pose=info))
        log(f"  wrist look {k + 1}: box ({xy[0] * 100:.1f}, {xy[1] * 100:.1f}) cm, score {score:.0f}, outline in image {vis:.0%}, clear of jaws {clr:.0%}, "
              f"{shift * 1000:.0f} mm from the front estimate{'  -> trusted' if ok else ''}")
        o = sl.outline(sl.outline(img, q, top_type, z_top, np.asarray(xy_guess), yaw_guess if yaw_guess is not None else 0.0,
                                  colour=(255, 0, 255), real=real), q, top_type, z_top, xy, yaw, real=real, colour=(0, 255, 0) if ok else (255, 160, 0))
        tiles.append(o)
        if ok:                                           # two good looks from different aims agree: their mean
            xy_m = (xy + cand[0]) / 2; dy = (np.degrees(yaw - cand[1]) + 90) % 180 - 90
            best = dict(looks[-1], xy=np.round(xy_m, 4).tolist(), yaw_deg=round(float(np.degrees(cand[1]) + dy / 2), 1)); break
        if good:
            goods.append((xy, yaw)); aim = xy            # look again, aimed at it (a wrong lock rarely repeats)
        elif score >= LOOK_REAIM_SCORE and shift <= LOOK_MAX_SHIFT:
            aim = xy                                     # centre on what was seen
        else:
            h += 0.04                                    # nothing usable: wider view of the same point
    move_to(arm, up, kin, paused=paused); go_home(arm, kin, home, lift=False, paused=paused)
    if tiles and save_path is not None:
        cv2.imwrite(str(save_path), cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR))
    if best is None:
        return dict(ok=False, looks=looks, why=f"no trusted look in {len(looks)} ({'move stopped: ' + why if why else 'see looks'})")
    return dict(ok=True, xy=best["xy"], yaw_deg=best["yaw_deg"], score=best["score"], vis=best["vis"], shift_mm=best["shift_mm"],
                looks=looks, why=None)


