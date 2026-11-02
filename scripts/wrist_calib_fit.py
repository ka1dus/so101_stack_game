"""Phase 8: wrist camera calibration fit from scripts/wrist_calib_capture.py sessions.

intrinsics <handheld session>: ChArUco corners -> cv2.calibrateCamera (fx = fy, principal point free, k1 k2; no
tangential, k3 = 0). Frames with < 6 corners are skipped; frames whose own reprojection error is > 3x the median are
dropped and the fit repeated. Writes <session>/intrinsics.json and prints f, centre, k1/k2, rms, per-frame errors.

    ~/.venvs/so101_eval/bin/python scripts/wrist_calib_fit.py intrinsics data/wrist_calib/<stamp>_handheld

mount <poses session> <handheld session>: camera pose in the gripper body from the pose tour (marker corners, joint
readings through the sim FK, the intrinsics above): OpenCV hand-eye for a start, then a joint least-squares fit of the
mount (6) and the board pose (6) on the reprojection error. Compared with the current sim mount (board pose refitted
with the old mount held). Writes <poses session>/mount.json.

    MUJOCO_GL=cgl ~/.venvs/so101_eval/bin/python scripts/wrist_calib_fit.py mount data/wrist_calib/<stamp>_poses data/wrist_calib/<stamp>_handheld
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W, H = 640, 480
BOARD_Z = 0.002          # printed board on card, lying on the table (m)


def board(bd):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, bd["dict"]))
    b = cv2.aruco.CharucoBoard(tuple(bd["squares"]), bd["square_m"], bd["marker_m"], d)
    return b, cv2.aruco.CharucoDetector(b)


def detect(sess, b, det, prefix="wrist"):
    out = []
    for f in sorted(sess.glob(f"{prefix}_*.png")):
        ch, ids, _, _ = det.detectBoard(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2GRAY))
        if ids is None or len(ids) < 6:
            continue
        obj, img = b.matchImagePoints(ch, ids)
        out.append((f.name, obj.reshape(-1, 3).astype(np.float32), img.reshape(-1, 2).astype(np.float32)))
    return out


def calib(views, K0=None):
    flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    K0 = np.array([[518.0, 0, W / 2], [0, 518.0, H / 2], [0, 0, 1]]) if K0 is None else K0
    rms, K, dist, rv, tv = cv2.calibrateCamera([v[1] for v in views], [v[2] for v in views], (W, H), K0.copy(), np.zeros(5),
                                               flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS)
    per = []
    for (name, o, i), r, t in zip(views, rv, tv):
        p, _ = cv2.projectPoints(o, r, t, K, dist); per.append(float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - i) ** 2, 1)))))
    return rms, K, dist.ravel(), rv, tv, np.array(per)


def intrinsics(sess):
    sess = Path(sess); bd = json.loads((sess / "captures.json").read_text())["board"]; b, det = board(bd)
    views = detect(sess, b, det); print(f"{len(views)} frames with >= 6 corners")
    rms, K, dist, rv, tv, per = calib(views)
    keep = per < 3 * np.median(per)
    if not keep.all():
        print(f"dropping {(~keep).sum()} outlier frames: {[v[0] for v, k in zip(views, keep) if not k]}")
        views = [v for v, k in zip(views, keep) if k]; rms, K, dist, rv, tv, per = calib(views, K)
    dists = [float(np.linalg.norm(t)) for t in tv]
    res = {"f_px": round(float(K[0, 0]), 2), "cx": round(float(K[0, 2]), 2), "cy": round(float(K[1, 2]), 2),
           "k1": round(float(dist[0]), 4), "k2": round(float(dist[1]), 4), "rms_px": round(float(rms), 3), "frames": len(views),
           "board_distance_m": [round(min(dists), 3), round(max(dists), 3)],
           "fovy_deg": round(float(np.degrees(2 * np.arctan(H / 2 / K[0, 0]))), 3)}
    (sess / "intrinsics.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    print("per-frame rms px:", np.round(per, 2).tolist())
    # coverage: where the detected corners fall in the image (3 x 3 grid counts)
    allp = np.vstack([v[2] for v in views]); g = np.zeros((3, 3), int)
    for u, v in allp:
        g[min(int(v / H * 3), 2), min(int(u / W * 3), 2)] += 1
    print("corner coverage (rows top->bottom, cols left->right):\n", g)
    return res




# ---------------------------------------------------------------- mount pose (hand-eye)
def marker_points(sess, b, prefix="wrist"):
    """Per frame: the 4 corners of every detected ArUco marker (board frame, m) and their pixels. Marker corners, not
    ChArUco corners: on the tour frames (blur, oblique views) far more markers are found than interpolated corners."""
    dp = cv2.aruco.DetectorParameters(); dp.adaptiveThreshWinSizeMax = 63; dp.adaptiveThreshWinSizeStep = 4
    dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX; dp.polygonalApproxAccuracyRate = 0.06; dp.errorCorrectionRate = 0.9
    cp = cv2.aruco.CharucoParameters(); cp.tryRefineMarkers = True; det = cv2.aruco.CharucoDetector(b, cp, dp); out = {}
    for f in sorted(sess.glob(f"{prefix}_*.png")):
        _, _, mc, mids = det.detectBoard(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2GRAY))
        if mids is None or len(mids) < 2:
            continue
        objp = b.getObjPoints(); idx = {int(i): k for k, i in enumerate(b.getIds().ravel())}       # CharucoBoard.matchImagePoints
        keep = [k for k, i in enumerate(mids.ravel()) if int(i) in idx]                                   # (ids not on the board: misreads)
        if len(keep) < 2:
            continue
        obj = np.vstack([np.asarray(objp[idx[int(mids.ravel()[k])]]).reshape(4, 3) for k in keep])        # takes ChArUco corners only
        img = np.vstack([np.asarray(mc[k]).reshape(4, 2) for k in keep]); mids = mids.ravel()[keep]
        out[int(f.stem.split("_")[1])] = (obj.astype(np.float64), img.astype(np.float64), len(mids))
    return out


def mount(sess, handheld):
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as Rot
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
    from wrist_calib_poses import Twin
    sess = Path(sess); cap = json.loads((sess / "captures.json").read_text()); bd = cap["board"]; b, _ = board(bd)
    I = json.loads((Path(handheld) / "intrinsics.json").read_text())
    K = np.array([[I["f_px"], 0, I["cx"]], [0, I["f_px"], I["cy"]], [0, 0, 1]]); dist = np.array([I["k1"], I["k2"], 0, 0, 0])
    pts = marker_points(sess, b); tw = Twin(); m = tw.m; gb = m.body("gripper").id
    CV = np.diag([1.0, -1.0, -1.0])                                   # MuJoCo camera axes (x right, y up, -z view) <-> OpenCV
    views = []
    for c in cap["captures"]:
        if c["i"] not in pts:
            continue
        tw.set(c["q_read"]); Rg = tw.d.xmat[gb].reshape(3, 3).copy(); xg = tw.d.xpos[gb].copy()
        obj, img, nm = pts[c["i"]]; ok, rv, tv = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE if len(obj) == 4 else cv2.SOLVEPNP_ITERATIVE)
        views.append(dict(name=c["name"], Rg=Rg, xg=xg, obj=obj, img=img, rv=rv.ravel(), tv=tv.ravel(), n=nm))
    print(f"{len(views)} tour frames with >= 2 markers ({sum(v['n'] for v in views)} markers, {sum(len(v['obj']) for v in views)} points)")
    # current sim mount, OpenCV camera frame in gripper coordinates
    cid = m.camera("wrist").id; Rs = Rot.from_quat(np.r_[m.cam_quat[cid][1:], m.cam_quat[cid][0]]).as_matrix() @ CV; ts = m.cam_pos[cid].copy()
    # initial guess: OpenCV hand-eye (eye in hand), then board pose from each view, averaged
    Rc2g, tc2g = cv2.calibrateHandEye([v["Rg"] for v in views], [v["xg"] for v in views],
                                      [cv2.Rodrigues(v["rv"])[0] for v in views], [v["tv"] for v in views], method=cv2.CALIB_HAND_EYE_PARK)
    tc2g = tc2g.ravel()
    Tb = [(v["Rg"] @ Rc2g @ cv2.Rodrigues(v["rv"])[0], v["Rg"] @ (Rc2g @ v["tv"] + tc2g) + v["xg"]) for v in views]
    Rb0 = Rot.from_matrix(np.array([t[0] for t in Tb])).mean().as_matrix(); xb0 = np.mean([t[1] for t in Tb], 0)

    def unpack(p):
        return Rot.from_rotvec(p[:3]).as_matrix(), p[3:6], Rot.from_rotvec(p[6:9]).as_matrix(), p[9:12]

    def resid(p, vs):
        Rc, tc, Rb, xb = unpack(p); r = []
        for v in vs:
            Rwc = v["Rg"] @ Rc; xwc = v["Rg"] @ tc + v["xg"]                      # camera in world
            Pw = v["obj"] @ Rb.T + xb; Pc = (Pw - xwc) @ Rwc                     # board points in camera frame
            uv, _ = cv2.projectPoints(Pc, np.zeros(3), np.zeros(3), K, dist); r.append((uv.reshape(-1, 2) - v["img"]).ravel())
        return np.concatenate(r)
    p0 = np.r_[Rot.from_matrix(Rc2g).as_rotvec(), tc2g, Rot.from_matrix(Rb0).as_rotvec(), xb0]
    sol = least_squares(resid, p0, args=(views,), loss="soft_l1", f_scale=3.0)
    Rc, tc, Rb, xb = unpack(sol.x)
    per = [float(np.sqrt(np.mean(resid(sol.x, [v]).reshape(-1, 2) ** 2 * 2))) for v in views]
    # same fit with the current sim mount held fixed (board pose only): how badly the old mount explains the frames
    sol0 = least_squares(lambda q: resid(np.r_[Rot.from_matrix(Rs).as_rotvec(), ts, q], views), np.r_[Rot.from_matrix(Rb0).as_rotvec(), xb0], loss="soft_l1", f_scale=3.0)
    per0 = [float(np.sqrt(np.mean(resid(np.r_[Rot.from_matrix(Rs).as_rotvec(), ts, sol0.x], [v]).reshape(-1, 2) ** 2 * 2))) for v in views]
    dR = Rot.from_matrix(Rs.T @ Rc); eul = dR.as_euler("xyz", degrees=True)            # in the (old) OpenCV camera frame
    dt_cam = Rs.T @ (tc - ts)
    nb = Rb[:, 2]; tilt = np.degrees(np.arccos(abs(nb[2])))
    print(f"reprojection rms per frame: fitted mount median {np.median(per):.1f} px (max {max(per):.1f}); "
          f"current sim mount median {np.median(per0):.1f} px (max {max(per0):.1f})")
    print(f"mount change (old camera frame: x right, y down, z forward): rotation {dR.magnitude() * 57.3:.1f} deg "
          f"[about x (tilt) {eul[0]:+.1f}, y (pan) {eul[1]:+.1f}, z (roll) {eul[2]:+.1f}], "
          f"translation {np.linalg.norm(dt_cam) * 1000:.0f} mm [x {dt_cam[0] * 1000:+.0f}, y {dt_cam[1] * 1000:+.0f}, z {dt_cam[2] * 1000:+.0f}]")
    print(f"board check: centre ({xb[0] * 100:.1f}, {xb[1] * 100:.1f}, {xb[2] * 100:.2f}) cm [origin corner], tilt from the table {tilt:.1f} deg")
    for v, e, e0 in zip(views, per, per0):
        print(f"  {v['name']}: {v['n']} markers, rms {e:.1f} px (old mount {e0:.1f})")
    # flat-board variant: board on the table (z = BOARD_Z, no tilt; x, y, yaw free). The sim renders the world through
    # the same FK, so arm-angle error must end up in the mount here, not in a floating board.
    yaw0 = np.arctan2(Rb[1, 0], Rb[0, 0]) if abs(Rb[2, 2]) > 0.5 else 0.0; flip = np.sign(Rb[2, 2])

    def board_flat(q):
        return Rot.from_euler("z", q[0]).as_matrix() @ np.diag([1.0, flip, flip]), np.r_[q[1], q[2], BOARD_Z]

    def resid_flat(p):
        Rbf, xbf = board_flat(p[6:9]); return resid(np.r_[p[:6], Rot.from_matrix(Rbf).as_rotvec(), xbf], views)
    solf = least_squares(resid_flat, np.r_[sol.x[:6], yaw0, xb[0], xb[1]], loss="soft_l1", f_scale=3.0)
    Rcf, tcf = Rot.from_rotvec(solf.x[:3]).as_matrix(), solf.x[3:6]; Rbf, xbf = board_flat(solf.x[6:9])
    perf = [float(np.sqrt(np.mean(resid(np.r_[solf.x[:6], Rot.from_matrix(Rbf).as_rotvec(), xbf], [v]).reshape(-1, 2) ** 2 * 2))) for v in views]
    dRf = Rot.from_matrix(Rs.T @ Rcf); eulf = dRf.as_euler("xyz", degrees=True); dtf = Rs.T @ (tcf - ts)
    print(f"flat board on the table: rms median {np.median(perf):.1f} px (max {max(perf):.1f}); mount change rotation {dRf.magnitude() * 57.3:.1f} deg "
          f"[tilt {eulf[0]:+.1f}, pan {eulf[1]:+.1f}, roll {eulf[2]:+.1f}], translation {np.linalg.norm(dtf) * 1000:.0f} mm "
          f"[x {dtf[0] * 1000:+.0f}, y {dtf[1] * 1000:+.0f}, z {dtf[2] * 1000:+.0f}]; board origin ({xbf[0] * 100:.1f}, {xbf[1] * 100:.1f}) cm")
    print("  per frame:", [round(e, 1) for e in perf])
    qf = Rot.from_matrix(Rc @ CV).as_quat(); (sess / "mount_free.json").write_text(json.dumps(
        {"cam_pos": [round(float(x), 5) for x in tc], "cam_quat": [round(float(x), 6) for x in np.r_[qf[3], qf[:3]]], "fovy_deg": I["fovy_deg"],
         "rms_px_median": round(float(np.median(per)), 2), "board": "free (absorbs arm-angle error)"}, indent=1))
    Rc, tc, per, dR, dt_cam = Rcf, tcf, perf, dRf, dtf                                   # the flat-board mount is the result
    Rm = Rc @ CV; qm = Rot.from_matrix(Rm).as_quat(); qm = np.r_[qm[3], qm[:3]]                 # back to the MuJoCo camera frame
    res = {"cam_pos": [round(float(x), 5) for x in tc], "cam_quat": [round(float(x), 6) for x in qm], "f_px": I["f_px"],
           "fovy_deg": I["fovy_deg"], "k1": I["k1"], "k2": I["k2"], "cx": I["cx"], "cy": I["cy"],
           "rms_px_median": round(float(np.median(per)), 2), "old_mount_rms_px_median": round(float(np.median(per0)), 2),
           "rotation_change_deg": round(float(dR.magnitude() * 57.3), 2), "translation_change_mm": round(float(np.linalg.norm(dt_cam) * 1000), 1),
           "free_board_tilt_deg": round(float(tilt), 2), "board": "flat on the table (z 2 mm)", "frames": len(views)}
    (sess / "mount.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res))
    return res


if __name__ == "__main__":
    if sys.argv[1] == "intrinsics":
        intrinsics(sys.argv[2])
    else:
        mount(sys.argv[2], sys.argv[3])
