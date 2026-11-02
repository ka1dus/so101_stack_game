"""Phase 8: capture frames for the wrist camera calibration (real arm, Mac, eval venv).

Part A, intrinsics (--handheld): the arm raises to R_raise (only the elbow moves; lens ~33 cm above the table,
looking forward and down), torque on, and returns home at the end. You hold the ChArUco board in front of the wrist
camera at different distances (10-40 cm), tilts and image positions; Enter saves a frame, q stops. Aim for 20.
Part B, mount pose (default): the board lies flat on the yellow area, edges along the yellow tape. The arm visits
the poses in notes/wrist_calib/poses.json (made and safety-checked in sim by scripts/wrist_calib_poses.py) at a
slow speed through S1 (jaws vertical ~7 cm above the X), holds still at each capture pose and saves wrist + front
frames and the averaged joint reading.

    ~/.venvs/so101_eval/bin/python scripts/wrist_calib_capture.py --port /dev/tty.usbmodem5B910311481 --handheld
    ~/.venvs/so101_eval/bin/python scripts/wrist_calib_capture.py --port /dev/tty.usbmodem5B910311481
Ctrl-C while moving: the arm holds; then r = go back via S1 / R_raise / home, t = torque off.
Saves data/wrist_calib/<time>_{handheld,poses}/: wrist_###.png, front_###.png, captures.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from game.backends import saved_home  # noqa: E402
from joint_mapping import JOINT_NAMES  # noqa: E402
from perception.drop import Kinematics  # noqa: E402
from skill.runner import RealArm, move_to  # noqa: E402

POSES = json.loads((ROOT / "notes/wrist_calib/poses.json").read_text())
BD = POSES["board"]
DICT = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, BD["dict"]))
BOARD = cv2.aruco.CharucoBoard(tuple(BD["squares"]), BD["square_m"], BD["marker_m"], DICT)
DETECTOR = cv2.aruco.CharucoDetector(BOARD)


def corners(bgr):
    ch, ids, _, _ = DETECTOR.detectBoard(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    return 0 if ids is None else len(ids)


def bgr(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def settle_read(arm, n=10):
    time.sleep(1.5); qs = []; last = None
    for _ in range(n):
        q, front, wrist = arm.read(); qs.append(q); last = (front, wrist); time.sleep(1 / 30)
    return np.mean(qs, 0), last


def handheld(arm, kin, out):
    caps = []; i = 0; names = {s["name"]: s["q"] for s in POSES["sequence"]}
    input("Hands clear: Enter to raise the arm to R_raise (only the elbow moves, the gripper rises ~20 cm): ")
    if (why := move_to(arm, np.array(names["R_raise"], float), kin, speed=15)):
        print(f"  stopped by the safety check: {why}"); return caps
    print("Hold the board in front of the wrist camera: near (10 cm) and far (40 cm), tilted up to ~45 deg, in the"
          " centre and near each edge and corner of the image. Keep it still for each shot.")
    while True:
        k = input(f"frame {i + 1}: Enter = capture, q = done: ").strip()
        if k == "q":
            break
        q, front, wrist = arm.read(); w = bgr(wrist); n = corners(w)
        cv2.imwrite(str(out / f"wrist_{i:03d}.png"), w); caps.append({"i": i, "corners": n}); i += 1
        print(f"  saved, {n}/24 board corners found" + ("  (too few, try again closer/steadier)" if n < 6 else ""))
    input("Hands clear: Enter to lower the arm back home: "); move_to(arm, saved_home(), kin, speed=15)
    return caps


def tour(arm, kin, out):
    home = saved_home(); caps = []; seq = POSES["sequence"]; auto = False; done = []
    for k, st in enumerate(seq):
        goal = home if st["q"] is None else np.array(st["q"], float)
        if not auto:
            r = input(f"move {k + 1}/{len(seq)} -> {st['name']}: Enter = go, a = go and run the rest without asking: ").strip()
            auto = r == "a"
        why = move_to(arm, goal, kin, speed=15)
        if why:
            print(f"  stopped by the safety check: {why}"); raise KeyboardInterrupt
        done.append(st)
        if st["capture"]:
            q, (front, wrist) = settle_read(arm); w = bgr(wrist); n = corners(w); j = len(caps)
            cv2.imwrite(str(out / f"wrist_{j:03d}.png"), w); cv2.imwrite(str(out / f"front_{j:03d}.png"), bgr(front))
            caps.append({"i": j, "name": st["name"], "q_cmd": st["q"], "q_read": [round(float(x), 3) for x in q], "corners": n})
            print(f"  {st['name']}: saved, {n}/24 board corners found; command - reading {np.round(np.array(st['q']) - q, 1).tolist()}")
    return caps


def back_out(arm, kin):
    """From wherever the arm stopped: S1, R_raise, home (the tour's own route)."""
    names = {s["name"]: s["q"] for s in POSES["sequence"]}
    for n in ("S1", "R_raise"):
        input(f"  Enter to move to {n}: "); move_to(arm, np.array(names[n], float), kin, speed=15)
    input("  Enter to move home: "); move_to(arm, saved_home(), kin, speed=15)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", required=True); ap.add_argument("--id", default="my_follower_arm")
    ap.add_argument("--front", type=int, default=0); ap.add_argument("--wrist", type=int, default=1); ap.add_argument("--handheld", action="store_true")
    a = ap.parse_args(); kin = Kinematics()
    out = ROOT / "data/wrist_calib" / f"{time.strftime('%Y%m%d_%H%M%S')}_{'handheld' if a.handheld else 'poses'}"; out.mkdir(parents=True)
    arm = RealArm(a.port, a.id, a.front, a.wrist); arm.connect(); caps = []
    try:
        q = arm.joints()
        if np.abs(q[:5] - saved_home()[:5]).max() > 3:
            print(f"the arm is not at home ({np.round(q, 1).tolist()}); run so101_go_home.py first"); return
        if a.handheld:
            try:
                caps = handheld(arm, kin, out)
            except KeyboardInterrupt:
                arm.hold(); print("\nheld.")
                if input("h = go home slowly, anything else = torque off: ").strip() == "h":
                    move_to(arm, saved_home(), kin, speed=15)
                else:
                    arm.torque_off()
        else:
            input("Board flat on the yellow area, edges along the tape, printed side up; table otherwise clear of boxes;"
                  " hands out of the arm's way. Enter to start (Ctrl-C holds the arm): ")
            try:
                caps = tour(arm, kin, out)
            except KeyboardInterrupt:
                arm.hold(); print("\nheld.")
                if input("r = go back via S1 / R_raise / home, anything else = torque off: ").strip() == "r":
                    back_out(arm, kin)
                else:
                    arm.torque_off()
    finally:
        (out / "captures.json").write_text(json.dumps({"joints": JOINT_NAMES, "board": BD, "captures": caps}, indent=1))
        arm.disconnect(); print(f"{len(caps)} captures in {out}")


if __name__ == "__main__":
    main()
