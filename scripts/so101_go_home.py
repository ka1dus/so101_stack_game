"""
Drive the SO-101 follower back to the home pose saved by so101_keyboard_record.py,
then leave torque ON so it holds there until your next lerobot-rollout.

Usage (between rollout runs):
    python so101_go_home.py --port /dev/tty.usbmodem5B910311481 --id my_follower_arm

Home pose file: ~/.cache/so101_home/<id>.json (written when you press m in the
record script). Ctrl-C stops the motion immediately.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--id", required=True, help="same id used for calibration / recording")
    p.add_argument("--home-file", default=None, help="default: ~/.cache/so101_home/<id>.json")
    p.add_argument("--speed", type=float, default=30.0, help="degrees per second")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--release", action="store_true", help="turn torque off when done (arm will go limp)")
    args = p.parse_args()

    home_file = Path(args.home_file).expanduser() if args.home_file else (
        Path.home() / ".cache" / "so101_home" / f"{args.id}.json"
    )
    if not home_file.exists():
        sys.exit(f"No home pose at {home_file}. Set one with m in so101_keyboard_record.py first.")
    home = {m: float(v) for m, v in json.loads(home_file.read_text()).items()}

    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=args.id,
            max_relative_target=10.0,
            disable_torque_on_disconnect=args.release,  # keep holding home unless --release
        )
    )
    robot.connect()

    obs = robot.get_observation()
    target = {m: obs[f"{m}.pos"] for m in home}
    max_step = args.speed / args.fps
    period = 1.0 / args.fps

    print(f"Returning to home from {home_file} ... (Ctrl-C to stop)")
    try:
        while True:
            t0 = time.perf_counter()
            done = True
            for m, goal in home.items():
                diff = goal - target[m]
                if abs(diff) > 0.3:
                    done = False
                target[m] += max(-max_step, min(max_step, diff))
            robot.send_action({f"{m}.pos": v for m, v in target.items()})
            if done:
                break
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
        time.sleep(0.5)  # let the motors settle
        print("Home." + ("" if args.release else " Holding position (torque on)."))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
