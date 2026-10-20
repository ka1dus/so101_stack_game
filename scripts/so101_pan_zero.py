"""
Rotate the SO-101 base (shoulder_pan) to 0 degrees, its calibrated middle,
so the arm points straight along its +x axis. Other joints stay where they are.

Usage:
    conda activate lerobot
    python so101_pan_zero.py --port /dev/tty.usbmodem5B910311481 --id my_follower_arm

The arm holds the pose with torque ON after the script exits, so you can lay
your tape line. Unplug power (or run so101_go_home.py) when done.
"""

import argparse
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--value", type=float, default=0.0, help="target shoulder_pan in degrees")
    p.add_argument("--speed", type=float, default=20.0, help="degrees per second")
    args = p.parse_args()

    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=args.id,
            max_relative_target=10.0,
            disable_torque_on_disconnect=False,  # keep holding after exit
        )
    )
    robot.connect()

    obs = robot.get_observation()
    target = {k.removesuffix(".pos"): v for k, v in obs.items() if k.endswith(".pos")}
    print(f"shoulder_pan now: {target['shoulder_pan']:.1f} deg -> moving to {args.value:.1f}")

    step = args.speed / 30
    try:
        while abs(target["shoulder_pan"] - args.value) > 0.2:
            diff = args.value - target["shoulder_pan"]
            target["shoulder_pan"] += max(-step, min(step, diff))
            robot.send_action({f"{m}.pos": v for m, v in target.items()})
            time.sleep(1 / 30)
        target["shoulder_pan"] = args.value
        robot.send_action({f"{m}.pos": v for m, v in target.items()})
        time.sleep(0.5)

        actual = robot.get_observation()["shoulder_pan.pos"]
        print(f"Done. shoulder_pan reads {actual:.1f} deg. Arm is holding position.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
