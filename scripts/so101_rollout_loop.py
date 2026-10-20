"""
Run lerobot-rollout repeatedly, putting the arm back at the home pose
BEFORE and AFTER every run, and keep a success tally.

Home pose: ~/.cache/so101_home/<robot.id>.json
    (set it with m in so101_keyboard_record.py, or from your dataset with
     so101_start_poses.py --write-home)

Usage: everything after `--` is passed straight to lerobot-rollout.

    python so101_rollout_loop.py --positions 5 -- \
        --strategy.type=base --policy.path=Maxctl352/act_pick_cup --policy.device=mps \
        --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5B910311481 \
        --robot.id=my_follower_arm \
        --robot.cameras="{ wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
        --task="Pick up the clip and drop it in the cup" --duration=40

Each trial:
    arm drives home -> you place the object -> Enter -> rollout runs
    -> arm lifts and drives home -> you type y/n for success
Quit with q at the prompt. Ctrl-C during a rollout stops that run only.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

ARM_JOINTS_LIFT_FIRST = ["shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def arg_value(args: list[str], key: str) -> str | None:
    for i, a in enumerate(args):
        if a.startswith(key + "="):
            return a.split("=", 1)[1]
        if a == key and i + 1 < len(args):
            return args[i + 1]
    return None


def drive(robot, goal: dict[str, float], joints: list[str], speed: float, fps: int):
    """Move the listed joints to their goal at `speed` deg/s; other joints hold still."""
    obs = robot.get_observation()
    target = {m: obs[f"{m}.pos"] for m in goal}
    step, period = speed / fps, 1.0 / fps
    while True:
        t0 = time.perf_counter()
        done = True
        for m in joints:
            diff = goal[m] - target[m]
            if abs(diff) > 0.3:
                done = False
            target[m] += max(-step, min(step, diff))
        robot.send_action({f"{m}.pos": v for m, v in target.items()})
        if done:
            return
        time.sleep(max(0.0, period - (time.perf_counter() - t0)))


def go_home(port: str, robot_id: str, home: dict[str, float], speed: float, direct: bool, release: bool):
    robot = SO101Follower(
        SO101FollowerConfig(
            port=port, id=robot_id, max_relative_target=10.0, disable_torque_on_disconnect=release
        )
    )
    robot.connect()
    try:
        if direct:
            drive(robot, home, list(home), speed, 30)
        else:
            # Lift/fold the arm first, then swing the base: avoids dragging through the cup
            drive(robot, home, [j for j in ARM_JOINTS_LIFT_FIRST if j in home], speed, 30)
            drive(robot, home, list(home), speed, 30)
        time.sleep(0.5)
    finally:
        robot.disconnect()


def main():
    if "--" not in sys.argv:
        sys.exit("Put the lerobot-rollout arguments after `--`. See the usage at the top of this file.")
    split = sys.argv.index("--")
    own, rollout_args = sys.argv[1:split], sys.argv[split + 1 :]

    p = argparse.ArgumentParser()
    p.add_argument("--home-file", default=None, help="default: ~/.cache/so101_home/<robot.id>.json")
    p.add_argument("--speed", type=float, default=30.0, help="homing speed, degrees per second")
    p.add_argument("--direct", action="store_true", help="straight-line homing (no lift-first)")
    p.add_argument("--positions", type=int, default=0, help="rotate through N object spots and prompt which")
    p.add_argument("--hold-at-end", action="store_true", help="keep torque on after quitting")
    args = p.parse_args(own)

    port = arg_value(rollout_args, "--robot.port")
    robot_id = arg_value(rollout_args, "--robot.id")
    if not port or not robot_id:
        sys.exit("The rollout arguments must include --robot.port=... and --robot.id=...")

    home_file = Path(args.home_file).expanduser() if args.home_file else (
        Path.home() / ".cache" / "so101_home" / f"{robot_id}.json"
    )
    if not home_file.exists():
        sys.exit(f"No home pose at {home_file}. Run so101_start_poses.py --write-home, or press m in the record script.")
    home = {m: float(v) for m, v in json.loads(home_file.read_text()).items()}

    # Keep the arm holding its last pose when lerobot-rollout exits, so it can't drop before homing
    if arg_value(rollout_args, "--robot.disable_torque_on_disconnect") is None:
        rollout_args = rollout_args + ["--robot.disable_torque_on_disconnect=false"]
    cmd = ["lerobot-rollout", *rollout_args]

    results: list[tuple[int, str, bool]] = []
    trial = 0
    print(f"Home pose: {home_file}")
    print("Moving to home...")
    go_home(port, robot_id, home, args.speed, args.direct, release=False)

    try:
        while True:
            spot = f" spot {trial % args.positions + 1}" if args.positions else ""
            ans = input(f"\nTrial {trial + 1}: place the object{' at' + spot if spot else ''}, Enter to run (q = quit): ")
            if ans.strip().lower() == "q":
                break

            try:
                subprocess.run(cmd, check=False)
            except KeyboardInterrupt:
                print("\nRollout interrupted.")

            print("Returning home...")
            go_home(port, robot_id, home, args.speed, args.direct, release=False)

            while True:
                r = input("Success? [y/n/s=skip, don't count]: ").strip().lower()
                if r in ("y", "n", "s"):
                    break
            if r != "s":
                results.append((trial + 1, spot.strip() or "-", r == "y"))
                trial += 1
            ok = sum(s for _, _, s in results)
            print(f"Tally: {ok}/{len(results)}")
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        print("Final return home" + ("" if args.hold_at_end else " (torque turns off after; support the arm)") + "...")
        try:
            go_home(port, robot_id, home, args.speed, args.direct, release=not args.hold_at_end)
        except Exception as e:  # never leave the user without a summary
            print(f"Could not return home: {e}")

        if results:
            print("\nTrial  Spot     Result")
            for t, s, ok in results:
                print(f"{t:<6} {s:<8} {'success' if ok else 'fail'}")
            if args.positions:
                print("\nPer spot:")
                for k in range(1, args.positions + 1):
                    rs = [ok for _, s, ok in results if s == f"spot {k}"]
                    if rs:
                        print(f"  spot {k}: {sum(rs)}/{len(rs)}")
            ok = sum(s for _, _, s in results)
            print(f"\nTotal: {ok}/{len(results)} ({100 * ok / len(results):.0f}%)")


if __name__ == "__main__":
    main()
