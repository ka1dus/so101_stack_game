"""
Record joint angles with the SO-101 gripper touching reference points on the table.
The angles are later converted to (x, y) in the arm's frame with the robot model.

Usage:
    conda activate lerobot
    python so101_record_points.py --port /dev/tty.usbmodem5B910311481 --id my_follower_arm

How to touch a point:
    - Close the gripper fully (hold h) so both jaw tips meet.
    - Jog until the jaw tips are just above the point, then lower until they
      barely touch the table. Don't press down; pressing bends the arm and
      skews the reading.
    - Press p to record. The script moves on to the next point.

Points are recorded in this order (press k to skip one you can't reach):
    1. table edges: the arm sits on one corner, so touch two points on each of the
       two table edges that meet at the arm's corner:
         left edge  = the edge running away from the arm on its LEFT (black-mat side)
         right edge = the edge running away from the arm on its RIGHT (yellow side)
         "near" = close to the arm, "far" = as far along the edge as the arm reaches
       (the further apart the two points on an edge, the more accurate)
    2. black supply-mat corners, 3. yellow stack-area corners:
       corner 1 = the corner nearest the arm, then go CLOCKWISE seen from above
    4. the stacking spot (where the bottom box sits)
Left/right are from the ARM's point of view (stand behind the arm):
    left = black-mat side, right = yellow side.

Keys (hold to move, combine freely):
    shoulder_pan   q / a        wrist_flex   r / f
    shoulder_lift  w / s*       wrist_roll   t / g
    elbow_flex     e / d        gripper      y / h  (open / close)
    speed          + / -  (default slow, for precise touching)
    p = record point    k = skip point    u = undo last    x = save and quit
    (*s moves shoulder_lift; skip is k)

macOS: the terminal app needs Accessibility + Input Monitoring permission
(already set up if the keyboard teleop script works).
"""

import argparse
import json
import select
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path

from pynput import keyboard

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

KEYMAP = {
    "q": ("shoulder_pan", +1), "a": ("shoulder_pan", -1),
    "w": ("shoulder_lift", +1), "s": ("shoulder_lift", -1),
    "e": ("elbow_flex", +1), "d": ("elbow_flex", -1),
    "r": ("wrist_flex", +1), "f": ("wrist_flex", -1),
    "t": ("wrist_roll", +1), "g": ("wrist_roll", -1),
    "y": ("gripper", +1), "h": ("gripper", -1),
}

POINTS = [
    # The arm sits at one corner of the table, so instead of corners we record two
    # points along each of the two table edges that meet at the arm's corner.
    # With the known table size (38 x 38 cm) that fixes the whole table.
    "table_left_edge_near", "table_left_edge_far",
    "table_right_edge_near", "table_right_edge_far",
    # Area corners: 1 = the corner nearest the arm, then go CLOCKWISE seen from above
    "supply_corner_1", "supply_corner_2", "supply_corner_3", "supply_corner_4",
    "stack_corner_1", "stack_corner_2", "stack_corner_3", "stack_corner_4",
    "stacking_spot",
]

# Phase 1 joint-offset protocol (--points multipose, see notes/phase1.md): touch the SAME
# three marks with clearly different wrist tilts. Exact tilt angles don't matter; only that
# they differ (steep = jaws close to vertical, then tilt the wrist so the jaws lean back
# toward the arm). Mark A = the X on the stacking spot, B = X on white tape on the far side
# of the black mat, C = X on the white table between the arm and the mats.
POINTS_MULTIPOSE = [
    "mark_A_steep", "mark_A_mid", "mark_A_flat",
    "mark_B_steep", "mark_B_mid", "mark_B_flat",
    "mark_C_steep", "mark_C_mid", "mark_C_flat",
]


class HeldKeys:
    def __init__(self):
        self._held: set[str] = set()
        self._lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self._press, on_release=self._release)
        self.listener.start()

    @staticmethod
    def _name(key):
        char = getattr(key, "char", None)
        return char.lower() if char else None

    def _press(self, key):
        if (n := self._name(key)) is not None:
            with self._lock:
                self._held.add(n)

    def _release(self, key):
        if (n := self._name(key)) is not None:
            with self._lock:
                self._held.discard(n)

    def snapshot(self) -> set[str]:
        with self._lock:
            return set(self._held)

    def stop(self):
        self.listener.stop()


def read_typed() -> list[str]:
    keys = []
    while select.select([sys.stdin], [], [], 0)[0]:
        keys.append(sys.stdin.read(1).lower())
    return keys


def save(path: Path, records: list, skipped: list, args):
    path.write_text(json.dumps({
        "robot_id": args.id,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "units": "LeRobot degrees (use_degrees=True); gripper 0-100",
        "touch_point": "closed gripper jaw tips touching the table",
        "points": records,
        "skipped": skipped,
    }, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--speed", type=float, default=10.0, help="deg/s (slow for precise touching)")
    p.add_argument("--out", default=None, help="output JSON (default: table_points_<time>.json)")
    p.add_argument("--points", choices=["table", "multipose"], default="table",
                   help="table = original 13 table/mat points; multipose = Phase 1 joint-offset protocol")
    args = p.parse_args()
    global POINTS
    if args.points == "multipose":
        POINTS = POINTS_MULTIPOSE

    out = Path(args.out or f"table_points_{datetime.now():%Y%m%d_%H%M%S}.json")

    robot = SO101Follower(SO101FollowerConfig(
        port=args.port, id=args.id, max_relative_target=10.0, disable_torque_on_disconnect=False,
    ))
    robot.connect()
    obs = robot.get_observation()
    target = {k.removesuffix(".pos"): v for k, v in obs.items() if k.endswith(".pos")}

    speed, period = args.speed, 1.0 / 30
    records: list[dict] = []
    skipped: list[str] = []
    idx = 0

    keys = HeldKeys()
    print("How to touch" + __doc__.split("How to touch")[1].split("macOS:")[0])
    print(f"Saving to {out}\n")
    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        while idx < len(POINTS):
            t0 = time.perf_counter()
            for k in read_typed():
                if k == "x":
                    raise KeyboardInterrupt
                if k in ("+", "="):
                    speed = min(speed + 5, 60)
                elif k == "-":
                    speed = max(speed - 5, 2)
                elif k == "p":
                    actual = robot.get_observation()
                    joints = {m: round(float(actual[f"{m}.pos"]), 3) for m in target}
                    records.append({"label": POINTS[idx], "joints": joints})
                    save(out, records, skipped, args)
                    print(f"\nRecorded {POINTS[idx]}: {joints}")
                    idx += 1
                elif k == "k":
                    skipped.append(POINTS[idx])
                    save(out, records, skipped, args)
                    print(f"\nSkipped {POINTS[idx]}")
                    idx += 1
                elif k == "u" and (records or skipped):
                    last = POINTS[idx - 1]
                    if records and records[-1]["label"] == last:
                        records.pop()
                    elif skipped and skipped[-1] == last:
                        skipped.pop()
                    idx -= 1
                    save(out, records, skipped, args)
                    print(f"\nUndid {last}")

            for k in keys.snapshot():
                if k in KEYMAP:
                    m, sign = KEYMAP[k]
                    target[m] += sign * speed * period * (2 if m == "gripper" else 1)
                    if m == "gripper":
                        target[m] = max(0.0, min(100.0, target[m]))
            robot.send_action({f"{m}.pos": v for m, v in target.items()})

            if idx < len(POINTS):
                sys.stdout.write(
                    f"\r[{idx + 1}/{len(POINTS)}] touch: {POINTS[idx]:<20} speed {speed:4.0f}°/s   "
                )
                sys.stdout.flush()
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
        print("\nAll points done.")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        keys.stop()
        save(out, records, skipped, args)
        robot.disconnect()  # torque stays on; arm holds its last pose
        print(f"Saved {len(records)} points to {out.resolve()}")
        print("Lift the arm clear of the table, then run so101_go_home.py.")


if __name__ == "__main__":
    main()
