"""
Keyboard joint teleop for an SO-101 follower arm (no leader arm needed).
Hold several keys at once to move several joints at the same time.

Uses LeRobot's official SO101Follower class, so it reuses your existing
calibration file (matched by --id).

macOS: give your terminal app (Terminal / iTerm / VS Code) permission in
System Settings > Privacy & Security > Accessibility AND Input Monitoring,
then fully quit and reopen the terminal. Without it, no joint will move.

Usage:
    conda activate lerobot
    python so101_keyboard_teleop.py --port /dev/tty.usbmodemXXXX --id my_follower_arm

Controls (hold to move, combine freely):
    shoulder_pan   q / a        wrist_flex   r / f
    shoulder_lift  w / s        wrist_roll   t / g
    elbow_flex     e / d        gripper      y / h  (open / close)
    speed          + / -
    quit           x  (or Ctrl-C)
"""

import argparse
import select
import sys
import termios
import threading
import time
import tty

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


class HeldKeys:
    """Tracks which keys are physically held down right now (needs pynput)."""

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
        if (name := self._name(key)) is not None:
            with self._lock:
                self._held.add(name)

    def _release(self, key):
        if (name := self._name(key)) is not None:
            with self._lock:
                self._held.discard(name)

    def snapshot(self) -> set[str]:
        with self._lock:
            return set(self._held)

    def stop(self):
        self.listener.stop()


def read_typed() -> list[str]:
    """Single-shot keys typed into the terminal (used for +, -, x)."""
    keys = []
    while select.select([sys.stdin], [], [], 0)[0]:
        keys.append(sys.stdin.read(1).lower())
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="e.g. /dev/tty.usbmodem58760431541")
    parser.add_argument("--id", required=True, help="same id you used for lerobot-calibrate")
    parser.add_argument("--speed", type=float, default=30.0, help="joint speed in degrees per second")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, max_relative_target=10.0))
    robot.connect()  # loads calibration for this id, or runs calibration if missing

    obs = robot.get_observation()
    target = {k.removesuffix(".pos"): v for k, v in obs.items() if k.endswith(".pos")}
    speed = args.speed
    period = 1.0 / args.fps

    keys = HeldKeys()
    print(__doc__.split("Controls")[1])
    old_term = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())  # stop typed keys echoing into the terminal
    try:
        while True:
            t0 = time.perf_counter()

            for key in read_typed():
                if key == "x":
                    return
                if key in ("+", "="):
                    speed = min(speed + 10.0, 120.0)
                elif key == "-":
                    speed = max(speed - 10.0, 5.0)

            # Every held key moves its joint this tick -> simultaneous multi-joint motion
            for key in keys.snapshot():
                if key in KEYMAP:
                    motor, sign = KEYMAP[key]
                    delta = speed * period * (2 if motor == "gripper" else 1)
                    target[motor] += sign * delta
                    if motor == "gripper":
                        target[motor] = max(0.0, min(100.0, target[motor]))

            robot.send_action({f"{m}.pos": v for m, v in target.items()})

            status = "  ".join(f"{m[:6]}:{v:6.1f}" for m, v in target.items())
            sys.stdout.write(f"\rspeed {speed:5.1f}°/s | {status}   ")
            sys.stdout.flush()

            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
        keys.stop()
        print("\nDisconnecting (torque off — support the arm).")
        robot.disconnect()


if __name__ == "__main__":
    main()
