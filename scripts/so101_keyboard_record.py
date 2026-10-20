"""
Keyboard teleop + dataset recording for an SO-101 follower (no leader arm).

Writes a standard LeRobotDataset (same format as `lerobot-record`), so
`lerobot-train`, `lerobot-replay` and the Hub dataset viewer all work on it.
Built on the same LeRobot APIs that lerobot_record.py uses on `main`.

Usage:
    conda activate lerobot
    python so101_keyboard_record.py \
        --port /dev/tty.usbmodem5B910311481 \
        --id my_follower_arm \
        --cameras wrist=1 front=2 \
        --repo-id $HF_USER/so101_keyboard_test \
        --task "Grab the black cube" \
        --num-episodes 5

macOS: give your terminal app permission in System Settings > Privacy &
Security > Accessibility AND Input Monitoring, then quit and reopen it.

Arm keys (hold to move, combine freely for multi-joint motion):
    shoulder_pan   q / a        wrist_flex   r / f
    shoulder_lift  w / s        wrist_roll   t / g
    elbow_flex     e / d        gripper      y / h  (open / close)
    speed          + / -

Recording keys:
    n   ready/reset: start recording the next episode
        recording:   end current episode and save it
    b   recording: discard current episode (then reset, press n to redo)
    x   stop the session, save, and upload
    m   ready/reset: save the current pose as home (also written to disk)
    o   ready/reset: drive the arm back to home

Home pose:
    - Set it with m. It is saved to ~/.cache/so101_home/<id>.json and loaded
      automatically next session, so every session starts from the same pose.
    - No saved home and no m? The pose at your first n becomes home.
    - --reset-home ignores the saved file and lets you set a new one.

Auto return:
    - After EVERY episode (saved or discarded) the arm drives back to home
      at the start of the reset phase. Not recorded.
    - After the last episode (or x) it also returns home before torque off.
    - Hold any move key to pause (e.g. to lift out of the cup), o to resume.
    - Disable with --no-return-home.

Add more episodes to an existing dataset later: rerun the same command with
--resume (same --repo-id, --cameras, --fps). --num-episodes then means how
many NEW episodes to add, not the total.

Nothing records until you press n. Between episodes it waits for n with no
time limit (set --reset-time to auto-start after N seconds instead).
"""

import argparse
import json
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

from pynput import keyboard

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts

KEYMAP = {
    "q": ("shoulder_pan", +1), "a": ("shoulder_pan", -1),
    "w": ("shoulder_lift", +1), "s": ("shoulder_lift", -1),
    "e": ("elbow_flex", +1), "d": ("elbow_flex", -1),
    "r": ("wrist_flex", +1), "f": ("wrist_flex", -1),
    "t": ("wrist_roll", +1), "g": ("wrist_roll", -1),
    "y": ("gripper", +1), "h": ("gripper", -1),
}


def read_keys() -> list[str]:
    """Single-shot keys typed into the terminal (used for n, b, x, +, -)."""
    keys = []
    while select.select([sys.stdin], [], [], 0)[0]:
        keys.append(sys.stdin.read(1).lower())
    return keys


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


class KeyboardArm:
    """Holds joint targets; every held key moves its joint each tick."""

    def __init__(self, start: dict[str, float], speed: float, fps: int, home_file: Path | None = None):
        self.target = dict(start)
        self.speed = speed  # degrees per second
        self.period = 1.0 / fps
        self.home: dict[str, float] | None = None
        self.homing = False
        self.home_file = home_file

    def mark_home(self, persist: bool = True):
        self.home = dict(self.target)
        if persist and self.home_file is not None:
            try:
                self.home_file.parent.mkdir(parents=True, exist_ok=True)
                self.home_file.write_text(json.dumps(self.home, indent=2))
            except OSError as e:
                print(f"\nCould not save home pose to {self.home_file}: {e}")

    def load_home(self) -> bool:
        if self.home_file is None or not self.home_file.exists():
            return False
        try:
            saved = json.loads(self.home_file.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        if set(saved) != set(self.target):
            print(f"Ignoring {self.home_file}: joint names don't match this arm.")
            return False
        self.home = {m: float(v) for m, v in saved.items()}
        return True

    def step_home(self):
        """Move all joints together toward the saved home pose, a little each tick."""
        max_step = self.speed * self.period  # same speed as manual control
        done = True
        for motor, goal in self.home.items():
            diff = goal - self.target[motor]
            if abs(diff) > 0.3:
                done = False
            self.target[motor] += max(-max_step, min(max_step, diff))
        if done:
            self.homing = False

    def at_home(self) -> bool:
        return self.home is not None and all(
            abs(goal - self.target[m]) <= 0.3 for m, goal in self.home.items()
        )

    def adjust_speed(self, key: str):
        if key in ("+", "="):
            self.speed = min(self.speed + 10.0, 120.0)
        elif key == "-":
            self.speed = max(self.speed - 10.0, 5.0)

    def update(self, held: set[str]):
        if any(k in KEYMAP for k in held):
            self.homing = False  # any manual move cancels auto-return
        elif self.homing:
            self.step_home()
            return
        for key in held:
            if key in KEYMAP:
                motor, sign = KEYMAP[key]
                delta = self.speed * self.period * (2 if motor == "gripper" else 1)
                self.target[motor] += sign * delta
                if motor == "gripper":
                    self.target[motor] = max(0.0, min(100.0, self.target[motor]))

    def action(self) -> dict[str, float]:
        return {f"{m}.pos": v for m, v in self.target.items()}


def run_phase(
    robot, arm, held_keys, fps, duration_s, label, dataset=None, task=None,
    exit_when_home=False, start_homing=False,
) -> str | None:
    """Teleop loop. Records frames if `dataset` is given.

    duration_s <= 0 means no time limit: the phase only ends on a key press.
    exit_when_home: also end the phase as soon as the arm reaches the home pose.
    start_homing: begin driving the arm to home as soon as the phase starts.
    Returns 'next', 'rerecord', 'stop', 'home' or None (timed out).
    """
    if start_homing and arm.home is not None and not arm.at_home():
        arm.homing = True
    period = 1.0 / fps
    start = time.perf_counter()
    result = None
    while duration_s <= 0 or time.perf_counter() - start < duration_s:
        t0 = time.perf_counter()

        for key in read_keys():
            if key == "n":
                result = "next"
            elif key == "b" and dataset is not None:  # re-record only makes sense while recording
                result = "rerecord"
            elif key == "x":
                result = "stop"
            elif key == "m" and dataset is None:
                arm.homing = False
                arm.mark_home()
                print(f"\nHome pose saved{f' to {arm.home_file}' if arm.home_file else ''}.")
            elif key == "o" and dataset is None:  # only outside recording
                if arm.home is None:
                    print("\nNo home pose yet: jog the arm to the start pose and press m.")
                else:
                    arm.homing = True
            else:
                arm.adjust_speed(key)
        if result:
            break

        arm.update(held_keys.snapshot())  # all held keys -> simultaneous joint motion

        obs = robot.get_observation()
        sent = robot.send_action(arm.action())  # may be clipped by max_relative_target

        if exit_when_home and arm.at_home():
            result = "home"
            break

        if dataset is not None:
            obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
            act_frame = build_dataset_frame(dataset.features, sent, prefix=ACTION)
            dataset.add_frame({**obs_frame, **act_frame, "task": task})

        elapsed = time.perf_counter() - start
        limit = f"/{duration_s:.0f}s" if duration_s > 0 else "s"
        homing = "  RETURNING HOME" if arm.homing else ""
        sys.stdout.write(f"\r[{label}] {elapsed:5.1f}{limit}  speed {arm.speed:5.1f}°/s{homing}   ")
        sys.stdout.flush()

        time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    print()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--id", required=True, help="same id used in lerobot-calibrate")
    p.add_argument(
        "--cameras",
        nargs="+",
        required=True,
        help="one or more name=index pairs from lerobot-find-cameras, e.g. wrist=1 front=2",
    )
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--repo-id", required=True, help="e.g. $HF_USER/so101_keyboard_test")
    p.add_argument("--task", required=True)
    p.add_argument("--num-episodes", type=int, default=5)
    p.add_argument("--episode-time", type=float, default=60)
    p.add_argument(
        "--reset-time",
        type=float,
        default=0,
        help="seconds before the next episode auto-starts; 0 (default) = wait until you press n",
    )
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--speed", type=float, default=30.0, help="joint speed in degrees per second")
    p.add_argument(
        "--resume",
        action="store_true",
        help="append new episodes to an existing local dataset with the same --repo-id",
    )
    p.add_argument("--root", default=None, help="local dataset folder (default: ~/.cache/huggingface/lerobot/<repo-id>)")
    p.add_argument(
        "--no-return-home",
        action="store_true",
        help="don't drive the arm back to home after each episode / at the end",
    )
    p.add_argument(
        "--home-file",
        default=None,
        help="where the home pose is stored (default: ~/.cache/so101_home/<id>.json)",
    )
    p.add_argument("--reset-home", action="store_true", help="ignore the saved home pose this session")
    p.add_argument(
        "--positions",
        type=int,
        default=0,
        help="number of object spots to rotate through (e.g. 5): prompts which spot to use each episode",
    )
    p.add_argument("--no-push", action="store_true", help="keep dataset local only")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    user, _, name = args.repo_id.partition("/")
    if not user or not name:
        sys.exit(
            f"--repo-id must look like <hf_username>/<dataset_name>, got '{args.repo_id}'.\n"
            "If it starts with '/', $HF_USER is empty in this terminal. Run:\n"
            "  HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}'); echo $HF_USER"
        )

    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=args.id,
            max_relative_target=10.0,
            cameras={
                name: OpenCVCameraConfig(
                    index_or_path=int(index),
                    width=args.cam_width,
                    height=args.cam_height,
                    fps=args.fps,
                )
                for name, index in (pair.split("=", 1) for pair in args.cameras)
            },
        )
    )

    # Same feature construction as lerobot_record.py (identity processors)
    teleop_proc, _, obs_proc = make_default_processors()
    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_proc,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=obs_proc,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    if args.resume:
        # resume() needs an explicit local folder; default to where create() wrote it
        resume_root = Path(args.root).expanduser() if args.root else HF_LEROBOT_HOME / args.repo_id
        if not (resume_root / "meta" / "info.json").exists():
            sys.exit(
                f"No dataset found at {resume_root}.\n"
                "Check --repo-id, or pass --root with the folder that contains meta/, data/ and videos/."
            )
        print(f"Resuming from {resume_root}")
        # Same call lerobot_record.py makes for --resume=true
        dataset = LeRobotDataset.resume(
            args.repo_id,
            root=resume_root,
            image_writer_processes=0,
            image_writer_threads=4 * len(robot.cameras),
        )
        # Refuses to continue if cameras, joints or fps differ from the original recording
        sanity_check_dataset_robot_compatibility(dataset, robot, args.fps, features)
        print(f"Resuming {args.repo_id}: {dataset.num_episodes} episodes already recorded.")
    else:
        dataset = LeRobotDataset.create(
            args.repo_id,
            args.fps,
            root=args.root,
            robot_type=robot.name,
            features=features,
            use_videos=True,
            image_writer_threads=4 * len(robot.cameras),
        )

    robot.connect()  # may prompt for calibration; done before the terminal goes raw
    obs = robot.get_observation()
    home_file = Path(args.home_file).expanduser() if args.home_file else (
        Path.home() / ".cache" / "so101_home" / f"{args.id}.json"
    )
    arm = KeyboardArm(
        {k.removesuffix(".pos"): v for k, v in obs.items() if k.endswith(".pos")},
        args.speed, args.fps, home_file,
    )
    if not args.reset_home and arm.load_home():
        print(f"Loaded home pose from {home_file}. Press o in READY to move there, or m to replace it.")
    auto_home = not args.no_return_home
    held_keys = HeldKeys()

    def spot(ep_index: int) -> str:
        """Round-robin spot for an episode: ep 0 -> spot 1, ep 1 -> spot 2, ... (continues across --resume)."""
        return f" | object at SPOT {ep_index % args.positions + 1}" if args.positions > 0 else ""

    print(__doc__.split("Arm keys")[1])
    old_term = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    saved = 0
    try:
        with VideoEncodingManager(dataset):
            # Nothing is recorded until you press n: set up the scene and jog the arm first.
            if run_phase(robot, arm, held_keys, args.fps, 0,
                         f"READY: m = save home, n = start{spot(dataset.num_episodes)}") == "stop":
                return
            if arm.home is None:
                # No saved home and no m: use the pose at the first n for this session only
                arm.mark_home(persist=False)
                print("Using the current pose as home for this session (press m in RESET to save it).")

            while saved < args.num_episodes:
                ep = dataset.num_episodes
                result = run_phase(robot, arm, held_keys, args.fps, args.episode_time,
                                   f"REC ep {ep}{spot(ep)}", dataset=dataset, task=args.task)

                if result == "rerecord":
                    dataset.clear_episode_buffer()
                    print(f"Episode {ep} discarded.")
                    if run_phase(robot, arm, held_keys, args.fps, args.reset_time,
                                 f"RESET: n = redo ep {ep}{spot(ep)}",
                                 start_homing=auto_home) == "stop":
                        break
                    continue
                else:
                    dataset.save_episode()
                    saved += 1
                    print(f"Episode {ep} saved ({saved}/{args.num_episodes}).")

                if result == "stop" or saved >= args.num_episodes:
                    break

                # Reset phase: arm still controllable, nothing recorded. Waits for n by default.
                if run_phase(robot, arm, held_keys, args.fps, args.reset_time,
                             f"RESET: n = start ep {dataset.num_episodes}{spot(dataset.num_episodes)}",
                             start_homing=auto_home) == "stop":
                    break

        # Session over (last episode saved or x pressed): drive the arm home before torque turns off.
        # Not recorded. Holding a move key pauses it so you can steer around obstacles; o resumes.
        if auto_home and arm.home is not None and not arm.at_home():
            arm.homing = True
            run_phase(
                robot, arm, held_keys, args.fps, 0,
                "DONE: returning home (move key = pause, o = resume, x = skip)",
                exit_when_home=True,
            )
            if arm.at_home():
                print("Arm is home.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
        held_keys.stop()
        dataset.finalize()
        print("Disconnecting (torque off — support the arm).")
        if robot.is_connected:
            robot.disconnect()
        if not args.no_push and dataset.num_episodes > 0:
            print("Uploading to the Hub...")
            dataset.push_to_hub(private=args.private)
        print(f"Done. {dataset.num_episodes} episodes in {args.repo_id}")


if __name__ == "__main__":
    main()
