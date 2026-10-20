"""
Record a short video from one camera at the exact recording settings
(default 640x480, 30 fps), through LeRobot's own OpenCV camera class.

Usage (from ~/so101_stack_game, lerobot env active):
    python scripts/so101_record_video.py --camera 0 --seconds 10 --out inputs/videos/stack_collapse.mp4
    python scripts/so101_record_video.py --camera 0 --seconds 15 --out inputs/videos/human_turn.mp4

A 3-second countdown runs before recording starts. Ctrl+C stops early and keeps the video.
"""

import argparse
import time
from pathlib import Path

import cv2

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0, help="camera index (front = 0)")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--out", required=True, help="output .mp4 path")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--countdown", type=int, default=3)
    args = p.parse_args()

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    cam = OpenCVCamera(OpenCVCameraConfig(
        index_or_path=args.camera, width=args.width, height=args.height, fps=args.fps,
    ))
    cam.connect()
    writer = None
    n = 0
    try:
        t_end = time.time() + 1.5          # let auto-exposure settle
        while time.time() < t_end:
            cam.read()
        for i in range(args.countdown, 0, -1):
            print(f"recording in {i}...")
            t_end = time.time() + 1
            while time.time() < t_end:
                cam.read()
        print(f"RECORDING ({args.seconds:.0f} s) -> {out}")
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
        t0 = time.time()
        while time.time() - t0 < args.seconds:
            frame = cam.read()
            h, w = frame.shape[:2]
            if (w, h) != (args.width, args.height):
                raise RuntimeError(f"camera gave {w}x{h}, expected {args.width}x{args.height}")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            n += 1
    except KeyboardInterrupt:
        print("stopped early")
    finally:
        if writer is not None:
            writer.release()
        cam.disconnect()
    dur = time.time() - t0 if n else 0
    print(f"done: {n} frames, {dur:.1f} s (~{n / dur if dur else 0:.0f} fps) -> {out}")


if __name__ == "__main__":
    main()
