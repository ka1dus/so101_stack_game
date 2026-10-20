"""
Save one frame from each camera at the exact resolution used for recording
(default 640x480), through LeRobot's own OpenCV camera class, so the frames
match what the policy sees.

Usage:
    conda activate lerobot
    python so101_capture_frames.py --cameras front=0 wrist=1 \
        --out ~/so101_stack_game/inputs/photos --suffix frame

    -> saves front_cam_frame.png and wrist_cam_frame.png

    For the 3-box stack shot:
    python so101_capture_frames.py --cameras front=0 \
        --out ~/so101_stack_game/inputs/photos --suffix stack3
"""

import argparse
import time
from pathlib import Path

import cv2

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cameras", nargs="+", required=True, help="name=index pairs, e.g. front=0 wrist=1")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", default=".", help="output folder")
    p.add_argument("--suffix", default="frame", help="file name: <name>_cam_<suffix>.png")
    p.add_argument("--warmup", type=float, default=2.0, help="seconds to let auto-exposure settle")
    args = p.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    for pair in args.cameras:
        name, index = pair.split("=", 1)
        cam = OpenCVCamera(OpenCVCameraConfig(
            index_or_path=int(index), width=args.width, height=args.height, fps=args.fps,
        ))
        cam.connect()
        try:
            t_end = time.time() + args.warmup
            frame = cam.read()
            while time.time() < t_end:   # discard early frames while exposure settles
                frame = cam.read()
            h, w = frame.shape[:2]
            path = out / f"{name}_cam_{args.suffix}.png"
            cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            status = "OK" if (w, h) == (args.width, args.height) else "MISMATCH"
            print(f"{name} (index {index}): {w}x{h} [{status}] -> {path}")
        finally:
            cam.disconnect()


if __name__ == "__main__":
    main()
