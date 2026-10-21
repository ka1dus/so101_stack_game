#!/usr/bin/env bash
# Phase 3: capture checkerboard frames for camera intrinsics (OpenCV calibration).
# Board: 9 x 7 squares of 20 mm (8 x 6 inner corners), printed at 100 % on A4, glued flat on card.
#   conda activate lerobot && cd ~/so101_stack_game && bash scripts/capture_checkerboard.sh wrist 1
#   (front camera: bash scripts/capture_checkerboard.sh front 0)
# Arm at home, torque on. Hold the board in view at different distances, tilts and image
# positions (corners and edges too). Keep it still for each shot; Enter takes a frame, q quits.
set -e
CAM=${1:-wrist}; IDX=${2:-1}; OUT=inputs/calibration/checkerboard_$CAM
mkdir -p "$OUT"; i=$(ls "$OUT" 2>/dev/null | wc -l | tr -d ' ')
while true; do
  read -r -p "[$CAM] frame $((i+1)): position the board, Enter to capture (q to quit): " k
  [ "$k" = q ] && break
  python scripts/so101_capture_frames.py --cameras "$CAM=$IDX" --out "$OUT" --suffix "cb$(printf %02d $i)" --warmup 0.5
  i=$((i+1))
done
echo "saved $i frames in $OUT (aim for 15-25)"
