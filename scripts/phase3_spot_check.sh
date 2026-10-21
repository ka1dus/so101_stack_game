#!/usr/bin/env bash
# Phase 3: check the new stacking X (23 cm from the pan axis) with the robot.
# The arm hovers its closed jaw tips 3 cm straight above where the sim puts the new X,
# you look (and photos are taken), then it goes back home. Asks before every move;
# Ctrl-C during a move stops the arm (it holds where it stopped).
#
#   conda activate lerobot && cd ~/so101_stack_game && bash scripts/phase3_spot_check.sh
set -e
PORT=${PORT:-/dev/tty.usbmodem5B910311481}
ID=my_follower_arm
SPEED=15
cd "$(dirname "$0")/.."

move() {
  echo; echo "NEXT MOVE: $2"
  read -r -p "  Press Enter to move (Ctrl-C to abort)... "
  if [ "$1" = home ]; then
    python scripts/so101_go_home.py --port "$PORT" --id "$ID" --speed "$SPEED"
  else
    python scripts/so101_go_home.py --port "$PORT" --id "$ID" --speed "$SPEED" --home-file "$1"
  fi
  echo "  (the script prints 'Home.' on arrival even for test poses; ignore that)"
}

echo "Checklist: table clear of boxes, new X marked, old X covered, nothing near the arm."
read -r -p "Press Enter when ready... "
move home                                           "arm to its normal home pose"
move notes/phase1_poses/R_raise.json               "R_raise: only the elbow moves; gripper rises ~20 cm"
move notes/phase3_poses/S1_above_new_spot.json     "S1: arm turns to its right; jaws pointing straight down, ~7 cm above the new X"
move notes/phase3_poses/S2_hover_new_spot.json     "S2: straight down to ~3 cm above the new X"
echo
echo "LOOK: stand behind the arm and look straight down past the jaw tips."
echo "  Are the closed tips directly above the NEW X? If not, estimate how far off (mm) and in which"
echo "  direction (toward/away from the arm, toward black mat/yellow side)."
read -r -p "  Press Enter to take front + wrist photos... "
python scripts/so101_capture_frames.py --cameras front=0 wrist=1 --out inputs/photos/phase3 --suffix spotcheck
move notes/phase3_poses/S1_above_new_spot.json     "S1: straight back up"
move notes/phase1_poses/R_raise.json               "R_raise"
move home                                           "home"
echo; echo "Done. Photos: inputs/photos/phase3/front_cam_spotcheck.png, wrist_cam_spotcheck.png"
