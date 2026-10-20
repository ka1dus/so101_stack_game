#!/usr/bin/env bash
# Phase 1 M4: guided run of the three test poses with photos (notes/phase1_measurement_guide.md).
# Asks before every move. Ctrl-C during a move stops the arm (it holds where it stopped);
# then rerun this script, which always starts by going home.
#
#   conda activate lerobot
#   cd ~/so101_stack_game && bash scripts/phase1_m4_run.sh
set -e
PORT=${PORT:-/dev/tty.usbmodem5B910311481}
ID=my_follower_arm
SPEED=15
cd "$(dirname "$0")/.."

move() {   # move <pose file or "home"> <what you should see>
  echo
  echo "NEXT MOVE: $2"
  read -r -p "  Press Enter to move (Ctrl-C to abort)... "
  if [ "$1" = home ]; then
    python scripts/so101_go_home.py --port "$PORT" --id "$ID" --speed "$SPEED"
  else
    python scripts/so101_go_home.py --port "$PORT" --id "$ID" --speed "$SPEED" --home-file "notes/phase1_poses/$1.json"
  fi
  echo "  (the script always prints 'Home.' when it arrives, even for test poses; ignore that)"
}

photo() {  # photo <P1|P2|P3>
  echo
  read -r -p "  Arm holding $1. Press Enter to take the front + wrist photos... "
  python scripts/so101_capture_frames.py --cameras front=0 wrist=1 --out inputs/photos/phase1 --suffix "$1"
}

echo "M4 checklist: boxes off the table, nothing near the arm, cables free, tripod untouched."
read -r -p "Press Enter when ready... "

move home        "arm goes to its normal home pose (start position)"
for P in P1_over_stack P2_side_raised P3_over_stack_roll45; do
  case $P in
    P1_over_stack)        DESC="P1: arm swings to its RIGHT; gripper ~12 cm above the yellow X, jaws open ~6 cm, pointing mostly down"; TAG=P1 ;;
    P2_side_raised)       DESC="P2: arm turns to its LEFT over the black mat, raised high; gripper ~23 cm up, jaws closed"; TAG=P2 ;;
    P3_over_stack_roll45) DESC="P3: same place as P1, but the gripper turned 45 deg about its own axis"; TAG=P3 ;;
  esac
  move R_raise "R_raise: only the elbow moves; the gripper rises ~20 cm"
  move "$P"    "$DESC"
  photo "$TAG"
done
move R_raise "R_raise: gripper back up"
move home    "back to home"
echo
echo "Done. Photos are in inputs/photos/phase1/ (front_cam_P1..P3.png, wrist_cam_P1..P3.png)."
