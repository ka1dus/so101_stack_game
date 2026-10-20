"""Cross-check joint_mapping.py against LeRobot's own normalization code (no hardware).

Run in the LeRobot env (read-only use of the installed package):
    conda activate lerobot && python scripts/check_mapping_vs_lerobot.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from joint_mapping import JOINT_NAMES, JointMapping  # noqa: E402

from lerobot.motors import Motor, MotorCalibration, MotorNormMode  # noqa: E402
from lerobot.motors.feetech import FeetechMotorsBus  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
cal = json.loads((ROOT / "inputs/calibration/my_follower_arm.json").read_text())
calib = {j: MotorCalibration(**c) for j, c in cal.items()}
motors = {j: Motor(calib[j].id, "sts3215", MotorNormMode.RANGE_0_100 if j == "gripper" else MotorNormMode.DEGREES)
          for j in JOINT_NAMES}
bus = FeetechMotorsBus(port="/dev/null", motors=motors, calibration=calib)  # never connected
ours = JointMapping.from_files()

worst_norm, worst_unnorm = 0.0, 0.0
for i, j in enumerate(JOINT_NAMES):
    mid = calib[j].id
    for raw in range(calib[j].range_min, calib[j].range_max + 1):
        theirs = bus._normalize({mid: raw})[mid]
        vec = np.zeros(6); vec[i] = raw
        worst_norm = max(worst_norm, abs(ours.raw_to_lerobot(vec)[i] - theirs))
        back = bus._unnormalize({mid: theirs})[mid]          # LeRobot truncates with int()
        vec_q = np.zeros(6); vec_q[i] = theirs
        worst_unnorm = max(worst_unnorm, abs(ours.lerobot_to_raw(vec_q)[i] - back))
print(f"max |ours - lerobot| normalize   : {worst_norm:.2e} (LeRobot units)")
print(f"max |ours - lerobot| unnormalize : {worst_unnorm:.3f} ticks (LeRobot int() truncation, expect < 1)")
assert worst_norm < 1e-9 and worst_unnorm < 1.0
print("OK")
