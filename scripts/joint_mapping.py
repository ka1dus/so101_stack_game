"""Joint mapping between LeRobot SO-101 units and the official MuJoCo model.

LeRobot side (use_degrees=True), verified in LeRobot main ca69a206:
  - body joints: deg = (raw - (range_min + range_max) / 2) * 360 / max_res
    (src/lerobot/motors/motors_bus.py:874-877; max_res = 4096 - 1 for sts3215,
    src/lerobot/motors/feetech/tables.py:190). drive_mode is ignored in DEGREES mode.
  - gripper (RANGE_0_100): g = (clip(raw, min, max) - min) / (max - min) * 100
    (motors_bus.py:871-873), 0 = closed, 100 = open (so_follower.py:80).
  - calibration: the half-turn homing pose reads raw 2047 (feetech.py:278-288), so the
    LeRobot zero is the midpoint of the recorded range, not the homing pose.

Sim side: sim/assets/so101/so101_new_calib.xml (TheRobotStudio SO-ARM100 a758567),
hinge joints in radians, same joint names and order.

Mapping:
  body:    q_sim[rad] = radians(q_lerobot[deg] - offset_deg[joint])
  gripper: q_sim[rad] = radians(gripper_closed_deg + g * gripper_deg_per_unit)
           gripper_deg_per_unit = (range_max - range_min) * 360 / max_res / 100
           (the moving jaw sits directly on the servo horn, so this slope is exact).

offset_deg and gripper_closed_deg come from sim/assets/joint_mapping.yaml; see
notes/phase1.md for how they were obtained and how well they are validated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION = ROOT / "inputs" / "calibration" / "my_follower_arm.json"
DEFAULT_CONFIG = ROOT / "sim" / "assets" / "joint_mapping.yaml"

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
BODY_JOINTS = JOINT_NAMES[:5]
STS3215_MAX_RES = 4096 - 1


@dataclass
class JointMapping:
    calibration: dict
    offset_deg: dict = field(default_factory=lambda: {j: 0.0 for j in BODY_JOINTS})
    gripper_closed_deg: float = -10.75

    @classmethod
    def from_files(cls, calibration_path=DEFAULT_CALIBRATION, config_path=DEFAULT_CONFIG) -> "JointMapping":
        calibration = json.loads(Path(calibration_path).read_text())
        cfg = yaml.safe_load(Path(config_path).read_text())
        offsets = {j: float(cfg["offset_deg"].get(j, 0.0)) for j in BODY_JOINTS}
        return cls(calibration, offsets, float(cfg["gripper_closed_deg"]))

    # ---- LeRobot normalization (mirrors motors_bus._normalize / _unnormalize, float-exact) ----
    def _range(self, joint):
        c = self.calibration[joint]
        return c["range_min"], c["range_max"]

    def raw_to_lerobot(self, raw) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        out = np.empty_like(raw)
        for i, j in enumerate(JOINT_NAMES):
            lo, hi = self._range(j)
            if j == "gripper":
                out[..., i] = (np.clip(raw[..., i], lo, hi) - lo) / (hi - lo) * 100
            else:
                out[..., i] = (raw[..., i] - (lo + hi) / 2) * 360 / STS3215_MAX_RES
        return out

    def lerobot_to_raw(self, q) -> np.ndarray:
        """Float inverse; LeRobot itself truncates with int() (<= 1 tick = 0.088 deg)."""
        q = np.asarray(q, dtype=float)
        out = np.empty_like(q)
        for i, j in enumerate(JOINT_NAMES):
            lo, hi = self._range(j)
            if j == "gripper":
                out[..., i] = np.clip(q[..., i], 0, 100) / 100 * (hi - lo) + lo
            else:
                out[..., i] = q[..., i] * STS3215_MAX_RES / 360 + (lo + hi) / 2
        return out

    @property
    def gripper_deg_per_unit(self) -> float:
        lo, hi = self._range("gripper")
        return (hi - lo) * 360 / STS3215_MAX_RES / 100

    # ---- LeRobot <-> sim ----
    def real_to_sim(self, q_lerobot) -> np.ndarray:
        """LeRobot units [deg x5, gripper 0-100] -> sim qpos [rad x6]."""
        q = np.asarray(q_lerobot, dtype=float)
        out = np.empty_like(q)
        for i, j in enumerate(BODY_JOINTS):
            out[..., i] = np.radians(q[..., i] - self.offset_deg[j])
        out[..., 5] = np.radians(self.gripper_closed_deg + q[..., 5] * self.gripper_deg_per_unit)
        return out

    def sim_to_real(self, q_sim) -> np.ndarray:
        """Sim qpos [rad x6] -> LeRobot units [deg x5, gripper 0-100] (gripper not clipped)."""
        q = np.asarray(q_sim, dtype=float)
        out = np.empty_like(q)
        for i, j in enumerate(BODY_JOINTS):
            out[..., i] = np.degrees(q[..., i]) + self.offset_deg[j]
        out[..., 5] = (np.degrees(q[..., 5]) - self.gripper_closed_deg) / self.gripper_deg_per_unit
        return out

    def lerobot_limits(self) -> np.ndarray:
        """(6, 2) min/max reachable LeRobot values given the calibrated ranges."""
        lo_hi = np.array([self._range(j) for j in JOINT_NAMES], dtype=float)
        return np.stack([self.raw_to_lerobot(lo_hi[:, 0]), self.raw_to_lerobot(lo_hi[:, 1])], axis=1)


def as_dict(q) -> dict:
    return {j: float(v) for j, v in zip(JOINT_NAMES, q)}


def from_dict(d: dict) -> np.ndarray:
    return np.array([d[j] for j in JOINT_NAMES], dtype=float)
