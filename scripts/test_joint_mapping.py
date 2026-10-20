"""Tests for joint_mapping.py (sim env):  python -m pytest scripts/test_joint_mapping.py -q"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from joint_mapping import JOINT_NAMES, JointMapping, from_dict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAP = JointMapping.from_files()
RNG = np.random.default_rng(0)


def random_lerobot(n):
    lim = MAP.lerobot_limits()
    return RNG.uniform(lim[:, 0], lim[:, 1], size=(n, 6))


def test_round_trip_real_sim_real():
    q = random_lerobot(10_000)
    np.testing.assert_allclose(MAP.sim_to_real(MAP.real_to_sim(q)), q, atol=1e-9)


def test_round_trip_sim_real_sim():
    q = MAP.real_to_sim(random_lerobot(10_000))
    np.testing.assert_allclose(MAP.real_to_sim(MAP.sim_to_real(q)), q, atol=1e-12)


def test_raw_round_trip():
    q = random_lerobot(10_000)
    np.testing.assert_allclose(MAP.raw_to_lerobot(MAP.lerobot_to_raw(q)), q, atol=1e-9)


def test_lerobot_formula_spot_values():
    # shoulder_pan range 712..3428 -> mid 2070; raw 2047 (homing pose) reads -2.02 deg
    raw = np.array([2047, 2013, 1963, 1999, 2047.5, 2029], dtype=float)
    q = MAP.raw_to_lerobot(raw)
    assert q[0] == pytest.approx((2047 - 2070) * 360 / 4095)
    assert q[4] == pytest.approx(0.0)
    assert q[5] == pytest.approx(0.0)           # gripper at range_min = closed
    assert MAP.raw_to_lerobot(np.array([0, 0, 0, 0, 0, 3511.0]))[5] == pytest.approx(100.0)


def test_gripper_slope():
    assert MAP.gripper_deg_per_unit == pytest.approx((3511 - 2029) * 360 / 4095 / 100)


def test_home_pose_inside_sim_limits():
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/assets/so101/so101_new_calib.xml"))
    home = from_dict(dict(shoulder_pan=-3.16, shoulder_lift=-87.30, elbow_flex=96.79,
                          wrist_flex=48.35, wrist_roll=8.66, gripper=0.74))
    q = MAP.real_to_sim(home)
    for i, j in enumerate(JOINT_NAMES):
        lo, hi = m.jnt_range[m.joint(j).id]
        assert lo - 1e-6 <= q[i] <= hi + 1e-6, (j, np.degrees([q[i], lo, hi]))


def test_calibrated_body_ranges_vs_sim_limits():
    """Report (not fail) where the real calibrated range exceeds the sim joint range."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/assets/so101/so101_new_calib.xml"))
    lim = MAP.real_to_sim(MAP.lerobot_limits().T)  # rows: min, max
    for i, j in enumerate(JOINT_NAMES):
        lo, hi = np.degrees(m.jnt_range[m.joint(j).id])
        print(f"{j:14s} real {np.degrees(lim[0, i]):7.1f}..{np.degrees(lim[1, i]):7.1f}  sim {lo:7.1f}..{hi:7.1f}")
