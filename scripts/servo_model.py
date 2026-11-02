"""Sim servo model: per-joint position gain kp, velocity damping kv and force cap of the sim position servos
(general affine actuators in sim/scene.xml), e.g. the real-arm fit in sim/assets/servo_fit.yaml (Phase 8)."""
from pathlib import Path

import numpy as np
import yaml


def apply_params(m, p):
    """p: {joint: (kp, kv, force)}."""
    for j, (kp, kv, f) in p.items():
        a = m.actuator(j).id; m.actuator_gainprm[a, 0] = kp; m.actuator_biasprm[a, 1] = -kp; m.actuator_biasprm[a, 2] = -kv
        m.actuator_forcerange[a] = [-f, f]


def load(path):
    return {j: (v["kp"], v["kv"], v["force"]) for j, v in yaml.safe_load(Path(path).read_text())["joints"].items()}


def jitter(p, rng, spread=0.3):
    """Per-joint random scale of kp and kv by exp(U(-spread, spread)) (data generation: servo randomization)."""
    return {j: (kp * float(np.exp(rng.uniform(-spread, spread))), kv * float(np.exp(rng.uniform(-spread, spread))), f)
            for j, (kp, kv, f) in p.items()}
