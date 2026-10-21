"""Phase 3: wrist camera overlays at home, P1 and P3 (real | sim | sim edges on real).

    MUJOCO_GL=cgl ~/.venvs/so101_stack_sim/bin/python scripts/render_wrist_check.py
"""
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import calibrate_cameras as C  # noqa: E402
from joint_mapping import JOINT_NAMES, JointMapping, from_dict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/scene.xml")); d = mujoco.MjData(m)
    jm = JointMapping.from_files(); MEAS = yaml.safe_load((ROOT / "inputs/measurements.yaml").read_text())
    rows = []
    with mujoco.Renderer(m, 480, 640) as r:
        for name, fr in C.WRIST_FRAMES.items():
            q = MEAS["arm"]["home_pose"] if fr["pose"] == "home" else json.loads((ROOT / fr["pose"]).read_text())
            q = dict(q, gripper=fr["gripper"]); mujoco.mj_resetData(m, d)
            for b in range(m.nbody):
                if m.body(b).name.startswith("box"):
                    a = m.jnt_qposadr[m.body_jntadr[b]]; d.qpos[a:a + 3] = [5 + b, 5, -5]
            for j, v in zip(JOINT_NAMES, jm.real_to_sim(from_dict(q))):
                d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
            mujoco.mj_forward(m, d); r.update_scene(d, "wrist"); sim = r.render()
            real = np.asarray(Image.open(ROOT / fr["image"]).convert("RGB"))
            e = cv2.Canny(cv2.cvtColor(sim, cv2.COLOR_RGB2GRAY), 40, 120); ov = real.copy(); ov[e > 0] = [255, 0, 255]
            rows.append(np.concatenate([real, sim, ov], 1))
    out = ROOT / "notes/img/phase3_wrist_overlays.png"
    Image.fromarray(np.concatenate(rows, 0)).resize((960, 720)).save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
