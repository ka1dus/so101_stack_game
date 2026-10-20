"""Phase 0 environment check: load a MuJoCo model, step physics, render offscreen to PNG.

Usage (sim env):
    MUJOCO_GL=cgl  python scripts/env_check.py   # macOS headless (CGL)
    MUJOCO_GL=glfw python scripts/env_check.py   # macOS via hidden GLFW window
    MUJOCO_GL=egl  python scripts/env_check.py   # Linux / WSL with GPU EGL

Writes notes/img/phase0_render_<backend>.png and prints timing numbers that bound
data-generation throughput (two 640x480 cameras per control step at 30 fps).
"""

import os
import platform
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notes" / "img"

# Placeholder scene: white table, yellow stack pad, black supply mat, three box types
# (dimensions/masses from inputs/measurements.yaml) dropped into a stack. Not the
# calibrated twin; that is Phase 3.
XML = """
<mujoco model="phase0_check">
  <option timestep="0.002"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0.2 0 1.2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="table" type="box" size="0.19 0.19 0.01" pos="0.21 0.03 -0.01" rgba="0.95 0.95 0.95 1"/>
    <geom name="stack_pad" type="box" size="0.107 0.073 0.0005" pos="0.199 -0.102 0.0005"
          euler="0 0 47.2" rgba="0.95 0.8 0.1 1"/>
    <geom name="supply_mat" type="box" size="0.1045 0.096 0.0005" pos="0.148 0.148 0.0005"
          euler="0 0 55.3" rgba="0.05 0.05 0.05 1"/>
    <body name="no3" pos="0.196 -0.101 0.010">
      <freejoint/>
      <geom type="box" size="0.035 0.020 0.0085" mass="0.038" rgba="0.2 0.6 0.6 1"/>
    </body>
    <body name="no35" pos="0.198 -0.100 0.030" euler="0 0 8">
      <freejoint/>
      <geom type="box" size="0.030 0.0175 0.008" mass="0.031" rgba="0.2 0.5 0.8 1"/>
    </body>
    <body name="no10" pos="0.195 -0.102 0.048" euler="0 0 -6">
      <freejoint/>
      <geom type="box" size="0.026 0.0145 0.0065" mass="0.021" rgba="0.3 0.7 0.4 1"/>
    </body>
    <body name="front_lookat" pos="0.03 0.03 0.0"/>
    <camera name="front" pos="0.865 0.156 0.38" mode="targetbody" target="front_lookat" fovy="55"/>
  </worldbody>
</mujoco>
"""


def main():
    backend = os.environ.get("MUJOCO_GL", "") or "default"
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    t0 = time.perf_counter()
    n_steps = int(1.0 / model.opt.timestep)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
    t_phys = time.perf_counter() - t0
    top_z = data.body("no10").xpos[2]

    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data, camera="front")
    img = renderer.render()

    n_frames = 60
    t0 = time.perf_counter()
    for _ in range(n_frames):
        renderer.update_scene(data, camera="front")
        renderer.render()
    t_render = (time.perf_counter() - t0) / n_frames

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"phase0_render_{backend}.png"
    Image.fromarray(img).save(out)

    print(f"platform        : {platform.system()} {platform.machine()}")
    print(f"mujoco          : {mujoco.__version__}")
    print(f"MUJOCO_GL       : {backend}")
    print(f"image           : {img.shape} {img.dtype}, mean={img.mean():.1f}, nonblank={bool(img.std() > 1)}")
    print(f"physics         : {n_steps} steps in {t_phys*1e3:.1f} ms ({n_steps/t_phys:.0f} steps/s)")
    print(f"stack settled   : top box z = {top_z*100:.2f} cm (expected 3.95 cm = 1.7+1.6+0.65)")
    print(f"render 640x480  : {t_render*1e3:.2f} ms/frame ({1/t_render:.0f} fps)")
    print(f"saved           : {out.relative_to(ROOT)}")
    renderer.close()


if __name__ == "__main__":
    main()
