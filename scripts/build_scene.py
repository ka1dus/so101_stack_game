"""Phase 3: build sim/scene.xml, the calibrated twin of the real play area.

Sources (nothing here is hand-tuned to look right):
  sim/assets/so101/so101_new_calib_camera.xml   official arm + wrist-camera mount (vendor, unmodified)
  sim/assets/joint_mapping.yaml                 offsets (for joint limits), base tilt
  sim/assets/layout.yaml                        table / mats / stacking spot (scripts/scene_layout.py)
  sim/assets/cameras.yaml                       front + wrist camera calibration (scripts/calibrate_cameras.py)
  inputs/measurements.yaml                      box sizes and masses, table size
  sim/assets/textures/                          box face textures (scripts/make_box_textures.py)
Items marked APPROX are placeholders until measured (see notes/phase3.md).

    ~/.venvs/so101_stack_sim/bin/python scripts/build_scene.py      # writes sim/scene.xml and checks it loads
"""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from joint_mapping import BODY_JOINTS, JOINT_NAMES, JointMapping  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "sim"
ROBOT_XML = SIM / "assets/so101/so101_new_calib_camera.xml"
CFG = yaml.safe_load((SIM / "assets/joint_mapping.yaml").read_text())
LAY = yaml.safe_load((SIM / "assets/layout.yaml").read_text())
MEAS = yaml.safe_load((ROOT / "inputs/measurements.yaml").read_text())
CAMS_PATH = SIM / "assets/cameras.yaml"

BOX_TYPES = {v["sim_name"]: v for v in MEAS["boxes"]["variants"]}
# cube-texture file slot -> photo, with the pre-rotation (deg CCW) verified in Phase 3 so each
# photo appears upright from outside the box (+x right/-x left ends, +-y long sides, +-z top/bottom)
CUBE_SLOTS = {"right": ("right", 90), "left": ("left", -90), "up": ("back", 180),
              "down": ("front", 0), "front": ("top", 0), "back": ("bottom", 180)}

# APPROX placeholders (not measured yet; visual only unless noted)
MAT_THICKNESS = 0.002          # black supply mat (M3 fit put mark B +2.7 mm incl. tape)
TAPE_THICKNESS = 0.0002        # yellow tape
BACKDROP = dict(start_from_mount=0.10, wrap=0.05, height=0.20, thickness=0.005)   # Phase 2 spec, collides
TABLE_TOP_THICKNESS = 0.02
LEG = 0.025                    # square leg tube
SHELF_DROP = 0.40              # lower shelf, below the top


def rot_xy(tx, ty):
    rx, ry = np.radians(tx), np.radians(ty)
    return (np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
            @ np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]))


def mat2quat(R):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, R.ravel()); return q


def yaw_quat(deg):
    return np.array([np.cos(np.radians(deg) / 2), 0, 0, np.sin(np.radians(deg) / 2)])


def make_cube_textures():
    """Square, pre-rotated copies of the face textures for MuJoCo cube textures."""
    out = SIM / "assets/textures/cube"
    out.mkdir(parents=True, exist_ok=True)
    for t in BOX_TYPES:
        for slot, (face, rot) in CUBE_SLOTS.items():
            im = Image.open(SIM / f"assets/textures/{t}_{face}.png").rotate(rot, expand=True).resize((512, 512), Image.LANCZOS)
            im.save(out / f"{t}_{slot}.png")


def build():
    make_cube_textures()
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    spec.modelname = "so101_stack_game"
    spec.meshdir = str(SIM / "assets/so101/assets")      # absolute while compiling here;
    spec.texturedir = str(SIM / "assets/textures")       # main() writes them relative to sim/
    spec.option.timestep = 0.002
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 640, 480

    # ---- arm placement: table = z 0, pan axis at the origin, fitted tilt ----
    R = rot_xy(*CFG["base_tilt_deg"].values())
    pan_off = np.array([-LAY["arm_base"]["base_origin_offset_from_pan_axis_m"][0], 0, 0])   # pan axis in base frame
    base = spec.body("base")
    base.pos = -R @ pan_off + np.array([0, 0, LAY["arm_base"]["base_origin_above_table_m"]])
    base.quat = mat2quat(R)

    # ---- joint limits = the real calibrated ranges mapped into sim (vendor limits are narrower) ----
    jm = JointMapping.from_files()
    lim = jm.lerobot_limits()                       # (6, 2) LeRobot units
    lo, hi = jm.real_to_sim(lim[:, 0]), jm.real_to_sim(lim[:, 1])
    for i, j in enumerate(JOINT_NAMES):
        a, b = sorted((lo[i], hi[i]))
        spec.joint(j).range = [a, b]
        spec.actuator(j).ctrlrange = [a, b]

    wb = spec.worldbody
    # ---- room (APPROX, visual; randomized in data generation) ----
    spec.add_texture(name="carpet", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                     rgb1=[0.12, 0.14, 0.25], rgb2=[0.5, 0.45, 0.3], mark=mujoco.mjtMark.mjMARK_RANDOM,
                     random=0.08, markrgb=[0.6, 0.55, 0.35], width=512, height=512)
    spec.add_material(name="carpet", textures=["", "carpet"], texrepeat=[8, 8], texuniform=True)
    table_h = MEAS["table"]["height_from_floor"]
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], pos=[0, 0, -table_h], material="carpet")

    # ---- table ----
    T = LAY["table"]; c = np.array(T["centre"]); hd = T["heading_deg"]; s = T["size_m"] / 2
    white = [0.93, 0.94, 0.96, 1]
    wb.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[s, s, TABLE_TOP_THICKNESS / 2],
                pos=[c[0], c[1], -TABLE_TOP_THICKNESS / 2], quat=yaw_quat(hd), rgba=white, friction=[0.5, 0.005, 0.0001])
    u = np.array([np.cos(np.radians(hd)), np.sin(np.radians(hd))]); v = np.array([-u[1], u[0]])
    for i, (a, b) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
        p = c + (s - LEG / 2) * (a * u + b * v)
        wb.add_geom(name=f"table_leg{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[LEG / 2, LEG / 2, table_h / 2],
                    pos=[p[0], p[1], -table_h / 2], quat=yaw_quat(hd), rgba=white, contype=0, conaffinity=0)
    wb.add_geom(name="table_shelf", type=mujoco.mjtGeom.mjGEOM_BOX, size=[s, s, 0.005], pos=[c[0], c[1], -SHELF_DROP],
                quat=yaw_quat(hd), rgba=white, contype=0, conaffinity=0)

    # ---- play areas ----
    S_ = LAY["supply_area"]
    wb.add_geom(name="supply_mat", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[S_["size_m"][0] / 2, S_["size_m"][1] / 2, MAT_THICKNESS / 2],
                pos=[*S_["centre"], MAT_THICKNESS / 2], quat=yaw_quat(S_["heading_1_2_deg"]), rgba=[0.03, 0.03, 0.03, 1],
                friction=[0.6, 0.005, 0.0001])
    K = LAY["stack_area"]
    wb.add_geom(name="stack_area", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[K["size_m"][0] / 2, K["size_m"][1] / 2, TAPE_THICKNESS / 2],
                pos=[*K["centre"], TAPE_THICKNESS / 2], quat=yaw_quat(K["heading_1_2_deg"]), rgba=[0.96, 0.76, 0.12, 1],
                friction=[0.45, 0.005, 0.0001])
    sp = K["stacking_spot"]
    wb.add_site(name="stacking_spot", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.004, 0.0001, 0],
                pos=[sp[0], sp[1], TAPE_THICKNESS], rgba=[0.2, 0.2, 0.2, 1], group=1)

    # ---- backdrop on the table's right edge (Phase 2), collides with the arm ----
    mc, rc = np.array(T["corners"]["mount"]), np.array(T["corners"]["right"])
    e = (rc - mc) / np.linalg.norm(rc - mc)
    out = -np.array([-e[1], e[0]])                    # outward normal of the right edge (away from the table)
    if np.dot(out, c - mc) > 0:
        out = -out
    L = np.linalg.norm(rc - mc) - BACKDROP["start_from_mount"]
    mid = mc + e * (BACKDROP["start_from_mount"] + L / 2) + out * BACKDROP["thickness"] / 2
    wb.add_geom(name="backdrop", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[L / 2, BACKDROP["thickness"] / 2, BACKDROP["height"] / 2],
                pos=[mid[0], mid[1], BACKDROP["height"] / 2], quat=yaw_quat(np.degrees(np.arctan2(e[1], e[0]))),
                rgba=[0.02, 0.02, 0.02, 1])
    e2 = -np.array([np.cos(np.radians(hd + 90)), np.sin(np.radians(hd + 90))])   # along the far edge from the right corner
    if np.dot(e2, c - rc) < 0:
        e2 = -e2
    mid2 = rc + e2 * BACKDROP["wrap"] / 2 + (-e) * 0 + e * BACKDROP["thickness"] / 2
    wb.add_geom(name="backdrop_wrap", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[BACKDROP["wrap"] / 2, BACKDROP["thickness"] / 2, BACKDROP["height"] / 2],
                pos=[mid2[0], mid2[1], BACKDROP["height"] / 2], quat=yaw_quat(np.degrees(np.arctan2(e2[1], e2[0]))),
                rgba=[0.02, 0.02, 0.02, 1])

    # ---- boxes: 3 x no3, 4 x no35, 1 x no10, free bodies, loose grid on the supply mat ----
    for t in BOX_TYPES:
        spec.add_texture(name=f"tex_{t}", type=mujoco.mjtTexture.mjTEXTURE_CUBE,
                         cubefiles=[f"cube/{t}_{slot}.png" for slot in ("right", "left", "up", "down", "front", "back")])
        spec.add_material(name=f"mat_{t}", textures=["", f"tex_{t}"])
    ang = np.radians(S_["heading_1_2_deg"]); su = np.array([np.cos(ang), np.sin(ang)]); sv = np.array([-su[1], su[0]])
    slots = [(i, j) for i in (-0.06, 0.0, 0.06) for j in (-0.05, 0.05)] + [(0.0, 0.0), (0.06, 0.0)]
    order = ["no3"] * 3 + ["no35"] * 4 + ["no10"]
    for n, (t, (a, b)) in enumerate(zip(order, [(-0.06, -0.05), (0.0, -0.05), (0.06, -0.05), (-0.06, 0.05),
                                                (0.0, 0.05), (0.06, 0.05), (-0.065, 0.0), (0.065, 0.0)][:8])):
        d = BOX_TYPES[t]["dimensions"]
        p = np.array(S_["centre"]) + a * su + b * sv
        body = wb.add_body(name=f"box{n}_{t}", pos=[p[0], p[1], MAT_THICKNESS + d["height"] / 2 + 0.0005],
                           quat=yaw_quat(S_["heading_1_2_deg"] + 90))
        body.add_freejoint(name=f"box{n}_free")
        body.add_geom(name=f"box{n}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[d["length"] / 2, d["width"] / 2, d["height"] / 2], mass=BOX_TYPES[t]["mass_g"] / 1000,
                      material=f"mat_{t}", friction=[0.5, 0.005, 0.0001])

    # ---- lights (ceiling, diffuse) ----
    wb.add_light(name="ceiling", pos=[0.2, 0, 1.6], dir=[0, 0, -1], diffuse=[0.55, 0.55, 0.55],
                 ambient=[0.35, 0.35, 0.35], specular=[0.05, 0.05, 0.05], castshadow=True)
    wb.add_light(name="fill", pos=[0.9, 0.2, 1.0], dir=[-0.6, -0.1, -0.8], diffuse=[0.25, 0.25, 0.25], castshadow=False)

    # ---- cameras ----
    cams = (yaml.safe_load(CAMS_PATH.read_text()) if CAMS_PATH.exists() else None) or {}
    f = cams.get("front")
    if f:
        wb.add_camera(name="front", pos=f["pos"], xyaxes=f["xyaxes"], fovy=f["fovy"])
    else:   # placeholder until scripts/calibrate_cameras.py has run
        tape = MEAS["cameras"]["front"]["position"]
        wb.add_camera(name="front", pos=[tape["x"], tape["y"], tape["z"]], mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
                      targetbody="world", fovy=45)
    w = cams.get("wrist")
    if w:
        spec.body(w["body"]).add_camera(name="wrist", pos=w["pos"], xyaxes=w["xyaxes"], fovy=w["fovy"])
    else:   # placeholder: measurements.yaml offsets from the fixed jaw tip, looking along the jaws
        spec.body("gripper").add_camera(name="wrist", pos=[-0.0089, 0.0128, -0.068], xyaxes=[0, -1, 0, -1, 0, 0], fovy=70)

    # ---- home keyframe (LeRobot home pose through the validated mapping) ----
    home = jm.real_to_sim(np.array([MEAS["arm"]["home_pose"][j] for j in JOINT_NAMES]))
    model = spec.compile()
    qpos = model.qpos0.copy()
    for i, j in enumerate(JOINT_NAMES):
        qpos[model.jnt_qposadr[model.joint(j).id]] = home[i]
    spec.add_key(name="home", qpos=qpos.tolist(), ctrl=home.tolist())
    return spec


def main():
    spec = build()
    xml = spec.to_xml()                                   # absolute asset dirs -> make them relative to sim/
    xml = xml.replace(str(SIM / "assets/so101/assets"), "assets/so101/assets").replace(str(SIM / "assets/textures"), "assets/textures")
    (SIM / "scene.xml").write_text(xml)
    m = mujoco.MjModel.from_xml_path(str(SIM / "scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    print(f"sim/scene.xml: {m.nbody} bodies, {m.ngeom} geoms, {m.ncam} cameras, {m.nu} actuators")
    # settle the boxes for 1 s, arm held at home
    for _ in range(500):
        mujoco.mj_step(m, d)
    for b in range(m.nbody):
        n = m.body(b).name
        if n.startswith("box"):
            print(f"  {n:12s} z after 1 s {d.xpos[b][2]*1000:6.1f} mm")
    print("  contacts arm-world at home:", sum(1 for i in range(d.ncon)
          if (m.geom_bodyid[d.contact[i].geom1] == 0) != (m.geom_bodyid[d.contact[i].geom2] == 0)
          and not any(m.body(m.geom_bodyid[g]).name.startswith("box") for g in (d.contact[i].geom1, d.contact[i].geom2))))


if __name__ == "__main__":
    main()
