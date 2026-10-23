"""Phase 7: sim test set for hand_in_workspace(): `front` video with a hand stand-in reaching into the play area.

The hand is added at run time (MjSpec; sim/scene.xml is untouched): a mocap forearm + palm + fingers, no collisions,
random skin tone and sleeve colour. Episode kinds:
  robot   the scripted expert does a full pick-and-place (the arm moves); half of them a hand reaches in meanwhile
  human   the arm holds at home for 8 s; a hand reaches in (to the stack or the supply) and out
  near    the arm at home; the hand moves around just outside the table edge (should NOT trigger)
Recorded per 30 Hz frame: front.mp4 (Phase 5 randomization, plus a small per-frame gain flicker in some episodes),
joint readings (LeRobot units), and truth: hand pixels inside the workspace mask (segmentation) and whether the palm
is inside the workspace volume.

    MUJOCO_GL=cgl ~/.venvs/so101_stack_sim/bin/python scripts/hand_testset.py --n 30 --seed 9001 --out data/perception/hand_v1
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(ROOT))
import expert as E  # noqa: E402
import game_states as G  # noqa: E402
import grasp_test as GT  # noqa: E402
from expert import home_fn  # noqa: E402
from joint_mapping import JOINT_NAMES  # noqa: E402
from perception.camera import from_mujoco  # noqa: E402
from perception.hand import workspace_mask  # noqa: E402
from randomize import H, W, Randomizer  # noqa: E402

FPS = 30
KIND_P = {"robot": 0.45, "human": 0.4, "near": 0.15}
SKIN = [(0.98, 0.84, 0.72), (0.92, 0.72, 0.58), (0.78, 0.57, 0.42), (0.6, 0.42, 0.3), (0.42, 0.28, 0.2), (0.3, 0.2, 0.15)]
TABLE = {k: np.array(v) for k, v in G.LAY["table"]["corners"].items()}
HAND_PX_MIN = 150             # truth "hand in workspace": at least this many hand pixels inside the workspace mask


def build_model():
    spec = mujoco.MjSpec.from_file(str(ROOT / "sim/scene.xml"))
    b = spec.worldbody.add_body(name="hand", mocap=True, pos=[3, 3, -2])
    kw = dict(contype=0, conaffinity=0, rgba=[0.8, 0.6, 0.5, 1])
    b.add_geom(name="hand_forearm", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[-0.32, 0, 0.035, -0.05, 0, 0.005], size=[0.032, 0, 0], **kw)
    b.add_geom(name="hand_palm", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, 0], size=[0.045, 0.04, 0.013], **kw)
    for i, y in enumerate((-0.027, -0.009, 0.009, 0.027)):
        b.add_geom(name=f"hand_finger{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0.045, y, 0, 0.11 - 0.008 * abs(i - 1.5), y, -0.012],
                   size=[0.009, 0, 0], **kw)
    b.add_geom(name="hand_thumb", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0.02, 0.04, -0.005, 0.06, 0.065, -0.012], size=[0.01, 0, 0], **kw)
    return spec.compile()


def hand_path(rng, kind, t_total):
    """(times, positions, yaws) of the palm; parked out of view outside [t_in, t_out]."""
    far = TABLE["far"]; side = TABLE["left"] if rng.random() < 0.5 else TABLE["right"]
    e = far + rng.uniform(0.15, 0.85) * (side - far)                  # entry point on a human-side table edge
    out_dir = e - G.LAY["table"]["centre"]; out_dir /= np.linalg.norm(out_dir)
    start = np.r_[e + out_dir * 0.2, rng.uniform(0.08, 0.2)]
    if kind == "near":                                               # stays 4-12 cm outside the edge
        tgt = np.r_[e + out_dir * rng.uniform(0.04, 0.12), rng.uniform(0.03, 0.15)]
    elif rng.random() < 0.5:
        tgt = np.r_[np.array(G.LAY["stack_area"]["stacking_spot"]) + rng.normal(0, 0.03, 2), rng.uniform(0.04, 0.16)]
    else:
        tgt = np.r_[np.array(G.LAY["supply_area"]["corners"]).mean(0) + rng.normal(0, 0.04, 2), rng.uniform(0.03, 0.1)]
    v = rng.uniform(0.25, 0.7); T = np.linalg.norm(tgt - start) / v; dwell = rng.uniform(0.3, 2.0)
    t_in = rng.uniform(0.5, max(0.6, t_total - 2 * T - dwell - 0.5))
    yaw = float(np.arctan2(*(tgt[:2] - start[:2])[::-1]))
    def at(t):
        if t < t_in or t > t_in + 2 * T + dwell:
            return None
        s = (t - t_in) / T if t < t_in + T else 1.0 if t < t_in + T + dwell else 1 - (t - t_in - T - dwell) / T
        s = s * s * (3 - 2 * s)
        return start + s * (tgt - start)
    return at, yaw, dict(start=start.round(3).tolist(), target=tgt.round(3).tolist(), speed=round(v, 2), dwell=round(dwell, 2), t_in=round(t_in, 2))


def episode(sim, ex, rand, ren, rng, out):
    m, d = sim.m, sim.d
    kind = str(rng.choice(list(KIND_P), p=list(KIND_P.values())))
    with_hand = kind != "robot" or rng.random() < 0.5
    vis = rand.apply(rng)
    skin = np.array(SKIN[rng.integers(len(SKIN))]) * rng.uniform(0.9, 1.1)
    sleeve = rng.uniform(0.05, 0.9, 3) if rng.random() < 0.5 else skin
    for g in range(m.ngeom):
        if m.geom(g).name.startswith("hand_"):
            m.geom_rgba[g, :3] = np.clip(sleeve if m.geom(g).name == "hand_forearm" else skin, 0, 1)
    flicker = rng.random() < 0.3
    st, _ = G.sample_state(rng, m, d, home=home_fn(sim), height_p=G.HEIGHT_P_V2, rest=1.0, hard_p=0.6, supply_p=G.SUPPLY_P_EVAL_GAME)
    cam = from_mujoco(m.cam_pos[rand.cam["front"]], m.cam_quat[rand.cam["front"]])   # true (jittered) camera for the truth mask
    ws = workspace_mask(cam)
    t_total = 14.0 if kind == "robot" else 8.0
    path, yaw, pinfo = hand_path(rng, kind, t_total) if with_hand else (lambda t: None, 0.0, None)
    hid = m.body("hand").id; mid = m.body_mocapid[hid]
    q = np.zeros(4); mujoco.mju_euler2Quat(q, np.r_[0, np.radians(15), yaw], "xyz"); d.mocap_quat[mid] = q
    hand_geoms = np.zeros(m.ngeom + 1, bool)
    for g in range(m.ngeom):
        hand_geoms[g] = m.geom(g).name.startswith("hand_")
    vw = cv2.VideoWriter(str(out / "front.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rows = []; tick = [0]

    def record():
        t = tick[0] / FPS; tick[0] += 1
        p = path(t)
        d.mocap_pos[mid] = p if p is not None else [3, 3, -2]
        mujoco.mj_kinematics(m, d)
        ren.update_scene(d, "front"); img = rand.post_front(ren.render())
        if flicker:
            img = np.clip(img.astype(np.float32) * rng.normal(1, 0.012), 0, 255).astype(np.uint8)
        vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        ren.enable_segmentation_rendering(); ren.update_scene(d, "front"); seg = ren.render()[..., 0]; ren.disable_segmentation_rendering()
        hp = hand_geoms[np.where(seg >= 0, seg, m.ngeom)]
        q_real = sim.jm.sim_to_real(np.r_[d.qpos[sim.qadr], d.qpos[sim.gadr]])
        rows.append(np.r_[q_real, float((hp & ws).sum()), float(hp.sum()), p if p is not None else [np.nan] * 3])

    if kind == "robot":
        ex.run(st, record=record, kind="clean")
        sim.on_tick = None
        while tick[0] < t_total * FPS:
            sim.wait(1 / FPS); record()
    else:
        home_fn(sim)(m, d); sim.sync_plan(); mujoco.mj_forward(m, d)
        for _ in range(int(t_total * FPS)):
            sim.wait(1 / FPS); record()
    vw.release(); d.mocap_pos[mid] = [3, 3, -2]
    A = np.array(rows, np.float32)
    meta = dict(kind=kind, with_hand=bool(with_hand), hand=pinfo, flicker=bool(flicker), frames=len(A), hand_px_min=HAND_PX_MIN,
                hand_frames=int((A[:, 6] >= HAND_PX_MIN).sum()), randomization=vis,
                camera=dict(pos=m.cam_pos[rand.cam["front"]].tolist(), quat=m.cam_quat[rand.cam["front"]].tolist()))
    np.save(out / "trace.npy", A); (out / "meta.json").write_text(json.dumps(meta))
    return meta


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30); ap.add_argument("--seed", type=int, default=9001)
    ap.add_argument("--out", default="data/perception/hand_v1")
    a = ap.parse_args()
    sim = GT.Sim(); m = build_model()
    sim.m, sim.d, sim.kin = m, mujoco.MjData(m), mujoco.MjData(m)      # body/geom/actuator ids unchanged: the hand is appended
    rng = np.random.default_rng(a.seed); ex = E.Expert(sim, rng)
    ex.arm_geoms = [g for g in ex.arm_geoms if not m.geom(g).name.startswith("hand_")]
    rand = Randomizer(m); ren = mujoco.Renderer(m, H, W)
    for i in range(a.n):
        out = Path(a.out) / f"h{a.seed}_{i:04d}"; out.mkdir(parents=True, exist_ok=True)
        mt = episode(sim, ex, rand, ren, rng, out)
        print(f"{out.name}: {mt['kind']:5s} hand={mt['with_hand']} hand_frames={mt['hand_frames']}/{mt['frames']}", flush=True)


if __name__ == "__main__":
    main()
