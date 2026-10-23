"""Phase 5: generate raw sim episodes with the scripted expert (successes only).

Per episode (data/raw/<run>/ep_NNNNNN/):
  front.mp4, wrist.mp4   640x480, 30 fps, H.264 crf 18 (re-encoded by export_lerobot.py)
  data.npz               state[T,6], action[T,6] in LeRobot units (sim_to_real), t[T] in s
  meta.json              task string, mode, picked type, stack composition, offsets, randomization, expert result
Frame t: observation after control tick t; action[t] = joint targets sent at tick t+1 (the command that follows
that observation); the last action repeats the final target.

Modes: --mode act  (deterministic nearest-to-stack pick, one task string)
       --mode vla  (requested type drawn uniformly, task string names the size)
An ACT episode is also a valid VLA episode (its picked size is in meta), so export_lerobot.py can build the VLA
dataset from both runs.

    MUJOCO_GL=cgl ~/.venvs/so101_stack_sim/bin/python scripts/generate.py --n 20 --mode act --run act_pilot
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import expert as E  # noqa: E402
import game_states as G  # noqa: E402
import grasp_test as GT  # noqa: E402
from joint_mapping import JOINT_NAMES, JointMapping  # noqa: E402
from randomize import Randomizer, H, W  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FPS = GT.CTRL_HZ
SIZE = {"no3": "large", "no35": "medium", "no10": "small"}
TASK_ACT = "Pick up the next box and place it on top of the stack"
TASK_VLA = "Pick up the {} box and place it on top of the stack"


class VideoWriter:
    def __init__(self, path):
        self.p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18",
                                   "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)

    def write(self, img):
        self.p.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())

    def close(self):
        self.p.stdin.close(); self.p.wait()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="successful episodes to write")
    ap.add_argument("--mode", choices=("act", "vla"), default="act")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", default=None)
    ap.add_argument("--max-attempts", type=int, default=None)
    ap.add_argument("--hard-p", type=float, default=0.6, help="share of hard states (Phase 5 section 10)")
    ap.add_argument("--tall-p", type=float, default=0.6, help="share of stack heights 4-7 (data v1 only)")
    ap.add_argument("--data-version", choices=("v1", "v2"), default="v2",
                    help="v2 (notes/phase6_data_v2_plan.md): v2 expert grasp, episode-kind mix, dense-supply heights, strict rest check")
    ap.add_argument("--kinds", default="clean:0.45,dart:0.40,retry:0.15", help="v2 episode-kind mix")
    ap.add_argument("--height-min", type=int, default=None, help="targeted (v3): stack heights height-min..7, uniform")
    ap.add_argument("--want-types", default=None, help="targeted (v3): comma list; only states where the rule picks these")
    ap.add_argument("--min-crooked-mm", type=float, default=0.0)
    ap.add_argument("--crooked-share", type=float, default=1.0, help="share of episodes that must meet --min-crooked-mm")
    ap.add_argument("--states", choices=("v2", "game"), default="game",
                    help="game (full v2 run): template-layout games + rule-spaced supplies + some dense (SUPPLY_P_GAME); v2: pilot sampling")
    a = ap.parse_args()
    out = ROOT / "data/raw" / (a.run or f"{a.mode}_seed{a.seed}"); out.mkdir(parents=True, exist_ok=True)
    sim = GT.Sim(); m, d = sim.m, sim.d; rng = np.random.default_rng(a.seed)
    ex = E.Expert(sim, rng); rand = Randomizer(m); jm = JointMapping.from_files()
    renderer = mujoco.Renderer(m, H, W)
    arm_q = list(sim.qadr) + [sim.gadr]; act_ids = [sim.act[j] for j in JOINT_NAMES]
    written, attempts, fails = len(list(out.glob("ep_*"))), 0, {}
    if written:                                   # resuming: don't replay the same random draws
        rng = np.random.default_rng([a.seed, written]); ex.rng = rng
    t0 = time.time(); start = written
    kind_names = [k.split(":")[0] for k in a.kinds.split(",")]
    kind_p = np.array([float(k.split(":")[1]) for k in a.kinds.split(",")]); kind_p /= kind_p.sum()
    while written < a.n and (a.max_attempts is None or attempts < a.max_attempts):
        attempts += 1
        requested = str(rng.choice(G.TYPE_ORDER)) if a.mode == "vla" else None
        if a.data_version == "v2":
            hp = G.HEIGHT_P_V2 if a.height_min is None else {h: 1.0 for h in range(a.height_min, G.MAX_STACK + 1)}
            need_crooked = rng.random() < a.crooked_share
            for _ in range(3000):
                st, _ = G.sample_state(rng, m, d, mode=a.mode, requested=requested, home=E.home_fn(sim),
                                       hard_p=a.hard_p, height_p=hp, rest=2.0,
                                       supply_p=G.SUPPLY_P_GAME if a.states == "game" else None)
                off = max((np.linalg.norm(o) for o in st.meta.get("offsets_mm", [])), default=0.0)
                if (not need_crooked or off >= a.min_crooked_mm) and \
                        (not a.want_types or st.supply[st.pick].type in a.want_types.split(",")):
                    break
            kind = str(rng.choice(kind_names, p=kind_p))
        else:
            st, _ = G.sample_state(rng, m, d, mode=a.mode, requested=requested, home=E.home_fn(sim),
                                   hard_p=a.hard_p, tall_p=a.tall_p)
            kind = "v1"
        rparams = rand.apply(rng)
        ep = out / f"ep_{written:06d}"; tmp = out / f".tmp_{written:06d}"
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
        vf, vw = VideoWriter(tmp / "front.mp4"), VideoWriter(tmp / "wrist.mp4")
        states, ctrls = [], []

        def record():
            states.append(d.qpos[arm_q].copy()); ctrls.append(sim.clean_ctrl())   # the plan (= d.ctrl unless DART noise is on)
            renderer.update_scene(d, "front"); vf.write(rand.post_front(renderer.render()))
            renderer.update_scene(d, "wrist"); vw.write(rand.post_wrist(renderer.render()))

        r = ex.run(st, record=record, kind=kind, mode=a.mode, requested=requested)
        vf.close(); vw.close()
        if not r["ok"] or len(states) < 10:
            fails[r["reason"]] = fails.get(r["reason"], 0) + 1
            shutil.rmtree(tmp); continue
        state = jm.sim_to_real(np.array(states)).astype(np.float32)
        ctrl = jm.sim_to_real(np.array(ctrls)).astype(np.float32)
        action = np.concatenate([ctrl[1:], ctrl[-1:]], 0)
        np.savez_compressed(tmp / "data.npz", state=state, action=action, t=np.arange(len(state)) / FPS)
        pick = st.supply[st.pick]
        meta = dict(task=TASK_ACT if a.mode == "act" else TASK_VLA.format(SIZE[pick.type]),
                    task_vla=TASK_VLA.format(SIZE[pick.type]), mode=a.mode, requested=requested,
                    picked_type=pick.type, stack_types=[b.type for b in st.stack], fps=FPS, frames=len(state),
                    robot_type="so101_follower", joint_names=list(JOINT_NAMES), seed=a.seed, attempt=attempts,
                    state_meta=st.meta, expert=r, randomization=rparams, data_version=a.data_version, kind=kind, states=a.states,
                    pairing_final=f"{pick.type}_on_{st.stack[-1].type}")   # retry starts may re-choose the box
        (tmp / "meta.json").write_text(json.dumps(jsonable(meta), indent=1))
        tmp.rename(ep); written += 1
        print(f"ep {written:4d}/{a.n} h={st.meta['stack_height']} {meta['pairing_final']:14s} {kind:5s} {len(state):4d} frames "
              f"| attempts {attempts}, fails {fails} | {(time.time() - t0) / (written - start):.1f} s/ep", flush=True)
    rand.reset(); renderer.close()
    print(f"done: {written} episodes in {out} ({attempts} attempts, failures {fails})")


if __name__ == "__main__":
    main()
