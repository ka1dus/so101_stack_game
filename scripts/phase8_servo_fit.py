"""Phase 8: fit the sim servo model (per joint: position gain kp, velocity damping kv, force cap) to real tracking.

Each real log is replayed open-loop in the sim twin (scripts/phase8_tracking.py SimArm: the sent targets, from the
real first reading, the held box as 38 g at the grip point from `load_from_s`) and the sim readings are compared with
the real ones. Coordinate search per joint (log grid over kp, then kv, then the force cap), 2 rounds; joints are
coupled only through the arm's dynamics, so each joint is scored on its own error.

    MUJOCO_GL=cgl ~/.venvs/so101_eval/bin/python scripts/phase8_servo_fit.py \
        --log data/real_tracking/<stamp>_empty.npz --log data/real_tracking/<stamp>_box.npz \
        --trial data/real_trials/20261027_115530/t001_trace.npy:6.5:12
    (--trial path:load_from_s:until_s, a skill trace before any contact)
Writes sim/assets/servo_fit.yaml (proposal; scene.xml is not changed) and notes/phase8_servo_fit.txt.
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from joint_mapping import JOINT_NAMES  # noqa: E402

ARM = JOINT_NAMES[:5]
START = {j: (998.22, 2.731, 3.35) for j in ARM}           # sim/scene.xml now
KP = np.geomspace(5, 1000, 12); KV = np.geomspace(0.05, 10, 9); FORCE = [0.8, 1.2, 1.8, 2.5, 3.35]


def load(spec):
    """-> dict(cmd, real, load_tick, name)"""
    if spec.endswith(".npz"):
        z = np.load(spec); log = z["log"]; boxed = "_box" in Path(spec).name
        return dict(name=Path(spec).name, cmd=log[:, 6:12].astype(float), real=log[:, :6].astype(float), load_tick=0 if boxed else None)
    path, a, b = spec.rsplit(":", 2); tr = np.load(path); k1 = int(float(b) * 30)
    return dict(name=f"{Path(path).parent.name}/{Path(path).name}[:{b}s]", cmd=tr[:k1, 6:12].astype(float),
                real=tr[:k1, :6].astype(float), load_tick=int(float(a) * 30))


def replay(args):
    seg, params = args
    from phase8_tracking import SimArm
    arm = SimArm(box=seg["load_tick"] == 0, params=params); arm.start(seg["real"][0]); out = []
    for k, c in enumerate(seg["cmd"]):
        if seg["load_tick"] and k == seg["load_tick"]:      # the box is picked up here
            add_load(arm)
        out.append(arm.joints()); arm.send(c)
    return np.array(out)


def add_load(arm):
    s, m, d, mj = arm.s, arm.s.m, arm.s.d, arm.mj; g = s.gb; mb = 0.038; mg = m.body_mass[g]
    pg = np.array([s.xin + 0.04 / 2 + 0.002, 0, s.ztip + 0.010])
    m.body_ipos[g] = (mg * m.body_ipos[g] + mb * pg) / (mg + mb); m.body_mass[g] = mg + mb; mj.mj_setConst(m, d)


def errors(segs, sims):
    """Mean |sim - real| per arm joint over all segments (LeRobot units)."""
    e = [np.abs(sm[:, :5] - sg["real"][:, :5]) for sg, sm in zip(segs, sims)]
    return np.concatenate(e).mean(0)


def score_all(pool, segs, cands):
    jobs = [(sg, p) for p in cands for sg in segs]; res = pool.map(replay, jobs); n = len(segs)
    return [errors(segs, res[i * n:(i + 1) * n]) for i in range(len(cands))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--log", action="append", default=[]); ap.add_argument("--trial", action="append", default=[])
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args(); segs = [load(s) for s in a.log + a.trial]; p = {j: list(v) for j, v in START.items()}
    lines = [f"segments: {[s['name'] for s in segs]}"]
    with Pool(a.workers) as pool:
        e0 = score_all(pool, segs, [p])[0]; lines.append(f"start (scene.xml) error per joint {np.round(e0, 2).tolist()} mean {e0.mean():.2f}")
        for r in range(a.rounds):
            for i, j in enumerate(ARM):
                for k, grid in ((0, KP), (1, KV), (2, FORCE)):
                    cands = []
                    for v in grid:
                        q = {jj: list(vv) for jj, vv in p.items()}; q[j][k] = float(v); cands.append(q)
                    errs = score_all(pool, segs, cands); best = int(np.argmin([e[i] for e in errs]))
                    p[j][k] = float(grid[best])
                print(f"round {r + 1} {j}: kp {p[j][0]:.1f} kv {p[j][1]:.2f} force {p[j][2]:.2f}", flush=True)
            e = score_all(pool, segs, [p])[0]
            lines.append(f"round {r + 1}: error per joint {np.round(e, 2).tolist()} mean {e.mean():.2f}")
    lines += [f"  {j}: kp {v[0]:.1f} N.m/rad, kv {v[1]:.2f} N.m.s/rad, force cap {v[2]:.2f} N.m" for j, v in p.items()]
    txt = "\n".join(lines); print(txt); (ROOT / "notes/phase8_servo_fit.txt").write_text(txt + "\n")
    (ROOT / "sim/assets/servo_fit.yaml").write_text(yaml.safe_dump(
        {"status": "proposal (not applied to scene.xml)", "segments": [s["name"] for s in segs],
         "joints": {j: {"kp": round(v[0], 2), "kv": round(v[1], 3), "force": v[2]} for j, v in p.items()}}, sort_keys=False))


if __name__ == "__main__":
    main()
