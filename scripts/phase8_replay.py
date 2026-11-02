"""Phase 8: replay a real trial's commands open-loop on the sim arm and compare the tracking (real vs sim).

The real trace (t###_trace.npy: per tick reading[6], sent target[6], time) is sent tick by tick to the MuJoCo twin,
started from the real first reading, with the template supply and the trial's stack built on the stacking spot.
Prints per phase (reach / grasp / carry) the target-minus-reading gap per joint, real vs sim, and the jaw-tip
height, and saves a plot to notes/img/.

    MUJOCO_GL=cgl ~/.venvs/so101_eval/bin/python scripts/phase8_replay.py data/real_trials/20261027_115530 1
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from game.backends import SimBackend  # noqa: E402
from joint_mapping import JOINT_NAMES  # noqa: E402
from perception.drop import Kinematics  # noqa: E402
from perception.stack import SPOT  # noqa: E402
from perception.supply import BOX  # noqa: E402

POLICY = ROOT / "data/policies/act_v4_ft_040000"


def set_box(m, d, body, xy, z, yaw):
    a = m.jnt_qposadr[m.body(body).jntadr[0]]
    d.qpos[a:a + 3] = [xy[0], xy[1], z]; d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    d.qvel[m.jnt_dofadr[m.body(body).jntadr[0]]:][:6] = 0


def replay(sess, n):
    sess = Path(sess); tr = np.load(sess / f"t{n:03d}_trace.npy"); real_q, cmd = tr[:, :6].astype(float), tr[:, 6:12].astype(float)
    t = json.loads((sess / "trials.jsonl").read_text().splitlines()[n - 1])
    b = SimBackend(str(POLICY), "cpu", seed=1); b.prepare_game(); g = b.game; s = g.sim; m, d = s.m, s.d
    z = 0.0005
    for slot, typ in zip(t["stack_slots"], t["stack_types"]):          # the trial's stack on the spot
        body = b.body_of[slot]; set_box(m, d, body, SPOT, z + BOX[typ]["height"] / 2, np.radians(-30)); z += BOX[typ]["height"]
    for slot in t["aside_slots"]:
        set_box(m, d, b.body_of[slot], (0.0, 0.30), 0.02, 0.0)
    q0 = g.jm.real_to_sim(real_q[0])                                      # start at the real first reading
    d.qpos[s.qadr] = q0[:5]; d.qpos[s.gadr] = q0[5]
    for j, v in zip(JOINT_NAMES, q0):
        d.ctrl[s.act[j]] = v
    mujoco.mj_forward(m, d); s.sync_plan(); s.wait(0.5)
    sim_q = []
    for a in cmd:
        sim_q.append(b.arm.joints()); b.arm.send(a)
    sim_q = np.array(sim_q); kin = Kinematics()
    tip_r = np.array([kin.tip_pos(q.tolist()) for q in real_q]); tip_s = np.array([kin.tip_pos(q.tolist()) for q in sim_q])
    picked = {bd: g.pose(bd)[0] for bd in b.body_of.values()}
    return t, real_q, cmd, sim_q, tip_r, tip_s, picked, b


def main():
    sess, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1
    t, rq, cmd, sq, tr_, ts_, picked, b = replay(sess, n)
    T = len(cmd); g = rq[:, 5]; closed = int(np.argmax((cmd[:, 5] < g - 5) & (np.arange(T) > 30))) or T // 3
    phases = {"reach (0-5 s)": slice(0, min(T, 150)), "carry (after the close + 1 s)": slice(min(T, closed + 30), T)}
    np.set_printoptions(suppress=True, precision=1, linewidth=160)
    lines = [f"session {Path(sess).name} trial {n}: {T} ticks; jaws closed at {closed / 30:.1f} s",
             "target minus reading, mean per joint (" + ", ".join(JOINT_NAMES) + "):"]
    for name, sl in phases.items():
        lines += [f"  {name}", f"    real {np.mean(cmd[sl] - rq[sl], 0)}", f"    sim  {np.mean(cmd[sl] - sq[sl], 0)}",
                  f"    tip z cm: real {tr_[sl, 2].mean() * 100:.1f}, sim {ts_[sl, 2].mean() * 100:.1f}; "
                  f"tip xy gap real-sim {np.linalg.norm(tr_[sl, :2] - ts_[sl, :2], axis=1).mean() * 100:.1f} cm"]
    held = [bd for bd, p in picked.items() if p[2] > 0.03]
    lines.append(f"sim boxes above 3 cm at the end (held/lifted): {[(bd, g_t) for bd, g_t in ((h, b.game.type_of[h]) for h in held)]}")
    txt = "\n".join(lines); print(txt)
    out = ROOT / "notes" / f"phase8_replay_{Path(sess).name}_t{n:03d}.txt"; out.write_text(txt + "\n")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(7, 1, figsize=(10, 15), sharex=True); x = np.arange(T) / 30
    for i, j in enumerate(JOINT_NAMES):
        ax[i].plot(x, cmd[:, i], "k:", label="target"); ax[i].plot(x, rq[:, i], "C0", label="real"); ax[i].plot(x, sq[:, i], "C1", label="sim")
        ax[i].set_ylabel(j, fontsize=8)
    ax[0].legend(fontsize=8)
    ax[6].plot(x, tr_[:, 2] * 100, "C0", label="real"); ax[6].plot(x, ts_[:, 2] * 100, "C1", label="sim"); ax[6].set_ylabel("tip z cm")
    ax[6].set_xlabel("s"); fig.tight_layout()
    img = ROOT / "notes/img" / f"phase8_replay_{Path(sess).name}_t{n:03d}.png"; fig.savefig(img, dpi=80); print(img)


if __name__ == "__main__":
    main()
