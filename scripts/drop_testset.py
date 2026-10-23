"""Phase 7: sim test set for drop_check(): 30 Hz traces of scripted pick-and-place turns with injected drops.

No rendering. The scripted expert (expert.py, clean v2 kind) runs on game-rule states; during the carry one of these
is injected at a random tick:
  none   nothing (the box is released on the stack as planned)
  slip   box and jaw friction cut to 0.02 (the box slides out of a closed grip)
  knock  a 0.1 s sideways/downward shove on the box (4-14 N)
  open   the gripper is commanded open before the stack
Recorded per tick, in LeRobot units like the real arm reports them: joint readings (state), the gripper command;
plus privileged truth: box-gripper contact, box and jaw-tip positions.
Label (offline, from the physics): a drop = the box leaves the jaws (no contact for 3 ticks) anywhere except in the
release zone just above the stack top.

    ~/.venvs/so101_stack_sim/bin/python scripts/drop_testset.py --n 150 --seed 8001 --out data/perception/drop_v1
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import expert as E  # noqa: E402
import game_states as G  # noqa: E402
import grasp_test as GT  # noqa: E402
from expert import home_fn  # noqa: E402
from joint_mapping import JOINT_NAMES  # noqa: E402

INJECT_P = {"none": 0.4, "slip": 0.2, "knock": 0.2, "open": 0.2}
GRIP_BODIES = ("gripper", "moving_jaw_so101_v1")


def episode(sim, ex, rng, out):
    m, d = sim.m, sim.d
    kw = dict(height_p=G.HEIGHT_P_V2, rest=1.0, hard_p=0.6, supply_p=G.SUPPLY_P_EVAL_GAME)
    st, _ = G.sample_state(rng, m, d, home=home_fn(sim), **kw)
    box = st.supply[st.pick]; bid = m.body(box.body).id; dim = G.BOX[box.type]
    box_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
    grip = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in GRIP_BODIES}
    top = st.stack[-1]; tid = m.body(top.body).id
    top_xy = d.xpos[tid][:2].copy(); top_face = float(d.xpos[tid][2] + G.BOX[top.type]["height"] / 2)
    z0 = float(d.xpos[bid][2])
    kind = str(rng.choice(list(INJECT_P), p=list(INJECT_P.values())))
    at = int(rng.integers(3, 55))                         # carry tick of the injection
    fric0 = m.geom_friction.copy()
    st_ = dict(carry=0, injected=None, open=False)
    rows = []
    orig_apply = sim._apply

    def apply_hook():
        orig_apply()
        if st_["open"]:
            d.ctrl[sim.act["gripper"]] = sim.gripper_sim(45.0)
    sim._apply = apply_hook

    def held():
        b = bid
        for c in d.contact[:d.ncon]:
            if (m.geom_bodyid[c.geom1] == b and c.geom2 in grip) or (m.geom_bodyid[c.geom2] == b and c.geom1 in grip):
                return True
        return False

    def record():
        h = held(); bz = float(d.xpos[bid][2])
        carrying = h and bz > z0 + 0.03
        if carrying or st_["carry"]:
            st_["carry"] += 1
        if st_["injected"] is None and st_["carry"] >= at and kind != "none":
            st_["injected"] = len(rows)
            if kind == "slip":
                for g in box_geoms + list(grip):
                    m.geom_friction[g] = [0.02, 0.0, 0.0]
            elif kind == "knock":
                a = rng.uniform(0, 2 * np.pi); f = rng.uniform(4, 14)
                d.xfrc_applied[bid, :3] = [f * np.cos(a), f * np.sin(a), -rng.uniform(0, 4)]
                st_["knock_until"] = len(rows) + 3
            elif kind == "open":
                st_["open"] = True
        if st_.get("knock_until") == len(rows):
            d.xfrc_applied[bid] = 0
        q_real = sim.jm.sim_to_real(np.r_[d.qpos[sim.qadr], d.qpos[sim.gadr]])
        cmd = sim.jm.sim_to_real(np.r_[[d.ctrl[sim.act[j]] for j in JOINT_NAMES[:5]], d.ctrl[sim.act["gripper"]]])
        R = d.xmat[sim.gb].reshape(3, 3); tip = d.xpos[sim.gb] + R @ np.array([sim.xin, 0.0, sim.ztip])
        rows.append(np.r_[q_real, cmd[5], float(h), d.xpos[bid], tip])

    try:
        r = ex.run(st, record=record, kind="clean")
        sim.on_tick = None
        for _ in range(45):                               # keep watching 1.5 s after the expert stops (or aborts)
            sim.wait(1 / 30); record()
    finally:
        sim._apply = orig_apply; m.geom_friction[:] = fric0; d.xfrc_applied[:] = 0
    A = np.array(rows, dtype=np.float32)
    # label: first lasting separation after the carry started, outside the release zone -> drop
    hold = A[:, 7] > 0.5; drop_tick = None; release_tick = None
    started = np.nonzero(hold & (A[:, 10] > z0 + 0.03))[0]
    if len(started):
        for k in range(started[0], len(A) - 3):
            if not hold[k:k + 3].any():
                bxy, bz = A[k, 8:10], A[k, 10] - dim["height"] / 2
                in_zone = np.linalg.norm(bxy - top_xy) < 0.03 and top_face - 0.003 < bz < top_face + 0.025
                if in_zone:
                    release_tick = k
                else:
                    drop_tick = k
                break
    meta = dict(kind=kind, inject_tick=st_["injected"], drop_tick=drop_tick, release_tick=release_tick,
                drop=drop_tick is not None, type=box.type, height=len(st.stack), top_xy=top_xy.tolist(), top_face=top_face,
                pick_z=z0, expert=r.get("reason"), grasped=len(started) > 0)
    np.save(out / "trace.npy", A); (out / "meta.json").write_text(json.dumps(meta))
    return meta


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=150); ap.add_argument("--seed", type=int, default=8001)
    ap.add_argument("--out", default="data/perception/drop_v1")
    a = ap.parse_args()
    sim = GT.Sim(); rng = np.random.default_rng(a.seed); ex = E.Expert(sim, rng)
    for i in range(a.n):
        out = Path(a.out) / f"d{a.seed}_{i:04d}"; out.mkdir(parents=True, exist_ok=True)
        mt = episode(sim, ex, rng, out)
        print(f"{out.name}: {mt['type']:4s} h{mt['height']} {mt['kind']:5s} drop={mt['drop']} drop_tick={mt['drop_tick']} "
              f"release_tick={mt['release_tick']} expert={mt['expert']}", flush=True)


if __name__ == "__main__":
    main()
