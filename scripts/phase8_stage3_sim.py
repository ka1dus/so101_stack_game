"""Phase 8: stage-3 setups in sim before the real trials: policy pick + scripted place onto 3-4 box mixed stacks, built
as phase8_trial.py plans them (stack slots from the template, crooked by +-OFF mm / +-YAW deg per level), from home.
Reports placed / landed offset / stack moved / any arm-backdrop contact.

    MUJOCO_GL=cgl python scripts/phase8_stage3_sim.py <device> "no3 no35 no35 > no3" ... [--runs N]
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import phase8_trial as T  # noqa: E402
import servo_model  # noqa: E402
from game.backends import SimBackend  # noqa: E402
from joint_mapping import JOINT_NAMES  # noqa: E402
from perception.drop import Kinematics  # noqa: E402
from perception.stack import SPOT  # noqa: E402
from perception.supply import BOX  # noqa: E402
from phase8_replay import set_box  # noqa: E402
from skill.grip_floor import GripFloor  # noqa: E402
from skill.runner import load_policy, run_place_skill, safety_check  # noqa: E402
from skill.scripted_place import ScriptedPlace  # noqa: E402

OFF, YAW = 0.006, 8.0


def main():
    args = sys.argv[2:]; runs = 2
    if "--runs" in args:
        i = args.index("--runs"); runs = int(args[i + 1]); args = args[:i] + args[i + 2:]
    pol = str(ROOT / "data/policies/act_wcal_060000"); kin = Kinematics(); policy = load_policy(pol, sys.argv[1])
    params = servo_model.load(str(ROOT / "sim/assets/servo_fit.yaml")); sp = ScriptedPlace(kin)
    home = np.array([json.loads((Path.home() / ".cache/so101_home/my_follower_arm.json").read_text())[j] for j in JOINT_NAMES])
    for spec in args:
        left, right = spec.split(">"); types = left.split(); pick = right.strip(); p = T.plan(types, pick)
        for r in range(runs):
            rng = np.random.default_rng(100 + r)
            b = SimBackend(pol, "cpu", seed=30000 + r); b.prepare_game(); g = b.game; s = g.sim; m, d = s.m, s.d
            servo_model.apply_params(m, params)
            z = 0.0005; yaw0 = np.radians(rng.uniform(40, 70)); stack_b = []; xy = np.array(SPOT, float)
            for slot, typ in zip(p["stack_slots"], types):
                xy_l = SPOT + rng.uniform(-OFF, OFF, 2); yaw_l = yaw0 + np.radians(rng.uniform(-YAW, YAW))
                set_box(m, d, b.body_of[slot], xy_l, z + BOX[typ]["height"] / 2, yaw_l); z += BOX[typ]["height"]
                stack_b.append(b.body_of[slot]); xy, yaw = xy_l, yaw_l
            q0 = g.jm.real_to_sim(home); d.qpos[s.qadr] = q0[:5]; d.qpos[s.gadr] = q0[5]
            for j, v in zip(JOINT_NAMES, q0):
                d.ctrl[s.act[j]] = v
            mujoco.mj_forward(m, d); s.sync_plan(); s.wait(1.5)
            st0 = {bd: g.pose(bd)[0].copy() for bd in stack_b}
            bd_g = m.geom("backdrop").id; arm_root = m.body_rootid[m.body("base").id]; hits = [0]

            def chk(*_):
                for i in range(d.ncon):
                    c = d.contact[i]
                    if bd_g in (c.geom1, c.geom2) and arm_root in (m.body_rootid[m.geom_bodyid[c.geom1]], m.body_rootid[m.geom_bodyid[c.geom2]]):
                        hits[0] += 1; return
            res = run_place_skill(b.arm, policy, kin, xy, z, act_fn=GripFloor(28), handover_lift=0.04, on_tick=lambda f, q: chk())
            info = {}
            if res["event"] == "handover":
                info = sp.run(b.arm, res["trace"], xy, z, yaw, log=lambda *a: None, safety=lambda c: safety_check(kin, c), on_tick=chk)
            s.wait(3.0)
            pid = b.body_of.get(info.get("slot")) if info.get("slot") else None
            pp = g.pose(pid)[0] if pid else None
            on = pp is not None and np.linalg.norm(pp[:2] - xy) < 0.03 and pp[2] > z
            land = (pp[:2] - xy) * 1000 if pp is not None else np.full(2, np.nan)
            moved = max(np.linalg.norm(g.pose(bd)[0] - st0[bd]) for bd in stack_b) * 1000
            print(f"RES {spec} run {r}: skill {res['event']} | place {info.get('event')} {info.get('type')} from slot {info.get('slot')} "
                  f"(planned {p['pick_slot']}) | landed ({land[0]:+.1f}, {land[1]:+.1f}) mm | on the stack {on}, stack moved {moved:.0f} mm, "
                  f"arm-backdrop contact ticks {hits[0]}", flush=True)


if __name__ == "__main__":
    main()
