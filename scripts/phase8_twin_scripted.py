"""Phase 8: sim twin of real trials with the policy pick + scripted place (skill/scripted_place.py).
Per run: skill event, place event, box-in-gripper estimate, release error (FK), landed offset of the box centre from the
top centre after 3 s (true sim pose), box on the stack, how far any stack box moved.

    MUJOCO_GL=cgl python scripts/phase8_twin_scripted.py <device> <session:trial> ... [--runs N]
"""
import sys, json; sys.path[:0] = [str(__import__("pathlib").Path(__file__).resolve().parents[1]), str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts")]
from pathlib import Path
import numpy as np, mujoco
from game.backends import SimBackend
from perception.drop import Kinematics
from perception.supply import BOX
from phase8_replay import set_box
from skill.grip_floor import GripFloor
from skill.runner import load_policy, run_place_skill, safety_check
from skill.scripted_place import ScriptedPlace
import servo_model
from joint_mapping import JOINT_NAMES
ROOT = Path(__file__).resolve().parents[1]; pol = str(ROOT / "data/policies/act_wcal_060000")
kin = Kinematics(); policy = load_policy(pol, sys.argv[1] if len(sys.argv) > 1 else "mps"); params = servo_model.load(str(ROOT / "sim/assets/servo_fit.yaml"))
sp = ScriptedPlace(kin)
args = sys.argv[2:]; runs = 2
if "--runs" in args:
    i = args.index("--runs"); runs = int(args[i + 1]); args = args[:i] + args[i + 2:]
jobs = [(a.split(":")[0], int(a.split(":")[1])) for a in (args or ["20261029_125445:1", "20261029_125445:5"])]
for sess, n in jobs:
    d0 = ROOT / "data/real_trials" / sess; t = json.loads((d0 / "trials.jsonl").read_text().splitlines()[n - 1]); top = np.array(t["top_xy"])
    rq = np.load(d0 / f"t{n:03d}_trace.npy")[:, :6].astype(float); yaw = np.radians(-30)
    for r in range(runs):
        b = SimBackend(pol, "cpu", seed=20000 + r); b.prepare_game(); g = b.game; s = g.sim; m, d = s.m, s.d; servo_model.apply_params(m, params)
        z = 0.0005; stack_b = []
        for slot, typ in zip(t["stack_slots"], t["stack_types"]):
            set_box(m, d, b.body_of[slot], top, z + BOX[typ]["height"] / 2, yaw); z += BOX[typ]["height"]; stack_b.append(b.body_of[slot])
        for slot in t["aside_slots"]:
            set_box(m, d, b.body_of[slot], (0.0, 0.30), 0.02, 0.0)
        q0 = g.jm.real_to_sim(rq[0]); d.qpos[s.qadr] = q0[:5]; d.qpos[s.gadr] = q0[5]
        for j, v in zip(JOINT_NAMES, q0): d.ctrl[s.act[j]] = v
        mujoco.mj_forward(m, d); s.sync_plan(); s.wait(1.0)
        st0 = {bd: g.pose(bd)[0].copy() for bd in stack_b}
        res = run_place_skill(b.arm, policy, kin, top, z, act_fn=GripFloor(28), handover_lift=0.04)
        info = {}
        if res["event"] == "handover":
            info = sp.run(b.arm, res["trace"], top, z, yaw, log=lambda *a: None, safety=lambda c: safety_check(kin, c))
        s.wait(3.0)
        pid = b.body_of[info["slot"]] if info.get("slot") else None
        on = pid is not None and np.linalg.norm(g.pose(pid)[0][:2] - top) < 0.03 and g.pose(pid)[0][2] > z
        land = (g.pose(pid)[0][:2] - top) * 1000 if pid is not None else np.full(2, np.nan)
        moved = max((np.linalg.norm(g.pose(bd)[0] - st0[bd]) for bd in stack_b), default=0) * 1000
        print(f"RES {sess} t{n} {t['stack_types']} run {r}: skill {res['event']} {res['duration_s']} s | place {info.get('event')} slot {info.get('slot')} "
              f"{info.get('type')} b_in_g {info.get('b_in_g_mm')} ik {info.get('ik_err_mm')} | release err {info.get('release_err_mm')} clear {info.get('release_clear_mm')} mm | landed ({land[0]:+.1f}, {land[1]:+.1f}) mm | "
              f"box on the stack {on}, stack moved {moved:.0f} mm", flush=True)
