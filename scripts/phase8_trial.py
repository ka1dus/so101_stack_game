"""Phase 8: real placement trials (no game yet). Run on the Mac, arm and cameras plugged in, in the eval venv
(~/.venvs/so101_eval: LeRobot 0.6.2 from ~/lerobot, MuJoCo 3.16, feetech-servo-sdk 1.0.0; the conda `lerobot` env's
MuJoCo 3.8 cannot load sim/scene.xml).

Session:
  1. connect; the arm must be at the saved home pose (offers a slow move there)
  2. reference frame: all 8 boxes in the template, stack area empty -> camera refit + supply-slot monitor
  3. per trial: you type the stack (types bottom to top) and the box type the robot should pick. The planner says
     which template slots form the stack and which boxes to set aside so that the ACT pick rule (nearest clear box
     to the stacking spot, Section 3c) picks that type; every such supply is a state of the training distribution
     (a template with boxes taken out). You build it, press Enter, the skill runs from home, the arm goes home,
     3 s settle, then perception gives its call and you give the real outcome (the ground truth).
  Ctrl-C while the arm moves: it holds where it is, then you choose: return home, torque off, or quit.
Logs: data/real_trials/<session>/trials.jsonl, per-trial front/wrist mp4 and joint trace.

    ~/.venvs/so101_eval/bin/python scripts/phase8_trial.py --port /dev/tty.usbmodem5B910311481 --stage 1
    ~/.venvs/so101_eval/bin/python scripts/phase8_trial.py --port ... --no-motion      # everything except motion
    ~/.venvs/so101_eval/bin/python scripts/phase8_trial.py --report                    # results by bucket
"""
import argparse
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import game_states as G  # noqa: E402
from joint_mapping import JOINT_NAMES  # noqa: E402
from perception.drop import Kinematics  # noqa: E402
from perception.register import refine  # noqa: E402
from perception.stack import SPOT, change, collapse_check, features, stack_region, top_xy  # noqa: E402
from perception.supply import BOX, LAYOUT, SupplyMonitor  # noqa: E402
from perception.twin import Twin, arm_mask  # noqa: E402
from perception_eval import template_poses  # noqa: E402
from skill.runner import FPS, RealArm, go_home, load_policy, run_place_skill, undistort_front  # noqa: E402

OUT = ROOT / "data/real_trials"
HOME_FILE = Path.home() / ".cache/so101_home/my_follower_arm.json"
SETTLE_S = 3.0
HOME_TOL = 2.0             # LeRobot units: "at home"
FAILURES = ["missed_grasp", "wrong_box", "drop", "knocked_approach", "knocked_release", "misplaced", "timeout", "other"]
STAGES = {1: "same-type stacks, heights 1-3", 2: "mixed pairings (small on large, then large on small)",
          3: "mixed stacks of height 3-4, some slightly crooked"}


# ---------------------------------------------------------------- planning
def template_boxes():
    C = np.array(G.SUPPLY_RECT); head = np.arctan2(*(C[1] - C[0])[::-1])
    return {s["id"]: G.Box(s["type"], np.array(s["centre_world_m"]), head + (0 if s["long_axis"] == "along edge 1->2" else np.pi / 2),
                           G.MAT_Z + BOX[s["type"]]["height"] / 2) for s in LAYOUT["boxes"]}


def plan(stack_types, pick_type):
    """-> dict(stack_slots bottom->top, aside_slots, pick_slot) with the fewest boxes set aside, or None."""
    T = template_boxes(); order = sorted(T, key=lambda i: np.linalg.norm(T[i].xy - SPOT))
    by_type = {t: [i for i in order if T[i].type == t] for t in BOX}
    best = None
    for combo in itertools.product(*[by_type[t] for t in stack_types]):
        if len(set(combo)) < len(combo):
            continue
        rest = [i for i in order if i not in combo]
        for r in range(len(rest)):
            if best is not None and r > len(best["aside_slots"]):
                break
            for aside in itertools.combinations(rest, r):
                sup = [i for i in rest if i not in aside]
                k = G.choose_pick(G.State(supply=[T[i] for i in sup]))
                if k >= 0 and T[sup[k]].type == pick_type:
                    cand = dict(stack_slots=list(combo), aside_slots=list(aside), pick_slot=sup[k],
                                score=(r, sum(order.index(i) for i in combo)))
                    if best is None or cand["score"] < best["score"]:
                        best = cand
                    break
    return best


# ---------------------------------------------------------------- helpers
def ask(prompt, choices=None):
    while True:
        a = input(prompt).strip().lower()
        if choices is None or a in choices:
            return a
        print(f"  please answer one of: {', '.join(choices)}")


def home_pose():
    h = json.loads(HOME_FILE.read_text()); return np.array([h[j] for j in JOINT_NAMES], float)


def front_bgr(arm):
    return cv2.cvtColor(undistort_front(arm.read()[1]), cv2.COLOR_RGB2BGR)


def write_mp4(path, frames):
    if not frames:
        return
    h, w = frames[0].shape[:2]; v = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for f in frames:
        v.write(f)
    v.release()


def estop(arm, kin, home):
    arm.hold()
    print("\n  STOPPED: the arm is holding where it is.")
    a = ask("  [h] return home slowly, [t] torque off (support the arm first!), [q] quit holding: ", ["h", "t", "q"])
    if a == "h":
        go_home(arm, kin, home, lift=True)
    elif a == "t":
        arm.torque_off(); print("  torque off.")
    return a


# ---------------------------------------------------------------- report
def report():
    rows = [json.loads(l) for f in sorted(OUT.glob("*/trials.jsonl")) for l in f.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("outcome") in ("success", "fail")]
    print(f"{len(rows)} judged trials in {OUT}")
    for key, name in ((lambda r: f"stage {r['stage']}", "stage"), (lambda r: r["picked_type"], "picked type"),
                      (lambda r: f"{r['picked_type']} on {r['stack_types'][-1] if r['stack_types'] else 'base'}", "pairing"),
                      (lambda r: f"height {len(r['stack_types'])}", "stack height")):
        g = defaultdict(list)
        for r in rows:
            g[key(r)].append(r)
        print(f"\nby {name}:")
        for k in sorted(g):
            v = g[k]; ok = sum(r["outcome"] == "success" for r in v)
            fails = defaultdict(int)
            for r in v:
                if r["outcome"] == "fail":
                    fails[r["failure"]] += 1
            print(f"  {k:22s} {ok}/{len(v)} ({100 * ok / len(v):.0f} %)" + (f"  failures {dict(fails)}" if fails else ""))
    calls = [r for r in rows if r.get("perception_call") is not None]
    agree = sum((r["perception_call"] == "standing") == (r["outcome"] == "success") for r in calls)
    print(f"\nperception call agrees with you: {agree}/{len(calls)}")


# ---------------------------------------------------------------- session
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port"); ap.add_argument("--id", default="my_follower_arm")
    ap.add_argument("--policy", default=str(ROOT / "data/policies/act_v4_ft_040000"))
    ap.add_argument("--front", type=int, default=0); ap.add_argument("--wrist", type=int, default=1)
    ap.add_argument("--stage", type=int, choices=STAGES, default=1); ap.add_argument("--device", default="mps")
    ap.add_argument("--no-motion", action="store_true", help="run everything, but the arm holds home (no actions sent)")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report()
    if not args.port:
        sys.exit("--port is required (find it with lerobot-find-port)")
    sess = OUT / time.strftime("%Y%m%d_%H%M%S"); sess.mkdir(parents=True)
    home = home_pose(); kin = Kinematics(); twin = Twin()
    print(f"stage {args.stage}: {STAGES[args.stage]}" + ("   [NO MOTION]" if args.no_motion else ""))
    print("loading the policy ..."); policy = load_policy(args.policy, args.device)
    arm = RealArm(args.port, args.id, args.front, args.wrist); arm.connect()
    try:
        q = arm.joints(); off = np.abs(q - home).max()
        if off > HOME_TOL:
            print(f"the arm is {off:.1f} units from home (joint readings {np.round(q, 1).tolist()})")
            if args.no_motion or ask("move it home slowly now? [y/n]: ", ["y", "n"]) != "y":
                sys.exit("put the arm at home first (scripts/so101_go_home.py)")
            go_home(arm, kin, home, lift=False)
        input("\nREFERENCE: all 8 boxes in the template slots, stack area EMPTY, hands out of view. Press Enter.")
        time.sleep(0.5); ref = front_bgr(arm); cv2.imwrite(str(sess / "ref.png"), ref)
        cam, rinfo = refine(ref, twin, template_poses(twin), home.tolist())
        print(f"  camera refit: moved {rinfo['delta_mm']} mm / {rinfo['delta_deg']} deg (mask cost {rinfo['cost_after']})")
        supply = SupplyMonitor(ref, cam); occ = supply.occupied(ref)
        if not all(occ.values()):
            print(f"  WARNING: slots read as empty in the reference: {[k for k, v in occ.items() if not v]}")
        home_mask = arm_mask(twin, cam, home.tolist()); n = 0
        while True:
            print("\n" + "-" * 70)
            spec = input(f"trial {n + 1}: stack types bottom->top and the pick type, e.g. 'no3 no3 > no3' (q = quit): ").strip()
            if spec.lower() in ("q", "quit"):
                break
            try:
                left, right = spec.split(">"); stack_types = left.split(); pick_type = right.strip()
                assert pick_type in BOX and all(t in BOX for t in stack_types)
            except (ValueError, AssertionError):
                print("  format: types from no3/no35/no10, e.g. 'no35 > no10' or '> no3' for the empty base"); continue
            if not stack_types:
                print("  NOTE: training never placed onto the empty base (the human always starts); height 0 is out of distribution.")
            p = plan(stack_types, pick_type)
            if p is None:
                print("  no supply arrangement makes the pick rule choose that type with this stack; try another"); continue
            print(f"  stack (bottom->top) from slots {p['stack_slots']}  |  set aside (off the table): {p['aside_slots'] or 'none'}"
                  f"  |  the robot should pick slot {p['pick_slot']} ({pick_type})")
            print("  leave every other box in its template slot; build the stack centred on the X; hands out of view.")
            if ask("  ready? [Enter = go, s = skip]: ", ["", "s"]) == "s":
                continue
            before = front_bgr(arm); cv2.imwrite(str(sess / f"t{n + 1:03d}_before.png"), before)
            occ = supply.occupied(before)
            want = {s["id"]: s["id"] not in p["stack_slots"] + p["aside_slots"] for s in LAYOUT["boxes"]}
            wrong = [k for k in want if occ[k] != want[k]]
            if wrong and ask(f"  slots {wrong} don't look as planned (full/empty). continue anyway? [y/n]: ", ["y", "n"]) != "y":
                continue
            z_top = 0.0002 + sum(BOX[t]["height"] for t in stack_types)
            fg = change(before, ref) & stack_region(cam, arm_mask=home_mask)
            xy = top_xy(cam, fg, z_top) if stack_types else None; xy = SPOT if xy is None else xy
            print(f"  stack top seen at ({xy[0] * 100:.1f}, {xy[1] * 100:.1f}) cm, face {z_top * 100:.1f} cm up")
            print("  starting in 3 s: keep hands clear. Ctrl-C = stop."); time.sleep(3)
            rec = []; stopped = None
            try:
                res = run_place_skill(arm, policy, kin, xy, z_top, move=not args.no_motion, record=rec)
                if not args.no_motion:
                    go_home(arm, kin, home, lift=not res["event"] == "released")
            except KeyboardInterrupt:
                stopped = estop(arm, kin, home); res = dict(event="stopped by user", duration_s=None, loop_hz=None, trace=np.zeros((0, 13)))
                if stopped in ("t", "q"):
                    write_mp4(sess / f"t{n + 1:03d}_front.mp4", [r[0] for r in rec]); break
            frames = []
            for _ in range(int(SETTLE_S * FPS)):
                t0 = time.perf_counter(); frames.append(front_bgr(arm)); time.sleep(max(0, 1 / FPS - (time.perf_counter() - t0)))
            after = frames[-1]; cv2.imwrite(str(sess / f"t{n + 1:03d}_after.png"), after)
            fb, fa = supply.features(before), supply.features(after)
            emptied = [k for k in fb if fb[k]["dark"] < 0.5 <= fa[k]["dark"]]
            slot = emptied[0] if len(emptied) == 1 else None
            ptype = next((s["type"] for s in LAYOUT["boxes"] if s["id"] == slot), None)
            call = None
            if slot is not None and res["event"] != "stopped by user":
                arm_frames = None
                if res.get("release_tick") is not None and rec:
                    tr = res["trace"]; k0 = res["release_tick"]
                    arm_frames = [(rec[k0][0], arm_mask(twin, cam, tr[k0, :6].tolist())), (rec[-1][0], arm_mask(twin, cam, tr[-1, :6].tolist()))]
                f = features(cam, ref, before, frames, z_top, BOX[ptype]["height"], arm_mask=home_mask, new_dims=BOX[ptype], arm_frames=arm_frames)
                col, why = collapse_check(f); call = ("collapse: " + ", ".join(why)) if col else "standing"
            print(f"  skill: {res['event']} after {res['duration_s']} s at {res['loop_hz']} Hz | picked slot {slot} ({ptype}) | perception: {call}")
            if args.no_motion:
                outcome, failure = "no_motion", None
            else:
                ok = ask("  YOUR CALL: box on top and the stack standing 3 s after release? [y/n]: ", ["y", "n"])
                outcome = "success" if ok == "y" else "fail"; failure = None
                if outcome == "fail":
                    for i, fl in enumerate(FAILURES):
                        print(f"    {i}: {fl}")
                    failure = FAILURES[int(ask("  failure type number: ", [str(i) for i in range(len(FAILURES))]))]
            note = input("  note (optional): ").strip()
            n += 1
            row = dict(trial=n, time=time.strftime("%Y-%m-%d %H:%M:%S"), stage=args.stage, policy=Path(args.policy).name,
                       stack_types=stack_types, stack_slots=p["stack_slots"], aside_slots=p["aside_slots"],
                       expected_pick_slot=p["pick_slot"], picked_slot=slot, picked_type=ptype or pick_type,
                       wrong_box=slot is not None and slot != p["pick_slot"], top_xy=np.round(xy, 4).tolist(),
                       skill_event=res["event"], skill_s=res["duration_s"], loop_hz=res["loop_hz"],
                       compute_ms_p50=res.get("compute_ms_p50"), compute_ms_p95=res.get("compute_ms_p95"),
                       perception_call=call, outcome=outcome, failure=failure, note=note, camera_refit=rinfo)
            with open(sess / "trials.jsonl", "a") as fh:
                fh.write(json.dumps(row) + "\n")
            write_mp4(sess / f"t{n:03d}_front.mp4", [r[0] for r in rec] + frames)
            write_mp4(sess / f"t{n:03d}_wrist.mp4", [r[1] for r in rec])
            np.save(sess / f"t{n:03d}_trace.npy", res["trace"])
            print(f"  logged -> {sess.relative_to(ROOT)}/trials.jsonl. Put the boxes back in their template slots.")
            if stopped == "h":
                print("  (stopped trial logged)")
    finally:
        arm.disconnect()
        print(f"session folder: {sess}")


if __name__ == "__main__":
    main()
