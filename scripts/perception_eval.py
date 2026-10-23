"""Phase 7: score the perception functions on the sim test set (scripts/perception_testset.py).

    ~/.venvs/so101_stack_sim/bin/python scripts/perception_eval.py --data data/perception/sim_v1 --check supply
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from perception.camera import Camera  # noqa: E402
from perception.supply import BOX, LAYOUT, SUPPLY, SupplyMonitor  # noqa: E402


def template_poses(twin):
    """Twin body -> pose for the template layout (the reference frame's boxes)."""
    C = np.array(SUPPLY["corners"]); head = np.arctan2(*(C[1] - C[0])[::-1])
    free = {t: [b for b in twin.boxes if b.endswith("_" + t)] for t in ("no3", "no35", "no10")}; out = {}
    for s in LAYOUT["boxes"]:
        yaw = head + (0 if s["long_axis"] == "along edge 1->2" else np.pi / 2)
        out[free[s["type"]].pop(0)] = (*s["centre_world_m"], 0.0005 + BOX[s["type"]]["height"] / 2, yaw)
    return out


def _refine_one(p):
    import yaml
    from joint_mapping import JOINT_NAMES
    from perception.register import refine
    from perception.twin import Twin
    global _TW
    if "_TW" not in globals():
        _TW = Twin()
    meas = yaml.safe_load((ROOT / "inputs/measurements.yaml").read_text())
    cam, info = refine(cv2.imread(str(p / "ref.png")), _TW, template_poses(_TW), [meas["arm"]["home_pose"][j] for j in JOINT_NAMES])
    (p / "cam_refined.json").write_text(json.dumps(dict(pos=cam.pos.tolist(), xyaxes=np.r_[cam.R[0], cam.R[1]].tolist(), f=cam.f, info=info)))


def refined_camera(p):
    c = json.loads((p / "cam_refined.json").read_text())
    return Camera(pos=c["pos"], xyaxes=c["xyaxes"], f=c["f"])


def refine_all(dirs, workers):
    todo = [p for p in dirs if not (p / "cam_refined.json").exists()]
    if todo:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(_refine_one, todo)):
                if i % 50 == 0:
                    print(f"refined {i}/{len(todo)}", flush=True)


UNSURE = (0.3, 0.7)       # dark share between these: the controller asks on screen instead of guessing (D1 fallback)


def eval_supply(dirs):
    """Per slot and frame (before/after): predicted occupancy vs truth. Also the turn-level answer D1 needs:
    exactly which slot was emptied between before and after."""
    conf = Counter(); feats = []; turns = Counter(); errors = []
    for p in dirs:
        t = json.loads((p / "truth.json").read_text()); ref = cv2.imread(str(p / "ref.png"))
        mon = SupplyMonitor(ref, cam=refined_camera(p) if REFINE else None)
        sup_before = set(t["supply_before"]); taken = t["event"].get("slot")
        truth = {"before": sup_before, "after": sup_before - ({taken} if taken else set())}
        pred = {}; unsure = False
        for fr in ("before", "after"):
            f = mon.features(cv2.imread(str(p / f"{fr}.png"))); occ = {k: v["dark"] < 0.5 for k, v in f.items()}
            pred[fr] = {k for k, v in occ.items() if v}
            unsure |= any(UNSURE[0] < v["dark"] < UNSURE[1] for v in f.values())
            for k, v in f.items():
                y = k in truth[fr]; feats.append((y, v["dark"], v["diff"]))
                conf[(y, occ[k])] += 1
                if y != occ[k]:
                    errors.append((p.name, fr, k, round(v["dark"], 3)))
        emptied = pred["before"] - pred["after"]
        want = {taken} if taken else set()
        right = emptied == want and not (pred["after"] - pred["before"])
        turns["correct" if right else "wrong"] += 1
        plain = t["randomization"]["box_plain"]
        turns[("plain " if plain else "printed ") + ("unsure" if unsure else "sure, correct" if right else "sure, WRONG")] += 1
    f = np.array([(y, dk) for y, dk, _ in feats])
    occ, emp = f[f[:, 0] == 1, 1], f[f[:, 0] == 0, 1]
    lines = [f"samples {len(dirs)}; slot-frames {len(feats)}",
             f"slot occupancy: TP {conf[(True, True)]}  FN {conf[(True, False)]}  TN {conf[(False, False)]}  FP {conf[(False, True)]}",
             f"dark share, occupied slots: max {occ.max():.3f}  p99 {np.percentile(occ, 99):.3f}; empty slots: min {emp.min():.3f}  p1 {np.percentile(emp, 1):.3f}",
             f"turn-level 'which slot was taken': {turns['correct']}/{turns['correct'] + turns['wrong']} correct",
             f"with an 'ask the player' band {UNSURE} on the dark share: " +
             ", ".join(f"{k} {v}" for k, v in sorted(turns.items()) if k not in ("correct", "wrong"))]
    lines += [f"  error: {e}" for e in errors[:20]]
    return "\n".join(lines)


REFINE = True


def main():
    global REFINE
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/perception/sim_v1")
    ap.add_argument("--no-refine", action="store_true", help="nominal calibration instead of the per-game refit")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check", default="supply"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dirs = sorted(p for p in Path(a.data).iterdir() if (p / "truth.json").exists())
    REFINE = not a.no_refine
    if REFINE:
        refine_all(dirs, a.workers)
    global a_data
    a_data = a.data
    out = {"supply": eval_supply, "collapse": lambda d: eval_collapse(d, a.workers)}[a.check](dirs)
    print(out)
    if a.out:
        Path(a.out).write_text(out + "\n")



def read_video(path):
    cap = cv2.VideoCapture(str(path)); out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


def stack_features(p):
    from perception.stack import features
    t = json.loads((p / "truth.json").read_text()); cam = refined_camera(p)
    h = {k: BOX[v]["height"] for k, v in ((int(a), b) for a, b in t["slot_type"].items())}
    z_top = 0.0002 + sum(h[s] for s in t["stack_before"])
    new_h = h[t["event"]["slot"]] if t["event"].get("slot") else 0.0
    import yaml
    from joint_mapping import JOINT_NAMES
    from perception.twin import Twin, arm_mask
    global _TW
    if "_TW" not in globals():
        _TW = Twin()
    home = [yaml.safe_load((ROOT / "inputs/measurements.yaml").read_text())["arm"]["home_pose"][j] for j in JOINT_NAMES]
    f = features(cam, cv2.imread(str(p / "ref.png")), cv2.imread(str(p / "before.png")), read_video(p / "after.mp4"), z_top, new_h,
                 arm_mask=arm_mask(_TW, cam, home), new_dims=BOX[t["slot_type"][str(t["event"]["slot"])]] if t["event"].get("slot") else None)
    L = t["labels"]
    new = t["event"].get("slot")
    new_fell = bool(new) and t["poses_after"][str(new)]["xyz"][2] < 0.0002 + BOX[t["slot_type"][str(new)]]["length"] / 2 + 0.004 \
        and not L["new_on_top"]                                          # on the table, flat or on its edge, or gone
    game = bool(L["boxes_on_table"]) or new_fell or (L["top_moved_after_landing_mm"] or 0) > 15 or L["max_old_move_mm"] > 15
    return dict(f, name=p.name, kind=t["event"]["kind"], height=len(t["stack_before"]), collapse=L["collapse"], game_collapse=game,
                top_moved=L["top_moved_after_landing_mm"], old_move=L["max_old_move_mm"],
                shadows=t["randomization"]["shadows"],
                true_top_xy=t["poses_after"][str(t["stack_before"][-1])]["xyz"][:2] if t["stack_before"] else None)


def eval_collapse(dirs, workers=8, threshold_mm=15.0):
    """collapse_check() against the truth. Truth collapse = the Judge rule (old stack moved > 5 mm / tilted > 5 deg,
    or the new box not on top) or the top box moved > threshold_mm after landing (CLAUDE.md: 'stack top moves >
    threshold')."""
    import multiprocessing as mp
    from perception.stack import collapse_check
    with mp.get_context("spawn").Pool(workers) as pool:
        rows = pool.map(stack_features, dirs)
    conf = Counter(); by = Counter(); lines = []; mist = []
    for r in rows:
        t = json.loads((Path(a_data) / r["name"] / "truth.json").read_text())
        truth = r["game_collapse"]
        pred, why = collapse_check(r)
        conf[(truth, pred)] += 1; by[(r["kind"], truth, pred)] += 1
        if truth != pred:
            mist.append((r["name"], r["kind"], f"h{r['height']}", "truth" if truth else "false call", why,
                         dict(shift=r["shift_mm"], score=r["shift_score"], outside=r["outside"], on_top_iou=r["on_top_iou"]),
                         f"top_moved {r['top_moved']} old_move {r['old_move']}"))
    lines += [f"samples {len(rows)}; truth collapse (game rule) = a stack box on the table (the new box included), or the "
              f"top moved > {threshold_mm:.0f} mm",
              f"collapse detected {conf[(True, True)]}/{conf[(True, True)] + conf[(True, False)]}; "
              f"false collapse calls {conf[(False, True)]}/{conf[(False, True)] + conf[(False, False)]}",
              "by event (kind, truth, predicted): " + ", ".join(f"{k}: {v}" for k, v in sorted(by.items()))]
    lines += [f"  miss: {m}" for m in mist]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
