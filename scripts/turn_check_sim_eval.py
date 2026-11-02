"""turn_check() on the Phase 7 sim perception set (data/perception/sim_v1): every turn that places a box (place,
push, nudge). Truth OK = the new box ends on top and no game collapse (no stack box on the table, nothing moved > 15 mm),
the same rule as scripts/perception_eval.py eval_collapse.

    MUJOCO_GL=cgl ~/.venvs/so101_eval/bin/python scripts/turn_check_sim_eval.py [--workers 8]
"""
import argparse
import json
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def one(p):
    import yaml
    from joint_mapping import JOINT_NAMES
    from perception.stack import turn_check
    from perception.supply import BOX
    from perception.twin import Twin, arm_mask
    from perception_eval import read_video, refined_camera, stack_features
    t = json.loads((p / "truth.json").read_text())
    if not t["event"].get("slot") or not t["stack_before"]:
        return None
    r = stack_features(p)                                  # truth (game rule) as in eval_collapse
    truth_ok = (not r["game_collapse"]) and t["labels"]["new_on_top"]
    cam = refined_camera(p); tw = Twin()
    home = [yaml.safe_load((ROOT / "inputs/measurements.yaml").read_text())["arm"]["home_pose"][j] for j in JOINT_NAMES]
    st = t["slot_type"]; z_top = 0.0002 + sum(BOX[st[str(s)]]["height"] for s in t["stack_before"])
    after = read_video(p / "after.mp4")[-1]
    # the game knows the old top (previous turn's result / wrist look): its true pose (+-3 mm) when the old stack did not
    # move (then after = before); otherwise the image fit (the eval can only be harder than the game there)
    import numpy as np
    known = (r["old_move"] or 0) < 3.0
    top0 = (np.array(t["poses_after"][str(t["stack_before"][-1])]["xyz"][:2]) + np.random.default_rng(len(p.name)).normal(0, 0.003, 2)) if known else None
    ok, why, f = turn_check(cam, cv2.imread(str(p / "ref.png")), cv2.imread(str(p / "before.png")), after, z_top,
                            st[str(t["stack_before"][-1])], st[str(t["event"]["slot"])], arm_mask=arm_mask(tw, cam, home),
                            top_xy0=top0, top_known=known)
    L = t["labels"]; truth_ok2 = bool(L["new_on_top"]) and not L["boxes_on_table"] and (L["max_old_move_mm"] or 0) <= 15
    asked = bool(f.get("unsure"))
    if asked:                                              # the game asks the player (sim: answered truthfully)
        ok = bool(L["new_on_top"])
    return dict(name=p.name, kind=t["event"]["kind"], h=len(t["stack_before"]), truth_ok=bool(truth_ok), truth_ok2=truth_ok2, ok=bool(ok), asked=asked, why=why,
                f={k: f[k] for k in ("fit", "off_mm", "blob", "cover", "top_fit", "stack_change", "stick_px")}, top_moved=r["top_moved"], old_move=r["old_move"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/perception/sim_v1"); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); dirs = sorted(p for p in (ROOT / a.data).iterdir() if (p / "truth.json").exists())
    with mp.get_context("spawn").Pool(a.workers) as pool:
        rows = [r for r in pool.map(one, dirs) if r]
    c = Counter((r["truth_ok"], r["ok"]) for r in rows); by = Counter((r["kind"], r["truth_ok"], r["ok"]) for r in rows)
    print(f"{len(rows)} placing turns: false collapse calls (truth OK, called not) {c[(True, False)]}/{c[(True, False)] + c[(True, True)]}; "
          f"missed (truth not OK, called OK) {c[(False, True)]}/{c[(False, True)] + c[(False, False)]}")
    print("by (kind, truth OK, called OK):", dict(sorted(by.items())))
    c2 = Counter((r["truth_ok2"], r["ok"]) for r in rows)
    print(f"truth without the new box's own slide (on top, nothing on the table, old stack moved <= 15 mm): false calls "
          f"{c2[(True, False)]}/{c2[(True, False)] + c2[(True, True)]}; missed {c2[(False, True)]}/{c2[(False, True)] + c2[(False, False)]}; "
          f"asked (unsure) {sum(r['asked'] for r in rows)}")
    for r in rows:
        if r["truth_ok"] != r["ok"]:
            print(f"  {'FALSE CALL' if r['truth_ok'] else 'MISS'} {r['name']} {r['kind']} h{r['h']} {r['why']} {r['f']} top_moved {r['top_moved']} old_move {r['old_move']}")
    (ROOT / "data/perception/turn_check_sim_v1.json").write_text(json.dumps(rows, default=str))


if __name__ == "__main__":
    main()
