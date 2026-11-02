"""Phase 8: replay the collapse check on every recorded real trial (as phase8_trial.py ran it live) and compare
with the user's call. Label per trial: OK = the new box ended on top and the stack stands (the user's "success"),
NOT = anything else, unless overridden in LABELS (checked by eye on the after images).

    MUJOCO_GL=cgl ~/.venvs/so101_eval/bin/python scripts/phase8_collapse_replay.py [--check perception.stack:collapse_check] [--show]
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from perception.drop import Kinematics  # noqa: E402
from perception.register import refine  # noqa: E402
from perception.supply import BOX  # noqa: E402
from perception.twin import Twin, arm_mask  # noqa: E402
from perception_eval import template_poses  # noqa: E402
from phase8_sim_twin import release_tick  # noqa: E402

JN = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
HOME = [json.loads((Path.home() / ".cache/so101_home/my_follower_arm.json").read_text())[j] for j in JN]
# Label = what the collapse check must answer: OK = the new box lies on top and the stack stands (also when it was the
# wrong box), NOT = anything else (box on the table, stack moved, no box placed). Overrides of the user's success/fail,
# checked by eye on notes/img/phase8_fail_after_*.png, phase8_turn_odd.png, phase8_turn_overrides.png (29 Oct); SKIP = a hand in the after image (the game restarts
# the settle window when a hand is seen, so these never reach the check).
LABELS = {("20261028_201746", 1): "NOT", ("20261028_201746", 2): "NOT", ("20261028_201746", 3): "OK",
          ("20261028_203311", 5): "NOT",   # user "success", but the box slid off the stack onto the table on the retract
          ("20261029_125445", 3): "OK", ("20261029_125445", 4): "OK", ("20261029_153706", 4): "OK",
          ("20261028_203311", 3): "SKIP", ("20261029_121913", 3): "SKIP", ("20261029_153706", 8): "SKIP"}


def frames_of(path):
    c = cv2.VideoCapture(str(path)); out = []
    while True:
        ok, im = c.read()
        if not ok:
            return out
        out.append(im)


def trials():
    for f in sorted((ROOT / "data/real_trials").glob("2026102*/trials.jsonl")):
        sess = f.parent
        if not (sess / "ref.png").exists():
            continue
        rows = [json.loads(x) for x in f.read_text().splitlines()]
        rows = [t for t in rows if t.get("perception_features") and t.get("outcome") not in (None, "no_motion")]
        if rows:
            yield sess, rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", default="perception.stack:collapse_check")
    ap.add_argument("--features", default="perception.stack:features"); ap.add_argument("--show", action="store_true")
    ap.add_argument("--v2", action="store_true", help="perception.stack.turn_check (photometric match + flat-box fit + unexplained blobs)")
    a = ap.parse_args()
    mod, fn = a.check.split(":"); check = getattr(importlib.import_module(mod), fn)
    mod, fn = a.features.split(":"); feats = getattr(importlib.import_module(mod), fn)
    kin = Kinematics(); tw = Twin(); out = []
    for sess, rows in trials():
        ref = cv2.imread(str(sess / "ref.png")); cam, _ = refine(ref, tw, template_poses(tw), HOME); hm = arm_mask(tw, cam, HOME)
        for t in rows:
            n = t["trial"]; before = cv2.imread(str(sess / f"t{n:03d}_before.png")); frames = frames_of(sess / f"t{n:03d}_settle.mp4")
            if before is None or len(frames) < 20 or t["picked_type"] is None or LABELS.get((sess.name, n)) == "SKIP":
                continue
            z_top = 0.0002 + sum(BOX[x]["height"] for x in t["stack_types"])
            tr = np.load(sess / f"t{n:03d}_trace.npy"); rec = frames_of(sess / f"t{n:03d}_front.mp4")
            sp = t.get("scripted_place") or {}
            if sp.get("release_index") is not None:
                kr = len(tr) - sp["ticks"] + sp["release_index"]
            else:
                kr = release_tick(kin, tr[:, :6].astype(float), tr[:, 6:12].astype(float), np.array(t["top_xy"]))
            arm_frames = None
            if rec and kr < len(tr):
                from perception.stack import landed_tick
                k0 = min(landed_tick(kin, tr, kr), len(rec) - 1)
                arm_frames = [(rec[k0], arm_mask(tw, cam, tr[k0, :6].tolist())), (rec[-1], arm_mask(tw, cam, tr[min(len(tr), len(rec)) - 1, :6].tolist()))]
            label_ = LABELS.get((sess.name, n), "OK" if t["outcome"] == "success" else "NOT")
            if a.v2:
                from perception.stack import turn_check
                ok2, why, f = turn_check(cam, ref, before, frames[-1], z_top, t["stack_types"][-1], t["picked_type"], arm_mask=hm,
                                         top_xy0=t["top_xy"])
                col = not ok2
                if f.get("unsure"):                         # the game asks the player: their answer is the label
                    col = label_ == "NOT"; why = why + ["(unsure: asked)"]
            else:
                f = feats(cam, ref, before, frames, z_top, BOX[t["picked_type"]]["height"], arm_mask=hm, new_dims=BOX[t["picked_type"]], arm_frames=arm_frames)
                col, why = check(f)
            label = LABELS.get((sess.name, n), "OK" if t["outcome"] == "success" else "NOT")
            live = "COLLAPSE" if (t["perception_call"] or "").startswith("collapse") else "standing"
            out.append(dict(sess=sess.name, n=n, stack=" ".join(t["stack_types"]), label=label, live=live, call="COLLAPSE" if col else "standing",
                            why=why, failure=t["failure"], note=t["note"], f=f))
            r = out[-1]; bad = (r["call"] == "COLLAPSE") != (label == "NOT")
            extra = (f" fit {f['fit']:.2f} off {f['off_mm']:.0f} blob {f['blob']}/{f['blob_max']:.0f} topfit {f['top_fit']:.2f} "
                     f"cover {f['cover'] if f['cover'] is None else round(f['cover'], 2)} stackchg {f['stack_change']:.2f} stick {f['stick_px']}") if a.v2 else ""
            print(f"{sess.name[-6:]} t{n} {r['stack'][:20]:20s} label {label:3s} live {live:8s} replay {r['call']:8s} {'WRONG' if bad else '     '}{extra} {why}", flush=True)
    fc = [r for r in out if r["label"] == "OK" and r["call"] == "COLLAPSE"]; ms = [r for r in out if r["label"] == "NOT" and r["call"] == "standing"]
    agree = sum(r["call"] == r["live"] for r in out)
    print(f"\n{len(out)} trials: replay agrees with the live call on {agree}; false collapse calls {len(fc)}/{sum(r['label'] == 'OK' for r in out)}, "
          f"missed {len(ms)}/{sum(r['label'] == 'NOT' for r in out)}; asked (unsure) {sum('(unsure: asked)' in r['why'] for r in out)}")
    (ROOT / f"data/real_trials/collapse_replay{'_v2' if a.v2 else ''}.json").write_text(json.dumps(
        [{k: (v if k != "f" else {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv) for kk, vv in v.items()}) for k, v in r.items()} for r in out], default=str))


if __name__ == "__main__":
    main()
