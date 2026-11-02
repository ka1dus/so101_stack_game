"""Phase 8: where did the placed box land on the real stack? From the trial's before/after front images.

Base top box: flat-box fit (perception.stack.fit_box) to the stack silhouette in `before` (vs the session ref), as
`phase8_trial.py --scripted-place` does for its target. New box: fit to the change between `before` and `after`
on the stack face. Prints both poses, the offset of the new box centre from the base centre and the yaw difference.

    ~/.venvs/so101_eval/bin/python scripts/phase8_landed.py data/real_trials/20261029_153706
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from perception.register import refine  # noqa: E402
from perception.stack import change, fit_box, stack_region  # noqa: E402
from perception.supply import BOX  # noqa: E402
from perception.twin import Twin, arm_mask  # noqa: E402
from perception_eval import template_poses  # noqa: E402

HOME = json.loads((Path.home() / ".cache/so101_home/my_follower_arm.json").read_text())
JN = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def main():
    sess = Path(sys.argv[1]); tw = Twin(); home = [HOME[j] for j in JN]
    ref = cv2.imread(str(sess / "ref.png")); cam, _ = refine(ref, tw, template_poses(tw), home)
    hm = arm_mask(tw, cam, home)
    for line in (sess / "trials.jsonl").read_text().splitlines():
        t = json.loads(line); n = t["trial"]
        b, a = cv2.imread(str(sess / f"t{n:03d}_before.png")), cv2.imread(str(sess / f"t{n:03d}_after.png"))
        if a is None or not t["stack_types"]:
            continue
        top = t["stack_types"][-1]; z_top = 0.0002 + sum(BOX[x]["height"] for x in t["stack_types"])
        fg = change(b, ref) & stack_region(cam, arm_mask=hm)
        s0, xy0, y0 = fit_box(cam, fg, z_top - BOX[top]["height"], BOX[top], np.array(t["top_xy"]))
        new = t["picked_type"] or "no3"
        ch = change(a, b) & stack_region(cam, arm_mask=hm)
        s1, xy1, y1 = fit_box(cam, ch, z_top, BOX[new], xy0)
        dy = (np.degrees(y1 - y0) + 90) % 180 - 90
        off = (xy1 - xy0) * 1000
        print(f"t{n} {' '.join(t['stack_types'])} > {new} {t['outcome']:7s}: base fit ({xy0[0]*100:.1f}, {xy0[1]*100:.1f}) cm yaw {np.degrees(y0) % 180:5.1f} (IoU {s0:.2f}), "
              f"perceived top ({t['top_xy'][0]*100:.1f}, {t['top_xy'][1]*100:.1f}) | new box ({xy1[0]*100:.1f}, {xy1[1]*100:.1f}) yaw {np.degrees(y1) % 180:5.1f} (IoU {s1:.2f}) | "
              f"offset ({off[0]:+.0f}, {off[1]:+.0f}) mm, yaw diff {dy:+.0f} deg")


if __name__ == "__main__":
    main()
