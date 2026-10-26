"""Phase 7: the classical perception on the real recordings of 2026-10-26 (current setup: expanded mat, reach tape,
trimmed backdrop). Arm at home throughout; frames undistorted as the controller will do.
  1. camera refit (perception/register.py) on both reference photos; agreement between the two refits
  2. supply slots (D1): all 8 full in ref_supply_full, all empty in ref_empty; per clip, slots at start vs end
  3. collapse (D2): each clip as one turn (before = 0.5 s, or just ahead of the first hand frame; settle window = 3 s after the human's turn ended, or the
     clip's last 3 s); static clips also with sliding 3 s windows, every 1 s (false-call test)
  4. hand (D3): HandMonitor on every frame, compared with the hand times in notes/phase7_clips.yaml
Box types come from notes/phase7_clips.yaml (guesses stripped of "?", unknown = no35). Report:
notes/phase7_real_clips.txt; overlays and per-clip strips in notes/img/phase7_real_*.png

    MUJOCO_GL=cgl ~/.venvs/so101_stack_sim/bin/python scripts/phase7_real_clips.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from joint_mapping import JOINT_NAMES  # noqa: E402
from perception.drop import Kinematics  # noqa: E402
from perception.hand import HandMonitor  # noqa: E402
from perception.register import refine  # noqa: E402
from perception.stack import AREA, SUPPLY, collapse_check, features  # noqa: E402
from perception.supply import BOX, SupplyMonitor  # noqa: E402
from perception.twin import Twin, arm_mask  # noqa: E402
from perception_eval import template_poses  # noqa: E402
from phase7_real_refit import undistort_img  # noqa: E402

MEAS = yaml.safe_load((ROOT / "inputs/measurements.yaml").read_text())
HOME = [MEAS["arm"]["home_pose"][j] for j in JOINT_NAMES]
CLIPS = yaml.safe_load((ROOT / "notes/phase7_clips.yaml").read_text())
PH, VID = ROOT / "inputs/photos/phase7", ROOT / "inputs/videos/phase7"
FPS, SETTLE = 30, 90
KEYPRESS_S = 0.5           # the human's keypress, taken as this long after the hand left (clips.yaml times)
HAND_TOL_S = 1.0           # clips.yaml times are read off 1-2 s contact sheets


def kind(t):
    t = t.rstrip("?"); return t if t in BOX else "no35"


def z_of(types):
    return 0.0002 + sum(BOX[kind(t)]["height"] for t in types)


def read(name):
    c = cv2.VideoCapture(str(VID / f"{name}.mp4")); out = []
    while True:
        ok, im = c.read()
        if not ok:
            return out
        out.append(undistort_img(im))


def main():
    L = []; P = lambda s="": (print(s, flush=True), L.append(s))
    twin = Twin(); ref = undistort_img(cv2.imread(str(PH / "front_cam_ref_supply_full.png")))
    empty = undistort_img(cv2.imread(str(PH / "front_cam_ref_empty.png")))
    # ---- 1. camera refit
    cam, info = refine(ref, twin, template_poses(twin), HOME)
    cam_e, info_e = refine(empty, twin, {}, HOME)
    P(f"refit on ref_supply_full: moved {info['delta_mm']} mm / {info['delta_deg']} deg; mask cost {info['cost_before']} -> {info['cost_after']}")
    P(f"refit on ref_empty:       moved {info_e['delta_mm']} mm / {info_e['delta_deg']} deg; mask cost {info_e['cost_before']} -> {info_e['cost_after']}")
    pts = np.vstack([np.c_[AREA, np.zeros(4)], np.c_[SUPPLY, np.zeros(4)]])
    P(f"  the two refits project the mat and stack-area corners {np.linalg.norm(cam.project(pts) - cam_e.project(pts), axis=1).max():.1f} px apart (max)")
    ov = ref.copy()
    for Q in (AREA, SUPPLY):
        cv2.polylines(ov, [cam.project(np.c_[Q, np.zeros(4)]).round().astype(np.int32)], True, (0, 0, 255), 1)
    sup = SupplyMonitor(ref, cam)
    for k, p in sup.polys.items():
        cv2.polylines(ov, [p.round().astype(np.int32)], True, (0, 255, 0), 1)
        cv2.putText(ov, str(k), tuple(p.mean(0).astype(int)), 0, 0.4, (0, 255, 0), 1)
    cv2.imwrite(str(ROOT / "notes/img/phase7_real_refit_new.png"), ov)
    # ---- 2. supply slots on the photos
    for nm, im, want in (("ref_supply_full", ref, True), ("ref_empty", empty, False)):
        f = sup.features(im); occ = {k: v["dark"] < 0.5 for k, v in f.items()}
        P(f"slots in {nm}: {sum(o == want for o in occ.values())}/8 correct; dark share per slot "
          + " ".join(f"{k}:{v['dark']:.2f}" for k, v in sorted(f.items())))
    home_mask = arm_mask(twin, cam, HOME)
    twh, kin = Twin(scale=2), Kinematics()
    P(); P("per clip (collapse call / hand / supply):")
    res = {}
    for name, c in CLIPS.items():
        fr = read(name); n = len(fr)
        # ---- 4. hand
        mon = HandMonitor(cam, twh, kin); flags = np.array([mon.tick(f, HOME) for f in fr])
        first = int(np.argmax(flags)) if flags.any() else n            # before = the turn's start, ahead of any hand
        before = fr[max(0, min(15, first - 6))]
        win = c.get("hand_in_s"); hit_s = np.nonzero(flags)[0] / FPS
        if win:
            in_win = (hit_s >= win[0] - HAND_TOL_S) & (hit_s <= win[1] + HAND_TOL_S)
            hand = f"hand {'seen' if in_win.any() else 'MISSED'} ({hit_s[in_win].min():.1f}-{hit_s[in_win].max():.1f} s)" if in_win.any() else "hand MISSED"
            false_hand = hit_s[~in_win]
        elif c["kind"] in ("collapse", "moved"):
            in_win = np.abs(hit_s - c["push_s"]) <= 1.5 + HAND_TOL_S
            hand = f"hand {'seen' if in_win.any() else 'MISSED'}" + (f" ({hit_s[in_win].min():.1f}-{hit_s[in_win].max():.1f} s)" if in_win.any() else "")
            false_hand = hit_s[~in_win]
        else:
            hand = "no hand expected"; false_hand = hit_s
        hand += f"; flagged outside: {len(false_hand)} frames" + (f" ({false_hand.min():.1f}-{false_hand.max():.1f} s)" if len(false_hand) else "")
        # ---- 2. supply, start vs end
        fb, fa = sup.features(before), sup.features(fr[-1])
        emptied = [k for k in fb if fb[k]["dark"] < 0.5 <= fa[k]["dark"]]; filled = [k for k in fb if fa[k]["dark"] < 0.5 <= fb[k]["dark"]]
        unsure = [k for k in fb if 0.3 < fb[k]["dark"] < 0.7 or 0.3 < fa[k]["dark"] < 0.7]
        supply = f"slots emptied {emptied}, filled {filled}" + (f", unsure {unsure}" if unsure else "")
        # ---- 3. collapse
        if c["kind"] == "human":
            stack = c["stack_before"]; z_top = z_of(stack)
            t0 = int((c["hand_in_s"][1] + KEYPRESS_S) * FPS); frames = fr[t0:t0 + SETTLE]
            slot_type = None
            if len(emptied) == 1:
                slot_type = next(s["type"] for s in sup_layout() if s["id"] == emptied[0])
            new_t = slot_type or kind(c["placed"]); dims = BOX[new_t]
            f = features(cam, ref, before, frames, z_top, dims["height"], land_idx=min(18, len(frames) - 1),
                         arm_mask=home_mask, new_dims=dims)
            note = f" (settle window {len(frames) / FPS:.1f} s; new box {new_t}{' from the slot' if slot_type else ' from clips.yaml'})"
        else:
            stack = c["stack"]; z_top = z_of(stack); frames = fr[-SETTLE:]
            f = features(cam, ref, before, frames, z_top, 0.0, arm_mask=home_mask); note = ""
        col, why = collapse_check(f)
        want = c["kind"] in ("collapse", "moved")           # game rule: the top moving > 15 mm is a loss too
        call = ("COLLAPSE: " + ", ".join(why)) if col else "standing"
        ok = col == want
        slide = ""
        if c["kind"] == "static":                          # sliding 3 s windows: any false call over the whole clip?
            calls = []
            for s in range(30, n - SETTLE + 1, 30):
                fs = features(cam, ref, before, fr[s:s + SETTLE], z_top, 0.0, arm_mask=home_mask)
                cc, ww = collapse_check(fs); calls.append((s / FPS, cc, ww))
            bad = [(t, w) for t, cc, w in calls if cc]
            slide = f"; sliding windows: {len(bad)}/{len(calls)} false calls" + (f" e.g. at {bad[0][0]:.0f} s: {bad[0][1]}" if bad else "")
            ok = ok and not bad
        res[name] = ok
        P(f"  {name:14s} truth: {c['truth']}")
        P(f"    call {call}{note} -> {'OK' if ok else 'WRONG'}{slide}")
        P(f"    features: shift {f['shift_mm']} mm (score {f['shift_score']}), outside {f['outside']} px, rise {f['top_rise_px']} px"
          f" (expected {f['top_rise_expected_px']}), cover {f['on_top_cover']}, vanished {f['vanished']}")
        P(f"    {hand}"); P(f"    supply: {supply}")
        strip = np.hstack([cv2.resize(x, (320, 240)) for x in (ref, before, frames[-1])])
        cv2.putText(strip, f"{name}: {call[:60]}", (5, 20), 0, 0.5, (0, 0, 255), 1)
        cv2.imwrite(str(ROOT / f"notes/img/phase7_real_{name}.png"), strip)
    P(); P(f"collapse calls correct: {sum(res.values())}/{len(res)}")
    (ROOT / "notes/phase7_real_clips.txt").write_text("\n".join(L) + "\n")


def sup_layout():
    return yaml.safe_load((ROOT / "sim/assets/supply_layout.yaml").read_text())["boxes"]


if __name__ == "__main__":
    main()
