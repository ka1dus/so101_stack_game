"""Phase 3: recompute the play-area layout with the validated joint mapping.

Inputs : touch sessions in TOUCH_FILES (19 + 21 Oct, averaged per feature), sim/assets/joint_mapping.yaml,
         notes/phase1_logs/fit_joint_offsets.json (table height from the M3 fit),
         inputs/measurements.yaml (ruler sizes).
Output : sim/assets/layout.yaml, used by scripts/build_scene.py.

World frame (same convention as measurements.yaml): origin on the shoulder_pan axis at
table height, z = table normal, +x = arm direction at pan = 0, +y = arm's left. The arm
base is placed with the fitted tilt, so the table is exactly z = 0.

Rectangles: the ruler gives the sizes; the touched corners give position and heading.
Each area is fitted as a rigid rectangle of the ruler size to its 4 touched corners.

    ~/.venvs/so101_stack_sim/bin/python scripts/scene_layout.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
import fit_joint_offsets as F  # noqa: E402  (touch_point model: last 9 mm of the jaws)

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "sim/assets/joint_mapping.yaml").read_text())
MEAS = yaml.safe_load((ROOT / "inputs/measurements.yaml").read_text())
FIT = json.loads((ROOT / "notes/phase1_logs/fit_joint_offsets.json").read_text())["roll_free_from+90"]

OFFS = {k: float(v) for k, v in CFG["offset_deg"].items()}
TILT = (CFG["base_tilt_deg"]["about_x"], CFG["base_tilt_deg"]["about_y"])
TABLE_Z_BASE = FIT["table_z"]
STACKING_SPOT_RADIUS_M = 0.230         # game stacking spot distance from the pan axis (was 0.255 at the touched X)          # table height in the (tilted) base frame, from the M3 fit [m]


def rect_fit(corners, L, W):
    """Fit a rectangle of size L (corner1->2) x W (corner2->3) to 4 ordered corners."""
    c0 = corners.mean(0)
    d = corners[1] - corners[0]
    th0 = np.arctan2(d[1], d[0])
    def model(p):
        cx, cy, th = p
        u = np.array([np.cos(th), np.sin(th)]); v = np.array([-np.sin(th), np.cos(th)])
        # orientation sign (corner 2->3 to the left or right of 1->2) taken from the data
        s = np.sign(np.cross(np.r_[corners[1] - corners[0], 0], np.r_[corners[2] - corners[1], 0])[2])
        v = v * s
        half = [(-L / 2, -W / 2), (L / 2, -W / 2), (L / 2, W / 2), (-L / 2, W / 2)]
        return np.array([[cx, cy] + a * u + b * v for a, b in half])
    sol = least_squares(lambda p: (model(p) - corners).ravel(), [c0[0], c0[1], th0])
    res = np.linalg.norm(model(sol.x) - corners, axis=1)
    return sol.x, model(sol.x), res


# Touch sessions averaged per feature (same arm pose: the 21 Oct touches match 19 Oct to a
# 0.36 deg / 2 mm rigid fit). Excluded: points that are physically impossible.
TOUCH_FILES = ["inputs/table_points_20261019_181950.json", "inputs/table_points_20261021.json"]
EXCLUDE = {("inputs/table_points_20261021.json", "stack_corner_2")}   # 17 mm below the table: pressed/deflected
NEW_X_FILE = "inputs/table_points_20261021.json"                     # its stacking_spot = the new X (23 cm)


# After the Phase 5 mat expansion (2 cm strip along the corner-1/corner-4 edge, user 2026-10-21) the 4 supply
# corners are re-touched with `so101_record_points.py --points supply`; the newest such file overrides them.
SUPPLY_FILES = sorted(ROOT.glob("inputs/supply_points_*.json"))
SUPPLY_SIZE_EXPANDED = (0.209 + 0.020, 0.192)      # side 1->2 grows by the 2 cm strip; side 2->3 unchanged


def touched_points():
    sums, counts, per_file = {}, {}, {}
    for f in TOUCH_FILES:
        pts = json.loads((ROOT / f).read_text())["points"]
        per_file[f] = {p["label"]: F.touch_point(p["joints"], OFFS, TILT) for p in pts}
        for k, v in per_file[f].items():
            if (f, k) in EXCLUDE or k == "stacking_spot":
                continue
            sums[k] = sums.get(k, 0) + v; counts[k] = counts.get(k, 0) + 1
    P = {k: sums[k] / counts[k] for k in sums}
    P["stacking_spot"] = per_file[NEW_X_FILE]["stacking_spot"]
    if SUPPLY_FILES:
        for p in json.loads(SUPPLY_FILES[-1].read_text())["points"]:
            P[p["label"]] = F.touch_point(p["joints"], OFFS, TILT)
    P["stacking_spot_original"] = per_file[TOUCH_FILES[0]]["stacking_spot"]
    return P


def main():
    P = touched_points()
    global SUPPLY_FILES
    _keep, SUPPLY_FILES = SUPPLY_FILES, []
    xy_old = {k: v[:2] for k, v in touched_points().items()}
    SUPPLY_FILES = _keep
    z = np.array([v[2] for k, v in P.items() if not k.startswith("stacking_spot")]) - TABLE_Z_BASE
    xy = {k: v[:2] for k, v in P.items()}

    # table: two edges through near/far touches, 38 x 38 square from the mount corner
    def unit(a, b): d = xy[b] - xy[a]; return d / np.linalg.norm(d)
    dL, dR = unit("table_left_edge_near", "table_left_edge_far"), unit("table_right_edge_near", "table_right_edge_far")
    t = np.linalg.solve(np.c_[dL, -dR], xy["table_right_edge_near"] - xy["table_left_edge_near"])
    mount = xy["table_left_edge_near"] + t[0] * dL
    size = MEAS["table"]["size_x"]
    edge_angle = np.degrees(np.arccos(abs(dL @ dR)))
    # square the table: bisect the two measured edge directions so they're exactly 90 deg apart
    bis = (dL + dR) / np.linalg.norm(dL + dR)
    perp = np.array([-bis[1], bis[0]])
    eL = (bis + perp * np.sign(perp @ dL)) / np.sqrt(2)
    eR = (bis + perp * np.sign(perp @ dR)) / np.sqrt(2)
    corners = {"mount": mount, "left": mount + size * eL, "right": mount + size * eR,
               "far": mount + size * (eL + eR)}
    centre = mount + size / 2 * (eL + eR)

    out = {"frame": "origin on the shoulder_pan axis at table height; z up (table normal); +x arm forward at pan=0; +y arm's left",
           "source": "old touches (2026-10-19) through the validated mapping (sim/assets/joint_mapping.yaml)",
           "arm_base": {"pan_axis_xy": [0.0, 0.0], "base_origin_offset_from_pan_axis_m": [float(-F.PAN[0]), 0.0],
                        "base_origin_above_table_m": float(-TABLE_Z_BASE), "tilt_deg": list(map(float, TILT))},
           "table": {"size_m": size, "centre": centre.round(4).tolist(),
                     "heading_deg": float(np.degrees(np.arctan2(eR[1], eR[0]))),
                     "corners": {k: v.round(4).tolist() for k, v in corners.items()},
                     "measured_edge_angle_deg": round(float(edge_angle), 2)},
           "touch_heights_mm": {"range": round(float(np.ptp(z) * 1e3), 1), "mean": round(float(z.mean() * 1e3), 1)}}

    for key, pre in (("supply_area", "supply"), ("stack_area", "stack")):
        c = np.array([xy[f"{pre}_corner_{i}"] for i in (1, 2, 3, 4)])
        L, W = MEAS[key]["side_1_2"], MEAS[key]["side_2_3"]
        if key == "supply_area" and SUPPLY_FILES:
            L, W = SUPPLY_SIZE_EXPANDED
        (cx, cy, th), model, res = rect_fit(c, L, W)
        out[key] = {"size_m": [L, W], "centre": [round(float(cx), 4), round(float(cy), 4)],
                    "heading_1_2_deg": round(float(np.degrees(th)), 2),
                    "corners": model.round(4).tolist(),
                    "corner_fit_residual_mm": (res * 1e3).round(1).tolist()}
    # The touched X (photos up to 2026-10-21 show it) and the game's stacking spot, moved 2.5 cm
    # toward the pan axis by user decision (reach margin for the 8-box game, notes/phase3.md §10).
    if SUPPLY_FILES:   # the pre-expansion mat (matches reference photos taken before 2026-10-21 16:23)
        c_old = np.array([xy_old[f"supply_corner_{i}"] for i in (1, 2, 3, 4)])
        (cx, cy, th), model, res = rect_fit(c_old, MEAS["supply_area"]["side_1_2"], MEAS["supply_area"]["side_2_3"])
        out["supply_area_pre_expansion"] = {"size_m": [MEAS["supply_area"]["side_1_2"], MEAS["supply_area"]["side_2_3"]],
                                            "centre": [round(float(cx), 4), round(float(cy), 4)], "corners": model.round(4).tolist(),
                                            "note": "black mat before the 2 cm strip; use with photos taken before the expansion"}
    out["stack_area"]["stacking_spot_original"] = xy["stacking_spot_original"].round(4).tolist()
    out["stack_area"]["stacking_spot"] = xy["stacking_spot"].round(4).tolist()
    out["stack_area"]["stacking_spot_note"] = (f"the new X (planned r = {STACKING_SPOT_RADIUS_M*100:.1f} cm, user decision 2026-10-21), "
                                               f"as touched on 21 Oct: r = {np.linalg.norm(xy['stacking_spot'])*100:.1f} cm")

    out = json.loads(json.dumps(out, default=float))   # numpy scalars -> plain floats for YAML
    (ROOT / "sim/assets/layout.yaml").write_text(yaml.safe_dump(out, sort_keys=False))
    print(yaml.safe_dump(out, sort_keys=False))
    old = MEAS
    for key in ("supply_area", "stack_area"):
        d = np.linalg.norm(np.array(out[key]["centre"]) - np.array(old[key]["centre"])) * 100
        print(f"{key}: centre moved {d:.1f} cm from measurements.yaml")
    print(f"stacking spot (new X) at r = {np.linalg.norm(xy['stacking_spot'])*100:.1f} cm")


if __name__ == "__main__":
    main()
