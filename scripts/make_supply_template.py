"""Printable 1:1 stencil of the template supply layout (sim/assets/supply_layout.yaml) for setting up a game.

Print at 100 % ("actual size", no fit-to-page) on A4 landscape, check the 10 cm scale bar with a ruler, cut out the
slot rectangles, lay the sheet on the black mat with its outline on the mat edges (corner 1 = the corner nearest the
arm base), drop each box into its labelled slot, then lift the sheet away. Nothing stays on the mat (the policy was
trained on a plain mat).

    ~/.venvs/so101_stack_sim/bin/python scripts/make_supply_template.py   # -> notes/supply_template.pdf (+ .png preview)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LAY = yaml.safe_load((ROOT / "sim/assets/layout.yaml").read_text())
SLOTS = yaml.safe_load((ROOT / "sim/assets/supply_layout.yaml").read_text())
MEAS = yaml.safe_load((ROOT / "inputs/measurements.yaml").read_text())
BOX = {v["sim_name"]: v["dimensions"] for v in MEAS["boxes"]["variants"]}
REACH = yaml.safe_load((ROOT / "sim/assets/reach_line.yaml").read_text())["reach_line"]
PAGE = (297.0, 210.0)                       # A4 landscape, mm


def main():
    C = np.array(LAY["supply_area"]["corners"])          # world metres, corner 1..4
    e1 = (C[1] - C[0]) / np.linalg.norm(C[1] - C[0])
    n = np.array([-e1[1], e1[0]])                        # e1 turned +90 deg (z up): a top view, not mirrored
    def to_mm(p):                                        # world -> page mm, corner 1 at the page offset
        d = np.asarray(p) - C[0]
        return np.array([d @ e1, d @ n]) * 1000
    pts = np.array([to_mm(c) for c in C])
    off = np.array([(PAGE[0] - (pts[:, 0].max() - pts[:, 0].min())) / 2 - pts[:, 0].min(),
                    (PAGE[1] - (pts[:, 1].max() - pts[:, 1].min())) / 2 - pts[:, 1].min()])
    P = lambda p: to_mm(p) + off
    fig = plt.figure(figsize=(PAGE[0] / 25.4, PAGE[1] / 25.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, PAGE[0]); ax.set_ylim(0, PAGE[1]); ax.set_aspect("equal"); ax.axis("off")
    mat = np.array([P(c) for c in C]); ax.plot(*np.r_[mat, mat[:1]].T, "k-", lw=1.2)
    for i, c in enumerate(mat):
        ax.annotate(f"corner {i + 1}" + (" (nearest the arm base)" if i == 0 else ""), c, fontsize=7,
                    xytext=(c - mat.mean(0)) / np.linalg.norm(c - mat.mean(0)) * 6 + c, ha="center")
    r = np.array([P(p) for p in REACH["ends"]]); ax.plot(*r.T, "k--", lw=0.8)
    ang = np.degrees(np.arctan2(*(r[1] - r[0])[::-1])); ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang
    ax.text(*(r.mean(0) + [4, 0]), "reach tape (boxes stay on the arm side)", fontsize=7, rotation=ang,
            ha="center", va="center", rotation_mode="anchor")
    head = np.arctan2(*(C[1] - C[0])[::-1])
    for s in SLOTS["boxes"]:
        d = BOX[s["type"]]; c = np.array(s["centre_world_m"])
        yaw = head + (0 if s["long_axis"] == "along edge 1->2" else np.pi / 2)
        u = np.array([np.cos(yaw), np.sin(yaw)]); v = np.array([-np.sin(yaw), np.cos(yaw)])
        cs = [P(c + a * d["length"] / 2 * u + b * d["width"] / 2 * v) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        poly = np.array(cs); ax.fill(*poly.T, color="0.85", ec="k", lw=0.8)
        name = {"no3": "No.3", "no35": "No.35", "no10": "No.10"}[s["type"]]
        ax.text(*P(c), f"{s['id']}\n{name}", ha="center", va="center", fontsize=8)
    x = (mat[:, 0].max() + PAGE[0]) / 2                  # vertical 100 mm bar in the free margin right of the mat
    ax.plot([x, x], [55, 155], "k-", lw=2)
    ax.text(x + 3, 105, "100 mm: check with a ruler (print at 100 %)", va="center", fontsize=7, rotation=90)
    ax.text(PAGE[0] / 2, PAGE[1] - 6, "SO-101 stacking game: supply template (cut out the grey slots; lift the sheet away "
            "after placing the boxes)", ha="center", fontsize=8)
    out = ROOT / "notes/supply_template.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=80); plt.close(fig)
    print("wrote", out, "and", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
