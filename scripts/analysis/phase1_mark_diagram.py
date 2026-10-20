# Phase 1 M3 diagrams: where to put marks A, B, C.
#   ~/.venvs/so101_stack_sim/bin/python scripts/analysis/phase1_mark_diagram.py
# Writes notes/img/phase1_marks_topdown.png and notes/img/phase1_marks_frontcam.png.
# Geometry = measurements.yaml [fk] positions (good to ~1.5-3 cm); exact mark positions
# don't matter for the fit, only that each is clearly marked and the A-B, B-C, A-C
# distances are measured with a ruler.

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from render_pose import _project, fit_camera  # noqa: E402

M = yaml.safe_load((ROOT / "inputs/measurements.yaml").read_text())
T = M["table"]["corners"]
table = np.array([T["mount_corner"], T["left_corner"], T["far_corner"], T["right_corner"]])
supply = np.array(M["supply_area"]["corners"])
stack = np.array(M["stack_area"]["corners"])
spot = np.array(M["stack_area"]["stacking_spot"])
far3 = supply[2]                                    # supply corner 3 = farthest from the arm
B = far3 + 0.04 * (np.array(M["supply_area"]["centre"]) - far3) / np.linalg.norm(np.array(M["supply_area"]["centre"]) - far3)
MARKS = {"A": spot, "B": B, "C": np.array([0.120, -0.030])}
HOME_TIP = np.array([0.131, 0.006])                 # jaw tips at home (sim), for orientation
CAM = np.array([M["cameras"]["front"]["position"]["x"], M["cameras"]["front"]["position"]["y"]])


def topdown():
    # view from behind the arm: +x (arm forward) points UP the page, +y (arm's left) to the LEFT
    def P(xy):
        xy = np.atleast_2d(xy)
        return np.c_[-xy[:, 1], xy[:, 0]] * 100     # page coords in cm
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.add_patch(plt.Polygon(P(table), closed=True, fc="#f4f4f4", ec="#888", lw=1.5))
    ax.add_patch(plt.Polygon(P(supply), closed=True, fc="#222", ec="#000"))
    ax.add_patch(plt.Polygon(P(stack), closed=True, fc="#f2c12e", ec="#c99a00"))
    ax.text(*P(np.array(M["supply_area"]["centre"]))[0] + [0, -2.5], "black supply mat", color="w", ha="center", fontsize=10)
    ax.text(*P(np.array(M["stack_area"]["centre"]))[0] + [6.5, -1.5], "yellow\nstack area", color="#5a4500", ha="center", fontsize=10)
    # arm
    ax.add_patch(plt.Rectangle((-4, -6), 8, 8, fc="#555", ec="k"))
    ax.annotate("", xy=(0, 9), xytext=(0, 1), arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
    ax.text(0.6, -4.5, "arm base\n(pan axis)", color="w", fontsize=8, va="bottom")
    ax.text(-10.5, 3.2, "arm points this\nway at pan = 0", fontsize=8, color="#555", bbox=dict(fc="w", ec="none", alpha=0.8))
    hp = P(HOME_TIP)[0]
    ax.plot(*hp, marker="+", ms=12, color="#1f77b4", mew=2, ls="none", label="jaw tips at home (look straight down)")
    ax.legend(loc="lower right", fontsize=8, bbox_to_anchor=(1.0, 0.07))
    colors = {"A": "#d62728", "B": "#2ca02c", "C": "#9467bd"}
    notes = {
        "A": "A = the X already on the\nstacking spot (yellow)",
        "B": "B = small white tape with an X,\n4 cm in from the mat's FAR corner\n(the corner farthest from the arm),\ntoward the mat centre",
        "C": "C = X on the white table,\n~12 cm in front of the base,\n3 cm toward the yellow side\n(just right of the home jaw tips)",
    }
    offs = {"A": (3.5, 5), "B": (-15, 8), "C": (6, -8)}
    for k, xy in MARKS.items():
        p = P(xy)[0]
        ax.plot(*p, marker="X", ms=14, color=colors[k], mec="w", mew=1.5)
        ax.annotate(notes[k], xy=p, xytext=p + offs[k], fontsize=9, color=colors[k],
                    bbox=dict(fc="w", ec=colors[k], alpha=0.9), arrowprops=dict(arrowstyle="-", color=colors[k], lw=1))
    for (a, b) in (("A", "B"), ("B", "C"), ("A", "C")):
        pa, pb = P(MARKS[a])[0], P(MARKS[b])[0]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], ls=":", color="#666", lw=1.2)
        d = np.linalg.norm(MARKS[a] - MARKS[b]) * 100
        ax.text(*(pa + pb) / 2 + [0.5, -2.5], f"{a}-{b} ~{d:.0f} cm", fontsize=8, color="#444",
                bbox=dict(fc="w", ec="none", alpha=0.75))
    cdir = P(CAM)[0] / np.linalg.norm(P(CAM)[0])
    ax.annotate("to front camera\n(tripod)", xy=cdir * 46, xytext=cdir * 36, fontsize=8, color="#444",
                arrowprops=dict(arrowstyle="->", color="#444"))
    ax.text(-30, -9, "LEFT of the arm\n(black-mat side)", fontsize=9, color="#444")
    ax.text(18, -9, "RIGHT of the arm\n(yellow side)", fontsize=9, color="#444", ha="right")
    ax.set_xlim(-34, 30); ax.set_ylim(-10, 52); ax.set_aspect("equal")
    ax.set_xlabel("cm (to the arm's left <-  -> to the arm's right)"); ax.set_ylabel("cm in front of the arm's pan axis")
    ax.set_title("M3 marks, seen from above, standing BEHIND the arm\n(positions within ~2 cm are fine)", fontsize=11)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / "notes/img/phase1_marks_topdown.png", dpi=110)


def frontcam():
    cam = fit_camera()
    x_mj, y_mj = cam["xyaxes"][:3], cam["xyaxes"][3:]
    Rcv = np.stack([x_mj, -y_mj, np.cross(x_mj, -y_mj)])
    rvec, _ = cv2.Rodrigues(Rcv)
    img = np.asarray(Image.open(ROOT / "inputs/photos/front_cam_frame.png").convert("RGB"))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    colors = {"A": "#d62728", "B": "#2ca02c", "C": "#9467bd"}
    for k, xy in MARKS.items():
        u, v = _project(np.array([[xy[0], xy[1], 0.0]]), rvec.ravel(), cam["pos"], cam["f"], cam["k1"])[0]
        ax.plot(u, v, marker="X", ms=16, color=colors[k], mec="w", mew=2)
        ax.text(u + 10, v - 10, k, color=colors[k], fontsize=16, weight="bold",
                bbox=dict(fc="w", alpha=0.7, ec="none", pad=1))
    ax.set_axis_off()
    ax.set_title("Same marks in the front-camera view (approximate, +/- 2 cm)", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "notes/img/phase1_marks_frontcam.png", dpi=110)
    return cam


if __name__ == "__main__":
    topdown()
    cam = frontcam()
    for k, xy in MARKS.items():
        print(k, np.round(xy * 100, 1), "cm")
    for a, b in (("A", "B"), ("B", "C"), ("A", "C")):
        print(f"{a}-{b} {np.linalg.norm(MARKS[a] - MARKS[b]) * 100:.1f} cm")
