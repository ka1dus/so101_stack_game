"""Phase 3: crop the box-face photos into MuJoCo textures (CLAUDE.md Section 3b).

inputs/photos/box_faces/<type>/<face>.jpg  ->  sim/assets/textures/<type>_<face>.png
Faces: top/bottom = largest faces (length x width), front/back = long sides
(length x height), left/right = short ends (width x height). Each crop is resized to
the face's real aspect ratio (dimensions from measurements.yaml) at 4 px/mm.
A contact sheet goes to notes/img/phase3_box_textures.png for checking.

    ~/.venvs/so101_stack_sim/bin/python scripts/make_box_textures.py
"""

from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sim/assets/textures"
PX_PER_MM = 4
FACES = ("top", "bottom", "front", "back", "left", "right")


def face_size_mm(dims, face):
    L, W, H = (dims[k] * 1000 for k in ("length", "width", "height"))
    return {"top": (L, W), "bottom": (L, W), "front": (L, H), "back": (L, H), "left": (W, H), "right": (W, H)}[face]


def crop_box(img: np.ndarray, aspect: float) -> np.ndarray:
    """Find the box in a face photo (box on a light, slightly shaded background)."""
    small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    edges = cv2.Canny(g, 20, 60)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(edges)
    h, w = g.shape
    best, best_score = None, -1
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < 0.1 * w or bh < 0.03 * h:
            continue
        cx, cy = x + bw / 2, y + bh / 2
        centred = 1 - (abs(cx - w / 2) / w + abs(cy - h / 2) / h)
        score = bw * bh * centred
        if score > best_score:
            best, best_score = (x, y, bw, bh), score
    x, y, bw, bh = best
    # the edge band is dilated by ~2 px; trim it back and scale to full resolution
    x, y, bw, bh = [int(round(v * 4)) for v in (x + 2, y + 2, bw - 4, bh - 4)]
    crop = img[y:y + bh, x:x + bw]
    return crop, (x, y, bw, bh), bw / max(bh, 1)


def main():
    meas = yaml.safe_load((ROOT / "inputs/measurements.yaml").read_text())
    dims = {v["sim_name"]: v["dimensions"] for v in meas["boxes"]["variants"]}
    OUT.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (6 * 300, 3 * 160), "white")
    report = []
    for r, t in enumerate(("no3", "no35", "no10")):
        for c, f in enumerate(FACES):
            img = np.asarray(ImageOps.exif_transpose(Image.open(ROOT / f"inputs/photos/box_faces/{t}/{f}.jpg")).convert("RGB"))
            wmm, hmm = face_size_mm(dims[t], f)
            crop, bbox, got = crop_box(img, wmm / hmm)
            if (got > 1) != (wmm / hmm > 1):        # photo rotated 90 deg relative to the face
                crop = np.rot90(crop)
                got = 1 / got
            tex = Image.fromarray(crop).resize((int(wmm * PX_PER_MM), int(hmm * PX_PER_MM)), Image.LANCZOS)
            tex.save(OUT / f"{t}_{f}.png")
            report.append(f"{t:5s} {f:6s} expected aspect {wmm/hmm:5.2f} cropped {got:5.2f} ({100*(got/(wmm/hmm)-1):+5.1f}%)")
            th = tex.copy(); th.thumbnail((290, 150)); sheet.paste(th, (c * 300 + 5, r * 160 + 5))
    sheet.save(ROOT / "notes/img/phase3_box_textures.png")
    print("\n".join(report))


if __name__ == "__main__":
    main()
