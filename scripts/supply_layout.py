"""Phase 6: a fixed starting layout of the 8 supply boxes that maximizes the jaw clearance (user request 2026-10-22).

Jaw clearance of a box = the free width of the strip beside each of its long sides, over the jaw contact zone
(±1 cm around the middle of the side; `game_states.JAW_ZONE`), up to the nearest other box. The ends of a box don't
matter for the grasp. Objective: maximize the smallest clearance over all boxes and both sides (secondary: total
clearance, and a pick-order margin: no two boxes nearly equally far from the stacking spot, so the robot's
nearest-box rule never sits on a tie).

Constraints: every box fully on the black mat; fully on the arm side of the 29 cm reach tape; centre pickable with
vertical jaws (Phase 5 pick map: `notes/phase5_pick_map.npz`). Boxes lie flat, long axis along one of the mat's edge
directions (easy to set up by hand with tape marks).

Search: simulated annealing over positions + {0°, 90°} yaw + box order, many restarts; the best layout is saved to
`sim/assets/supply_layout.yaml` and drawn to `notes/img/phase6_supply_layout.png`.

    ~/.venvs/so101_stack_sim/bin/python scripts/supply_layout.py [--restarts 40] [--iters 6000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import game_states as G  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LAY = yaml.safe_load((ROOT / "sim/assets/layout.yaml").read_text())
REACH = yaml.safe_load((ROOT / "sim/assets/reach_line.yaml").read_text())["reach_line"]
MAT = np.array(LAY["supply_area"]["corners"])
SPOT = np.array(LAY["stack_area"]["stacking_spot"])
HEAD = np.radians(LAY["supply_area"]["heading_1_2_deg"])
PM = np.load(ROOT / "notes/phase5_pick_map.npz")
TYPES = ["no3"] * 3 + ["no35"] * 4 + ["no10"]
JAW = G.JAW_ZONE
CAP = 0.04            # clearances above 4 cm don't matter
REACH_R = REACH["radius_m"] - 0.005
SEED_FILE = "data/layout_search/final_s2.yaml"   # starting point for the search
ORDER_W, ORDER_TARGET = 0.1, 0.0   # --order-weight / --order-target (finalized layout: 0.5 / 12 mm)
MIN_GAP = 0.01        # >= 1 cm in every direction (2 cm everywhere does not fit: 297 cm2 reachable vs ~373 needed); the long sides get the 2 cm+ via the objective


def corners(c, yaw, t):
    d = G.BOX[t]; u = np.array([np.cos(yaw), np.sin(yaw)]); v = np.array([-np.sin(yaw), np.cos(yaw)])
    L, Wd = d["length"] / 2, d["width"] / 2
    return np.array([c + a * L * u + b * Wd * v for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def sat_overlap(P, Q):
    for poly in (P, Q):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]; n = np.array([-e[1], e[0]])
            if (P @ n).max() <= (Q @ n).min() or (Q @ n).max() <= (P @ n).min():
                return False
    return True


def seg_point_dist(p, a, b):
    t = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
    return float(np.linalg.norm(p - (a + t * (b - a))))


def poly_dist(P, Q):
    """Distance between two non-overlapping convex polygons (0 if they overlap)."""
    if sat_overlap(P, Q):
        return 0.0
    return min(min(seg_point_dist(p, R[i], R[(i + 1) % 4]) for p in O for i in range(4)) for O, R in ((P, Q), (Q, P)))


def side_clearance(c, yaw, t, side, others):
    """Free strip width beside one long side over the jaw zone (binary search, 0.25 mm), capped at CAP."""
    d = G.BOX[t]; u = np.array([np.cos(yaw), np.sin(yaw)]); v = np.array([-np.sin(yaw), np.cos(yaw)]) * side
    base = c + v * d["width"] / 2
    def free(w):
        S = np.array([base + a * JAW * u + b * w * v for a, b in ((-1, 0), (1, 0), (1, 1), (-1, 1))])
        return not any(sat_overlap(S, Q) for Q in others)
    if free(CAP):
        return CAP
    lo, hi = 0.0, CAP
    while hi - lo > 0.00025:
        m = (lo + hi) / 2
        lo, hi = (m, hi) if free(m) else (lo, m)
    return lo


# --- constraints -----------------------------------------------------------------------------------------------
def inside_convex(pts, poly):
    """All points inside the convex polygon (any vertex order): same side of every edge."""
    signs = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        signs.append((b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0]))
    signs = np.array(signs)
    return bool(np.all(signs >= 0) or np.all(signs <= 0))


RA, RB = np.array(REACH["ends"][0]), np.array(REACH["ends"][1])
ARM_SIDE = np.sign((RB[0] - RA[0]) * (0 - RA[1]) - (RB[1] - RA[1]) * (0 - RA[0]))   # the pan axis (origin) side


def arm_side(pts):
    s = (RB[0] - RA[0]) * (pts[:, 1] - RA[1]) - (RB[1] - RA[1]) * (pts[:, 0] - RA[0])
    return bool(np.all(np.sign(s) == ARM_SIDE))


def pickable(c):
    i = np.argmin(((PM["pts"] - c) ** 2).sum(1))
    return bool(PM["ok"][i]) and np.linalg.norm(PM["pts"][i] - c) < 0.012 and np.linalg.norm(c) <= REACH_R


EDGE_MARGIN = float(__import__("os").environ.get("EDGE_MARGIN", "0.005"))   # inside the mat edges and the tape near edge


def feasible_box(c, yaw, t):
    """Fully on the usable supply as game_states defines it (mat edges + the reach tape's NEAR edge, the tape being
    1.7 cm wide), with EDGE_MARGIN to spare, and pickable."""
    b = G.Box(t, np.asarray(c, float), float(yaw), 0.0)
    return G.inside_supply(b, margin=EDGE_MARGIN) and pickable(c)


# --- objective -------------------------------------------------------------------------------------------------
def evaluate(state):
    polys = [corners(c, y, t) for c, y, t in state]
    gaps = []
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if sat_overlap(polys[i], polys[j]):
                return -1.0, None
            gaps.append(poly_dist(polys[i], polys[j]))    # gap rule enforced as a soft penalty below
    cl = []
    for i, (c, y, t) in enumerate(state):
        others = polys[:i] + polys[i + 1:]
        cl.append((side_clearance(c, y, t, 1, others), side_clearance(c, y, t, -1, others)))
    cl = np.array(cl)
    dist = np.sort([np.linalg.norm(c - SPOT) for c, _, _ in state])
    order_margin = np.min(np.diff(dist))
    violation = sum(max(0.0, MIN_GAP - g) for g in gaps)          # soft constraint: pushes boxes apart gradually
    order_pen = max(0.0, ORDER_TARGET - order_margin)               # near-ties in the nearest-box rule (Section 3c)
    score = cl.min() + 0.02 * cl.mean() + ORDER_W * min(order_margin, 0.015) - 5.0 * violation - 2.0 * order_pen
    return score, dict(clear=cl, order_margin=order_margin, min_gap=min(gaps))


def random_state(rng):
    state = []
    for t in TYPES:
        for _ in range(2000):
            c = MAT[0] + rng.random() * (MAT[1] - MAT[0]) + rng.random() * (MAT[3] - MAT[0])
            y = HEAD + rng.integers(2) * np.pi / 2
            if feasible_box(c, y, t) and all(poly_dist(corners(c, y, t), corners(c2, y2, t2)) >= MIN_GAP for c2, y2, t2 in state):
                state.append((c, y, t)); break
        else:
            return None
    return state


def seed_state(rng):
    """Start from a saved layout (SEED_FILE; default: the 1 cm-gap optimum) when present, jittered."""
    f = ROOT / SEED_FILE
    if f.exists():
        boxes = yaml.safe_load(f.read_text())["boxes"]
        st = []
        for b in boxes:
            y = HEAD + (0 if b["long_axis"] == "along edge 1->2" else np.pi / 2)
            st.append((np.array(b["centre_world_m"]) + rng.normal(0, 0.002, 2), y, b["type"]))
        inner = MAT.mean(0) + 0.3 * (np.array([0.0, 0.0]) - MAT.mean(0))     # pull infeasible slots toward the mat centre
        fixed = []
        for c, y, t in st:
            for _ in range(60):
                if feasible_box(c, y, t):
                    break
                c = c + 0.001 * (inner - c) / np.linalg.norm(inner - c)
            fixed.append((c, y, t))
        if all(feasible_box(*x) for x in fixed):
            return fixed
    return random_state(rng)


def anneal(rng, iters):
    state = None
    while state is None:
        state = seed_state(rng)
    score, info = evaluate(state)
    best = (score, [tuple(s) for s in state], info)
    T0 = 0.004
    for k in range(iters):
        T = T0 * (1 - k / iters) + 1e-5
        i = rng.integers(len(state)); c, y, t = state[i]
        r = rng.random()
        if r < 0.75:
            c2, y2 = c + rng.normal(0, 0.006 * (1 - k / iters) + 0.0005, 2), y
        elif r < 0.9:
            c2, y2 = c, HEAD + (np.pi / 2 if abs(((y - HEAD) % np.pi)) < 0.1 else 0)
        else:                                            # swap two boxes of different type
            j = rng.integers(len(state))
            if state[j][2] == t:
                continue
            new = list(state); new[i] = (state[j][0], state[j][1], t); new[j] = (c, y, state[j][2])
            if all(feasible_box(*b) for b in (new[i], new[j])):
                s2, inf2 = evaluate(new)
                if inf2 is not None and s2 > score or rng.random() < np.exp((s2 - score) / T):
                    state, score, info = new, s2, inf2
            continue
        if not feasible_box(c2, y2, t):
            continue
        new = list(state); new[i] = (c2, y2, t)
        s2, inf2 = evaluate(new)
        if s2 == -1.0 and inf2 is None:
            continue
        if s2 > score or rng.random() < np.exp((s2 - score) / T):
            state, score, info = new, s2, inf2
            if score > best[0]:
                best = (score, list(state), info)
    return best


def mat_coords(c):
    """Box centre in the mat frame: (along edge 1→2, along edge 1→4) from corner 1, in cm."""
    e1 = (MAT[1] - MAT[0]) / np.linalg.norm(MAT[1] - MAT[0]); e2 = (MAT[3] - MAT[0]) / np.linalg.norm(MAT[3] - MAT[0])
    return float((c - MAT[0]) @ e1 * 100), float((c - MAT[0]) @ e2 * 100)


def draw(best, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    score, state, info = best
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.fill(*np.r_[MAT, MAT[:1]].T, color="0.15", alpha=0.9, label="black mat")
    ax.plot(*np.array(REACH["ends"]).T, color="w", lw=3, label="reach tape (29 cm)")
    col = {"no3": "#e07b39", "no35": "#3a86c8", "no10": "#5fb35f"}
    for k, (c, y, t) in enumerate(state):
        P = corners(c, y, t); ax.fill(*np.r_[P, P[:1]].T, color=col[t], alpha=0.9)
        a, b = mat_coords(c)
        ax.text(*c, f"{k + 1}\n{t}\n{min(info['clear'][k]) * 1000:.0f} mm", ha="center", va="center", fontsize=8, color="w")
    for i, nm in enumerate(("corner 1", "corner 2", "corner 3", "corner 4")):
        ax.annotate(nm, MAT[i], color="k", fontsize=9)
    ax.plot(0, 0, "k+", ms=12); ax.annotate("arm pan axis", (0, 0))
    ax.set_aspect("equal"); ax.set_xlim(-0.02, 0.33); ax.set_ylim(-0.0, 0.35)
    ax.set_title(f"max-gap supply layout: min jaw clearance {info['clear'].min() * 1000:.1f} mm "
                 f"(numbers: box id, type, its smaller side clearance)", fontsize=9)
    ax.legend(loc="lower right", fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--restarts", type=int, default=40); ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sim/assets/supply_layout.yaml", help="layout yaml (the png goes next to notes/img)")
    ap.add_argument("--order-weight", type=float, default=0.1); ap.add_argument("--order-target-mm", type=float, default=0.0)
    a = ap.parse_args()
    global ORDER_W, ORDER_TARGET
    ORDER_W, ORDER_TARGET = a.order_weight, a.order_target_mm / 1000
    rng = np.random.default_rng(a.seed); results = []
    for r in range(a.restarts):
        best = anneal(rng, a.iters)
        results.append(best)
        print(f"restart {r:2d}: min clearance {best[2]['clear'].min() * 1000:5.1f} mm, mean {best[2]['clear'].mean() * 1000:5.1f} mm, "
              f"pick-order margin {best[2]['order_margin'] * 1000:4.1f} mm, min gap {best[2]['min_gap'] * 1000:4.1f} mm", flush=True)
    results.sort(key=lambda b: -b[0]); best = results[0]
    score, state, info = best
    out = {"note": "fixed starting supply layout maximizing jaw clearance (scripts/supply_layout.py); mat frame = "
                   "cm from corner 1 along edge 1->2 (a) and along edge 1->4 (b); yaw relative to edge 1->2",
           "min_jaw_clearance_mm": round(float(info["clear"].min() * 1000), 1),
           "pick_order_margin_mm": round(float(info["order_margin"] * 1000), 1),
           "min_gap_any_direction_mm": round(float(info["min_gap"] * 1000), 1), "boxes": []}
    for k, (c, y, t) in enumerate(state):
        a_, b_ = mat_coords(c)
        out["boxes"].append(dict(id=k + 1, type=t, centre_world_m=[round(float(c[0]), 4), round(float(c[1]), 4)],
                                 centre_mat_cm=[round(a_, 1), round(b_, 1)],
                                 long_axis="along edge 1->2" if abs(((y - HEAD) % np.pi)) < 0.1 else "along edge 1->4",
                                 jaw_clearance_mm=[round(float(x * 1000), 1) for x in info["clear"][k]],
                                 dist_to_stack_spot_cm=round(float(np.linalg.norm(c - SPOT) * 100), 1)))
    out["score"] = round(float(score), 5)
    (ROOT / a.out).write_text(yaml.safe_dump(out, sort_keys=False))
    draw(best, ROOT / "notes/img" / f"phase6_{Path(a.out).stem}.png")
    print(yaml.safe_dump(out, sort_keys=False))


if __name__ == "__main__":
    main()
