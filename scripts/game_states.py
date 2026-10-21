"""Phase 5: randomized starting states for the place skill (CLAUDE.md Phase 5, Section 3c).

A state = a stack of n boxes (0..7; the robot places box n+1) on the stacking spot plus the
remaining 8-n boxes in the supply, and the box the robot must pick.
  - stack box types drawn per level from the real mix (3 x no3, 4 x no35, 1 x no10)
  - each level offset (x/y) and rotated (yaw) relative to the one below, mimicking imperfect
    human placement; a share of deliberately precarious stacks; every stack is settled in
    physics for 1 s and rejected if anything moves (it must stand at the start)
  - hard combinations over-sampled: large onto small, small top surfaces, top box rotated
  - supply: 60 % "game progression" (a loose grid of all 8 like the real start, then turns played:
    human = a random box, robot = the pick rule; removed boxes become the stack in order), 40 %
    random non-overlapping poses; always entirely on the arm side of the reach-line tape
  - pick rule: ACT = the nearest box to the stacking spot with room for the jaws (10 mm free on each
    long side, within +/-1 cm of the box middle) (fixed
    tie order no3 < no35 < no10, then body index); VLA = the nearest clear box of a requested type

    ~/.venvs/so101_stack_sim/bin/python scripts/game_states.py --n 200     # distribution report
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
LAY = yaml.safe_load((ROOT / "sim/assets/layout.yaml").read_text())
REACH = yaml.safe_load((ROOT / "sim/assets/reach_line.yaml").read_text())["reach_line"]
MEAS = yaml.safe_load((ROOT / "inputs/measurements.yaml").read_text())
BOX = {v["sim_name"]: v["dimensions"] for v in MEAS["boxes"]["variants"]}
MIX = {"no3": 3, "no35": 4, "no10": 1}
TYPE_ORDER = ("no3", "no35", "no10")
MAT_Z = 0.0005
TAPE_Z = 0.0002
JAW_CLEAR = 0.022          # free space needed on each width side of the picked box (finger + swing margin)
MAX_STACK = 7              # boxes already on the stack (the robot places the 8th at most)
MIN_STACK = 1              # the human always starts (Phase 2), so the robot never places onto the empty base
REJECT = {"supply": 0, "no_clear_pick": 0, "fell": 0}


@dataclass
class Box:
    type: str
    xy: np.ndarray
    yaw: float
    z: float = 0.0          # centre height
    body: str = ""


@dataclass
class State:
    stack: list = field(default_factory=list)      # bottom -> top
    supply: list = field(default_factory=list)
    pick: int = -1                                 # index into supply
    meta: dict = field(default_factory=dict)


def rect_corners(b, pad=0.0):
    d = BOX[b.type]; L, W = d["length"] / 2 + pad, d["width"] / 2 + pad
    c, s = np.cos(b.yaw), np.sin(b.yaw)
    u, v = np.array([c, s]), np.array([-s, c])
    return np.array([b.xy + a * L * u + w * W * v for a, w in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def overlap(a, b, pad=0.005):
    """Separating-axis test for two oriented rectangles (with padding)."""
    A, B = rect_corners(a, pad / 2), rect_corners(b, pad / 2)
    for poly in (A, B):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]; n = np.array([-e[1], e[0]])
            pa, pb = A @ n, B @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def usable_supply():
    """Supply polygon = mat rectangle on the arm side of the reach line (returned as half-planes)."""
    S = np.array(LAY["supply_area"]["corners"])
    planes = []
    for i in range(4):                                  # mat edges, inward normals
        a, b = S[i], S[(i + 1) % 4]; e = b - a; n = np.array([-e[1], e[0]]); n /= np.linalg.norm(n)
        if n @ (S.mean(0) - a) < 0:
            n = -n
        planes.append((a, n))
    p1, p2 = np.array(REACH["ends"]); e = p2 - p1; n = np.array([-e[1], e[0]]); n /= np.linalg.norm(n)
    if n @ (-p1) < 0:
        n = -n                                          # toward the arm (origin)
    planes.append((p1 + n * 0.0085, n))                 # keep off the 1.7 cm tape itself
    return S, planes


SUPPLY_RECT, SUPPLY_PLANES = usable_supply()


def inside_supply(b, margin=0.004):
    return all(((rect_corners(b) - a) @ n >= margin).all() for a, n in SUPPLY_PLANES)


JAW_ZONE = 0.010           # the jaws touch the box only within +/-1 cm of the middle of its long side


def side_free(b, others, side, clear):
    """Is a strip `clear` wide beside the box's long side (side = +/-1 along its width axis), over the
    jaw contact zone (+/- JAW_ZONE along the length, centred), free of other boxes?"""
    d = BOX[b.type]; c, s = np.cos(b.yaw), np.sin(b.yaw); u, v = np.array([c, s]), np.array([-s, c])
    ctr = b.xy + side * v * (d["width"] / 2 + clear / 2)
    P = np.array([ctr + a * JAW_ZONE * u + w * clear / 2 * v for a, w in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    for o in others:
        if o is b:
            continue
        Q = rect_corners(o); sep = False
        for poly in (P, Q):
            for i in range(4):
                e = poly[(i + 1) % 4] - poly[i]; n = np.array([-e[1], e[0]])
                if (P @ n).max() < (Q @ n).min() or (Q @ n).max() < (P @ n).min():
                    sep = True; break
            if sep:
                break
        if not sep:
            return False
    return True


# fixed jaw: 2 mm gap + 6 mm finger (+2 mm); moving jaw: pre-grasp opening width + 6 mm -> ~3 mm + 7 mm finger
# (Phase 5 re-evaluation, notes/phase5.md section 3; confirmed by the grasp test with neighbours)
FIXED_CLEAR, MOVING_CLEAR = 0.010, 0.010


def jaw_sides(b, others):
    """All feasible moving-jaw sides (+1/-1) for box b."""
    return [side for side in (1, -1) if side_free(b, others, side, MOVING_CLEAR) and side_free(b, others, -side, FIXED_CLEAR)]


def jaw_side(b, others):
    """+1/-1 = which width side the MOVING jaw goes on (the other side gets the fixed jaw); 0 = not pickable."""
    for side in (1, -1):
        if side_free(b, others, side, MOVING_CLEAR) and side_free(b, others, -side, FIXED_CLEAR):
            return side
    return 0


def sample_stack(rng, n, hard, types=None):
    """Types per level (respecting the mix, or given) + offsets/yaws; returns list of Box (z filled)."""
    pool = dict(MIX)
    given = types is not None
    if given:
        for t in types:
            pool[t] -= 1
    types = list(types) if given else []
    for lvl in range(0 if not given else n, n):
        avail = [t for t in TYPE_ORDER if pool[t] > 0]
        if hard and lvl == n - 1 and pool["no10"] > 0 and rng.random() < 0.5:
            t = "no10"                                   # small top surface (and large-on-small next)
        else:
            w = np.array([pool[t] for t in avail], float); t = avail[rng.choice(len(avail), p=w / w.sum())]
        types.append(t); pool[t] -= 1
    sp = np.array(LAY["stack_area"]["stacking_spot"]); head = np.radians(LAY["stack_area"]["heading_1_2_deg"])
    precarious = hard and rng.random() < 0.4
    sig_xy = 0.009 if precarious else 0.004
    boxes, z = [], TAPE_Z
    for lvl, t in enumerate(types):
        if lvl == 0:
            xy = sp + rng.normal(0, 0.006, 2); yaw = head + rng.normal(0, np.radians(8))
        else:
            below = boxes[-1]
            xy = below.xy + np.clip(rng.normal(0, sig_xy, 2), -0.02, 0.02)
            ysig = np.radians(15 if (hard and lvl == n - 1) else 7)
            yaw = below.yaw + rng.normal(0, ysig)
        h = BOX[t]["height"]
        boxes.append(Box(t, xy, yaw, z + h / 2)); z += h
    return boxes, pool, dict(precarious=bool(precarious), sig_xy_mm=sig_xy * 1000)


def sample_supply(rng, pool, grid):
    boxes = []
    types = [t for t in TYPE_ORDER for _ in range(pool[t])]
    rng.shuffle(types)
    S = SUPPLY_RECT; e1, e2 = S[1] - S[0], S[3] - S[0]
    mat_yaw = np.arctan2(e1[1], e1[0])
    # larger boxes first; mostly square to the mat (like the real loose grid), sometimes free yaw
    types.sort(key=lambda t: -BOX[t]["length"])
    for t in types:
        for k in range(2000):
            p = S[0] + rng.random() * e1 + rng.random() * e2
            if grid or k > 1000:
                yaw = mat_yaw + (np.pi / 2) * rng.integers(2) + rng.normal(0, np.radians(6)) + np.pi * rng.integers(2)
            else:
                yaw = rng.uniform(0, np.pi)
            b = Box(t, p, yaw, MAT_Z + BOX[t]["height"] / 2)
            if inside_supply(b) and not any(overlap(b, o, pad=0.002) for o in boxes):
                boxes.append(b); break
        else:
            return None
    return boxes


def grid_supply(rng):
    """All 8 boxes in a loose grid (rows along the long mat edge, ~1 cm side gaps, 3-6 mm end gaps), like the
    real start of a game; small jitter in position (+/- 2 mm) and yaw (+/- 3 deg). None if it doesn't fit."""
    S = SUPPLY_RECT; e1 = S[1] - S[0]; e2 = S[3] - S[0]; L1, L2 = np.linalg.norm(e1), np.linalg.norm(e2); e1, e2 = e1 / L1, e2 / L2
    # only rows along the long (22.9 cm) mat edge with ~1 cm side gaps hold all 8 on the arm side of the tape
    # (Phase 5 check: 30/30 at 10 mm, 3/30 at 12 mm, 0/30 for the other orientation)
    along = 1; U, V, LU, LV = e1, e2, L1, L2
    O, Vs = [(S[0], V), (S[3], -V)][int(rng.integers(2))]
    gs, ge = rng.uniform(0.0095, 0.0110), rng.uniform(0.003, 0.006)
    for _ in range(60):
        ty = [t for t in TYPE_ORDER for _ in range(MIX[t])]; rng.shuffle(ty)
        boxes = []; v = 0.004
        while ty and v < LV:
            u = 0.004; rw = max(BOX[t]["width"] for t in ty)
            for t in list(ty):
                d = BOX[t]
                if u + d["length"] > LU:
                    continue
                xy = O + U * (u + d["length"] / 2) + Vs * (v + rw / 2) + rng.normal(0, 0.001, 2)
                yaw = float(np.arctan2(U[1], U[0])) + rng.normal(0, np.radians(1.5)) + np.pi * rng.integers(2)
                bb = Box(t, xy, yaw, MAT_Z + d["height"] / 2)
                if inside_supply(bb, margin=0.002) and not any(overlap(bb, o, pad=0.001) for o in boxes):
                    boxes.append(bb); ty.remove(t); u += d["length"] + ge
            v += rw + gs
        if not ty:
            return boxes
    return None


def progress_game(rng, n, mode="act", requested=None):
    """Play n turns from a full grid (human first: a random box; robot: its pick rule). The removed boxes,
    in order, become the stack (bottom first); what's left is the supply. None if the robot got stuck."""
    supply = grid_supply(rng)
    if supply is None:
        return None
    removed = []
    for k in range(n):
        if k % 2 == 0:                                   # human turn
            removed.append(supply.pop(int(rng.integers(len(supply)))))
        else:
            st = State(supply=supply); i = choose_pick(st, "act")
            if i < 0:
                return None
            removed.append(supply.pop(i))
    return [b.type for b in removed], supply


def choose_pick(state, mode="act", requested=None):
    sp = np.array(LAY["stack_area"]["stacking_spot"])
    cand = []
    for i, b in enumerate(state.supply):
        if mode == "vla" and b.type != requested:
            continue
        if jaw_side(b, state.supply) == 0:
            continue
        cand.append((round(float(np.linalg.norm(b.xy - sp)), 4), TYPE_ORDER.index(b.type), i))
    return min(cand)[2] if cand else -1


def assign_bodies(state, model):
    """Map stack + supply boxes to the scene's named bodies (box0-2 no3, box3-6 no35, box7 no10)."""
    free = {t: [model.body(b).name for b in range(model.nbody) if model.body(b).name.endswith("_" + t)] for t in TYPE_ORDER}
    for b in state.stack + state.supply:
        b.body = free[b.type].pop(0)


def apply(state, model, data):
    for b in state.stack + state.supply:
        bid = model.body(b.body).id; adr = model.jnt_qposadr[model.body_jntadr[bid]]
        data.qpos[adr:adr + 3] = [b.xy[0], b.xy[1], b.z + 0.0002]
        data.qpos[adr + 3:adr + 7] = [np.cos(b.yaw / 2), 0, 0, np.sin(b.yaw / 2)]


def stands(model, data, state, settle=0.4, t=0.6, tol=0.001, tilt_tol=2.0):
    """Let contacts settle (soft contacts sink a fraction of a mm per level), then require that
    no stack box moves > tol or tilts > tilt_tol over the next t seconds (arm parked at home)."""
    mujoco.mj_forward(model, data)
    for _ in range(int(settle / model.opt.timestep)):
        mujoco.mj_step(model, data)
    p0 = {b.body: data.xpos[model.body(b.body).id].copy() for b in state.stack}
    for _ in range(int(t / model.opt.timestep)):
        mujoco.mj_step(model, data)
    for b in state.stack:
        bid = model.body(b.body).id
        if np.linalg.norm(data.xpos[bid] - p0[b.body]) > tol:
            return False
        R = data.xmat[bid].reshape(3, 3)
        if np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))) > tilt_tol:
            return False
    return True


def sample_state(rng, model, data, n=None, mode="act", requested=None, hard_p=0.5, home=None, max_tries=200, tall_p=None):
    """Returns (State, settled qpos) with the stack standing and a pickable box chosen.
    tall_p: share of states with height 4..MAX_STACK (uniform within each group); None = uniform over all heights."""
    for _ in range(max_tries):
        if n is not None:
            nn = n
        elif tall_p is None:
            nn = int(rng.integers(MIN_STACK, MAX_STACK + 1))
        else:
            nn = int(rng.integers(4, MAX_STACK + 1)) if rng.random() < tall_p else int(rng.integers(MIN_STACK, 4))
        hard = rng.random() < hard_p
        progression = rng.random() < 0.6
        if progression:                                   # supply = what's left of the real starting grid
            g = progress_game(rng, nn)
            if g is None:
                REJECT["supply"] += 1; continue
            stack_types, supply = g
            stack, pool, info = sample_stack(rng, nn, hard, types=stack_types)
        else:
            stack, pool, info = sample_stack(rng, nn, hard)
            supply = None
            for _ in range(10):
                supply = sample_supply(rng, pool, grid=rng.random() < 0.8)
                if supply is not None:
                    break
            if supply is None:
                REJECT["supply"] += 1; continue
        if mode == "vla" and requested is not None and pool[requested] == 0:
            continue
        st = State(stack=stack, supply=supply, meta=dict(info, hard=bool(hard), stack_height=nn, mode=mode,
                                                         supply_mode="game_progression" if progression else "random"))
        st.pick = choose_pick(st, mode, requested)
        if st.pick < 0:
            REJECT["no_clear_pick"] += 1; continue
        mujoco.mj_resetData(model, data)
        if home is not None:
            home(model, data)
        assign_bodies(st, model); apply(st, model, data)
        if not stands(model, data, st):
            REJECT["fell"] += 1; continue
        for b in st.stack + st.supply:                 # record settled poses
            bid = model.body(b.body).id
            b.xy = data.xpos[bid][:2].copy(); b.z = float(data.xpos[bid][2])
            R = data.xmat[bid].reshape(3, 3); b.yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        p = st.supply[st.pick]
        st.meta["moving_jaw_side"] = int(jaw_side(p, st.supply))
        st.meta.update(stack_types=[b.type for b in st.stack], pick_type=p.type,
                       top_type=st.stack[-1].type if st.stack else "base",
                       pairing=f"{p.type}_on_{st.stack[-1].type if st.stack else 'base'}",
                       offsets_mm=[np.round((st.stack[i].xy - st.stack[i - 1].xy) * 1000, 1).tolist()
                                   for i in range(1, len(st.stack))],
                       yaw_rel_deg=[round(float(np.degrees((st.stack[i].yaw - st.stack[i - 1].yaw + np.pi / 2) % np.pi - np.pi / 2)), 1)
                                    for i in range(1, len(st.stack))])
        return st, data.qpos.copy()
    raise RuntimeError("could not sample a standing, pickable state")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/scene.xml")); d = mujoco.MjData(m)
    def home(m_, d_):
        mujoco.mj_resetDataKeyframe(m_, d_, m_.key("home").id)
    rng = np.random.default_rng(a.seed)
    H, pairs, picks, maxoff = [], {}, {}, []
    for _ in range(a.n):
        st, _ = sample_state(rng, m, d, home=home)
        H.append(st.meta["stack_height"]); pairs[st.meta["pairing"]] = pairs.get(st.meta["pairing"], 0) + 1
        picks[st.meta["pick_type"]] = picks.get(st.meta["pick_type"], 0) + 1
        maxoff.append(max([np.linalg.norm(o) for o in st.meta["offsets_mm"]], default=0))
    print("stack heights:", {h: H.count(h) for h in range(MIN_STACK, MAX_STACK + 1)})
    print("picked type:", picks)
    print("pairings (picked_on_top):", dict(sorted(pairs.items(), key=lambda kv: -kv[1])))
    print("rejections:", REJECT)
    print(f"max level offset (mm): median {np.median(maxoff):.1f}, 90th pct {np.percentile(maxoff, 90):.1f}, max {np.max(maxoff):.1f}")


if __name__ == "__main__":
    main()
