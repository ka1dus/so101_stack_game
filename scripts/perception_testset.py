"""Phase 7: sim test set for the classical perception (`front` camera), with ground truth.

One sample = one game turn, arm parked at home (what the camera sees when perception judges):
  ref.png      game start in the same scene: all 8 boxes in their template slots, empty stack area (the real game
               captures this reference at setup)
  before.png   the state before the turn: n boxes on the stack (types from a played-out game), the rest of the
               supply still in their slots
  after.mp4    3 s from the moment the new box is dropped onto the stack (30 fps); after.png = its last frame
  seg_*.png    labels: 0 background, 1..8 = template slot id of the box, 20 arm, 21 the other static scene
  truth.json   slot -> type, stack levels (slot ids, bottom first), event, final box poses, labels
Events: "place" (box dropped on top with a human-like offset), "push" (place, then a sideways push on one level),
"nudge" (place, then the top box slid 5-25 mm), "none" (nothing changes: false-call test).
Labels follow sim_eval.Judge: the old stack is standing if no old box moved > 5 mm or tilted > 5 deg;
the new box is on top if it rests over the old top box at the right height.

    MUJOCO_GL=cgl ~/.venvs/so101_stack_sim/bin/python scripts/perception_testset.py --n 50 --seed 7001 --out data/perception/sim_v1
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import game_states as G  # noqa: E402
import grasp_test as GT  # noqa: E402
from expert import home_fn  # noqa: E402
from randomize import H, W, Randomizer  # noqa: E402

FPS = 30
EVENT_P = {"place": 0.55, "push": 0.2, "nudge": 0.15, "none": 0.1}
SEG_ARM, SEG_SCENE = 20, 21


def play(rng, n):
    """Template start, then n turns (human: any box; robot: its pick rule). Returns (start, stack slots, supply)."""
    start = G.layout_supply(rng)
    if start is None:
        return None
    for i, b in enumerate(start):
        b.slot = i + 1
    supply, removed = list(start), []
    for k in range(n):
        if k % 2 == 0:
            removed.append(supply.pop(int(rng.integers(len(supply)))))
        else:
            i = G.choose_pick(G.State(supply=supply), "act")
            if i < 0:
                return None
            removed.append(supply.pop(i))
    return start, removed, supply


def place_pose(rng, top, t):
    """Where the human (or robot) lets go of the new box: offset/yaw around the top box, 1 mm above it."""
    sig = rng.choice([0.004, 0.008, 0.015], p=[0.5, 0.35, 0.15])
    xy = top.xy + np.clip(rng.normal(0, sig, 2), -0.035, 0.035) if top is not None else \
        np.array(G.LAY["stack_area"]["stacking_spot"]) + rng.normal(0, 0.006, 2)
    yaw = (top.yaw if top is not None else np.radians(G.LAY["stack_area"]["heading_1_2_deg"])) + rng.normal(0, np.radians(10))
    z = (top.z + G.BOX[top.type]["height"] / 2 if top is not None else G.TAPE_Z) + G.BOX[t]["height"] / 2 + 0.001
    return xy, float(yaw), float(z)


def body_pose(m, d, name):
    i = m.body(name).id; R = d.xmat[i].reshape(3, 3)
    return dict(xyz=d.xpos[i].round(5).tolist(), yaw_deg=round(float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), 2),
                tilt_deg=round(float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))), 2))


def seg_labels(m, seg, slot_of_body):
    """Renderer segmentation (geom ids) -> slot ids / arm / scene."""
    gid = seg[..., 0]; out = np.zeros(gid.shape, np.uint8)
    lut = np.zeros(m.ngeom + 1, np.uint8)
    arm_root = m.body_rootid[m.body("base").id]
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]; name = m.body(b).name
        if name in slot_of_body:
            lut[g] = slot_of_body[name]
        elif m.body_rootid[b] == arm_root:
            lut[g] = SEG_ARM
        else:
            lut[g] = SEG_SCENE
    ok = gid >= 0
    out[ok] = lut[gid[ok]]
    return out


def sample(sim, rand, ren, rng, out):
    m, d = sim.m, sim.d; home = home_fn(sim)
    vis = rand.apply(rng)
    for _ in range(100):
        n = int(rng.integers(0, 8))                      # boxes already on the stack (0..7; the 8th is placed now)
        g = play(rng, n)
        if g is None:
            continue
        start, removed, supply = g
        stack, _, info = G.sample_stack(rng, n, hard=rng.random() < 0.4, types=[b.type for b in removed])
        for b, r in zip(stack, removed):
            b.slot = r.slot
        st = G.State(stack=stack, supply=supply)
        mujoco.mj_resetData(m, d); home(m, d); sim.sync_plan()
        G.assign_bodies(st, m); G.apply(st, m, d); mujoco.mj_forward(m, d)     # camera/body poses for the render
        if n and not G.stands(m, d, st, rest=1.0):
            continue
        break
    else:
        raise RuntimeError("no standing state")
    slot_type = {b.slot: b.type for b in start}
    # reference: game start in the same scene (same bodies per slot as far as types allow)
    d2 = mujoco.MjData(m); home(m, d2)
    ref = G.State(supply=[G.Box(b.type, b.xy, b.yaw, b.z) for b in start]); G.assign_bodies(ref, m); G.apply(ref, m, d2)
    mujoco.mj_forward(m, d2)
    ren.update_scene(d2, "front"); cv2.imwrite(str(out / "ref.png"), cv2.cvtColor(rand.post_front(ren.render()), cv2.COLOR_RGB2BGR))
    slot_of_body = {b.body: b.slot for b in stack + supply}

    def snap(name):
        ren.update_scene(d, "front"); img = rand.post_front(ren.render())
        ren.enable_segmentation_rendering(); ren.update_scene(d, "front"); seg = ren.render(); ren.disable_segmentation_rendering()
        cv2.imwrite(str(out / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"seg_{name}.png"), seg_labels(m, seg, slot_of_body))
        return img

    snap("before")
    old = {b.body: (d.xpos[m.body(b.body).id].copy(), d.xmat[m.body(b.body).id].copy()) for b in stack}
    event = str(rng.choice(list(EVENT_P), p=list(EVENT_P.values())))
    new = None; ev = dict(kind=event)
    if event != "none":
        human = n % 2 == 0
        i = int(rng.integers(len(supply))) if human else G.choose_pick(G.State(supply=supply), "act")
        if i < 0:
            i = int(rng.integers(len(supply)))
        new = supply.pop(i)
        xy, yaw, z = place_pose(rng, stack[-1] if stack else None, new.type)
        bid = m.body(new.body).id; adr = m.jnt_qposadr[m.body_jntadr[bid]]
        d.qpos[adr:adr + 3] = [xy[0], xy[1], z]; d.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qvel[m.jnt_dofadr[m.body_jntadr[bid]]:][:6] = 0
        mujoco.mj_forward(m, d)
        ev.update(player="human" if human else "robot", slot=new.slot, type=new.type,
                  drop_offset_mm=(np.r_[xy] - (stack[-1].xy if stack else xy)).round(4).__mul__(1000).tolist())
    vw = cv2.VideoWriter(str(out / "after.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    push = None
    if event == "push":
        lvls = stack + [new]; lv = int(rng.integers(len(lvls))); ang = rng.uniform(0, 2 * np.pi)
        push = dict(level=lv, force_N=round(float(rng.uniform(0.15, 0.6)), 3), dir_deg=round(float(np.degrees(ang)), 1),
                    t0=round(float(rng.uniform(0.8, 1.5)), 2), max_dur=0.5, stop_after_mm=round(float(rng.uniform(15, 40)), 1))
        ev["push"] = push; pb = m.body(lvls[lv].body).id; p0 = None    # a finger push: stops once the box has moved
    nudge_t = None
    if event == "nudge":
        nudge_t = round(float(rng.uniform(0.8, 1.5)), 2); ev["nudge_t"] = nudge_t
    rest = None                                          # new box pose once it has landed (before any push/nudge)
    for k in range(3 * FPS):
        t = k / FPS
        if new is not None and k == int(round(0.6 * FPS)):
            rest = d.xpos[m.body(new.body).id].copy()
        if push is not None:
            on = push["t0"] <= t < push["t0"] + push["max_dur"]
            if on and p0 is None:
                p0 = d.xpos[pb].copy()
            if on and np.linalg.norm(d.xpos[pb] - p0) * 1000 > push["stop_after_mm"]:
                push["max_dur"] = round(t - push["t0"], 3); on = False
            d.xfrc_applied[pb, :3] = [push["force_N"] * np.cos(ang), push["force_N"] * np.sin(ang), 0] if on else 0
        if nudge_t is not None and k == int(round(nudge_t * FPS)):
            top = new if new is not None else stack[-1]; tb = m.body(top.body).id; adr = m.jnt_qposadr[m.body_jntadr[tb]]
            dist = rng.uniform(0.005, 0.025); a2 = rng.uniform(0, 2 * np.pi)
            d.qpos[adr:adr + 2] += [dist * np.cos(a2), dist * np.sin(a2)]; mujoco.mj_forward(m, d)
            ev["nudge_mm"] = round(dist * 1000, 1)
        sim.wait(1 / FPS)
        ren.update_scene(d, "front"); vw.write(cv2.cvtColor(rand.post_front(ren.render()), cv2.COLOR_RGB2BGR))
    vw.release(); d.xfrc_applied[:] = 0
    snap("after")
    unstable = int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) > 0     # MuJoCo reset the state: drop the sample
    # labels
    moved = {k: float(np.linalg.norm(d.xpos[m.body(k).id] - p)) for k, (p, _) in old.items()}
    tilt = {k: float(np.degrees(np.arccos(np.clip(d.xmat[m.body(k).id].reshape(3, 3)[2, 2], -1, 1)))) for k in old}
    standing = all(moved[k] < 0.005 and tilt[k] < 5 for k in old)
    on_top = None
    if new is not None:
        nb = d.xpos[m.body(new.body).id]
        if stack:
            top = stack[-1]; tid = m.body(top.body).id; Rt = d.xmat[tid].reshape(3, 3); rel = Rt.T @ (nb - d.xpos[tid])
            exp_z = d.xpos[tid][2] + G.BOX[top.type]["height"] / 2 + G.BOX[new.type]["height"] / 2
            on_top = bool(abs(rel[0]) < G.BOX[top.type]["length"] / 2 and abs(rel[1]) < G.BOX[top.type]["width"] / 2
                          and abs(nb[2] - exp_z) < 0.005)
        else:
            on_top = bool(abs(nb[2] - (G.TAPE_Z + G.BOX[new.type]["height"] / 2)) < 0.005)
    lifted = (stack + ([new] if new else []))[1:]           # every level but the bottom one should be off the table
    on_table = [b.slot for b in lifted if d.xpos[m.body(b.body).id][2] < G.BOX[b.type]["height"] / 2 + 0.004]
    levels = [b.slot for b in stack] + ([new.slot] if new is not None else [])
    truth = dict(
        slot_type=slot_type, stack_before=[b.slot for b in stack], supply_before=sorted([b.slot for b in supply] + ([new.slot] if new else [])),
        stack_after_intended=levels, event=ev, stack_info=info,
        labels=dict(old_stack_standing=standing, new_on_top=on_top, collapse=bool(not standing or on_top is False),
                    top_moved_after_landing_mm=round(1000 * float(np.linalg.norm(d.xpos[m.body(new.body).id] - rest)), 2) if new else None,
                    max_old_move_mm=round(1000 * max(moved.values(), default=0.0), 2),
                    max_old_tilt_deg=round(max(tilt.values(), default=0.0), 2), boxes_on_table=on_table),
        poses_after={str(slot_of_body[k]): body_pose(m, d, k) for k in slot_of_body},
        camera=dict(pos=m.cam_pos[rand.cam["front"]].round(5).tolist(), quat=m.cam_quat[rand.cam["front"]].round(6).tolist()),
        randomization=vis, sim_unstable=unstable)
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    return truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50); ap.add_argument("--seed", type=int, default=7001)
    ap.add_argument("--out", default="data/perception/sim_v1")
    a = ap.parse_args()
    sim = GT.Sim(); rand = Randomizer(sim.m); ren = mujoco.Renderer(sim.m, H, W)
    rng = np.random.default_rng(a.seed); root = Path(a.out)
    for i in range(a.n):
        out = root / f"s{a.seed}_{i:04d}"; out.mkdir(parents=True, exist_ok=True)
        t = sample(sim, rand, ren, rng, out)
        L = t["labels"]
        print(f"{out.name}: h{len(t['stack_before'])} {t['event']['kind']:5s} collapse={L['collapse']} "
              f"move={L['max_old_move_mm']}mm on_top={L['new_on_top']}", flush=True)


if __name__ == "__main__":
    main()
