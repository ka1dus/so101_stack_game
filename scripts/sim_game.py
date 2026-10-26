"""Phase 6/7: full games in sim, ACT policy vs a scripted "human", judged by the classical perception (in the loop).

Each game: template supply, empty stack, the human starts. The controller sees only what the real one will:
rendered `front` frames, joint readings and the gripper command. Per turn it runs
  supply-slot check (D1)  which slot was emptied -> the type of the new level (ambiguous: "ask", answered from truth)
  drop_check              robot turns, from the joint readings
  collapse_check          before/after frames and the 3 s settle window (robot turns: from the moment the arm is home)
and calls the turn: ok, or loss (drop / collapse / no placement). The sim's exact state judges the same turn by the
game rule (a stack box on the table, the top moved > 15 mm, or the new box not on the stack), so every call is scored.
The game ends at the first loss (called or true) or when all 8 boxes are stacked.

Human turns: a random supply box is let go above the stack with human-like error (perception_testset.place_pose);
5 % are careless (offset 2.5-4 cm). Robot turns: the policy runs from home (temporal ensembling, 320x240 input) until
drop_check reports the release and the jaw tips have risen 3 cm (or a drop, or 25 s), then a scripted move home.

    MUJOCO_GL=cgl ~/.venvs/so101_eval/bin/python scripts/sim_game.py --policy data/policies/act_v3_ft_060000 --games 2 --device cpu
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(ROOT))
import game_states as G  # noqa: E402
import grasp_test as GT  # noqa: E402
from generate import TASK_ACT  # noqa: E402
from joint_mapping import JOINT_NAMES, JointMapping  # noqa: E402
from perception.drop import DropMonitor, Kinematics  # noqa: E402
from perception.register import refine  # noqa: E402
from perception.stack import collapse_check, features, top_xy, change, stack_region, SPOT  # noqa: E402
from perception.supply import SupplyMonitor  # noqa: E402
from perception.twin import Twin, arm_mask  # noqa: E402
from perception_eval import template_poses  # noqa: E402
from perception_testset import place_pose  # noqa: E402
from randomize import H, W, Randomizer  # noqa: E402
from sim_eval import MAX_REL, load_policy  # noqa: E402

FPS = 30
SETTLE_S = 3.0
SKILL_T_MAX = 25.0
RAISE_M = 0.035            # after the release the policy keeps control until the tips are this far up (its own
                           # straight-up retract) or RELEASE_HOLD_S has passed, then the scripted move home
RELEASE_HOLD_S = 1.5       # (5 cm was too much: the policy's retract often stops short, and an idle policy wanders)
UNSURE = (0.3, 0.7)
MEAS = yaml.safe_load((ROOT / "inputs/measurements.yaml").read_text())
HOME_REAL = [MEAS["arm"]["home_pose"][j] for j in JOINT_NAMES]


class Game:
    def __init__(self, sim, rand, ren, twin, kin, policy, rng, resize="320x240", video=None, debug=None):
        self.sim, self.rand, self.ren, self.twin, self.kin, self.policy, self.rng = sim, rand, ren, twin, kin, policy, rng
        self.resize = tuple(int(x) for x in resize.split("x")); self.video = video; self.debug = debug; self.trace = []
        self.jm = JointMapping.from_files()

    # ---------- rendering / sim helpers
    def front(self):
        self.ren.update_scene(self.sim.d, "front"); img = cv2.cvtColor(self.rand.post_front(self.ren.render()), cv2.COLOR_RGB2BGR)
        if self.video is not None:
            self.video.write(img)
        return img

    def wrist(self):
        self.ren.update_scene(self.sim.d, "wrist"); return self.rand.post_wrist(self.ren.render())

    def q_real(self):
        s = self.sim; return self.jm.sim_to_real(np.r_[s.d.qpos[s.qadr], s.d.qpos[s.gadr]])

    def go_home(self, T=1.5, lift=0.0):
        """Scripted return home; lift > 0 first raises the jaw tips straight up (after a drop call or a timeout the
        jaws may still be next to the stack: never move sideways from there)."""
        s = self.sim; s.sync_plan()
        if lift:
            R = s.d.xmat[s.gb].reshape(3, 3); p_grip = np.array([s.xin, 0.0, s.ztip])
            tip = s.d.xpos[s.gb] + R @ p_grip; xdir = R[:2, 0] / np.linalg.norm(R[:2, 0])
            q_up, err, _ = s.ik(p_grip, tip + [0, 0, lift], xdir, [s.d.qpos[s.qadr].copy()], w_orient=10.0)
            if err < 10:
                s.move(q_up, s.plan_g, 0.8)
        s.move(s.home[:5], s.home[5], T)

    def pose(self, body):
        i = self.sim.m.body(body).id; return self.sim.d.xpos[i].copy(), self.sim.d.xmat[i].reshape(3, 3).copy()

    # ---------- truth (game rule, from the sim's exact state)
    def truth_turn(self, stack_bodies, new_body, old_poses):
        m, d = self.sim.m, self.sim.d
        on_table = any(d.xpos[m.body(b).id][2] < G.BOX[self.type_of[b]]["height"] / 2 + 0.004 for b in stack_bodies[1:])
        top_moved = 0.0
        if stack_bodies:
            top = stack_bodies[-1]; top_moved = float(np.linalg.norm(d.xpos[m.body(top).id][:2] - old_poses[top][0][:2]))
        new_ok = None
        if new_body is not None:
            nb = d.xpos[m.body(new_body).id]
            if stack_bodies:
                top = stack_bodies[-1]; tp, tR = self.pose(top); rel = tR.T @ (nb - tp)
                exp_z = tp[2] + G.BOX[self.type_of[top]]["height"] / 2 + G.BOX[self.type_of[new_body]]["height"] / 2
                new_ok = bool(abs(rel[0]) < G.BOX[self.type_of[top]]["length"] / 2 and abs(rel[1]) < G.BOX[self.type_of[top]]["width"] / 2
                              and abs(nb[2] - exp_z) < 0.006)
            else:
                new_ok = bool(abs(nb[2] - (G.TAPE_Z + G.BOX[self.type_of[new_body]]["height"] / 2)) < 0.006
                              and np.linalg.norm(nb[:2] - SPOT) < 0.08)
        loss = on_table or top_moved > 0.015 or new_ok is False
        why = "box on the table" if on_table else f"top moved {top_moved * 1000:.0f} mm" if top_moved > 0.015 else \
            "new box not on the stack" if new_ok is False else ""
        return bool(loss), why

    # ---------- turns
    def human_place(self, body, stack_bodies):
        m, d, rng = self.sim.m, self.sim.d, self.rng
        top = None
        if stack_bodies:
            p, R = self.pose(stack_bodies[-1]); top = G.Box(self.type_of[stack_bodies[-1]], p[:2], float(np.arctan2(R[1, 0], R[0, 0])), float(p[2]))
        xy, yaw, z = place_pose(rng, top, self.type_of[body])
        if rng.random() < 0.05:                            # careless placement
            a = rng.uniform(0, 2 * np.pi); xy = (top.xy if top is not None else SPOT) + rng.uniform(0.025, 0.04) * np.r_[np.cos(a), np.sin(a)]
        bid = m.body(body).id; adr = m.jnt_qposadr[m.body_jntadr[bid]]; dof = m.jnt_dofadr[m.body_jntadr[bid]]
        d.qpos[adr:adr + 3] = [xy[0], xy[1], z]; d.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qvel[dof:dof + 6] = 0; mujoco.mj_forward(m, d)
        frames = []
        for _ in range(int(SETTLE_S * FPS)):
            self.sim.wait(1 / FPS); frames.append(self.front())
        return frames, None

    def robot_skill(self, top_xy_est, top_face_est):
        sim, m, d = self.sim, self.sim.m, self.sim.d
        pol, pre, post = self.policy; pol.reset(); pre.reset(); post.reset()
        act_ids = [sim.act[j] for j in JOINT_NAMES]; lo, hi = m.actuator_ctrlrange[act_ids, 0], m.actuator_ctrlrange[act_ids, 1]
        mon = DropMonitor(self.kin, top_xy_est, top_face_est); rel_z = None; event = "timeout"; self.trace = []; self.arm_frames = None
        for k in range(int(SKILL_T_MAX * FPS)):
            state = self.q_real().astype(np.float32)
            self.last_front = self.front()
            f = cv2.resize(cv2.cvtColor(self.last_front, cv2.COLOR_BGR2RGB), self.resize, interpolation=cv2.INTER_AREA)
            w = cv2.resize(self.wrist(), self.resize, interpolation=cv2.INTER_AREA)
            obs = {"observation.state": torch.from_numpy(state)[None],
                   "observation.images.front": torch.from_numpy(f).permute(2, 0, 1)[None].float() / 255,
                   "observation.images.wrist": torch.from_numpy(w).permute(2, 0, 1)[None].float() / 255, "task": [TASK_ACT]}
            with torch.inference_mode():
                a = post(pol.select_action(pre(obs))).squeeze(0).float().cpu().numpy()
            a = np.clip(a, state - MAX_REL, state + MAX_REL)
            d.ctrl[act_ids] = np.clip(self.jm.real_to_sim(a), lo, hi)
            for _ in range(sim.steps_per_ctrl):
                mujoco.mj_step(m, d)
            ev = mon.tick(self.q_real(), float(a[5])); self.trace.append(np.r_[self.q_real(), a[5]])
            if ev and ev.startswith("drop"):
                event = ev; break
            if ev == "released":
                tip = self.kin.tip_pos(self.q_real())
                if rel_z is None:
                    rel_z, rel_k = tip[2], k
                    self.arm_frames = [(self.last_front, arm_mask(self.twin, self.cam, self.q_real()))]
                if tip[2] > rel_z + RAISE_M or k - rel_k >= RELEASE_HOLD_S * FPS:
                    self.arm_frames.append((self.front(), arm_mask(self.twin, self.cam, self.q_real())))
                    event = "released"; break
        return event, (k + 1) / FPS

    def play(self, game_i):
        sim, m, d, rng = self.sim, self.sim.m, self.sim.d, self.rng
        self.rand.apply(rng)
        start = G.layout_supply(rng)
        for i, b in enumerate(start):
            b.slot = i + 1
        st = G.State(supply=start); mujoco.mj_resetData(m, d); G.assign_bodies(st, m); G.apply(st, m, d)
        d.qpos[sim.qadr] = sim.home[:5]; d.qpos[sim.gadr] = sim.home[5]
        for j, v in zip(JOINT_NAMES, sim.home):
            d.ctrl[sim.act[j]] = v
        mujoco.mj_forward(m, d); sim.sync_plan(); sim.wait(0.5)
        self.type_of = {b.body: b.type for b in start}; slot_of = {b.body: b.slot for b in start}; body_of = {b.slot: b.body for b in start}
        ref = self.front()
        cam, rinfo = refine(ref, self.twin, template_poses(self.twin), HOME_REAL); self.cam = cam; self.arm_frames = None
        supply = SupplyMonitor(ref, cam); home_mask = arm_mask(self.twin, cam, HOME_REAL)
        stack_bodies = []; perceived_slots = []; turns = []
        supply_pose0 = {b.body: self.pose(b.body)[0] for b in start}
        for turn in range(8):
            player = "human" if turn % 2 == 0 else "robot"
            before = self.front()
            old = {b: self.pose(b) for b in stack_bodies}
            left = [b for b in body_of.values() if b not in stack_bodies and np.linalg.norm(self.pose(b)[0] - supply_pose0[b]) < 0.01]
            # perception's view of the stack before the turn
            z_top = 0.0002 + sum(G.BOX[self.type_of[body_of[s]]]["height"] for s in perceived_slots)
            fg_b = change(before, ref) & stack_region(cam, arm_mask=home_mask)
            xy_est = top_xy(cam, fg_b, z_top) if perceived_slots else None
            xy_est = SPOT if xy_est is None else xy_est
            info = dict(turn=turn, player=player, height=len(stack_bodies))
            if player == "human":
                body = left[int(rng.integers(len(left)))]
                frames, ev = self.human_place(body, stack_bodies); info["drop_call"] = None
            else:
                ev, dur = self.robot_skill(xy_est, z_top); info.update(skill_event=ev, skill_s=round(dur, 1))
                self.go_home(lift=0.0 if ev == "released" else 0.06)
                frames = [self.front() for _ in range(int(SETTLE_S * FPS)) if sim.wait(1 / FPS) is None]
                moved = [b for b in left if np.linalg.norm(self.pose(b)[0] - supply_pose0[b]) > 0.02]
                body = max(moved, key=lambda b: self.pose(b)[0][2]) if moved else None   # the highest: the placed box
                info["supply_boxes_moved"] = len(moved)                                    # (others were knocked)
                info["drop_call"] = ev if ev.startswith("drop") else None
            after = frames[-1]
            # perception: which slot was emptied
            fb, fa = supply.features(before), supply.features(after)
            emptied = [k for k in fb if fb[k]["dark"] < 0.5 <= fa[k]["dark"]]
            unsure = any(UNSURE[0] < v["dark"] < UNSURE[1] for v in list(fb.values()) + list(fa.values()))
            truth_slot = slot_of[body] if body is not None else None
            if unsure:
                info["asked"] = True; slot = truth_slot            # the player answers on screen
            else:
                slot = emptied[0] if len(emptied) == 1 else None
            info.update(slot_pred=slot, slot_true=truth_slot, slot_ok=slot == truth_slot)
            new_h = G.BOX[self.type_of[body_of[slot]]]["height"] if slot else 0.0
            dims = G.BOX[self.type_of[body_of[slot]]] if slot else None
            f = features(cam, ref, before, frames, z_top, new_h, arm_mask=home_mask, new_dims=dims,
                         arm_frames=self.arm_frames if player == "robot" else None)
            col, why = collapse_check(f)
            info["features"] = {k: v for k, v in f.items() if k != "top_xy"}
            if self.debug is not None:                     # frames + trace for diagnosing a turn
                tdir = self.debug / f"g{game_i:03d}_t{turn}"; tdir.mkdir(parents=True, exist_ok=True)
                for nm, im in (("ref", ref), ("before", before), ("after", after)):
                    cv2.imwrite(str(tdir / f"{nm}.png"), im)
                if player == "robot":
                    np.save(tdir / "skill_trace.npy", np.array(self.trace, np.float32))
            call_loss = bool(col or info["drop_call"] or slot is None)
            info["call"] = ("loss: " + (info["drop_call"] or ("no placement seen" if slot is None else ", ".join(why)))) if call_loss else "ok"
            t_loss, t_why = self.truth_turn(stack_bodies, body, old)
            if body is None:
                t_loss, t_why = True, "no placement (skill failed)"
            info.update(truth="loss: " + t_why if t_loss else "ok", call_ok=call_loss == t_loss)
            turns.append(info)
            print(f"  game {game_i} turn {turn} {player:5s} h{info['height']}: call {info['call']:40s} truth {info['truth']}"
                  f"{'' if info['call_ok'] else '  <-- MISMATCH'}", flush=True)
            if call_loss or t_loss:
                break
            stack_bodies.append(body); perceived_slots.append(slot)
        return dict(game=game_i, refit=rinfo, turns=turns, boxes_stacked=len(stack_bodies),
                    loser=turns[-1]["player"] if turns and turns[-1]["call"] != "ok" else None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--policy", required=True); ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--seed", type=int, default=12001); ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="data/sim_game/run"); ap.add_argument("--videos", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    policy = load_policy(a.policy, a.device, None, 0.01)
    sim = GT.Sim(); rand = Randomizer(sim.m); ren = mujoco.Renderer(sim.m, H, W)
    twin, kin = Twin(), Kinematics(); rng = np.random.default_rng(a.seed)
    games = []; t0 = time.time()
    for g in range(a.games):
        vw = cv2.VideoWriter(str(out / f"game_{g:03d}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)) if g < a.videos else None
        res = Game(sim, rand, ren, twin, kin, policy, rng, video=vw, debug=out / "turns").play(g)
        if vw is not None:
            vw.release()
        games.append(res); (out / "games.json").write_text(json.dumps(games, indent=1, default=str))
        print(f"game {g}: {res['boxes_stacked']} boxes stacked, loser {res['loser']}, {(time.time() - t0) / (g + 1) / 60:.1f} min/game", flush=True)
    turns = [t for g in games for t in g["turns"]]
    c = Counter((t["truth"] != "ok", t["call"] != "ok") for t in turns)
    lines = [f"games {len(games)}, turns {len(turns)} (robot {sum(t['player'] == 'robot' for t in turns)})",
             f"turn calls: correct {sum(t['call_ok'] for t in turns)}/{len(turns)}; losses caught {c[(True, True)]}/{c[(True, True)] + c[(True, False)]}; "
             f"false loss calls {c[(False, True)]}/{c[(False, True)] + c[(False, False)]}",
             f"slot check: correct {sum(t['slot_ok'] for t in turns)}/{len(turns)}, asked {sum(t.get('asked', False) for t in turns)}",
             f"robot skill events: {dict(Counter(t.get('skill_event') for t in turns if t['player'] == 'robot'))}",
             f"games whose outcome call matched the truth: {sum(all(t['call_ok'] for t in g['turns']) for g in games)}/{len(games)}"]
    txt = "\n".join(lines); print(txt); (out / "summary.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
