"""Phase 6: evaluate a trained policy (ACT or SmolVLA) in the MuJoCo twin, closed loop at 30 Hz.

Runs in the eval venv (MuJoCo + LeRobot): ~/.venvs/so101_eval (created 2026-10-22; numpy/opencv follow LeRobot's caps).

Per episode: a game state from game_states.sample_state (own seed, so not a training state), the arm at home (+-3 deg),
the policy drives the arm from the rendered `front`/`wrist` frames and the joint state (LeRobot units). Like the
real arm (max_relative_target=10), each target is clipped to within 10 units of the present position.
Judged with privileged state, CLAUDE.md success = picked box on top of the stack AND the stack standing 3 s after release.

Failure types (Phase 6): missed_grasp, drop, knocked_approach, knocked_release, misplaced, timeout_holding;
`wrong_box` is a separate flag (ACT: not the deterministic rule's box; VLA: not the requested size).
Breakdown by stack height, crookedness (max per-level offset), picked type and pairing (picked_on_top).

    MUJOCO_GL=cgl ~/.venvs/so101_eval/bin/python scripts/sim_eval.py --policy <checkpoint>/pretrained_model --n 100 [--mode vla]
"""
import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import expert as E  # noqa: E402
import game_states as G  # noqa: E402
import grasp_test as GT  # noqa: E402
from generate import SIZE, TASK_ACT, TASK_VLA  # noqa: E402
from joint_mapping import JOINT_NAMES, JointMapping  # noqa: E402
from randomize import H, W, Randomizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAX_REL = 10.0           # LeRobot units per step (the real arm's max_relative_target)
T_MAX = 25.0             # s; expert episodes are 11-16 s
SETTLE = 3.0
OFFSET_BINS = (5, 10, 20)  # mm, crookedness buckets


def load_policy(path, device, n_action_steps=None, temporal_ensemble=None):
    """Inference-time re-planning (ACT, configuration_act.py:85-146): n_action_steps = steps executed per
    prediction (training default 100 = 3.3 s open loop); temporal_ensemble = coefficient for per-step re-planning
    with exponentially weighted averaging of overlapping chunks (forces n_action_steps=1)."""
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    cfg = PreTrainedConfig.from_pretrained(path)
    if n_action_steps is not None:
        cfg.n_action_steps = n_action_steps
    if temporal_ensemble is not None:
        cfg.temporal_ensemble_coeff = temporal_ensemble; cfg.n_action_steps = 1
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=path,
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


class Judge:
    """Privileged bookkeeping during one policy episode."""

    def __init__(self, sim, st):
        m, d = sim.m, sim.d
        self.sim, self.st = sim, st
        self.bid = {b.body: m.body(b.body).id for b in st.supply + st.stack}
        self.z0 = {k: d.xpos[i][2] for k, i in self.bid.items()}
        self.stack0 = {b.body: d.xpos[self.bid[b.body]].copy() for b in st.stack}
        self.grip_geoms = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in ("gripper", "moving_jaw_so101_v1")}
        self.picked = None; self.released_at = None; self.free_ticks = 0; self.moved_before_release = 0.0
        self.backdrop = {g for g in range(m.ngeom) if m.geom(g).name.startswith("backdrop")}; self.hit_backdrop = False

    def touching_gripper(self, body):
        m, d = self.sim.m, self.sim.d; b = self.bid[body]
        for c in d.contact[:d.ncon]:
            g1, g2 = c.geom1, c.geom2
            if (m.geom_bodyid[g1] == b and g2 in self.grip_geoms) or (m.geom_bodyid[g2] == b and g1 in self.grip_geoms):
                return True
        return False

    def stack_motion(self):
        d = self.sim.d
        moved = max((np.linalg.norm(d.xpos[self.bid[k]] - p) for k, p in self.stack0.items()), default=0.0)
        tilt = max((np.degrees(np.arccos(np.clip(d.xmat[self.bid[k]].reshape(3, 3)[2, 2], -1, 1))) for k in self.stack0), default=0.0)
        return moved, tilt

    def tick(self, t):
        d, m = self.sim.d, self.sim.m
        for c in d.contact[:d.ncon]:                    # safety flag: any arm part touching the backdrop
            g = c.geom2 if c.geom1 in self.backdrop else c.geom1 if c.geom2 in self.backdrop else None
            if g is not None and m.geom_bodyid[g] != 0 and not m.body(m.geom_bodyid[g]).name.startswith("box"):
                self.hit_backdrop = True
        if self.picked is None:
            for b in self.st.supply:
                if d.xpos[self.bid[b.body]][2] > self.z0[b.body] + 0.02:
                    self.picked = b; break
        elif self.released_at is None:
            moved, tilt = self.stack_motion()
            self.moved_before_release = max(self.moved_before_release, moved if tilt < 5 else 1.0)
            v = np.linalg.norm(d.cvel[self.bid[self.picked.body]][3:])
            self.free_ticks = self.free_ticks + 1 if (not self.touching_gripper(self.picked.body) and v < 0.01) else 0
            if self.free_ticks >= 10:
                self.released_at = t

    def verdict(self, mode, requested):
        st, d = self.st, self.sim.d
        out = dict(picked_type=self.picked.type if self.picked else None, wrong_box=False, hit_backdrop=self.hit_backdrop)
        if self.picked is not None:
            exp = st.supply[st.pick]
            out["wrong_box"] = (self.picked.body != exp.body) if mode == "act" else (self.picked.type != requested)
        moved, tilt = self.stack_motion()
        standing = moved < 0.005 and tilt < 5
        if self.picked is None:
            return dict(out, ok=False, reason="missed_grasp")
        if self.released_at is None:
            return dict(out, ok=False, reason="timeout_holding")
        top = st.stack[-1]; tid = self.bid[top.body]; nb = d.xpos[self.bid[self.picked.body]]
        exp_z = d.xpos[tid][2] + G.BOX[top.type]["height"] / 2 + G.BOX[self.picked.type]["height"] / 2
        Rt = d.xmat[tid].reshape(3, 3); rel = Rt.T @ (nb - d.xpos[tid])
        over = abs(rel[0]) < G.BOX[top.type]["length"] / 2 and abs(rel[1]) < G.BOX[top.type]["width"] / 2
        nb_tilt = np.degrees(np.arccos(np.clip(d.xmat[self.bid[self.picked.body]].reshape(3, 3)[2, 2], -1, 1)))
        on_top = abs(nb[2] - exp_z) < 0.004 and over and nb_tilt < 5
        out.update(stack_moved_mm=float(moved * 1000), place_offset_mm=float(np.linalg.norm(rel[:2]) * 1000))
        if on_top and standing:
            return dict(out, ok=True, reason="ok")
        if not standing:
            return dict(out, ok=False, reason="knocked_approach" if self.moved_before_release >= 0.005 else "knocked_release")
        if nb[2] < self.z0[self.picked.body] + 0.01 and not over:
            return dict(out, ok=False, reason="drop")
        return dict(out, ok=False, reason="misplaced")


def run_episode(sim, st, policy, pre, post, jm, rand, renderer, task, device, rng, video=None, resize=None):
    m, d = sim.m, sim.d
    arm_q = list(sim.qadr) + [sim.gadr]; act_ids = [sim.act[j] for j in JOINT_NAMES]
    lo, hi = m.actuator_ctrlrange[act_ids, 0], m.actuator_ctrlrange[act_ids, 1]
    q0 = sim.home.copy(); q0[:5] += np.radians(rng.uniform(-3, 3, 5))
    q0[:5] = np.clip(q0[:5], sim.rng_lim[:, 0] + np.radians(0.5), sim.rng_lim[:, 1] - np.radians(0.5))
    d.qpos[sim.qadr] = q0[:5]; d.qpos[sim.gadr] = q0[5]; d.ctrl[act_ids] = q0; d.qvel[:] = 0; mujoco.mj_forward(m, d)
    judge = Judge(sim, st); policy.reset(); pre.reset(); post.reset()
    n_ticks = int(T_MAX * GT.CTRL_HZ); post_release = int(2.0 * GT.CTRL_HZ); end = None
    for k in range(n_ticks):
        state = jm.sim_to_real(d.qpos[arm_q]).astype(np.float32)
        renderer.update_scene(d, "front"); front = rand.post_front(renderer.render())
        renderer.update_scene(d, "wrist"); wrist = rand.post_wrist(renderer.render())
        if video is not None:
            video.write(np.concatenate([front, wrist], 1))
        if resize:
            rw, rh = (int(x) for x in resize.split("x"))
            front = cv2.resize(front, (rw, rh), interpolation=cv2.INTER_AREA); wrist = cv2.resize(wrist, (rw, rh), interpolation=cv2.INTER_AREA)
        obs = {"observation.state": torch.from_numpy(state)[None],
               "observation.images.front": torch.from_numpy(front).permute(2, 0, 1)[None].float() / 255,
               "observation.images.wrist": torch.from_numpy(wrist).permute(2, 0, 1)[None].float() / 255,
               "task": [task]}
        with torch.inference_mode():
            a = post(policy.select_action(pre(obs)))
        a = a.squeeze(0).float().cpu().numpy()
        a = np.clip(a, state - MAX_REL, state + MAX_REL)
        d.ctrl[act_ids] = np.clip(jm.real_to_sim(a), lo, hi)
        for _ in range(sim.steps_per_ctrl):
            mujoco.mj_step(m, d)
        judge.tick(k / GT.CTRL_HZ)
        if judge.released_at is not None and end is None:
            end = k + post_release
        if end is not None and k >= end:
            break
    sim.sync_plan(); sim.wait(SETTLE)                  # hold the policy's last targets, then judge
    return judge, (k + 1) / GT.CTRL_HZ


def resting_drift(sim, st, secs=2.0):
    """Max displacement (m) of any stack box while everything rests (arm at home) for `secs`."""
    ids = [sim.m.body(b.body).id for b in st.stack]; p0 = [sim.d.xpos[i].copy() for i in ids]
    sim.sync_plan()        # hold the home targets just set by sample_state (else wait() replays the last episode's plan)
    sim.wait(secs)
    return max(float(np.linalg.norm(sim.d.xpos[i] - p)) for i, p in zip(ids, p0))


def max_offset_mm(st):
    return max((np.linalg.norm(o) for o in st.meta.get("offsets_mm", [])), default=0.0)


def wanted(st, a):
    """Targeted-evaluation filters (confirmation of the Section 1 criteria per box type / pairing)."""
    pick = st.supply[st.pick].type; top = st.stack[-1].type
    if a.want_type and pick != a.want_type:
        return False
    if a.want_pairing and f"{pick}_on_{top}" != a.want_pairing:
        return False
    return max_offset_mm(st) >= a.min_crooked_mm


def bucket_offsets(st):
    off = max((np.linalg.norm(o) for o in st.meta.get("offsets_mm", [])), default=0.0)
    for b in OFFSET_BINS:
        if off < b:
            return f"<{b}mm"
    return f">={OFFSET_BINS[-1]}mm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="path to .../checkpoints/<step>/pretrained_model")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--mode", choices=("act", "vla"), default="act")
    ap.add_argument("--seed", type=int, default=777, help="eval states; training used 100-2003")
    ap.add_argument("--randomize", action="store_true", help="randomized visuals (default: nominal calibrated scene)")
    ap.add_argument("--videos", type=int, default=5, help="save the first N episodes as mp4")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-action-steps", type=int, default=None, help="re-plan every K steps (default: the trained value, 100)")
    ap.add_argument("--temporal-ensemble", type=float, default=None, help="temporal ensembling coefficient (e.g. 0.01)")
    ap.add_argument("--height-min", type=int, default=None, help="targeted: stack heights height-min..height-max (uniform)")
    ap.add_argument("--height-max", type=int, default=None)
    ap.add_argument("--hard-p", type=float, default=0.5, help="share of hard (crooked/precarious) stacks")
    ap.add_argument("--min-crooked-mm", type=float, default=0.0, help="targeted: max per-level stack offset >= this")
    ap.add_argument("--want-type", default=None, help="targeted: only states where the robot's rule picks this type")
    ap.add_argument("--want-pairing", default=None, help="targeted: only states with this picked_on_top pairing")
    ap.add_argument("--states", choices=("v1", "game"), default="v1",
                    help="v1: the original sampler (the 200-state comparison set); game: game-rule supplies (fixed layout + rule gaps)")
    ap.add_argument("--servo", default=None, help="sim servo model yaml (e.g. sim/assets/servo_fit.yaml, the real-arm fit)")
    ap.add_argument("--servo-scale", type=float, default=1.0, help="scale kp and kv of --servo on every joint (robustness check)")
    ap.add_argument("--resize", default=None, help="WxH policy input size (must match the training dataset, e.g. 320x240)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    a = ap.parse_args()
    out = Path(a.out or ROOT / "data/eval" / f"{Path(a.policy).parent.name}_{a.mode}_seed{a.seed}"); out.mkdir(parents=True, exist_ok=True)
    policy, pre, post = load_policy(a.policy, a.device, a.n_action_steps, a.temporal_ensemble)
    print(f"inference: n_action_steps={policy.config.n_action_steps}, temporal_ensemble_coeff={policy.config.temporal_ensemble_coeff}")
    sim = GT.Sim(); rng = np.random.default_rng(a.seed); jm = JointMapping.from_files()
    if a.servo:
        import servo_model
        sp = {j: (kp * a.servo_scale, kv * a.servo_scale, f) for j, (kp, kv, f) in servo_model.load(a.servo).items()}
        servo_model.apply_params(sim.m, sp); print(f"servo model: {a.servo} x{a.servo_scale}")
    rand = Randomizer(sim.m); renderer = mujoco.Renderer(sim.m, H, W)
    rows = []; t0 = time.time(); n_drift = 0; n_filtered = 0
    height_p = None if a.height_min is None else {h: 1.0 for h in range(a.height_min, (a.height_max or G.MAX_STACK) + 1)}
    for i in range(a.n):
        requested = str(rng.choice(G.TYPE_ORDER)) if a.mode == "vla" else None
        for _ in range(3000):                          # targeted filters, then skip states whose resting stack creeps
            st, _ = G.sample_state(rng, sim.m, sim.d, mode=a.mode, requested=requested, home=E.home_fn(sim),
                                   supply_p=G.SUPPLY_P_EVAL_GAME if a.states == "game" else None,
                                   height_p=height_p, hard_p=a.hard_p)
            if not wanted(st, a):
                n_filtered += 1; continue
            if resting_drift(sim, st) < 0.0005:        # rare: ~1 in 40; not the policy's doing
                break
            n_drift += 1
        if a.randomize:
            rand.apply(rng)
        else:
            rand.reset()
        task = TASK_ACT if a.mode == "act" else TASK_VLA.format(SIZE[requested])
        vw = _wide_writer(out / f"ep_{i:03d}.mp4") if i < a.videos else None
        judge, dur = run_episode(sim, st, policy, pre, post, jm, rand, renderer, task, a.device, rng, vw, resize=a.resize)
        if vw is not None:
            vw.close()
        v = judge.verdict(a.mode, requested)
        row = dict(i=i, supply_mode=st.meta.get("supply_mode"), height=st.meta["stack_height"], max_offset_mm=round(max_offset_mm(st), 1), pairing=st.meta["pairing"], expected_type=st.supply[st.pick].type,
                   requested=requested, crooked=bucket_offsets(st), hard=st.meta.get("hard"), duration_s=round(dur, 2), **v)
        rows.append(row)
        print(f"{i:3d} h={row['height']} {row['pairing']:14s} {row['crooked']:7s} -> {v['reason']:16s}"
              f"{' WRONG_BOX' if v['wrong_box'] else ''}{' HIT_BACKDROP' if v['hit_backdrop'] else ''} ({dur:.1f} s) | {sum(r['ok'] for r in rows)}/{len(rows)} ok,"
              f" {(time.time() - t0) / len(rows):.0f} s/ep", flush=True)
    (out / "episodes.json").write_text(json.dumps(rows, indent=1))
    summary = summarize(rows, a) + f"eval states skipped for resting-stack drift > 0.5 mm in 2 s: {n_drift}\n" \
        + f"states rejected by targeted filters: {n_filtered}\n"
    (out / "summary.txt").write_text(summary); print(summary)


def _wide_writer(path):
    class Wide:
        def __init__(self):
            self.p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                       "-s", f"{2 * W}x{H}", "-r", str(GT.CTRL_HZ), "-i", "-", "-c:v", "libx264", "-crf", "23",
                                       "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)

        def write(self, img):
            self.p.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())

        def close(self):
            self.p.stdin.close(); self.p.wait()
    return Wide()


def summarize(rows, a):
    lines = [f"policy {a.policy}", f"mode {a.mode}, seed {a.seed}, {'randomized' if a.randomize else 'nominal'} visuals, n={len(rows)}",
             f"test states: {getattr(a, 'states', 'v1')} | policy input: {a.resize or '640x480'} | inference: n_action_steps={1 if a.temporal_ensemble else (a.n_action_steps or 'trained default (100)')}, temporal_ensemble={a.temporal_ensemble}",
             f"success {sum(r['ok'] for r in rows)}/{len(rows)} ({100 * np.mean([r['ok'] for r in rows]):.1f} %)",
             "failures: " + json.dumps(dict(sorted(Counter(r["reason"] for r in rows if not r["ok"]).items()))),
             f"arm touched the backdrop (safety flag): {sum(r['hit_backdrop'] for r in rows)}/{len(rows)}",
             f"wrong box picked: {sum(r['wrong_box'] for r in rows)}/{len(rows)}" + (" (instruction accuracy "
             f"{100 * (1 - np.mean([r['wrong_box'] for r in rows])):.1f} %)" if a.mode == "vla" else "")]
    for name, key in (("stack height", lambda r: r["height"]), ("height group", lambda r: "0-3" if r["height"] <= 3 else "4+"),
                      ("crookedness", lambda r: r["crooked"]), ("picked type (expected)", lambda r: r["expected_type"]),
                      ("pairing", lambda r: r["pairing"]), ("height 4+ and crooked >=10mm",
                                                            lambda r: "yes" if r["height"] >= 4 and r["crooked"] in ("<20mm", ">=20mm") else "no")):
        b = defaultdict(list)
        for r in rows:
            b[key(r)].append(r["ok"])
        lines.append(f"by {name}: " + ", ".join(f"{k} {sum(v)}/{len(v)} ({100 * np.mean(v):.0f}%)" for k, v in sorted(b.items(), key=lambda kv: str(kv[0]))))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
