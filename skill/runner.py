"""run_place_skill(): the ACT place skill on the real SO-101, in our own 30 Hz loop (CLAUDE.md Section 4).

Per tick: read the joints and both cameras (LeRobot's SO101Follower + OpenCV cameras, RGB), undistort `front` the way
the sim's corner mask expects (sim/assets/cameras.yaml), resize both to the policy's 320x240, run the policy
(temporal ensembling 0.01, as in every sim evaluation), clip the target to +-MAX_REL of the reading and to the
calibrated ranges, check it with the twin's kinematics (SAFETY), then send it. The skill ends as in sim_game.py:
  released  drop_check saw the jaws open in the release zone above the stack, and the tips then rose RAISE_M or
            RELEASE_HOLD_S passed (the policy's own straight-up retract)
  drop      drop_check saw the box leave the jaws away from the stack
  guard     a target failed the safety check (the arm holds where it is)
  timeout   SKILL_T_MAX
then the controller sends the arm home with go_home() (lifting the tips first unless the box was released).
move=False runs the whole loop with the arm holding still (timing and image checks without motion).
"""
import time

import cv2
import numpy as np
import torch
import yaml

from perception.camera import ROOT
from perception.drop import DropMonitor

import sys
sys.path.insert(0, str(ROOT / "scripts"))
from joint_mapping import JOINT_NAMES, JointMapping  # noqa: E402

FPS = 30
MAX_REL = 10.0             # LeRobot units per tick (training clip and the arm's max_relative_target)
SKILL_T_MAX = 25.0
RAISE_M = 0.035
RELEASE_HOLD_S = 1.5
HOME_SPEED = 30.0          # LeRobot units per second for scripted moves (so101_go_home.py default)
LIFT_M = 0.06              # tips raised straight up before going home after a drop or a timeout
RESIZE = (320, 240)
TASK = "Pick up the next box and place it on top of the stack"
CAMS = yaml.safe_load((ROOT / "sim/assets/cameras.yaml").read_text())
# safety envelope for the commanded jaw tips (m, arm frame); the joint mapping is good to ~1 cm at the tips
TIP_Z_MIN = -0.012
TIP_Z_MAX = 0.35
TABLE = np.array(list(yaml.safe_load((ROOT / "sim/assets/layout.yaml").read_text())["table"]["corners"].values()), float)
TABLE_PAD = 0.05


_UNDIST = {}


def undistort_front(img):
    """Same result as cv2.undistort with the calibrated f/k1 (scripts/phase7_real_refit.py), from cached maps
    (6 -> ~1 ms per frame)."""
    h, w = img.shape[:2]
    if (h, w) not in _UNDIST:
        c = CAMS["front"]; K = np.array([[c["f_px"], 0, w / 2], [0, c["f_px"], h / 2], [0, 0, 1]])
        _UNDIST[(h, w)] = cv2.initUndistortRectifyMap(K, np.array([c["k1"], c.get("k2", 0.0), 0, 0]), None, K, (w, h), cv2.CV_16SC2)
    return cv2.remap(img, *_UNDIST[(h, w)], cv2.INTER_LINEAR)


def _inside_table(xy, pad=TABLE_PAD):
    c = TABLE.mean(0); T = TABLE[np.argsort(np.arctan2(*(TABLE - c).T[::-1]))]
    for a, b in zip(T, np.roll(T, -1, 0)):
        e = b - a; n = np.array([e[1], -e[0]]) / np.linalg.norm(e)
        if np.dot(n, c - a) < 0:
            n = -n
        if np.dot(n, xy - a) < -pad:
            return False
    return True


def safety_check(kin, q):
    """None if the target q (LeRobot units) keeps the jaw tips in the envelope, else the reason."""
    tip = kin.tip_pos(q)
    if tip[2] < TIP_Z_MIN:
        return f"tips {tip[2] * 1000:.0f} mm below the table"
    if tip[2] > TIP_Z_MAX:
        return f"tips {tip[2] * 100:.0f} cm up"
    if not _inside_table(tip[:2]):
        return "tips outside the table"
    return None


class RealArm:
    """SO101Follower with the `front` and `wrist` cameras at the recording settings (640x480, 30 fps)."""

    def __init__(self, port, robot_id, front=0, wrist=1):
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        cams = {"front": OpenCVCameraConfig(index_or_path=front, fps=FPS, width=640, height=480),
                "wrist": OpenCVCameraConfig(index_or_path=wrist, fps=FPS, width=640, height=480)}
        self.robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, max_relative_target=MAX_REL,
                                                       disable_torque_on_disconnect=False, cameras=cams))
        self.jm = JointMapping.from_files(); self.lo, self.hi = self.jm.lerobot_limits().T

    def connect(self):
        self.robot.connect()

    def disconnect(self):
        self.robot.disconnect()

    def read(self):
        o = self.robot.get_observation()
        return np.array([o[f"{j}.pos"] for j in JOINT_NAMES], float), o["front"], o["wrist"]

    def joints(self):
        return self.read()[0]

    def send(self, q):
        self.robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINT_NAMES, q)})

    def hold(self):
        self.send(self.joints())

    def torque_off(self):
        self.robot.bus.disable_torque()

    def move_to(self, goal, kin=None, speed=HOME_SPEED, paused=None):
        return move_to(self, goal, kin, speed, paused)


def move_to(arm, goal, kin=None, speed=HOME_SPEED, paused=None):
    """Straight joint-space move at `speed` units/s (so101_go_home.py style); holds while paused() is True. Stops at a
    target that fails the safety check. Returns None or the reason it stopped."""
    q = arm.joints(); goal = np.asarray(goal, float); step = speed / FPS; virtual = getattr(arm, "virtual", False)
    while np.abs(goal - q).max() > 0.3:
        t0 = time.perf_counter()
        if paused is not None and paused():
            arm.send(q)                                     # hold at the current target (not the reading: no sag)
        else:
            q = q + np.clip(goal - q, -step, step)
            if kin is not None and (why := safety_check(kin, q)):
                arm.hold(); return why
            arm.send(q)
        if not virtual:
            time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - t0)))
    if not virtual:
        time.sleep(0.3)
    return None


def lift_target(kin, q, dz=LIFT_M, iters=30):
    """Joint target with the jaw tips dz straight above where they are now (shoulder_lift, elbow_flex, wrist_flex;
    damped least squares on the twin's kinematics). None if it does not converge to 5 mm."""
    q = np.asarray(q, float).copy(); goal = kin.tip_pos(q) + [0, 0, dz]; idx = [1, 2, 3]
    for _ in range(iters):
        p = kin.tip_pos(q); err = goal - p
        if np.linalg.norm(err) < 0.002:
            return q
        J = np.zeros((3, 3))
        for c, i in enumerate(idx):
            dq = np.zeros(6); dq[i] = 0.5; J[:, c] = (kin.tip_pos(q + dq) - p) / 0.5
        q[idx] += np.clip(J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), err), -5, 5)
    return q if np.linalg.norm(goal - kin.tip_pos(q)) < 0.005 else None


def go_home(arm, kin, home, lift=False, paused=None):
    """Scripted return home; lift=True first raises the tips straight up (the jaws may still be next to the stack)."""
    if lift:
        q_up = lift_target(kin, arm.joints())
        if q_up is not None:
            move_to(arm, q_up, kin, speed=HOME_SPEED / 2, paused=paused)
    return move_to(arm, home, kin, paused=paused)


def load_policy(path, device="mps"):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    cfg = PreTrainedConfig.from_pretrained(path); cfg.temporal_ensemble_coeff = 0.01; cfg.n_action_steps = 1
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg); policy.to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=path,
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


def run_place_skill(arm, policy, kin, top_xy, top_face, move=True, t_max=SKILL_T_MAX, record=None, on_tick=None, paused=None):
    """Run the skill from home. top_xy / top_face: the stack top from perception (m). record: list to append
    (front_undistorted_bgr, wrist_bgr) frames to. on_tick(front_bgr, q): called every tick with the undistorted front
    frame and the reading (hand check, live view). paused(): while it returns True the arm holds where it is and the
    policy is not run (safety pause, e.g. a hand in the workspace); on resume the policy restarts from the current
    observation (its action queue / ensemble is reset). The arm may be a sim stand-in: `pinhole_front` (its front
    frame needs no undistortion) and `virtual` (send() advances the physics; no real-time sleep).
    Returns dict(event, duration_s (active time), ticks, pauses, paused_s, loop_hz, release_tick (index into
    trace/record: the reading on which the release was seen), compute times, trace: per tick reading, target,
    compute time)."""
    pol, pre, post = policy; pol.reset(); pre.reset(); post.reset()
    pinhole, virtual = getattr(arm, "pinhole_front", False), getattr(arm, "virtual", False)
    mon = DropMonitor(kin, top_xy, top_face); rel_z = rel_k = None; event = "timeout"; trace = []; g_cmd = None; dts = []
    was_paused = False; pauses = 0; paused_ticks = 0
    while len(dts) < int(t_max * FPS):
        t0 = time.perf_counter()
        q, front, wrist = arm.read()
        fr = front if pinhole else undistort_front(front)
        if on_tick is not None:
            on_tick(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), q)
        if paused is not None and paused():
            if not was_paused:
                held = q.copy(); arm.send(held); was_paused = True; pauses += 1   # stop where it is now
            elif virtual:
                arm.send(held)                              # sim: time only passes when a command is sent
            paused_ticks += 1
            if not virtual:
                time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - t0)))
            continue
        if was_paused:                                      # resume: plan afresh from what the cameras see now
            pol.reset(); pre.reset(); post.reset(); was_paused = False
        if g_cmd is not None:                               # drop_check on this reading and the last command
            ev = mon.tick(q, g_cmd)
            if ev and ev.startswith("drop"):
                event = ev; break
            if ev == "released":
                tip = kin.tip_pos(q)
                if rel_z is None:
                    rel_z, rel_k = tip[2], len(trace)
                if tip[2] > rel_z + RAISE_M or len(trace) - rel_k >= RELEASE_HOLD_S * FPS:
                    event = "released"; break
        f = cv2.resize(fr, RESIZE, interpolation=cv2.INTER_AREA); w = cv2.resize(wrist, RESIZE, interpolation=cv2.INTER_AREA)
        obs = {"observation.state": torch.from_numpy(q.astype(np.float32))[None],
               "observation.images.front": torch.from_numpy(f).permute(2, 0, 1)[None].float() / 255,
               "observation.images.wrist": torch.from_numpy(w).permute(2, 0, 1)[None].float() / 255, "task": [TASK]}
        with torch.inference_mode():
            a = post(pol.select_action(pre(obs))).squeeze(0).float().cpu().numpy()
        a = np.clip(np.clip(a, q - MAX_REL, q + MAX_REL), arm.lo, arm.hi)
        why = safety_check(kin, a)
        if why:
            event = "guard: " + why; arm.hold(); break
        if move:
            arm.send(a)
        g_cmd = float(a[5]); trace.append(np.r_[q, a, time.perf_counter() - t0])
        if record is not None:
            record.append((cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)))
        dt = time.perf_counter() - t0; dts.append(dt)
        if not virtual:
            time.sleep(max(0.0, 1 / FPS - dt))
    hz = 1 / np.mean([max(d, 1 / FPS) for d in dts]) if dts else 0.0
    return dict(event=event, duration_s=round(len(dts) / FPS, 2), ticks=len(dts), pauses=pauses,
                paused_s=round(paused_ticks / FPS, 2), loop_hz=round(float(hz), 1), release_tick=rel_k,
                compute_ms_p50=round(float(np.median(dts) * 1000), 1) if dts else None,
                compute_ms_p95=round(float(np.percentile(dts, 95) * 1000), 1) if dts else None,
                trace=np.array(trace, np.float32))
