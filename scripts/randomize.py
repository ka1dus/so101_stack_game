"""Phase 5: per-episode visual randomization (CLAUDE.md Phase 5 + Section 3b).

Only rendering properties change (camera pose/intrinsics, lights, colours, box tints); physics is untouched.
`Randomizer.apply(rng)` sets a new draw and returns its parameters (stored in episode metadata);
`reset()` restores the nominal scene. Image-space effects (wrist lens distortion, the front corner mask)
are applied by `post_front()` / `post_wrist()` on the rendered frames.

Ranges:
  front camera   position +-1 cm per axis, rotation <= 1.5 deg about a random axis
  wrist camera   position +-3 mm per axis, rotation <= 1 deg, f (cameras.yaml, 508 px) +-5 %, k1 in [-0.05, 0.05]; blur sigma 0.6-2.5 px
                 (no checkerboard calibration: Decisions log 2026-10-21)
  lights         ceiling/fill/headlight intensity, ceiling position +-20 cm
  colours        white table, black mat/backdrop, yellow tape, reach-line tape, floor: around the real values
  boxes          per type: brightness +-15 %, per-channel tint +-6 %; 5 % of episodes plain colour (no print)
Front frames: real front frames are undistorted before the policy (Phase 3 proposal), which leaves black
corners; the same mask is applied to sim front frames so both look alike.
"""
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CAMS = yaml.safe_load((ROOT / "sim/assets/cameras.yaml").read_text())
W, H = 640, 480
BOX_MATS = ("mat_no3", "mat_no35", "mat_no10")


def _small_rot_quat(rng, max_deg):
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, axis, np.radians(rng.uniform(0, max_deg)))
    return q


def _K(f):
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])


class Randomizer:
    def __init__(self, m):
        self.m = m
        self.cam = {n: m.camera(n).id for n in ("front", "wrist")}
        self.geom = {n: m.geom(n).id for n in ("table_top", "supply_mat", "stack_area", "reach_line_tape", "backdrop",
                                               "wall_lower", "wall_upper")}
        self.mats = {n: m.material(n).id for n in BOX_MATS}
        self.carpet = m.material("carpet").id
        self.arm_mats = [i for i in range(m.nmat) if np.allclose(m.mat_rgba[i, :3], [0.07, 0.07, 0.08], atol=0.01)]
        self.nominal = dict(cam_pos=m.cam_pos.copy(), cam_quat=m.cam_quat.copy(), cam_fovy=m.cam_fovy.copy(),
                            light_pos=m.light_pos.copy(), light_diffuse=m.light_diffuse.copy(), light_castshadow=m.light_castshadow.copy(),
                            head_amb=m.vis.headlight.ambient.copy(), head_dif=m.vis.headlight.diffuse.copy(),
                            geom_rgba=m.geom_rgba.copy(), geom_pos=m.geom_pos.copy(), mat_rgba=m.mat_rgba.copy(), mat_texid=m.mat_texid.copy())
        # front: black corners left by undistorting the real frame
        fc = CAMS["front"]
        self.front_mask = cv2.undistort(np.full((H, W), 255, np.uint8), _K(fc["f_px"]),
                                        np.array([fc["k1"], fc.get("k2", 0.0), 0, 0]))[..., None] > 127
        self.wrist_maps = None; self.img = None; self.noise_rng = np.random.default_rng(0)

    def reset(self):
        m, n = self.m, self.nominal
        m.cam_pos[:] = n["cam_pos"]; m.cam_quat[:] = n["cam_quat"]; m.cam_fovy[:] = n["cam_fovy"]
        m.light_pos[:] = n["light_pos"]; m.light_diffuse[:] = n["light_diffuse"]; m.light_castshadow[:] = n["light_castshadow"]
        m.vis.headlight.ambient[:] = n["head_amb"]; m.vis.headlight.diffuse[:] = n["head_dif"]
        m.geom_rgba[:] = n["geom_rgba"]; m.geom_pos[:] = n["geom_pos"]; m.mat_rgba[:] = n["mat_rgba"]; m.mat_texid[:] = n["mat_texid"]
        self.wrist_maps = None; self.img = None

    def apply(self, rng):
        self.reset()
        m, n = self.m, self.nominal; p = {}
        # cameras
        for name, dpos, drot in (("front", 0.010, 1.5), ("wrist", 0.003, 1.0)):
            i = self.cam[name]; dp = rng.uniform(-dpos, dpos, 3); dq = _small_rot_quat(rng, drot)
            m.cam_pos[i] = n["cam_pos"][i] + dp
            q = np.zeros(4); mujoco.mju_mulQuat(q, n["cam_quat"][i], dq); m.cam_quat[i] = q
            p[f"{name}_dpos_mm"] = (dp * 1000).round(2).tolist(); p[f"{name}_dquat"] = dq.round(5).tolist()
        f = CAMS["wrist"]["f_px"] * rng.uniform(0.95, 1.05); k1 = rng.uniform(-0.05, 0.05)
        m.cam_fovy[self.cam["wrist"]] = np.degrees(2 * np.arctan(H / 2 / f))
        p["wrist_f_px"] = round(float(f), 1); p["wrist_k1"] = round(float(k1), 4)
        self._wrist_distortion(f, k1)
        # lights
        s_ceil, s_fill, s_head = rng.uniform(0.75, 1.25), rng.uniform(0.5, 1.5), rng.uniform(0.6, 1.4)
        m.light_diffuse[0] = n["light_diffuse"][0] * s_ceil; m.light_diffuse[1] = n["light_diffuse"][1] * s_fill
        m.light_pos[0] = n["light_pos"][0] + np.r_[rng.uniform(-0.2, 0.2, 2), 0]
        m.vis.headlight.ambient[:] = n["head_amb"] * s_head; m.vis.headlight.diffuse[:] = n["head_dif"] * s_head
        shadows = bool(rng.random() < 0.5)                   # real room light is diffuse: hard shadows only half the time
        m.light_castshadow[:] = n["light_castshadow"] & shadows; p["shadows"] = shadows
        p["light_scale"] = [round(float(s_ceil), 3), round(float(s_fill), 3), round(float(s_head), 3)]
        # surface colours around the real values
        g = self.geom; rgba = m.geom_rgba
        rgba[g["table_top"], :3] = np.clip(n["geom_rgba"][g["table_top"], :3] + rng.uniform(-0.06, 0.03) + rng.uniform(-0.02, 0.02, 3), 0, 1)
        for k in ("supply_mat", "backdrop"):
            rgba[g[k], :3] = rng.uniform(0.01, 0.07) + rng.uniform(-0.005, 0.005, 3)
        rgba[g["stack_area"], :3] = np.clip(n["geom_rgba"][g["stack_area"], :3] * rng.uniform(0.85, 1.1) + rng.uniform(-0.05, 0.05, 3), 0, 1)
        rgba[g["reach_line_tape"], :3] = np.clip(n["geom_rgba"][g["reach_line_tape"], :3] + rng.uniform(-0.06, 0.04, 3), 0, 1)
        m.mat_rgba[self.carpet, :3] = np.clip(rng.uniform(0.6, 1.4) * (1 + rng.uniform(-0.1, 0.1, 3)), 0, 2)
        rgba[g["wall_lower"], :3] = np.clip(rng.uniform(0.7, 0.95) + rng.uniform(-0.04, 0.04, 3), 0, 1)
        rgba[g["wall_upper"], :3] = np.clip(n["geom_rgba"][g["wall_upper"], :3] * rng.uniform(0.5, 2.0) + rng.uniform(-0.03, 0.03, 3), 0, 1)
        # wall distance (APPROX in the scene) and lower-panel height
        look = n["geom_pos"][g["wall_lower"], :2] / np.linalg.norm(n["geom_pos"][g["wall_lower"], :2])
        dd = rng.uniform(-0.15, 0.25); dz = rng.uniform(-0.06, 0.08)
        for k in ("wall_lower", "wall_upper"):
            m.geom_pos[g[k], :2] = n["geom_pos"][g[k], :2] + look * dd; m.geom_pos[g[k], 2] = n["geom_pos"][g[k], 2] + dz
        p["wall_shift_m"] = [round(float(dd), 3), round(float(dz), 3)]
        # black printed arm
        arm = rng.uniform(0.03, 0.15)
        for i in self.arm_mats:
            m.mat_rgba[i, :3] = arm + rng.uniform(-0.01, 0.01, 3)
        p["arm_grey"] = round(float(arm), 3)
        # image-space augmentation (per camera): brightness, contrast, saturation, gamma, blur, noise
        self.img = {}
        for cam in ("front", "wrist"):
            # wrist: the real Gemini 2 is out of focus at close range (about sigma 2 px at 640x480, notes/phase8.md section 10)
            blurs = [0.6, 1.0, 1.5, 2.0, 2.5] if cam == "wrist" else [0, 0, 0.6, 1.0, 1.5]
            self.img[cam] = dict(bright=rng.uniform(-0.08, 0.08), contrast=rng.uniform(0.8, 1.1), sat=rng.uniform(0.6, 1.05),
                                 gamma=rng.uniform(0.85, 1.15), blur=float(rng.choice(blurs)),
                                 noise=rng.uniform(0, 0.02))
        p["image_aug"] = {c: {k: round(float(v), 3) for k, v in d.items()} for c, d in self.img.items()}
        # boxes: tint per type, sometimes a plain colour
        plain = bool(rng.random() < 0.05); p["box_plain"] = plain; p["box_tint"] = {}
        for name, i in self.mats.items():
            if plain:
                m.mat_texid[i, :] = -1; m.mat_rgba[i, :3] = rng.uniform(0.3, 0.95, 3)
            else:
                m.mat_rgba[i, :3] = np.clip(rng.uniform(0.85, 1.15) * (1 + rng.uniform(-0.06, 0.06, 3)), 0, 1)
            p["box_tint"][name] = m.mat_rgba[i, :3].round(3).tolist()
        return p

    def _wrist_distortion(self, f, k1):
        """remap grids so a pinhole render looks like a lens with radial k1 (output pixel -> pinhole source)."""
        K = _K(f); u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        pts = np.stack([u.ravel(), v.ravel()], -1)[:, None, :]
        und = cv2.undistortPoints(pts, K, np.array([k1, 0, 0, 0]), P=K).reshape(H, W, 2)
        self.wrist_maps = (und[..., 0].astype(np.float32), und[..., 1].astype(np.float32))

    def _augment(self, img, cam):
        """uint8 pipeline: saturation (blend with grey), contrast/brightness/gamma (one LUT), blur, noise."""
        if not self.img:
            return img
        a = self.img[cam]
        if "lut" not in a:
            x = np.arange(256, dtype=np.float32) / 255
            a["lut"] = (np.clip(np.clip((x - 0.5) * a["contrast"] + 0.5 + a["bright"], 0, 1) ** a["gamma"], 0, 1) * 255).astype(np.uint8)
        grey = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        x = cv2.addWeighted(img, a["sat"], grey, 1 - a["sat"], 0)
        x = cv2.LUT(x, a["lut"])
        if a["blur"] > 0:
            x = cv2.GaussianBlur(x, (0, 0), a["blur"])
        if a["noise"] > 0:
            nz = np.empty(x.shape, np.int16); cv2.randn(nz, 0, a["noise"] * 255)
            x = np.clip(x.astype(np.int16) + nz, 0, 255).astype(np.uint8)
        return x

    def post_front(self, img):
        return self._augment(img, "front") * self.front_mask.astype(np.uint8)

    def post_wrist(self, img):
        if self.wrist_maps is not None:
            img = cv2.remap(img, *self.wrist_maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return self._augment(img, "wrist")
