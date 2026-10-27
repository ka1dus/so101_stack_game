"""hand_in_workspace(): is a hand (or anything else that moves, other than the arm) inside the workspace?

On the `front` camera, frame by frame:
  workspace   the table top (grown 2 cm) up to 30 cm, projected through the calibrated camera
  arm mask    the twin's arm at the current joint readings, dilated (decision D3), plus a zone around the jaws for
              the box being carried
  foreground  changed versus a background frame (shadow-like changes ignored), inside the workspace, outside the mask
  motion      changed versus the frame 2 ticks earlier
A hand is called when a foreground blob that is also moving is larger than AREA_PX inside the workspace and reaches
out across a human-side table edge (ENTRY_PX in the entry band: a hand comes with a forearm, the arm's shadow does
not) or is not uniformly darker than the background (a shadow is), for HIT_FRAMES frames in a row.
It stays called while a foreground blob remains where it was (a hand held still), and is released when that
region is back to the background, or after LATCH_S without motion. Pixels that stay changed but still for ABSORB_S
(a placed box, a settled stack) are taken into the background; the background is frozen while a hand is suspected.
Motion is only measured where both frames were outside the arm mask, so the gripper moving away from a box it just
placed is not motion.
"""
import cv2
import numpy as np

from .camera import Camera, H, W
from .stack import _fill, _grow, change
from .twin import arm_mask

import yaml
from .camera import ROOT

TABLE = np.array(list(yaml.safe_load((ROOT / "sim/assets/layout.yaml").read_text())["table"]["corners"].values()), float)
WS_TOP = 0.30
AREA_PX = 400          # a hand at the far edge of the table is ~2,000 px; a finger tip ~300
HIT_FRAMES = 2
ABSORB_S = 1.0
FPS = 30
ARM_MARGIN_PX = 8
MASK_REUSE_UNITS = 0.1 # re-render the arm mask only when a joint reading moved more than this (LeRobot units)
SHADE_CHROMA = 0.05    # looser than the stack checks: the moving arm casts hard shadows in sim
SHADOW_DARKER = 0.85   # a blob darker than the background on >= this share of its pixels could be the arm's shadow
ENTRY_M, ENTRY_PX = 0.15, 40   # entry band: up to 15 cm outside the two human-side table edges
JAW_ZONE_M = 0.07      # radius around the grip point: a carried box, or supply boxes the closing jaws nudge
LATCH_S = 4.0          # a called hand with no motion for this long is released (its region taken into the background)


def entry_band(cam: Camera, top=WS_TOP):
    """Image region just outside the two table edges at the far corner (the human's side), up to `top`."""
    C = {k: np.array(v, float) for k, v in yaml.safe_load((ROOT / "sim/assets/layout.yaml").read_text())["table"]["corners"].items()}
    c = TABLE.mean(0); m = np.zeros((H, W), bool)
    for a, b in ((C["far"], C["left"]), (C["far"], C["right"])):
        n = (a + b) / 2 - c; n /= np.linalg.norm(n)
        P = np.array([a, b, b + n * ENTRY_M, a + n * ENTRY_M])
        m |= _fill((H, W), cam.project(np.vstack([np.c_[P, np.zeros(4)], np.c_[P, np.full(4, top)]])))
    return m & ~workspace_mask(cam)


def workspace_mask(cam: Camera, pad=0.02, top=WS_TOP):
    c = TABLE.mean(0); order = np.argsort(np.arctan2(*(TABLE - c).T[::-1])); P = _grow(TABLE[order], pad)
    return _fill((H, W), cam.project(np.vstack([np.c_[P, np.zeros(4)], np.c_[P, np.full(4, top)]])))


class HandMonitor:
    def __init__(self, cam: Camera, twin, kin=None):
        self.cam, self.twin, self.kin = cam, twin, kin
        self.ws = workspace_mask(cam); self.band = entry_band(cam)
        self.bg = None; self.hist = []; self.still = np.zeros((H, W), np.int32)
        self.hits = 0; self.present = False; self.track = None; self.valid_hist = []; self.idle = 0
        self.mask_q = None; self.mask_cache = None

    def rebase(self, frame):
        """Take `frame` as the empty-workspace background and drop any called hand (the player has confirmed the
        workspace is clear and nothing moved for a while, e.g. a latched placed-box region or an exposure shift)."""
        self.bg = None                                    # the next tick takes its frame as the background
        self.still[:] = 0; self.hits = 0; self.present = False; self.track = None; self.idle = 0

    def mask(self, q_real):
        q = np.asarray(q_real, float)
        if self.mask_q is not None and np.abs(q - self.mask_q).max() < MASK_REUSE_UNITS:
            return self.mask_cache                        # arm still (e.g. frozen during the human turn): reuse
        m = arm_mask(self.twin, self.cam, q_real, ARM_MARGIN_PX)
        if self.kin is not None:                          # carried-box zone around the grip point
            tip = self.kin.tip_pos(q_real); c = tip + np.r_[0, 0, 0.01]
            a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            ring = np.vstack([np.c_[c[0] + JAW_ZONE_M * np.cos(a), c[1] + JAW_ZONE_M * np.sin(a), np.full(16, c[2] + dz)]
                              for dz in (-JAW_ZONE_M, JAW_ZONE_M)])
            m |= _fill((H, W), self.cam.project(ring))
        self.mask_q, self.mask_cache = q, m
        return m

    def tick(self, frame, q_real):
        """frame: BGR, undistorted. Returns True while a hand is in the workspace."""
        arm = self.mask(q_real); valid = (self.ws | self.band) & ~arm
        if self.bg is None:
            self.bg = frame.copy(); self.hist = [frame]; self.valid_hist = [valid]
            return False
        fg = change(frame, self.bg, shadows=True, chroma=SHADE_CHROMA) & valid
        prev = self.hist[0] if len(self.hist) >= 2 else self.hist[-1]
        pv = self.valid_hist[0] if len(self.valid_hist) >= 2 else self.valid_hist[-1]
        mot = change(frame, prev, shadows=True, chroma=SHADE_CHROMA) & valid & pv
        n, lab, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8))
        moving = []
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] <= AREA_PX:
                continue
            blob = lab == i
            if (blob & self.ws).sum() <= AREA_PX or mot[blob].mean() <= 0.05:
                continue
            darker = (frame[blob].astype(np.int16).sum(1) < self.bg[blob].astype(np.int16).sum(1) - 15).mean()
            if (blob & self.band).sum() >= ENTRY_PX or darker < SHADOW_DARKER:   # comes from the human's side, or not a shadow
                moving.append(i)
        if moving:
            self.hits += 1; self.idle = 0
            if self.hits >= HIT_FRAMES:
                self.present = True
                self.track = cv2.dilate(np.isin(lab, moving).astype(np.uint8), np.ones((61, 61), np.uint8)).astype(bool)
        else:
            self.hits = 0; self.idle += 1
            if self.present:                              # held still? keep it while foreground stays in its region
                keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > AREA_PX // 2 and self.track[lab == i].any()]
                if keep and self.idle >= LATCH_S * FPS:   # nothing moved for LATCH_S: take it as scenery
                    np.copyto(self.bg, frame, where=self.track[..., None]); keep = []
                if keep:
                    self.track = cv2.dilate(np.isin(lab, keep).astype(np.uint8), np.ones((61, 61), np.uint8)).astype(bool)
                else:
                    self.present = False; self.track = None
        self.hist = (self.hist + [frame])[-2:]; self.valid_hist = (self.valid_hist + [valid])[-2:]
        if self.present or self.hits:                     # background frozen while a hand is suspected or present
            return self.present
        # background upkeep: changed-but-still pixels become background after ABSORB_S
        self.still = np.where(fg & ~mot, self.still + 1, 0)
        absorb = self.still >= ABSORB_S * FPS
        np.copyto(self.bg, frame, where=absorb[..., None]); self.still[absorb] = 0
        quiet = ~fg                                       # slow drift (light) where nothing is going on
        np.copyto(self.bg, cv2.addWeighted(self.bg, 0.95, frame, 0.05, 0), where=quiet[..., None])
        return self.present
