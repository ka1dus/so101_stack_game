"""Phase 5: inspect a run of raw episodes (data/raw/<run>/).

Writes to notes/img/:
  phase5_<run>_joints.png   state (solid) and action (dashed) per joint, LeRobot units, all episodes
  phase5_<run>_dist.png     stack height, picked type, pairing, max stack offset, frames per episode
  phase5_<run>_frames.png   front + wrist frames at 4 moments for up to 6 episodes
and prints checks: units/limits, |action - state| (the real arm runs max_relative_target=10),
per-frame action steps, episode lengths, distribution counts.

    ~/.venvs/so101_stack_sim/bin/python scripts/inspect_episodes.py act_pilot
"""
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from joint_mapping import JOINT_NAMES, JointMapping  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "notes/img"


def frame_strip(ep, n_frames, out_png):
    idx = [0, n_frames // 3, 2 * n_frames // 3, n_frames - 1]
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    tiles = []
    for cam in ("front", "wrist"):
        p = out_png.with_name(out_png.stem + f"_{cam}.png")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(ep / f"{cam}.mp4"), "-vf",
                        f"select='{sel}',scale=240:180,tile=4x1", "-frames:v", "1", str(p)], check=True)
        tiles.append(plt.imread(p)); p.unlink()
    return np.concatenate(tiles, 0)


def main():
    run = sys.argv[1]; d = ROOT / "data/raw" / run
    eps = sorted(d.glob("ep_*"))
    metas = [json.loads((e / "meta.json").read_text()) for e in eps]
    data = [np.load(e / "data.npz") for e in eps]
    lim = JointMapping.from_files().lerobot_limits()
    print(f"{len(eps)} episodes in {d}")
    # ---- checks
    S = np.concatenate([x["state"] for x in data]); A = np.concatenate([x["action"] for x in data])
    print("state range vs calibrated limits (LeRobot units):")
    for i, j in enumerate(JOINT_NAMES):
        lo, hi = sorted(lim[i])
        flag = "" if (S[:, i].min() >= lo - 0.5 and S[:, i].max() <= hi + 0.5) else "  <-- OUTSIDE"
        print(f"  {j:14s} state [{S[:, i].min():7.1f}, {S[:, i].max():7.1f}]  action [{A[:, i].min():7.1f}, {A[:, i].max():7.1f}]"
              f"  limits [{lo:7.1f}, {hi:7.1f}]{flag}")
    gap = np.abs(A - S).max(0); step = np.concatenate([np.abs(np.diff(x["action"], axis=0)) for x in data]).max(0)
    print("max |action - state| per joint:", dict(zip(JOINT_NAMES, gap.round(1))))
    print("max per-frame action step:     ", dict(zip(JOINT_NAMES, step.round(2))))
    body_gap = np.concatenate([np.abs(x["action"] - x["state"])[:, :5] for x in data])
    print(f"share of frames with a body-joint |action - state| > 10 deg (max_relative_target): {(body_gap > 10).any(1).mean()*100:.2f} %")
    lens = [m["frames"] for m in metas]
    print(f"frames per episode: min {min(lens)}, median {int(np.median(lens))}, max {max(lens)} ({np.mean(lens)/30:.1f} s mean)")
    heights = Counter(m["state_meta"]["stack_height"] for m in metas)
    types = Counter(m["picked_type"] for m in metas); pairs = Counter(m["state_meta"]["pairing"] for m in metas)
    offs = [max((np.hypot(*o) for o in m["state_meta"].get("offsets_mm", [[0, 0]])), default=0) for m in metas]
    print("heights:", dict(sorted(heights.items()))); print("picked types:", dict(types)); print("pairings:", dict(pairs))
    print(f"max per-level stack offset (mm): median {np.median(offs):.1f}, max {max(offs):.1f}")
    print("supply modes:", dict(Counter(m["state_meta"]["supply_mode"] for m in metas)),
          "| plain-colour boxes:", sum(m["randomization"]["box_plain"] for m in metas),
          "| shadows on:", sum(m["randomization"]["shadows"] for m in metas))
    # ---- joint plot
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    for x in data:
        t = x["t"]
        for i in range(6):
            l, = ax.flat[i].plot(t, x["state"][:, i], lw=0.8)
            ax.flat[i].plot(t, x["action"][:, i], "--", lw=0.6, color=l.get_color())
    for i, j in enumerate(JOINT_NAMES):
        ax.flat[i].set_title(j + (" (0-100)" if j == "gripper" else " (deg)")); ax.flat[i].set_xlabel("s")
        for v in lim[i]:
            ax.flat[i].axhline(v, color="k", lw=0.5, ls=":")
    fig.suptitle(f"{run}: state (solid) and action (dashed), LeRobot units; dotted = calibrated limits"); fig.tight_layout()
    fig.savefig(IMG / f"phase5_{run}_joints.png", dpi=90); plt.close(fig)
    # ---- distributions
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    ax[0].bar(*zip(*sorted(heights.items()))); ax[0].set_title("stack height (boxes before placing)")
    ax[1].bar(*zip(*types.items())); ax[1].set_title("picked type")
    pk = sorted(pairs); ax[2].barh(pk, [pairs[k] for k in pk]); ax[2].set_title("pairing (picked_on_top)")
    ax[3].hist(offs, bins=12); ax[3].set_title("max per-level stack offset (mm)")
    fig.tight_layout(); fig.savefig(IMG / f"phase5_{run}_dist.png", dpi=90); plt.close(fig)
    # ---- frames
    rows = [frame_strip(e, m["frames"], IMG / f"_tmp_{k}.png") for k, (e, m) in enumerate(zip(eps[:6], metas[:6]))]
    fig = plt.figure(figsize=(12, 3.6 * len(rows)))
    for k, (r, m) in enumerate(zip(rows, metas)):
        a = fig.add_subplot(len(rows), 1, k + 1); a.imshow(r); a.axis("off")
        a.set_title(f"ep {k}: h={m['state_meta']['stack_height']} {m['state_meta']['pairing']} | {m['task']}", fontsize=9)
    fig.tight_layout(); fig.savefig(IMG / f"phase5_{run}_frames.png", dpi=80); plt.close(fig)
    print("figures:", *(f"notes/img/phase5_{run}_{k}.png" for k in ("joints", "dist", "frames")))


if __name__ == "__main__":
    main()
