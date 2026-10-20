"""
Check whether your dataset really shows grasps at different spots.

For every episode, finds the moment the gripper first closes on the object and
prints the joint angles at that moment. If you recorded 5 spots properly you'll
see ~5 distinct clusters of shoulder_pan / shoulder_lift / elbow_flex values.
If most episodes grasp at nearly the same angles, the dataset itself only
really covers one spot.

Usage:
    python so101_grasp_poses.py --repo-id Maxctl352/so101_pick_cup
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def find_dataset(repo_id: str, root: str | None) -> Path:
    if root:
        return Path(root).expanduser()
    local = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
    if (local / "meta" / "info.json").exists():
        return local
    from huggingface_hub import snapshot_download

    print(f"Downloading {repo_id} tables from the Hub...")
    return Path(snapshot_download(repo_id, repo_type="dataset", allow_patterns=["meta/*", "data/*"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--close-drop", type=float, default=15.0,
                   help="how far the gripper must close (0-100 scale) to count as a grasp")
    args = p.parse_args()

    root = find_dataset(args.repo_id, args.root)
    info = json.loads((root / "meta" / "info.json").read_text())
    names = info["features"]["observation.state"]["names"]
    if isinstance(names, dict):
        names = next(iter(names.values()))
    joints = [n.removesuffix(".pos") for n in names]
    g = joints.index("gripper")
    show = [j for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex") if j in joints]
    idx = [joints.index(j) for j in show]

    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    df = pd.concat([
        pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state", "action"]) for f in files
    ]).sort_values(["episode_index", "frame_index"])

    rows = []
    for ep, d in df.groupby("episode_index"):
        state = np.stack(d["observation.state"].to_numpy())
        act = np.stack(d["action"].to_numpy())
        grip = act[:, g]
        start = grip[: max(1, len(grip) // 20)].mean()  # gripper value at the start
        closed = np.where(grip < start - args.close_drop)[0]
        if len(closed) == 0:
            rows.append((ep, None, len(d)))
            continue
        k = closed[0]
        rows.append((ep, state[k, idx], len(d)))

    ok = [(ep, v, n) for ep, v, n in rows if v is not None]
    print("Grasp pose per episode, sorted by shoulder_pan (degrees):\n")
    print("ep    " + "".join(f"{j[:12]:>14}" for j in show) + "   frames")
    for ep, v, n in sorted(ok, key=lambda r: r[1][0]):
        print(f"{ep:<5} " + "".join(f"{x:>14.1f}" for x in v) + f"   {n:>6}")
    missing = [ep for ep, v, _ in rows if v is None]
    if missing:
        print(f"\nNo grasp detected in episodes {missing} (gripper never closed by {args.close_drop}).")

    pans = np.array([v[0] for _, v, _ in ok])
    if len(pans) > 1:
        print(f"\nshoulder_pan at grasp: min {pans.min():.1f}, max {pans.max():.1f}, "
              f"range {pans.max() - pans.min():.1f} deg")
        # crude cluster count: gaps bigger than 3 deg between sorted values
        s = np.sort(pans)
        clusters = 1 + int(np.sum(np.diff(s) > 3.0))
        print(f"Roughly {clusters} distinct pan positions (gaps > 3 deg). "
              "Spots side by side should show as separate groups; spots at different distances "
              "show up in shoulder_lift / elbow_flex instead.")


if __name__ == "__main__":
    main()
