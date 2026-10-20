"""
Show the exact joint positions the arm started from in every recorded episode
(the first frame of observation.state), plus the average and spread.

Optionally save the average as the home pose used by so101_go_home.py and
so101_keyboard_record.py, so rollouts start exactly where training did.

Usage:
    python so101_start_poses.py --repo-id Maxctl352/so101_pick_cup
    python so101_start_poses.py --repo-id Maxctl352/so101_pick_cup --write-home --id my_follower_arm
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
    # Not on this machine: fetch only the tables (skips the large videos)
    from huggingface_hub import snapshot_download

    print(f"Downloading {repo_id} tables from the Hub...")
    return Path(snapshot_download(repo_id, repo_type="dataset", allow_patterns=["meta/*", "data/*"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", default=None, help="local dataset folder (optional)")
    p.add_argument("--write-home", action="store_true", help="save the average start pose as home")
    p.add_argument("--id", default="my_follower_arm", help="robot id for the home file name")
    args = p.parse_args()

    root = find_dataset(args.repo_id, args.root)
    info = json.loads((root / "meta" / "info.json").read_text())
    names = info["features"]["observation.state"]["names"]
    if isinstance(names, dict):  # some versions nest the list
        names = next(iter(names.values()))
    joints = [n.removesuffix(".pos") for n in names]

    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    df = pd.concat(
        [pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"]) for f in files]
    )
    first = df[df["frame_index"] == 0].sort_values("episode_index")
    starts = np.stack(first["observation.state"].to_numpy())
    eps = first["episode_index"].to_numpy()

    mean = starts.mean(axis=0)
    std = starts.std(axis=0)
    dev = np.abs(starts - mean).max(axis=1)

    w = 9
    header = "ep   " + "".join(f"{j[:w]:>{w + 1}}" for j in joints) + "   max_dev"
    print(header)
    print("-" * len(header))
    for e, row, d in zip(eps, starts, dev):
        flag = "  <-- outlier" if d > 5 else ""
        print(f"{e:<4} " + "".join(f"{v:>{w + 1}.1f}" for v in row) + f"   {d:6.1f}{flag}")
    print("-" * len(header))
    print("mean " + "".join(f"{v:>{w + 1}.1f}" for v in mean))
    print("std  " + "".join(f"{v:>{w + 1}.1f}" for v in std))
    print(f"\n{len(eps)} episodes. Joints in degrees, gripper 0-100.")

    home_file = Path.home() / ".cache" / "so101_home" / f"{args.id}.json"
    if home_file.exists():
        saved = json.loads(home_file.read_text())
        diff = {j: saved[j] - m for j, m in zip(joints, mean) if j in saved}
        print(f"\nSaved home ({home_file}) minus dataset average:")
        print("     " + "".join(f"{diff.get(j, float('nan')):>{w + 1}.1f}" for j in joints))

    if args.write_home:
        home_file.parent.mkdir(parents=True, exist_ok=True)
        home_file.write_text(json.dumps({j: float(m) for j, m in zip(joints, mean)}, indent=2))
        print(f"\nWrote dataset average start pose to {home_file}")


if __name__ == "__main__":
    main()
