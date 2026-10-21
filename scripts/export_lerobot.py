"""Phase 5: raw sim episodes -> LeRobotDataset (run in the LeRobot env that trains: WSL `lerobot`, Working rule 2).

Features follow the real SO-101 follower recordings (CLAUDE.md Section 2): built with LeRobot's own
`hw_to_dataset_features` from `{joint}.pos` floats and the `front` / `wrist` 480x640x3 cameras
(src/lerobot/utils/feature_utils.py:48, names from src/lerobot/robots/so_follower/so_follower.py:88).

  --dataset act : only episodes generated with --mode act; task = the single ACT string
  --dataset vla : every episode (ACT episodes too: their picked size is known); task names the size

    python scripts/export_lerobot.py --raw data/raw/act_main --dataset act --repo-id Maxctl352/so101_stack_sim_act
    python scripts/export_lerobot.py --raw data/raw/act_main data/raw/vla_main --dataset vla --repo-id Maxctl352/so101_stack_sim_vla [--push]
Authentication for --push: `hf auth login` beforehand (no tokens in files).
"""
import argparse
import json
from pathlib import Path

import av
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.feature_utils import hw_to_dataset_features

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
CAMS = ("front", "wrist")


def frames(path):
    with av.open(str(path)) as c:
        for f in c.decode(video=0):
            yield f.to_ndarray(format="rgb24")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True)
    ap.add_argument("--dataset", choices=("act", "vla"), required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", default=None, help="local output dir (default: LeRobot's cache)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    joints = {f"{j}.pos": float for j in JOINTS}
    features = (hw_to_dataset_features(joints | {c: (480, 640, 3) for c in CAMS}, "observation", use_video=True)
                | hw_to_dataset_features(joints, "action", use_video=True))
    ds = LeRobotDataset.create(a.repo_id, fps=30, features=features, root=a.root, robot_type="so101_follower", use_videos=True)
    eps = sorted(e for r in a.raw for e in Path(r).glob("ep_*"))
    n = 0; index = []
    for e in eps:
        meta = json.loads((e / "meta.json").read_text())
        if a.dataset == "act" and meta["mode"] != "act":
            continue
        task = meta["task"] if a.dataset == "act" else meta["task_vla"]
        z = np.load(e / "data.npz"); T = len(z["state"])
        vids = {c: frames(e / f"{c}.mp4") for c in CAMS}
        for t in range(T):
            fr = {"observation.state": z["state"][t], "action": z["action"][t], "task": task}
            for c in CAMS:
                fr[f"observation.images.{c}"] = next(vids[c])
            ds.add_frame(fr)
        ds.save_episode()
        index.append(dict(episode_index=n, raw=str(e), picked_type=meta["picked_type"], stack_types=meta["stack_types"],
                          stack_height=meta["state_meta"]["stack_height"], pairing=meta["state_meta"]["pairing"],
                          offsets_mm=meta["state_meta"].get("offsets_mm"), task=task))
        n += 1
        print(f"\r{n} episodes", end="", flush=True)
        if a.limit and n >= a.limit:
            break
    ds.finalize()
    # per-episode game metadata (Section 3c: evaluation split by type and pairing)
    (Path(ds.root) / "meta" / "so101_stack_episodes.json").write_text(json.dumps(index, indent=1))
    print(f"\nwrote {n} episodes to {ds.root}")
    if a.push:
        ds.push_to_hub(private=True)


if __name__ == "__main__":
    main()
