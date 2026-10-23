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
import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.feature_utils import hw_to_dataset_features

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
CAMS = ("front", "wrist")


def verify_videos(root):
    """Each camera's video files must hold exactly sum(episode lengths) frames, and each episode's
    [from, to) span must equal its length. Returns a list of problems (empty = OK)."""
    import glob
    import pandas as pd
    root = Path(root); bad = []
    eps = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(root / "meta/episodes/*/*.parquet")))])
    for cam in CAMS:
        k = f"videos/observation.images.{cam}/"
        span = (eps[k + "to_timestamp"] - eps[k + "from_timestamp"]) * 30
        for e in eps[(span - eps["length"]).abs() > 0.01]["episode_index"]:
            bad.append(f"{cam}: episode {e} video span != length")
        for (ch, fi), g in eps.groupby([k + "chunk_index", k + "file_index"]):
            f = root / f"videos/observation.images.{cam}/chunk-{ch:03d}/file-{fi:03d}.mp4"
            with av.open(str(f)) as c:
                nf = c.streams.video[0].frames or sum(1 for _ in c.decode(video=0))
            if nf != int(g["length"].sum()):
                bad.append(f"{cam}: {f.name} has {nf} frames, episodes need {int(g['length'].sum())}")
    return bad


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
    ap.add_argument("--streaming", action="store_true", help="encode frames directly (no PNG round trip; LeRobot ca69a206+)")
    ap.add_argument("--queue", type=int, default=600,
                    help="streaming encoder queue (frames per camera). LeRobot DROPS frames when it is full "
                         "(video_utils.py:1028-1061); >= the longest episode (480) means nothing can be dropped")
    ap.add_argument("--resize", default=None, help="WxH, e.g. 320x240: store downscaled frames (half-resolution pilot); "
                                                     "the evaluator and the real runner must resize the same way")
    a = ap.parse_args()
    W, H = (int(x) for x in a.resize.split("x")) if a.resize else (640, 480)
    joints = {f"{j}.pos": float for j in JOINTS}
    features = (hw_to_dataset_features(joints | {c: (H, W, 3) for c in CAMS}, "observation", use_video=True)
                | hw_to_dataset_features(joints, "action", use_video=True))
    ds = LeRobotDataset.create(a.repo_id, fps=30, features=features, root=a.root, robot_type="so101_follower", use_videos=True,
                                streaming_encoding=a.streaming, encoder_queue_maxsize=a.queue)
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
                img = next(vids[c])
                fr[f"observation.images.{c}"] = img if a.resize is None else cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
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
    bad = verify_videos(ds.root)
    if bad:
        print("VIDEO CHECK FAILED:", *bad[:10], sep="\n  "); raise SystemExit(2)
    print("video check: every video file has exactly the frames of its episodes")
    if a.push:
        ds.push_to_hub(private=True)


if __name__ == "__main__":
    main()
