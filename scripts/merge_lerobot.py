"""Phase 5: merge LeRobot dataset parts (exported in parallel by export_lerobot.py) into one dataset.

Uses LeRobot's own `merge_datasets` (src/lerobot/datasets/dataset_tools.py:278). The per-episode game metadata
(meta/so101_stack_episodes.json) of each part is concatenated with its episode_index offset, in part order.

    python scripts/merge_lerobot.py --parts <root0> <root1> ... --repo-id Maxctl352/so101_stack_sim_act --out <root>
"""
import argparse
import json
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from export_lerobot import verify_videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True, help="local roots of the part datasets, in order")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    parts = []
    for root in a.parts:
        bad = verify_videos(root)
        if bad:
            raise SystemExit(f"part {root} fails the video check ({len(bad)} problems, e.g. {bad[0]}); not merging")
        info = json.loads((Path(root) / "meta/info.json").read_text())
        parts.append(LeRobotDataset(f"local/{Path(root).name}", root=root))
        print(f"{root}: {parts[-1].num_episodes} episodes, {parts[-1].num_frames} frames ({info['total_episodes']} in info.json)")
    merged = merge_datasets(parts, output_repo_id=a.repo_id, output_dir=a.out)
    index, off = [], 0
    for ds, root in zip(parts, a.parts):
        for e in json.loads((Path(root) / "meta/so101_stack_episodes.json").read_text()):
            e = dict(e); e["episode_index"] += off; index.append(e)
        off += ds.num_episodes
    (Path(a.out) / "meta/so101_stack_episodes.json").write_text(json.dumps(index, indent=1))
    print(f"merged: {merged.num_episodes} episodes, {merged.num_frames} frames -> {a.out}; game metadata for {len(index)} episodes")
    assert merged.num_episodes == len(index), "episode count mismatch"
    bad = verify_videos(a.out)
    if bad:
        raise SystemExit(f"merged dataset fails the video check: {bad[:3]}")
    print("video check of the merged dataset: OK")


if __name__ == "__main__":
    main()
