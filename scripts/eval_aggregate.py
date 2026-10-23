"""Combine several sim_eval.py runs (episodes.json) into one summary (same breakdowns as sim_eval.summarize).

    ~/.venvs/so101_eval/bin/python scripts/eval_aggregate.py --title "100k, temporal ensembling 0.01" data/eval/te200_s100*
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
from sim_eval import summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--title", default="combined")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resize", default=None); ap.add_argument("--states", default="v1"); ap.add_argument("--temporal-ensemble", type=float, default=None)
    a = ap.parse_args()
    rows = []
    for r in a.runs:
        rows += json.loads((Path(r) / "episodes.json").read_text())
    s = summarize(rows, SimpleNamespace(policy=a.title, mode="act", seed="+".join(Path(r).name for r in a.runs),
                                        randomize=False, n_action_steps=None, temporal_ensemble=a.temporal_ensemble,
                                        resize=a.resize, states=a.states))
    print(s)
    if a.out:
        Path(a.out).write_text(s)


if __name__ == "__main__":
    main()
