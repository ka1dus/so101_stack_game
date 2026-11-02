"""Summarize the sim-twin gripper-floor test (scripts/b4_gripfloor_twin.ps1): per condition and trial group, right-box
picks, placements, missed first grasps (lowest tip in 2-8 s > 1.5 cm), max opening before the grasp, neighbour bumps.

    python3 scripts/phase8_gripfloor_summary.py <dir with gf_<tag>_<session>_t<n>.log> floor0 cap24 floor33
"""
import json
import re

import yaml
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
d = Path(sys.argv[1])
SLOT_TYPE = {b["id"]: b["type"] for b in yaml.safe_load((ROOT / "sim/assets/supply_layout.yaml").read_text())["boxes"]}
for tag in sys.argv[2:]:
    agg = defaultdict(lambda: defaultdict(list))
    for f in sorted(d.glob(f"gf_{tag}_*_t*.log")):
        m = re.match(rf"gf_{tag}_(\d+_\d+)_t(\d+)\.log", f.name); sess, n = m.group(1), int(m.group(2))
        t = json.loads((ROOT / "data/real_trials" / sess / "trials.jsonl").read_text().splitlines()[n - 1])
        grp = f"{SLOT_TYPE[t['expected_pick_slot']]} s{t['expected_pick_slot']}"
        txt = f.read_text(errors="ignore")
        for lo, gmax in re.findall(r"GRASP sim \d+: lowest tip in 2-8 s ([\d.]+) cm at [\d.]+ s, gripper max before ([\d.]+)", txt):
            agg[grp]["tip"].append(float(lo)); agg[grp]["gmax"].append(float(gmax))
        for res, on, bump in re.findall(r"PICK sim \d+: .*-> (\w+); placed on the stack (\w+); neighbours moved > 5 mm: (.*)", txt):
            agg[grp]["right"].append(res == "right"); agg[grp]["on"].append(on == "True")
            agg[grp]["bump"].append(bump.strip() != "none")
            agg[grp]["bump_mm"] += [int(x) for x in re.findall(r":(\d+)mm", bump)]
    print(f"== {tag}")
    for grp, v in sorted(agg.items()):
        k = len(v["right"]); miss = sum(x > 1.5 for x in v["tip"])
        print(f"  pick {grp:8s} runs {k:3d}: right box {sum(v['right']):3d}, on stack {sum(v['on']):3d}, first grasp missed {miss:3d}, "
              f"opening max med {sorted(v['gmax'])[len(v['gmax']) // 2]:.1f}, runs with a neighbour moved >5 mm {sum(v['bump']):3d} "
              f"(max {max(v['bump_mm'], default=0)} mm)")
