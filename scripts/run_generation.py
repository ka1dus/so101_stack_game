"""Phase 5 large generation, any OS: split ACT and VLA episode totals over N parallel workers.

Each worker runs generate.py for its ACT share, then its VLA share, logging to data/raw/logs/<prefix>_wK.log.
Re-running the same command resumes: each run continues from its existing ep_* folders.
The renderer is the platform default (macOS: cgl; Windows: glfw/WGL on the NVIDIA driver); MUJOCO_GL overrides it.

    python scripts/run_generation.py --workers 8 --act 1674 --vla 1000 --prefix win --seed-base 3000
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def shares(total, n):
    return [total // n + (1 if k < total % n else 0) for k in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--act", type=int, required=True, help="total ACT episodes over all workers")
    ap.add_argument("--vla", type=int, required=True, help="total VLA episodes over all workers")
    ap.add_argument("--prefix", required=True, help="run-name prefix (keep it different per machine)")
    ap.add_argument("--seed-base", type=int, required=True, help="seeds: ACT base+k, VLA base+500+k")
    a, extra = ap.parse_known_args()                  # unknown options are passed on to generate.py
    env = dict(os.environ)
    if sys.platform == "darwin":
        env.setdefault("MUJOCO_GL", "cgl")
    logs = ROOT / "data/raw/logs"; logs.mkdir(parents=True, exist_ok=True)
    gen = [sys.executable, str(ROOT / "scripts/generate.py")]
    procs = []
    for k, (na, nv) in enumerate(zip(shares(a.act, a.workers), shares(a.vla, a.workers))):
        cmds = [gen + ["--n", str(na), "--mode", "act", "--seed", str(a.seed_base + k), "--run", f"act_{a.prefix}_w{k}"] + extra,
                gen + ["--n", str(nv), "--mode", "vla", "--seed", str(a.seed_base + 500 + k), "--run", f"vla_{a.prefix}_w{k}"] + extra]
        log = open(logs / f"{a.prefix}_w{k}.log", "a")
        # one shell-free chain per worker: a small Python runner executes the two commands in order
        runner = "import subprocess,sys,json; [subprocess.run(c, check=True) for c in json.loads(sys.argv[1])]"
        procs.append(subprocess.Popen([sys.executable, "-c", runner, json.dumps(cmds)], cwd=ROOT, env=env,
                                      stdout=log, stderr=subprocess.STDOUT))
        print(f"worker {k}: {na} ACT + {nv} VLA -> data/raw/logs/{a.prefix}_w{k}.log")
    codes = [p.wait() for p in procs]
    print("exit codes:", codes)


if __name__ == "__main__":
    main()
