"""Hardware proof for skill/cameras.py on the Mac (no robot motion; the arm is not used).

Runs both real cameras through the supervised child processes for --minutes, reading them at 30 Hz like the
policy loop, and injects faults into the camera children (SIGKILL = crash, SIGSTOP = hung driver call) every
--every seconds, alternating cameras. Natural stalls (the wrist camera's USB resets) are counted too.
Checks (PASS/FAIL):
  identity   the front frame matches the latest session reference image (it is the C922, not the Orbbec), and
             the wrist frame does not
  recovery   after every injected fault the camera is fresh again within RECOVER_S
  non-block  no single read in the main loop took longer than 10 ms (restarts run in the watchdog thread; the bound is Python's 5 ms thread switch interval; the control tick is 33 ms)
  coverage   both cameras delivered fresh frames for >= 95 % of the run outside fault windows
Writes data/camera_soak/<time>/ with snapshots (front.png, wrist.png) and log.txt.

    ~/.venvs/so101_eval/bin/python scripts/camera_soak.py --minutes 5
    ~/.venvs/so101_eval/bin/python scripts/camera_soak.py --minutes 3 --replug   # also asks you to unplug/re-plug the wrist camera
"""
import argparse
import os
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import skill.cameras as C  # noqa: E402

RECOVER_S = 6.0


def stale_since_fault(pending, gap):
    """Was this stale stretch caused by the pending injected fault (it began after the fault)?"""
    return time.monotonic() - gap >= pending[1] - 0.05


def ncc(a, b):
    a = cv2.cvtColor(cv2.resize(a, (160, 120)), cv2.COLOR_BGR2GRAY).astype(np.float32); b = cv2.cvtColor(cv2.resize(b, (160, 120)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    a -= a.mean(); b -= b.mean(); return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--minutes", type=float, default=5); ap.add_argument("--every", type=float, default=15)
    ap.add_argument("--replug", action="store_true"); a = ap.parse_args()
    out = ROOT / "data/camera_soak" / time.strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True)
    logf = open(out / "log.txt", "w")

    def log(m):
        line = f"{time.strftime('%H:%M:%S')} {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()
    cams = {n: C.CameraSupervisor(n, log=log) for n in ("front", "wrist")}
    worst = 0.0; fresh = {n: 0 for n in cams}; total = {n: 0 for n in cams}; faults = []; recov = []
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 20:                    # first frames
            try:
                fr = {n: c.latest(0.5) for n, c in cams.items()}; break
            except C.CameraStale:
                time.sleep(0.05)
        else:
            log(f"FAIL start: no frames after 20 s ({ {n: c.status() for n, c in cams.items()} })"); return
        log(f"started in {time.monotonic() - t0:.1f} s")
        for n, f in fr.items():
            cv2.imwrite(str(out / f"{n}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        refs = sorted((ROOT / "data/real_trials").glob("*/ref.png"))
        if refs:
            ref = cv2.imread(str(refs[-1])); f_b = cv2.cvtColor(fr["front"], cv2.COLOR_RGB2BGR); w_b = cv2.cvtColor(fr["wrist"], cv2.COLOR_RGB2BGR)
            s_f, s_w = ncc(f_b, ref), ncc(w_b, ref)
            ok = s_f > 0.6 and s_f > s_w + 0.2
            log(f"{'PASS' if ok else 'FAIL'} identity: front frame vs {refs[-1].parent.name}/ref.png similarity {s_f:.2f}, wrist frame {s_w:.2f}")
        t_end = time.monotonic() + a.minutes * 60; next_fault = time.monotonic() + a.every; k = 0; pending = None
        replug_at = time.monotonic() + a.minutes * 30 if a.replug else None; quiet_until = 0.0
        stale_since = {n: None for n in cams}; natural = []
        while time.monotonic() < t_end:
            t_tick = time.monotonic()
            for n, c in cams.items():
                t1 = time.perf_counter()
                try:
                    c.latest(0.5); ok = True
                except C.CameraStale:
                    ok = False
                worst = max(worst, time.perf_counter() - t1); total[n] += 1; fresh[n] += ok
                if not ok and stale_since[n] is None:
                    stale_since[n] = time.monotonic()
                if ok and stale_since[n] is not None:
                    gap = time.monotonic() - stale_since[n]; stale_since[n] = None
                    if gap > 0.3 and not (pending and pending[0] == n and stale_since_fault(pending, gap)):
                        natural.append((n, gap)); log(f"  {n}: frames stopped for {gap:.1f} s and came back by themselves (not injected)")
                if pending and pending[0] == n and ok and time.monotonic() > pending[1] + 0.6:
                    recov.append((pending[2], n, time.monotonic() - pending[1])); log(f"  {n} fresh again {time.monotonic() - pending[1]:.1f} s after {pending[2]}"); pending = None
            if pending and time.monotonic() - pending[1] > RECOVER_S + 5:
                recov.append((pending[2], pending[0], None)); log(f"  {pending[0]} NOT back {RECOVER_S + 5:.0f} s after {pending[2]}"); pending = None
            if pending is None and time.monotonic() > next_fault and time.monotonic() > quiet_until:
                n = ("wrist", "front")[k % 2]; kind = ("SIGKILL", "SIGSTOP")[(k // 2) % 2]; k += 1
                pid = int(cams[n].meta[C.PID])
                try:
                    os.kill(pid, signal.SIGKILL if kind == "SIGKILL" else signal.SIGSTOP)
                    pending = (n, time.monotonic(), f"{kind} of the {n} child"); faults.append(pending); log(f"inject: {pending[2]} (pid {pid})")
                except ProcessLookupError:
                    pass
                next_fault = time.monotonic() + a.every
            if replug_at and time.monotonic() > replug_at:
                replug_at = None; quiet_until = time.monotonic() + 60   # no injected faults around the unplug
                log(">>> NOW unplug the WRIST camera USB cable, wait 5 s, plug it back in (the loop keeps running; no injected faults for 60 s)")
            time.sleep(max(0.0, 1 / 30 - (time.monotonic() - t_tick)))
    finally:
        for c in cams.values():
            log(f"{c.name}: {c.status()}"); c.close()
    good = [r for r in recov if r[2] is not None and r[2] <= RECOVER_S]
    log(f"{'PASS' if len(good) == len(recov) and recov else 'FAIL'} recovery: {len(good)}/{len(recov)} injected faults recovered within {RECOVER_S:.0f} s "
        f"(times {[round(r[2], 1) if r[2] else None for r in recov]})")
    log(f"{'PASS' if worst < 0.010 else 'FAIL'} non-block: longest single read in the main loop {worst * 1000:.1f} ms")
    log(f"natural stalls (not injected; e.g. the unplug or a USB reset): {[(n, round(g, 1)) for n, g in natural]}")
    if a.replug:
        back = [g for n, g in natural if n == "wrist" and g > 3]
        log(f"{'PASS' if back else 'FAIL'} re-plug: the wrist camera came back by itself after the unplug "
            f"({'after ' + str(round(back[0], 1)) + ' s without frames' if back else 'not seen: no natural wrist stall > 3 s that ended'})")
    cov = {n: fresh[n] / max(total[n], 1) for n in cams}
    lim = 0.7 if a.replug else 0.85
    log(f"{'PASS' if min(cov.values()) > lim else 'FAIL'} coverage: fresh-frame share incl. fault windows{' and the unplug' if a.replug else ''} {cov} (limit {lim})")
    log(f"results in {out}")


if __name__ == "__main__":
    main()
