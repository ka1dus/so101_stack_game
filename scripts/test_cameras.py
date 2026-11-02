"""Fault-injection proof for skill/cameras.py (no hardware: the 'fake' source). Every test also records the longest
single latest() call in the main process, which must stay tiny (the main process never waits on a camera).

    ~/.venvs/so101_eval/bin/python scripts/test_cameras.py
"""
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import skill.cameras as C  # noqa: E402

TAG = {"front": 60, "wrist": 200}
results = []


def poll(cams, seconds, worst, frames=None):
    """Read every camera at ~100 Hz for `seconds`; track the longest call; check identity and completeness."""
    t_end = time.monotonic() + seconds; fresh = {n: 0 for n in cams}
    while time.monotonic() < t_end:
        for n, c in cams.items():
            t0 = time.perf_counter()
            try:
                f = c.latest(0.5); ok = True
            except C.CameraStale:
                ok = False
            dt = time.perf_counter() - t0; worst[0] = max(worst[0], dt)
            if ok:
                fresh[n] += 1
                body = f.reshape(-1, 3)[1:]
                assert (body == TAG[n]).all(), f"{n}: wrong or torn frame (values {np.unique(body)[:5]})"
                if frames is not None:
                    frames[n] += 1
        time.sleep(0.01)
    return fresh


def wait_fresh(cams, timeout, worst):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        f = poll(cams, 0.1, worst)
        if all(v > 0 for v in f.values()):
            return time.monotonic() - t0
    return None


def check(name, cond, detail):
    results.append((name, bool(cond), detail)); print(f"{'PASS' if cond else 'FAIL'}  {name}: {detail}", flush=True)


def main():
    cams = {n: C.CameraSupervisor(n, source="fake", log=lambda m: None) for n in ("front", "wrist")}
    worst = [0.0]
    try:
        t = wait_fresh(cams, 10, worst); check("T1 start", t is not None, f"both cameras fresh after {t:.2f} s" if t else "no frames")
        f = poll(cams, 2.0, worst); check("T1 identity", min(f.values()) > 100, f"fresh reads {f}, every frame had its own camera's tag")

        w = cams["wrist"]
        for test, fault, extra in (("T2 driver call hangs", 1, None), ("T3 child crashes", 2, None),
                                   ("T5 child frozen (SIGSTOP)", 0, "stop"), ("T6 wrong frame size", 4, None),
                                   ("T4 device absent 5 s", 3, None)):
            r0 = w.restarts; t0 = time.monotonic()
            if extra == "stop":
                os.kill(int(w.meta[C.PID]), signal.SIGSTOP)
            else:
                w.fault[0] = fault
            stale_at = None
            while time.monotonic() - t0 < 8:
                fr = poll({"wrist": w}, 0.05, worst)
                if fr["wrist"] == 0 and stale_at is None:
                    stale_at = time.monotonic() - t0
                if fault in (1, 2) and w.restarts > r0:
                    w.fault[0] = 0                         # the fault clears once the child was replaced
                if extra == "stop" and w.restarts > r0:
                    break
                if fault in (3, 4) and time.monotonic() - t0 > 5:
                    w.fault[0] = 0; break
            rec = wait_fresh({"wrist": w}, 10, worst)
            f2 = poll(cams, 1.0, worst)
            needs_stale = fault != 2                       # a crashed child is replaced before frames go stale
            check(test, rec is not None and (stale_at is not None or not needs_stale) and f2["front"] > 50
                  and (w.restarts > r0 or fault in (3, 4)),
                  f"wrist stale after {stale_at}, restarts +{w.restarts - r0}, fresh again {rec and round(rec, 2)} s later; front unaffected ({f2['front']} fresh reads)")

        # T7: unplug + re-plug that the running child never notices (stale device list), as on the real Mac 17:53
        r0 = w.restarts; w.fault[0] = 6; t0 = time.monotonic(); poll({"wrist": w}, 1.0, worst); w.fault[0] = 0   # 'plugged back'
        rec = wait_fresh({"wrist": w}, 15, worst)
        check("T7 re-plug, stale device list", rec is not None and w.restarts > r0,
              f"fresh again {rec and round(time.monotonic() - t0, 1)} s after the unplug via {w.restarts - r0} new child process(es) "
              f"(the old child exited after {C.CHILD_EXIT_NO_FRAME_S:.0f} s without frames)")

        # T8: SIGKILL the child at random moments while reading at full speed: no torn or foreign frame may appear
        rng = np.random.default_rng(0); n_kill = 0; frames = {"front": 0, "wrist": 0}
        for _ in range(30):
            poll(cams, rng.uniform(0.0, 0.15), worst, frames)
            try:
                os.kill(int(w.meta[C.PID]), signal.SIGKILL); n_kill += 1
            except ProcessLookupError:
                pass
            poll(cams, 0.05, worst, frames)
            w.t_start -= 10                                # allow an immediate restart for this stress test
            time.sleep(0.12)                               # (the watchdog thread restarts it)
        rec = wait_fresh(cams, 15, worst)
        check("T8 kill mid-write x30", rec is not None, f"{n_kill} kills, {frames} complete frames read, none torn or swapped; fresh {rec and round(rec, 2)} s after the last kill")
        check("T9 main never waits", worst[0] < 0.010, f"longest single latest() call {worst[0] * 1000:.2f} ms over all tests")
    finally:
        for c in cams.values():
            c.close()
    print(f"\n{sum(r[1] for r in results)}/{len(results)} passed")
    sys.exit(0 if all(r[1] for r in results) else 1)


if __name__ == "__main__":
    main()
