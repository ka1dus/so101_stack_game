"""E-stop and hand watch.

EStop      set by the UI's e-stop key or Ctrl-C; once set, every motion stops: the arm holds, then torque is turned off
           (CLAUDE.md Section 4: "e-stop key -> torque off"). Clearing it needs a new game.
HandWatch  hand_in_workspace() (perception/hand.py, decision D3) in its own thread, so its ~18 ms per frame does not
           slow the 30 Hz skill loop: the loop hands over its latest frame and joint reading, the thread checks the
           newest one it has. `present` is the monitor's current call; `clear(s)` is True once no hand has been seen
           for s seconds. The robot moves only while clear(RESUME_S) (start of a turn and after a pause).
"""
import threading
import time

from perception.hand import HandMonitor

RESUME_S = 1.0             # no hand for this long before the robot starts or resumes


class EStopped(Exception):
    pass


class EStop:
    def __init__(self):
        self.flag = threading.Event(); self.reason = None

    def trigger(self, reason):
        if not self.flag.is_set():
            self.reason = reason; self.flag.set()

    def __bool__(self):
        return self.flag.is_set()


class HandWatch:
    def __init__(self, cam, twin, kin=None, clock=time.monotonic, threaded=True):
        self.mon = HandMonitor(cam, twin, kin); self.clock = clock
        self.lock = threading.Lock(); self.latest = None; self.new = threading.Event()
        self.present = False; self.last_seen = -1e9; self.frames = 0; self.stop_ = False; self.pending_rebase = None
        self.thread = threading.Thread(target=self._run, daemon=True) if threaded else None
        if self.thread:
            self.thread.start()

    def feed(self, frame_bgr, q_real):
        if self.thread is None:                             # single-threaded (sim time): check every frame now
            self.check_now(frame_bgr, q_real); return
        with self.lock:
            self.latest = (frame_bgr, q_real)
        self.new.set()

    def _run(self):
        while not self.stop_:
            if not self.new.wait(0.2):
                continue
            with self.lock:
                item, self.latest = self.latest, None; self.new.clear()
                rb, self.pending_rebase = self.pending_rebase, None
            if rb is not None:
                self.mon.rebase(rb); self.present = False
            if item is None:
                continue
            p = self.mon.tick(*item); self.frames += 1
            self.present = p
            if p:
                self.last_seen = self.clock()

    def quiet_s(self):
        """Seconds without a moving blob (as counted by the monitor, at the rate it checks frames)."""
        return self.mon.idle / 30.0

    def rebase(self, frame_bgr):
        """New empty-workspace background (see HandMonitor.rebase); applied by the checking thread at its next frame."""
        if self.thread is None:
            self.mon.rebase(frame_bgr); self.present = False; return
        with self.lock:
            self.latest = None; self.pending_rebase = frame_bgr
        self.new.set()

    def check_now(self, frame_bgr, q_real):
        """Synchronous tick (single-threaded callers, e.g. the sim dry run where wall time is not real time)."""
        p = self.mon.tick(frame_bgr, q_real); self.frames += 1; self.present = p
        if p:
            self.last_seen = self.clock()
        return p

    def clear(self, s=RESUME_S):
        return not self.present and self.clock() - self.last_seen >= s

    def close(self):
        self.stop_ = True; self.new.set()
        if self.thread:
            self.thread.join(timeout=1)
