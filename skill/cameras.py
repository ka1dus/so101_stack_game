"""Crash-proof camera capture: one supervised child process per camera, opened by device NAME.

Why (29 Oct, notes/phase8.md section 22): the Orbbec Gemini 2 wrist camera browns out on the Mac port (900 mA
available; Orbbec lists 5 V >= 1.5 A, peaks ~6.5 W) and resets itself on the USB bus (kernel log: "terminateDevice",
re-"enumerated", "pipe stalled"). With LeRobot's in-process OpenCVCamera that (1) kills its read thread for good,
(2) can block the whole program inside a driver call when reopening a device mid-reset, and (3) after re-enumeration
the OpenCV index order can change (the macOS list read 0 = Orbbec, 1 = C922 at 17:30), silently swapping front/wrist.

Design, and why each failure is contained:
- Each camera runs in its own process (multiprocessing "spawn"). The main process makes NO camera driver call; a
  driver call that blocks can only block the child. The child writes a heartbeat; the supervisor SIGKILLs a child
  whose heartbeat is older than HANG_S (SIGKILL cannot be caught or blocked) and starts a new one.
- The child opens the device by its macOS name (AVFoundation via pyobjc, uniqueID lookup each time), never by index:
  re-enumeration cannot swap cameras. If the device is absent the child keeps looking (state ABSENT).
- Frames go through shared memory with a lock-free seqlock (two slots), so a child killed mid-write can never leave a
  lock held: the reader retries a torn slot or reports the frame as not fresh.
- Reads in the main process are O(copy), never wait (restarts happen in a watchdog thread): latest() returns the frame if it is younger than max_age, else
  raises CameraStale at once. Waiting is explicit (RealArm.wait_cameras) and the callers decide what the arm does
  meanwhile (the policy loop holds the arm still and resumes when frames are back).
- A child that exits for any reason (crash, exception) is restarted by the supervisor (rate limited).

Test without hardware: scripts/test_cameras.py (fault injection on the "fake" source: hang, crash, freeze, absent,
wrong frame size, reordered device list). On hardware: scripts/camera_soak.py.
"""
import multiprocessing as mp
import os
import signal
import threading
import time
from multiprocessing import shared_memory

import numpy as np

H, W = 480, 640
HANG_S = 3.0            # child heartbeat older than this: hung -> SIGKILL + restart
RESTART_GAP_S = 1.0     # at most one restart per camera per this
SUPERVISE_EVERY_S = 0.1  # watchdog thread period
REOPEN_NO_FRAME_S = 1.5  # child: no frame for this long while streaming -> close and re-open the device
CHILD_EXIT_NO_FRAME_S = 4.0   # child: no frame for this long -> exit, so a fresh process sees the current device list
                              # (real 29 Oct 17:53: after an unplug/re-plug a long-running child never saw the
                              # re-enumerated camera, macOS does not refresh its device list there; a new child did)
DEVICE_NAMES = {"front": "C922 Pro Stream Webcam", "wrist": "Orbbec Gemini 2 RGB Camera"}
# meta layout (float64): heartbeat, state, cur slot, seq0, ts0, seq1, ts1, frames, opens, bad_frames, pid
HB, ST, CUR, SEQ0, TS0, SEQ1, TS1, NFR, NOPEN, NBAD, PID = range(11)
NMETA = 11
STATE = {0: "starting", 1: "absent", 2: "opening", 3: "streaming", 4: "error"}


class CameraStale(RuntimeError):
    """No frame younger than the requested age (camera resetting, absent or restarting)."""


# ------------------------------------------------------------------------------------------- child side
def _publish(buf, meta, frame, n):
    """Write a frame into the slot not currently published, then publish it (seqlock: seq -1 while writing)."""
    k = 1 - int(meta[CUR]); s, t = (SEQ0, TS0) if k == 0 else (SEQ1, TS1)
    meta[s] = -1.0
    buf[k][:] = frame
    meta[t] = time.monotonic(); meta[s] = float(n); meta[CUR] = float(k); meta[NFR] = float(n)


def _child(name, shm_name, meta, source, fault):
    """Capture loop in the child process. source: 'avf' (macOS camera by name) or 'fake' (tests)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)          # Ctrl-C goes to the main process only
    shm = shared_memory.SharedMemory(name=shm_name)
    buf = [np.ndarray((H, W, 3), np.uint8, shm.buf, offset=i * H * W * 3) for i in range(2)]
    meta[PID] = float(os.getpid())
    try:
        (_avf_loop if source == "avf" else _fake_loop)(name, buf, meta, fault)
    finally:
        del buf
        shm.close()


def _beat(meta, state):
    meta[HB] = time.monotonic(); meta[ST] = float(state)


def _avf_loop(name, buf, meta, fault):
    import AVFoundation as AV
    import CoreMedia as CM
    import libdispatch
    import objc
    import Quartz
    from Foundation import NSDate, NSObject, NSRunLoop
    want = DEVICE_NAMES[name]
    count = [int(meta[NFR])]; last = [time.monotonic()]

    class Delegate(NSObject, protocols=[objc.protocolNamed("AVCaptureVideoDataOutputSampleBufferDelegate")]):
        def captureOutput_didOutputSampleBuffer_fromConnection_(self, out, sbuf, conn):
            img = CM.CMSampleBufferGetImageBuffer(sbuf)
            if img is None:
                return
            Quartz.CVPixelBufferLockBaseAddress(img, 1)
            try:
                w, h = Quartz.CVPixelBufferGetWidth(img), Quartz.CVPixelBufferGetHeight(img)
                bpr = Quartz.CVPixelBufferGetBytesPerRow(img)
                if (w, h) != (W, H):
                    meta[NBAD] += 1; return
                raw = Quartz.CVPixelBufferGetBaseAddress(img).as_buffer(bpr * h)
                bgra = np.frombuffer(raw, np.uint8).reshape(h, bpr // 4, 4)[:, :w]
                count[0] += 1; _publish(buf, meta, bgra[..., 2::-1], count[0]); last[0] = time.monotonic()  # -> RGB
            finally:
                Quartz.CVPixelBufferUnlockBaseAddress(img, 1)

    delegate = Delegate.alloc().init(); queue = libdispatch.dispatch_queue_create(b"cam", None)
    t_child = time.monotonic()
    while True:
        _beat(meta, 1)
        if time.monotonic() - max(last[0], t_child) > CHILD_EXIT_NO_FRAME_S:
            return                                        # the supervisor starts a fresh process
        devs = [d for d in AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo) if d.localizedName() == want]
        if not devs:
            time.sleep(0.3); continue
        dev = devs[0]; _beat(meta, 2)
        session = AV.AVCaptureSession.alloc().init()
        inp, err = AV.AVCaptureDeviceInput.deviceInputWithDevice_error_(dev, None)
        if inp is None or not session.canAddInput_(inp):
            _beat(meta, 4); time.sleep(0.5); continue
        session.addInput_(inp)
        if session.canSetSessionPreset_(AV.AVCaptureSessionPreset640x480):
            session.setSessionPreset_(AV.AVCaptureSessionPreset640x480)
        out = AV.AVCaptureVideoDataOutput.alloc().init()
        out.setVideoSettings_({Quartz.kCVPixelBufferPixelFormatTypeKey: Quartz.kCVPixelFormatType_32BGRA,
                               Quartz.kCVPixelBufferWidthKey: W, Quartz.kCVPixelBufferHeightKey: H})
        out.setAlwaysDiscardsLateVideoFrames_(True); out.setSampleBufferDelegate_queue_(delegate, queue)
        session.addOutput_(out); meta[NOPEN] += 1
        n_open = count[0]; session.startRunning(); last[0] = time.monotonic()
        while True:                                       # stream until the device stops delivering
            _beat(meta, 3 if count[0] > n_open and time.monotonic() - last[0] < 0.5 else 2)
            NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))
            if time.monotonic() - last[0] > REOPEN_NO_FRAME_S or not dev.isConnected():
                break
        _beat(meta, 2)
        session.stopRunning()                             # may block in the driver: the heartbeat stops, the
        del session, inp, out                             # supervisor kills and restarts this process


def _fake_loop(name, buf, meta, fault):
    """Test source: frames tagged with the camera's identity (pixel value) at 30 fps; fault[0] selects a failure:
    0 ok, 1 hang in a 'driver call', 2 crash, 3 device absent, 4 wrong frame size, 5 freeze (SIGSTOP by the test),
    6 unplug + re-plug that this process never notices (stale device list: only a new process sees the camera)."""
    tag = {"front": 60, "wrist": 200}[name]; n = int(meta[NFR]); meta[NOPEN] += 1
    stale_list = False; last = time.monotonic()
    while True:
        f = int(fault[0])
        _beat(meta, 3)
        if f == 6:
            stale_list = True                             # this process saw the device vanish: never sees it again
        if stale_list:
            _beat(meta, 1)
            if time.monotonic() - last > CHILD_EXIT_NO_FRAME_S:
                return                                    # same rule as the real child
            time.sleep(0.03); continue
        if f == 1:
            time.sleep(1e6)                               # a driver call that never returns
        if f == 2:
            os._exit(3)
        if f == 3:
            _beat(meta, 1); time.sleep(0.03); continue
        if f == 4:
            meta[NBAD] += 1; time.sleep(0.03); continue
        n += 1; fr = np.full((H, W, 3), tag, np.uint8); fr[0, 0] = n % 256
        _publish(buf, meta, fr, n); last = time.monotonic(); time.sleep(1 / 30)


# ------------------------------------------------------------------------------------------ main side
class CameraSupervisor:
    def __init__(self, name, source="avf", log=print):
        self.name, self.source, self.log = name, source, log
        self.ctx = mp.get_context("spawn")
        self.shm = shared_memory.SharedMemory(create=True, size=2 * H * W * 3)
        self.buf = [np.ndarray((H, W, 3), np.uint8, self.shm.buf, offset=i * H * W * 3) for i in range(2)]
        self.meta = self.ctx.RawArray("d", NMETA); self.fault = self.ctx.RawArray("i", 1)
        self.proc = None; self.t_start = 0.0; self.restarts = 0; self.events = []
        self._start("start")
        self._stop = threading.Event()                    # restarts (kill + spawn, ~10-25 ms) run in this thread,
        self._thr = threading.Thread(target=self._watch, daemon=True, name=f"camwatch-{name}")   # never in a read
        self._thr.start()

    def _watch(self):
        while not self._stop.wait(SUPERVISE_EVERY_S):
            try:
                self.supervise()
            except Exception as e:                        # the watchdog must never die
                self.log(f"  [camera {self.name}] watchdog error: {e}")

    def _start(self, why):
        self.meta[HB] = time.monotonic(); self.meta[ST] = 0.0
        self.proc = self.ctx.Process(target=_child, args=(self.name, self.shm.name, self.meta, self.source, self.fault),
                                     daemon=True, name=f"cam-{self.name}")
        self.proc.start(); self.t_start = time.monotonic()
        if why != "start":
            self.restarts += 1; self.events.append((time.time(), why)); self.log(f"  [camera {self.name}] restarted ({why})")

    def _kill(self):
        p = self.proc
        if p is not None and p.pid is not None:
            try:
                os.kill(p.pid, signal.SIGKILL)            # cannot be blocked, even inside a driver call
            except ProcessLookupError:
                pass
            p.join(timeout=0.5)                           # reaps it (a SIGKILLed process exits at once)

    def supervise(self):
        """Non-blocking health check: restart a dead or hung child. Call often (every read does)."""
        now = time.monotonic()
        if now - self.t_start < RESTART_GAP_S:
            return
        if not self.proc.is_alive():
            self._start(f"child exited, code {self.proc.exitcode}")
        elif now - self.meta[HB] > HANG_S:
            self._kill(); self._start(f"child hung {now - self.meta[HB]:.1f} s (state {STATE.get(int(self.meta[ST]), '?')})")

    def latest(self, max_age_s=0.5):
        """The newest frame (RGB, 480x640) if younger than max_age_s, else CameraStale. Never waits: only a copy
        from shared memory (the watchdog thread restarts children)."""
        for _ in range(3):
            k = int(self.meta[CUR]); s, t = (SEQ0, TS0) if k == 0 else (SEQ1, TS1)
            s1 = self.meta[s]
            if s1 <= 0:
                continue
            fr = self.buf[k].copy(); ts = self.meta[t]
            if self.meta[s] == s1:
                age = time.monotonic() - ts
                if age <= max_age_s:
                    return fr
                raise CameraStale(f"camera {self.name}: newest frame {age:.1f} s old ({self.status()})")
        raise CameraStale(f"camera {self.name}: no complete frame ({self.status()})")

    def status(self):
        return (f"state {STATE.get(int(self.meta[ST]), '?')}, frames {int(self.meta[NFR])}, opens {int(self.meta[NOPEN])}, "
                f"restarts {self.restarts}")

    def close(self):
        self._stop.set(); self._thr.join(timeout=2.0)
        self._kill()
        del self.buf
        try:
            self.shm.close(); self.shm.unlink()
        except Exception:
            pass
