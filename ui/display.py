"""OpenCV game window and a headless stand-in.

Window: the live `front` frame (undistorted) with the stack column and the detected stack top drawn in, the supply
slots coloured full/empty, a banner (whose turn and what to do) and, while a hand is in the workspace, a red frame.
Keys: SPACE = done (ends the human's turn, confirms a prompt); l / m / s = large / medium / small (answers "which box
did you take?"); e or ESC = E-STOP; q = quit after the current turn.
Sounds: macOS system sounds through afplay (non-blocking); silent elsewhere.
cv2.imshow must run on the main thread on macOS: the controller calls show() from its own loop (and from the skill
loop's per-tick hook), so the window stays live while the arm moves.
"""
import shutil
import subprocess

import cv2
import numpy as np

SOUNDS = {"turn": "Glass", "robot": "Pop", "loss": "Basso", "win": "Hero", "alert": "Sosumi"}
COLORS = {"human": (60, 170, 60), "robot": (200, 120, 30), "judge": (90, 90, 90), "over": (40, 40, 200), "alert": (0, 0, 220)}
WIN = "SO-101 stacking game"


def sound(name):
    if shutil.which("afplay") and name in SOUNDS:
        subprocess.Popen(["afplay", f"/System/Library/Sounds/{SOUNDS[name]}.aiff"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class UI:
    def __init__(self, scale=1.5):
        self.scale = scale; self.banner = ("", "", "judge"); self.overlay = {}; self.last = None
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    def set(self, title, sub="", mode="judge"):
        self.banner = (title, sub, mode)

    def draw(self, frame, hand=False):
        img = frame.copy(); ov = self.overlay
        if "column" in ov:
            cv2.polylines(img, [ov["column"]], True, (255, 255, 255), 1)
        if "top_uv" in ov:
            cv2.drawMarker(img, tuple(int(v) for v in ov["top_uv"]), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        for uv, full in ov.get("slots", []):
            cv2.circle(img, tuple(int(v) for v in uv), 6, (0, 200, 0) if full else (0, 0, 200), -1)
        img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        title, sub, mode = self.banner
        bar = np.full((90, img.shape[1], 3), COLORS[mode], np.uint8)
        cv2.putText(bar, title, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(bar, sub, (16, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if ov.get("stack"):
            cv2.putText(bar, "stack: " + " ".join(ov["stack"]), (img.shape[1] - 420, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        out = np.vstack([bar, img])
        if hand:
            cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), COLORS["alert"], 10)
            cv2.putText(out, "HAND IN WORKSPACE", (16, out.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS["alert"], 3)
        return out

    def show(self, frame, hand=False, wait_ms=1):
        self.last = self.draw(frame, hand); cv2.imshow(WIN, self.last)
        k = cv2.waitKey(wait_ms) & 0xFF
        return None if k == 255 else ("esc" if k == 27 else "space" if k == 32 else chr(k))

    def close(self):
        cv2.destroyWindow(WIN)


class NullUI(UI):
    """Headless: nothing drawn; keys come from a script (dry runs). After the script, `then` is returned on every
    call (e.g. "space": a player who keeps pressing SPACE until the controller accepts it)."""

    def __init__(self, keys=None, then=None):
        self.banner = ("", "", "judge"); self.overlay = {}; self.last = None; self.keys = list(keys or []); self.then = then

    def show(self, frame, hand=False, wait_ms=1):
        return self.keys.pop(0) if self.keys else self.then

    def close(self):
        pass
