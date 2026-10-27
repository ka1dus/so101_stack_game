"""Phase 9: the safety paths in sim (they cannot be exercised with the recorded clips).
  pause   a robot turn with a "hand" (forced) in the workspace for 2 s mid-skill: the arm must hold still while it
          is there, resume 1 s after it left, and still finish the placement
  e-stop  `e` pressed during the robot's turn: the game ends as "stopped", the arm does not move afterwards

    MUJOCO_GL=cgl ~/.venvs/so101_eval/bin/python scripts/test_safety.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from game.backends import SimBackend  # noqa: E402
from game.controller import Controller  # noqa: E402
from skill.runner import FPS, load_policy  # noqa: E402
from ui.display import NullUI  # noqa: E402

POLICY = ROOT / "data/policies/act_v4_ft_040000"
OUT = ROOT / "data/games/test_safety"


class HandAt:
    """Wraps the controller's hand watch: a hand is 'present' during [t0, t1) of sim time."""

    def __init__(self, real, clock, t0, t1):
        self.real, self.clock, self.t0, self.t1 = real, clock, t0, t1

    @property
    def present(self):
        return self.t0 <= self.clock() < self.t1 or self.real.present

    def clear(self, s=1.0):
        t = self.clock()
        return self.real.clear(s) and not (self.t0 <= t < self.t1 + s)

    def __getattr__(self, k):
        return getattr(self.real, k)


class EAfter(NullUI):
    """Presses `e` on the n-th window update of the robot's turn."""

    def __init__(self, n):
        super().__init__(); self.n = n; self.calls = 0

    def show(self, frame, hand=False, wait_ms=1):
        if self.banner[0] == "Robot's turn":
            self.calls += 1
            if self.calls == self.n:
                return "e"
        return None


def main():
    policy = load_policy(str(POLICY), "mps"); b = SimBackend(str(POLICY), "mps", seed=15001); lines = []
    # ---- pause: one human turn, then a robot turn with a hand from 3 s to 5 s into it
    c = Controller(b, policy, NullUI(), OUT / "pause", threaded_hand=False, max_turns=2)
    log = []; orig_robot = c._robot_turn

    def robot(xy, z):
        t_start = b.clock(); c.hand = HandAt(c.hand, b.clock, t_start + 3.0, t_start + 5.0)
        orig_tick = c._tick

        def tick(f, q=None, show=True):
            log.append((b.clock() - t_start, b.joints().copy(), c.hand.present)); return orig_tick(f, q, show)
        c._tick = tick
        return orig_robot(xy, z)
    c._robot_turn = robot
    r = c.play(); t = c.turns[-1]
    L = [(s, q) for s, q, _ in log]; held = [q for s, q in L if 3.1 <= s < 5.9]
    move_held = float(np.abs(np.diff(np.array(held), axis=0)).max()) if len(held) > 1 else None
    before = [q for s, q in L if 2.0 <= s < 3.0]; move_before = float(np.abs(np.diff(np.array(before), axis=0)).max()) if len(before) > 1 else None
    lines.append(f"pause: robot turn {t.get('skill_event')}, call {t['call']}, truth {t.get('truth')}; pauses {t.get('pauses')}, paused {t.get('paused_s')} s")
    lines.append(f"  joint change per frame while the hand was there (3.1-5.9 s incl. the 1 s resume delay): max {move_held:.2f} units"
                 f" (moving before it: {move_before:.2f})")
    # ---- e-stop: pressed ~2 s into the robot's turn
    ui = EAfter(30); c = Controller(b, policy, ui, OUT / "estop", threaded_hand=False, max_turns=2)
    r = c.play(); q0 = b.joints().copy()
    for _ in range(30):
        b.frame()
    lines.append(f"e-stop: game result '{r['why']}', turns logged {len(c.turns)}; arm motion in the 1 s after: {np.abs(b.joints() - q0).max():.2f} units")
    txt = "\n".join(lines); print(txt); (OUT / "summary.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
