"""Minimum gripper opening on the way to the first grasp (runtime fix, no retraining).

Real stage-2 trials (28 Oct, notes/phase8.md §17): wcal60 aimed at the right no3 but opened the gripper to only
~24 units (stage-1 successes ~35, sim twin ~27-28), so the jaws landed on the 4 cm box and closed on nothing.

Rule on the policy's gripper command, before the runner's +-MAX_REL clip: once the command first rises above
`start` (the pre-grasp opening has begun), command at least `floor` until the policy starts closing (its command
falls `close_drop` below its running maximum); after that the policy is passed through for the rest of the skill
(retries and the release are its own). `cap` (sim tests only) instead limits the opening to reproduce the real
narrow opening.
"""
import numpy as np


class GripFloor:
    def __init__(self, floor=None, cap=None, start=15.0, close_drop=3.0):
        self.floor, self.cap, self.start, self.close_drop = floor, cap, start, close_drop
        self.state, self.peak, self.ticks = "wait", 0.0, 0         # wait -> armed -> done

    def __call__(self, a, q):
        """a: policy action (LeRobot units), q: reading. Returns the action to clip and send."""
        g = float(a[5])
        if self.state == "wait" and g > self.start:
            self.state = "armed"
        if self.state == "armed":
            self.peak = max(self.peak, g)
            if g < self.peak - self.close_drop:
                self.state = "done"
            else:
                a = np.array(a, copy=True); self.ticks += 1
                if self.floor is not None:
                    a[5] = max(g, self.floor)
                if self.cap is not None:
                    a[5] = min(g, self.cap)
        return a
