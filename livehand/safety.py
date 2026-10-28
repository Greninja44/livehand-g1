"""Safety layer: actuator clamping + slew limiting, packet watchdog, and the enable / e-stop state machine."""
import enum
import math

import numpy as np


class Mode(enum.Enum):
    DISABLED = "DISABLED"     # startup default: targets held, live input ignored
    ENABLED = "ENABLED"       # mirroring live
    ESTOP = "E-STOP"          # latched hold; must be cleared explicitly


class Limiter:
    """Clamp commands to the actuator ctrlrange and limit their rate of change. NaN/inf commands are rejected."""

    def __init__(self, lo, hi, max_rate, initial):
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.rate = np.asarray(max_rate, float) * np.ones_like(self.lo)
        self.out = np.clip(np.asarray(initial, float), self.lo, self.hi)
        self.rejected = 0

    def apply(self, target, dt):
        t = np.asarray(target, float)
        bad = ~np.isfinite(t)
        if bad.any():
            self.rejected += int(bad.sum())
            t = np.where(bad, self.out, t)
        t = np.clip(t, self.lo, self.hi)
        step = self.rate * dt
        self.out = self.out + np.clip(t - self.out, -step, step)
        return self.out.copy()

    def hold(self):
        return self.out.copy()


class Watchdog:
    """Rejects non-finite or teleporting targets; tracks tracking loss / reacquisition."""

    def __init__(self, max_jump, stale_timeout, reacquire_frames=3):
        self.max_jump, self.stale_timeout, self.reacquire_frames = max_jump, stale_timeout, reacquire_frames
        self.last_target = None
        self.rejected_jumps = 0
        self.rejected_invalid = 0
        self._good_run = 0
        self.tracking = False           # True once reacquire_frames consecutive valid packets were seen

    def packet_valid(self, now, state):
        """Packet is usable if tracked, finite and fresh."""
        if state is None or not state.tracked:
            return False
        if state.palm_uv is None or not np.isfinite(state.palm_uv).all() or not math.isfinite(state.size) or state.size <= 0:
            self.rejected_invalid += 1
            return False
        return (now - state.t_capture) <= self.stale_timeout

    def update(self, now, state):
        """Returns True while tracking is established (enough consecutive fresh valid packets)."""
        if self.packet_valid(now, state):
            self._good_run += 1
            if self._good_run >= self.reacquire_frames:
                self.tracking = True
        else:
            self._good_run = 0
            self.tracking = False
            self.last_target = None    # first packet after a loss becomes the new baseline (no jump rejection)
        return self.tracking

    def accept_target(self, target):
        """False (reject) if target is non-finite or jumps farther than max_jump from the last accepted one."""
        t = np.asarray(target, float)
        if not np.isfinite(t).all():
            self.rejected_invalid += 1
            return False
        if self.last_target is not None and np.linalg.norm(t - self.last_target) > self.max_jump:
            self.rejected_jumps += 1
            return False
        self.last_target = t.copy()
        return True
