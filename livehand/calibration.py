"""Keyboard-triggered calibration: neutral pose (position/orientation reference) and gesture ranges."""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

FINGERS = ("index", "middle", "ring", "pinky")

# Typical-hand priors (hand-size-normalised); overwritten by calibration captures.
PRIOR = dict(finger_open=1.95, finger_closed=0.90, thumb_open=1.25, thumb_closed=0.35, pinch_open=0.95, pinch_closed=0.12)


@dataclass
class Calibration:
    palm_uv0: list = None           # neutral palm position (image, frame-height units)
    size0: float = None             # neutral apparent hand size
    R0: list = None                 # neutral palm rotation (3x3, camera coords)
    finger_open: dict = field(default_factory=lambda: {k: PRIOR["finger_open"] for k in FINGERS})
    finger_closed: dict = field(default_factory=lambda: {k: PRIOR["finger_closed"] for k in FINGERS})
    thumb_open: float = PRIOR["thumb_open"]
    thumb_closed: float = PRIOR["thumb_closed"]
    pinch_open: float = PRIOR["pinch_open"]
    pinch_closed: float = PRIOR["pinch_closed"]
    cue_span: list = None           # [u, v, s] comfortable motion range from neutral (range capture); None = config gains
    done: dict = field(default_factory=lambda: dict(neutral=False, range=False, open=False, fist=False, pinch=False))

    @property
    def has_neutral(self):
        return self.done["neutral"]

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, d):
        c = cls()
        for k, v in d.items():
            if k == "done":
                c.done.update(v)            # files saved before a capture kind existed keep its default
            elif hasattr(c, k):
                setattr(c, k, v)
        return c

    def update_from(self, other):
        """Copy another calibration's values in place (the Calibrator keeps a reference to this object)."""
        self.__dict__.update({k: v for k, v in other.__dict__.items() if k != "done"})
        self.done.clear(); self.done.update(other.done)


def save_set(path, cals):
    """Save {side: Calibration} as {"right": {...}, "left": {...}}."""
    Path(path).write_text(json.dumps({s: asdict(c) for s, c in cals.items()}, indent=2))


def load_set(path, sides):
    """Load {side: Calibration} for `sides`. A pre-two-hand file (one flat calibration) applies to sides[0]."""
    d = json.loads(Path(path).read_text())
    if "done" in d:
        d = {sides[0]: d}
    return {s: Calibration.from_dict(d[s]) if s in d else Calibration() for s in sides}


def norm01(x, lo_val, hi_val):
    """Map x so that lo_val -> 1 (fully 'on') and hi_val -> 0, clipped. Works for either ordering; degenerate -> 0."""
    span = hi_val - lo_val
    if abs(span) < 1e-6 or not np.isfinite(x):
        return 0.0
    return float(np.clip((hi_val - x) / span, 0.0, 1.0))


def curl_norm(cal, finger, raw):
    return norm01(raw, cal.finger_closed[finger], cal.finger_open[finger])


def thumb_norm(cal, raw):
    return norm01(raw, cal.thumb_closed, cal.thumb_open)


def pinch_norm(cal, raw):
    return norm01(raw, cal.pinch_closed, cal.pinch_open)


class Calibrator:
    """Averages `samples` valid frames per capture. kinds: neutral | open | fist | pinch.
    'range' collects `range_samples` frames while the user sweeps the hand around neutral (needs neutral first)."""

    def __init__(self, cal: Calibration, samples=20, range_samples=60):
        self.cal, self.samples, self.range_samples = cal, samples, range_samples
        self.kind = None
        self._acc = []

    @property
    def active(self):
        return self.kind is not None

    @property
    def _need(self):
        return self.range_samples if self.kind == "range" else self.samples

    @property
    def progress(self):
        return len(self._acc) / self._need if self.kind else 0.0

    def start(self, kind):
        """Begin a capture. Returns False (and does nothing) if 'range' is requested before 'neutral'."""
        assert kind in ("neutral", "range", "open", "fist", "pinch")
        if kind == "range" and not self.cal.has_neutral:
            return False
        self.kind, self._acc = kind, []
        return True

    def cancel(self):
        self.kind, self._acc = None, []

    def feed(self, state):
        """Feed a HandState; returns the completed kind when a capture finishes, else None."""
        if not self.kind or state is None or not state.tracked:
            return None
        self._acc.append(state)
        if len(self._acc) < self._need:
            return None
        s, c, kind = self._acc, self.cal, self.kind
        mean = lambda f: float(np.mean([f(x) for x in s]))
        if kind == "range":
            dev = np.array([[*(np.asarray(x.palm_uv) - c.palm_uv0), np.log(x.size / c.size0)] for x in s])
            c.cue_span = np.percentile(np.abs(dev), 95, axis=0).tolist()   # robust to a few outlier frames
        elif kind == "neutral":
            c.palm_uv0 = np.mean([x.palm_uv for x in s], 0).tolist()
            c.size0 = mean(lambda x: x.size)
            R = np.mean([x.R_palm for x in s], 0)
            u, _, vt = np.linalg.svd(R)           # project the mean back onto SO(3)
            R = u @ vt
            if np.linalg.det(R) < 0:
                R = u @ np.diag([1, 1, -1]) @ vt
            c.R0 = R.tolist()
        elif kind in ("open", "fist"):
            tgt = c.finger_open if kind == "open" else c.finger_closed
            for f in FINGERS:
                tgt[f] = mean(lambda x, f=f: x.curl_raw[f])
            if kind == "open":
                c.thumb_open = mean(lambda x: x.thumb_raw)
            else:
                c.thumb_closed = mean(lambda x: x.thumb_raw)
            if kind == "open":
                c.pinch_open = mean(lambda x: x.pinch_raw)
        elif kind == "pinch":
            c.pinch_closed = mean(lambda x: x.pinch_raw)
        c.done[kind] = True
        self.kind, self._acc = None, []
        return kind
