"""Human-to-robot retargeting: palm motion -> bounded Cartesian offset/rotation; finger gestures -> Dex3-1 targets."""
import numpy as np

from .calibration import FINGERS, curl_norm, pinch_norm, thumb_norm


def _deadzone(x, dz):
    x = np.asarray(x, float)
    return np.sign(x) * np.maximum(np.abs(x) - dz, 0.0)


def map_position(palm_uv, size, cal, cfg):
    """Relative palm displacement -> robot-frame offset (m) from the home end-effector position, clipped to the
    workspace box. Cues: u (screen right), v (screen down), s = ln(size/size0) (toward camera). Not metric depth."""
    if not cal.has_neutral or palm_uv is None or size <= 0:
        return np.zeros(3)
    u, v = np.asarray(palm_uv, float) - np.asarray(cal.palm_uv0, float)
    s = np.log(size / cal.size0)
    cue = _deadzone([u, v, s], cfg.position.deadzone) * np.array([cfg.position.gain, cfg.position.gain, cfg.position.depth_gain])
    delta = np.asarray(cfg.position.axis_map, float) @ cue
    half = np.asarray(cfg.position.workspace_half_extent, float)
    return np.clip(delta, -half, half)


def _rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    ang = np.arccos(c)
    if ang < 1e-8:
        return np.zeros(3)
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    return ax * ang


def exp_rot(v):
    v = np.asarray(v, float)
    a = np.linalg.norm(v)
    if a < 1e-9:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def map_orientation(R_now, cal, cfg):
    """Palm rotation relative to the neutral palm -> robot-frame rotation matrix, clamped to cfg.orientation.max_deg."""
    if not cal.has_neutral or R_now is None or cal.R0 is None:
        return np.eye(3)
    R_rel_cam = np.asarray(R_now, float) @ np.asarray(cal.R0, float).T      # rotation expressed in camera axes
    rv = _rotvec(R_rel_cam)
    C = np.asarray(cfg.orientation.axis_map, float)
    rv_robot = np.linalg.det(C) * (C @ rv) * cfg.orientation.gain            # pseudovector under a reflection
    mx = np.deg2rad(cfg.orientation.max_deg)
    n = np.linalg.norm(rv_robot)
    if n > mx:
        rv_robot = rv_robot * (mx / n)
    return exp_rot(rv_robot)


def smoothstep(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


class HandRetargeter:
    """Dex3-1 mapping. Actuator order (cfg.hand.open/fist/pinch): thumb_0, thumb_1, thumb_2, index_0, index_1,
    middle_0, middle_1. Human cues (normalised 0..1 with calibration):
      index curl                                 -> index_0, index_1   (open -> fist pose)
      mean(middle, ring, pinky curl)             -> middle_0, middle_1 (the robot has no ring/pinky)
      thumb curl                                 -> thumb_0..2         (open -> fist pose)
      thumb-index pinch (suppressed in a fist)   -> blend thumb_0..2 and index_0/1 toward the pinch pose
    """

    def __init__(self, cfg, ctrl_lo, ctrl_hi):
        h = cfg.hand
        self.open, self.fist, self.pinch = (np.array(x, float) for x in (h.open, h.fist, h.pinch))
        self.lo, self.hi = np.asarray(ctrl_lo, float), np.asarray(ctrl_hi, float)
        self.max_fraction = h.max_fraction

    def cues(self, state, cal):
        c = {f: curl_norm(cal, f, state.curl_raw[f]) for f in FINGERS}
        return dict(index=c["index"], middle=float(np.mean([c["middle"], c["ring"], c["pinky"]])),
                    thumb=thumb_norm(cal, state.thumb_raw), pinch=pinch_norm(cal, state.pinch_raw))

    def targets(self, cues):
        idx, mid, th, p = cues["index"], cues["middle"], cues["thumb"], cues["pinch"]
        p_eff = p * (1.0 - smoothstep(mid, 0.5, 0.9))
        lerp = lambda a, b, t: a + (b - a) * t
        q = self.open.copy()
        q[0:3] = lerp(self.open[0:3], self.fist[0:3], th)
        q[3:5] = lerp(self.open[3:5], self.fist[3:5], idx)
        q[5:7] = lerp(self.open[5:7], self.fist[5:7], mid)
        q[0:3] = lerp(q[0:3], self.pinch[0:3], p_eff)
        q[3:5] = lerp(q[3:5], self.pinch[3:5], p_eff)
        return self.clamp(q)

    def clamp(self, q):
        """Stay inside max_fraction of each actuator's range (measured from the range centre) and inside ctrlrange."""
        c, half = (self.lo + self.hi) / 2, (self.hi - self.lo) / 2 * self.max_fraction
        # never shrink the validated open pose: keep the allowed interval wide enough to contain it
        lo = np.minimum(c - half, self.open)
        hi = np.maximum(c + half, self.open)
        return np.clip(np.clip(q, lo, hi), self.lo, self.hi)
