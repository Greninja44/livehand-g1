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
    cue = _deadzone([u, v, s], cfg.position.deadzone) * cue_gains(cal, cfg)
    delta = np.asarray(cfg.position.axis_map, float) @ cue
    half = np.asarray(cfg.position.workspace_half_extent, float)
    return np.clip(delta, -half, half)


def cue_gains(cal, cfg):
    """Metres per unit of each cue [u, v, s]. Config gains, or -- after a range capture -- the gain that maps the
    user's comfortable span onto the workspace half-extent of the robot axis that cue drives, bounded to
    [range_gain_min, range_gain_max] x the config gain."""
    p = cfg.position
    base = np.array([p.gain, p.gain, p.depth_gain], float)
    if cal.cue_span is None:
        return base
    A = np.abs(np.asarray(p.axis_map, float))
    half = np.asarray(p.workspace_half_extent, float)
    span = np.maximum(np.asarray(cal.cue_span, float) - p.deadzone, 1e-6)
    want = (A.T @ half) / np.maximum(A.sum(0), 1e-9) / span    # each cue's (mean) target half-extent / span
    return np.clip(want, base * p.range_gain_min, base * p.range_gain_max)


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
    """Gesture -> hand actuator targets for one hand profile (config [hand.<model>]).
    Human cues, normalised 0..1 with calibration: index, middle, ring, pinky curl; rest = mean(middle, ring, pinky);
    thumb curl; thumb-index pinch. Each actuator k is driven by cue drive[k] from its open toward its fist value;
    then the actuators in pinch_mask blend toward the pinch pose by the pinch cue (suppressed when `rest` is curled,
    so a fist is not read as a pinch). Targets are convex combinations of the model-derived poses.
      dex3     thumb_0..2 <- thumb, index_0/1 <- index, middle_0/1 <- rest (no ring/pinky); pinch: thumb + index
      inspire  thumb yaw, thumb pitch <- thumb, index/middle/ring/pinky <- own finger; pinch: thumb yaw/pitch + index
    """

    def __init__(self, profile, ctrl_lo, ctrl_hi, max_fraction):
        p = profile
        self.open, self.fist, self.pinch = (np.array(x, float) for x in (p.open, p.fist, p.pinch))
        self.drive = list(p.drive)
        self.pinch_mask = np.array(p.pinch_mask, bool)
        self.lo, self.hi = np.asarray(ctrl_lo, float), np.asarray(ctrl_hi, float)
        self.max_fraction = max_fraction
        assert len(self.drive) == len(self.open) == len(self.lo), "hand profile length must match the actuator count"

    def cues(self, state, cal):
        c = {f: curl_norm(cal, f, state.curl_raw[f]) for f in FINGERS}
        return dict(**c, rest=float(np.mean([c["middle"], c["ring"], c["pinky"]])),
                    thumb=thumb_norm(cal, state.thumb_raw), pinch=pinch_norm(cal, state.pinch_raw))

    def targets(self, cues):
        p_eff = cues["pinch"] * (1.0 - smoothstep(cues["rest"], 0.5, 0.9))
        t = np.array([cues[k] for k in self.drive])
        q = self.open + (self.fist - self.open) * t
        q = np.where(self.pinch_mask, q + (self.pinch - q) * p_eff, q)
        return self.clamp(q)

    def clamp(self, q):
        """Stay inside max_fraction of each actuator's range (measured from the range centre) and inside ctrlrange."""
        c, half = (self.lo + self.hi) / 2, (self.hi - self.lo) / 2 * self.max_fraction
        # never shrink the validated open pose: keep the allowed interval wide enough to contain it
        lo = np.minimum(c - half, self.open)
        hi = np.maximum(c + half, self.open)
        return np.clip(np.clip(q, lo, hi), self.lo, self.hi)
