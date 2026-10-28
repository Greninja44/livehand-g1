"""Hand-state estimation from 21 MediaPipe landmarks, plus a synthetic hand generator for tests/demos.

Scale-invariant measurements use MediaPipe's *world* landmarks (metric, hand-centred) and are expressed in units
of hand size (wrist -> middle-finger MCP). Hand position uses the *image* landmarks (palm centre in normalised
image coordinates, in units of frame height) and the apparent hand size; MediaPipe's z is NOT used as depth.
"""
from dataclasses import dataclass, field

import numpy as np

WRIST, THUMB_CMC, THUMB_TIP = 0, 1, 4
INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP = 5, 8, 9, 12
RING_MCP, RING_TIP, PINKY_MCP, PINKY_TIP = 13, 16, 17, 20
PALM_IDS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
FINGER_TIPS = dict(index=INDEX_TIP, middle=MIDDLE_TIP, ring=RING_TIP, pinky=PINKY_TIP)
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
               (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


@dataclass
class HandState:
    seq: int = 0
    t_capture: float = 0.0          # time.monotonic() when the frame was read
    t_done: float = 0.0             # when the state was computed
    tracked: bool = False
    handedness: str = ""
    score: float = 0.0
    palm_uv: np.ndarray = None      # (2,) palm centre, image x scaled by aspect, units of frame height
    size: float = 0.0               # apparent wrist->middle-MCP length, units of frame height
    R_palm: np.ndarray = None       # 3x3 palm frame in camera coordinates (x right, y down, z away)
    curl_raw: dict = field(default_factory=dict)   # finger -> |tip-wrist| / hand size (smaller = more curled)
    thumb_raw: float = 0.0          # |thumb tip - pinky MCP| / hand size
    pinch_raw: float = 0.0          # |thumb tip - index tip| / hand size
    landmarks_px: np.ndarray = None  # (21,2) normalised image coords for overlay
    synthetic: bool = False


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def palm_rotation(w, handedness="Right"):
    """Palm frame from world landmarks: x along the fingers, y = palm normal, z = x cross y (camera coords)."""
    x = _unit(w[MIDDLE_MCP] - w[WRIST])
    n = np.cross(w[INDEX_MCP] - w[WRIST], w[PINKY_MCP] - w[WRIST])
    if handedness == "Left":
        n = -n
    y = _unit(n - np.dot(n, x) * x)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


def extract(img_lm, world_lm, handedness="Right", score=1.0, aspect=4 / 3):
    """img_lm: (21,>=2) normalised image coords; world_lm: (21,3) metres. Returns a partially filled HandState."""
    img_lm, w = np.asarray(img_lm, float), np.asarray(world_lm, float)
    if not (np.isfinite(img_lm).all() and np.isfinite(w).all()):
        return HandState(tracked=False)
    size_w = np.linalg.norm(w[MIDDLE_MCP] - w[WRIST])
    if size_w < 1e-4:
        return HandState(tracked=False)
    uv = img_lm[list(PALM_IDS), :2].mean(0) * np.array([aspect, 1.0])
    size_img = np.linalg.norm((img_lm[MIDDLE_MCP, :2] - img_lm[WRIST, :2]) * np.array([aspect, 1.0]))
    return HandState(
        tracked=True, handedness=handedness, score=float(score), palm_uv=uv, size=float(size_img),
        R_palm=palm_rotation(w, handedness),
        curl_raw={k: float(np.linalg.norm(w[i] - w[WRIST]) / size_w) for k, i in FINGER_TIPS.items()},
        thumb_raw=float(np.linalg.norm(w[THUMB_TIP] - w[PINKY_MCP]) / size_w),
        pinch_raw=float(np.linalg.norm(w[THUMB_TIP] - w[INDEX_TIP]) / size_w),
        landmarks_px=img_lm[:, :2].copy())


# ------------------------------------------------------------------ synthetic hand (tests / demo source)
_SEG = dict(index=(0.043, 0.025, 0.020), middle=(0.046, 0.028, 0.022), ring=(0.042, 0.026, 0.020), pinky=(0.034, 0.020, 0.018))
_MCP = dict(index=(-0.015, 0.075), middle=(0.0, 0.080), ring=(0.015, 0.074), pinky=(0.030, 0.064))   # (x, y) in the palm plane (m)
_FIRST = dict(index=5, middle=9, ring=13, pinky=17)


def synthetic_hand(curl=(0, 0, 0, 0), thumb=0.0, pinch=0.0, R=None, center=(0.5, 0.5), scale=1.0, aspect=4 / 3):
    """21 world landmarks (m) and image landmarks for a parametric right hand.
    curl: index, middle, ring, pinky in [0,1]; thumb in [0,1]; pinch in [0,1] (thumb tip -> index tip)."""
    R = np.eye(3) if R is None else np.asarray(R, float)
    p = np.zeros((21, 3))   # hand-local: x right, y along fingers (palm plane y-z), palm normal -z (toward camera)
    for name, (cx, cy) in _MCP.items():
        c = curl[("index", "middle", "ring", "pinky").index(name)]
        i0 = _FIRST[name]
        pos = np.array([cx, cy, 0.0]); p[i0] = pos
        ang = 0.0
        for j, (L, f) in enumerate(zip(_SEG[name], (1.5, 1.7, 1.2))):
            ang += c * f * (0.9 if j == 0 else 1.0) * 0.6
            pos = pos + L * np.array([0.0, np.cos(ang), -np.sin(ang)])
            p[i0 + 1 + j] = pos
    tang = 0.7 * (1 - thumb)                      # thumb abduction angle shrinks as it folds across the palm
    base = np.array([-0.025, 0.02, 0.0]); p[1] = base
    pos = base
    for j, L in enumerate((0.04, 0.032, 0.028)):
        a = tang + thumb * (0.5 + 0.4 * j) * 1.2
        d = np.array([-np.sin(a) * (1 - 0.7 * thumb) + 0.0, np.cos(a) * (1 - 0.3 * thumb), -0.25 * thumb * j])
        d = d + thumb * np.array([0.9, 0.0, 0.0]) * (0.4 + 0.3 * j)
        pos = pos + L * _unit(d)
        p[2 + j] = pos
    if pinch > 0:
        for j in (2, 3, 4):
            k = pinch * (0.3 if j == 2 else 0.65 if j == 3 else 1.0)
            p[j] = p[j] * (1 - k) + p[INDEX_TIP] * k
    world = (p @ R.T) * scale
    img = np.zeros((21, 3))
    img[:, 0] = center[0] + world[:, 0] / aspect * 2.0
    img[:, 1] = center[1] - world[:, 1] * 2.0   # fingers point up in the image
    img[:, 2] = world[:, 2]
    return world, img
