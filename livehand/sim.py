"""MuJoCo side: model construction (reusing the project's Dex3-1 G1 scene) and the live Controller.

The Controller is a plain object (no threads, no viewer): `update(now, dt)` consumes the newest HandState from a
`get_state` callable and writes d.ctrl. It is driven by the real-time loop in app.py and directly by the tests.
"""
import sys
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from .arm_ik import ArmIK
from .calibration import FINGERS, Calibration, Calibrator
from .filters import OneEuro
from .retarget import HandRetargeter, _rotvec, exp_rot, map_orientation, map_position
from .safety import Limiter, Mode, Watchdog

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import grasp_scene as gs  # noqa: E402  (existing project module: Menagerie G1 + Dex3-1 hands, validated arm-forward pose)

HAND_JOINTS = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")


def build_sim():
    """Existing scene: scene_with_hands.xml via MjSpec, right arm in the validated arm-forward pose, hand open."""
    m, d, info = gs.build_scene({}, with_object=False)
    gs.reset(m, d, info)
    return m, d, info


def actuator_for_joint(m, joint):
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
    assert j >= 0, f"joint {joint!r} not in model"
    ids = [i for i in range(m.nu) if m.actuator_trnid[i, 0] == j]
    assert len(ids) == 1, f"expected exactly one actuator for {joint!r}, found {len(ids)}"
    i = ids[0]
    assert m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE and m.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_FIXED, \
        f"actuator for {joint!r} is not a position actuator"
    return i


class Controller:
    def __init__(self, m, d, info, cfg, get_state, calibration=None, clock=None):
        assert cfg.hand.side == "right", "only the right arm/hand is wired"
        self.m, self.d, self.cfg, self.get_state = m, d, cfg, get_state
        self.arm_act = [actuator_for_joint(m, n) for n in gs.ARM]
        self.hand_act = [actuator_for_joint(m, f"{cfg.hand.side}_hand_{n}_joint") for n in HAND_JOINTS]
        self.ctrl_hold = np.array(info["ctrl0"], float)
        lo, hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        out = np.where((self.ctrl_hold < lo - 1e-9) | (self.ctrl_hold > hi + 1e-9))[0]
        self.clipped_home = [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i)), float(self.ctrl_hold[i]), float(lo[i]), float(hi[i]))
                             for i in out]       # e.g. Menagerie's stand keyframe has left_hand_thumb_1 = 1.05 > its 1.0472 limit
        self.ctrl_hold = np.clip(self.ctrl_hold, lo, hi)
        rate = np.full(m.nu, 2.0)
        rate[self.arm_act] = cfg.safety.arm_slew_margin * cfg.arm.max_joint_velocity
        rate[self.hand_act] = cfg.hand.slew_rate
        self.limiter = Limiter(lo, hi, rate, self.ctrl_hold)
        self.home_arm = self.ctrl_hold[self.arm_act].copy()
        self.ik = ArmIK(m, gs.ARM, cfg.arm.site, self.home_arm, cfg)
        self.pos_home, self.R_home = self.ik.site_pose_local(d.qpos, self.home_arm)
        self.q_arm = self.home_arm.copy()
        self.hand = HandRetargeter(cfg, lo[self.hand_act], hi[self.hand_act])
        self.hand_open = self.hand.clamp(self.hand.open)
        self.hand_cmd = self.hand_open.copy()
        self.cart_cmd, self.cart_des = np.zeros(3), np.zeros(3)
        self.rot_cmd, self.rot_des = np.eye(3), np.eye(3)
        self.cal = calibration or Calibration()
        self.calibrator = Calibrator(self.cal, cfg.calibration.samples)
        f = cfg.filter
        self.f_uv = OneEuro(f.pos_min_cutoff, f.pos_beta); self.f_size = OneEuro(f.pos_min_cutoff, f.pos_beta)
        self.f_curl = {k: OneEuro(f.curl_min_cutoff, f.curl_beta) for k in (*FINGERS, "thumb", "pinch")}
        self.wd = Watchdog(cfg.safety.max_target_jump, cfg.safety.stale_timeout, cfg.safety.reacquire_frames)
        self.mode = Mode.DISABLED
        self.orientation_on = bool(cfg.orientation.enabled)
        self.hand_on = bool(cfg.hand.enabled)
        self.homing = False
        self.last_seq = 0
        self.lost_since = None
        self.latency = deque(maxlen=300)
        self.messages = deque(maxlen=4)
        self.cues = dict(index=0.0, middle=0.0, thumb=0.0, pinch=0.0)
        self.tracking_ok = False
        self.now = 0.0

    # ------------------------------------------------------------------ commands
    def say(self, msg):
        self.messages.append(msg)

    def enable(self):
        if self.mode == Mode.ESTOP:
            self.mode = Mode.DISABLED
            self.say("E-STOP cleared (mirroring still DISABLED; press e again to enable)")
        elif self.mode == Mode.ENABLED:
            self.disable()
        elif not self.cal.has_neutral:
            self.say("calibrate neutral first (key n) before enabling")
        else:
            self.mode, self.homing = Mode.ENABLED, False
            self.say("mirroring ENABLED")

    def disable(self):
        self.mode = Mode.DISABLED
        self._hold()
        self.say("mirroring DISABLED (holding current pose)")

    def estop(self):
        self.mode, self.homing = Mode.ESTOP, False
        self._hold()
        self.say("EMERGENCY STOP: holding; press e to clear")

    def home(self):
        if self.mode == Mode.ENABLED:
            self.say("disable mirroring before homing (e)")
            return
        self.homing = True
        self.say("returning to home pose")

    def toggle_orientation(self):
        self.orientation_on = not self.orientation_on
        self.say(f"palm orientation control {'ON' if self.orientation_on else 'OFF'}")

    def toggle_hand(self):
        self.hand_on = not self.hand_on
        self.say(f"hand gestures {'ON' if self.hand_on else 'OFF'}")

    def calibrate(self, kind):
        self.calibrator.start(kind)
        self.say(f"calibrating '{kind}': hold the pose for {self.cfg.calibration.samples} frames")

    def _hold(self):
        self.cart_des = self.cart_cmd.copy()
        self.rot_des = self.rot_cmd.copy()
        self.hand_des = self.hand_cmd.copy()

    # ------------------------------------------------------------------ per-tick update
    hand_des = None

    def _measure(self, state, t):
        """Filtered, calibrated measurements -> (position offset, rotation, hand targets, cues)."""
        c = self.cal
        uv = self.f_uv(state.palm_uv, t)
        size = float(self.f_size(state.size, t))
        raw = {k: float(self.f_curl[k](state.curl_raw[k], t)) for k in FINGERS}
        raw_thumb = float(self.f_curl["thumb"](state.thumb_raw, t))
        raw_pinch = float(self.f_curl["pinch"](state.pinch_raw, t))
        fs = type("S", (), dict(curl_raw=raw, thumb_raw=raw_thumb, pinch_raw=raw_pinch))
        cues = self.hand.cues(fs, c)
        off = map_position(uv, size, c, self.cfg)
        R = map_orientation(state.R_palm, c, self.cfg) if self.orientation_on else np.eye(3)
        return off, R, self.hand.targets(cues), cues

    def update(self, now, dt):
        cfg, st = self.cfg, self.cfg.safety
        seq, state = self.get_state()
        new = seq != self.last_seq and state is not None
        if new:
            self.last_seq = seq
            self.latency.append(now - state.t_capture)
            if self.calibrator.active:
                done = self.calibrator.feed(state)
                if done:
                    self.say(f"calibration '{done}' captured")
            self.wd.update(now, state)
        fresh = state is not None and state.tracked and (now - state.t_capture) <= st.stale_timeout
        self.tracking_ok = bool(self.wd.tracking and fresh)
        if not self.tracking_ok and self.wd.tracking and not fresh:
            self.wd.update(now, None)                  # packet went stale between frames
        self.now = now

        live = self.mode == Mode.ENABLED
        if live and self.tracking_ok:
            self.lost_since = None
            if new:
                off, R, hq, cues = self._measure(state, state.t_capture)
                self.cues = cues
                if self.wd.accept_target(off):
                    self.cart_des, self.rot_des = off, R
                    if self.hand_on:
                        self.hand_des = hq
        elif live:
            if self.lost_since is None:
                self.lost_since = now
                self._hold()
            elif now - self.lost_since > st.lost_return_after:
                self.cart_des, self.rot_des, self.hand_des = np.zeros(3), np.eye(3), self.hand_open.copy()
        elif self.homing:
            self.cart_des, self.rot_des, self.hand_des = np.zeros(3), np.eye(3), self.hand_open.copy()
            if np.linalg.norm(self.cart_cmd) < 1e-4 and np.allclose(self.hand_cmd, self.hand_open, atol=1e-3):
                self.homing = False
                self.say("home reached")
        else:
            self._hold()
        if self.hand_des is None:
            self.hand_des = self.hand_cmd.copy()

        # Cartesian target: first-order lag with a hard speed cap
        a = 1 - np.exp(-dt / max(cfg.position.smoothing_tau, 1e-3))
        step = (self.cart_des - self.cart_cmd) * a
        n = np.linalg.norm(step)
        if n > cfg.position.max_speed * dt:
            step *= cfg.position.max_speed * dt / n
        self.cart_cmd = self.cart_cmd + step
        # rotation: rate-limited slerp-like step toward the target
        dR = _rotvec(self.rot_des @ self.rot_cmd.T)
        mx = np.deg2rad(cfg.orientation.max_rate_deg) * dt
        nr = np.linalg.norm(dR)
        self.rot_cmd = exp_rot(dR * min(1.0, mx / nr) if nr > 1e-9 else dR) @ self.rot_cmd
        self.hand_cmd = np.asarray(self.hand_des, float)

        # IK for the arm, then actuator clamping/slew for everything
        use_rot = self.orientation_on and self.mode == Mode.ENABLED
        tR = self.rot_cmd @ self.R_home if use_rot else None
        if self.mode == Mode.ENABLED or self.homing:   # DISABLED / E-STOP: commands stay exactly where they are
            self.q_arm = self.ik.step(self.d.qpos, self.q_arm, self.pos_home + self.cart_cmd, tR, dt)
        target = self.ctrl_hold.copy()
        target[self.arm_act] = self.q_arm
        target[self.hand_act] = self.hand_cmd
        self.d.ctrl[:] = self.limiter.apply(target, dt)

    # ------------------------------------------------------------------ status
    def status(self):
        lat = np.array(self.latency) * 1000
        return dict(mode=self.mode.value, tracking=self.tracking_ok, lost_for=(self.now - self.lost_since) if self.lost_since else 0.0,
                    calib=dict(self.cal.done), calibrating=self.calibrator.kind, calib_progress=self.calibrator.progress,
                    orientation=self.orientation_on, hand=self.hand_on, homing=self.homing,
                    offset=self.cart_cmd.copy(), cues=dict(self.cues),
                    latency_ms_median=float(np.median(lat)) if len(lat) else float('nan'),
                    latency_ms_p95=float(np.percentile(lat, 95)) if len(lat) else float('nan'),
                    rejected_jumps=self.wd.rejected_jumps, rejected_invalid=self.wd.rejected_invalid + self.limiter.rejected,
                    ik_sigma_min=self.ik.last_sigma_min, messages=list(self.messages))
