"""MuJoCo side: model construction (reusing the project's scenes) and the live Controller.

Two robot models (config `[robot].model`):
  inspire  G1 29-DoF + 5-finger Inspire RH56DFQ hands (robots/inspire_scene.py, Unitree URDF), both arms
  dex3     Menagerie G1 + 3-finger Dex3-1 hands (robots/grasp_scene.py), right arm only

The Controller is a plain object (no threads, no viewer): `update(now, dt)` consumes the newest hand packet from a
`get_state` callable and writes d.ctrl. A packet is a dict {"right": HandState, "left": HandState} (either may be
missing); a bare HandState is accepted as the primary (first) side. Each robot side has its own IK, filters,
watchdog, calibration and hand mapping; the enable / e-stop / home state machine is shared.
"""
import sys
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from .arm_ik import ArmIK
from .calibration import FINGERS, Calibration, Calibrator
from .config import SUPPORTED_SIDES, active_sides
from .filters import OneEuro
from .retarget import HandRetargeter, _rotvec, exp_rot, map_orientation, map_position
from .safety import Limiter, Mode, Watchdog

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # for robots/
from robots import grasp_scene as gs  # noqa: E402  (project module: Menagerie G1 + Dex3-1 hands, validated arm-forward pose)
from robots import inspire_scene as isc  # noqa: E402  (project module: G1 + Inspire hands from Unitree's URDF)

ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
HAND_JOINTS = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")    # Dex3-1
INSPIRE_JOINTS = ("thumb_proximal_yaw", "thumb_proximal_pitch", "index_proximal", "middle_proximal",
                  "ring_proximal", "pinky_proximal")

# Inspire home (right arm; the left is the mirror image: roll and yaw joints negated). From
# `python -m livehand.tools.find_inspire_poses --home`: IK for the palm site at INSPIRE_HOME_PALM (pelvis frame),
# fingers forward, palm facing the midline. Chosen so the whole workspace box is reachable (corner error <= 2.8 mm;
# palms 4 cm farther forward put the far-low corners past full arm extension) without the near corners touching the
# torso. Checked by the tests: palm position, joint-limit margin, no arm/hand contacts, robot upright.
INSPIRE_HOME_PALM = (0.26, -0.20, 0.16)
INSPIRE_HOME_RIGHT = (0.3317, -0.3227, -0.2204, -0.6095, 0.4076, 0.3241, 0.0727)
MIRROR = np.array([1, -1, -1, 1, -1, 1, -1])        # joint sign flips right -> left for ARM_JOINTS
CAL_STEPS = ("neutral", "open", "fist", "pinch")    # a hand is calibrated when these are captured (range is optional)


def _inspire_edit(spec):
    for s in "RL":
        # palm centre, 2.5 cm in front of the palm surface (palm normal = -x of the hand base), between wrist and knuckles
        spec.body(f"{s}_hand_base_link").add_site(name=f"{s}_palm", pos=[-0.025, -0.09, 0.0], size=[0.01] * 3,
                                                  rgba=[1, 0.3, 0.3, 0.5])
        # the URDF's thumb-base meshes overlap the palm by 3-5 mm at rest; that contact fought the thumb actuators
        # (pitch 0.45 -> 0.38 reached). Excluding just these pairs makes the thumb track exactly.
        for b in ("thumb_proximal_base", "thumb_proximal"):
            spec.add_exclude(bodyname1=f"{s}_hand_base_link", bodyname2=f"{s}_{b}")


def _build_inspire():
    stand = {}
    for side, sgn in (("right", np.ones(7)), ("left", MIRROR)):
        stand.update({f"{side}_{j}_joint": float(v) for j, v in zip(ARM_JOINTS, sgn * INSPIRE_HOME_RIGHT)})
    return isc.build(stand=stand, edit=_inspire_edit)


def _build_dex3():
    m, d, info = gs.build_scene({}, with_object=False)
    gs.reset(m, d, info)
    return m, d, info


MODELS = {
    "inspire": dict(build=_build_inspire, sides=SUPPORTED_SIDES["inspire"], hand=INSPIRE_JOINTS,
                    site=lambda side: f"{side[0].upper()}_palm",
                    hand_joint=lambda side, n: f"{side[0].upper()}_{n}_joint"),
    "dex3": dict(build=_build_dex3, sides=SUPPORTED_SIDES["dex3"], hand=HAND_JOINTS,
                 site=lambda side: "grasp_site",
                 hand_joint=lambda side, n: f"{side}_hand_{n}_joint"),
}


def build_sim(cfg=None):
    """Build the configured robot (default: the original Dex3-1 scene, for callers without a config)."""
    name = cfg.robot.model if cfg is not None else "dex3"
    return MODELS[name]["build"]()


def actuator_for_joint(m, joint):
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
    assert j >= 0, f"joint {joint!r} not in model"
    ids = [i for i in range(m.nu) if m.actuator_trnid[i, 0] == j]
    assert len(ids) == 1, f"expected exactly one actuator for {joint!r}, found {len(ids)}"
    i = ids[0]
    assert m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE and m.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_FIXED, \
        f"actuator for {joint!r} is not a position actuator"
    return i


class Side:
    """Everything one robot arm + hand needs: IK, hand mapping, filters, watchdog, calibration, targets."""

    def __init__(self, name, m, d, cfg, ctrl_hold, lo, hi, calibration=None):
        prof = MODELS[cfg.robot.model]
        self.name, self.cfg = name, cfg
        arm = [f"{name}_{j}_joint" for j in ARM_JOINTS]
        self.arm_act = [actuator_for_joint(m, n) for n in arm]
        self.hand_act = [actuator_for_joint(m, prof["hand_joint"](name, n)) for n in prof["hand"]]
        self.home_arm = ctrl_hold[self.arm_act].copy()
        self.ik = ArmIK(m, arm, prof["site"](name), self.home_arm, cfg)
        self.pos_home, self.R_home = self.ik.site_pose_local(d.qpos, self.home_arm)
        self.q_arm = self.home_arm.copy()
        self.hand = HandRetargeter(getattr(cfg.hand, cfg.robot.model), lo[self.hand_act], hi[self.hand_act], cfg.hand.max_fraction)
        self.hand_open = self.hand.clamp(self.hand.open)
        self.hand_cmd = self.hand_open.copy()
        self.hand_des = self.hand_open.copy()
        self.cart_cmd, self.cart_des = np.zeros(3), np.zeros(3)
        self.rot_cmd, self.rot_des = np.eye(3), np.eye(3)
        self.cal = calibration or Calibration()
        self.calibrator = Calibrator(self.cal, cfg.calibration.samples, cfg.calibration.range_samples)
        f = cfg.filter
        self.f_uv = OneEuro(f.pos_min_cutoff, f.pos_beta); self.f_size = OneEuro(f.pos_min_cutoff, f.pos_beta)
        self.f_curl = {k: OneEuro(f.curl_min_cutoff, f.curl_beta) for k in (*FINGERS, "thumb", "pinch")}
        self.wd = Watchdog(cfg.safety.max_target_jump, cfg.safety.stale_timeout, cfg.safety.reacquire_frames)
        self.lost_since = None
        self.cues = dict(index=0.0, middle=0.0, ring=0.0, pinky=0.0, rest=0.0, thumb=0.0, pinch=0.0)
        self.tracking_ok = False
        self.state = None

    def hold(self):
        self.cart_des = self.cart_cmd.copy()
        self.rot_des = self.rot_cmd.copy()
        self.hand_des = self.hand_cmd.copy()

    def go_home(self):
        self.cart_des, self.rot_des, self.hand_des = np.zeros(3), np.eye(3), self.hand_open.copy()

    def at_home(self):
        return np.linalg.norm(self.cart_cmd) < 1e-4 and np.allclose(self.hand_cmd, self.hand_open, atol=1e-3)

    def measure(self, state, t, orientation_on):
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
        R = map_orientation(state.R_palm, c, self.cfg) if orientation_on else np.eye(3)
        return off, R, self.hand.targets(cues), cues

    def observe(self, now, state, new):
        """Calibration feed, watchdog and freshness for this side's newest packet."""
        st = self.cfg.safety
        self.state = state
        done = None
        if new:
            if self.calibrator.active:
                done = self.calibrator.feed(state)
            self.wd.update(now, state)
        fresh = state is not None and state.tracked and (now - state.t_capture) <= st.stale_timeout
        self.tracking_ok = bool(self.wd.tracking and fresh)
        if not self.tracking_ok and self.wd.tracking and not fresh:
            self.wd.update(now, None)                  # packet went stale between frames
        return done

    def plan(self, now, new, live, homing, orientation_on, hand_on):
        """Update this side's desired targets for one tick."""
        st = self.cfg.safety
        if live and not self.cal.has_neutral:
            self.hold()                                # not calibrated: this side stays where it is
        elif live and self.tracking_ok:
            self.lost_since = None
            if new:
                off, R, hq, cues = self.measure(self.state, self.state.t_capture, orientation_on)
                self.cues = cues
                if self.wd.accept_target(off):
                    self.cart_des, self.rot_des = off, R
                    if hand_on:
                        self.hand_des = hq
        elif live:
            if self.lost_since is None:
                self.lost_since = now
                self.hold()
            elif now - self.lost_since > st.lost_return_after:
                self.go_home()
        elif homing:
            self.go_home()
        else:
            self.hold()

    def smooth(self, dt):
        """Cartesian target: first-order lag with a hard speed cap; rotation: rate-limited step; hand: as desired."""
        cfg = self.cfg
        a = 1 - np.exp(-dt / max(cfg.position.smoothing_tau, 1e-3))
        step = (self.cart_des - self.cart_cmd) * a
        n = np.linalg.norm(step)
        if n > cfg.position.max_speed * dt:
            step *= cfg.position.max_speed * dt / n
        self.cart_cmd = self.cart_cmd + step
        dR = _rotvec(self.rot_des @ self.rot_cmd.T)
        mx = np.deg2rad(cfg.orientation.max_rate_deg) * dt
        nr = np.linalg.norm(dR)
        self.rot_cmd = exp_rot(dR * min(1.0, mx / nr) if nr > 1e-9 else dR) @ self.rot_cmd
        self.hand_cmd = np.asarray(self.hand_des, float)

    def status(self, now):
        return dict(tracking=self.tracking_ok, lost_for=(now - self.lost_since) if self.lost_since else 0.0,
                    calib=dict(self.cal.done), calibrating=self.calibrator.kind, calib_progress=self.calibrator.progress,
                    offset=self.cart_cmd.copy(), cues=dict(self.cues), ik_sigma_min=self.ik.last_sigma_min,
                    rejected_jumps=self.wd.rejected_jumps, rejected_invalid=self.wd.rejected_invalid)


def separate(right, left, min_sep):
    """Keep the palms at least min_sep apart laterally (pelvis y) by pushing both outward symmetrically."""
    gap = (left.pos_home[1] + left.cart_cmd[1]) - (right.pos_home[1] + right.cart_cmd[1])
    if gap < min_sep:
        push = (min_sep - gap) / 2
        right.cart_cmd[1] -= push
        left.cart_cmd[1] += push
        return True
    return False


class Controller:
    # attributes of the primary side reachable as ctrl.<name> (single-hand callers, tools and tests)
    _PRIMARY = ("arm_act", "hand_act", "home_arm", "ik", "pos_home", "R_home", "q_arm", "hand", "hand_open", "hand_cmd",
                "hand_des", "cart_cmd", "cart_des", "rot_cmd", "rot_des", "cal", "calibrator", "wd", "cues",
                "tracking_ok", "lost_since")

    def __init__(self, m, d, info, cfg, get_state, calibration=None, clock=None):
        """calibration: {side: Calibration} or a single Calibration for the primary side."""
        self.m, self.d, self.cfg, self.get_state = m, d, cfg, get_state
        self.side_names = active_sides(cfg)
        assert self.side_names, f"robot.sides {cfg.robot.sides} not supported by model {cfg.robot.model!r}"
        self.ctrl_hold = np.array(info["ctrl0"], float)
        lo, hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        out = np.where((self.ctrl_hold < lo - 1e-9) | (self.ctrl_hold > hi + 1e-9))[0]
        self.clipped_home = [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i)), float(self.ctrl_hold[i]), float(lo[i]), float(hi[i]))
                             for i in out]       # e.g. Menagerie's stand keyframe has left_hand_thumb_1 = 1.05 > its 1.0472 limit
        self.ctrl_hold = np.clip(self.ctrl_hold, lo, hi)
        cals = calibration if isinstance(calibration, dict) else ({self.side_names[0]: calibration} if calibration else {})
        self.sides = {s: Side(s, m, d, cfg, self.ctrl_hold, lo, hi, cals.get(s)) for s in self.side_names}
        self.primary = self.sides[self.side_names[0]]
        # hand being calibrated: the first one (right) that is not fully calibrated yet
        self.cal_side = next((s for s in self.side_names if not all(self.sides[s].cal.done[k] for k in CAL_STEPS)),
                             self.side_names[0])
        rate = np.full(m.nu, 2.0)
        for sd in self.sides.values():
            rate[sd.arm_act] = cfg.safety.arm_slew_margin * cfg.arm.max_joint_velocity
            rate[sd.hand_act] = cfg.hand.slew_rate
        self.limiter = Limiter(lo, hi, rate, self.ctrl_hold)
        self.mode = Mode.DISABLED
        self.orientation_on = bool(cfg.orientation.enabled)
        self.hand_on = bool(cfg.hand.enabled)
        self.homing = False
        self.last_seq = 0
        self.separations = 0
        self.latency = deque(maxlen=300)
        self.messages = deque(maxlen=4)
        self.now = 0.0

    def __getattr__(self, name):
        if name in Controller._PRIMARY:
            return getattr(self.__dict__["primary"], name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in Controller._PRIMARY and "primary" in self.__dict__:
            setattr(self.primary, name, value)
        else:
            super().__setattr__(name, value)

    # ------------------------------------------------------------------ commands
    def say(self, msg):
        self.messages.append(msg)

    def enable(self):
        if self.mode == Mode.ESTOP:
            self.mode = Mode.DISABLED
            self.say("E-STOP cleared (mirroring still DISABLED; press e again to enable)")
        elif self.mode == Mode.ENABLED:
            self.disable()
        elif not any(sd.cal.has_neutral for sd in self.sides.values()):
            self.say("calibrate neutral first (key n) before enabling")
        else:
            self.mode, self.homing = Mode.ENABLED, False
            off = [s for s, sd in self.sides.items() if not sd.cal.has_neutral]
            self.say("mirroring ENABLED" + (f" ({', '.join(off)} not calibrated: held)" if off else ""))

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

    def _who(self, side):
        return f"{side.upper()} hand" if len(self.sides) > 1 else "your hand"

    def calibrate(self, kind):
        """Start a capture on the current calibration hand only (one hand at a time: the other types the keys)."""
        s = self.cal_side
        sd = self.sides[s]
        if len(self.sides) > 1 and not sd.tracking_ok:
            self.say(f"{self._who(s)} not visible: hold it up in the {s} half of the image (c = other hand)")
        elif not sd.calibrator.start(kind):
            self.say(f"{self._who(s)}: calibrate neutral (n) before '{kind}'")
        elif kind == "range":
            self.say(f"range ({self._who(s)}): sweep slowly left/right, up/down, toward/away "
                     f"({self.cfg.calibration.range_samples} frames)")
        else:
            self.say(f"calibrating '{kind}' ({self._who(s)}): hold the pose for {self.cfg.calibration.samples} frames")

    def switch_cal_side(self):
        """Calibrate the other hand next (key c). Cancels a capture in progress on the current one."""
        if len(self.sides) < 2:
            self.say("only one hand is driven")
            return
        self.sides[self.cal_side].calibrator.cancel()
        self.cal_side = next(s for s in self.side_names if s != self.cal_side)
        self.say(f"now calibrating the {self._who(self.cal_side)}: hold it up in the {self.cal_side} half of the image")

    def cal_complete(self, side):
        return all(self.sides[side].cal.done[k] for k in CAL_STEPS)

    def _after_capture(self, side):
        """Once the current hand has every step, move on to the next hand that still needs calibrating."""
        if side != self.cal_side or not self.cal_complete(side):
            return
        nxt = next((s for s in self.side_names if not self.cal_complete(s)), None)
        if nxt is not None:
            self.cal_side = nxt
            self.say(f"{self._who(side)} calibrated. Now the {self._who(nxt)}: hold it up in the {nxt} half, press n")

    def _hold(self):
        for sd in self.sides.values():
            sd.hold()

    # ------------------------------------------------------------------ per-tick update
    def _state_for(self, frame, side):
        if isinstance(frame, dict):
            return frame.get(side)
        return frame if side == self.side_names[0] else None

    def update(self, now, dt):
        seq, frame = self.get_state()
        new = seq != self.last_seq and frame is not None
        if new:
            self.last_seq = seq
        self.now = now
        live = self.mode == Mode.ENABLED
        for s, sd in self.sides.items():
            state = self._state_for(frame, s)
            if new and state is not None and state.tracked:
                self.latency.append(now - state.t_capture)
            done = sd.observe(now, state, new and state is not None)
            if done:
                self.say(f"calibration '{done}' captured" + (f" ({s})" if len(self.sides) > 1 else ""))
                self._after_capture(s)
            sd.plan(now, new and state is not None, live, self.homing, self.orientation_on, self.hand_on)
        if self.homing and all(sd.at_home() for sd in self.sides.values()):
            self.homing = False
            self.say("home reached")
        for sd in self.sides.values():
            sd.smooth(dt)
        if "right" in self.sides and "left" in self.sides:
            self.separations += separate(self.sides["right"], self.sides["left"], self.cfg.safety.min_hand_separation)

        # IK for the arms, then actuator clamping/slew for everything
        target = self.ctrl_hold.copy()
        for sd in self.sides.values():
            use_rot = self.orientation_on and self.mode == Mode.ENABLED
            tR = sd.rot_cmd @ sd.R_home if use_rot else None
            if self.mode == Mode.ENABLED or self.homing:   # DISABLED / E-STOP: commands stay exactly where they are
                sd.q_arm = sd.ik.step(self.d.qpos, sd.q_arm, sd.pos_home + sd.cart_cmd, tR, dt)
            target[sd.arm_act] = sd.q_arm
            target[sd.hand_act] = sd.hand_cmd
        self.d.ctrl[:] = self.limiter.apply(target, dt)

    # ------------------------------------------------------------------ status
    def status(self):
        lat = np.array(self.latency) * 1000
        sides = {s: sd.status(self.now) for s, sd in self.sides.items()}
        p = sides[self.side_names[0]]
        return dict(mode=self.mode.value, model=self.cfg.robot.model, sides=sides, cal_side=self.cal_side,
                    tracking=any(v["tracking"] for v in sides.values()), lost_for=p["lost_for"],
                    calib=p["calib"], calibrating=p["calibrating"], calib_progress=p["calib_progress"],
                    offset=p["offset"], cues=p["cues"],
                    orientation=self.orientation_on, hand=self.hand_on, homing=self.homing,
                    latency_ms_median=float(np.median(lat)) if len(lat) else float('nan'),
                    latency_ms_p95=float(np.percentile(lat, 95)) if len(lat) else float('nan'),
                    rejected_jumps=sum(v["rejected_jumps"] for v in sides.values()),
                    rejected_invalid=sum(v["rejected_invalid"] for v in sides.values()) + self.limiter.rejected,
                    separations=self.separations, ik_sigma_min=p["ik_sigma_min"], messages=list(self.messages))
