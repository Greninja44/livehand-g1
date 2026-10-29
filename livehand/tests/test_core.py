"""LiveHand tests (stdlib unittest). Run: python -m unittest livehand.tests.test_core -v
No webcam or display needed: sim tests use a fake state source and a simulated clock."""
import time
import unittest

import mujoco
import numpy as np

from livehand import config as config_mod
from livehand.calibration import Calibration, Calibrator, curl_norm, pinch_norm, thumb_norm
from livehand.features import extract, synthetic_hand
from livehand.retarget import HandRetargeter, cue_gains, map_orientation, map_position
from livehand.safety import Limiter, Mode, Watchdog
from livehand.sim import INSPIRE_HOME_PALM, INSPIRE_JOINTS, Controller, HAND_JOINTS, actuator_for_joint, build_sim, gs
from livehand.tracking import VisionSystem

CFG = config_mod.load(overrides={"robot": {"model": "dex3", "sides": ["right"]}})     # the original single-hand setup
CFG_INS = config_mod.load(overrides={"robot": {"model": "inspire", "sides": ["right", "left"]}})
ASPECT = 4 / 3


def cal_neutral():
    c = Calibration()
    c.palm_uv0, c.size0, c.R0 = [0.5 * ASPECT, 0.5], 0.2, np.eye(3).tolist()
    c.done["neutral"] = True
    return c


def make_state(t, dx=0.0, dy=0.0, scale=1.0, curl=0.0, pinch=0.0, tracked=True, left=False, curls=None):
    w, im = synthetic_hand(curl=curls or (curl,) * 4, thumb=curl, pinch=pinch, center=(0.5 + dx / ASPECT, 0.5 + dy),
                           scale=scale, aspect=ASPECT, left=left)
    st = extract(im, w, "Left" if left else "Right", 1.0, ASPECT)
    st.t_capture = t
    st.tracked = tracked and st.tracked
    return st


class FakeSource:
    def __init__(self):
        self.seq, self.state = 0, None

    def push(self, st):
        self.seq += 1
        self.state = st

    def get(self):
        return self.seq, self.state


_SIM = {}


def user_cal(left=False):
    """Neutral reference + the o / f / p captures a user would do, for one synthetic hand."""
    cal = cal_neutral()
    cb = Calibrator(cal, samples=5)
    for kind, st in (("open", make_state(0, left=left)), ("fist", make_state(0, curl=1.0, left=left)),
                     ("pinch", make_state(0, pinch=1.0, left=left))):
        cb.start(kind)
        for _ in range(5):
            cb.feed(st)
    return cal


def sim(cfg=CFG):
    """Fresh model/data/controller per call (model build is cached per robot model). Every side is calibrated."""
    key = cfg.robot.model
    if key not in _SIM:
        _SIM[key] = build_sim(cfg)
    m, _, info = _SIM[key]
    d = mujoco.MjData(m)
    gs.reset(m, d, info)
    src = FakeSource()
    cals = {s: user_cal(left=s == "left") for s in config_mod.active_sides(cfg)}
    ctrl = Controller(m, d, info, cfg, src.get, calibration=cals)
    return m, d, info, src, ctrl


def run(m, d, src, ctrl, seconds, feed=None, n_ctrl=5):
    """Step physics; at each control tick optionally push a new packet from feed(t) -> HandState|None."""
    dt, k = m.opt.timestep, 0
    while d.time < seconds + run.t0:
        if k % n_ctrl == 0:
            if feed is not None:
                st = feed(d.time)
                if st is not None:
                    src.push(st)
            ctrl.update(d.time, n_ctrl * dt)
        mujoco.mj_step(m, d)
        k += 1
    run.t0 = d.time


run.t0 = 0.0


def fresh_run(m, d):
    run.t0 = d.time


class TestMappingAndCalibration(unittest.TestCase):
    def test_axis_map_signs_and_zero(self):
        c = cal_neutral()
        self.assertTrue(np.allclose(map_position(c.palm_uv0, c.size0, c, CFG), 0))
        base = np.array(c.palm_uv0)
        right = map_position(base + [0.2, 0], c.size0, c, CFG)       # hand to the screen right -> robot right (-y)
        up = map_position(base + [0, -0.2], c.size0, c, CFG)         # hand up on screen -> robot up (+z)
        near = map_position(base, c.size0 * 1.3, c, CFG)             # hand toward camera -> robot forward (+x)
        self.assertLess(right[1], 0); self.assertAlmostEqual(right[0], 0); self.assertAlmostEqual(right[2], 0)
        self.assertGreater(up[2], 0)
        self.assertGreater(near[0], 0)

    def test_position_bounded_by_workspace(self):
        c, half = cal_neutral(), np.array(CFG.position.workspace_half_extent)
        for uv, s in (([50, 50], 1e3), ([-50, -50], 1e-3), ([1e6, -1e6], 1.0)):
            d = map_position(np.array(c.palm_uv0) + uv, c.size0 * s, c, CFG)
            self.assertTrue(np.all(np.abs(d) <= half + 1e-12), d)

    def test_deadzone(self):
        c = cal_neutral()
        d = map_position(np.array(c.palm_uv0) + [CFG.position.deadzone * 0.5, 0], c.size0, c, CFG)
        self.assertTrue(np.allclose(d, 0))

    def test_no_neutral_means_no_motion(self):
        self.assertTrue(np.allclose(map_position([0.9, 0.9], 0.3, Calibration(), CFG), 0))

    def test_orientation_clamped(self):
        c = cal_neutral()
        a = np.deg2rad(120)
        Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        R = map_orientation(Ry, c, CFG)
        ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        self.assertLessEqual(np.rad2deg(ang), CFG.orientation.max_deg + 1e-6)
        self.assertTrue(np.allclose(R @ R.T, np.eye(3), atol=1e-9))
        self.assertTrue(np.allclose(map_orientation(np.eye(3), c, CFG), np.eye(3)))

    def test_calibrator_averages_and_normalises(self):
        cal = Calibration(); cb = Calibrator(cal, samples=5)
        cb.start("open")
        done = None
        for i in range(5):
            done = cb.feed(make_state(i * 0.03, curl=0.0))
        self.assertEqual(done, "open"); self.assertTrue(cal.done["open"])
        cb.start("fist")
        for i in range(5):
            cb.feed(make_state(i * 0.03, curl=1.0))
        self.assertTrue(cal.done["fist"])
        for curl, lo, hi in ((0.0, 0.0, 0.05), (1.0, 0.95, 1.0)):
            st = make_state(0, curl=curl)
            for f in ("index", "middle", "ring", "pinky"):
                v = curl_norm(cal, f, st.curl_raw[f]); self.assertTrue(lo - 1e-9 <= v <= hi + 1e-9, (curl, f, v))
        for raw in (-5, 0, 0.5, 5, float("nan")):                   # always bounded
            for fn in (lambda r: curl_norm(cal, "index", r), lambda r: thumb_norm(cal, r), lambda r: pinch_norm(cal, r)):
                self.assertTrue(0.0 <= fn(raw) <= 1.0)

    def test_calibrator_ignores_untracked(self):
        cal = Calibration(); cb = Calibrator(cal, samples=3); cb.start("neutral")
        for _ in range(10):
            self.assertIsNone(cb.feed(make_state(0, tracked=False)))
        self.assertFalse(cal.done["neutral"]); self.assertEqual(cb.progress, 0.0)
        for i in range(3):
            cb.feed(make_state(i * 0.03))
        self.assertTrue(cal.done["neutral"])
        self.assertAlmostEqual(abs(np.linalg.det(cal.R0)), 1.0, places=6)

    def test_range_capture_sets_bounded_gains(self):
        c = Calibration()
        cb = Calibrator(c, samples=3, range_samples=20)
        self.assertFalse(cb.start("range"), "range needs neutral first")
        cb.start("neutral")                              # captured from the hand itself, as in real use
        for _ in range(3):
            cb.feed(make_state(0))
        self.assertTrue(np.allclose(cue_gains(c, CFG), [CFG.position.gain, CFG.position.gain, CFG.position.depth_gain]))
        self.assertTrue(cb.start("range"))
        for i in range(20):                              # sweep: +-0.3 frame-heights horizontally, +-0.15 vertically
            a = 2 * np.pi * i / 20
            cb.feed(make_state(0, dx=0.3 * np.cos(a), dy=0.15 * np.sin(a), scale=1 + 0.1 * np.sin(a)))
        self.assertTrue(c.done["range"])
        g = cue_gains(c, CFG)
        base = np.array([CFG.position.gain, CFG.position.gain, CFG.position.depth_gain])
        self.assertTrue(np.all(g >= base * CFG.position.range_gain_min - 1e-12) and np.all(g <= base * CFG.position.range_gain_max + 1e-12))
        self.assertGreater(g[1], g[0], "smaller vertical sweep -> larger vertical gain")
        # sweeping the captured span reaches (close to) the workspace edge, never beyond it
        half = np.array(CFG.position.workspace_half_extent)
        right = map_position(np.array(c.palm_uv0) + [c.cue_span[0], 0], c.size0, c, CFG)
        self.assertGreater(abs(right[1]), 0.5 * half[1]); self.assertLessEqual(abs(right[1]), half[1] + 1e-12)
        # a tiny sweep cannot blow the gain up past the cap
        c.cue_span = [1e-4, 1e-4, 1e-4]
        self.assertTrue(np.allclose(cue_gains(c, CFG), base * CFG.position.range_gain_max))

    def test_old_calibration_file_loads(self):
        import json, tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(dict(palm_uv0=[0.6, 0.5], size0=0.2, done=dict(neutral=True, open=False, fist=False, pinch=False)), f)
        c = Calibration.load(f.name)
        self.assertTrue(c.has_neutral); self.assertFalse(c.done["range"]); self.assertIsNone(c.cue_span)

    def test_features_monotonic(self):
        r = [make_state(0, curl=c).curl_raw["index"] for c in (0, 0.25, 0.5, 0.75, 1.0)]
        self.assertTrue(all(a > b for a, b in zip(r, r[1:])), r)
        p = [make_state(0, pinch=x).pinch_raw for x in (0, 0.5, 1.0)]
        self.assertTrue(all(a > b for a, b in zip(p, p[1:])), p)

    def test_handedness_label_does_not_flip_palm_frame(self):
        a = np.deg2rad(30)
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        w, im = synthetic_hand(R=R)
        self.assertTrue(np.allclose(extract(im, w, "Left").R_palm, extract(im, w, "Right").R_palm))

    def test_user_hand_label_for_mirrored_frames(self):
        from livehand.tracking import user_hand
        self.assertEqual(user_hand("Left", True), "Right")     # measured: user's right hand, mirrored -> 'Left'
        self.assertEqual(user_hand("Right", True), "Left")
        self.assertEqual(user_hand("Left", False), "Left")

    def test_extract_rejects_nan(self):
        w, im = synthetic_hand()
        im[3, 0] = np.nan
        self.assertFalse(extract(im, w).tracked)


class TestSafety(unittest.TestCase):
    def test_limiter_clamps_and_slews(self):
        lim = Limiter([-1, -1], [1, 1], [2.0, 2.0], [0, 0])
        out = lim.apply([5, -5], 0.1)
        self.assertTrue(np.allclose(out, [0.2, -0.2]))                 # slew: 2 rad/s * 0.1 s
        for _ in range(100):
            out = lim.apply([5, -5], 0.1)
        self.assertTrue(np.allclose(out, [1, -1]))                     # clamped to range
        held = lim.apply([np.nan, np.inf * 0], 0.1)
        self.assertTrue(np.isfinite(held).all()); self.assertEqual(lim.rejected, 2)

    def test_watchdog_jump_and_reacquire(self):
        wd = Watchdog(max_jump=0.2, stale_timeout=0.25, reacquire_frames=3)
        s = make_state(10.0)
        self.assertFalse(wd.update(10.0, s)); self.assertFalse(wd.update(10.0, s)); self.assertTrue(wd.update(10.0, s))
        self.assertTrue(wd.accept_target([0, 0, 0])); self.assertFalse(wd.accept_target([0.5, 0, 0]))
        self.assertEqual(wd.rejected_jumps, 1); self.assertFalse(wd.accept_target([np.nan, 0, 0]))
        self.assertFalse(wd.update(11.0, s))                           # stale packet (1 s old) -> tracking lost
        self.assertFalse(wd.tracking)
        for _ in range(3):
            wd.update(11.0, make_state(11.0))
        self.assertTrue(wd.tracking); self.assertTrue(wd.accept_target([0.5, 0, 0]))   # baseline reset after loss

    def test_watchdog_rejects_missing_hand(self):
        wd = Watchdog(0.2, 0.25, 1)
        self.assertFalse(wd.update(0, None)); self.assertFalse(wd.update(0, make_state(0, tracked=False)))


class TestModelWiring(unittest.TestCase):
    def test_mapped_names_exist_and_are_position_actuators(self):
        m, d, info, src, ctrl = sim()
        for j in gs.ARM:
            actuator_for_joint(m, j)
        for n in HAND_JOINTS:
            actuator_for_joint(m, f"right_hand_{n}_joint")
        with self.assertRaises(AssertionError):
            actuator_for_joint(m, "no_such_joint")

    def test_home_command_is_inside_ctrlrange(self):
        m, d, info, src, ctrl = sim()
        self.assertTrue(np.all(ctrl.ctrl_hold >= m.actuator_ctrlrange[:, 0]) and np.all(ctrl.ctrl_hold <= m.actuator_ctrlrange[:, 1]))
        for name, v, lo, hi in ctrl.clipped_home:      # only the known keyframe quirk (left thumb_1, 0.003 rad) may be clipped
            self.assertEqual(name, "left_hand_thumb_1_joint"); self.assertLess(abs(v - hi), 0.01)

    def test_gesture_poses_inside_ctrlrange(self):
        m, d, info, src, ctrl = sim()
        lo, hi = m.actuator_ctrlrange[ctrl.hand_act, 0], m.actuator_ctrlrange[ctrl.hand_act, 1]
        for pose in (CFG.hand.dex3.open, CFG.hand.dex3.fist, CFG.hand.dex3.pinch):
            q = ctrl.hand.clamp(np.array(pose))
            self.assertTrue(np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9))
            self.assertTrue(np.allclose(q, pose, atol=1e-6), "validated gesture pose must not be altered by clamping")
        q = ctrl.hand.clamp(np.array([9, -9, 9, -9, 9, -9, 9.0]))
        self.assertTrue(np.all(q >= lo) and np.all(q <= hi))

    def test_start_pose_has_no_arm_hand_penetration(self):
        m, d, info, src, ctrl = sim()
        mujoco.mj_forward(m, d)
        bad = []
        for x in d.contact[:d.ncon]:
            names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) for g in (x.geom1, x.geom2)]
            if x.dist < -0.001 and any(("right_" in n and "ankle" not in n and "hip" not in n) for n in names):
                bad.append((names, x.dist))
        self.assertEqual(bad, [])
        self.assertTrue(np.all(ctrl.home_arm >= ctrl.ik.lo - 1e-9) and np.all(ctrl.home_arm <= ctrl.ik.hi + 1e-9))

    def test_ik_reaches_interior_targets_within_limits(self):
        m, d, info, src, ctrl = sim()
        half = np.array(CFG.position.workspace_half_extent)
        for corner in ([0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [0.5, -0.5, 0.5], [0, 0, 0.8]):
            q = ctrl.home_arm.copy()
            target = ctrl.pos_home + np.array(corner) * half
            for _ in range(600):
                q = ctrl.ik.step(d.qpos, q, target, None, 0.01)
                self.assertTrue(np.all(q >= ctrl.ik.lo - 1e-9) and np.all(q <= ctrl.ik.hi + 1e-9))
            pos, _ = ctrl.ik.site_pose_local(d.qpos, q)
            self.assertLess(np.linalg.norm(pos - target), 0.005, (corner, pos - target))

    def test_ik_step_bounded_by_velocity(self):
        m, d, info, src, ctrl = sim()
        q = ctrl.home_arm.copy()
        q2 = ctrl.ik.step(d.qpos, q, ctrl.pos_home + [0.1, 0.1, 0.1], None, 0.01)
        self.assertTrue(np.all(np.abs(q2 - q) <= CFG.arm.max_joint_velocity * 0.01 + 1e-12))


class TestControlBehaviour(unittest.TestCase):
    def setUp(self):
        run.t0 = 0.0

    def test_disabled_by_default_holds_home(self):
        m, d, info, src, ctrl = sim()
        self.assertEqual(ctrl.mode, Mode.DISABLED)
        feed = lambda t: make_state(t, dx=0.3 * np.sin(t * 3), dy=-0.2, scale=1.4, curl=1.0)
        run(m, d, src, ctrl, 2.0, feed)
        self.assertTrue(np.allclose(d.ctrl, ctrl.ctrl_hold, atol=1e-9), "live input must be ignored while disabled")
        self.assertGreater(d.xmat[1].reshape(3, 3)[2, 2], 0.98)
        self.assertTrue(np.isfinite(d.qpos).all())

    def test_enable_needs_neutral_calibration(self):
        m, d, info, src, ctrl = sim()
        ctrl.cal.done["neutral"] = False
        ctrl.enable()
        self.assertEqual(ctrl.mode, Mode.DISABLED)
        self.assertIn("calibrate neutral", ctrl.status()["messages"][-1])

    def _live(self):
        m, d, info, src, ctrl = sim()
        t = lambda tt: make_state(tt)
        run(m, d, src, ctrl, 0.3, t)                       # establish tracking at the neutral pose
        ctrl.enable()
        return m, d, info, src, ctrl

    def test_enable_follows_and_disable_holds(self):
        m, d, info, src, ctrl = self._live()
        feed = lambda t: make_state(t, dx=0.15, dy=-0.1, curl=1.0)
        run(m, d, src, ctrl, 2.0, feed)
        self.assertEqual(ctrl.mode, Mode.ENABLED); self.assertTrue(ctrl.tracking_ok)
        off = ctrl.cart_cmd.copy()
        self.assertGreater(np.linalg.norm(off), 0.02, "robot target should follow the hand")
        self.assertTrue(np.all(np.abs(off) <= np.array(CFG.position.workspace_half_extent) + 1e-9))
        self.assertGreater(ctrl.hand_cmd[3], 1.0, "closed hand should curl the index finger")
        ctrl.disable()
        run(m, d, src, ctrl, 1.0, lambda t: make_state(t, dx=-0.3, curl=0.0))
        self.assertTrue(np.allclose(ctrl.cart_cmd, off, atol=1e-3), "disable must hold the current pose")

    def test_estop_freezes_and_needs_two_presses(self):
        m, d, info, src, ctrl = self._live()
        feed = lambda t: make_state(t, dx=0.2, dy=-0.15, curl=1.0)
        run(m, d, src, ctrl, 0.5, feed)
        ctrl.estop()
        self.assertEqual(ctrl.mode, Mode.ESTOP)
        run(m, d, src, ctrl, 0.4, feed)
        frozen = ctrl.cart_cmd.copy(); hand = ctrl.hand_cmd.copy()
        run(m, d, src, ctrl, 1.0, lambda t: make_state(t, dx=-0.3, dy=0.2, curl=0.0))
        self.assertTrue(np.allclose(ctrl.cart_cmd, frozen, atol=1e-6)); self.assertTrue(np.allclose(ctrl.hand_cmd, hand))
        ctrl.enable()                                      # first press only clears the e-stop
        self.assertEqual(ctrl.mode, Mode.DISABLED)
        ctrl.enable()
        self.assertEqual(ctrl.mode, Mode.ENABLED)

    def test_stale_tracking_holds_then_returns_home(self):
        m, d, info, src, ctrl = self._live()
        feed = lambda t: make_state(t, dx=0.2, dy=-0.1, curl=1.0)
        run(m, d, src, ctrl, 1.5, feed)
        pos = ctrl.cart_cmd.copy()
        self.assertGreater(np.linalg.norm(pos), 0.02)
        run(m, d, src, ctrl, 1.0, None)                    # no new packets at all (camera stalled)
        self.assertFalse(ctrl.tracking_ok)
        self.assertGreater(ctrl.status()["lost_for"], 0.5)
        self.assertTrue(np.allclose(ctrl.cart_cmd, pos, atol=2e-3), "hold during the first seconds of loss")
        run(m, d, src, ctrl, CFG.safety.lost_return_after + 1.0, None)
        self.assertLess(np.linalg.norm(ctrl.cart_cmd), np.linalg.norm(pos), "drifts back toward home after a long loss")
        # untracked packets (hand left the frame) behave the same way
        run(m, d, src, ctrl, 0.5, lambda t: make_state(t, tracked=False))
        self.assertFalse(ctrl.tracking_ok)
        # recovery: needs reacquire_frames consecutive good packets
        run(m, d, src, ctrl, 0.5, lambda t: make_state(t, dx=0.1))
        self.assertTrue(ctrl.tracking_ok)

    def test_jump_rejected_and_commands_always_bounded(self):
        m, d, info, src, ctrl = self._live()
        run(m, d, src, ctrl, 0.5, lambda t: make_state(t, dx=0.05))
        before = ctrl.cart_des.copy()
        src.push(make_state(d.time, dx=1.0, dy=0.8))       # teleporting glitch
        ctrl.update(d.time, 0.01)
        self.assertGreaterEqual(ctrl.wd.rejected_jumps, 1)
        self.assertTrue(np.allclose(ctrl.cart_des, before))
        # extreme but 'smooth' input: everything stays inside limits and slew bounds
        lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
        prev = d.ctrl.copy()
        rate = ctrl.limiter.rate
        for k in range(1500):
            t = d.time
            if k % 5 == 0:
                src.push(make_state(t, dx=0.0004 * k, dy=-0.0004 * k, scale=1 + 0.0006 * k, curl=min(1.0, k / 600)))
                ctrl.update(t, 0.01)
                self.assertTrue(np.all(d.ctrl >= lo - 1e-9) and np.all(d.ctrl <= hi + 1e-9))
                self.assertTrue(np.all(np.abs(d.ctrl - prev) <= rate * 0.01 + 1e-9))
                prev = d.ctrl.copy()
            mujoco.mj_step(m, d)
        self.assertTrue(np.all(np.abs(ctrl.cart_cmd) <= np.array(CFG.position.workspace_half_extent) + 1e-9))
        self.assertTrue(np.isfinite(d.qpos).all()); self.assertGreater(d.xmat[1].reshape(3, 3)[2, 2], 0.95)

    def test_hand_gestures_open_fist_pinch(self):
        m, d, info, src, ctrl = self._live()
        run(m, d, src, ctrl, 1.5, lambda t: make_state(t, curl=0.0))
        self.assertTrue(np.allclose(ctrl.hand_cmd, ctrl.hand_open, atol=0.05))
        run(m, d, src, ctrl, 1.5, lambda t: make_state(t, curl=1.0))
        self.assertTrue(np.allclose(ctrl.hand_cmd[3:], CFG.hand.dex3.fist[3:], atol=0.15), ctrl.hand_cmd)
        run(m, d, src, ctrl, 1.5, lambda t: make_state(t, curl=0.0, pinch=1.0))
        pinch = np.array(CFG.hand.dex3.pinch)
        self.assertTrue(np.allclose(ctrl.hand_cmd[[0, 1, 2, 3, 4]], pinch[:5], atol=0.2), ctrl.hand_cmd)


class TestVisionLifecycle(unittest.TestCase):
    def test_synthetic_source_publishes_and_stops(self):
        for cfg, sides in ((CFG, ["right"]), (CFG_INS, ["right", "left"])):
            cfg = config_mod.load(overrides={"camera": {"source": "synthetic"},
                                             "robot": {"model": cfg.robot.model, "sides": cfg.robot.sides}})
            v = VisionSystem(cfg); v.start()
            seq, pk = v.state_box.wait_newer(0, 2.0)
            self.assertEqual(sorted(pk), sorted(sides))
            for s, st in pk.items():
                self.assertTrue(st.tracked and st.synthetic)
            if len(sides) == 2:                          # the synthetic right hand is on the screen right
                self.assertGreater(pk["right"].palm_uv[0], pk["left"].palm_uv[0])
            self.assertTrue(v.stop())

    def test_missing_camera_reports_error_and_shuts_down(self):
        cfg = config_mod.load(overrides={"camera": {"source": "camera", "device": "/nonexistent/video.mp4"}})
        v = VisionSystem(cfg); v.start()
        deadline = time.time() + 5
        while not v.error and time.time() < deadline:
            time.sleep(0.05)
        self.assertIn("camera", v.error)
        self.assertIsNone(v.state_box.get()[1], "no hand state without a camera")
        self.assertTrue(v.stop(timeout=5)); self.assertIsNone(v.cap)

    def test_v4l2_ctrl_missing_device_is_harmless(self):
        from livehand.tracking import V4L2_CID_EXPOSURE_AUTO_PRIORITY, v4l2_ctrl
        self.assertIsNone(v4l2_ctrl(97, V4L2_CID_EXPOSURE_AUTO_PRIORITY))
        self.assertIsNone(v4l2_ctrl("clip.mp4", V4L2_CID_EXPOSURE_AUTO_PRIORITY, 0))

    def test_status_panel_prompts_and_draws(self):
        from livehand.status import draw_panel, next_step
        m, d, info, src, ctrl = sim(CFG_INS)
        st = ctrl.status()
        R, L = st["sides"]["right"]["calib"], st["sides"]["left"]["calib"]
        for c in (R, L):
            c.update(neutral=False, range=False, open=False, fist=False, pinch=False)
        st["cal_side"] = "right"
        self.assertIn("RIGHT HAND - STEP 1", next_step(st)); self.assertIn("right half", next_step(st))
        R["neutral"] = True
        self.assertIn("RIGHT HAND - STEP 2", next_step(st))
        R.update(open=True, fist=True, pinch=True)
        self.assertIn("press c to calibrate the LEFT hand", next_step(st))
        st["cal_side"] = "left"
        self.assertIn("LEFT HAND - STEP 1", next_step(st)); self.assertIn("left half", next_step(st))
        L.update(neutral=True, open=True, fist=True, pinch=True)
        self.assertIn("press e", next_step(st))
        st["mode"] = "E-STOP"
        self.assertIn("E-STOP", next_step(st))
        out = draw_panel(np.zeros((480, 640, 3), np.uint8), st, 15.0, 15.0, "", 9.0)
        self.assertEqual(out.shape[1], 960)
        self.assertEqual(draw_panel(np.zeros((480, 640, 3), np.uint8), sim()[4].status(), 15.0, 15.0).shape[1], 960)

    def test_latest_box_never_queues(self):
        from livehand.buffer import LatestBox
        b = LatestBox()
        for i in range(1000):
            b.put(i)
        self.assertEqual(b.get(), (1000, 999))
        self.assertEqual(b.wait_newer(1000, 0.01), (1000, None))


def two_hands(t, r=None, l=None):
    """Packet with a synthetic right hand at the screen right and a left hand at the screen left (None = not seen)."""
    pk = {}
    for side, kw, x in (("right", r, 0.2), ("left", l, -0.2)):
        if kw is None:
            st = make_state(t, tracked=False)
        else:
            kw = dict(kw); kw["dx"] = x + kw.get("dx", 0.0)
            st = make_state(t, left=side == "left", **kw)
        pk[side] = st
    return pk


def inspire_live():
    """Inspire, both sides: neutral captured from the actual synthetic hands (as a user would), then enabled."""
    m, d, info, src, ctrl = sim(CFG_INS)
    run.t0 = 0.0
    for s, x in (("right", 0.2), ("left", -0.2)):
        cb = Calibrator(ctrl.sides[s].cal, samples=3)
        cb.start("neutral")
        for _ in range(3):
            cb.feed(make_state(0, dx=x, left=s == "left"))
    run(m, d, src, ctrl, 0.3, lambda t: two_hands(t, {}, {}))
    ctrl.enable()
    return m, d, info, src, ctrl


def palm_local(m, d, ctrl, side):
    sd = ctrl.sides[side]
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    Rp = d.xmat[pel].reshape(3, 3)
    return Rp.T @ (d.site_xpos[sd.ik.site] - d.xpos[pel])


def bad_contacts(m, d, depth=-0.001):
    """Penetrating contacts that involve an arm or hand body (feet on the floor are fine)."""
    out = []
    for x in d.contact[:d.ncon]:
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "world" for g in (x.geom1, x.geom2)]
        if x.dist < depth and any(n.startswith(("R_", "L_")) or any(k in n for k in ("shoulder", "elbow", "wrist")) for n in names):
            out.append((names, round(x.dist * 1000, 1)))
    return out


class TestInspireModel(unittest.TestCase):
    def test_both_sides_wired_to_position_actuators(self):
        m, d, info, src, ctrl = sim(CFG_INS)
        self.assertEqual(list(ctrl.sides), ["right", "left"])
        for s, S in (("right", "R"), ("left", "L")):
            for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"):
                actuator_for_joint(m, f"{s}_{j}_joint")
            for n in INSPIRE_JOINTS:
                actuator_for_joint(m, f"{S}_{n}_joint")
            self.assertEqual(len(ctrl.sides[s].hand_act), 6)
        self.assertEqual(ctrl.clipped_home, [], "Inspire home command must be inside every ctrlrange")

    def test_home_pose_validated(self):
        m, d, info, src, ctrl = sim(CFG_INS)
        mujoco.mj_forward(m, d)
        for s, mirror in (("right", 1), ("left", -1)):
            sd = ctrl.sides[s]
            nominal = np.array(INSPIRE_HOME_PALM) * [1, mirror, 1]
            self.assertLess(np.linalg.norm(palm_local(m, d, ctrl, s) - nominal), 0.01)
            self.assertGreater(min(np.min(sd.home_arm - sd.ik.lo), np.min(sd.ik.hi - sd.home_arm)), 0.35)
        self.assertEqual(bad_contacts(m, d), [])
        p0 = {s: palm_local(m, d, ctrl, s) for s in ctrl.sides}
        for _ in range(1500):                                  # 3 s holding the home commands
            mujoco.mj_step(m, d)
        self.assertGreater(d.xmat[1].reshape(3, 3)[2, 2], 0.99)
        for s in ctrl.sides:
            self.assertLess(np.linalg.norm(palm_local(m, d, ctrl, s) - p0[s]), 0.01)

    def test_thumb_tracks_after_palm_contact_exclusion(self):
        m, d, info, src, ctrl = sim(CFG_INS)
        for S in "RL":
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{S}_thumb_proximal_pitch_joint")
            d.ctrl[i] = 0.45
        for _ in range(1000):
            mujoco.mj_step(m, d)
        for S in "RL":
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{S}_thumb_proximal_pitch_joint")
            self.assertAlmostEqual(d.qpos[m.jnt_qposadr[j]], 0.45, delta=0.01)

    def test_gesture_poses_in_range_and_penetration_free(self):
        from livehand.tools.find_inspire_poses import Hand
        m, d, info, src, ctrl = sim(CFG_INS)
        prof = CFG_INS.hand.inspire
        for s in ("right", "left"):
            sd = ctrl.sides[s]
            lo, hi = m.actuator_ctrlrange[sd.hand_act, 0], m.actuator_ctrlrange[sd.hand_act, 1]
            h = Hand(m, mujoco.MjData(m), s, CFG_INS)
            for name in ("open", "fist", "pinch"):
                q = np.array(getattr(prof, name))
                self.assertTrue(np.all(q >= lo) and np.all(q <= hi), (s, name))
                self.assertTrue(np.allclose(sd.hand.clamp(q), q), "validated gesture pose must not be altered by clamping")
                self.assertEqual(h.new_pen(q), set(), (s, name))
            P = np.array(prof.pinch)
            self.assertLess(np.linalg.norm(h.tip(P, "thumb") - h.tip(P, "index")), 0.015, "pinch brings thumb and index together")

    def test_workspace_corners_reachable_without_body_collisions(self):
        m, d, info, src, ctrl = sim(CFG_INS)
        half = np.array(CFG_INS.position.workspace_half_extent)     # the full box the mapping can command
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        for s, sd in ctrl.sides.items():
            for c in corners:
                q, target = sd.home_arm.copy(), sd.pos_home + c * half
                for _ in range(800):
                    q = sd.ik.step(d.qpos, q, target, None, 0.01)
                pos, _ = sd.ik.site_pose_local(d.qpos, q)
                self.assertLess(np.linalg.norm(pos - target), 0.005, (s, c))
                d2 = mujoco.MjData(m); d2.qpos[:] = d.qpos; d2.qpos[sd.ik.qadr] = q
                mujoco.mj_forward(m, d2)
                self.assertEqual(bad_contacts(m, d2), [], (s, c))


class TestTwoHands(unittest.TestCase):
    def test_each_hand_drives_its_own_arm(self):
        m, d, info, src, ctrl = inspire_live()
        R, L = ctrl.sides["right"], ctrl.sides["left"]
        run(m, d, src, ctrl, 1.5, lambda t: two_hands(t, dict(dx=0.12, dy=-0.1), dict(dx=-0.12, dy=0.1)))
        self.assertTrue(R.tracking_ok and L.tracking_ok)
        self.assertLess(R.cart_cmd[1], -0.02); self.assertGreater(R.cart_cmd[2], 0.02)     # screen right/up -> robot right/up
        self.assertGreater(L.cart_cmd[1], 0.02); self.assertLess(L.cart_cmd[2], -0.02)     # screen left/down -> robot left/down
        for s in ctrl.sides:                                   # the simulated palms really went there
            sd = ctrl.sides[s]
            self.assertLess(np.linalg.norm(palm_local(m, d, ctrl, s) - (sd.pos_home + sd.cart_cmd)), 0.015)
        self.assertGreater(d.xmat[1].reshape(3, 3)[2, 2], 0.98)

    def test_independent_fingers(self):
        m, d, info, src, ctrl = inspire_live()
        run(m, d, src, ctrl, 1.5, lambda t: two_hands(t, dict(curls=(1, 0, 0, 0)), dict(curls=(0, 0, 0, 1))))
        R, L = ctrl.sides["right"].hand_cmd, ctrl.sides["left"].hand_cmd       # [thumb yaw, thumb pitch, index, middle, ring, pinky]
        self.assertGreater(R[2], 1.0); self.assertLess(max(R[3], R[4], R[5]), 0.5)
        self.assertGreater(L[5], 1.0); self.assertLess(max(L[2], L[3], L[4]), 0.5)

    def test_missing_hand_holds_its_arm_while_other_moves(self):
        m, d, info, src, ctrl = inspire_live()
        run(m, d, src, ctrl, 1.5, lambda t: two_hands(t, dict(dx=0.12), None))
        self.assertTrue(ctrl.sides["right"].tracking_ok); self.assertFalse(ctrl.sides["left"].tracking_ok)
        self.assertGreater(np.linalg.norm(ctrl.sides["right"].cart_cmd), 0.02)
        self.assertLess(np.linalg.norm(ctrl.sides["left"].cart_cmd), 0.005)

    def test_palms_never_closer_than_min_separation(self):
        m, d, info, src, ctrl = inspire_live()
        min_sep = CFG_INS.safety.min_hand_separation
        R, L = ctrl.sides["right"], ctrl.sides["left"]
        gaps = []
        def feed(t):                                            # both hands sweep toward the middle and past it
            k = min(1.0, t / 2.0)
            return two_hands(t, dict(dx=-0.45 * k), dict(dx=0.45 * k))
        for _ in range(60):
            run(m, d, src, ctrl, 0.05, feed)
            gaps.append((L.pos_home[1] + L.cart_cmd[1]) - (R.pos_home[1] + R.cart_cmd[1]))
        self.assertGreater(ctrl.separations, 0, "the inputs must actually have pushed the hands together")
        self.assertGreaterEqual(min(gaps), min_sep - 1e-9)
        self.assertEqual(bad_contacts(m, d), [])

    def test_calibration_one_hand_at_a_time_right_first(self):
        """The user shows one hand and types with the other: right is calibrated first, then it moves to the left."""
        m, d, info, src, ctrl = sim(CFG_INS)
        for sd in ctrl.sides.values():                      # start from nothing, like a first session
            sd.cal.update_from(Calibration())
        ctrl.cal_side = "right"
        run.t0 = 0.0
        right_only = lambda **kw: (lambda t: two_hands(t, kw, None))
        left_only = lambda **kw: (lambda t: two_hands(t, None, kw))
        run(m, d, src, ctrl, 0.3, right_only())
        for key, kw in (("neutral", {}), ("open", {}), ("fist", dict(curl=1.0)), ("pinch", dict(pinch=1.0))):
            ctrl.calibrate(key)
            self.assertTrue(ctrl.sides["right"].calibrator.active, key)
            self.assertFalse(ctrl.sides["left"].calibrator.active, "only the hand being calibrated captures")
            run(m, d, src, ctrl, 0.6, right_only(**kw))     # 20 samples at the 100 Hz test feed
        R, L = ctrl.sides["right"].cal, ctrl.sides["left"].cal
        self.assertTrue(all(R.done[k] for k in ("neutral", "open", "fist", "pinch")))
        self.assertFalse(L.has_neutral)
        self.assertEqual(ctrl.cal_side, "left", "after the right hand is done, calibration moves to the left hand")
        self.assertIn("Now the LEFT hand", " ".join(ctrl.status()["messages"]))
        ctrl.calibrate("neutral")                            # left hand not shown yet: refused with a hint
        self.assertFalse(ctrl.sides["left"].calibrator.active)
        self.assertIn("LEFT hand not visible", ctrl.status()["messages"][-1])
        run(m, d, src, ctrl, 0.3, left_only())
        ctrl.calibrate("neutral")
        run(m, d, src, ctrl, 0.6, left_only())
        self.assertTrue(L.has_neutral)
        self.assertLess(L.palm_uv0[0], R.palm_uv0[0], "left neutral is on the screen left of the right neutral")
        ctrl.switch_cal_side()                               # c: back to the right hand (e.g. to redo it)
        self.assertEqual(ctrl.cal_side, "right")

    def test_enable_with_one_calibrated_hand_holds_the_other(self):
        m, d, info, src, ctrl = sim(CFG_INS)
        run.t0 = 0.0
        ctrl.sides["left"].cal.done["neutral"] = False
        run(m, d, src, ctrl, 0.3, lambda t: two_hands(t, {}, {}))
        ctrl.enable()
        self.assertEqual(ctrl.mode, Mode.ENABLED)
        run(m, d, src, ctrl, 1.0, lambda t: two_hands(t, dict(dx=0.1), dict(dx=-0.1)))
        self.assertLess(np.linalg.norm(ctrl.sides["left"].cart_cmd), 1e-6)
        self.assertGreater(np.linalg.norm(ctrl.sides["right"].cart_cmd), 0.02)

    def test_calibration_set_roundtrip_and_old_format(self):
        import json, tempfile
        from livehand.calibration import load_set, save_set
        a, b = user_cal(), user_cal(left=True)
        b.size0 = 0.33
        path = tempfile.mktemp(suffix=".json")
        save_set(path, {"right": a, "left": b})
        got = load_set(path, ["right", "left"])
        self.assertEqual(got["left"].size0, 0.33); self.assertTrue(got["right"].has_neutral)
        a.save(path)                                            # a single-hand file from before two-hand support
        got = load_set(path, ["right", "left"])
        self.assertTrue(got["right"].has_neutral); self.assertFalse(got["left"].has_neutral)


class TestHandAssigner(unittest.TestCase):
    def test_two_hands_by_screen_position(self):
        from livehand.tracking import HandAssigner
        a = HandAssigner(mirrored=True)
        self.assertEqual(a([("Left", 0.7), ("Left", 0.3)]), {"right": 0, "left": 1})      # labels ignored
        self.assertEqual(HandAssigner(mirrored=False)([("Right", 0.7), ("Right", 0.3)]), {"right": 1, "left": 0})

    def test_single_hand_flicker_and_switch(self):
        from livehand.tracking import HandAssigner
        a = HandAssigner(mirrored=True, switch_after=5)
        self.assertEqual(a([("Left", 0.65)]), {"right": 0}, "a new hand takes its screen half, not a (wrong) label")
        self.assertEqual(a([("Right", 0.65)]), {"right": 0})   # label agrees: disagreement count resets
        for i in range(4):                                      # 4 disagreeing labels: continuity still wins
            self.assertEqual(a([("Left", 0.65 + 0.005 * i)]), {"right": 0})
        self.assertEqual(a([("Left", 0.67)]), {"left": 0}, "the 5th consecutive disagreeing label switches the side")
        self.assertEqual(a([]), {})
        self.assertEqual(a([("Left", 0.3)]), {"left": 0})


if __name__ == "__main__":
    unittest.main()
