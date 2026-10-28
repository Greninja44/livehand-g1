"""LiveHand tests (stdlib unittest). Run: python -m unittest livehand.tests.test_core -v
No webcam or display needed: sim tests use a fake state source and a simulated clock."""
import time
import unittest

import mujoco
import numpy as np

from livehand import config as config_mod
from livehand.calibration import Calibration, Calibrator, curl_norm, pinch_norm, thumb_norm
from livehand.features import extract, synthetic_hand
from livehand.retarget import HandRetargeter, map_orientation, map_position
from livehand.safety import Limiter, Mode, Watchdog
from livehand.sim import Controller, HAND_JOINTS, actuator_for_joint, build_sim, gs
from livehand.tracking import VisionSystem

CFG = config_mod.load()
ASPECT = 4 / 3


def cal_neutral():
    c = Calibration()
    c.palm_uv0, c.size0, c.R0 = [0.5 * ASPECT, 0.5], 0.2, np.eye(3).tolist()
    c.done["neutral"] = True
    return c


def make_state(t, dx=0.0, dy=0.0, scale=1.0, curl=0.0, pinch=0.0, tracked=True):
    w, im = synthetic_hand(curl=(curl,) * 4, thumb=curl, pinch=pinch, center=(0.5 + dx / ASPECT, 0.5 + dy), scale=scale, aspect=ASPECT)
    st = extract(im, w, "Right", 1.0, ASPECT)
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


def sim():
    """Fresh model/data/controller per call (model build is cached)."""
    if "m" not in _SIM:
        _SIM["m"], _SIM["d"], _SIM["info"] = build_sim()
    m, info = _SIM["m"], _SIM["info"]
    d = mujoco.MjData(m)
    gs.reset(m, d, info)
    src = FakeSource()
    cal = cal_neutral()                                  # neutral reference + the o / f / p captures a user would do
    cb = Calibrator(cal, samples=5)
    for kind, st in (("open", make_state(0)), ("fist", make_state(0, curl=1.0)), ("pinch", make_state(0, pinch=1.0))):
        cb.start(kind)
        for _ in range(5):
            cb.feed(st)
    ctrl = Controller(m, d, info, CFG, src.get, calibration=cal)
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

    def test_features_monotonic(self):
        r = [make_state(0, curl=c).curl_raw["index"] for c in (0, 0.25, 0.5, 0.75, 1.0)]
        self.assertTrue(all(a > b for a, b in zip(r, r[1:])), r)
        p = [make_state(0, pinch=x).pinch_raw for x in (0, 0.5, 1.0)]
        self.assertTrue(all(a > b for a, b in zip(p, p[1:])), p)

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
        for pose in (CFG.hand.open, CFG.hand.fist, CFG.hand.pinch):
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
        self.assertTrue(np.allclose(ctrl.hand_cmd[3:], CFG.hand.fist[3:], atol=0.15), ctrl.hand_cmd)
        run(m, d, src, ctrl, 1.5, lambda t: make_state(t, curl=0.0, pinch=1.0))
        pinch = np.array(CFG.hand.pinch)
        self.assertTrue(np.allclose(ctrl.hand_cmd[[0, 1, 2, 3, 4]], pinch[:5], atol=0.2), ctrl.hand_cmd)


class TestVisionLifecycle(unittest.TestCase):
    def test_synthetic_source_publishes_and_stops(self):
        cfg = config_mod.load(overrides={"camera": {"source": "synthetic"}})
        v = VisionSystem(cfg); v.start()
        seq, st = v.state_box.wait_newer(0, 2.0)
        self.assertIsNotNone(st); self.assertTrue(st.tracked and st.synthetic)
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

    def test_latest_box_never_queues(self):
        from livehand.buffer import LatestBox
        b = LatestBox()
        for i in range(1000):
            b.put(i)
        self.assertEqual(b.get(), (1000, 999))
        self.assertEqual(b.wait_newer(1000, 0.01), (1000, None))


if __name__ == "__main__":
    unittest.main()
