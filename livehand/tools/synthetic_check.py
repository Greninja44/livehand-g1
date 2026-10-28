"""Real-time end-to-end check with the SYNTHETIC hand source (not a webcam test).

Runs the real threads + Controller + MuJoCo for ~14 s: disabled -> enabled (live mirroring) -> e-stop, and records
bounds/stability/latency to logs/livehand_synthetic_check.json. With --viewer it shows the MuJoCo window so you can watch.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from livehand import config as config_mod
from livehand.calibration import Calibration, Calibrator
from livehand.features import extract, synthetic_hand
from livehand.sim import Controller, build_sim
from livehand.tracking import VisionSystem

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--seconds", type=float, default=14.0)
    a = ap.parse_args()
    cfg = config_mod.load(overrides={"camera": {"source": "synthetic"}})
    aspect = cfg.camera.width / cfg.camera.height
    m, d, info = build_sim()
    vis = VisionSystem(cfg)
    cal = Calibration()
    cb = Calibrator(cal, 5)
    def snap(**kw):
        w, im = synthetic_hand(center=(0.5, 0.5), aspect=aspect, **kw)
        return extract(im, w, "Right", 1.0, aspect)
    for kind, st in (("neutral", snap()), ("open", snap()), ("fist", snap(curl=(1,) * 4, thumb=1.0)), ("pinch", snap(pinch=1.0))):
        cb.start(kind)
        for _ in range(5):
            cb.feed(st)
    ctrl = Controller(m, d, info, cfg, vis.state_box.get, calibration=cal)
    vis.start()
    viewer = mujoco.viewer.launch_passive(m, d) if a.viewer else None
    if viewer:
        site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, cfg.arm.site)
        viewer.cam.lookat[:] = d.site_xpos[site]; viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.4, 150, -15
    dt = m.opt.timestep
    n_ctrl = round(1 / cfg.sim.control_hz / dt)
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    rec = dict(ctrl_in_range=True, max_slew_ratio=0.0, min_upright=1.0, max_offset_m=0.0, max_tracking_err_m=0.0, hand_min=np.full(7, 9.0), hand_max=np.full(7, -9.0),
               hold_ok_disabled=True, estop_frozen=True)
    site, pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, cfg.arm.site), mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    prev, n, wall0 = d.ctrl.copy(), 0, time.perf_counter()
    T = a.seconds
    marks = dict(enable=2.0, estop=T - 3.0)
    state_before_estop = None
    try:
        while d.time < T:
            t = d.time
            if n % n_ctrl == 0:
                if t >= marks["enable"] and ctrl.mode.value == "DISABLED" and "enabled" not in rec:
                    ctrl.enable(); rec["enabled"] = ctrl.mode.value == "ENABLED"
                if t >= marks["estop"] and "estop" not in rec:
                    ctrl.estop(); rec["estop"] = True; state_before_estop = None
                ctrl.update(time.monotonic(), n_ctrl * dt)
                rec["ctrl_in_range"] &= bool(np.all(d.ctrl >= lo - 1e-9) and np.all(d.ctrl <= hi + 1e-9))
                rec["max_slew_ratio"] = max(rec["max_slew_ratio"], float(np.max(np.abs(d.ctrl - prev) / (ctrl.limiter.rate * n_ctrl * dt))))
                prev = d.ctrl.copy()
                if ctrl.mode.value == "ENABLED":
                    Rp = d.xmat[pel].reshape(3, 3)
                    actual = Rp.T @ (d.site_xpos[site] - d.xpos[pel])        # simulated hand site in the pelvis frame
                    rec["max_tracking_err_m"] = max(rec["max_tracking_err_m"], float(np.linalg.norm(actual - (ctrl.pos_home + ctrl.cart_cmd))))
                    rec["max_offset_m"] = max(rec["max_offset_m"], float(np.linalg.norm(ctrl.cart_cmd)))
                if ctrl.mode.value == "DISABLED" and t < marks["enable"]:
                    rec["hold_ok_disabled"] &= bool(np.allclose(d.ctrl, ctrl.ctrl_hold, atol=1e-9))
                rec["hand_min"] = np.minimum(rec["hand_min"], d.ctrl[ctrl.hand_act]); rec["hand_max"] = np.maximum(rec["hand_max"], d.ctrl[ctrl.hand_act])
                if "estop" in rec and t > marks["estop"] + 0.5:
                    if state_before_estop is None:
                        state_before_estop = d.ctrl.copy()
                    rec["estop_frozen"] &= bool(np.allclose(d.ctrl, state_before_estop, atol=2e-3))
            mujoco.mj_step(m, d); n += 1
            rec["min_upright"] = min(rec["min_upright"], float(d.xmat[1].reshape(3, 3)[2, 2]))
            if viewer and n % 8 == 0:
                viewer.sync()
            lag = d.time - (time.perf_counter() - wall0)
            if lag > 0:
                time.sleep(lag)
            if viewer and not viewer.is_running():
                break
    finally:
        st = ctrl.status()
        stopped = vis.stop()
        if viewer: viewer.close()
    rtf = d.time / (time.perf_counter() - wall0)
    out = dict(source="SYNTHETIC (not a webcam)", sim_time=d.time, real_time_factor=rtf,
               **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in rec.items()},
               cam_fps=vis.cam_fps.v, latency_ms_median=st["latency_ms_median"], latency_ms_p95=st["latency_ms_p95"],
               final_mode=st["mode"], rejected_jumps=st["rejected_jumps"], vision_threads_stopped=stopped,
               final_offset_m=st["offset"].tolist(), finite=bool(np.isfinite(d.qpos).all()))
    (ROOT / "logs" / "livehand_synthetic_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    ok = (out["ctrl_in_range"] and out["max_slew_ratio"] <= 1.0001 and out["min_upright"] > 0.95 and out["hold_ok_disabled"]
          and out["estop_frozen"] and out["finite"] and stopped and rec.get("enabled"))
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import mujoco.viewer  # noqa: F401
    sys.exit(main())
