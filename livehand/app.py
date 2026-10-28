"""Entry point: `python -m livehand`. Wires vision threads, the MuJoCo loop (main thread), preview window and keys."""
import argparse
import signal
import sys
import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from . import config as config_mod
from .calibration import Calibration
from .sim import Controller, build_sim
from .status import draw_panel
from .tracking import VisionSystem

WIN = "LiveHand-G1 camera (focus this window for keys)"
KEYS = {ord("e"): "enable", 32: "estop", ord("n"): "cal_neutral", ord("o"): "cal_open", ord("f"): "cal_fist",
        ord("p"): "cal_pinch", ord("r"): "orient", ord("g"): "hand", ord("h"): "home", ord("s"): "save",
        ord("l"): "load", ord("q"): "quit", 27: "quit"}


def dispatch(ctrl, action, cal_path, quit_flag):
    if action == "enable":
        ctrl.enable()
    elif action == "estop":
        ctrl.estop()
    elif action.startswith("cal_"):
        ctrl.calibrate(action[4:])
    elif action == "orient":
        ctrl.toggle_orientation()
    elif action == "hand":
        ctrl.toggle_hand()
    elif action == "home":
        ctrl.home()
    elif action == "save":
        ctrl.cal.save(cal_path)
        ctrl.say(f"calibration saved to {cal_path.name}")
    elif action == "load":
        if cal_path.exists():
            new = Calibration.load(cal_path)
            ctrl.cal.__dict__.update(new.__dict__)
            ctrl.say("calibration loaded")
        else:
            ctrl.say("no saved calibration")
    elif action == "quit":
        quit_flag.append(True)


def parse(argv):
    ap = argparse.ArgumentParser(prog="python -m livehand", description=__doc__)
    ap.add_argument("--config", help="TOML overriding livehand/config.toml")
    ap.add_argument("--source", choices=["camera", "synthetic"], help="override camera.source")
    ap.add_argument("--camera", help="camera index or video file path (override camera.device)")
    ap.add_argument("--width", type=int); ap.add_argument("--height", type=int)
    ap.add_argument("--no-preview", action="store_true", help="no OpenCV window (keys only via the MuJoCo window)")
    ap.add_argument("--no-viewer", action="store_true", help="headless MuJoCo (benchmarks); implies --duration or Ctrl-C")
    ap.add_argument("--duration", type=float, default=0.0, help="exit after N wall seconds (0 = until closed)")
    ap.add_argument("--vision-only", action="store_true", help="run only the camera + MediaPipe + preview (no MuJoCo)")
    return ap.parse_args(argv)


def run_vision_only(cfg, args):
    vis = VisionSystem(cfg)
    vis.start()
    t0, quit_flag = time.time(), []
    try:
        while not quit_flag and (not args.duration or time.time() - t0 < args.duration):
            _, frame = vis.frame_box.get()
            seq, st = vis.state_box.get()
            if frame is not None and not args.no_preview:
                txt = f"cam {vis.cam_fps.v:4.1f} fps  infer {vis.inf_fps.v:4.1f} fps  hand {'YES' if st is not None and st.tracked else 'no'}"
                out = frame.copy()
                cv2.putText(out, txt + (f"  ERR {vis.error}" if vis.error else ""), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow(WIN, out)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    quit_flag.append(True)
            else:
                time.sleep(0.02)
            if args.no_preview and int(time.time() * 2) % 4 == 0:
                print(f"cam {vis.cam_fps.v:.1f} fps infer {vis.inf_fps.v:.1f} fps frames {vis.frames} err={vis.error!r}", flush=True)
                time.sleep(0.5)
    finally:
        ok = vis.stop()
        cv2.destroyAllWindows()
        print("vision stopped cleanly" if ok else "WARNING: vision threads did not stop", flush=True)
    return 0


def main(argv=None):
    args = parse(argv)
    over = {}
    if args.source: over.setdefault("camera", {})["source"] = args.source
    if args.camera is not None: over.setdefault("camera", {})["device"] = args.camera
    if args.width: over.setdefault("camera", {})["width"] = args.width
    if args.height: over.setdefault("camera", {})["height"] = args.height
    cfg = config_mod.load(args.config, over)
    if args.vision_only:
        return run_vision_only(cfg, args)

    m, d, info = build_sim()
    vis = VisionSystem(cfg)
    cal_path = config_mod.PKG / cfg.calibration.file
    cal = Calibration.load(cal_path) if cal_path.exists() else Calibration()
    ctrl = Controller(m, d, info, cfg, vis.state_box.get, calibration=cal)
    print(f"model: {m.nbody} bodies, {m.nu} actuators, dt={m.opt.timestep}; arm joints/hand actuators wired; "
          f"mirroring DISABLED until you press 'e' (after calibrating neutral with 'n')", flush=True)
    print(f"source={cfg.camera.source} device={cfg.camera.device}; calibration {'loaded' if cal.has_neutral else 'not found'}", flush=True)
    vis.start()

    quit_flag = []
    signal.signal(signal.SIGINT, lambda *_: quit_flag.append(True))
    key_q = []
    viewer_keys = {ord("E"): "enable", 32: "estop", ord("Q"): "quit"}
    on_key = lambda code: key_q.append(viewer_keys[code]) if code in viewer_keys else None
    viewer = None if args.no_viewer else mujoco.viewer.launch_passive(m, d, key_callback=on_key)
    if viewer is not None:
        site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, cfg.arm.site)
        viewer.cam.lookat[:] = d.site_xpos[site]; viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.4, 150, -15

    dt = m.opt.timestep
    n_ctrl = max(1, round(1.0 / cfg.sim.control_hz / dt))
    n_view = max(1, round(1.0 / cfg.sim.viewer_hz / dt))
    prev_dt = n_ctrl * dt
    wall0 = time.perf_counter(); n = 0
    next_preview = 0.0
    try:
        while not quit_flag and (viewer is None or viewer.is_running()):
            now = time.monotonic()
            if args.duration and time.perf_counter() - wall0 > args.duration:
                break
            if n % n_ctrl == 0:
                ctrl.update(now, prev_dt)
            mujoco.mj_step(m, d)
            n += 1
            if not np.isfinite(d.qpos).all():
                ctrl.estop()
                print("ERROR: non-finite simulation state; E-STOP", file=sys.stderr, flush=True)
                break
            if viewer is not None and n % n_view == 0:
                viewer.sync()
            if not args.no_preview and now >= next_preview:
                next_preview = now + 1.0 / cfg.sim.preview_hz
                _, frame = vis.frame_box.get()
                if frame is None:
                    frame = np.zeros((cfg.camera.height, cfg.camera.width, 3), np.uint8)
                cv2.imshow(WIN, draw_panel(frame, ctrl.status(), vis.cam_fps.v, vis.inf_fps.v, vis.error))
                k = cv2.waitKey(1) & 0xFF
                if k in KEYS:
                    key_q.append(KEYS[k])
            while key_q:
                dispatch(ctrl, key_q.pop(0), cal_path, quit_flag)
            if cfg.sim.realtime:
                lag = d.time - (time.perf_counter() - wall0)
                if lag > 0:
                    time.sleep(lag)
    finally:
        st = ctrl.status()
        ok = vis.stop()
        if viewer is not None:
            viewer.close()
        cv2.destroyAllWindows()
        print(f"shutdown: vision threads {'stopped' if ok else 'STILL RUNNING'}, camera released={vis.cap is None}, "
              f"sim_time={d.time:.1f}s, latency median {st['latency_ms_median']:.0f} ms p95 {st['latency_ms_p95']:.0f} ms "
              f"(capture->command)", flush=True)
    return 0
