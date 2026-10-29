"""Entry point: `python -m livehand`. Wires vision threads, the MuJoCo loop (main thread), preview window and keys."""
import argparse
import os
import signal
import sys
import time
from pathlib import Path

# Native Ubuntu (GNOME/Wayland) defaults, set before cv2/glfw load; export the variables yourself to override.
# The glfw wheel defaults to its Wayland build, whose MuJoCo window gets no title bar here (libdecor fails), while
# OpenCV's bundled Qt already runs through XWayland -- so put both windows on X11. Qt in the cv2 wheel ships no fonts.
if os.environ.get("DISPLAY"):
    os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
_FONTDIR = os.environ.get("QT_QPA_FONTDIR") or ("/usr/share/fonts/truetype/dejavu" if os.path.isdir("/usr/share/fonts/truetype/dejavu") else None)

import cv2

if _FONTDIR:                     # cv2's import points this at its own (empty) font dir; Qt reads it at the first window
    os.environ["QT_QPA_FONTDIR"] = _FONTDIR
import mujoco
import mujoco.viewer
import numpy as np

from . import config as config_mod
from .calibration import load_set, save_set
from .sim import Controller, build_sim
from .status import draw_panel
from .tracking import VisionSystem

WIN = "LiveHand-G1 camera (focus this window for keys)"
KEYS = {ord("e"): "enable", 32: "estop", ord("n"): "cal_neutral", ord("m"): "cal_range", ord("o"): "cal_open", ord("f"): "cal_fist",
        ord("p"): "cal_pinch", ord("c"): "cal_switch", ord("r"): "orient", ord("g"): "hand", ord("h"): "home", ord("s"): "save",
        ord("l"): "load", ord("q"): "quit", 27: "quit"}


def dispatch(ctrl, action, cal_path, quit_flag):
    if action == "enable":
        ctrl.enable()
    elif action == "estop":
        ctrl.estop()
    elif action == "cal_switch":
        ctrl.switch_cal_side()
    elif action.startswith("cal_"):
        ctrl.calibrate(action[4:])
    elif action == "orient":
        ctrl.toggle_orientation()
    elif action == "hand":
        ctrl.toggle_hand()
    elif action == "home":
        ctrl.home()
    elif action == "save":
        save_set(cal_path, {s: sd.cal for s, sd in ctrl.sides.items()})
        ctrl.say(f"calibration saved to {cal_path.name}")
    elif action == "load":
        if cal_path.exists():
            for s, c in load_set(cal_path, ctrl.side_names).items():
                ctrl.sides[s].cal.update_from(c)
            ctrl.cal_side = next((s for s in ctrl.side_names if not ctrl.cal_complete(s)), ctrl.cal_side)
            ctrl.say("calibration loaded")
        else:
            ctrl.say("no saved calibration")
    elif action == "quit":
        quit_flag.append(True)


def parse(argv):
    ap = argparse.ArgumentParser(prog="python -m livehand", description=__doc__)
    ap.add_argument("--config", help="TOML overriding livehand/config.toml")
    ap.add_argument("--robot", choices=["inspire", "dex3"], help="robot hands: inspire = 5-finger (default), dex3 = 3-finger")
    ap.add_argument("--hands", choices=["both", "right", "left"], help="which arms to drive (default both; dex3: right)")
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
            seq, pk = vis.state_box.get()
            if frame is not None and not args.no_preview:
                seen = [f"{s} YES" if st.tracked else f"{s} no" for s, st in (pk or {}).items()]
                txt = (f"cam {vis.cam_fps.v:4.1f} fps  infer {vis.inf_fps.v:4.1f} fps ({vis.inf_ms:.0f} ms)  "
                       f"{'  '.join(seen) or 'no hand'}  (q quits)")
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
    if args.robot: over.setdefault("robot", {})["model"] = args.robot
    if args.hands: over.setdefault("robot", {})["sides"] = ["right", "left"] if args.hands == "both" else [args.hands]
    cfg = config_mod.load(args.config, over)
    if not config_mod.active_sides(cfg):
        print(f"ERROR: robot {cfg.robot.model!r} cannot drive {cfg.robot.sides} (dex3 has the right arm wired only)", file=sys.stderr)
        return 2
    if args.vision_only:
        return run_vision_only(cfg, args)

    m, d, info = build_sim(cfg)
    vis = VisionSystem(cfg)
    cal_path = config_mod.PKG / cfg.calibration.file
    sides = config_mod.active_sides(cfg)
    cals = load_set(cal_path, sides) if cal_path.exists() else {}
    ctrl = Controller(m, d, info, cfg, vis.state_box.get, calibration=cals)
    print(f"robot: {cfg.robot.model}, arms: {', '.join(sides)}; {m.nu} actuators, dt={m.opt.timestep}; "
          f"mirroring DISABLED until you press 'e' (after calibrating neutral with 'n')", flush=True)
    loaded = [s for s, c in cals.items() if c.has_neutral]
    print(f"source={cfg.camera.source} device={cfg.camera.device}; calibration "
          f"{'loaded for ' + ', '.join(loaded) if loaded else 'not found'}", flush=True)
    vis.start()

    quit_flag = []
    signal.signal(signal.SIGINT, lambda *_: quit_flag.append(True))
    key_q = []
    viewer_keys = {ord("E"): "enable", 32: "estop", ord("Q"): "quit"}
    on_key = lambda code: key_q.append(viewer_keys[code]) if code in viewer_keys else None
    viewer = None if args.no_viewer else mujoco.viewer.launch_passive(m, d, key_callback=on_key)
    if viewer is not None:
        viewer.cam.lookat[:] = np.mean([d.site_xpos[sd.ik.site] for sd in ctrl.sides.values()], 0)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.4 + 0.3 * (len(sides) > 1), 160, -15

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
                cv2.imshow(WIN, draw_panel(frame, ctrl.status(), vis.cam_fps.v, vis.inf_fps.v, vis.error, vis.inf_ms))
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
        lat = (f"latency median {st['latency_ms_median']:.0f} ms p95 {st['latency_ms_p95']:.0f} ms (capture->command)"
               if np.isfinite(st["latency_ms_median"]) else "latency n/a (no hand was tracked)")
        print(f"shutdown: vision threads {'stopped' if ok else 'STILL RUNNING'}, camera released={vis.cap is None}, "
              f"sim_time={d.time:.1f}s, {lat}", flush=True)
    return 0
