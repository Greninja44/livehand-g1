"""Camera + MediaPipe diagnostic (no MuJoCo, no window). Optionally records a replayable clip.

    python -m livehand.tools.camera_check                       # list devices, run the real vision threads for 8 s
    python -m livehand.tools.camera_check --camera 0 --seconds 15 --record logs/hand_clip.mp4
    python -m livehand --camera logs/hand_clip.mp4              # replay the clip through the full pipeline

Reports capture fps, inference fps/time, hand-detection rate, handedness and capture->state latency, and writes a
summary JSON next to the clip (or to logs/camera_check.json). The clip holds raw (un-mirrored) frames, so a replay
is mirrored and tracked exactly like the live camera.
"""
import argparse
import glob
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from livehand import config as config_mod
from livehand.tracking import V4L2_CID_EXPOSURE_AUTO_PRIORITY, VisionSystem, v4l2_ctrl

ROOT = Path(__file__).resolve().parents[2]


def list_devices():
    out = []
    for p in sorted(glob.glob("/dev/video*")):
        n = p.removeprefix("/dev/video")
        name = Path(f"/sys/class/video4linux/video{n}/name")
        out.append(dict(device=p, name=name.read_text().strip() if name.exists() else "?",
                        rw=os.access(p, os.R_OK | os.W_OK),
                        exposure_auto_priority=v4l2_ctrl(n, V4L2_CID_EXPOSURE_AUTO_PRIORITY)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default=None, help="index or video file (default: config camera.device)")
    ap.add_argument("--width", type=int); ap.add_argument("--height", type=int)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--record", help="write raw frames to this .mp4 (replay with python -m livehand --camera FILE)")
    ap.add_argument("--snapshot", help="save the last annotated (mirrored) frame with a detected hand to this image")
    ap.add_argument("--auto-exposure-priority", action="store_true", help="let auto-exposure lower the fps (driver default)")
    a = ap.parse_args()

    devs = list_devices()
    print("video devices:" if devs else "video devices: NONE (is the camera connected / enabled in BIOS?)")
    for d in devs:
        print(f"  {d['device']}: {d['name']}  rw={'yes' if d['rw'] else 'NO'}  exposure_auto_priority={d['exposure_auto_priority']}")
    if any(not d["rw"] for d in devs):
        print("  no read/write access: sudo usermod -aG video $USER, then log out and in")

    cam = {"source": "camera", "auto_exposure_priority": a.auto_exposure_priority}
    if a.camera is not None: cam["device"] = a.camera
    if a.width: cam["width"] = a.width
    if a.height: cam["height"] = a.height
    cfg = config_mod.load(overrides={"camera": cam})
    vis = VisionSystem(cfg)
    writer, rec_fps = None, None
    if a.record:
        Path(a.record).parent.mkdir(parents=True, exist_ok=True)
        stamps = []

        def record(frame):                           # capture thread; opens the writer once the real fps is known
            nonlocal writer, rec_fps
            stamps.append(time.monotonic())
            if writer is None:
                if len(stamps) < 15:
                    return
                rec_fps = round((len(stamps) - 1) / (stamps[-1] - stamps[0]), 1)
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), rec_fps, (w, h))
            writer.write(frame)
        vis.on_raw_frame = record

    print(f"opening {cfg.camera.device!r} {cfg.camera.width}x{cfg.camera.height}, running {a.seconds:.0f} s"
          + (f", recording to {a.record}" if a.record else "") + " ... show your hand", flush=True)
    vis.start()
    t0, seq, n, n_hand, hands, lat, inf_ms, snap = time.monotonic(), 0, 0, 0, {}, [], [], None
    try:
        while time.monotonic() - t0 < a.seconds:
            seq, st = vis.state_box.wait_newer(seq, 0.5)
            if st is None:
                continue
            n += 1
            inf_ms.append(vis.inf_ms)
            lat.append((st.t_done - st.t_capture) * 1000)
            if st.tracked:
                n_hand += 1
                hands[st.handedness] = hands.get(st.handedness, 0) + 1
                if a.snapshot:
                    snap = vis.frame_box.get()[1]
            if n % 15 == 0:
                print(f"  t={time.monotonic() - t0:4.1f}s cam {vis.cam_fps.v:4.1f} fps infer {vis.inf_fps.v:4.1f} fps "
                      f"hand {'YES ' + st.handedness if st.tracked else 'no'}" + (f" err={vis.error}" if vis.error else ""), flush=True)
    finally:
        cam_fps, inf_fps, frames, err = vis.cam_fps.v, vis.inf_fps.v, vis.frames, vis.error
        ok = vis.stop()
        if writer is not None:
            writer.release()
    if snap is not None:
        cv2.imwrite(a.snapshot, snap)
    pct = lambda x, q: round(float(np.percentile(x, q)), 1) if x else None
    res = dict(device=str(cfg.camera.device), devices=devs, seconds=a.seconds, frames_captured=frames, states=n,
               cam_fps=round(cam_fps, 1), infer_fps=round(inf_fps, 1), infer_ms_median=pct(inf_ms, 50),
               capture_to_state_ms_median=pct(lat, 50), capture_to_state_ms_p95=pct(lat, 95),
               hand_detect_rate=round(n_hand / n, 3) if n else 0.0, handedness=hands, error=err,
               recorded=a.record if writer is not None else None, record_fps=rec_fps, threads_stopped=ok)
    out = Path(a.record).with_suffix(".json") if a.record else ROOT / "logs" / "camera_check.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "devices"}, indent=2))
    print(f"summary written to {out}")
    if err or frames == 0:
        print("FAIL: no frames" if frames == 0 else f"WARNING: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
