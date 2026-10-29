"""Vision pipeline, fully separate from the simulation: capture thread -> inference thread -> latest-value boxes.

  CameraThread    reads frames as fast as the camera delivers and keeps only the newest one.
  TrackerThread   runs MediaPipe HandLandmarker on the newest frame (older frames are dropped, never queued),
                  publishes a HandState and an annotated preview frame.
  SyntheticSource a scripted fake hand (NOT a camera test) that feeds the same boxes through the same feature code.
"""
import fcntl
import os
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .buffer import LatestBox
from .features import CONNECTIONS, HandState, extract, synthetic_hand

PKG = Path(__file__).resolve().parent


class Fps:
    def __init__(self, a=0.1):
        self.a, self.v, self._t = a, 0.0, None

    def tick(self):
        t = time.monotonic()
        if self._t is not None and t > self._t:
            self.v = (1 - self.a) * self.v + self.a / (t - self._t) if self.v else 1 / (t - self._t)
        self._t = t


VIDIOC_G_CTRL, VIDIOC_S_CTRL = 0xC008561B, 0xC008561C
V4L2_CID_EXPOSURE_AUTO_PRIORITY = 0x009A0903   # 1: auto-exposure may lower the frame rate (UVC default)


def v4l2_ctrl(device, cid, value=None):
    """Read (value=None) or write a V4L2 control on /dev/video<device>. Returns the value, or None if unsupported.
    OpenCV exposes no setter for exposure_auto_priority, so this goes through the raw ioctl."""
    try:
        fd = os.open(f"/dev/video{int(device)}", os.O_RDWR)
    except (OSError, ValueError):
        return None
    try:
        if value is not None:
            fcntl.ioctl(fd, VIDIOC_S_CTRL, struct.pack("Ii", cid, int(value)))
        buf = bytearray(struct.pack("Ii", cid, 0))
        fcntl.ioctl(fd, VIDIOC_G_CTRL, buf)
        return struct.unpack("Ii", buf)[1]
    except OSError:
        return None
    finally:
        os.close(fd)


def user_hand(label, mirrored):
    """MediaPipe Tasks label -> the user's actual hand. Measured on this webcam with mediapipe 1.2.0: the user's right
    hand in a mirrored (selfie) frame is labelled 'Left' (184/200 frames), so the label is swapped for mirrored input."""
    if not mirrored:
        return label
    return {"Left": "Right", "Right": "Left"}.get(label, label)


class HandAssigner:
    """Map detections [(user_label, mean image x), ...] to robot sides -> {side: detection index}, frame by frame.
    Two hands: by screen position (mirrored view: your right hand is on the screen right), stable even when
    MediaPipe's labels flicker (measured 8% of frames). One hand: a hand continuing from the previous frame (within
    reuse_dist) keeps its side through label flickers; a newly appearing hand takes the side of the screen half it
    is in (its first labels were wrong for 8 frames on a real clip). Either way, switch_after consecutive frames
    whose label disagrees move it to the labelled side (e.g. a hand that crossed the midline)."""

    def __init__(self, mirrored, switch_after=5, reuse_dist=0.12):
        self.mirrored, self.switch_after, self.reuse_dist = mirrored, switch_after, reuse_dist
        self.prev_x, self.disagree = {}, 0

    def __call__(self, dets):
        if not dets:
            self.prev_x, self.disagree = {}, 0
            return {}
        if len(dets) >= 2:
            order = sorted(range(len(dets)), key=lambda i: dets[i][1])[:: (1 if self.mirrored else -1)]
            pick = {"left": order[0], "right": order[-1]}
            self.disagree = 0
        else:
            lab, x = dets[0]
            labelled = lab.lower() if lab.lower() in ("right", "left") else "right"
            near = [s for s, px in self.prev_x.items() if abs(px - x) < self.reuse_dist]
            by_half = "right" if (x > 0.5) == self.mirrored else "left"     # appearing hand: labels were unreliable
            side = near[0] if len(near) == 1 else by_half
            self.disagree = self.disagree + 1 if side != labelled else 0
            if self.disagree >= self.switch_after:
                side, self.disagree = labelled, 0
            pick = {side: 0}
        self.prev_x = {s: dets[i][1] for s, i in pick.items()}
        return pick


SIDE_COLOR = {"right": (0, 255, 0), "left": (255, 200, 0)}


def draw_hand(img, lm_px, color=(0, 255, 0), tag=None):
    h, w = img.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y in lm_px]
    for a, b in CONNECTIONS:
        cv2.line(img, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(img, p, 3, (0, 0, 255), -1)
    if tag:
        cv2.putText(img, tag, (pts[0][0] - 8, pts[0][1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


class VisionSystem:
    """Owns the threads and boxes. `state_box` holds {side: HandState} per processed frame (every active robot side
    present, untracked if its hand was not found); `frame_box` holds the annotated preview (BGR)."""

    def __init__(self, cfg):
        from .config import active_sides
        self.cfg = cfg
        self.sides = active_sides(cfg)
        self.state_box, self.frame_box = LatestBox(), LatestBox()
        self._frame_box = LatestBox()           # raw newest frame (capture -> inference)
        self._stop = threading.Event()
        self._threads = []
        self.cap = None
        self._prio_restore = None               # original exposure_auto_priority, restored on release
        self.cam_fps, self.inf_fps = Fps(), Fps()
        self.inf_ms = 0.0                       # last MediaPipe inference time
        self.on_raw_frame = None                # optional callback(frame) in the capture thread, before mirroring
        self.error = ""                         # last camera/model error, shown in the status panel
        self.frames = 0

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        c = self.cfg.camera
        if c.source == "synthetic":
            self._threads = [threading.Thread(target=self._synthetic_loop, name="synthetic", daemon=True)]
        else:
            self._threads = [threading.Thread(target=self._capture_loop, name="capture", daemon=True),
                             threading.Thread(target=self._inference_loop, name="inference", daemon=True)]
        for t in self._threads:
            t.start()

    def stop(self, timeout=3.0):
        self._stop.set()
        for t in self._threads:
            t.join(timeout)
        self._release()
        return not any(t.is_alive() for t in self._threads)

    def _release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self._prio_restore is not None:
            v4l2_ctrl(self.cfg.camera.device, V4L2_CID_EXPOSURE_AUTO_PRIORITY, self._prio_restore)
            self._prio_restore = None

    # ------------------------------------------------------------------ camera
    def _open(self):
        c = self.cfg.camera
        dev = int(c.device) if str(c.device).isdigit() else str(c.device)
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2) if isinstance(dev, int) else cv2.VideoCapture(dev)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"cannot open camera/video {c.device!r}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.height)
        cap.set(cv2.CAP_PROP_FPS, c.fps); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if isinstance(dev, int):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            # In dim rooms UVC auto-exposure stretches frames (measured 5 fps vs 15 fps on the LOQ camera).
            if not c.auto_exposure_priority:
                old = v4l2_ctrl(dev, V4L2_CID_EXPOSURE_AUTO_PRIORITY)
                if old not in (None, 0) and v4l2_ctrl(dev, V4L2_CID_EXPOSURE_AUTO_PRIORITY, 0) == 0:
                    self._prio_restore = old
        return cap

    def _capture_loop(self):
        fails, is_file = 0, not str(self.cfg.camera.device).isdigit()
        while not self._stop.is_set():
            if self.cap is None:
                try:
                    self.cap = self._open()
                    self.error = ""
                except Exception as e:
                    self.error = f"camera: {e}"
                    self._stop.wait(2.0)
                    continue
            ok, frame = self.cap.read()
            t = time.monotonic()
            if not ok or frame is None:
                fails += 1
                if is_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)       # loop video files
                if fails > 30:
                    self.error = "camera: read failures, reopening"
                    self._release(); fails = 0
                self._stop.wait(0.01)
                continue
            fails = 0
            self.frames += 1
            self.cam_fps.tick()
            if self.on_raw_frame is not None:
                self.on_raw_frame(frame)                            # e.g. recorder: un-mirrored, so replays mirror the same way
            if self.cfg.camera.mirror:
                frame = cv2.flip(frame, 1)
            self._frame_box.put((t, frame))
            if is_file:                                             # pace files like a camera, at their own rate
                fps = self.cap.get(cv2.CAP_PROP_FPS) or self.cfg.camera.fps
                self._stop.wait(1.0 / max(fps, 1))

    def _make_landmarker(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        from mediapipe.tasks.python.vision import RunningMode
        tr = self.cfg.tracking
        path = PKG / tr.model_path
        if not path.exists():
            raise FileNotFoundError(f"MediaPipe model not found: {path} (see README: download hand_landmarker.task)")
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)), running_mode=RunningMode.VIDEO,
            num_hands=2 if len(self.sides) > 1 or tr.preferred_hand != "any" else 1,
            min_hand_detection_confidence=tr.min_detection_confidence,
            min_hand_presence_confidence=tr.min_presence_confidence, min_tracking_confidence=tr.min_tracking_confidence)
        return mp, vision.HandLandmarker.create_from_options(opts)

    def _inference_loop(self):
        try:
            mp, lmk = self._make_landmarker()
        except Exception as e:
            self.error = f"mediapipe: {e}"
            return
        seq, last_ts, pref = 0, -1, self.cfg.tracking.preferred_hand
        assign = HandAssigner(self.cfg.camera.mirror)
        try:
            while not self._stop.is_set():
                seq, item = self._frame_box.wait_newer(seq, 0.1)
                if item is None:
                    continue
                t_cap, frame = item
                h, w = frame.shape[:2]
                ts = int(t_cap * 1000)
                ts = max(ts, last_ts + 1); last_ts = ts
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                t0 = time.perf_counter()
                try:
                    res = lmk.detect_for_video(img, ts)
                except Exception as e:
                    self.error = f"mediapipe: {e}"
                    continue
                self.inf_ms = (time.perf_counter() - t0) * 1000
                labels = [user_hand(hd[0].category_name, self.cfg.camera.mirror) for hd in res.handedness]
                if len(self.sides) > 1:
                    xs = [float(np.mean([p.x for p in lm])) for lm in res.hand_landmarks]
                    pick = assign(list(zip(labels, xs)))
                else:
                    k = next((i for i, lab in enumerate(labels) if pref == "any" or lab.lower() == pref.lower()), None)
                    pick = {} if k is None else {self.sides[0]: k}
                packet = {}
                for s in self.sides:
                    st = HandState(tracked=False)
                    if s in pick:
                        k = pick[s]
                        lm = np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[k]])
                        wl = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[k]])
                        st = extract(lm, wl, labels[k], res.handedness[k][0].score, w / h)
                    st.t_capture, st.t_done = t_cap, time.monotonic()
                    packet[s] = st
                self.inf_fps.tick()
                self.state_box.put(packet)
                for s, st in packet.items():
                    if st.tracked:
                        draw_hand(frame, st.landmarks_px, SIDE_COLOR[s], s[0].upper() if len(self.sides) > 1 else None)
                self.frame_box.put(frame)
        finally:
            lmk.close()

    # ------------------------------------------------------------------ synthetic
    def _synthetic_loop(self):
        c, aspect = self.cfg.camera, self.cfg.camera.width / self.cfg.camera.height
        t0 = time.monotonic()
        period = 1.0 / max(c.fps, 1)
        while not self._stop.is_set():
            t = time.monotonic()
            frame = np.zeros((c.height, c.width, 3), np.uint8)
            packet, two = {}, len(self.sides) > 1
            for k, side in enumerate(self.sides):
                s = t - t0 + 2.5 * k                    # each hand on its own phase
                curl = np.clip(0.5 - 0.5 * np.cos(2 * np.pi * s / 6.0), 0, 1)
                pinch = float(np.clip(np.sin(2 * np.pi * s / 9.0), 0, 1))
                x0 = 0.5 + (0.2 if side == "right" else -0.2) * two     # mirrored view: right hand on screen right
                center = (x0 + 0.06 * np.sin(2 * np.pi * s / 8.0), 0.5 + 0.10 * np.sin(2 * np.pi * s / 5.0))
                scale = 1.0 + 0.15 * np.sin(2 * np.pi * s / 10.0)
                w, im = synthetic_hand(curl=(curl,) * 4, thumb=curl, pinch=0.0 if curl > 0.2 else pinch, center=center,
                                       scale=scale, aspect=aspect, left=side == "left")
                st = extract(im, w, side.capitalize(), 1.0, aspect)
                st.t_capture = time.monotonic(); st.t_done = st.t_capture; st.synthetic = True
                packet[side] = st
                draw_hand(frame, im[:, :2], SIDE_COLOR[side], side[0].upper() if two else None)
            self.cam_fps.tick(); self.inf_fps.tick(); self.frames += 1
            self.state_box.put(packet)
            cv2.putText(frame, "SYNTHETIC SOURCE (not a camera)", (10, c.height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
            self.frame_box.put(frame)
            self._stop.wait(max(0.0, period - (time.monotonic() - t)))
