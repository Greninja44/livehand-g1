"""Vision pipeline, fully separate from the simulation: capture thread -> inference thread -> latest-value boxes.

  CameraThread    reads frames as fast as the camera delivers and keeps only the newest one.
  TrackerThread   runs MediaPipe HandLandmarker on the newest frame (older frames are dropped, never queued),
                  publishes a HandState and an annotated preview frame.
  SyntheticSource a scripted fake hand (NOT a camera test) that feeds the same boxes through the same feature code.
"""
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


def draw_hand(img, lm_px, color=(0, 255, 0)):
    h, w = img.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y in lm_px]
    for a, b in CONNECTIONS:
        cv2.line(img, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(img, p, 3, (0, 0, 255), -1)


class VisionSystem:
    """Owns the threads and boxes. `state_box` holds HandState, `frame_box` the annotated preview (BGR)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.state_box, self.frame_box = LatestBox(), LatestBox()
        self._frame_box = LatestBox()           # raw newest frame (capture -> inference)
        self._stop = threading.Event()
        self._threads = []
        self.cap = None
        self.cam_fps, self.inf_fps = Fps(), Fps()
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
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        return not any(t.is_alive() for t in self._threads)

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
                    self.cap.release(); self.cap = None; fails = 0
                self._stop.wait(0.01)
                continue
            fails = 0
            self.frames += 1
            self.cam_fps.tick()
            if self.cfg.camera.mirror:
                frame = cv2.flip(frame, 1)
            self._frame_box.put((t, frame))
            if is_file:
                self._stop.wait(1.0 / max(self.cfg.camera.fps, 1))   # pace files like a camera

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
            num_hands=2 if tr.preferred_hand != "any" else 1,
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
                try:
                    res = lmk.detect_for_video(img, ts)
                except Exception as e:
                    self.error = f"mediapipe: {e}"
                    continue
                state, k = HandState(tracked=False), -1
                for i, hd in enumerate(res.handedness):
                    if pref == "any" or hd[0].category_name == pref:
                        k = i
                        break
                if k >= 0:
                    lm = np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[k]])
                    wl = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[k]])
                    state = extract(lm, wl, res.handedness[k][0].category_name, res.handedness[k][0].score, w / h)
                state.t_capture, state.t_done = t_cap, time.monotonic()
                self.inf_fps.tick()
                self.state_box.put(state)
                if state.tracked:
                    draw_hand(frame, state.landmarks_px)
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
            s = t - t0
            curl = np.clip(0.5 - 0.5 * np.cos(2 * np.pi * s / 6.0), 0, 1)
            pinch = float(np.clip(np.sin(2 * np.pi * s / 9.0), 0, 1))
            center = (0.5 + 0.06 * np.sin(2 * np.pi * s / 8.0), 0.5 + 0.10 * np.sin(2 * np.pi * s / 5.0))
            scale = 1.0 + 0.15 * np.sin(2 * np.pi * s / 10.0)
            w, im = synthetic_hand(curl=(curl,) * 4, thumb=curl, pinch=0.0 if curl > 0.2 else pinch, center=center, scale=scale, aspect=aspect)
            st = extract(im, w, "Right", 1.0, aspect)
            st.t_capture = time.monotonic(); st.t_done = st.t_capture; st.synthetic = True
            self.cam_fps.tick(); self.inf_fps.tick(); self.frames += 1
            self.state_box.put(st)
            frame = np.zeros((c.height, c.width, 3), np.uint8)
            draw_hand(frame, im[:, :2])
            cv2.putText(frame, "SYNTHETIC SOURCE (not a camera)", (10, c.height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
            self.frame_box.put(frame)
            self._stop.wait(max(0.0, period - (time.monotonic() - t)))
