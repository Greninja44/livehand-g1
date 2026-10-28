# LiveHand-G1: webcam hand -> simulated Unitree G1 (Dex3-1) in MuJoCo

Simulation only. Webcam -> OpenCV -> MediaPipe HandLandmarker (21 landmarks) -> hand-state estimation -> calibration +
retargeting -> bounded IK for the right arm/wrist and a Dex3-1 gesture mapper -> the existing project scene
(`scripts/grasp_scene.py`: Menagerie G1 + Dex3-1 hands, validated arm-forward start pose, open hand).

## Status (read first)
- **Tested here (no webcam in this WSL session):** all unit/integration tests, the vision pipeline on a video file (capture +
  MediaPipe threads, clean shutdown), missing-camera error handling, and a real-time end-to-end run with a *synthetic* hand source.
- **NOT tested here:** live capture from a physical webcam, MediaPipe on a real hand, and the on-screen preview/key handling.
  Do the manual steps in "First live test" below; until then treat live mirroring as unverified.

## Install
```bash
cd ~/robotics/chess2bot
uv venv --python 3.12 .venv-livehand            # separate env: your existing .venv is untouched
uv pip install --python .venv-livehand/bin/python -r requirements-livehand.txt
# hand model (official MediaPipe, ~7.8 MB, sha256 fbc2a300...cde1); already present if you used this checkout:
mkdir -p livehand/models && curl -fsSL -o livehand/models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## Run
```bash
export GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA      # GPU GL on WSLg (optional, see main README)
.venv-livehand/bin/python -m livehand                      # webcam 0, MuJoCo viewer + camera preview window
.venv-livehand/bin/python -m livehand --camera 1 --width 1280 --height 720
.venv-livehand/bin/python -m livehand --camera path/or/url # video file / stream instead of a webcam
.venv-livehand/bin/python -m livehand --source synthetic   # scripted fake hand: shows the control path, NOT your hand
.venv-livehand/bin/python -m livehand --vision-only        # camera + MediaPipe + preview only, no MuJoCo
```
Mirroring starts **DISABLED**. Keys (press them with the *camera preview window* focused):

| key | action |
|---|---|
| `n` | capture **neutral** pose (position + palm-orientation reference; hold your hand centred, relaxed) |
| `o` / `f` / `p` | capture **open hand**, **fist**, **thumb-index pinch** (each averages 20 valid frames) |
| `e` | enable / disable mirroring (disable holds the current pose). Refused until neutral is calibrated. |
| `SPACE` | **E-STOP**: freeze all commands immediately. First `e` only clears it; a second `e` enables. |
| `h` | return home (only while disabled / e-stopped) |
| `r` | toggle palm-orientation control (default OFF) |
| `g` | toggle hand-gesture control |
| `s` / `l` | save / load `livehand/calibration.json` |
| `q` / ESC | quit (releases the webcam, closes viewer and windows) |

Typical session: `n` (hold still) -> `o` (open hand) -> `f` (fist) -> `p` (pinch) -> `s` -> `e`.
The MuJoCo window also accepts `SPACE` (e-stop), `E`, `Q`.

## Architecture
```
CameraThread --latest frame--> TrackerThread (MediaPipe) --latest HandState--> Controller.update()  [main thread]
                                    \--annotated frame--> preview                  |-> Calibrator / filters (One-Euro)
                                                                                    |-> map_position / map_orientation
                                                                                    |-> HandRetargeter -> 7 Dex3-1 targets
MuJoCo physics (fixed dt, real time) <- Limiter(clamp+slew) <- ArmIK (7 joints) + hand targets
```
Files: `tracking.py` (threads, synthetic source), `features.py` (landmarks -> HandState), `calibration.py`, `retarget.py`,
`arm_ik.py`, `safety.py` (Limiter, Watchdog, Mode), `sim.py` (model + Controller), `status.py`, `app.py`, `config.toml`,
`buffer.py` (latest-value box: no queues), `tools/` (pose derivation, synthetic check), `tests/`.
Vision never blocks physics: the sim reads whatever HandState is newest; stale frames are dropped, not queued.

## Mapping (documented, configurable in `config.toml`)
**Position.** Palm centre in the mirrored image, relative to the neutral capture, in units of frame height: `u` (screen right),
`v` (screen down); `s = ln(apparent hand size / neutral size)` for toward/away. MediaPipe's z is **not** used as metric depth;
`s` is a relative cue. `axis_map` (rows = robot x forward, y left, z up) turns `[u, v, s]` into metres with `gain`/`depth_gain`,
then clips to `workspace_half_extent` (default +-12/15/12 cm around the home hand position). Default: hand right -> robot right,
hand up -> robot up, hand toward camera -> robot forward.
**Orientation (optional, off by default).** Palm frame from world landmarks, relative to the neutral palm, mapped through a
signed axis matrix (as a pseudovector), clamped to 35 deg total / 60 deg/s. Axis signs are unverified on a real camera.
**Arm.** Damped-least-squares IK on the 7 right-arm joints of the real model (site `grasp_site` on the palm), targets in the
pelvis frame, per-step joint-velocity limit, joint-limit margin, adaptive damping near singularities, null-space pull to home.
Legs, waist, head, left arm and left hand are held at their keyframe commands.
**Hand (Dex3-1, 7 position actuators: thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1).** Cues normalised 0..1 by
your open/fist/pinch calibration:
- index curl -> `index_0/1` (open -> fist pose)
- mean(middle, ring, pinky curl) -> `middle_0/1` (Dex3-1 has no ring/pinky)
- thumb curl -> `thumb_0/1/2`
- thumb-index pinch (suppressed when the other fingers are curled) -> blends thumb and index toward the pinch pose.
Gesture poses come from the model (`tools/find_hand_poses.py`): open = validated start pose, fist = no self-penetration, pinch =
12 mm fingertip gap with no new self-penetration. All targets stay within 90% of each actuator range.

## Safety behaviour
Startup DISABLED (home pose held, input ignored). Actuator commands are clamped to `ctrlrange` and slew-limited; NaN/inf commands
are rejected; raw target jumps > `max_target_jump` are rejected; Cartesian target speed is capped; stale data
(`stale_timeout` 0.25 s) or a lost hand freezes the pose, and after `lost_return_after` (3 s) it drifts back home with an open
hand. Tracking resumes only after `reacquire_frames` consecutive good frames. A non-finite sim state triggers E-STOP.
While DISABLED/E-STOP the IK does not run, so commands are exactly the held values.

## Tests
```bash
.venv-livehand/bin/python -m unittest livehand.tests.test_core -v      # 28 tests, ~10 s, no camera/display
.venv-livehand/bin/python -m livehand.tools.synthetic_check            # real-time e2e with the synthetic source (add --viewer to watch)
```
Covers: coordinate mapping/signs/bounds/deadzone, orientation clamp, calibration (averaging, normalisation, untracked frames),
actuator clamping + slew + NaN, watchdog (jumps, staleness, reacquisition), model wiring (all joints/actuators exist, position
actuators, poses inside ranges, no arm/hand penetration at start), IK accuracy and limits, disabled/enable/disable/e-stop semantics,
stale-tracking behaviour, extreme-input bounds, gestures, vision lifecycle (synthetic source, missing camera, clean shutdown).

## First live test (manual, needs your webcam)
1. Get the camera into WSL (below), confirm `ls /dev/video*`.
2. `python -m livehand --vision-only` -> you should see yourself mirrored with a green skeleton, "hand YES", and fps numbers.
3. `python -m livehand`; press `n` with your hand centred and relaxed, then `o`, `f`, `p`, then `e`.
4. Move your palm slowly: the robot's right hand should move the same way in a small box; open/close your fingers.
5. Cover the camera / drop your hand: the robot must freeze (panel shows LOST), then drift home after 3 s. `SPACE` must freeze it at once.
6. If a direction is inverted, change the signs in `[position].axis_map` (see Troubleshooting).

## Troubleshooting
**No webcam in WSL2 (`ls /dev/video*` empty).** This machine's kernel has UVC as a module, and `usbipd-win` is installed. In an
*Administrator* PowerShell: `usbipd list` (your camera was `1-7  04f2:b7b9  Integrated Camera`), `usbipd bind --busid 1-7`,
`usbipd attach --wsl --busid 1-7`. Then in WSL: `ls /dev/video*` (if empty: `sudo modprobe uvcvideo`). While attached, Windows apps
(Teams, Camera) lose the camera; `usbipd detach --busid 1-7` gives it back. If `/dev/video0` is not accessible: `sudo usermod -aG video $USER`
and restart WSL. Alternatives: run the same code on native Windows Python, or pass an IP-camera/stream URL via `--camera`.
**Camera opens but is black/slow.** Try `--camera 1`, lower `--width/--height`, check `cam ... fps` in the panel. The MJPG fourcc is requested for 30 fps.
**`cv2.imshow` errors / no window.** Needs WSLg (`echo $DISPLAY $WAYLAND_DISPLAY`). Qt platform plugin errors: `export QT_QPA_PLATFORM=xcb`.
Use `--no-preview` to run without the OpenCV window (keys then only via the MuJoCo window: SPACE, E, Q).
**MediaPipe.** Version 1.2.0 provides only the Tasks API and needs `livehand/models/hand_landmarker.task` (see Install). Error text is shown
in the panel (`ERROR: mediapipe: ...`). Do not install `opencv-python` next to `opencv-contrib-python` (they overwrite each other); the
requirements pin only the contrib build that MediaPipe depends on.
**MuJoCo rendering.** See the main README (WSLg, `GALLIUM_DRIVER=d3d12`). The sim runs without a window via `--no-viewer`.
**Wrong or inverted motion.** Edit `config.toml`: `[position].axis_map` (rows robot x/y/z, columns u/v/s) flips or swaps axes;
`gain`/`depth_gain` scale. `[orientation].axis_map` likewise (orientation is off until you press `r`).
**Robot ignores you.** Panel must show MODE ENABLED and TRACKING; enabling is refused until `n` is captured; check `calib:` flags.
**Fingers saturate or never close.** Re-run `o` and `f` (and `p`); the priors in `calibration.py` are only typical-hand values.
**Known quirk.** Menagerie's `stand` keyframe sets `left_hand_thumb_1_joint` = 1.05 rad, 0.003 rad above its limit; the controller
clips it. A 2-3 mm thumb-base/palm overlap in the Dex3-1 collision geometry makes the thumb settle slightly off zero.

## Known limitations
- Live webcam path unverified here (see Status). Latency numbers printed by the app are capture-timestamp -> command only; they exclude
  camera exposure/USB, viewer and display delay, so they understate true glass-to-glass latency.
- Monocular depth is a relative cue only; forward/back control is coarse. Orientation control is experimental.
- Only the right arm and hand are controlled; the Dex3-1 has 3 digits, so ring/pinky are averaged into the middle finger.
- The sim does not model contact with a table or objects in this scene; collisions between the arm and the body are avoided only via the workspace box.
