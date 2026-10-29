# LiveHand-G1: webcam hands -> simulated Unitree G1 in MuJoCo

Simulation only. Webcam -> OpenCV -> MediaPipe HandLandmarker (21 landmarks per hand) -> hand-state estimation ->
calibration + retargeting -> bounded IK per arm and a gesture mapper per hand -> one of the project's existing scenes:

| `--robot` | scene | hands | arms driven |
|---|---|---|---|
| `inspire` (default) | `robots/inspire_scene.py`: G1 29-DoF from Unitree's URDF + **5-finger Inspire RH56DFQ** hands | 6 actuators/hand (thumb yaw + pitch, index, middle, ring, pinky; 6 more joints coupled) | **right and left** |
| `dex3` | `robots/grasp_scene.py`: Menagerie G1 + 3-finger Dex3-1 hands | 7 actuators/hand | right |

Your right hand drives the robot's right arm and hand; your left hand drives the left.

## Status (read first)
Native Ubuntu 26.04 (GNOME/Wayland), Lenovo LOQ 15IRH8 integrated camera, Python 3.14.4, 2026-10-29.

**Verified on this machine**
- 48 unit/integration tests pass; the real-time synthetic end-to-end check passes for both robots (Inspire: both arms
  track within 8 mm, both 5-finger hands sweep their full range, robot upright, real time).
- **5-finger G1 with a recording of your (right) hand:** palm follows the command with 1.8 mm median / 3.2 mm p95
  error, correlation 0.996 / 0.992 / 0.994; all six hand actuators move with your fingers.
- Webcam `/dev/video0` (in-kernel `uvcvideo`, no extra driver needed): 640x480 MJPG at ~15 fps (hardware cap) once
  `auto_exposure_priority = false` is applied (5 fps without it in a dim room).
- MediaPipe on your real hand: 8.8 ms/frame on CPU, capture->hand-state 10 ms median, hand tracked continuously
  while in view (`livehand.tools.camera_check`).
- **Full pipeline on a recording of your hand** (`livehand.tools.replay_check`): real MediaPipe -> neutral
  calibration -> enable -> IK -> MuJoCo physics. The simulated palm followed the command with correlation
  0.996 / 0.989 / 0.994 (x/y/z), 4.3 mm median error; commands always inside workspace and actuator ranges;
  robot upright; Dex3-1 commands moved with real thumb/finger motion.
- The full GUI app with the live webcam + MuJoCo viewer + preview runs in real time and shuts down cleanly.

**Not yet verified (needs you at the keyboard):** interactive key handling in the preview window, the o/f/p gesture
captures on your hand, judging by eye that the motion directions feel right (see "First live session"), and **two
real hands at once** (two-hand control is tested with synthetic hands; the clip had only your right hand).

## Install
The existing `.venv-livehand` (Python 3.14.4, created with `python3 -m venv`) already has everything; nothing needed
installing. MediaPipe 1.2.0, MuJoCo 3.17.0, opencv-contrib-python 5.0.0.93 and NumPy 2.5.4 all support Python 3.14.
To recreate it on a fresh machine (exact pins in `requirements-livehand.txt`):
```bash
cd livehand-g1            # repo root
python3 -m venv .venv-livehand
.venv-livehand/bin/pip install -r requirements-livehand.txt
mkdir -p livehand/models && curl -fsSL -o livehand/models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```
Do not install `opencv-python` next to `opencv-contrib-python` (they overwrite each other).

## Run
```bash
cd livehand-g1            # repo root
.venv-livehand/bin/python -m livehand.tools.camera_check     # diagnose: devices, fps, MediaPipe detection (no windows)
.venv-livehand/bin/python -m livehand --vision-only          # camera-only mode: mirrored preview + 21-landmark skeleton
.venv-livehand/bin/python -m livehand                        # full teleoperation: 5-finger G1, both hands
.venv-livehand/bin/python -m livehand --hands right          # 5-finger G1, right arm/hand only (left held)
.venv-livehand/bin/python -m livehand --robot dex3           # 3-finger Dex3-1 G1 (right arm)
.venv-livehand/bin/python -m livehand --camera 1 --width 1280 --height 720
.venv-livehand/bin/python -m livehand --source synthetic     # scripted fake hand: shows the control path, NOT your hand
```
Record a replayable clip and run it through the whole pipeline (no webcam needed for the replay):
```bash
.venv-livehand/bin/python -m livehand.tools.camera_check --seconds 20 --record logs/hand_clip/hand_clip.mp4
.venv-livehand/bin/python -m livehand --camera logs/hand_clip/hand_clip.mp4     # interactive, with windows
.venv-livehand/bin/python -m livehand.tools.replay_check logs/hand_clip/hand_clip.mp4   # headless, writes json + png
```
`logs/hand_clip/` is git-ignored (it contains video of you).

## Keyboard controls
Mirroring starts **DISABLED**. Press keys with the *camera preview window* focused:

| key | action |
|---|---|
| `n` | capture **neutral** (hand relaxed, palm toward camera, in the middle of its half of the image; averages 20 frames) |
| `m` | capture **range** (optional, after `n`): sweep your hand gently left/right, up/down, toward/away for ~4 s; scales motion so your sweep spans the robot workspace (bounded to 0.33x-3x of the config gains) |
| `o` / `f` / `p` | capture **open hand**, **fist**, **thumb-index pinch** (each averages 20 frames) |
| `c` | calibrate the **other hand** next (cancels a capture in progress) |
| `e` | enable / disable mirroring (disable holds the current pose). Refused until neutral is captured. |
| `SPACE` | **E-STOP**: freeze all commands immediately. First `e` only clears it; a second `e` enables. |
| `h` | return home (only while disabled / e-stopped) |
| `r` | toggle palm-orientation control (default OFF) |
| `g` | toggle hand-gesture control |
| `s` / `l` | save / load `livehand/calibration.json` |
| `q` / ESC | quit (releases the webcam, closes viewer and windows) |

The MuJoCo window also accepts `SPACE` (e-stop), `E` (enable), `Q` (quit). The preview panel shows the next calibration
step, mode, tracking state, camera/inference fps, inference ms and measured capture->command latency.

## First live session
1. `python -m livehand --vision-only`: you should see yourself mirrored, your right hand with a green "R" skeleton on the
   screen right and your left hand with a blue "L" skeleton on the screen left ("right YES  left YES").
   Light your hands from the front; a bright window behind you makes the camera underexpose them.
2. `python -m livehand`. The 5-finger G1 stands with both forearms forward, palms facing each other, hands open; DISABLED.
3. Calibrate **one hand at a time, right first**, so your other hand is free for the keyboard (the panel shows
   "calibrating: RIGHT hand" and the next step):
   - Hold your **right** hand up in the right half of the image, type with your left: `n`, optionally `m` (sweep),
     then `o`, `f`, `p` (make each gesture with the right hand).
   - Calibration then switches to the **left** hand by itself: hold it up in the left half, type with your right:
     `n`, `o`, `f`, `p`.
   - `c` switches hands by hand (e.g. to redo one). `s` saves both. A hand without neutral stays still. Press `e`.
4. Move your palms slowly. Each robot hand moves the same way inside its own box (default +-12/15/12 cm). Curl single
   fingers: each Inspire finger follows its own finger. Pinch thumb and index.
5. Drop one hand out of view: that arm shows LOST and freezes (after 3 s it drifts home), the other keeps following.
   Bring both hands toward each other: the robot hands stop 16 cm apart. `SPACE` must freeze everything at once.
6. If a direction is inverted, flip the sign in `[position].axis_map` in `config.toml`.

## Architecture
```
CameraThread --latest frame--> TrackerThread (MediaPipe, up to 2 hands, HandAssigner)
                                    |--latest {right: HandState, left: HandState}--> Controller.update()  [main thread]
                                    \--annotated frame--> preview                      for each Side (right, left):
                                                                                         Calibrator / filters (One-Euro)
                                                                                         map_position / map_orientation
                                                                                         HandRetargeter -> hand targets
                                                                                         ArmIK (7 joints)
                                                                                       palm-separation guard (two arms)
MuJoCo physics (fixed dt, real time) <- Limiter(clamp+slew, all actuators) <-------------+
```
| responsibility | file |
|---|---|
| camera capture + MediaPipe threads, V4L2 exposure control, synthetic source | `tracking.py` |
| landmarks -> hand state (palm position, palm frame, curls, pinch) | `features.py` |
| neutral / range / open / fist / pinch calibration | `calibration.py` |
| human -> robot mapping (position, orientation, hand targets from a per-model profile) | `retarget.py` |
| 7-DoF arm IK (one instance per arm) | `arm_ik.py` |
| clamping, slew limits, watchdog, enable/e-stop state machine | `safety.py` |
| robot registry (`MODELS`), Inspire home/palm sites, Side + Controller (one tick = newest packet -> `d.ctrl`) | `sim.py` |
| entry point, viewer + preview loop, keys | `app.py`, `__main__.py`, `status.py` |
| latest-value box (no queues), One-Euro filter | `buffer.py`, `filters.py` |
| diagnostics: `camera_check`, `replay_check`, `synthetic_check`; pose derivation: `find_hand_poses` (Dex3-1), `find_inspire_poses` (Inspire, `--home`) | `tools/` |

Vision never blocks physics: the sim reads whatever HandState is newest; stale frames are dropped, not queued.

## Mapping (configurable in `config.toml`)
**Position.** Palm centre in the mirrored image, relative to the neutral capture, in frame-height units: `u` (screen right),
`v` (screen down); `s = ln(apparent hand size / neutral size)` for toward/away. MediaPipe's z is **not** used as metric depth;
`s` is a relative cue. `axis_map` (rows robot x forward, y left, z up) turns `[u, v, s]` into metres with `gain`/`depth_gain`
(or the range-capture gains), then clips to `workspace_half_extent` around the home hand position. Default: hand right ->
robot right, hand up -> robot up, hand toward camera -> robot forward.
**Handedness.** With mirrored frames MediaPipe 1.2.0 labels your right hand "Left" (measured 184/200 frames); `tracking.py`
swaps the label so `preferred_hand` and the panel name your real hand. The palm frame no longer depends on the label (it
flickers), so a mislabelled frame cannot flip the orientation target.
**Which hand drives which arm (two arms).** Two hands in view: by screen position (your right hand is on the screen right).
One hand: a hand that continues from the previous frame keeps its arm through label flickers; a newly appearing hand takes
the arm of the screen half it is in; 5 consecutive disagreeing labels move it (a hand that crossed the midline). On your
clip this put all 194 tracked frames on the right arm (label-first assignment got the first 8 wrong).
**Two-arm safety.** The palms are kept at least `min_hand_separation` (16 cm) apart sideways by pushing both outward; the
home palms are 40 cm apart, and each arm has its own workspace box, IK and watchdog.
**Orientation (optional, off by default, key `r`).** Palm frame from world landmarks, relative to the neutral palm, mapped
through a signed axis matrix, clamped to 35 deg total / 60 deg/s. Axis signs not yet confirmed by eye on the real camera.
**Arm.** Damped-least-squares IK per arm on its 7 joints (`<side>_shoulder_pitch/roll/yaw`, `<side>_elbow`,
`<side>_wrist_roll/pitch/yaw`), targets in the pelvis frame, per-step joint-velocity limit, joint-limit margin, adaptive
damping near singularities, null-space pull to home. Legs, waist, head (and any undriven arm/hand) hold their home commands.
Controlled point: Inspire `R_palm` / `L_palm` (added on the hand base, palm centre); Dex3-1 `grasp_site`.
**Inspire home.** Palms at (0.26, -/+0.20, 0.16) m in the pelvis frame, fingers forward, palms facing each other. Chosen by
search so every corner of the workspace box is reachable (<= 2.8 mm) without touching the torso; 4 cm farther forward the
far-low corners were past full arm extension (25-46 mm short). Left = right with roll/yaw joints negated.
**Hand mapping.** Cues normalised 0..1 by your open/fist/pinch calibration; each actuator moves from its open toward its fist
value by one cue, then thumb/index blend toward the pinch pose by the pinch cue (suppressed when the other fingers are curled).
- Inspire (6 actuators per hand): thumb curl -> `thumb_proximal_yaw` + `thumb_proximal_pitch`; index, middle, ring, pinky
  curl -> each finger's own `*_proximal` (its `*_intermediate` follows by the URDF coupling); pinch -> thumb yaw/pitch + index.
- Dex3-1 (7 actuators): thumb curl -> `thumb_0/1/2`; index curl -> `index_0/1`; mean(middle, ring, pinky) -> `middle_0/1`.
Gesture poses come from the models: `tools/find_inspire_poses.py` (Inspire: pinch = thumb fully opposed, 10.2 / 12.9 mm
fingertip gap R / L; fist = fingers at 90%, thumb clear; no new self-penetration on either hand) and `tools/find_hand_poses.py`
(Dex3-1). All targets stay within 90% of each actuator range; Inspire thumb pitch is usable only to 0.5 rad (its coupled
joints hit their limits there).
**Inspire model fix.** The URDF's thumb-base meshes overlap the palm by 3-5 mm at rest; that contact fought the thumb
(pitch target 0.45 reached 0.38). LiveHand excludes just that pair (thumb base/proximal vs. hand base), after which the thumb
tracks exactly; all other hand contacts stay on. `robots/inspire_scene.py` gained an optional `edit` hook for this; its
default output is unchanged.

## Safety behaviour
Startup DISABLED (home pose held, input ignored). Actuator commands are clamped to `ctrlrange` and slew-limited; NaN/inf commands
are rejected; raw target jumps > `max_target_jump` are rejected; Cartesian target speed is capped (`max_speed` 0.25 m/s), so
enabling with your hand away from neutral moves the robot there smoothly instead of jumping. Stale data (`stale_timeout`
0.25 s) or a lost hand freezes the pose, and after `lost_return_after` (3 s) it drifts back home with an open hand. Tracking
resumes only after `reacquire_frames` consecutive good frames. A non-finite sim state triggers E-STOP. While DISABLED/E-STOP the
IK does not run, so commands are exactly the held values.

## Tests
```bash
.venv-livehand/bin/python -m unittest livehand.tests.test_core -v      # 48 tests, ~6 s, no camera/display
.venv-livehand/bin/python -m livehand.tools.synthetic_check            # real-time e2e, both arms (add --viewer to watch)
.venv-livehand/bin/python -m livehand.tools.synthetic_check --robot dex3
```
Covers: coordinate mapping/signs/bounds/deadzone, range calibration and gain bounds, orientation clamp, calibration
(averaging, normalisation, untracked frames, old calibration files), handedness handling, actuator clamping + slew + NaN,
watchdog (jumps, staleness, reacquisition), model wiring, IK accuracy and limits, disabled/enable/disable/e-stop semantics,
stale-tracking behaviour, extreme-input bounds, gestures, status prompts, vision lifecycle (synthetic source, missing
camera, clean shutdown). Inspire: wiring of both arms/hands, home pose (position, joint margins, no contacts, stable),
thumb tracking, gesture poses in range and penetration-free on both hands, full workspace reachable without body
collisions. Two hands: each hand drives its own arm (and the simulated palms get there), independent fingers, one hand
missing, palm separation under converging input, per-hand calibration and enabling, calibration file round-trip,
hand-to-arm assignment.

## Native Ubuntu notes / troubleshooting
**Camera drivers.** None needed: the LOQ camera is a standard UVC device handled by the in-kernel `uvcvideo`. `/dev/video0` is
the camera, `/dev/video1` its metadata node (cannot capture; expected). Access is granted to the logged-in user by an ACL; if
`camera_check` says `rw=NO`, run `sudo usermod -aG video $USER` and log in again.
**Low fps.** In dim light, UVC auto-exposure lowers the frame rate (measured 5 fps). LiveHand sets `exposure_auto_priority=0`
through V4L2 while it runs (auto-exposure stays on; the frame rate is kept) and restores the driver value on exit. Set
`[camera].auto_exposure_priority = true` to disable this. The camera tops out at ~15 fps at 640x480; more front light gives a
cleaner, less blurred image.
**Windows.** On GNOME/Wayland, `app.py` puts the MuJoCo window on X11 (`PYGLFW_LIBRARY_VARIANT=x11`), because the Wayland
GLFW build gets no title bar here. It also points OpenCV's bundled Qt at the system DejaVu fonts. Export either variable
yourself to override. The Qt line "Ignoring XDG_SESSION_TYPE=wayland on Gnome" is harmless.
**Exit crash workaround.** MuJoCo 3.17's passive viewer crashes during Python 3.14 interpreter teardown here (a bare
`launch_passive` script segfaults 3/3). `python -m livehand` does all cleanup and then exits with `os._exit` to skip that
teardown. `scripts/run_g1.py` is unchanged and may still crash or hang on exit.
**Wrong or inverted motion.** `[position].axis_map` (rows robot x/y/z, columns u/v/s) flips or swaps axes; `gain`/`depth_gain`
scale (or use the `m` range capture). `[orientation].axis_map` likewise.
**Robot ignores you.** Panel must show MODE ENABLED and TRACKING; enabling is refused until `n` is captured.
**Fingers half-curled with an open hand / never close.** Run `o` and `f` (and `p`); the priors are typical-hand values.
On the recorded clip, the uncalibrated open hand read ~0.2-0.4 curl; after an open capture it read 0.02.
**Known quirk.** Menagerie's `stand` keyframe sets `left_hand_thumb_1_joint` = 1.05 rad, 0.003 rad above its limit; the controller
clips it. A 2-3 mm thumb-base/palm overlap in the Dex3-1 collision geometry makes the thumb settle slightly off zero.

## Known limitations
- Latency printed by the app is capture-timestamp -> command (measured 14-19 ms median live). It excludes exposure/USB
  (~67 ms frame period at 15 fps), viewer and display delay, so true glass-to-glass latency is higher.
- Monocular depth is a relative cue only; forward/back control is coarse. Orientation control is experimental.
- Steady-state ~5 mm palm sag below the command (gravity vs. position actuators).
- Dex3 drives the right arm only (its left home pose and gesture poses were never validated); its 3 digits average
  ring/pinky into the middle finger. The Inspire hand has 6 actuators for 12 joints: each finger bends as one unit.
- Two-hand control is verified with synthetic hands only; real two-hand tracking (occlusion, hands crossing) is untested.
  Hands crossing the midline can briefly swap arms until 5 frames of consistent labels move them back.
- Arm-arm clashes are prevented only by the lateral palm separation, not by full collision checking.
- The scene has no table or objects; arm/body collisions are avoided by the workspace box (verified collision-free
  at every corner for the Inspire arms).
