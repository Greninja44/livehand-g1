# LiveHand-G1

**Real-time webcam teleoperation of a Unitree G1 humanoid in MuJoCo.**

Control a simulated Unitree G1 with your bare hands. A laptop webcam tracks both of your hands with MediaPipe; each
robot arm follows your palm, and the G1's 5-finger hands copy your fingers, including thumb-index pinches. No gloves,
no depth camera, no robot hardware: **simulation only**.

![Unitree G1 with 5-finger Inspire hands in MuJoCo](docs/g1_inspire_teleop.png)

## Highlights
- **Two arms, ten fingers.** Your right hand drives the robot's right arm and hand, your left drives the left. On the
  default 5-finger Inspire hands, index, middle, ring and pinky each follow your own finger; your thumb drives thumb swing
  and bend.
- **Real time on a laptop CPU.** Measured on the target machine: MediaPipe 8.8 ms per frame, ~15 ms capture-to-command,
  physics at a fixed 500 Hz in real time. Vision and simulation run in separate threads and exchange only the newest state,
  so stale frames are dropped, never queued.
- **Accurate following.** Replaying a real recording of a hand: the simulated palm stays within 1.8 mm (median) of its
  command, with correlation above 0.99 on every axis.
- **Safety first.** Starts disabled, explicit enable, emergency stop, bounded workspace, rate limits, joint-limit margins,
  jump rejection, stale/lost-tracking holds, and a guard that keeps the two hands apart.
- **Calibration you can do alone.** One hand at a time with on-screen prompts, so the other hand stays on the keyboard.

## Quick start
Tested on Ubuntu 26.04 (native, GNOME/Wayland), Python 3.14, Lenovo LOQ 15IRH8 integrated webcam. First-time setup is in
[Setup from scratch](#setup-from-scratch).
```bash
cd ~/chess2bot
.venv-livehand/bin/python -m livehand.tools.camera_check   # is the webcam OK? (fps, hand detection; no windows)
.venv-livehand/bin/python -m livehand --vision-only        # camera only: see both hands tracked
.venv-livehand/bin/python -m livehand                      # full teleoperation (5-finger G1, both hands)
```
Options: `--hands right` (one arm), `--robot dex3` (3-finger Dex3-1 G1, right arm), `--camera 1`,
`--camera clip.mp4` (replay a recording), `--source synthetic` (scripted fake hands, no camera).

### First session
Click the **camera window** so it gets the keys. The panel's green line always says what to do next.
1. **Right hand:** hold it up in the right half of the image, palm to the camera, and type with your left:
   `n` (neutral, hold still), then `o` (open hand), `f` (fist), `p` (thumb-index pinch).
2. **Left hand:** calibration switches over by itself. Hold your left hand in the left half and type with your right:
   `n`, `o`, `f`, `p`.
3. `s` saves the calibration (`l` loads it next time). Press `e` and move slowly.

| key | action | | key | action |
|---|---|---|---|---|
| `e` | enable / disable teleoperation | | `n` `m` `o` `f` `p` | calibrate: neutral, range (optional sweep), open, fist, pinch |
| `SPACE` | **emergency stop** (freeze everything) | | `c` | calibrate the other hand next |
| `h` | return home (while disabled) | | `r` / `g` | palm-orientation control / finger control on-off |
| `q`, `ESC` | quit | | `s` / `l` | save / load calibration |

Details, tuning and troubleshooting: **[livehand/README.md](livehand/README.md)**.

## How it works
```
 webcam ──> capture thread ──latest frame──> MediaPipe thread ──latest hand states──> control loop (100 Hz)
   (V4L2, 15 fps)        (mirror, drop stale)     (21 landmarks x 2 hands,               │
                                                   left/right assignment)                │ per arm:
                                                                                         │  filter (One-Euro)
                                                                                         │  calibrate + retarget
                                                                                         │  IK (7 joints, damped LS)
                                                                                         │  finger targets
                                                                                         ▼
                       MuJoCo physics (500 Hz, real time) <── clamp + rate-limit every actuator
```
- **Position.** Palm motion relative to your calibrated neutral, in image units, plus apparent hand size for
  toward/away. Monocular depth is never treated as metres. It is mapped into a bounded box (+-12/15/12 cm) around each
  robot hand's home position.
- **Arms.** Damped-least-squares IK on each arm's 7 joints, with per-step velocity limits, joint-limit margins, adaptive
  damping near singularities and a null-space pull toward home. Legs, waist and head hold their pose (no locomotion).
- **Fingers.** Curl, thumb and pinch measures from MediaPipe's 3-D landmarks, normalised by your open/fist/pinch
  captures, blend between gesture poses derived from the robot model and checked for self-collision.

## Robots
| `--robot` | model | hands | arms |
|---|---|---|---|
| `inspire` (default) | G1 29-DoF from Unitree's URDF + Inspire RH56DFQ hands (`scripts/inspire_scene.py`) | 5 fingers, 6 actuators per hand | right + left |
| `dex3` | MuJoCo Menagerie G1 + Dex3-1 hands (`scripts/grasp_scene.py`) | 3 fingers, 7 actuators per hand | right |

## Repository layout
```
livehand/            teleoperation package (python -m livehand)
  tracking.py          camera + MediaPipe threads, two-hand assignment, synthetic source
  features.py          landmarks -> palm position, palm frame, finger curls, pinch
  calibration.py       neutral / range / open / fist / pinch captures, save/load
  retarget.py          human -> robot mapping (position, orientation, fingers)
  arm_ik.py            7-DoF arm IK
  safety.py            clamping, rate limits, watchdog, enable / e-stop state machine
  sim.py               robot models, per-arm controller
  app.py, status.py    entry point, viewer + camera window, keys, status panel
  config.toml          every tunable (mapping, filters, limits, gesture poses)
  tools/               camera_check, replay_check, synthetic_check, gesture/home pose derivation
  tests/               unit + integration tests (no camera or display needed)
scripts/             robot scenes, viewers, physics and grasp experiments
docs/                model notes, grasp experiments, images
logs/                recorded results of the checks and experiments
models/              downloaded robot models (git-ignored)
```

## Setup from scratch
Large, re-downloadable assets are git-ignored. On a fresh machine:
```bash
git clone git@github.com:Greninja44/chess2bot.git ~/chess2bot && cd ~/chess2bot

# 1. Robot models: MuJoCo Menagerie (pinned commit) and Unitree's URDFs (for the Inspire hands)
git clone https://github.com/google-deepmind/mujoco_menagerie.git models/mujoco_menagerie
git -C models/mujoco_menagerie checkout f3b6013629c7b94f8ee93fd977b161fa8350f9bc
git -C models/mujoco_menagerie sparse-checkout set unitree_g1
git clone --depth 1 --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_ros.git models/unitree_ros_probe/src
git -C models/unitree_ros_probe/src sparse-checkout set robots/g1_description

# 2. Teleoperation env + MediaPipe hand model (~7.8 MB)
python3 -m venv .venv-livehand && .venv-livehand/bin/pip install -r requirements-livehand.txt
mkdir -p livehand/models && curl -fsSL -o livehand/models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

# 3. Optional: base env for the scripts/ experiments
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```
Never install into the system Python. All versions are pinned (MuJoCo 3.17.0, MediaPipe 1.2.0,
opencv-contrib-python 5.0.0.93, NumPy 2.5.4), and all of them support Python 3.14. The laptop webcam needs no extra
driver (in-kernel `uvcvideo`). Not backed up: `livehand/calibration.json` (machine- and person-specific; recalibrate).

## Tests
```bash
.venv-livehand/bin/python -m unittest livehand.tests.test_core -v      # 48 tests, ~6 s, no camera or display
.venv-livehand/bin/python -m livehand.tools.synthetic_check            # real-time end-to-end, both arms
.venv-livehand/bin/python -m livehand.tools.replay_check CLIP.mp4      # a recorded webcam clip through the full pipeline
```
The tests cover coordinate mapping, calibration, clamping and invalid input, tracking loss, enable / disable / e-stop,
IK accuracy and limits, the model wiring and home poses, self-collision-free gestures, full-workspace reachability
without body collisions, two-hand control and hand separation.

## Roadmap
- [x] G1 in MuJoCo, standing and stable; Dex3-1 and Inspire hands actuated
- [x] Real-time webcam teleoperation: both arms, 5-finger hands, safety layer
- [x] Scripted pawn grasp with the Dex3-1 hand, plus a map of where it succeeds ([docs/experiments.md](docs/experiments.md))
- [ ] Live two-hand session tuning (directions, gains, gesture feel)
- [ ] Objects in the scene: teleoperated pick-and-place (first target: chess pieces)
- [ ] Recording teleoperation demonstrations for learning a policy

## Limitations
- Simulation only; nothing here talks to a physical robot.
- A single RGB camera gives only coarse toward/away control; orientation control is experimental (off by default).
- Two-hand teleoperation is verified with synthetic hands and a one-hand real recording; a live two-hand session is still
  to be tested.
- The robot stands under stiff position control (no balance controller); the two arms are kept apart by a sideways
  separation limit, not by full collision checking.

## More documentation
- [livehand/README.md](livehand/README.md): teleoperation details, mapping, safety, troubleshooting.
- [docs/g1_model.md](docs/g1_model.md): G1 model source, joints and actuators, verification, the Inspire hand scene.
- [docs/experiments.md](docs/experiments.md): scripted Dex3-1 pawn grasp and grasp-region mapping.

## Credits and licenses
- Unitree G1 model: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) `unitree_g1/`, BSD-3-Clause,
  Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics").
- G1 with Inspire hands: Unitree's [unitree_ros](https://github.com/unitreerobotics/unitree_ros) URDF, BSD-3-Clause.
- Hand tracking: [MediaPipe](https://github.com/google-ai-edge/mediapipe) HandLandmarker, Apache-2.0.
- Simulation: [MuJoCo](https://github.com/google-deepmind/mujoco), Apache-2.0.

The models are downloaded, not redistributed here. This repository's own code has no license file yet.
