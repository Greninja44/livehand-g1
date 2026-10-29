# G1 model setup and verification notes

The first stage of this project: getting the Unitree G1 (MuJoCo Menagerie) running reliably in MuJoCo, before any
teleoperation. Kept as a record of what was verified and how. These notes were written on **WSL2**; the project now runs on
native Ubuntu 26.04, where the `GALLIUM_DRIVER` / WSLg variables below are not needed (for offscreen renders use
`MUJOCO_GL=egl`). Scripts referred to here live in `scripts/` and use the base env `.venv` (see the main README).

## Environment used
- Ubuntu 26.04 on **WSL2** (WSLg display), x86_64, 13 GB RAM visible to WSL
- GPU: RTX 4050 Laptop 6 GB, driver 596.49 (Windows side). CUDA is not needed.
- Python 3.12.14 (via `uv`; system Python is 3.14 and was left untouched), MuJoCo 3.17.0

## Install (as originally done on WSL, with `uv` and Python 3.12; for the current setup see the main README)
```bash
mkdir -p ~/robotics/chess2bot && cd ~/robotics/chess2bot
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt   # or: uv pip install mujoco pillow
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git models/mujoco_menagerie
git -C models/mujoco_menagerie sparse-checkout set unitree_g1   # only the G1 (~0.5 GB instead of 2.3 GB)
```
Activate: `source .venv/bin/activate` (or call `.venv/bin/python` directly).

## Run
```bash
# GPU-backed OpenGL on WSLg (recommended; see Troubleshooting)
export GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
python scripts/run_g1.py                  # base 29-DoF G1, runs until the window is closed
python scripts/run_g1.py --variant hands  # G1 with Dex3-1 hands
python scripts/inspect_model.py [--model PATH]   # joints / actuators / ranges
python scripts/test_physics.py [--zero-ctrl] [--model PATH]   # >=1000 steps, finite + standing check
MUJOCO_GL=glfw python scripts/snapshot_g1.py     # optional offscreen PNG (logs/g1_snapshot.png)
```
`run_g1.py --duration N` auto-closes after N seconds (smoke tests). Scripts resolve paths from their own location, not the CWD.

## Model source and license
- Source: https://github.com/google-deepmind/mujoco_menagerie, directory `unitree_g1/`
- Revision: `f3b6013629c7b94f8ee93fd977b161fa8350f9bc` (2026-10-23)
- Derived from Unitree's `g1_29dof_rev_1_0` MJCF (unitree_ros). License: **BSD-3-Clause**, Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics"); full text in `models/mujoco_menagerie/unitree_g1/LICENSE`.
- Entry points: `unitree_g1/scene.xml` (includes `g1.xml`, 29 DoF, no fingers), `scene_with_hands.xml` (includes `g1_with_hands.xml`). Assets: `unitree_g1/assets/` (STL meshes).
- Menagerie notes the actuators are simple position actuators that "need tuning". MJX/mjlab variants exist but are not used here.

## Verification results (actually run, see `logs/`)
| Check | Result |
|---|---|
| `import mujoco` in `.venv` | OK, 3.17.0 |
| Base model: 31 bodies, 30 joints (1 free + 29 hinge), 29 actuators, dt 0.002 s | loads, no missing assets |
| Hands model: 45 bodies, 44 joints, 43 actuators | loads, no missing assets |
| 2500 steps (5 s) from `stand` keyframe, all three configs | finite state; pelvis z 0.790→0.792, upright 1.000; **stable** |
| Same with ctrl=0 (all targets 0, near-straight legs) | also stable (qvel ~0.006) |
| Interactive viewer (`launch_passive`), base and hands | opens, sim time advances, exits normally on close/duration |
| Offscreen render to PNG | robot visible standing on the ground plane |

Stability caveat: this is quiet standing with stiff position control (kp=500) and no disturbance. It is **not** a balance controller; no push test was done.
I could not literally watch the live window from this CLI session. "Viewer works" means it launched without error, ran and closed cleanly. Robot visibility was confirmed from the offscreen render of the same model.

## Joints / actuators (base model; full dump in `logs/inspect_g1.txt`)
- Free base + 12 leg joints (hip pitch/roll/yaw, knee, ankle pitch/roll ×2), 3 waist (yaw/roll/pitch), 7 per arm (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw).
- One **position actuator per hinge**, kp=500 for all, kv between 4.5 and 16.2, `ctrlrange` = joint range. No actuator-level `forcerange`, **but the joints carry `actuatorfrcrange`** (e.g. wrist ±5 N·m, Dex3 fingers ±1.4 N·m, thumb_0 ±2.45 N·m), which clamps the applied torque (check `qfrc_actuator`, not `actuator_force`, which is the unclamped value). Control = target angle in radians.
- Example ranges (rad): knee [-0.087, 2.880], elbow [-1.047, 2.094], waist_yaw ±2.618, wrist_pitch/yaw ±1.614.
- A `stand` keyframe is provided (pose + ctrl); the scripts reset to it and hold its ctrl.
- Note (corrected): earlier text here said there were no force limits; that only looked at actuator-level `forcerange`. Joint `actuatorfrcrange` limits do exist. Hand joints also have `frictionloss=0.3` and `armature=0.01`.

## Dexterous hand availability (separate from base G1)
- `scene_with_hands.xml` **does** include controllable hands, but they are **Unitree Dex3-1 three-finger hands, not five-finger hands**.
- Per hand 7 position actuators: `*_hand_thumb_0/1/2`, `*_hand_index_0/1`, `*_hand_middle_0/1` (limits e.g. left index ±: [-1.571, 0] / [-1.745, 0]; right side signs are mirrored). Full list in `logs/inspect_g1_hands.txt`.
- Mechanically represented: these are real hinge joints with actuators; 14 of 28 hand geoms have collision enabled. Loads and stands stably.
- Not verified: grasping an object, contact quality, finger gain tuning. Not tested with chess pieces. Five-finger hands (e.g. Inspire) are not in Menagerie's G1. (Done later from Unitree's URDF: see the Inspire section below.)

## Known limitations
- Viewer runs at about 0.6-0.75x real time on this WSLg setup (raw physics alone is ~38x). Viewer sync, not physics, is the limit.
- Position gains are untuned; no walking/balance controller exists.
- `models/mujoco_menagerie` is a nested git clone (gitignored); re-clone with the commands above.

## Troubleshooting (WSL / Linux graphics)
- Check you are on WSLg: `echo $DISPLAY $WAYLAND_DISPLAY`, `ls /mnt/wslg`, `ls /dev/dxg`.
- Default GL here is `llvmpipe` (CPU software). Works but slower. Startup prints harmless `libEGL warning` / `ZINK: failed to choose pdev` lines.
- To use the GPU: `export GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`. Without the second variable Mesa picks the Intel iGPU. Confirm with the `GL_RENDERER` string (`D3D12 (NVIDIA GeForce RTX 4050 Laptop GPU)`).
- `libdecor-gtk.so failed to init` is harmless (window has no decorations).
- Offscreen `Renderer` is limited to 640x480 unless the XML sets `<visual><global offwidth= offheight=/>`.
- If no window opens: confirm `DISPLAY` is set, run `wsl --update` from Windows, and restart WSL.

## 5-finger Inspire hands (`robots/inspire_scene.py`)
- Source: Unitree's official URDF `g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf` from
  [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros) (BSD-3-Clause), sparse-cloned into
  `models/unitree_ros_probe/src` (git-ignored; see "Setup from scratch" in the main README).
- The URDF has no actuators, free base, floor, damping or armature; `inspire_scene.build()` adds them with `MjSpec`: body
  position actuators with the same gains as Menagerie's G1, and per hand 6 finger position actuators (thumb yaw, thumb
  pitch, index, middle, ring, pinky). The other 6 finger joints follow through the URDF mimic constraints.
- `experiments/test_inspire_actuation.py` (`logs/inspire_actuation.json`): 9/12 single-actuator checks pass on the default
  scene; the 3 failures are the thumb. Cause (verified later): the thumb-base meshes overlap the palm by 3-5 mm at rest and
  that contact fights the thumb actuators. LiveHand builds the scene with that one contact pair excluded (via the optional
  `edit` hook of `build()`), after which the thumb tracks exactly; the default scene is unchanged.
- Thumb pitch is usable only to 0.5 rad of its 0.6 rad range: its coupled joints reach their limits there.
