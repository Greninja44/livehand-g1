# chess2bot — Unitree G1 in MuJoCo (setup stage)

Goal of this stage: get the Unitree G1 running reliably in MuJoCo. No Chess2Bot logic yet.

## Environment used
- Ubuntu 26.04 on **WSL2** (WSLg display), x86_64, 13 GB RAM visible to WSL
- GPU: RTX 4050 Laptop 6 GB, driver 596.49 (Windows side). CUDA is not needed.
- Python 3.12.14 (via `uv`; system Python is 3.14 and was left untouched), MuJoCo 3.17.0

## Install (exact commands used)
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
- Not verified: grasping an object, contact quality, finger gain tuning. Not tested with chess pieces. Five-finger hands (e.g. Inspire) are not in Menagerie's G1; integrating one is a future task and was not attempted.

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


## Experiment: Dex3-1 pawn grasp (right hand)

Goal: show that the G1 right hand can **physically** grasp and lift a pawn-sized cylinder. No learned policy.

### Result: SUCCESS (scripted, open-loop) — `logs/grasp_test.json`
- Pawn: cylinder r=12 mm (24 mm dia), h=45 mm, 10 g, friction (1.2, 0.01, 0.001), on a table box. Arm lift commanded +50 mm.
- Pawn rose **52.5 mm**, stayed above the 30 mm success height for the full **1.0 s** hold, slip in the wrist frame **1.1 mm**.
- Contacts at the end of the hold: `right_hand_thumb_2_link`, `right_hand_middle_0_link` **and the palm** (`right_wrist_yaw_link`). It is a thumb + middle-finger pinch with the palm cradling the pawn, not a pure fingertip pinch. The index finger does not touch it (it sits 31 mm away from the middle finger across the palm and is higher than the pawn).
- Max pawn acceleration 9.8 m/s²; no non-finite state; no joint-limit violations; robot upright (min 0.9999).
- Negative control (same arm lift, fingers left open): pawn stays on the table, rise 0.0 mm.
- Determinism: 3 fresh builds/resets give identical final pawn positions (spread 0.0).
- Starting clearance from the pawn to every hand collision geom is at least 22.7 mm (no initial penetration). No weld/equality constraints (`m.neq == 0`), no teleporting, no collisions disabled, actuator gains unchanged.
- Snapshots: `logs/grasp_init.png`, `logs/grasp_closed.png`, `logs/grasp_final.png`; per-step command/response trace in `logs/grasp_trace.csv`.

### How it is built
- `scripts/grasp_scene.py` builds the scene with `MjSpec` from the **unmodified** `scene_with_hands.xml`: adds a `grasp_site` on `right_wrist_yaw_link`, a table, and a free-joint pawn.
- The right arm is moved to a pre-grasp pose by numerical IK (wrist x → world +x, z → up). The pawn is placed from **forward kinematics** of the grasp site (hand-frame x=0.145, y=0.045 m), table top 5 mm below the pawn base so the lowest finger-hull point stays clear.
- The lift is a second IK solution 50 mm higher, interpolated over 1.5 s.
- The closing targets and pawn position were found by an **offline random search** with constraints (start clearance ≥ 2 mm) and then checked for robustness (12/12 under ±3 mm pawn position and ±0.05 rad target noise). The first search lacked the clearance constraint and produced a grasp that started with a 2.3 mm penetration of thumb_2; the test caught this and the search was redone. So these numbers are tuned to this scene; they are not a general grasp.

### Run
```bash
export MUJOCO_GL=glfw GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA   # snapshots (see Troubleshooting)
python scripts/test_hand_actuation.py --pose pregrasp --screenshots   # 9/9 expected
python scripts/test_hand_actuation.py --pose keyframe                 # 8/9 expected (see below)
python scripts/test_chess_grasp.py            # add --no-render to skip PNGs; exit code 0 only if all checks pass
```

### Hand actuation findings (`logs/hand_actuation_*.json`)
Each of the seven right-hand actuators was commanded individually to 50% of each allowed side of its range from a fresh reset; all respond with ~0 tracking error and move no other hand joint, so the fingers are independently controllable (position control, targets in rad).
- **Pre-grasp pose: 9/9 pass.**
- **Menagerie `stand` keyframe: 8/9.** `right_hand_thumb_1_joint` in the + direction fails. Cause (verified): at the keyframe the arm hangs with the hand against the thigh, and `right_hand_thumb_2_link` **penetrates `right_hip_pitch_link` by up to ~3.9 cm** for thumb_1 ≥ −0.24 rad. The contact throws the thumb to its limit (the joint torque is clamped at 1.4 N·m so the actuator cannot push back). The hands are therefore not meaningfully usable from the stock keyframe; the earlier "stable standing" test did not look at the hands.
- Intrinsic: `thumb_1_link` (box) vs the palm hull has a persistent ~0.3 mm contact at the rest pose in both hands. Benign in the tests run here.

### Limitations
- Open-loop and scene-specific; success depends on the exact pawn pose. Not a perception/policy result.
- Only the right hand, one pawn orientation (upright, hand z-axis vertical). Pawn is held partly by the palm, which a real hand might not allow.
- The robot is free-standing (no pelvis weld) with stiff PD control; it stays upright for this ~5 s sequence, but this is not a balance controller.
- A 31 mm gap between index and middle fingers and a thumb that sweeps in the same plane limit what the Dex3-1 can grasp; larger or lying objects were not explored.
- Simulated friction/mass values are assumptions. No sim-to-real claims.


## Experiment: grasp-region mapping (`scripts/map_grasp_region.py`)

Measures where the **fixed, scripted** baseline grasp works as the pawn pose varies. Each trial is a fresh build and reset that replays the exact `test_chess_grasp.py` sequence (same arm pose, open-hand pose, closing targets, lift, 1.0 s hold). Nothing is re-optimised per trial; actuator gains and physics are unchanged. Only the pawn body moves, after the arm pose and table are fixed from the nominal pose (`grasp_scene.build_scene(pawn_offset_mm=..., pawn_yaw_deg=...)`).

### Run
```bash
export MUJOCO_GL=glfw GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
python scripts/map_grasp_region.py                    # quick: dx,dy in -6..6 step 3 mm, yaw -30/0/30 = 75 trials (~1.5 min)
python scripts/map_grasp_region.py --full --out-dir logs/grasp_region_full   # +-9 mm, yaw 0..+-45 step 15 = 343 trials (~7 min)
# custom: --x-offsets-mm -6,0,6 --y-offsets-mm ... --yaws-deg ... --repeats 3 --shuffle --seed 1 --min-clearance-mm 2
python scripts/test_map_grasp_region.py               # lightweight checks (grid, symmetry, ordering, classification, aggregation, clearance)
```
A baseline gate (nominal grasp + open-finger control) runs first; if it fails the sweep aborts with an error. Outputs go under `logs/` (`--out-dir`): `grasp_region_trials.csv` (one row per trial), `grasp_region_summary.json`, `grasp_region_summary.txt` (grids per yaw), `grasp_region_map.png` (PIL heat-map, no plotting dependency).

### Coordinate convention
- `dx` (mm): along the right-wrist x axis = the direction the fingers point = world +x (forward).
- `dy` (mm): along the wrist y axis = palm normal = world +y (toward the robot's midline). Positive dy moves the pawn away from the palm toward the fingertip/thumb side.
- `dz` is always 0 (the pawn rests on the table). The offset is applied through the wrist rotation at the pre-grasp pose (identity to ~1e-16 there).
- `yaw` (deg): rotation of the upright pawn about its vertical axis. **The pawn is an axisymmetric cylinder, so yaw does not change its geometry.** Differences between yaw rows are numerical sensitivity of contact generation at marginal grasps, not a physical yaw effect. Yaw is kept as an independent dimension for future non-symmetric pieces.

### Success definition and categories
Success is exactly the per-trial part of the baseline test (`test_chess_grasp.trial_checks`): finite state, no joint-limit violation, upright robot, at least two right-hand links (not counting the table) touching the pawn at the end, pawn at least 30 mm above its rest height at the end, above 30 mm for the full 1.0 s hold, no table/floor contact during the hold, slip in the wrist frame at most 10 mm.
- `invalid_initial_clearance`: the pawn starts closer than `--min-clearance-mm` (default 2) to any hand collision geom. Closing is never run; **excluded from the success-rate denominator** and counted separately. It means the placement is geometrically infeasible, not that the grasp failed.
- `grasp_failure`: no hand contact after closure. `lift_failure`: peak rise never reached 30 mm. `hold_failure`: it rose past 30 mm but a hold criterion failed (typically the pawn was dropped). `sim_error`: exception, non-finite state, joint-limit or balance violation.
- CSV columns include trial id, seed (informational: the simulation is deterministic and uses no randomness; only `--shuffle` uses the seed), offsets, initial minimum clearance, closure/hand-contact flags, peak height above the table-rest height, threshold/hold flags, slip, max pawn acceleration (same definition as the baseline: max per-step |dv|/dt of the pawn, m/s^2), final contacting links, category and diagnostic.

### Observed results (only the tested ranges)
- Quick grid (dx,dy ±6 mm, yaw ±30°): **73/75 valid trials succeeded (97.3%)**, 0 invalid. The two failures were at (dx=0, dy=+6) for yaw 0° and +30°: the pawn rose 34–43 mm and then fell to the floor.
- Full grid (dx,dy ±9 mm, yaw 0..±45°): **319/343 (93.0%)**, 0 invalid (minimum start clearance 13.7 mm). Failures: 17 hold, 6 lift, 1 grasp.
- The working region is the central ~±6 mm square in dx and dy, with its reliable edge at about dy ≤ +3 mm and dx ≤ +6 mm. Failures concentrate on the +dx edge (pawn 9 mm farther along the fingers, at dy=0 and dy=+9) and the +dy edge (dy +6/+9, mostly at dx = 0 or −6). Negative dy (toward the palm) succeeded at every tested dx and yaw down to −9 mm.
- Nothing is claimed beyond ±9 mm, upright pawn, this one hand/arm pose, this table, this mass and friction. Sparse isolated failures inside the region (e.g. dx=+6, dy=+3 at yaw 15° only) show the boundary is ragged, not a clean contour.
- Repeats are bit-identical (determinism checked with `--repeats 2 --shuffle`), so repeated trials add no statistical information unless noise is added.


## LiveHand-G1 (webcam hand -> simulated G1 + Dex3-1)
Real-time hand retargeting lives in `livehand/` (own env: `.venv-livehand`, deps in `requirements-livehand.txt`). Run
`.venv-livehand/bin/python -m livehand`; see `livehand/README.md` for install, keys, mapping, safety behaviour, tests and the
manual steps to attach your webcam to WSL (not testable from this session).
