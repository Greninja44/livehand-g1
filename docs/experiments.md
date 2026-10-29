# Grasp experiments (Dex3-1, scripted)

Scripted, open-loop grasp experiments with the 3-finger Dex3-1 hand, done before teleoperation. They are groundwork for the
chess pick-and-place goal: they show the simulated hand can physically lift a pawn-sized piece, and how sensitive that
fixed grasp is to where the piece is. These notes were written on WSL2: on native Ubuntu drop the `GALLIUM_DRIVER` /
`MESA_D3D12_*` variables and use `MUJOCO_GL=egl` for snapshots. Scripts use the base env `.venv`.

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
