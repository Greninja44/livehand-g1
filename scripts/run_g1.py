"""Launch the Unitree G1 (MuJoCo Menagerie) in the interactive MuJoCo viewer.

Runs until the viewer window is closed (or --duration elapses, for smoke tests).
The robot is reset to the model's `stand` keyframe and holds that keyframe's
position targets; nothing else is commanded.
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
G1_DIR = ROOT / "models" / "mujoco_menagerie" / "unitree_g1"
SCENES = {"base": G1_DIR / "scene.xml", "hands": G1_DIR / "scene_with_hands.xml"}


def load(path: Path):
    try:
        m = mujoco.MjModel.from_xml_path(str(path))
    except Exception as e:  # missing file / mesh / bad XML
        sys.exit(f"ERROR: failed to load model {path}\n  {type(e).__name__}: {e}")
    return m, mujoco.MjData(m)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=SCENES, default="base",
                    help="base = 29-DoF G1 (no fingers); hands = G1 with Dex3-1 3-finger hands")
    ap.add_argument("--model", type=Path, help="override: explicit MJCF path")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="auto-close after N wall seconds (0 = run until closed)")
    args = ap.parse_args()

    path = (args.model or SCENES[args.variant]).resolve()
    m, d = load(path)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(m, d, key)
    ctrl0 = d.ctrl.copy()
    mujoco.mj_forward(m, d)

    print(f"model path : {path}")
    print(f"bodies={m.nbody} joints={m.njnt} actuators={m.nu} timestep={m.opt.timestep}s")
    print(f"MuJoCo {mujoco.__version__}")

    try:
        viewer = mujoco.viewer.launch_passive(m, d)
    except Exception as e:
        sys.exit(f"ERROR: could not open viewer ({type(e).__name__}: {e}).\n"
                 "  Check DISPLAY/WAYLAND_DISPLAY and GL; see README troubleshooting.")

    wall0 = time.time()
    last_report = wall0
    steps_per_sync = max(1, round(1 / 60 / m.opt.timestep))  # ~60 Hz viewer updates
    n = 0
    with viewer:
        while viewer.is_running():
            step_start = time.time()
            d.ctrl[:] = ctrl0
            mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all():
                print("ERROR: non-finite state; stopping", file=sys.stderr)
                break
            n += 1
            if n % steps_per_sync == 0:
                viewer.sync()
            now = time.time()
            if now - last_report >= 2.0:
                print(f"wall={now - wall0:5.1f}s sim_time={d.time:6.2f}s pelvis_z={d.qpos[2]:.3f}")
                last_report = now
            if args.duration and now - wall0 > args.duration:
                break
            sleep = m.opt.timestep - (time.time() - step_start)
            if sleep > 0:
                time.sleep(sleep)
    print(f"viewer closed normally; final sim_time={d.time:.2f}s")


if __name__ == "__main__":
    main()
