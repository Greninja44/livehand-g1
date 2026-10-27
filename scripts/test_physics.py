"""Physics sanity checks for the Menagerie G1: finite state, asset load, standing stability."""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "models" / "mujoco_menagerie" / "unitree_g1" / "scene.xml"


def run(model_path, steps, seconds_stand, use_keyframe_ctrl):
    m = mujoco.MjModel.from_xml_path(str(model_path))  # raises on missing assets
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(m, d, key)
    ctrl0 = d.ctrl.copy() if use_keyframe_ctrl else np.zeros(m.nu)
    h0 = d.qpos[2]
    print(f"model={model_path}\nstart height z={h0:.3f}  nu={m.nu}  using_keyframe_ctrl={use_keyframe_ctrl}")

    n = max(steps, int(seconds_stand / m.opt.timestep))
    ok = True
    min_z = h0
    for i in range(n):
        d.ctrl[:] = ctrl0
        mujoco.mj_step(m, d)
        min_z = min(min_z, d.qpos[2])
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and np.isfinite(d.qacc).all()):
            print(f"FAIL: non-finite state at step {i}")
            ok = False
            break
    z = d.qpos[2]
    # pelvis up-axis z-component: 1 = upright
    up = d.xmat[1].reshape(3, 3)[2, 2]
    print(f"steps={i+1}  sim_time={d.time:.2f}s  final pelvis z={z:.3f} (min {min_z:.3f})  upright={up:.3f}  max|qvel|={np.abs(d.qvel).max():.3f}")
    standing = z > 0.6 * h0 and up > 0.9
    print("finite state:", "PASS" if ok else "FAIL")
    print("standing at end (z>0.6*start, upright>0.9):", "PASS" if standing else "FAIL (robot fell/collapsed)")
    return ok, standing


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT))
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seconds", type=float, default=5.0, help="sim seconds for standing check")
    ap.add_argument("--zero-ctrl", action="store_true", help="ctrl=0 instead of the keyframe's ctrl")
    a = ap.parse_args()
    ok, standing = run(a.model, a.steps, a.seconds, not a.zero_ctrl)
    sys.exit(0 if ok and standing else 1)
