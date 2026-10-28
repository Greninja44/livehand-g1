"""Interactive viewer: Unitree G1 with 5-finger Inspire hands. Both hands open and close in a loop.
Runs until the window is closed. Needs a display (WSLg); see README for the GPU env vars."""
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inspire_scene as sc  # noqa: E402


def main():
    m, d, info = sc.build()
    sc.reset(m, d, info)
    aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    acts = {f"{s}_{h}_joint": aid(f"{s}_{h}_joint") for s in "RL" for h in sc.ACTIVE_HAND}
    lo = {n: max(m.actuator_ctrlrange[i, 0], 0.0) for n, i in acts.items()}   # stay on the reachable (>=0) side
    hi = {n: m.actuator_ctrlrange[i, 1] for n, i in acts.items()}
    # thumb pitch only to 60% of range: its coupled joints reach their limits sooner
    for n in acts:
        if "thumb_proximal_pitch" in n:
            hi[n] *= 0.6
    print(f"{m.nu} actuators; 5-finger hands open/close, period 4 s (close the window to exit)", flush=True)
    wrist = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = [0.1, 0, 0.95]; v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.3, 150, -10
        wall0, n = time.time(), 0
        ctrl = info["ctrl0"].copy()
        while v.is_running():
            s = 0.5 - 0.5 * np.cos(2 * np.pi * (d.time - 1.0) / 4.0) if d.time > 1.0 else 0.0
            for name, i in acts.items():
                ctrl[i] = lo[name] + (hi[name] - lo[name]) * s
            d.ctrl[:] = ctrl
            mujoco.mj_step(m, d); n += 1
            if n % 8 == 0:
                v.sync()
            time.sleep(max(0.0, d.time - (time.time() - wall0)))
    print("viewer closed", flush=True)


if __name__ == "__main__":
    main()
