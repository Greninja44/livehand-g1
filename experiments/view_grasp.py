"""Watch the baseline pawn grasp in the interactive MuJoCo viewer (same sequence as test_chess_grasp.py).

Plays settle -> close -> pause -> lift -> hold once in real time, then keeps the viewer open until closed.
Needs a display (WSLg); see README for the GPU env vars.
"""
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, for robots/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robots import grasp_scene as gs  # noqa: E402
from test_chess_grasp import CONFIG as C  # noqa: E402


def main():
    m, d, info = gs.build_scene({})
    gs.reset(m, d, info)
    aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    f_act, arm_act = [aid(n) for n in gs.FINGERS], [aid(n) for n in gs.ARM]
    key2, _ = gs.solve_pregrasp({**info["params"], "grasp_world_z": info["params"]["grasp_world_z"] + C["lift_height"]})
    qa = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
    arm0, arm1 = info["ctrl0"][arm_act].copy(), np.array([key2[qa(n)] for n in gs.ARM])
    open_t, close_t = np.array(info["params"]["open_pose"], float), np.array(C["finger_close"])
    ctrl = info["ctrl0"].copy()
    t1 = C["t_settle"]; t2 = t1 + C["t_close"]; t3 = t2 + C["t_pause"]; t4 = t3 + C["t_lift"]; t5 = t4 + C["t_hold"]

    def command(t):
        if t1 <= t < t2: ctrl[f_act] = open_t + (close_t - open_t) * (t - t1) / C["t_close"]
        elif t >= t2: ctrl[f_act] = close_t
        if t3 <= t < t4: ctrl[arm_act] = arm0 + (arm1 - arm0) * (t - t3) / C["t_lift"]
        elif t >= t4: ctrl[arm_act] = arm1
        d.ctrl[:] = ctrl

    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = d.site_xpos[site]; v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.7, 150, -20
        print("playing grasp sequence (close the window to exit)")
        wall0, n = time.time(), 0
        while v.is_running():
            command(d.time)
            mujoco.mj_step(m, d); n += 1
            if n % 8 == 0:
                v.sync()
            if d.time < t5:  # real-time pacing while the sequence plays; afterwards just hold
                time.sleep(max(0.0, d.time - (time.time() - wall0)))
            else:
                if d.time - t5 < m.opt.timestep * 1.5:
                    print(f"done: pawn z={d.qpos[-5]:.3f} (rest 0.860)")
                time.sleep(m.opt.timestep)
    print("viewer closed")


if __name__ == "__main__":
    main()
