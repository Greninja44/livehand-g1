"""Optional: render one offscreen frame of the G1 to a PNG (visual check, no interactive viewer)."""
import argparse
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
G1_DIR = ROOT / "models" / "mujoco_menagerie" / "unitree_g1"

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["base", "hands"], default="base")
ap.add_argument("--out", type=Path, default=ROOT / "logs" / "g1_snapshot.png")
ap.add_argument("--seconds", type=float, default=2.0)
a = ap.parse_args()

m = mujoco.MjModel.from_xml_path(str(G1_DIR / ("scene.xml" if a.variant == "base" else "scene_with_hands.xml")))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
ctrl0 = d.ctrl.copy()
while d.time < a.seconds:
    d.ctrl[:] = ctrl0
    mujoco.mj_step(m, d)

r = mujoco.Renderer(m, height=480, width=640)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0, 0, 0.7]; cam.distance = 3.0; cam.azimuth = 140; cam.elevation = -15
r.update_scene(d, cam)
img = r.render()
r.close()
import PIL.Image  # installed with mujoco's viewer deps? fall back below
PIL.Image.fromarray(img).save(a.out)
print(f"saved {a.out}  nonuniform pixels: {np.std(img) > 5}  sim_time={d.time:.2f} pelvis_z={d.qpos[2]:.3f}")
