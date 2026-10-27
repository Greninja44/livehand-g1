"""Small offscreen-render helper shared by the hand/grasp experiment scripts.

Run with MUJOCO_GL=glfw (and the d3d12 env vars from the README on WSLg).
"""
from pathlib import Path

import mujoco
import numpy as np


def save_png(m, d, path, lookat=(0, 0, 0.7), distance=3.0, azimuth=140, elevation=-15,
             width=640, height=480):
    """Render the current state of (m, d) to a PNG. Returns False (no exception) if rendering fails."""
    try:
        import PIL.Image
        r = mujoco.Renderer(m, height=height, width=width)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        r.update_scene(d, cam)
        img = r.render()
        r.close()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        PIL.Image.fromarray(img).save(path)
        return bool(np.std(img) > 5)
    except Exception as e:  # rendering is optional for the tests
        print(f"  (render failed: {type(e).__name__}: {e})")
        return False
