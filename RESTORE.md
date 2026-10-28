# Restoring on a fresh machine

The repo holds code, configs, logs and docs. Large, re-downloadable assets are git-ignored; recreate them:

```bash
git clone <this repo> chess2bot && cd chess2bot

# 1. Base env (G1 + Dex3-1 sim, grasp scripts)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. MuJoCo Menagerie (pinned commit used for all results)
git clone https://github.com/google-deepmind/mujoco_menagerie.git models/mujoco_menagerie
git -C models/mujoco_menagerie checkout f3b6013629c7b94f8ee93fd977b161fa8350f9bc
git -C models/mujoco_menagerie sparse-checkout set unitree_g1

# 3. Unitree URDF (only needed for scripts/inspire_scene.py, view_inspire.py)
git clone --depth 1 --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_ros.git models/unitree_ros_probe/src
git -C models/unitree_ros_probe/src sparse-checkout set robots/g1_description

# 4. LiveHand env + MediaPipe hand model
python3 -m venv .venv-livehand && .venv-livehand/bin/pip install -r requirements-livehand.txt
mkdir -p livehand/models
curl -L -o livehand/models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

Not backed up (machine-specific): `livehand/calibration.json` (recalibrate with n/o/f/p).
WSL-only: webcam needs usbipd-win; native Ubuntu sees /dev/video0 directly.
