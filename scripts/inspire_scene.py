"""G1 with 5-finger Inspire (RH56DFQ) hands, built with MjSpec from Unitree's official URDF.

Source: unitreerobotics/unitree_ros (BSD-3), robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf
The URDF has no actuators, free base, floor, damping or armature, so this module adds them:
  * pelvis kept as a body (fusestatic off) with a free joint, placed at standing height
  * body-joint position actuators: same kp/kv as the Menagerie G1 (identical joint names), torque limit = URDF effort
  * finger position actuators on the 6 active joints per hand (the 6 coupled joints follow via the URDF's
    mimic constraints, which MuJoCo imports as joint equality constraints)
  * floor + sky/lights like the Menagerie scene. Nothing in models/ is modified.
"""
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
URDF = ROOT / "models" / "unitree_ros_probe" / "src" / "robots" / "g1_description" / "g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"
MENAGERIE = ROOT / "models" / "mujoco_menagerie" / "unitree_g1" / "scene.xml"

ACTIVE_HAND = ["thumb_proximal_yaw", "thumb_proximal_pitch", "index_proximal", "middle_proximal",
               "ring_proximal", "pinky_proximal"]
HAND_KP, HAND_KV = 10.0, 0.2      # N*m/rad, N*m*s/rad; torque is clamped by the URDF effort (1 N*m)
HAND_ARMATURE, HAND_FRICTIONLOSS = 0.001, 0.02
BODY_ARMATURE, BODY_FRICTIONLOSS = 0.01, 0.3     # same as Menagerie g1 defaults
STAND_Z = 0.793

# Standing pose for the 29 body joints. Legs/waist straight, arms slightly out so the larger hands clear the thighs.
STAND = dict(left_shoulder_pitch_joint=0.2, left_shoulder_roll_joint=0.45, left_elbow_joint=1.0,
             right_shoulder_pitch_joint=0.2, right_shoulder_roll_joint=-0.45, right_elbow_joint=1.0)


def hand_joint(side, name):
    return f"{side}_{name}_joint"


def build(stand=None, edit=None):
    """Return (model, data, info) with data reset to the standing pose, hands open, ctrl = held targets.
    edit: optional callable(spec) applied just before compiling (e.g. to add sites); default changes nothing."""
    men = mujoco.MjModel.from_xml_path(str(MENAGERIE))
    men_gain = {}
    for i in range(men.nu):
        j = mujoco.mj_id2name(men, mujoco.mjtObj.mjOBJ_JOINT, men.actuator_trnid[i, 0])
        men_gain[j] = (men.actuator_gainprm[i, 0], -men.actuator_biasprm[i, 2])

    spec = mujoco.MjSpec.from_file(str(URDF))
    spec.meshdir = "."
    spec.compiler.fusestatic = 0
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    pelvis = spec.body("pelvis")
    pelvis.pos = [0, 0, STAND_Z]
    pelvis.add_freejoint(name="floating_base_joint")

    # visuals / ground (copied from Menagerie scene.xml)
    spec.visual.headlight.diffuse = [0.6] * 3
    spec.visual.headlight.ambient = [0.1] * 3
    spec.visual.headlight.specular = [0.9] * 3
    spec.visual.rgba.haze = [0.15, 0.25, 0.35, 1]
    spec.visual.global_.azimuth, spec.visual.global_.elevation = 140, -20
    spec.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.3, 0.5, 0.7], rgb2=[0, 0, 0], width=512, height=3072)
    spec.add_texture(name="groundplane", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     mark=mujoco.mjtMark.mjMARK_EDGE, rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3],
                     markrgb=[0.8, 0.8, 0.8], width=300, height=300)
    spec.add_material(name="groundplane", textures=["", "groundplane"], texuniform=True, texrepeat=[5, 5], reflectance=0.2)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="groundplane")

    # joint dynamics (URDF has none)
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        hand = j.name.startswith(("R_", "L_"))
        j.armature = HAND_ARMATURE if hand else BODY_ARMATURE
        j.frictionloss = HAND_FRICTIONLOSS if hand else BODY_FRICTIONLOSS
        if hand:
            j.damping[0] = 0.01

    # actuators
    for j in spec.joints:
        n = j.name
        if n in men_gain:
            kp, kv = men_gain[n]
        elif n.startswith(("R_", "L_")) and any(n == f"{n[0]}_{h}_joint" for h in ACTIVE_HAND):
            kp, kv = HAND_KP, HAND_KV
        else:
            continue
        a = spec.add_actuator(name=n, target=n, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                              gaintype=mujoco.mjtGain.mjGAIN_FIXED, biastype=mujoco.mjtBias.mjBIAS_AFFINE)
        a.gainprm[0] = kp
        a.biasprm[:3] = [0, -kp, -kv]
        a.ctrllimited = True
        a.ctrlrange = [j.range[0], j.range[1]]

    if edit is not None:
        edit(spec)
    m = spec.compile()
    d = mujoco.MjData(m)
    pose = {**STAND, **(stand or {})}
    mujoco.mj_resetData(m, d)
    d.qpos[3] = 1.0  # free-joint quaternion w (pelvis at spec pos)
    for n, v in pose.items():
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]] = v
    ctrl0 = np.zeros(m.nu)
    for i in range(m.nu):
        ctrl0[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
    d.ctrl[:] = ctrl0
    mujoco.mj_forward(m, d)
    return m, d, dict(ctrl0=ctrl0, qpos0=d.qpos.copy())


def reset(m, d, info):
    mujoco.mj_resetData(m, d)
    d.qpos[:] = info["qpos0"]
    d.ctrl[:] = info["ctrl0"]
    mujoco.mj_forward(m, d)
