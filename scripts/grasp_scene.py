"""Programmatic grasp-test scene: Menagerie G1+Dex3-1 hands, plus a table and a pawn-sized cylinder.

Nothing under models/mujoco_menagerie is modified: the scene is built with MjSpec from the
original scene_with_hands.xml. The right arm is moved to a pre-grasp pose by numerical IK, and the
object is placed from forward kinematics of a grasp site defined in the right-wrist frame.
"""
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "models" / "mujoco_menagerie" / "unitree_g1" / "scene_with_hands.xml"

ARM = [f"right_{j}_joint" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                                    "wrist_roll", "wrist_pitch", "wrist_yaw")]
FINGERS = ["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
           "right_hand_index_0_joint", "right_hand_index_1_joint",
           "right_hand_middle_0_joint", "right_hand_middle_1_joint"]

DEFAULTS = dict(
    # object (hand = right_wrist_yaw_link frame: x along fingers, y palm normal, z across palm)
    obj_radius=0.012, obj_height=0.045, obj_mass=0.010, obj_friction=(1.2, 0.01, 0.001),
    obj_hand_xy=(0.1452, 0.0447),
    open_pose=(-0.07, 0.43, 0, 0, 0, 0, 0),   # right-hand joint angles (FINGERS order) at reset; also the initial ctrl          # object axis position in the hand frame (x, y)
    table_top_hand_z=-0.047,           # table surface in hand z: below the lowest finger-hull point (-0.042)
    # where the grasp site should be in the world for the pre-grasp (hand z up, x forward)
    grasp_world_xy=(0.30, -0.16),
    grasp_world_z=0.86,                # world height of the table-top-referenced hand origin (see build)
)


def _jid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)


def solve_pregrasp(params, iters=400):
    """IK for the 7 right-arm joints so the wrist frame has x=world +x, z=world up and the
    hand-frame point `grasp site` is at the requested world position. Returns (qpos_keyframe, report)."""
    p = {**DEFAULTS, **params}
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    site_hand = np.array([p["obj_hand_xy"][0], p["obj_hand_xy"][1], p["table_top_hand_z"] + p["obj_height"] / 2])
    target_site = np.array([*p["grasp_world_xy"], p["grasp_world_z"]])
    Rd = np.eye(3)  # columns: hand x -> world x, hand y -> world y, hand z -> world z
    dofs = [m.jnt_dofadr[_jid(m, n)] for n in ARM]
    qadr = [m.jnt_qposadr[_jid(m, n)] for n in ARM]
    lo = np.array([m.jnt_range[_jid(m, n)][0] for n in ARM]); hi = np.array([m.jnt_range[_jid(m, n)][1] for n in ARM])
    # start from an elbow-bent, forward-reaching guess to avoid the hip-contact keyframe pose
    d.qpos[qadr] = [-0.6, -0.2, 0.0, 1.0, 0.0, 0.0, 0.0]
    for _ in range(iters):
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        R = d.xmat[body].reshape(3, 3)
        pos = d.xpos[body] + R @ site_hand
        e_pos = target_site - pos
        e_rot = 0.5 * sum(np.cross(R[:, i], Rd[:, i]) for i in range(3))
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jp, jr, pos, body)
        J = np.vstack([jp[:, dofs], jr[:, dofs]])
        e = np.concatenate([e_pos, e_rot])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        d.qpos[qadr] = np.clip(d.qpos[qadr] + 0.5 * dq, lo, hi)
    mujoco.mj_kinematics(m, d)
    R = d.xmat[body].reshape(3, 3)
    pos = d.xpos[body] + R @ site_hand
    rep = dict(pos_err=float(np.linalg.norm(target_site - pos)),
               rot_err=float(np.linalg.norm(0.5 * sum(np.cross(R[:, i], Rd[:, i]) for i in range(3)))),
               arm_q=d.qpos[qadr].tolist(), at_limit=[n for n, q, l, h in zip(ARM, d.qpos[qadr], lo, hi)
                                                     if min(q - l, h - q) < 1e-3])
    q = d.qpos.copy()
    mujoco.mj_resetDataKeyframe(m, d, 0)
    key = d.qpos.copy(); key[qadr] = q[qadr]
    return key, rep


def build_scene(params=None, with_object=True):
    """Return (model, data, info). Data is reset to the pre-grasp keyframe ('pregrasp')."""
    p = {**DEFAULTS, **(params or {})}
    key_q, rep = solve_pregrasp(p)
    spec = mujoco.MjSpec.from_file(str(SCENE))
    wrist = spec.body("right_wrist_yaw_link")
    site_hand = [p["obj_hand_xy"][0], p["obj_hand_xy"][1], p["table_top_hand_z"] + p["obj_height"] / 2]
    wrist.add_site(name="grasp_site", pos=site_hand, size=[0.004] * 3, rgba=[1, 0, 0, 1], group=4)

    # world pose of the grasp site from FK (needed to place object and table)
    m0 = spec.compile(); d0 = mujoco.MjData(m0)
    d0.qpos[:] = key_q; mujoco.mj_forward(m0, d0)
    sid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    obj_pos = d0.site_xpos[sid].copy()

    if with_object:
        table_top = obj_pos[2] - p["obj_height"] / 2
        wb = spec.worldbody
        # support surface: world-aligned box, starting past the palm so it never overlaps the hand at init
        tx0 = obj_pos[0] - 0.035
        tb = wb.add_body(name="table", pos=[tx0 + 0.20, obj_pos[1], table_top - 0.02])
        tb.add_geom(name="table_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.20, 0.30, 0.02],
                    friction=[1.0, 0.005, 0.0001], rgba=[0.55, 0.4, 0.25, 1])
        ob = wb.add_body(name="pawn", pos=obj_pos.tolist())
        ob.add_freejoint(name="pawn_free")
        ob.add_geom(name="pawn_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=[p["obj_radius"], p["obj_height"] / 2, 0], mass=p["obj_mass"],
                    friction=list(p["obj_friction"]), condim=4, rgba=[0.95, 0.95, 0.9, 1])
    m = spec.compile()
    d = mujoco.MjData(m)
    # pre-grasp keyframe: robot keyframe with IK'd arm (+ object pose appended if present)
    qpos = np.zeros(m.nq)
    qpos[:len(key_q)] = key_q
    if with_object:
        qpos[len(key_q):len(key_q) + 7] = [*obj_pos, 1, 0, 0, 0]
    for n, v in zip(FINGERS, p["open_pose"]):
        qpos[m.jnt_qposadr[_jid(m, n)]] = v
    return m, d, dict(qpos0=qpos, ctrl0=_ctrl_from_qpos(m, qpos), obj_pos=obj_pos, ik=rep, params=p,
                      site_hand=site_hand)


def _ctrl_from_qpos(m, qpos):
    """Position-actuator ctrl that holds the joint angles in qpos (joint-targets = current angles)."""
    ctrl = np.zeros(m.nu)
    for i in range(m.nu):
        j = m.actuator_trnid[i, 0]
        ctrl[i] = qpos[m.jnt_qposadr[j]]
    return ctrl


def reset(m, d, info):
    mujoco.mj_resetData(m, d)
    d.qpos[:] = info["qpos0"]
    d.ctrl[:] = info["ctrl0"]
    mujoco.mj_forward(m, d)


def hand_clearance(m, d, margin=0.5):
    """Min signed distance (m) from the pawn collision geom to each collidable right-hand/palm body."""
    pawn_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pawn_geom")
    out = {}
    for g in range(m.ngeom):
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
        if (b.startswith("right_hand") or b == "right_wrist_yaw_link") and (m.geom_contype[g] or m.geom_conaffinity[g]):
            out[b] = min(out.get(b, 9.0), float(mujoco.mj_geomDistance(m, d, pawn_geom, g, margin, None)))
    return out
