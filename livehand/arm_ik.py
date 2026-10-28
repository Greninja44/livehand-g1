"""Bounded damped-least-squares IK for the 7 right-arm joints of the actual G1 model.

Only the arm joints are solved; torso, head, legs and all other actuators are held by the caller. Targets are
expressed in the pelvis frame so small base sway does not move the target. Each call performs ONE bounded step
(joint velocity limit, joint-limit margin, error clamp, adaptive damping near singularities, null-space pull to home).
"""
import mujoco
import numpy as np


class ArmIK:
    def __init__(self, m, joint_names, site, home_q, cfg, pelvis="pelvis"):
        self.m = m
        self.cfg = cfg.arm
        self.kd = mujoco.MjData(m)               # kinematic scratch data (never stepped)
        self.jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        assert all(j >= 0 for j in self.jid), f"unknown arm joint in {joint_names}"
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.jid])
        self.dadr = np.array([m.jnt_dofadr[j] for j in self.jid])
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
        self.pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, pelvis)
        assert self.site >= 0 and self.pelvis >= 0, "site/pelvis not found"
        self.lo = np.array([m.jnt_range[j][0] for j in self.jid]) + self.cfg.joint_margin
        self.hi = np.array([m.jnt_range[j][1] for j in self.jid]) - self.cfg.joint_margin
        self.home_q = np.clip(np.asarray(home_q, float), self.lo, self.hi)
        self.last_sigma_min = np.inf

    def _fk(self, qpos_full, q_arm):
        self.kd.qpos[:] = qpos_full
        self.kd.qpos[self.qadr] = q_arm
        mujoco.mj_kinematics(self.m, self.kd)
        mujoco.mj_comPos(self.m, self.kd)

    def site_pose_local(self, qpos_full, q_arm):
        """Site pose in the pelvis frame for the given configuration."""
        self._fk(qpos_full, q_arm)
        Rp = self.kd.xmat[self.pelvis].reshape(3, 3)
        pos = Rp.T @ (self.kd.site_xpos[self.site] - self.kd.xpos[self.pelvis])
        R = Rp.T @ self.kd.site_xmat[self.site].reshape(3, 3)
        return pos, R

    def step(self, qpos_full, q_cmd, target_pos_local, target_R_local=None, dt=0.01):
        """One IK step. Returns the new commanded arm joint vector (bounded, within limits)."""
        c = self.cfg
        pos, R = self.site_pose_local(qpos_full, q_cmd)
        e_pos = np.asarray(target_pos_local, float) - pos
        n = np.linalg.norm(e_pos)
        if n > c.max_pos_error:
            e_pos *= c.max_pos_error / n
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.kd, jp, jr, self.site)
        Rp = self.kd.xmat[self.pelvis].reshape(3, 3)
        Jp = Rp.T @ jp[:, self.dadr]                 # express in pelvis frame
        if target_R_local is not None:
            Jr = Rp.T @ jr[:, self.dadr]
            Rt = np.asarray(target_R_local, float)
            e_rot = 0.5 * sum(np.cross(R[:, i], Rt[:, i]) for i in range(3))
            w = c.rot_weight
            J = np.vstack([Jp, w * Jr]); e = np.concatenate([e_pos, w * e_rot])
        else:
            J, e = Jp, e_pos
        sig = np.linalg.svd(J, compute_uv=False)
        self.last_sigma_min = float(sig[-1])
        lam2 = c.damping ** 2
        if sig[-1] < c.singularity_eps:               # adaptive damping close to a singularity
            lam2 += (1 - (sig[-1] / c.singularity_eps) ** 2) * c.max_damping ** 2
        Jpinv = J.T @ np.linalg.inv(J @ J.T + lam2 * np.eye(J.shape[0]))
        dq = Jpinv @ e
        dq += (np.eye(len(q_cmd)) - Jpinv @ J) @ (c.nullspace_gain * (self.home_q - q_cmd) * dt)   # pull toward home
        vmax = c.max_joint_velocity * dt
        dq = np.clip(dq, -vmax, vmax)
        out = np.clip(np.asarray(q_cmd, float) + dq, self.lo, self.hi)
        return out if np.isfinite(out).all() else np.asarray(q_cmd, float)
