"""Derive Inspire (RH56DFQ) gesture poses from the actual model (run once; results are stored in livehand/config.toml).

Actuated joints per hand: thumb_proximal_yaw, thumb_proximal_pitch, index/middle/ring/pinky_proximal. The other 6
finger joints follow through the URDF mimic equalities, which forward kinematics does not enforce, so `couple()` sets
them from eq_data (q_follower = c0 + c1 * q_leader) before every evaluation.
Fingertip point = mean of the 3% collision-mesh vertices farthest from the link origin (last link of each digit).
  pinch: exhaustive grid over (thumb yaw, thumb pitch, index) -> thumb-tip/index-tip gap closest to GAP, no new
         hand self-penetration; other fingers open.
  fist:  fingers at max_fraction of range; grid over the thumb -> thumb tip closest to the middle finger without
         penetrating it.
Both hands are checked (same joint conventions; the geometry is mirrored).
  --home: re-derive sim.INSPIRE_HOME_RIGHT (arm joints for the home palm position).
"""
import itertools
import sys

import mujoco
import numpy as np

from livehand import config as config_mod
from livehand.arm_ik import ArmIK
from livehand.sim import ARM_JOINTS, INSPIRE_HOME_PALM, MODELS, build_sim

GAP = 0.012            # m, pinch fingertip gap: nearly touching, never overlapping
PEN = -0.002           # m, contacts deeper than this count as penetration


def tip_local(m, body):
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == body and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and (m.geom_contype[g] or m.geom_conaffinity[g]):
            a, n = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
            R = np.zeros(9); mujoco.mju_quat2Mat(R, m.geom_quat[g])
            pts.append(m.mesh_vert[a:a + n] @ R.reshape(3, 3).T + m.geom_pos[g])
    v = np.vstack(pts)
    k = max(3, int(0.03 * len(v)))
    return v[np.argsort(np.linalg.norm(v, axis=1))[-k:]].mean(0)


class Hand:
    def __init__(self, m, d, side, cfg):
        self.m, self.d, self.S = m, d, side[0].upper()
        prof = MODELS["inspire"]
        self.q0 = d.qpos.copy()
        J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.names = [prof["hand_joint"](side, n) for n in prof["hand"]]
        self.qa = [m.jnt_qposadr[J(n)] for n in self.names]
        frac = cfg.hand.max_fraction
        r = np.array([m.jnt_range[J(n)] for n in self.names])
        self.lo, self.hi = r[:, 0].clip(min=0.0), r[:, 1] * frac       # stay on the reachable (>= 0) side
        self.eq = [(m.jnt_qposadr[m.eq_obj1id[e]], m.jnt_qposadr[m.eq_obj2id[e]], m.eq_data[e, 0], m.eq_data[e, 1])
                   for e in range(m.neq) if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT
                   and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.eq_obj1id[e]).startswith(self.S + "_")]
        # a coupled joint's own limit caps its leader (thumb pitch: 0.8/1.6 = 1.2/2.4 = 0.5 rad)
        for f, l, c0, c1 in self.eq:
            jf = next(j for j in range(m.njnt) if m.jnt_qposadr[j] == f)
            k = self.qa.index(l)
            if c1 > 0:
                self.hi[k] = min(self.hi[k], (m.jnt_range[jf][1] - c0) / c1 * frac)
        self.tips = {k: (B(f"{self.S}_{b}"), None) for k, b in
                     (("thumb", "thumb_distal"), ("index", "index_intermediate"), ("middle", "middle_intermediate"))}
        self.tips = {k: (b, tip_local(m, b)) for k, (b, _) in self.tips.items()}
        self.base_pen = self.penetrations(np.zeros(6))

    def set(self, q):
        self.d.qpos[:] = self.q0
        self.d.qpos[self.qa] = q
        for f, l, c0, c1 in self.eq:
            self.d.qpos[f] = c0 + c1 * self.d.qpos[l]

    def tip(self, q, k):
        self.set(q); mujoco.mj_kinematics(self.m, self.d)
        b, p = self.tips[k]
        return self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ p

    def penetrations(self, q):
        self.set(q); mujoco.mj_forward(self.m, self.d)
        out = set()
        for x in self.d.contact[:self.d.ncon]:
            b1, b2 = (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[g]) for g in (x.geom1, x.geom2))
            if (b1.startswith(self.S + "_") or b2.startswith(self.S + "_")) and x.dist < PEN:
                out.add(tuple(sorted((b1, b2))))
        return out

    def new_pen(self, q):
        return self.penetrations(q) - self.base_pen


def pinch(h):
    best = []
    for yaw, pitch, idx in itertools.product(np.linspace(h.lo[0], h.hi[0], 27), np.linspace(h.lo[1], h.hi[1], 19),
                                             np.linspace(h.lo[2], h.hi[2], 35)):
        q = np.array([yaw, pitch, idx, 0, 0, 0])
        best.append((abs(np.linalg.norm(h.tip(q, "thumb") - h.tip(q, "index")) - GAP), q))
    best.sort(key=lambda x: x[0])
    for c, q in best[:300]:                       # nearest-gap candidates first; first one without new penetration
        if not h.new_pen(q):
            return q, np.linalg.norm(h.tip(q, "thumb") - h.tip(q, "index"))
    raise RuntimeError("no penetration-free pinch found")


def fist(h):
    best = []
    for yaw, pitch in itertools.product(np.linspace(h.lo[0], h.hi[0], 27), np.linspace(h.lo[1], h.hi[1], 19)):
        q = np.array([yaw, pitch, *h.hi[2:]])
        best.append((np.linalg.norm(h.tip(q, "thumb") - h.tip(q, "middle")), q))
    best.sort(key=lambda x: x[0])
    for dist, q in best:
        if dist > 0.008 and not h.new_pen(q):
            return q, dist
    raise RuntimeError("no penetration-free fist found")


def solve_home(m, d, cfg, palm=INSPIRE_HOME_PALM):
    """IK for the right palm site at `palm` (pelvis frame), fingers forward (-y of the hand base -> +x), palm toward
    the midline (-x of the hand base -> +y). Returns the 7 right-arm joint values."""
    jn = [f"right_{j}_joint" for j in ARM_JOINTS]
    xh, yh = np.array([0, -1.0, 0]), np.array([-1.0, 0, 0])
    ik = ArmIK(m, jn, "R_palm", np.array([-0.6, -0.2, 0, 1.0, 0, 0, 0]), cfg)
    q, tgt = ik.home_q.copy(), np.array(palm)
    for _ in range(3000):
        q = ik.step(d.qpos, q, tgt, np.stack([xh, yh, np.cross(xh, yh)], 1), 0.01)
    pos, _ = ik.site_pose_local(d.qpos, q)
    return q, np.linalg.norm(pos - tgt)


def main():
    cfg = config_mod.load(overrides={"robot": {"model": "inspire"}})
    m, d, info = build_sim(cfg)
    if "--home" in sys.argv:
        q, err = solve_home(m, d, cfg)
        print(f"INSPIRE_HOME_RIGHT = {tuple(np.round(q, 4).tolist())}   # palm error {1000 * err:.1f} mm")
        return
    for side in ("right", "left"):
        h = Hand(m, d, side, cfg)
        print(f"{side}: actuators {h.names}")
        print(f"  usable ranges lo {np.round(h.lo, 3).tolist()} hi {np.round(h.hi, 3).tolist()}")
        print(f"  open-pose self-penetrations (ignored baseline): {sorted(h.base_pen)}")
        qp, gap = pinch(h)
        print(f"  PINCH {np.round(qp, 3).tolist()}  tip gap {1000 * gap:.1f} mm")
        qf, dist = fist(h)
        print(f"  FIST  {np.round(qf, 3).tolist()}  thumb tip -> middle tip {1000 * dist:.1f} mm, new penetrations {sorted(h.new_pen(qf))}")


if __name__ == "__main__":
    main()
