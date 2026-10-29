"""Derive Dex3-1 gesture poses from the actual model (run once; results are stored in livehand/config.toml).

Fingertip points = mean of the farthest 3% of each distal link's collision-mesh vertices along the link's long axis.
Pinch pose = random search + local refinement of (thumb_0, thumb_1, thumb_2, index_0, index_1) minimising the
thumb-tip/index-tip distance, subject to joint ranges (90%) and no new hand self-penetration.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, for robots/
from robots import grasp_scene as gs  # noqa: E402

SIDE = "right"
H = [f"{SIDE}_hand_{n}_joint" for n in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")]


def tip_local(m, body, axis):
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == body and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and (m.geom_contype[g] or m.geom_conaffinity[g]):
            a, n = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
            R = np.zeros(9); mujoco.mju_quat2Mat(R, m.geom_quat[g])
            pts.append(m.mesh_vert[a:a + n] @ R.reshape(3, 3).T + m.geom_pos[g])   # mesh frame -> body frame
    v = np.vstack(pts)
    k = max(3, int(0.03 * len(v)))
    return v[np.argsort(v[:, axis])[-k:]].mean(0)


def main():
    m, d, info = gs.build_scene({}, with_object=False)
    B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    tips = {"thumb": (B(f"{SIDE}_hand_thumb_2_link"), 1), "index": (B(f"{SIDE}_hand_index_1_link"), 0),
            "middle": (B(f"{SIDE}_hand_middle_1_link"), 0)}
    loc = {k: tip_local(m, b, ax) for k, (b, ax) in tips.items()}
    print("tip points in link frames:", {k: np.round(v, 4).tolist() for k, v in loc.items()})
    qa = [m.jnt_qposadr[J(n)] for n in H]
    rng_ = np.array([m.jnt_range[J(n)] for n in H]) * 0.9

    def tip_pos(q):
        d.qpos[:] = info["qpos0"]; d.qpos[qa] = q
        mujoco.mj_kinematics(m, d)
        return {k: d.xpos[b] + d.xmat[b].reshape(3, 3) @ loc[k] for k, (b, _) in tips.items()}

    def self_pen(q):
        d.qpos[:] = info["qpos0"]; d.qpos[qa] = q
        mujoco.mj_forward(m, d)
        out = set()
        for x in d.contact[:d.ncon]:
            b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[x.geom1]); b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[x.geom2])
            if "right_hand" in b1 + b2 or "right_wrist" in b1 + b2:
                if x.dist < -0.002: out.add((b1, b2))
        return out

    base_pen = self_pen(np.array(gs.DEFAULTS["open_pose"], float))
    print("open pose self-penetrations:", base_pen)
    r = np.random.default_rng(0)
    lo, hi = rng_[:5, 0], rng_[:5, 1]
    GAP = 0.012  # target tip gap (m): pinch pose should nearly touch, never overlap
    cost = lambda q: abs(np.linalg.norm(tip_pos(q)["thumb"] - tip_pos(q)["index"]) - GAP) + (1.0 if self_pen(q) - base_pen else 0.0)
    best, bq = 9, None
    for _ in range(6000):
        q = np.array(gs.DEFAULTS["open_pose"], float); q[:5] = r.uniform(lo, hi)
        c = cost(q)
        if c < best: best, bq = c, q.copy()
    for s_ in (0.2, 0.1, 0.05, 0.02, 0.01):
        for _ in range(1500):
            q = bq.copy(); q[:5] = np.clip(q[:5] + r.normal(0, s_, 5), lo, hi)
            c = cost(q) + 0.002 * np.linalg.norm(q[:5] - bq[:5])
            if c < best: best, bq = c, q.copy()
    t = tip_pos(bq); pen = self_pen(bq) - base_pen
    print("PINCH thumb_0,thumb_1,thumb_2,index_0,index_1 =", np.round(bq[:5], 3).tolist(),
          "tip distance mm:", round(1000 * np.linalg.norm(t['thumb'] - t['index']), 1), "new self-penetrations:", pen)
    # fist candidate
    fist = np.array([0.0, -0.9, -1.2, 1.4, 1.5, 1.4, 1.5])
    fist = np.clip(fist, rng_[:, 0], rng_[:, 1])
    print("FIST candidate:", np.round(fist, 3).tolist(), "new self-penetrations:", self_pen(fist) - base_pen)


if __name__ == "__main__":
    main()
