"""Individually actuate each of the 6 active Inspire-hand joints (right by default) and check the response,
that the coupled (mimic) joints follow their constraint, that other active joints stay put, and balance holds."""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inspire_scene as sc  # noqa: E402
from render_util import save_png  # noqa: E402

MIMIC = {"thumb_proximal_pitch": [("thumb_intermediate", 1.6), ("thumb_distal", 2.4)],
         "index_proximal": [("index_intermediate", 1.0)], "middle_proximal": [("middle_intermediate", 1.0)],
         "ring_proximal": [("ring_intermediate", 1.0)], "pinky_proximal": [("pinky_intermediate", 1.0)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["R", "L"], default="R")
    ap.add_argument("--screenshots", action="store_true")
    ap.add_argument("--out", type=Path, default=sc.ROOT / "logs" / "inspire_actuation.json")
    a = ap.parse_args()
    m, d, info = sc.build()
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    qa = lambda n: m.jnt_qposadr[jid(n)]
    names = [f"{a.side}_{h}_joint" for h in sc.ACTIVE_HAND]
    results, failures = [], []

    def run(T):
        worst = 0.0
        for _ in range(int(T / m.opt.timestep)):
            mujoco.mj_step(m, d)
            if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
                return None
            worst = max(worst, np.abs(d.qvel).max())
        return worst

    for n in names:
        lo, hi = m.jnt_range[jid(n)]
        print(f"\n{n}: range [{lo:+.3f},{hi:+.3f}] ctrlrange {m.actuator_ctrlrange[aid(n)]} "
              f"torque limit {m.jnt_actfrcrange[jid(n)]}")
        for sign, lim in (("+", hi), ("-", lo)):
            target = 0.5 * lim
            sc.reset(m, d, info)
            if run(0.5) is None:
                failures.append((n, sign, "non-finite")); continue
            before = {k: d.qpos[qa(k)] for k in names}
            z0 = d.qpos[2]
            d.ctrl[:] = info["ctrl0"]; d.ctrl[aid(n)] = target
            for _ in range(int(1.5 / m.opt.timestep)):
                d.ctrl[aid(n)] = target
                mujoco.mj_step(m, d)
            q = d.qpos[qa(n)]
            base = n[2:-6]
            mim = [(f"{a.side}_{c}_joint", k) for c, k in MIMIC.get(base, [])]
            mim_err = max([abs(d.qpos[qa(c)] - k * q) for c, k in mim], default=0.0)
            others = max([abs(d.qpos[qa(k)] - before[k]) for k in names if k != n], default=0.0)
            up = d.xmat[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")].reshape(3, 3)[2, 2]
            rec = dict(actuator=n, direction=sign, target=target, q_final=float(q), tracking_error=float(abs(q - target)),
                       mimic_follow_error=float(mim_err), max_other_active_joint_motion=float(others),
                       pelvis_dz=float(d.qpos[2] - z0), upright=float(up))
            rec["ok"] = bool(rec["tracking_error"] < 0.1 * abs(target) + 0.02 and mim_err < 0.05 and others < 0.05
                             and abs(rec["pelvis_dz"]) < 0.02 and up > 0.98)
            results.append(rec)
            if a.screenshots:
                w = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{'right' if a.side == 'R' else 'left'}_wrist_yaw_link")
                save_png(m, d, sc.ROOT / "logs" / "inspire_actuation" / f"{n}_{'pos' if sign == '+' else 'neg'}.png",
                         lookat=d.xpos[w], distance=0.5, azimuth=110, elevation=-20)
            print(f"  {sign} target={target:+.3f} final={q:+.3f} err={rec['tracking_error']:.3f} mimic_err={mim_err:.3f} "
                  f"others={others:.3f} dz={rec['pelvis_dz']:+.4f} -> {'OK' if rec['ok'] else 'FAIL'}")
            if not rec["ok"]:
                failures.append((n, sign, "criteria not met"))
    a.out.parent.mkdir(exist_ok=True)
    a.out.write_text(json.dumps(dict(side=a.side, results=results, failures=failures, mujoco=mujoco.__version__), indent=2))
    print(f"\n{sum(r['ok'] for r in results)}/{len(results)} OK; failures: {failures or 'none'} -> {a.out}")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
