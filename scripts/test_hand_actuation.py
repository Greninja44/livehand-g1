"""Individually actuate each of the seven right-hand (or left-hand) Dex3-1 actuators.

For every actuator and each allowed direction: reset to the `stand` keyframe, settle, command a
conservative target (50% of the way to the joint limit), hold, and log the response. Also checks
that the other hand joints do not move (independence) and that the robot stays stable.
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_util import save_png  # noqa: E402
import grasp_scene as gs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "models" / "mujoco_menagerie" / "unitree_g1" / "scene_with_hands.xml"
FRACTION = 0.5          # fraction of the joint range side to command
SETTLE_S, HOLD_S = 0.5, 1.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["left", "right"], default="right")
    ap.add_argument("--screenshots", action="store_true", help="save PNG per test (needs MUJOCO_GL=glfw)")
    ap.add_argument("--pose", choices=["keyframe", "pregrasp"], default="keyframe",
                    help="keyframe: Menagerie `stand` (arm at side, hand against the hip). "
                         "pregrasp: same legs/torso, right arm forward from IK (hand clear of the body), all finger joints 0")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    a.out = a.out or ROOT / "logs" / f"hand_actuation_{a.pose}.json"

    if a.pose == "keyframe":
        m = mujoco.MjModel.from_xml_path(str(SCENE))
        d = mujoco.MjData(m)
        key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
        reset = lambda: mujoco.mj_resetDataKeyframe(m, d, key)
    else:
        m, d, info = gs.build_scene({"open_pose": [0.0] * 7}, with_object=False)
        reset = lambda: gs.reset(m, d, info)
    prefix = f"{a.side}_hand_"
    acts = [i for i in range(m.nu) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i).startswith(prefix)]
    assert len(acts) == 7, f"expected 7 {a.side}-hand actuators, found {len(acts)}"
    all_hand = [i for i in range(m.nu) if "_hand_" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)]
    jadr = lambda i: m.jnt_qposadr[m.actuator_trnid[i, 0]]
    results, failures = [], []

    def run(ctrl, seconds):
        worst = 0.0
        for _ in range(int(seconds / m.opt.timestep)):
            d.ctrl[:] = ctrl
            mujoco.mj_step(m, d)
            if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
                return None
            worst = max(worst, np.abs(d.qvel).max())
        return worst

    for i in acts:
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        jid = m.actuator_trnid[i, 0]
        lo, hi = m.jnt_range[jid]
        frc = m.jnt_actfrcrange[jid]
        print(f"\n{name}: joint range [{lo:+.3f},{hi:+.3f}] ctrlrange {m.actuator_ctrlrange[i]} "
              f"jnt_actuatorfrcrange {frc} kp={m.actuator_gainprm[i,0]:.0f}")
        for sign, limit in (("+", hi), ("-", lo)):
            if abs(limit) < 1e-6:
                continue
            target = float(np.clip(FRACTION * limit, *m.actuator_ctrlrange[i]))
            reset()
            ctrl0 = d.ctrl.copy()
            if run(ctrl0, SETTLE_S) is None:
                failures.append((name, sign, "non-finite during settle")); continue
            q_before = d.qpos[[jadr(k) for k in all_hand]].copy()
            z0 = d.qpos[2]
            ctrl = ctrl0.copy(); ctrl[i] = target
            vmax = run(ctrl, HOLD_S)
            if vmax is None:
                failures.append((name, sign, "non-finite during hold")); continue
            q = d.qpos[jadr(i)]
            q_after = d.qpos[[jadr(k) for k in all_hand]]
            others = [abs(q_after[n] - q_before[n]) for n, k in enumerate(all_hand) if k != i]
            upright = d.xmat[1].reshape(3, 3)[2, 2]
            rec = dict(actuator=name, direction=sign, q_start=float(q_before[all_hand.index(i)]), target=target,
                       q_final=float(q), tracking_error=float(abs(q - target)),
                       in_limits=bool(lo - 1e-3 <= q <= hi + 1e-3),
                       max_other_hand_joint_motion=float(max(others)),
                       pelvis_dz=float(d.qpos[2] - z0), upright=float(upright), max_qvel=float(vmax),
                       actuator_force=float(d.actuator_force[i]))
            rec["responded"] = bool(rec["tracking_error"] < 0.1 * abs(target) + 0.02)
            rec["ok"] = bool(rec["responded"] and rec["in_limits"] and rec["max_other_hand_joint_motion"] < 0.05
                         and abs(rec["pelvis_dz"]) < 0.02 and upright > 0.98)
            if a.screenshots:
                p = ROOT / "logs" / f"hand_actuation_{a.pose}" / f"{name}_{'pos' if sign == '+' else 'neg'}.png"
                # camera on the hand
                w = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{a.side}_wrist_yaw_link")
                save_png(m, d, p, lookat=d.xpos[w], distance=0.5, azimuth=90, elevation=-20)
            results.append(rec)
            print(f"  {sign} target={target:+.3f} final={q:+.3f} err={rec['tracking_error']:.3f} "
                  f"others_moved={rec['max_other_hand_joint_motion']:.3f} dz={rec['pelvis_dz']:+.4f} "
                  f"force={rec['actuator_force']:+.2f} -> {'OK' if rec['ok'] else 'FAIL'}")
            if not rec["ok"]:
                failures.append((name, sign, "criteria not met"))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(side=a.side, pose=a.pose, model=str(SCENE), mujoco=mujoco.__version__,
                                     fraction=FRACTION, results=results, failures=failures), indent=2))
    n_ok = sum(r["ok"] for r in results)
    print(f"\n{n_ok}/{len(results)} tests OK; failures: {failures or 'none'}  -> {a.out}")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
