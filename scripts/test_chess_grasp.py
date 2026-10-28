"""Scripted physical grasp of a pawn-sized cylinder with the G1 right Dex3-1 hand.

Sequence (open-loop position targets; no teleporting, welds, or disabled collisions):
  settle (open hand) -> close fingers (ramp) -> pause -> lift by moving the arm (ramp) -> hold.
Also runs a negative control (identical arm lift, fingers left open) and repeats the grasp from
fresh resets to check determinism. Writes logs/grasp_test.json, a CSV trace and PNG snapshots.

Run with MUJOCO_GL=glfw (+ the d3d12 env vars from the README) for snapshots, or --no-render.
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grasp_scene as gs  # noqa: E402
from render_util import save_png  # noqa: E402

LOGS = gs.ROOT / "logs"

# Closing targets for [thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1] (rad), found by
# an offline search around a pawn pose in the hand frame (see README). All inside joint/ctrl ranges.
CONFIG = dict(
    finger_close=[-0.226, -0.902, -0.924, 0.154, 0.724, 1.095, 1.01],
    t_settle=0.5, t_close=1.5, t_pause=0.5, t_lift=1.5, t_hold=1.0,
    lift_height=0.05,           # commanded rise of the grasp site (m)
    success_min_rise=0.03,      # pawn must be at least this far above its rest height throughout the hold (m)
    success_max_slip=0.01,      # pawn pose in the wrist frame may drift at most this much during the hold (m)
    accel_warn=500.0,           # m/s^2, reported if exceeded
)
nm = lambda m, t, i: mujoco.mj_id2name(m, t, i)


def trial_checks(met):
    """Per-trial pass/fail checks (the success definition shared with scripts/map_grasp_region.py)."""
    c = CONFIG
    return dict(
        finite_state=not met["nonfinite_state"],
        no_initial_penetration=len(met["init_pawn_penetrations"]) == 0,
        no_joint_limit_violation=met["joint_limit_violation_steps"] == 0,
        robot_upright=met["min_pelvis_upright"] > 0.95,
        no_equality_constraints=True,
        at_least_two_hand_links_touch_pawn=len(met["finger_links_in_contact_at_end"]) >= 2,
        pawn_risen=met["rise"] >= c["success_min_rise"],
        held_for_full_interval=met["hold_time_above_threshold_s"] >= c["t_hold"] - 1e-9,
        no_table_or_floor_contact_in_hold=met["hold_samples_with_table_or_floor_contact"] == 0,
        no_slip_in_hand=met["hold_slip_m"] <= c["success_max_slip"],
    )


def run_sequence(params, close=True, snapshots=None, min_clearance=None):
    """Fresh build + reset + full sequence. If min_clearance (m) is given and the pawn starts closer than that
    to any hand collision geom, returns early with met['invalid_initial_clearance']=True (no closing is run)."""
    c = CONFIG
    m, d, info = gs.build_scene(params)
    assert m.neq == 0, "model must have no equality constraints (no welds)"
    gs.reset(m, d, info)
    dt = m.opt.timestep
    OBJ = mujoco.mjtObj
    bid = lambda n: mujoco.mj_name2id(m, OBJ.mjOBJ_BODY, n)
    jid = lambda n: mujoco.mj_name2id(m, OBJ.mjOBJ_JOINT, n)
    aid = lambda n: mujoco.mj_name2id(m, OBJ.mjOBJ_ACTUATOR, n)
    pawn, wrist = bid("pawn"), bid("right_wrist_yaw_link")
    pq = m.jnt_qposadr[jid("pawn_free")]
    pv = m.jnt_dofadr[jid("pawn_free")]
    qadr = lambda n: m.jnt_qposadr[jid(n)]
    f_act, arm_act = [aid(n) for n in gs.FINGERS], [aid(n) for n in gs.ARM]
    site = mujoco.mj_name2id(m, OBJ.mjOBJ_SITE, "grasp_site")

    def body_of(g): return nm(m, OBJ.mjOBJ_BODY, m.geom_bodyid[g])

    # --- initial-state checks: no pawn penetration, report clearance to the hand ---
    init_pen = [(body_of(x.geom1), body_of(x.geom2), float(x.dist)) for x in d.contact[:d.ncon]
                if pawn in (m.geom_bodyid[x.geom1], m.geom_bodyid[x.geom2]) and x.dist < -1e-4]
    pawn_geom = mujoco.mj_name2id(m, OBJ.mjOBJ_GEOM, "pawn_geom")
    clearance = {}
    for g in range(m.ngeom):
        b = body_of(g)
        if b.startswith("right_hand") or b == "right_wrist_yaw_link":
            if m.geom_contype[g] or m.geom_conaffinity[g]:
                dist = mujoco.mj_geomDistance(m, d, pawn_geom, g, 0.5, None)
                clearance[b] = min(clearance.get(b, 9.0), float(dist))

    if min_clearance is not None and min(clearance.values()) < min_clearance:
        return dict(invalid_initial_clearance=True, init_clearance_to_hand_m=clearance,
                    init_pawn_penetrations=init_pen, pawn_initial_world=d.qpos[pq:pq + 3].tolist()), [], info

    key2, _ = gs.solve_pregrasp({**info["params"], "grasp_world_z": info["params"]["grasp_world_z"] + c["lift_height"]})
    arm0 = info["ctrl0"][arm_act].copy()
    arm1 = np.array([key2[qadr(n)] for n in gs.ARM])
    open_t = np.array(info["params"]["open_pose"], float)
    close_t = np.array(c["finger_close"]) if close else open_t
    ctrl = info["ctrl0"].copy()
    rows = []
    st = dict(peak_z=-9.0, max_acc=0.0, nonfinite=False, limit_viol=0, min_up=1.0, hold_support=0, prev_v=d.qvel[pv:pv + 3].copy())
    hinge = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    jlo, jhi = m.jnt_range[hinge, 0], m.jnt_range[hinge, 1]

    def pawn_in_wrist():
        return d.xmat[wrist].reshape(3, 3).T @ (d.xpos[pawn] - d.xpos[wrist])

    def contacts():
        out = []
        for i in range(d.ncon):
            x = d.contact[i]
            b1, b2 = m.geom_bodyid[x.geom1], m.geom_bodyid[x.geom2]
            if pawn in (b1, b2):
                f = np.zeros(6); mujoco.mj_contactForce(m, d, i, f)
                out.append(dict(body=nm(m, OBJ.mjOBJ_BODY, b2 if b1 == pawn else b1), dist=float(x.dist),
                                normal_force=float(f[0])))
        return out

    def run(name, T, fn):
        n = int(round(T / dt))
        for k in range(n):
            fn((k + 1) / n)
            mujoco.mj_step(m, d)
            if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
                st["nonfinite"] = True
                return
            st["peak_z"] = max(st["peak_z"], float(d.qpos[pq + 2]))
            v = d.qvel[pv:pv + 3]
            st["max_acc"] = max(st["max_acc"], float(np.linalg.norm(v - st["prev_v"]) / dt)); st["prev_v"] = v.copy()
            st["min_up"] = min(st["min_up"], float(d.xmat[1].reshape(3, 3)[2, 2]))
            q = d.qpos[m.jnt_qposadr[hinge]]
            st["limit_viol"] += int(np.any((q < jlo - 1e-2) | (q > jhi + 1e-2)))
            if k % 10 == 0:
                rows.append([d.time, name, *d.ctrl[f_act], *d.qpos[[qadr(n) for n in gs.FINGERS]],
                             *d.ctrl[arm_act], *d.qpos[pq:pq + 3]])
                if name == "hold" and any(x["body"] in ("table", "world") for x in contacts()):
                    st["hold_support"] += 1

    def keep(s): d.ctrl[:] = ctrl
    def close_f(s): ctrl[f_act] = open_t + (close_t - open_t) * s; d.ctrl[:] = ctrl
    def lift_f(s): ctrl[arm_act] = arm0 + (arm1 - arm0) * s; d.ctrl[:] = ctrl
    def hold_f(s): ctrl[arm_act] = arm1; d.ctrl[:] = ctrl

    def snap(tag):
        if snapshots:
            save_png(m, d, snapshots[tag], lookat=d.site_xpos[site], distance=0.45, azimuth=150, elevation=-20)

    mujoco.mj_forward(m, d)
    pawn_init = d.qpos[pq:pq + 3].copy()
    snap("init")
    run("settle_open", c["t_settle"], keep)
    z_rest = float(d.qpos[pq + 2])
    run("close", c["t_close"], close_f)
    run("pause", c["t_pause"], keep)
    contacts_closed = contacts()
    hand_q_closed = {n: float(d.qpos[qadr(n)]) for n in gs.FINGERS}
    snap("closed")
    run("lift", c["t_lift"], lift_f)
    z_after_lift = float(d.qpos[pq + 2]); p_hold0 = pawn_in_wrist()
    # hold: arm fixed at the lifted pose; track how long the pawn stays above the success height
    n_hold = int(round(c["t_hold"] / dt)); above = 0
    def hold_track(s):
        nonlocal above
        hold_f(s)
    t_hold_start = d.time
    for blk in range(int(round(c["t_hold"] / 0.01))):
        run("hold", 0.01, hold_f)
        if d.qpos[pq + 2] - z_rest >= c["success_min_rise"]:
            above += 1
    hold_above_s = above * 0.01
    p_hold1 = pawn_in_wrist()
    z_end = float(d.qpos[pq + 2])
    contacts_hold = contacts()
    snap("final")

    finger_links = sorted({x["body"] for x in contacts_hold if x["body"].startswith("right_hand_")})
    slip = float(np.linalg.norm(p_hold1 - p_hold0))
    pawn_xy_end = d.qpos[pq:pq + 2]
    met = dict(
        z_rest=z_rest, z_after_lift=z_after_lift, z_end=z_end, rise=z_end - z_rest,
        commanded_lift=c["lift_height"], hold_time_above_threshold_s=hold_above_s,
        pawn_in_wrist_frame_start_hold=p_hold0.tolist(), pawn_in_wrist_frame_end_hold=p_hold1.tolist(), hold_slip_m=slip,
        pawn_final_world=d.qpos[pq:pq + 3].tolist(),
        contacts_after_close=contacts_closed, contacts_end_of_hold=contacts_hold,
        finger_links_in_contact_at_end=finger_links,
        palm_in_contact_at_end=any(x["body"] == "right_wrist_yaw_link" for x in contacts_hold),
        hold_samples_with_table_or_floor_contact=st["hold_support"],
        hand_joint_q_after_close=hand_q_closed,
        hand_joint_q_end={n: float(d.qpos[qadr(n)]) for n in gs.FINGERS},
        max_object_accel=st["max_acc"], object_accel_above_warn=st["max_acc"] > c["accel_warn"],
        joint_limit_violation_steps=st["limit_viol"], nonfinite_state=st["nonfinite"],
        min_pelvis_upright=st["min_up"], final_pelvis_z=float(d.qpos[2]),
        init_pawn_penetrations=init_pen, init_clearance_to_hand_m=clearance,
        sim_time=float(d.time),
        invalid_initial_clearance=False, peak_z=st["peak_z"], peak_rise=st["peak_z"] - z_rest,
        pawn_initial_world=pawn_init.tolist(), closure_hand_contacts=sorted(
            {x["body"] for x in contacts_closed if x["body"].startswith("right_")}),
    )
    return met, rows, info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=3, help="fresh-reset repeats for determinism check")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--out", type=Path, default=LOGS / "grasp_test.json")
    a = ap.parse_args()
    c = CONFIG
    snaps = None if a.no_render else {k: LOGS / f"grasp_{k}.png" for k in ("init", "closed", "final")}

    print("== grasp run (fresh build + reset) ==")
    met, rows, info = run_sequence({}, close=True, snapshots=snaps)
    print(f"pawn rest z={met['z_rest']:.4f} -> end z={met['z_end']:.4f} (rise {met['rise']*1000:.1f} mm, commanded {c['lift_height']*1000:.0f} mm)")
    print(f"hold above {c['success_min_rise']*1000:.0f} mm for {met['hold_time_above_threshold_s']:.2f}/{c['t_hold']:.2f} s; slip in wrist frame {met['hold_slip_m']*1000:.2f} mm")
    print("contacts at end of hold:", [(x['body'], round(x['normal_force'], 3)) for x in met["contacts_end_of_hold"]])

    print("== negative control: same arm lift, fingers open ==")
    ctl, _, _ = run_sequence({}, close=False)
    print(f"control rise {ctl['rise']*1000:.1f} mm, final pawn z={ctl['z_end']:.4f}, contacts {[x['body'] for x in ctl['contacts_end_of_hold']]}")

    print(f"== determinism: {a.repeats} fresh runs ==")
    finals = [met["pawn_final_world"]]
    for _ in range(a.repeats - 1):
        r, _, _ = run_sequence({}, close=True)
        finals.append(r["pawn_final_world"])
    spread = float(np.abs(np.array(finals) - np.array(finals[0])).max())
    print(f"max spread of final pawn position across runs: {spread:.3e} m")

    checks = dict(
        **trial_checks(met),
        control_does_not_lift=ctl["rise"] < 0.01,
        reproducible=spread < 1e-6,
    )
    success = all(checks.values())
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("OVERALL:", "SUCCESS" if success else "FAILED")

    LOGS.mkdir(exist_ok=True)
    out = dict(success=success, checks=checks, config=c, mujoco=mujoco.__version__,
               scene_params={k: (list(v) if isinstance(v, tuple) else v) for k, v in info["params"].items()},
               arm_ik=info["ik"], pawn_initial_world=info["obj_pos"].tolist(), grasp_site_in_wrist_frame=info["site_hand"],
               grasp=met, negative_control=ctl, determinism=dict(runs=a.repeats, max_spread_m=spread, finals=finals),
               notes="Position actuators/gains unchanged from Menagerie; joint actuatorfrcrange (<=1.4 N*m fingers) limits torque.")
    a.out.write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    hdr = ["t", "phase", *[f"ctrl_{n}" for n in gs.FINGERS], *[f"q_{n}" for n in gs.FINGERS],
           *[f"armctrl_{n}" for n in gs.ARM], "pawn_x", "pawn_y", "pawn_z"]
    with open(LOGS / "grasp_trace.csv", "w") as f:
        f.write(",".join(hdr) + "\n")
        for r in rows:
            f.write(",".join(str(v) if isinstance(v, str) else f"{v:.5f}" for v in r) + "\n")
    print(f"wrote {a.out}")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
