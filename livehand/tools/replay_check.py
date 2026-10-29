"""Replay a recorded clip through the FULL pipeline, headless: MediaPipe -> calibration -> retargeting -> IK -> MuJoCo.

    python -m livehand.tools.camera_check --seconds 20 --record logs/hand_clip/hand_clip.mp4   # record (needs webcam)
    python -m livehand.tools.replay_check logs/hand_clip/hand_clip.mp4                          # replay + verify

Scripted like a user session: once the hand has been tracked for `--settle` s, capture neutral, then enable mirroring
and keep going to the end of the clip. Logs, per control tick, the commanded Cartesian offset, the palm position the
simulated arm actually reached, the gesture cues and the hand commands; writes <clip>_<robot>_replay.json and a plot.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

from livehand import config as config_mod
from livehand.calibration import Calibration
from livehand.safety import Mode
from livehand.sim import MODELS, Controller, build_sim
from livehand.tracking import VisionSystem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--settle", type=float, default=1.0, help="s of continuous tracking before the neutral capture")
    ap.add_argument("--range", action="store_true", help="also run the range capture right after neutral")
    ap.add_argument("--robot", choices=["inspire", "dex3"], help="default: config robot.model")
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.clip)
    n_frames, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 15.0
    cap.release()
    assert n_frames > 0, f"cannot read {a.clip}"
    seconds = n_frames / fps                                # play the clip once (the capture loop would loop it)

    over = {"camera": {"source": "camera", "device": a.clip}}
    if a.robot:
        over["robot"] = {"model": a.robot}
    cfg = config_mod.load(overrides=over)
    m, d, info = build_sim(cfg)
    vis = VisionSystem(cfg)
    ctrl = Controller(m, d, info, cfg, vis.state_box.get, calibration=Calibration())
    dt = m.opt.timestep
    n_ctrl = max(1, round(1.0 / cfg.sim.control_hz / dt))
    log, events, tracked_since, phase = [], [], None, "wait"
    vis.start()
    wall0, k = time.perf_counter(), 0
    try:
        while time.perf_counter() - wall0 < seconds:
            if k % n_ctrl == 0:
                now = time.monotonic()
                ctrl.update(now, n_ctrl * dt)
                t = time.perf_counter() - wall0
                tracked_since = (tracked_since or t) if ctrl.tracking_ok else None
                if phase == "wait" and tracked_since is not None and t - tracked_since >= a.settle:
                    ctrl.calibrate("neutral"); phase = "neutral"; events.append((t, "neutral capture start"))
                elif phase == "neutral" and ctrl.cal.has_neutral:
                    if a.range:
                        ctrl.calibrate("range"); phase = "range"; events.append((t, "range capture start"))
                    else:
                        phase = "go"
                elif phase == "range" and ctrl.cal.done["range"]:
                    phase = "go"
                if phase == "go" and ctrl.mode == Mode.DISABLED:
                    ctrl.enable(); phase = "live"; events.append((t, "enabled"))
                pos, _ = ctrl.ik.site_pose_local(d.qpos, d.qpos[ctrl.ik.qadr])     # palm position actually reached
                log.append(dict(t=t, mode=ctrl.mode.value, tracking=ctrl.tracking_ok, cmd=ctrl.cart_cmd.tolist(),
                                reached=(pos - ctrl.pos_home).tolist(), hand=ctrl.hand_cmd.tolist(), cues=dict(ctrl.cues)))
            mujoco.mj_step(m, d)
            k += 1
            lag = d.time - (time.perf_counter() - wall0)
            if lag > 0:
                time.sleep(lag)
    finally:
        ok = vis.stop()

    live = [r for r in log if r["mode"] == "ENABLED"]
    cmd = np.array([r["cmd"] for r in live]) if live else np.zeros((0, 3))
    got = np.array([r["reached"] for r in live]) if live else np.zeros((0, 3))
    hand = np.array([r["hand"] for r in live]) if live else np.zeros((0, 7))
    cues = {c: [r["cues"][c] for r in live] for c in ("index", "middle", "thumb", "pinch")}
    half = np.array(cfg.position.workspace_half_extent)
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    rng = lambda x: [round(float(v), 4) for v in (x.max(0) - x.min(0))] if len(x) else None
    res = dict(clip=a.clip, robot=cfg.robot.model, side=ctrl.side_names[0], clip_seconds=round(seconds, 1), events=events,
               tracked_fraction_live=round(float(np.mean([r["tracking"] for r in live])), 3) if live else 0.0,
               cmd_range_m=rng(cmd), reached_range_m=rng(got),
               follow_error_m_median=round(float(np.median(np.linalg.norm(cmd - got, axis=1))), 4) if live else None,
               follow_error_m_p95=round(float(np.percentile(np.linalg.norm(cmd - got, axis=1), 95)), 4) if live else None,
               follow_corr_xyz=[round(float(np.corrcoef(cmd[:, i], got[:, i])[0, 1]), 3) if np.std(cmd[:, i]) > 1e-4 else None
                                for i in range(3)] if len(cmd) > 2 else None,
               hand_cmd_range_rad=rng(hand),
               cue_range={c: [round(min(v), 2), round(max(v), 2)] for c, v in cues.items() if v},
               within_workspace=bool(np.all(np.abs(cmd) <= half + 1e-9)),
               ctrl_within_range=bool(np.all(d.ctrl >= lo - 1e-9) and np.all(d.ctrl <= hi + 1e-9)),
               finite=bool(np.isfinite(d.qpos).all()), upright=bool(d.xmat[1].reshape(3, 3)[2, 2] > 0.95),
               rejected_jumps=ctrl.wd.rejected_jumps, latency_ms_median=round(ctrl.status()["latency_ms_median"], 1),
               vision_threads_stopped=ok)
    stem = Path(str(Path(a.clip).with_suffix("")) + f"_{cfg.robot.model}")
    Path(f"{stem}_replay.json").write_text(json.dumps(res, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        T = np.array([r["t"] for r in log]); C = np.array([r["cmd"] for r in log]); G = np.array([r["reached"] for r in log])
        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i, (lab, col) in enumerate(zip("xyz", ("tab:red", "tab:green", "tab:blue"))):
            ax[0].plot(T, C[:, i], col, label=f"{lab} commanded"); ax[0].plot(T, G[:, i], col, ls="--", label=f"{lab} reached")
        ax[0].set_ylabel("palm offset (m)"); ax[0].legend(ncol=3, fontsize=7)
        for c in ("index", "middle", "thumb", "pinch"):
            ax[1].plot(T, [r["cues"][c] for r in log], label=c)
        ax[1].set_ylabel("gesture cue 0..1"); ax[1].legend(ncol=4, fontsize=7)
        H = np.array([r["hand"] for r in log])
        for j, n in enumerate(MODELS[cfg.robot.model]["hand"]):
            ax[2].plot(T, H[:, j], label=n)
        ax[2].set_ylabel("hand cmd (rad)"); ax[2].set_xlabel("s"); ax[2].legend(ncol=4, fontsize=7)
        for t, e in events:
            for x in ax:
                x.axvline(t, color="k", lw=0.6, ls=":")
            ax[0].text(t, ax[0].get_ylim()[1], e, fontsize=7, va="top")
        fig.tight_layout(); fig.savefig(f"{stem}_replay.png", dpi=90)
    except Exception as e:                                   # plotting is optional
        print(f"(plot skipped: {e})")
    print(json.dumps(res, indent=2))
    ok_all = res["within_workspace"] and res["ctrl_within_range"] and res["finite"] and res["upright"] and bool(live)
    print("RESULT:", "PASS" if ok_all else "FAIL", f"(details in {stem}_replay.json / .png)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
