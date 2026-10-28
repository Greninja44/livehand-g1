"""On-screen status panel for the preview window."""
import cv2

GREEN, RED, AMBER, WHITE = (80, 220, 80), (60, 60, 255), (0, 190, 255), (240, 240, 240)
HELP = "e enable/disable  SPACE e-stop  n/o/f/p calibrate  r orient  g hand  h home  s/l save/load  q quit"


def draw_panel(frame, st, cam_fps, inf_fps, error=""):
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 118), (0, 0, 0), -1)
    mode_col = GREEN if st["mode"] == "ENABLED" else RED if st["mode"] == "E-STOP" else AMBER
    put = lambda txt, y, col=WHITE: cv2.putText(out, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    put(f"MODE: {st['mode']}" + ("  (homing)" if st["homing"] else ""), 18, mode_col)
    trk = "TRACKING" if st["tracking"] else (f"LOST {st['lost_for']:.1f}s" if st["lost_for"] else "NO HAND")
    put(f"{trk}   cam {cam_fps:4.1f} fps   infer {inf_fps:4.1f} fps   latency med {st['latency_ms_median']:.0f} / p95 {st['latency_ms_p95']:.0f} ms",
        36, GREEN if st["tracking"] else AMBER)
    cal = st["calib"]
    put("calib: " + " ".join(f"{k}:{'ok' if v else '--'}" for k, v in cal.items())
        + (f"   >> capturing {st['calibrating']} {100 * st['calib_progress']:.0f}%" if st["calibrating"] else ""), 54,
        AMBER if st["calibrating"] else WHITE)
    c = st["cues"]
    put(f"orient:{'ON' if st['orientation'] else 'off'} hand:{'ON' if st['hand'] else 'off'}  "
        f"cues idx {c['index']:.2f} mid {c['middle']:.2f} thumb {c['thumb']:.2f} pinch {c['pinch']:.2f}  "
        f"d=({st['offset'][0]:+.2f},{st['offset'][1]:+.2f},{st['offset'][2]:+.2f}) m", 72)
    msgs = list(st["messages"])
    put((msgs[-1] if msgs else ""), 90, AMBER)
    if error:
        put("ERROR: " + error[:90], 108, RED)
    else:
        put(f"rejected jumps {st['rejected_jumps']} invalid {st['rejected_invalid']}", 108)
    put(HELP, h - 8, (170, 170, 170))
    return out
