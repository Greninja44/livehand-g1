"""On-screen status panel for the preview window."""
import cv2

GREEN, RED, AMBER, WHITE = (80, 220, 80), (60, 60, 255), (0, 190, 255), (240, 240, 240)
HELP = ("e enable/disable   SPACE e-stop   h home   q/ESC quit",
        "calib: n neutral  m range  o open  f fist  p pinch  c other hand   r orient  g hand  s/l save/load")
SHORT = dict(neutral="n", range="m", open="o", fist="f", pinch="p")
STEPS = ("neutral", "open", "fist", "pinch")


def next_step(st):
    """The guided calibration prompt: what the user should do next. Hands are calibrated one at a time (right first)
    so the other hand is free for the keyboard."""
    sides = st["sides"]
    both = len(sides) > 1
    s = st["cal_side"]
    c = sides[s]["calib"]
    who = f"{s.upper()} hand" if both else "hand"
    where = f" in the {s} half of the image, other hand on the keyboard" if both else ""
    if st["mode"] == "E-STOP":
        return "E-STOP latched: press e to clear (then e again to enable)"
    if any(v["calibrating"] for v in sides.values()):
        return "keep holding / sweeping ..." if sides[s]["tracking"] else f"show your {who} to the camera to continue capture"
    if not c["neutral"]:
        return f"{who.upper()} - STEP 1: hold it up{where}, palm to camera, relaxed -> n"
    if not all(c[k] for k in STEPS):
        return f"{who.upper()} - STEP 2: o = open, f = fist, p = thumb-index pinch (hold each)   [optional: m + sweep]"
    todo = [x for x, v in sides.items() if not all(v["calib"][k] for k in STEPS)]
    if todo:
        return f"{who} done - press c to calibrate the {todo[0].upper()} hand"
    if st["mode"] == "DISABLED":
        return "ready: s saves calibration; press e to enable mirroring"
    return ""


def draw_panel(frame, st, cam_fps, inf_fps, error="", inf_ms=0.0):
    h, w = frame.shape[:2]
    if w < 960:                                 # display-only upscale so the panel text fits (tracking is unaffected)
        out = cv2.resize(frame, (960, round(h * 960 / w)), interpolation=cv2.INTER_LINEAR)
    else:
        out = frame.copy()
    h, w = out.shape[:2]
    s = min(1.0, w / 960)
    lh = int(18 * s + 0.5)
    sides = st["sides"]
    n_lines = 5 + 2 * len(sides)
    cv2.rectangle(out, (0, 0), (w, n_lines * lh + 6), (0, 0, 0), -1)
    cv2.rectangle(out, (0, h - 2 * lh - 6), (w, h), (0, 0, 0), -1)
    mode_col = GREEN if st["mode"] == "ENABLED" else RED if st["mode"] == "E-STOP" else AMBER
    put = lambda txt, y, col=WHITE: cv2.putText(out, txt, (6, y * lh), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, col, 1, cv2.LINE_AA)
    put(f"MODE: {st['mode']}" + ("  (homing)" if st["homing"] else "") + f"   robot: {st['model']}   "
        f"orient:{'ON' if st['orientation'] else 'off'}  hand:{'ON' if st['hand'] else 'off'}"
        + (f"   calibrating: {st['cal_side'].upper()} hand (c switches)" if len(sides) > 1 else ""), 1, mode_col)
    put(f"cam {cam_fps:4.1f} fps  infer {inf_fps:4.1f} fps ({inf_ms:.0f} ms)  "
        f"capture->cmd med {st['latency_ms_median']:.0f} / p95 {st['latency_ms_p95']:.0f} ms", 2)
    y = 3
    for name, v in sides.items():
        trk = "TRACKING" if v["tracking"] else (f"LOST {v['lost_for']:.1f}s" if v["lost_for"] else "NO HAND")
        cal = " ".join(f"{SHORT[k]}:{'ok' if ok else '--'}" for k, ok in v["calib"].items())
        cap = f"  >> {v['calibrating']} {100 * v['calib_progress']:.0f}%" if v["calibrating"] else ""
        put(f"{name.upper():5s} {trk:9s}  calib {cal}{cap}", y, GREEN if v["tracking"] else AMBER); y += 1
        c, o = v["cues"], v["offset"]
        put(f"      idx {c['index']:.2f} mid {c['middle']:.2f} ring {c['ring']:.2f} pinky {c['pinky']:.2f} "
            f"thumb {c['thumb']:.2f} pinch {c['pinch']:.2f}  d=({o[0]:+.2f},{o[1]:+.2f},{o[2]:+.2f}) m", y); y += 1
    put(next_step(st), y, GREEN); y += 1
    msgs = list(st["messages"])
    put((msgs[-1] if msgs else ""), y, AMBER); y += 1
    if error:
        put("ERROR: " + error[:90], y, RED)
    else:
        put(f"rejected jumps {st['rejected_jumps']} invalid {st['rejected_invalid']}  hand-separation holds {st['separations']}", y)
    cv2.putText(out, HELP[0], (6, h - lh - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.putText(out, HELP[1], (6, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (190, 190, 190), 1, cv2.LINE_AA)
    return out
