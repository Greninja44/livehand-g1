"""Map where the scripted G1 right-Dex3-1 pawn grasp succeeds, as a function of pawn placement.

Every trial replays the *exact* baseline sequence from experiments/test_chess_grasp.py (same arm pose, open-hand
pose, closing targets, lift and 1.0 s hold; no per-trial optimisation). Only the pawn pose changes:

  dx  [mm]  offset along the right-wrist x axis (the direction the fingers point; = world +x, forward)
  dy  [mm]  offset along the right-wrist y axis (the palm normal, toward the thumb/fingers' curl side;
            = world +y, toward the robot's midline)
  dz        always 0: the pawn rests on the table (not a free dimension)
  yaw [deg] rotation of the upright pawn about its own vertical axis

The arm pose and table are fixed from the nominal pose; only the pawn body moves. NOTE: the pawn is an
axisymmetric cylinder, so yaw cannot change the geometry; it is kept as an independent dimension (and
reported) mainly as a consistency check / for future non-symmetric pieces.

Outcome categories: success | invalid_initial_clearance (excluded from rates) | grasp_failure (no hand
contact after closure) | lift_failure (never rose >= 30 mm) | hold_failure (rose, but a hold criterion of
the baseline test failed) | sim_error (exception, non-finite state, joint-limit or balance violation).

Examples:
  python experiments/map_grasp_region.py                 # quick grid
  python experiments/map_grasp_region.py --full          # +-3/6/9 mm, yaw 0/+-15/+-30/+-45 (343 trials)
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, for robots/
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_chess_grasp as tcg  # noqa: E402

ROOT = tcg.gs.ROOT
CATS = ["success", "hold_failure", "lift_failure", "grasp_failure", "sim_error", "invalid_initial_clearance"]
SYMBOL = dict(success="S", hold_failure="h", lift_failure="l", grasp_failure="g", sim_error="E",
              invalid_initial_clearance="x")
RGB = dict(success=(70, 160, 80), hold_failure=(235, 160, 50), lift_failure=(210, 80, 70),
           grasp_failure=(130, 40, 40), sim_error=(130, 70, 170), invalid_initial_clearance=(160, 160, 160))


# ------------------------------------------------------------------ grid / ordering
def parse_list(text):
    return [float(v) for v in str(text).replace(" ", "").split(",") if v != ""]


def is_symmetric(vals):
    v = sorted(vals)
    return all(abs(a + b) < 1e-9 for a, b in zip(v, v[::-1]))


def make_grid(xs, ys, yaws):
    """Cartesian product in a fixed lexicographic order (yaw, dy, dx), de-duplicated."""
    out = []
    for yaw in sorted(set(yaws)):
        for dy in sorted(set(ys)):
            for dx in sorted(set(xs)):
                out.append(dict(dx_mm=float(dx), dy_mm=float(dy), yaw_deg=float(yaw)))
    return out


def order_trials(grid, repeats, seed, shuffle):
    """Trials with canonical IDs (index in grid x repeats); run order is shuffled only if requested, using a
    deterministic generator seeded by `seed`."""
    trials = []
    for i, cfg in enumerate(grid):
        for r in range(repeats):
            tid = i * repeats + r
            trials.append(dict(cfg, trial_id=tid, repeat=r, seed=int(seed * 1_000_003 + tid)))
    order = np.random.default_rng(seed).permutation(len(trials)) if shuffle else np.arange(len(trials))
    return [dict(trials[j], run_order=k) for k, j in enumerate(order)]


# ------------------------------------------------------------------ classification
def classify(met, min_rise=None):
    """Return (category, diagnostic, success) for a run_sequence metrics dict."""
    min_rise = tcg.CONFIG["success_min_rise"] if min_rise is None else min_rise
    if met.get("invalid_initial_clearance"):
        cl = met["init_clearance_to_hand_m"]
        b = min(cl, key=cl.get)
        return "invalid_initial_clearance", f"pawn starts {cl[b]*1000:.1f} mm from {b}", False
    if met["nonfinite_state"]:
        return "sim_error", "non-finite simulation state", False
    shift = np.linalg.norm(np.array(met["pawn_final_world"])[:2] - np.array(met["pawn_initial_world"])[:2]) * 1000
    ends = sorted({x["body"] for x in met["contacts_end_of_hold"]})
    if not met["closure_hand_contacts"]:
        return "grasp_failure", f"no hand contact after closure; pawn moved {shift:.1f} mm in xy; end contacts {ends}", False
    if met["peak_rise"] < min_rise:
        return "lift_failure", (f"peak rise {met['peak_rise']*1000:.1f} mm < {min_rise*1000:.0f} mm; "
                                f"closure contacts {met['closure_hand_contacts']}; end contacts {ends}"), False
    checks = tcg.trial_checks(met)
    failed = [k for k, v in checks.items() if not v]
    if not failed:
        return "success", "", True
    robot = {"finite_state", "no_joint_limit_violation", "robot_upright", "no_initial_penetration"}
    if set(failed) <= robot:
        return "sim_error", "robot-level check failed: " + ",".join(failed), False
    return "hold_failure", (f"failed {','.join(failed)}; held {met['hold_time_above_threshold_s']:.2f} s above threshold; "
                            f"slip {met['hold_slip_m']*1000:.1f} mm; end contacts {ends}"), False


def row_from(trial, met, category, diagnostic, success, wall, min_clr):
    inv = category == "invalid_initial_clearance"
    err = category == "sim_error" and "init_clearance_to_hand_m" not in met
    clr = met.get("init_clearance_to_hand_m")
    blank = inv or err
    g = lambda k, f=lambda v: v: "" if blank or k not in met else f(met[k])
    return dict(
        trial_id=trial["trial_id"], run_order=trial["run_order"], repeat=trial["repeat"], seed=trial["seed"],
        dx_mm=trial["dx_mm"], dy_mm=trial["dy_mm"], dz_mm=0.0, yaw_deg=trial["yaw_deg"],
        init_min_clearance_mm="" if not clr else round(min(clr.values()) * 1000, 3),
        min_clearance_required_mm=round(min_clr * 1000, 3),
        valid=int(not inv and not err),
        closure_completed="" if blank else int(not met["nonfinite_state"]),
        hand_contact_after_closure="" if blank else int(bool(met["closure_hand_contacts"])),
        peak_height_mm=g("peak_rise", lambda v: round(v * 1000, 3)),
        exceeded_threshold=g("peak_rise", lambda v: int(v >= tcg.CONFIG["success_min_rise"])),
        held_full_hold=g("hold_time_above_threshold_s", lambda v: int(v >= tcg.CONFIG["t_hold"] - 1e-9)),
        hold_time_above_s=g("hold_time_above_threshold_s", lambda v: round(v, 3)),
        hold_slip_mm=g("hold_slip_m", lambda v: round(v * 1000, 3)),
        max_object_accel=g("max_object_accel", lambda v: round(v, 3)),
        final_contacts=";".join(sorted({x["body"] for x in met.get("contacts_end_of_hold", [])})),
        success=int(success), category=category, diagnostic=diagnostic, wall_time_s=round(wall, 2))


def run_trial(trial, min_clr):
    params = dict(pawn_offset_mm=(trial["dx_mm"], trial["dy_mm"]), pawn_yaw_deg=trial["yaw_deg"])
    t0 = time.time()
    try:
        met, _, _ = tcg.run_sequence(params, close=True, min_clearance=min_clr)
        cat, diag, ok = classify(met)
    except Exception as e:  # model/IK/simulation error: report, do not skip silently
        met, cat, diag, ok = {}, "sim_error", f"{type(e).__name__}: {e}", False
    return row_from(trial, met, cat, diag, ok, time.time() - t0, min_clr)


# ------------------------------------------------------------------ aggregation / reporting
def rate(rows):
    valid = [r for r in rows if r["valid"]]
    n_ok = sum(r["success"] for r in valid)
    return dict(valid=len(valid), success=n_ok, invalid=len(rows) - len(valid),
                rate=(n_ok / len(valid)) if valid else None)


def aggregate(rows):
    def group(keyf):
        d = defaultdict(list)
        for r in rows:
            d[keyf(r)].append(r)
        return d
    fmt = lambda k: ",".join(f"{v:g}" for v in k) if isinstance(k, tuple) else f"{k:g}"
    out = dict(
        overall=dict(**rate(rows), trials=len(rows),
                     categories={c: sum(1 for r in rows if r["category"] == c) for c in CATS}),
        by_position_mm={f"dx={k[0]:g},dy={k[1]:g}": rate(v) for k, v in sorted(group(lambda r: (r["dx_mm"], r["dy_mm"])).items())},
        by_yaw_deg={f"{k:g}": rate(v) for k, v in sorted(group(lambda r: r["yaw_deg"]).items())},
        by_dx_mm={f"{k:g}": rate(v) for k, v in sorted(group(lambda r: r["dx_mm"]).items())},
        by_dy_mm={f"{k:g}": rate(v) for k, v in sorted(group(lambda r: r["dy_mm"]).items())},
    )
    return out


def cell_category(rs):
    """Dominant category in a grid cell (success if every valid repeat succeeded)."""
    valid = [r for r in rs if r["valid"]]
    if not valid:
        return "invalid_initial_clearance"
    cnt = Counter(r["category"] for r in valid)
    return "success" if cnt["success"] == len(valid) else max((c for c in cnt if c != "success"), key=lambda c: cnt[c])


def text_summary(rows, agg, cfg):
    xs = sorted({r["dx_mm"] for r in rows}); ys = sorted({r["dy_mm"] for r in rows}); yaws = sorted({r["yaw_deg"] for r in rows})
    o = agg["overall"]
    L = ["Grasp region map (scripted baseline grasp, fixed targets; only the pawn pose varies)", "",
         f"trials: {o['trials']}  valid: {o['valid']}  invalid_initial_clearance (excluded): {o['invalid']}",
         f"valid-trial success: {o['success']}/{o['valid']}" + (f" = {100*o['rate']:.1f}%" if o["rate"] is not None else ""),
         "categories: " + ", ".join(f"{c}={n}" for c, n in o["categories"].items()), "",
         "Legend: S success, h hold_failure, l lift_failure, g grasp_failure, E sim_error, x invalid start",
         "Axes: columns = dx (mm, along fingers/forward), rows = dy (mm, along palm normal/toward midline)", ""]
    cells = defaultdict(list)
    for r in rows:
        cells[(r["yaw_deg"], r["dy_mm"], r["dx_mm"])].append(r)
    for yaw in yaws:
        L.append(f"yaw = {yaw:g} deg   (success {agg['by_yaw_deg'][f'{yaw:g}']['success']}/{agg['by_yaw_deg'][f'{yaw:g}']['valid']})")
        L.append("  dy\\dx " + " ".join(f"{x:+5.0f}" for x in xs))
        for dy in reversed(ys):
            L.append(f"  {dy:+5.0f} " + " ".join(f"{SYMBOL[cell_category(cells[(yaw, dy, x)])]:>5}" for x in xs))
        L.append("")
    L.append("Success rate by position (all yaws, valid trials only):")
    for k, v in agg["by_position_mm"].items():
        L.append(f"  {k:16s} {v['success']}/{v['valid']}" + (f"  ({v['invalid']} invalid)" if v["invalid"] else ""))
    L += ["", "By yaw: " + "; ".join(f"{k}deg {v['success']}/{v['valid']}" for k, v in agg["by_yaw_deg"].items()),
          "", "Only configurations listed above were tested; nothing is claimed outside the tested ranges "
          f"(dx {min(xs):g}..{max(xs):g} mm, dy {min(ys):g}..{max(ys):g} mm, yaw {min(yaws):g}..{max(yaws):g} deg)."]
    return "\n".join(L) + "\n"


def draw_map(rows, path):
    from PIL import Image, ImageDraw
    xs = sorted({r["dx_mm"] for r in rows}); ys = sorted({r["dy_mm"] for r in rows}); yaws = sorted({r["yaw_deg"] for r in rows})
    cs, pad, top, lab = 44, 16, 40, 46
    pw, ph = lab + cs * len(xs) + pad, top + cs * len(ys) + 34
    img = Image.new("RGB", (max(pw * len(yaws) + pad, pad + len(CATS) * 190), ph + 70), (250, 250, 250))
    dr = ImageDraw.Draw(img)
    cells = defaultdict(list)
    for r in rows:
        cells[(r["yaw_deg"], r["dy_mm"], r["dx_mm"])].append(r)
    for pi, yaw in enumerate(yaws):
        ox = pad + pi * pw
        dr.text((ox + lab, 6), f"yaw {yaw:g} deg", fill=(0, 0, 0))
        dr.text((ox + lab, 20), "x: dx mm (fingers)   y: dy mm (palm normal)", fill=(80, 80, 80))
        for ci, x in enumerate(xs):
            dr.text((ox + lab + ci * cs + 10, top - 14), f"{x:+.0f}", fill=(0, 0, 0))
        for ri, dy in enumerate(reversed(ys)):
            dr.text((ox + 6, top + ri * cs + 16), f"{dy:+.0f}", fill=(0, 0, 0))
            for ci, x in enumerate(xs):
                rs = cells[(yaw, dy, x)]
                c = cell_category(rs)
                x0, y0 = ox + lab + ci * cs, top + ri * cs
                dr.rectangle([x0, y0, x0 + cs - 2, y0 + cs - 2], fill=RGB[c])
                v = [r for r in rs if r["valid"]]
                if len(rs) > 1:
                    dr.text((x0 + 6, y0 + 14), f"{sum(r['success'] for r in v)}/{len(v)}", fill=(255, 255, 255))
                if abs(x) < 1e-9 and abs(dy) < 1e-9:
                    dr.rectangle([x0, y0, x0 + cs - 2, y0 + cs - 2], outline=(0, 0, 0), width=2)
    ly = ph + 8
    for i, c in enumerate(CATS):
        dr.rectangle([pad + i * 190, ly, pad + i * 190 + 14, ly + 14], fill=RGB[c])
        dr.text((pad + i * 190 + 20, ly + 1), c, fill=(0, 0, 0))
    dr.text((pad, ly + 24), "black outline = nominal (0,0); whole-cell colour = dominant outcome", fill=(80, 80, 80))
    img.save(path)


# ------------------------------------------------------------------ main
def baseline_gate():
    """Replay the nominal grasp and the open-finger control; they must pass before any mapping is trusted."""
    met, _, _ = tcg.run_sequence({}, close=True)
    ctl, _, _ = tcg.run_sequence({}, close=False)
    checks = tcg.trial_checks(met)
    ok = all(checks.values()) and ctl["rise"] < 0.01
    return ok, dict(checks=checks, rise_mm=met["rise"] * 1000, control_rise_mm=ctl["rise"] * 1000,
                    failed=[k for k, v in checks.items() if not v])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x-offsets-mm", default="-6,-3,0,3,6", help="dx values (comma list)")
    ap.add_argument("--y-offsets-mm", default="-6,-3,0,3,6", help="dy values (comma list)")
    ap.add_argument("--yaws-deg", default="-30,0,30", help="pawn yaw values (comma list)")
    ap.add_argument("--full", action="store_true", help="larger sweep: +-3/6/9 mm and yaw 0,+-15,+-30,+-45")
    ap.add_argument("--repeats", type=int, default=1, help="repetitions per configuration")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true", help="shuffle run order (deterministic given --seed)")
    ap.add_argument("--min-clearance-mm", type=float, default=2.0, help="initial pawn-hand clearance required")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "logs", help="must be under logs/")
    ap.add_argument("--skip-baseline-gate", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    if a.full:
        a.x_offsets_mm = a.y_offsets_mm = "-9,-6,-3,0,3,6,9"
        a.yaws_deg = "-45,-30,-15,0,15,30,45"
    xs, ys, yaws = parse_list(a.x_offsets_mm), parse_list(a.y_offsets_mm), parse_list(a.yaws_deg)
    out_dir = a.out_dir.resolve()
    if (ROOT / "logs").resolve() not in [out_dir, *out_dir.parents]:
        sys.exit("ERROR: --out-dir must be under logs/")
    for name, v in (("x", xs), ("y", ys), ("yaw", yaws)):
        if not is_symmetric(v):
            print(f"WARNING: {name} offsets {v} are not symmetric around 0")
    if 0.0 not in xs or 0.0 not in ys:
        print("WARNING: nominal (0,0) offset is not in the grid")
    out_dir.mkdir(parents=True, exist_ok=True)
    min_clr = a.min_clearance_mm / 1000.0

    gate = None
    if not a.skip_baseline_gate:
        print("== baseline gate: nominal grasp + open-finger control ==")
        ok, gate = baseline_gate()
        print(f"nominal rise {gate['rise_mm']:.1f} mm, control rise {gate['control_rise_mm']:.1f} mm, "
              f"{'PASS' if ok else 'FAIL ' + str(gate['failed'])}")
        if not ok:
            (out_dir / "grasp_region_summary.json").write_text(json.dumps(dict(error="baseline gate failed", gate=gate), indent=2))
            sys.exit("ERROR: baseline grasp/control failed; mapping aborted (not skipping silently)")

    trials = order_trials(make_grid(xs, ys, yaws), a.repeats, a.seed, a.shuffle)
    print(f"== mapping {len(trials)} trials ({len(xs)}x{len(ys)} positions, {len(yaws)} yaws, {a.repeats} repeat(s)) ==")
    rows, t0 = [], time.time()
    for i, tr in enumerate(trials):
        r = run_trial(tr, min_clr)
        rows.append(r)
        print(f"[{i+1:3d}/{len(trials)}] id={r['trial_id']:3d} dx={r['dx_mm']:+.0f} dy={r['dy_mm']:+.0f} yaw={r['yaw_deg']:+.0f} "
              f"-> {r['category']:26s} peak={r['peak_height_mm']}mm {r['diagnostic'][:70]}", flush=True)
    rows.sort(key=lambda r: r["trial_id"])

    fields = list(rows[0].keys())
    with open(out_dir / "grasp_region_trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    agg = aggregate(rows)
    cfg = dict(x_offsets_mm=xs, y_offsets_mm=ys, yaws_deg=yaws, repeats=a.repeats, seed=a.seed, shuffle=a.shuffle,
               min_clearance_mm=a.min_clearance_mm, grasp_config=tcg.CONFIG,
               convention="dx along wrist x (fingers/forward = world +x); dy along wrist y (palm normal = world +y); "
                          "dz=0 (on table); yaw about pawn vertical axis; cylinder is axisymmetric so yaw is geometrically a no-op",
               wall_time_s=round(time.time() - t0, 1))
    (out_dir / "grasp_region_summary.json").write_text(json.dumps(dict(config=cfg, baseline_gate=gate, **agg), indent=2))
    txt = text_summary(rows, agg, cfg)
    (out_dir / "grasp_region_summary.txt").write_text(txt)
    if not a.no_plot:
        draw_map(rows, out_dir / "grasp_region_map.png")
    print("\n" + txt)
    print(f"wrote {out_dir}/grasp_region_trials.csv, grasp_region_summary.json/.txt" + ("" if a.no_plot else ", grasp_region_map.png"))


if __name__ == "__main__":
    main()
