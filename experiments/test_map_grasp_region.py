"""Lightweight checks for experiments/map_grasp_region.py (grid, symmetry, ordering, classification, aggregation).
Run: python experiments/test_map_grasp_region.py   (no viewer; one scene build for the clearance check)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, for robots/
sys.path.insert(0, str(Path(__file__).resolve().parent))
import map_grasp_region as mg  # noqa: E402
import test_chess_grasp as tcg  # noqa: E402


def test_grid():
    g = mg.make_grid([-6, -3, 0, 3, 6], [-6, -3, 0, 3, 6], [-30, 0, 30])
    assert len(g) == 75 and len({(c["dx_mm"], c["dy_mm"], c["yaw_deg"]) for c in g}) == 75
    assert len(mg.make_grid([0, 0, 3], [0], [0])) == 2  # de-duplication
    pts = {(c["dx_mm"], c["dy_mm"], c["yaw_deg"]) for c in g}
    assert all((-x, -y, -w) in pts for x, y, w in pts), "grid not point-symmetric"
    assert mg.is_symmetric([-9, -3, 0, 3, 9]) and not mg.is_symmetric([-9, 0, 3])
    assert g == mg.make_grid([6, 3, 0, -3, -6], [3, 0, -3, 6, -6], [30, 0, -30]), "grid must not depend on input order"


def test_ordering():
    g = mg.make_grid([-3, 0, 3], [-3, 0, 3], [0])
    a = mg.order_trials(g, 2, seed=7, shuffle=True)
    b = mg.order_trials(g, 2, seed=7, shuffle=True)
    c = mg.order_trials(g, 2, seed=8, shuffle=True)
    assert a == b, "same seed must give identical order"
    assert [t["trial_id"] for t in a] != [t["trial_id"] for t in c], "different seed should reorder"
    assert sorted(t["trial_id"] for t in a) == list(range(18))
    assert [t["trial_id"] for t in mg.order_trials(g, 2, 7, False)] == list(range(18))


def _met(**kw):
    m = dict(invalid_initial_clearance=False, nonfinite_state=False, closure_hand_contacts=["right_hand_thumb_2_link"],
             peak_rise=0.052, rise=0.052, hold_time_above_threshold_s=1.0, hold_slip_m=0.001,
             hold_samples_with_table_or_floor_contact=0, finger_links_in_contact_at_end=["a", "b"],
             init_pawn_penetrations=[], joint_limit_violation_steps=0, min_pelvis_upright=1.0,
             pawn_final_world=[0.3, -0.16, 0.91], pawn_initial_world=[0.3, -0.16, 0.86],
             contacts_end_of_hold=[{"body": "right_hand_thumb_2_link"}])
    m.update(kw)
    return m


def test_classification():
    assert mg.classify(_met())[0] == "success"
    assert mg.classify(_met(invalid_initial_clearance=True, init_clearance_to_hand_m={"right_hand_thumb_2_link": -0.002}))[0] \
        == "invalid_initial_clearance"
    assert mg.classify(_met(nonfinite_state=True))[0] == "sim_error"
    assert mg.classify(_met(closure_hand_contacts=[]))[0] == "grasp_failure"
    assert mg.classify(_met(peak_rise=0.01, rise=0.0))[0] == "lift_failure"
    assert mg.classify(_met(hold_time_above_threshold_s=0.4, rise=0.0))[0] == "hold_failure"
    assert mg.classify(_met(hold_slip_m=0.02))[0] == "hold_failure"
    assert mg.classify(_met(min_pelvis_upright=0.5))[0] == "sim_error"


def test_aggregation():
    mk = lambda dx, dy, yaw, valid, ok, cat: dict(dx_mm=dx, dy_mm=dy, yaw_deg=yaw, valid=valid, success=ok, category=cat)
    rows = [mk(0, 0, 0, 1, 1, "success"), mk(3, 0, 0, 1, 0, "lift_failure"), mk(6, 0, 0, 0, 0, "invalid_initial_clearance"),
            mk(0, 0, 30, 1, 1, "success")]
    a = mg.aggregate(rows)
    assert a["overall"]["valid"] == 3 and a["overall"]["success"] == 2 and a["overall"]["invalid"] == 1
    assert abs(a["overall"]["rate"] - 2 / 3) < 1e-12
    assert a["by_yaw_deg"]["0"]["valid"] == 2 and a["by_position_mm"]["dx=6,dy=0"]["rate"] is None
    assert a["overall"]["categories"]["invalid_initial_clearance"] == 1


def test_clearance_real_scene():
    """A pawn pushed 40 mm into the hand must be flagged invalid (before closing); the nominal must be valid."""
    row = mg.run_trial(dict(dx_mm=-40.0, dy_mm=-40.0, yaw_deg=0.0, trial_id=0, repeat=0, seed=0, run_order=0), 0.002)
    assert row["category"] == "invalid_initial_clearance" and row["valid"] == 0 and row["success"] == 0, row
    m, _, _ = tcg.run_sequence({}, close=True, min_clearance=0.002)
    assert not m["invalid_initial_clearance"] and min(m["init_clearance_to_hand_m"].values()) > 0.02


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
