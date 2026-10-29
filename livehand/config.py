"""TOML configuration with attribute access and light validation."""
import tomllib
from pathlib import Path
from types import SimpleNamespace

PKG = Path(__file__).resolve().parent
DEFAULT_PATH = PKG / "config.toml"


def _ns(d):
    return SimpleNamespace(**{k: _ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


def load(path=None, overrides=None):
    with open(DEFAULT_PATH, "rb") as f:
        raw = tomllib.load(f)
    if path and Path(path).resolve() != DEFAULT_PATH:
        with open(path, "rb") as f:
            _merge(raw, tomllib.load(f))
    if overrides:
        _merge(raw, overrides)
    validate(raw)
    return _ns(raw)


SUPPORTED_SIDES = {"inspire": ("right", "left"), "dex3": ("right",)}


def active_sides(cfg):
    """Robot sides that will be driven: configured AND supported by the model, in configured order."""
    return [s for s in cfg.robot.sides if s in SUPPORTED_SIDES[cfg.robot.model]]


def validate(c):
    assert c["camera"]["source"] in ("camera", "synthetic"), "camera.source must be 'camera' or 'synthetic'"
    assert c["camera"]["width"] > 0 and c["camera"]["height"] > 0
    assert c["tracking"]["preferred_hand"] in ("Right", "Left", "any")
    for k in ("axis_map",):
        for sec in ("position", "orientation"):
            m = c[sec][k]
            assert len(m) == 3 and all(len(r) == 3 for r in m), f"{sec}.{k} must be 3x3"
    assert len(c["position"]["workspace_half_extent"]) == 3
    assert all(h > 0 for h in c["position"]["workspace_half_extent"])
    assert c["robot"]["model"] in ("inspire", "dex3"), "robot.model must be 'inspire' or 'dex3'"
    assert c["robot"]["sides"] and set(c["robot"]["sides"]) <= {"right", "left"}, "robot.sides: 'right' and/or 'left'"
    cues = {"index", "middle", "ring", "pinky", "rest", "thumb"}
    for model, n in (("dex3", 7), ("inspire", 6)):
        prof = c["hand"][model]
        for p in ("open", "fist", "pinch", "drive", "pinch_mask"):
            assert len(prof[p]) == n, f"hand.{model}.{p} must have {n} entries"
        assert set(prof["drive"]) <= cues, f"hand.{model}.drive entries must be in {sorted(cues)}"
    assert c["safety"]["min_hand_separation"] >= 0
    assert 0 < c["hand"]["max_fraction"] <= 1.0
    assert c["safety"]["stale_timeout"] > 0 and c["position"]["max_speed"] > 0
    assert 0 < c["position"]["range_gain_min"] <= 1.0 <= c["position"]["range_gain_max"]
    assert c["sim"]["control_hz"] > 0
