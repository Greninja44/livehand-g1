"""Print joints, actuators, and ranges for a Menagerie G1 scene."""
import argparse
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parent.parent
MENAGERIE_G1 = ROOT / "models" / "mujoco_menagerie" / "unitree_g1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MENAGERIE_G1 / "scene.xml"))
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.model)
    print(f"model: {args.model}")
    print(f"nbody={m.nbody} njnt={m.njnt} nu={m.nu} nq={m.nq} nv={m.nv} dt={m.opt.timestep}")
    print(f"mesh={m.nmesh} tex={m.ntex}  keyframes={[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(m.nkey)]}")

    print("\nJOINTS (name, type, limited, range[rad]):")
    for i in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        t = mujoco.mjtJoint(m.jnt_type[i]).name.replace("mjJNT_", "")
        rng = m.jnt_range[i] if m.jnt_limited[i] else None
        print(f"  {i:2d} {name or '<free>':32s} {t:6s} {'' if rng is None else f'[{rng[0]:+.3f}, {rng[1]:+.3f}]'}")

    print("\nACTUATORS (name -> joint, ctrlrange, forcerange, kp/kv):")
    for i in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        j = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0])
        kind = "position" if m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE else "other"
        cr = m.actuator_ctrlrange[i] if m.actuator_ctrllimited[i] else None
        fr = m.actuator_forcerange[i] if m.actuator_forcelimited[i] else None
        crs = "unlimited" if cr is None else f"[{cr[0]:+.3f}, {cr[1]:+.3f}]"
        frs = "unlimited" if fr is None else f"[{fr[0]:+.1f}, {fr[1]:+.1f}]"
        print(f"  {i:2d} {name:32s} -> {j:30s} {kind:8s} ctrl={crs} force={frs} kp={m.actuator_gainprm[i,0]:.1f} kv={-m.actuator_biasprm[i,2]:.1f}")


if __name__ == "__main__":
    main()
