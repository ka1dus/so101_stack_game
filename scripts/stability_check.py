"""Phase 4: reproduce the real stability tests (inputs/stability/overhang_tests.yaml) in sim.

Physics parameters live in sim/assets/physics.yaml (also used by build_scene.py), so tuning
here and the game scene can't drift apart.

Tests, mirroring the real protocol:
  overhang : stack built aligned and centred on the yellow pad; the TOP box is offset along
             its long side; overhang = top box edge past the edge of the box below
             (= centre offset + (L_top - L_below)/2). Bisection on the overhang at which the
             top box topples within 2 s (tilt > 10 deg or drops a box height). 0.25 mm resolution.
  slide    : top box resting on a support (box type / yellow tape / black mat); gravity tilted
             by theta (= tilting the board); slide onset = top box moves > 3 mm within 2 s.
  tap      : 4-box stack; a sideways velocity kick on the tapped level (impulse J); bisection
             on the J that topples the stack. Real taps aren't calibrated, so the check is the
             ORDER and ratio: the stack that toppled more often must have the lower threshold.

    ~/.venvs/so101_stack_sim/bin/python scripts/stability_check.py [--physics alt.yaml]
"""

import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
MEAS = yaml.safe_load((ROOT / "inputs/measurements.yaml").read_text())
REAL = yaml.safe_load((ROOT / "inputs/stability/overhang_tests.yaml").read_text())
PHYS_PATH = Path(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--physics=")), ROOT / "sim/assets/physics.yaml"))
PHYS = yaml.safe_load(PHYS_PATH.read_text())
BOX = {v["sim_name"]: dict(v["dimensions"], mass=v["mass_g"] / 1000) for v in MEAS["boxes"]["variants"]}


def opt_xml():
    o = PHYS["option"]
    return (f'<option timestep="{o["timestep"]}" cone="{o["cone"]}" impratio="{o["impratio"]}" '
            f'noslip_iterations="{o.get("noslip_iterations", 0)}"><flag warmstart="enable"/></option>')


def contact_attrs(kind):
    c = PHYS["contact"]
    return f'condim="{c["condim"]}" solref="{c["solref"][0]} {c["solref"][1]}" solimp="{" ".join(map(str, c["solimp"]))}"'


def box_geom(name, t, extra=""):
    b = BOX[t]
    fr = PHYS["friction"]["box"][t]
    return (f'<geom name="{name}" type="box" size="{b["length"]/2} {b["width"]/2} {b["height"]/2}" '
            f'mass="{b["mass"]}" friction="{fr} {PHYS["friction"]["torsional"]} {PHYS["friction"]["rolling"]}" '
            f'{contact_attrs("box")} {extra}/>')


def surface_geom(name, kind, size=(0.15, 0.15, 0.005), pos=(0, 0, -0.005)):
    fr = PHYS["friction"]["surface"][kind]
    return (f'<geom name="{name}" type="box" size="{size[0]} {size[1]} {size[2]}" pos="{pos[0]} {pos[1]} {pos[2]}" '
            f'friction="{fr} {PHYS["friction"]["torsional"]} {PHYS["friction"]["rolling"]}" priority="1" '
            f'{contact_attrs("surface")}/>')


def stack_model(stack, top_offset=0.0, gravity=(0, 0, -9.81)):
    """Bodies box0..boxN bottom->top, aligned along x (long side); top box shifted by top_offset in x."""
    z = 0.0
    bodies = []
    for i, t in enumerate(stack):
        h = BOX[t]["height"]
        dx = top_offset if i == len(stack) - 1 else 0.0
        bodies.append(f'<body name="b{i}" pos="{dx} 0 {z + h/2 + 1e-4}"><freejoint/>{box_geom(f"g{i}", t)}</body>')
        z += h + 1e-4
    xml = (f'<mujoco>{opt_xml()}<option gravity="{gravity[0]} {gravity[1]} {gravity[2]}"/>'
           f'<worldbody>{surface_geom("pad", "yellow_tape")}{"".join(bodies)}</worldbody></mujoco>')
    return mujoco.MjModel.from_xml_string(xml)


def toppled(m, d, i_top, z0):
    R = d.xmat[m.body(f"b{i_top}").id].reshape(3, 3)
    tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
    return tilt > 10 or d.xpos[m.body(f"b{i_top}").id][2] < z0 - 0.008


def overhang_threshold(stack):
    lt, lb = BOX[stack[-1]]["length"], BOX[stack[-2]]["length"]
    def topples(o):
        m = stack_model(stack, top_offset=o - (lt - lb) / 2); d = mujoco.MjData(m)
        mujoco.mj_forward(m, d); z0 = d.xpos[m.body(f"b{len(stack)-1}").id][2]
        for _ in range(int(2.0 / m.opt.timestep)):
            mujoco.mj_step(m, d)
            if toppled(m, d, len(stack) - 1, z0):
                return True
        return False
    lo, hi = 0.0, lt
    while hi - lo > 0.00025:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if topples(mid) else (mid, hi)
    return (lo + hi) / 2


def slide_angle(top, support):
    """support = box type or a surface kind ('yellow_tape', 'black_mat')."""
    def slides(theta):
        g = 9.81 * np.array([np.sin(np.radians(theta)), 0, -np.cos(np.radians(theta))])
        ht = BOX[top]["height"]
        if support in BOX:
            hs = BOX[support]["height"]
            base = f'<body name="s" pos="0 0 {hs/2}">{box_geom("gs", support)}</body>'      # fixed (welded) support box
            z = hs + ht / 2 + 1e-4
        else:
            base = ""; z = ht / 2 + 1e-4
        xml = (f'<mujoco>{opt_xml()}<option gravity="{g[0]} {g[1]} {g[2]}"/><worldbody>'
               f'{surface_geom("pad", "yellow_tape" if support in BOX else support)}{base}'
               f'<body name="t" pos="0 0 {z}"><freejoint/>{box_geom("gt", top)}</body></worldbody></mujoco>')
        m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
        mujoco.mj_forward(m, d); x0 = d.xpos[m.body("t").id].copy()
        for _ in range(int(2.0 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        return np.linalg.norm(d.xpos[m.body("t").id] - x0) > 0.003
    lo, hi = 0.0, 60.0
    while hi - lo > 0.25:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if slides(mid) else (mid, hi)
    return (lo + hi) / 2


TAP_MODES = ("impulse", "velocity", "push")   # fixed impulse / fixed speed / 50 ms force pulse


def tap_threshold(stack, level, mode="push", axis=1):
    """Tap on level (1 = bottom), sideways: axis 1 = across the long side (y), 0 = along it (x).
    mode impulse: J (N s) as a velocity kick J/m; velocity: a kick of J m/s regardless of mass;
    push: a force of J*100 N for 50 ms at the box centre (a fingertip push of fixed strength)."""
    def topples(J):
        m = stack_model(stack); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        for _ in range(100):
            mujoco.mj_step(m, d)
        z0 = [d.xpos[m.body(f"b{i}").id][2] for i in range(len(stack))]
        b = m.body(f"b{level-1}").id
        if mode == "push":
            n = int(0.05 / m.opt.timestep)
            for _ in range(n):
                d.xfrc_applied[b, axis] = J * 100; mujoco.mj_step(m, d)
            d.xfrc_applied[b, axis] = 0
        else:
            dv = J / BOX[stack[level - 1]]["mass"] if mode == "impulse" else J
            d.qvel[m.jnt_dofadr[m.body_jntadr[b]] + axis] += dv
        for _ in range(int(2.0 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        return any(toppled(m, d, i, z0[i]) for i in range(len(stack)))
    lo, hi = 0.0, 2.0
    while hi - lo > 0.0005:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if topples(mid) else (mid, hi)
    return (lo + hi) / 2


def tap_threshold_old(stack, level):
    """Impulse (N s) on the tapped level (1 = bottom), sideways (y, across the long side)."""
    def topples(J):
        m = stack_model(stack); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        for _ in range(100):          # settle
            mujoco.mj_step(m, d)
        z0 = [d.xpos[m.body(f"b{i}").id][2] for i in range(len(stack))]
        b = m.body(f"b{level-1}").id
        d.qvel[m.jnt_dofadr[m.body_jntadr[b]] + 1] += J / BOX[stack[level - 1]]["mass"]
        for _ in range(int(2.0 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        return any(toppled(m, d, i, z0[i]) for i in range(len(stack)))
    lo, hi = 0.0, 0.2
    while hi - lo > 0.0005:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if topples(mid) else (mid, hi)
    return (lo + hi) / 2


def main():
    print(f"physics: {PHYS_PATH.relative_to(ROOT) if PHYS_PATH.is_relative_to(ROOT) else PHYS_PATH}")
    print("\nOVERHANG (mm at topple)          real mean (runs)        sim     sim-real")
    worst = 0
    for row in REAL["overhang_tests"]:
        st = row["stack"]; real = np.mean(row["topple_overhang_mm"])
        sim = overhang_threshold(st) * 1000; worst = max(worst, abs(sim - real))
        print(f"  {' -> '.join(st):22s} {real:6.1f} {str(row['topple_overhang_mm']):14s} {sim:7.1f} {sim-real:+7.1f}")
    print(f"  worst |sim - real| = {worst:.1f} mm (target <= 2-3 mm)")
    print("\nSLIDE (deg at slide onset)       real mean   sim")
    for k, v in REAL["slide_tests"].items():
        if k == "box_on_box_deg":
            for pair, vals in v.items():
                top, support = pair.split("_on_")
                print(f"  {top:5s} on {support:12s}        {np.mean(vals):5.1f}  {slide_angle(top, support):5.1f}")
        else:
            support = {"box_on_yellow_area_deg": "yellow_tape", "box_on_black_mat_deg": "black_mat"}[k]
            print(f"  no35  on {support:12s}        {np.mean(v):5.1f}  {slide_angle('no35', support):5.1f}")
    print("\nTAP (impulse threshold, N s; lower = less stable)   real toppled")
    units = {"impulse": "N s", "velocity": "m/s", "push": "x100 N for 50 ms"}
    for mode in TAP_MODES:
        for axis, axname in ((1, "across"), (0, "along")):
            th = []
            for row in REAL["tap_tests"]:
                J = tap_threshold(row["stack"], row["tapped_level"], mode, axis)
                th.append((J, row["toppled_count"] / row["trials"]))
            order_ok = (th[0][0] > th[1][0]) == (th[0][1] < th[1][1])
            print(f"  {mode:8s} {axname:6s}: A {th[0][0]:.3f}  B {th[1][0]:.3f} ({units[mode]})  A/B {th[0][0]/th[1][0]:4.2f}  "
                  f"order {'OK' if order_ok else 'WRONG'} (real: A 1/5, B 3/5 -> B should need less)")


if __name__ == "__main__":
    main()
