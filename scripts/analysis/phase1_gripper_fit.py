# Phase 1 analysis (exploratory, kept for reproducibility). Run with the sim env:
#   ~/.venvs/so101_stack_sim/bin/python scripts/analysis/phase1_gripper_fit.py
import mujoco, numpy as np, json
from scipy.spatial import cKDTree
m = mujoco.MjModel.from_xml_path("/Users/maxchan/so101_stack_game/sim/assets/so101/so101_new_calib.xml")
d = mujoco.MjData(m)
def verts(g):
    mid = m.geom_dataid[g]; a = m.mesh_vertadr[mid]; n = m.mesh_vertnum[mid]
    return d.geom_xpos[g] + m.mesh_vert[a:a+n] @ d.geom_xmat[g].reshape(3,3).T
gm = [g for g in range(m.ngeom) if m.geom_group[g]==2 and m.body(m.geom_bodyid[g]).name=="moving_jaw_so101_v1"][0]
gf = [g for g in range(m.ngeom) if m.geom_group[g]==2 and m.mesh(m.geom_dataid[g]).name=="wrist_roll_follower_so101_v1"][0]
jg = m.joint("gripper").id
def gaps(theta):
    d.qpos[:] = 0; d.qpos[m.jnt_qposadr[jg]] = theta; mujoco.mj_forward(m, d)
    h, ax = d.xanchor[jg], d.xaxis[jg]
    mj, fj = verts(gm), verts(gf)
    v = mj - h; r = np.linalg.norm(v - np.outer(v@ax, ax), axis=1); rt = r.max()
    tree = cKDTree(fj)
    tip = tree.query(mj[r > rt-2e-3])[0].min()
    d10 = tree.query(mj[(r > rt-11e-3) & (r < rt-9e-3)])[0].min()
    return tip, d10
for deg in [-20,-18,-16,-14,-12,-10]:
    print(deg, ["%.1f"%(x*1e3) for x in gaps(np.radians(deg))])
cal = json.load(open("/Users/maxchan/so101_stack_game/inputs/calibration/my_follower_arm.json"))["gripper"]
k = (cal["range_max"]-cal["range_min"])*360/4095/100
print("deg per gripper unit k = %.4f; full 0..100 = %.1f deg servo travel" % (k, 100*k))
obs = [(0.13,"tip",0.0),(24.0,"d10",0.040),(18.6,"d10",0.035),(15.5,"d10",0.029),(99.66,"tip",0.150),(99.66,"d10",0.122)]
best=None
for tc in np.arange(-25,5,0.25):
    res=[]
    for rd,kind,val in obs:
        t,d10 = gaps(np.radians(tc + rd*k)); res.append(((t if kind=="tip" else d10)-val)*1e3)
    rms=np.sqrt(np.mean(np.square(res)))
    if best is None or rms<best[0]: best=(rms,tc,res)
print("best theta_closed = %.2f deg, rms %.1f mm" % (best[1],best[0]))
for (rd,kind,val),r in zip(obs,best[2]): print(f"  reading {rd:6.2f} {kind}: measured {val*1e3:5.1f} mm  sim-meas {r:+5.1f} mm  (sim angle {best[1]+rd*k:6.1f} deg)")
# fit without the open-gripper rows (box widths + closed only)
best2=None
for tc in np.arange(-25,5,0.25):
    res=[(gaps(np.radians(tc+rd*k))[0 if kind=="tip" else 1]-val)*1e3 for rd,kind,val in obs[:4]]
    rms=np.sqrt(np.mean(np.square(res)))
    if best2 is None or rms<best2[0]: best2=(rms,tc,res)
print("box+closed only: theta_closed %.2f, rms %.1f, res %s" % (best2[1],best2[0],np.round(best2[2],1)))

def tip2tip(theta, back=0.0):
    d.qpos[:] = 0; d.qpos[m.jnt_qposadr[jg]] = theta; mujoco.mj_forward(m, d)
    h, ax = d.xanchor[jg], d.xaxis[jg]
    mj, fj = verts(gm), verts(gf)
    v = mj - h; r = np.linalg.norm(v - np.outer(v@ax, ax), axis=1)
    tm = mj[np.argmax(r)]
    R = d.xmat[m.body("gripper").id].reshape(3,3); loc = (fj - d.xpos[m.body("gripper").id]) @ R
    tf = fj[np.argmin(loc[:,2])]
    if back:  # points `back` metres from each tip, along each jaw toward its root
        tm = tm - back * (tm - h) / np.linalg.norm(tm - h)
        tf = tf + back * R[:,2]
    return np.linalg.norm(tm - tf)
tc = -10.75
for rd in [0.13, 99.66]:
    th = tc + rd*k
    print(f"reading {rd}: sim angle {th:.1f} deg, tip-tip {tip2tip(np.radians(th))*1e3:.1f} mm, 1cm back {tip2tip(np.radians(th),0.01)*1e3:.1f} mm")
print("at sim limit 100 deg: tip-tip %.1f mm" % (tip2tip(np.radians(100))*1e3))
for w,name in [(0.040,'no3'),(0.035,'no35'),(0.029,'no10')]:
    print(name, "width", w)

print("\n### grasp clearance per box (gap 1 cm back from the tips, mapping theta = -10.75 + 1.303*g)")
from scipy.optimize import brentq
def reading_for_gap(gap):
    f = lambda g: gaps(np.radians(tc + g*k))[1] - gap
    return brentq(f, 1, 99)
meas = {"no3": 24.0, "no35": 18.6, "no10": 15.5}
for name, w in (("no3",0.040),("no35",0.035),("no10",0.029)):
    gc = reading_for_gap(w); gp = reading_for_gap(w + 0.015)
    print(f"  {name}: width {w*1e3:.0f} mm | contact reading sim {gc:5.1f} (measured {meas[name]}) | pre-grasp width+15 mm -> reading {gp:5.1f} | max opening margin {150 - w*1e3:.0f} mm")
