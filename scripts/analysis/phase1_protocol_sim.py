# Phase 1 analysis (exploratory, kept for reproducibility). Run with the sim env:
#   ~/.venvs/so101_stack_sim/bin/python scripts/analysis/phase1_protocol_sim.py
"""Monte-Carlo identifiability of joint offsets for two touch protocols."""
import mujoco, numpy as np
from scipy.optimize import least_squares
m = mujoco.MjModel.from_xml_path("/Users/maxchan/so101_stack_game/sim/assets/so101/so101_new_calib.xml")
d = mujoco.MjData(m)
J = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll"]
adr = [m.jnt_qposadr[m.joint(j).id] for j in J]; gadr = m.jnt_qposadr[m.joint("gripper").id]
def verts(g):
    mid = m.geom_dataid[g]; a = m.mesh_vertadr[mid]; n = m.mesh_vertnum[mid]
    return d.geom_xpos[g] + m.mesh_vert[a:a+n] @ d.geom_xmat[g].reshape(3,3).T
jaw = [g for g in range(m.ngeom) if m.geom_group[g]==2 and m.body(m.geom_bodyid[g]).name in ("gripper","moving_jaw_so101_v1")]
d.qpos[:]=0; mujoco.mj_forward(m,d); PAN = np.r_[d.xanchor[m.joint("shoulder_pan").id][:2],0]
gb = m.body("gripper").id
def fk(qdeg):  # sim degrees for 5 body joints; gripper closed
    for a,v in zip(adr,qdeg): d.qpos[a]=np.radians(v)
    d.qpos[gadr]=np.radians(-10.6); mujoco.mj_forward(m,d)
    V=np.vstack([verts(g) for g in jaw]); return V[np.argmin(V[:,2])]-PAN, d.xmat[gb].reshape(3,3)[:,2]
def ik(target, pitch_deg, roll=5.0, q0=(0,20,10,60), wrist=None, seed=None):
    def r(x):
        x = np.r_[x[:3], wrist] if wrist is not None else x
        p, z = fk([x[0],x[1],x[2],x[3],roll])
        if wrist is not None: return (p-target)*1e3
        tilt = np.degrees(np.arccos(np.clip(z[2],-1,1)))    # jaws point along body -z; 0 = straight down
        return np.r_[(p-target)*1e3, (tilt-pitch_deg)*0.5]
    best=None
    for s in ([seed] if seed is not None else [])+[q0,(0,40,-30,60),(0,0,40,60),(0,50,-50,90)]:
        x0=np.array(s,float)
        if seed is None or s is not seed: x0[0]=-np.degrees(np.arctan2(target[1],target[0]))  # +pan turns toward -y
        if wrist is not None: x0=x0[:3]
        n=len(x0); sol=least_squares(r,x0,bounds=([-110,-100,-96,-95][:n],[110,100,96,95][:n]))
        if best is None or sol.cost<best.cost: best=sol
    x = np.r_[best.x[:3], wrist] if wrist is not None else best.x
    return np.r_[x, roll], best.cost
marks = {"A": np.array([0.193,-0.100,0]), "B": np.array([0.250,0.150,0]), "C": np.array([0.130,0.030,0])}
designs = {

  "multi-pitch (3 marks x pitch 0/30/50)": [(k,p) for k in marks for p in (0,30,50)],
  "multi-pitch (3 marks x pitch 10/30/50, x2 repeats)": [(k,p) for k in marks for p in (10,30,50)]*2,
  "single pitch 30 (3 marks x3 repeats)": [(k,30) for k in marks]*3,
}
TRUE = np.array([0.0,-4.0,6.0,5.0])  # pan,lift,elbow,wrist offsets used to synthesize readings
rng = np.random.default_rng(1)
# old-style: reuse real recorded poses' geometry: synthesize from H0 touch positions with wrist fixed 62.8
import json
pts = json.load(open("/Users/maxchan/so101_stack_game/inputs/table_points_20261019_181950.json"))["points"]
def nominal_old():
    out=[]
    for p in pts:
        q=[p["joints"][j] for j in J]; out.append((p["label"], q))
    return out
def synth(design):
    """returns list of (mark_label, readings_deg[5])"""
    obs=[]
    if design is None:
        for lab,q in nominal_old():
            nominal,_ = fk(np.array(q)); nominal[2]=0      # where the real touch roughly was
            tgt = nominal + np.r_[rng.normal(0,0.002,2), rng.normal(0,0.001)]
            qs,c = ik(tgt, None, wrist=62.8-TRUE[3], seed=np.array(q[:4])-TRUE); assert c<1, (lab,c)
            obs.append((lab, qs + np.r_[TRUE,0] + rng.normal(0,0.2,5)))
        return obs, "z_only"
    for k,pitch in design:
        tgt = marks[k] + np.r_[rng.normal(0,0.002,2), rng.normal(0,0.001)]
        q,c = ik(tgt,pitch)
        if c>1: continue
        obs.append((k, q + np.r_[TRUE,0] + rng.normal(0,0.2,5)))
    return obs, "marks"
def fit(obs, kind):
    labs = sorted(set(l for l,_ in obs)); nm=len(labs)
    def r(x):
        off=np.r_[x[:4],0]; zt=x[4]; res=[]
        P={}
        for l,q in obs:
            p,_=fk(q-off); P.setdefault(l,[]).append(p)
            res.append((p[2]-zt)*1e3/1.5)
        if kind=="marks":
            for i,l in enumerate(labs):
                c=x[5+2*i:7+2*i]
                for p in P[l]: res += list((p[:2]-c)*1e3/2.0)
            # ruler distances between marks (+/-1 mm)
            for a,b in (("A","B"),("B","C"),("A","C")):
                ia,ib=labs.index(a),labs.index(b)
                dist=np.linalg.norm(x[5+2*ia:7+2*ia]-x[5+2*ib:7+2*ib]); true=np.linalg.norm(marks[a][:2]-marks[b][:2])+rng_meas[(a,b)]
                res.append((dist-true)*1e3/1.0)
        return np.array(res)
    x0=np.zeros(5+(2*nm if kind=="marks" else 0))
    if kind=="marks":
        for i,l in enumerate(labs): x0[5+2*i:7+2*i]=marks[l][:2]
    sol=least_squares(r,x0,diff_step=1e-4)
    return sol.x[1:4]
print("reachability (IK cost<1 => reachable; jaw low point on mark at given tilt from vertical):")
for k_ in marks:
    row=[]
    for p_ in (0,10,20,30,40,50,60):
        q_,c_=ik(marks[k_],p_); row.append(f"{p_}:{'ok' if c_<1 else 'no'}" + (f"(wf={q_[3]:.0f})" if c_<1 else ""))
    print("  mark",k_,np.round(marks[k_][:2]*100,1),"cm  ", " ".join(row))
for name,des in designs.items():
    errs=[]
    for trial in range(25):
        rng_meas={k:rng.normal(0,0.001) for k in (("A","B"),("B","C"),("A","C"))}
        obs,kind=synth(des); est=fit(obs,kind); errs.append(est-TRUE[1:])
    errs=np.array(errs)
    print(f"{name:42s} offset error (lift, elbow, wrist): bias {np.round(errs.mean(0),2)}  sd {np.round(errs.std(0),2)} deg")
