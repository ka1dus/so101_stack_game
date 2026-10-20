# Phase 1 analysis (exploratory, kept for reproducibility). Run with the sim env:
#   ~/.venvs/so101_stack_sim/bin/python scripts/analysis/phase1_touch_fit.py
import mujoco, numpy as np, json
m = mujoco.MjModel.from_xml_path("/Users/maxchan/so101_stack_game/sim/assets/so101/so101_new_calib.xml")
d = mujoco.MjData(m)
J = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
cal = json.load(open("/Users/maxchan/so101_stack_game/inputs/calibration/my_follower_arm.json"))
pts = json.load(open("/Users/maxchan/so101_stack_game/inputs/table_points_20261019_181950.json"))["points"]
k = (cal["gripper"]["range_max"]-cal["gripper"]["range_min"])*360/4095/100
home_deg = {j: (2047 - (cal[j]["range_min"]+cal[j]["range_max"])/2)*360/4095 for j in J[:5]}
print("LeRobot reading at the half-turn homing pose:", {j: round(v,2) for j,v in home_deg.items()})
def verts(g):
    mid = m.geom_dataid[g]; a = m.mesh_vertadr[mid]; n = m.mesh_vertnum[mid]
    return d.geom_xpos[g] + m.mesh_vert[a:a+n] @ d.geom_xmat[g].reshape(3,3).T
jaw_geoms = [g for g in range(m.ngeom) if m.geom_group[g]==2 and m.body(m.geom_bodyid[g]).name in ("gripper","moving_jaw_so101_v1")]
base = m.body("base").id
def fk(q, offs):
    for i,j in enumerate(J[:5]):
        d.qpos[m.jnt_qposadr[m.joint(j).id]] = np.radians(q[j] - offs.get(j,0.0))
    d.qpos[m.jnt_qposadr[m.joint("gripper").id]] = np.radians(-10.75 + q["gripper"]*k)
    mujoco.mj_forward(m, d)
    V = np.vstack([verts(g) for g in jaw_geoms])
    low = V[np.argmin(V[:,2])]
    return d.site_xpos[m.site("gripperframe").id].copy(), low
# measurement frame: origin on pan axis at table height. pan axis in base frame:
d.qpos[:]=0; mujoco.mj_forward(m,d); pan_xy = d.xanchor[m.joint("shoulder_pan").id][:2].copy()
print("pan axis in base frame (mm):", np.round(pan_xy*1e3,2))
def report(name, offs, use="low"):
    P = {}
    for p in pts:
        tcp, low = fk(p["joints"], offs); P[p["label"]] = (low if use=="low" else tcp) - np.r_[pan_xy, 0]
    z = np.array([v[2] for v in P.values()])
    def ang(a,b,c,dd):
        u=P[b][:2]-P[a][:2]; v=P[dd][:2]-P[c][:2]; return np.degrees(np.arccos(abs(u@v)/np.linalg.norm(u)/np.linalg.norm(v)))
    edge = ang("table_left_edge_near","table_left_edge_far","table_right_edge_near","table_right_edge_far")
    def side(a,b): return np.linalg.norm(P[a][:2]-P[b][:2])*100
    s = dict(sup12=side("supply_corner_1","supply_corner_2"), sup23=side("supply_corner_2","supply_corner_3"),
             sup34=side("supply_corner_3","supply_corner_4"), sup41=side("supply_corner_4","supply_corner_1"),
             stk12=side("stack_corner_1","stack_corner_2"), stk23=side("stack_corner_2","stack_corner_3"),
             stk34=side("stack_corner_3","stack_corner_4"), stk41=side("stack_corner_4","stack_corner_1"))
    print(f"\n== {name} [{use}]  heights mm: range {np.ptp(z)*1e3:.1f}, std {z.std()*1e3:.1f}, mean {z.mean()*1e3:.1f} | table-edge angle {edge:.1f} deg")
    print("   sides cm:", {k_: round(v,1) for k_,v in s.items()}, " (ruler: supply 20.9 x 19.2, stack 21.4 x 14.6)")
    return P, z
P0, z0 = report("H0 plain deg->rad (as measurements.yaml)", {}, "tcp")
P0, z0 = report("H0 plain deg->rad", {}, "low")
for lab,(v) in P0.items(): print(f"   {lab:22s} xyz mm {np.round(v*1e3,0)}")
P1, z1 = report("H1 zero = half-turn homing pose", home_deg, "low")

from scipy.optimize import least_squares
RUL = dict(sup=(20.9,19.2), stk=(21.4,14.6))
def resid(x, joints, w_side=1.0):
    offs = dict(zip(joints, x[:-1])); zt = x[-1]
    P = {p["label"]: fk(p["joints"], offs)[1] - np.r_[pan_xy,0] for p in pts}
    r = [(P[l][2]-zt)*1e3/2.0 for l in P]                  # heights: 2 mm ~ 1 unit
    def side(a,b): return np.linalg.norm(P[a][:2]-P[b][:2])*100
    for pre,(L,W) in (("supply",RUL["sup"]),("stack",RUL["stk"])):
        c=[f"{pre}_corner_{i}" for i in (1,2,3,4)]
        r += [w_side*(side(c[0],c[1])-L)/0.5, w_side*(side(c[2],c[3])-L)/0.5, w_side*(side(c[1],c[2])-W)/0.5, w_side*(side(c[3],c[0])-W)/0.5]  # 5 mm ~ 1 unit
    u=P["table_left_edge_far"][:2]-P["table_left_edge_near"][:2]; v=P["table_right_edge_far"][:2]-P["table_right_edge_near"][:2]
    r.append((np.degrees(np.arccos(abs(u@v)/np.linalg.norm(u)/np.linalg.norm(v)))-90)/2.0)
    return np.array(r)
for joints in (["shoulder_lift","elbow_flex","wrist_flex"], ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex"]):
    x0 = np.zeros(len(joints)+1)
    sol = least_squares(resid, x0, args=(joints,), diff_step=1e-3)
    offs = dict(zip(joints, sol.x[:-1]))
    print("\nFIT", {j: round(v,2) for j,v in offs.items()}, "table z %.1f mm" % (sol.x[-1]*1e3), "cost", round(sol.cost,1))
    J_ = sol.jac; cov = np.linalg.pinv(J_.T@J_)*(2*sol.cost/max(1,len(sol.fun)-len(x0)))
    print("   1-sigma deg:", np.round(np.sqrt(np.diag(cov))[:-1],2))
    report("fitted", offs, "low")
print("\nH0 cost", round(0.5*np.sum(resid(np.zeros(5),["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex"])**2),1))
x = np.r_[[home_deg[j] for j in ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex"]], 0.0]
best = least_squares(lambda zt: resid(np.r_[x[:-1], zt], ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex"]), [0.0]).cost
print("H1 cost (table z free)", round(best,1))
# H1 but wrist_flex offset left at 0
x2 = [home_deg["shoulder_pan"],home_deg["shoulder_lift"],home_deg["elbow_flex"],0.0]
print("H1 without wrist_flex offset cost", round(least_squares(lambda zt: resid(np.r_[x2, zt], ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex"]), [0.0]).cost,1))
report("H1 without wrist offset", dict(zip(["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex"],x2)), "low")

print("\n################ base tilt hypothesis")
def Rxy(rx, ry):
    rx, ry = np.radians(rx), np.radians(ry)
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    return Ry @ Rx
def resid_t(x, joints, use_sides=True):
    offs = dict(zip(joints, x[:len(joints)])); rx, ry, zt = x[len(joints):]
    R = Rxy(rx, ry)
    P = {p["label"]: R @ (fk(p["joints"], offs)[1] - np.r_[pan_xy,0]) for p in pts}
    r = [(P[l][2]-zt)*1e3/2.0 for l in P]
    if use_sides:
        def side(a,b): return np.linalg.norm(P[a][:2]-P[b][:2])*100
        for pre,(L,W) in (("supply",RUL["sup"]),("stack",RUL["stk"])):
            c=[f"{pre}_corner_{i}" for i in (1,2,3,4)]
            r += [(side(c[0],c[1])-L)/0.5, (side(c[2],c[3])-L)/0.5, (side(c[1],c[2])-W)/0.5, (side(c[3],c[0])-W)/0.5]
        u=P["table_left_edge_far"][:2]-P["table_left_edge_near"][:2]; v=P["table_right_edge_far"][:2]-P["table_right_edge_near"][:2]
        r.append((np.degrees(np.arccos(abs(u@v)/np.linalg.norm(u)/np.linalg.norm(v)))-90)/2.0)
    return np.array(r), P
for joints in ([], ["shoulder_lift","elbow_flex","wrist_flex"]):
    sol = least_squares(lambda x: resid_t(x, joints)[0], np.zeros(len(joints)+3), diff_step=1e-3)
    r, P = resid_t(sol.x, joints); z = np.array([P[l][2] for l in P])
    J_ = sol.jac; cov = np.linalg.pinv(J_.T@J_)*(2*sol.cost/max(1,len(sol.fun)-len(sol.x)))
    names = joints+["tilt_x","tilt_y","table_z"]
    print("offsets+tilt:", {n: f"{v:.2f}+/-{s:.2f}" for n,v,s in zip(names, sol.x, np.sqrt(np.diag(cov)))})
    print(f"   heights after tilt: range {np.ptp(z)*1e3:.1f} mm, std {z.std()*1e3:.1f} mm, cost {sol.cost:.1f}")
    for l in P: print(f"     {l:22s} z {P[l][2]*1e3:+5.1f} mm")

print("\n################ wrist_roll offset vs touch data (tilt free, other offsets 0 / free)")
for roll in (0.0, 90.0, -90.0):
    for joints in ([], ["shoulder_lift","elbow_flex","wrist_flex"]):
        def rr(x):
            offs = dict(zip(joints, x[:len(joints)])); offs["wrist_roll"] = roll
            rx, ry, zt = x[len(joints):]; R = Rxy(rx, ry)
            P = {p["label"]: R @ (fk(p["joints"], offs)[1] - np.r_[pan_xy,0]) for p in pts}
            r = [(P[l][2]-zt)*1e3/2.0 for l in P]
            def side(a,b): return np.linalg.norm(P[a][:2]-P[b][:2])*100
            for pre,(L,W) in (("supply",RUL["sup"]),("stack",RUL["stk"])):
                c=[f"{pre}_corner_{i}" for i in (1,2,3,4)]
                r += [(side(c[0],c[1])-L)/0.5, (side(c[2],c[3])-L)/0.5, (side(c[1],c[2])-W)/0.5, (side(c[3],c[0])-W)/0.5]
            u=P["table_left_edge_far"][:2]-P["table_left_edge_near"][:2]; v=P["table_right_edge_far"][:2]-P["table_right_edge_near"][:2]
            r.append((np.degrees(np.arccos(abs(u@v)/np.linalg.norm(u)/np.linalg.norm(v)))-90)/2.0)
            return np.array(r), P
        sol = least_squares(lambda x: rr(x)[0], np.zeros(len(joints)+3), diff_step=1e-3)
        _, P = rr(sol.x); z = np.array([P[l][2] for l in P])
        print(f"roll {roll:+5.0f} free={joints or 'tilt only'}: cost {sol.cost:6.1f}  heights range {np.ptp(z)*1e3:4.1f} std {z.std()*1e3:3.1f} mm  x={np.round(sol.x[:-1],2)}")

print("\n################ with M1: jaw tips 24 mm above the table at home (notes/phase1_measurements.yaml)")
HOME_Q = dict(shoulder_pan=-3.16, shoulder_lift=-87.30, elbow_flex=96.79, wrist_flex=48.35, wrist_roll=8.66, gripper=0.74)
for roll in (0.0, 90.0, -90.0):
    for joints in (["shoulder_lift","elbow_flex","wrist_flex"],):
        def rr(x):
            offs = dict(zip(joints, x[:len(joints)])); offs["wrist_roll"] = roll
            rx, ry, zt = x[len(joints):]; R = Rxy(rx, ry)
            P = {p["label"]: R @ (fk(p["joints"], offs)[1] - np.r_[pan_xy,0]) for p in pts}
            r = [(P[l][2]-zt)*1e3/2.0 for l in P]
            def side(a,b): return np.linalg.norm(P[a][:2]-P[b][:2])*100
            for pre,(L,W) in (("supply",RUL["sup"]),("stack",RUL["stk"])):
                c=[f"{pre}_corner_{i}" for i in (1,2,3,4)]
                r += [(side(c[0],c[1])-L)/0.5, (side(c[2],c[3])-L)/0.5, (side(c[1],c[2])-W)/0.5, (side(c[3],c[0])-W)/0.5]
            u=P["table_left_edge_far"][:2]-P["table_left_edge_near"][:2]; v=P["table_right_edge_far"][:2]-P["table_right_edge_near"][:2]
            r.append((np.degrees(np.arccos(abs(u@v)/np.linalg.norm(u)/np.linalg.norm(v)))-90)/2.0)
            zh = (R @ (fk(HOME_Q, offs)[1] - np.r_[pan_xy,0]))[2]
            r.append((zh - zt - 0.024)*1e3/2.0)            # M1, ruler +/- 2 mm
            return np.array(r), P, zh - zt
        sol = least_squares(lambda x: rr(x)[0], np.zeros(len(joints)+3), diff_step=1e-3)
        _, P, hh = rr(sol.x); z = np.array([P[l][2] for l in P])
        J_ = sol.jac; cov = np.linalg.pinv(J_.T@J_)*(2*sol.cost/max(1,len(sol.fun)-len(sol.x)))
        names = joints+["tilt_x","tilt_y","table_z"]
        print(f"roll {roll:+4.0f}: cost {sol.cost:5.1f} | home tip {hh*1e3:4.1f} mm | touch heights range {np.ptp(z)*1e3:4.1f} std {z.std()*1e3:3.1f} mm")
        print("   ", {n: f"{v:+.2f}+/-{s:.2f}" for n,v,s in zip(names[:-1], sol.x[:-1], np.sqrt(np.diag(cov))[:-1])})
# what H0 (+tilt only) says about home, for reference
for roll in (0.0, 90.0, -90.0):
    zh = fk(HOME_Q, {"wrist_roll": roll})[1][2] - (-0.0047)
    print(f"H0 roll {roll:+4.0f}: home tip {zh*1e3:5.1f} mm above the touch-fitted table (measured 24)")
