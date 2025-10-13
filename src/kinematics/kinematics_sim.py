import numpy as np
import pyvista as pv

# SORRY ALL UNITS ARE CM

# setup geometry
b_a = 20.32     # distance pivot -> actuator attach along barrel
b_l = 60.0     # barrel length
L_d = 20.0     # distance between actuator base pivots
h   = 20.32     # actuator base height
base_x = -15.0 # actuator base offset from pivot

pivot  = np.array([0.0, 0.0, 0.0])
mount1 = np.array([base_x,  L_d/2, h])
mount2 = np.array([base_x, -L_d/2, h])

# actuator length constraints
L_MIN = 15.5
L_MAX = L_MIN + 5.0

BASE_X_RANGE = (-60.0, 10.0)
LD_RANGE     = (  5.0, 100.0)
H_RANGE      = (  5.0, 120.0)
B_A_RANGE = (5.0, 80.0)   # reasonable range for barrel attachment (cm)

L_RANGE = (0.0, 150.0)   # overall allowed range for either slider
EPS = 0.1                # minimum gap so L_MIN < L_MAX





# theta = azimuth around +Z from +X; phi = elevation from XY plane
def u_from_angles_deg(theta_deg, phi_deg):
    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(phi_deg)
    cph, sph = np.cos(ph), np.sin(ph)
    cth, sth = np.cos(th), np.sin(th)
    return np.array([sph*cth, sph*sth, cph])

# mesh builders
def make_barrel(u):
    center = pivot + 0.5*b_l*u
    return pv.Cylinder(center=center, direction=u, radius=2.0, height=b_l, resolution=64)

def make_actuator(p0, p1, radius=0.6):
    return pv.Line(p0, p1).tube(radius=radius, n_sides=24)

# initial geometry
theta = 0.0
phi   = 0.0

u0        = u_from_angles_deg(theta, phi)
attach0   = pivot + b_a*u0
barrel0   = make_barrel(u0)
act1_0    = make_actuator(mount1, attach0)
act2_0    = make_actuator(mount2, attach0)
L1_0      = np.linalg.norm(attach0 - mount1)
L2_0      = np.linalg.norm(attach0 - mount2)

# pyvista scene setup
pv.set_jupyter_backend(None)
plotter = pv.Plotter(window_size=(900, 700))
plotter.add_axes()

# camera start position
plotter.camera.position = (120, 120, 200)
plotter.camera.focal_point = (0, 0, 0)
plotter.camera.up = (0, 0, 1)


# xy ground plane
plane = pv.Plane(
    center=(0, 0, 0),        
    direction=(0, 0, 1),     
    i_size=200, j_size=200,  
    i_resolution=20, j_resolution=20  
)
plotter.add_mesh(
    plane,
    style='wireframe',       
    color='lightgray',
    opacity=0.8,            
    pickable=False
)


# wall plane
Y, Z = np.meshgrid(np.linspace(-2*L_d, 2*L_d, 2), np.linspace(-20, 60, 2))
X = np.zeros_like(Y)  
wall = pv.StructuredGrid(X, Y, Z).extract_geometry()
plotter.add_mesh(wall, color="lightgray", opacity=0.3, smooth_shading=False)

# Persistent PolyData containers; add actors ONCE
barrel_poly = barrel0
act1_poly   = act1_0
act2_poly   = act2_0
pts_poly    = pv.PolyData(np.vstack([attach0, mount1, mount2]))

barrel_actor = plotter.add_mesh(barrel_poly, color="black", smooth_shading=True)
act1_actor   = plotter.add_mesh(act1_poly,   color="blue")
act2_actor   = plotter.add_mesh(act2_poly,   color="green")
plotter.add_points(pivot[None, :], color="black", point_size=12, render_points_as_spheres=True)
pts_actor = plotter.add_mesh(pts_poly, color="green", point_size=12, render_points_as_spheres=True)

# hud text
status_txt = plotter.add_text(
    f"theta={theta:.1f}°, phi={phi:.1f}° | L1={L1_0:.2f}, L2={L2_0:.2f} | Limits [{L_MIN:.1f}, {L_MAX:.1f}]",
    position='upper_edge',   
    font_size=12,
    name='status'            
)

state = {
    "theta": theta,
    "phi":   phi,
    "u":     u0,
    "attach": attach0,
    "colors_ok": True,  # track whether we tinted actors due to violation
}

def set_actuator_colors(valid: bool):
    """Optional visual cue when a requested pose violates limits."""
    if valid and not state["colors_ok"]:
        act1_actor.prop.color = "blue"
        act2_actor.prop.color = "green"
        state["colors_ok"] = True
    elif not valid and state["colors_ok"]:
        act1_actor.prop.color = "red"
        act2_actor.prop.color = "red"
        state["colors_ok"] = False

# update callback
def update_by_angles(theta_deg, phi_deg):
    # Compute direction & geometry for the *requested* angles
    u_req = u_from_angles_deg(theta_deg, phi_deg)
    attach_req = pivot + b_a*u_req

    # Compute actuator lengths for this pose
    L1 = np.linalg.norm(attach_req - mount1)
    L2 = np.linalg.norm(attach_req - mount2)

    # Check limits
    within = (L_MIN <= L1 <= L_MAX) and (L_MIN <= L2 <= L_MAX)

    # Update status text regardless (so you can see the attempted pose)
    lim_str = f"[{L_MIN:.1f}, {L_MAX:.1f}]"
    if within:
        msg = f"theta={theta_deg:.1f}°, phi={phi_deg:.1f}° | L1={L1:.2f}, L2={L2:.2f} | Limits [{L_MIN:.1f}, {L_MAX:.1f}]"
        plotter.add_text(msg, position='upper_edge', name='status', font_size=12)  
        set_actuator_colors(True)

        # Build new meshes in memory and copy into existing polydata
        new_barrel = make_barrel(u_req)
        new_act1   = make_actuator(mount1, attach_req)
        new_act2   = make_actuator(mount2, attach_req)

        barrel_poly.shallow_copy(new_barrel)
        act1_poly.shallow_copy(new_act1)
        act2_poly.shallow_copy(new_act2)

        # Update points (attach, mount1, mount2)
        pts_poly.points = np.vstack([attach_req, mount1, mount2])

        # Record last *valid* state
        state["theta"], state["phi"] = theta_deg, phi_deg
        state["u"], state["attach"] = u_req, attach_req

        # update points
        pts_poly.points = np.vstack([attach_req, mount1, mount2])

    else:
        # Out-of-bounds: keep geometry at last valid pose, warn user
        warn = []
        if not (L_MIN <= L1 <= L_MAX): warn.append(f"L1={L1:.2f}")
        if not (L_MIN <= L2 <= L_MAX): warn.append(f"L2={L2:.2f}")
        status_txt.SetText(0, f"! OUT OF RANGE: {', '.join(warn)} | Limits {lim_str}  (pose not applied)")
        set_actuator_colors(False)

        # keep last valid geometry
        attach_vis = state["attach"]
        pts_poly.points = np.vstack([attach_vis, mount1, mount2])

    plotter.render()

def on_theta(val):
    global theta
    theta = float(val)
    update_by_angles(theta, phi)

def on_phi(val):
    global phi
    phi = float(val)
    update_by_angles(theta, phi)






def on_base_x(val):
    global base_x, mount1, mount2
    base_x = float(val)
    mount1 = np.array([base_x,  L_d/2.0, h])
    mount2 = np.array([base_x, -L_d/2.0, h])
    update_by_angles(theta, phi)
    plotter.render()

def on_Ld(val):
    global L_d, mount1, mount2
    L_d = float(val)
    mount1 = np.array([base_x,  L_d/2.0, h])
    mount2 = np.array([base_x, -L_d/2.0, h])
    update_by_angles(theta, phi)
    plotter.render()

def on_h(val):
    global h, mount1, mount2
    h = float(val)
    mount1 = np.array([base_x,  L_d/2.0, h])
    mount2 = np.array([base_x, -L_d/2.0, h])
    update_by_angles(theta, phi)
    plotter.render()

def on_Lmin_slider(val: float):
    global L_MIN
    L_MIN = float(val)
    # enforce L_MIN < L_MAX
    if L_MIN > L_MAX - EPS:
        L_MIN = L_MAX - EPS
    update_by_angles(theta, phi)
    plotter.render()

def on_Lmax_slider(val: float):
    global L_MAX
    L_MAX = float(val)
    # enforce L_MIN < L_MAX
    if L_MAX < L_MIN + EPS:
        L_MAX = L_MIN + EPS
    update_by_angles(theta, phi)
    plotter.render()


def on_ba(val):
    global b_a
    b_a = float(val)
    update_by_angles(theta, phi)
    plotter.render()


# sliders

# current actuator spec

# right column
# Add the sliders (placed above your existing base/angle sliders)
s_min = plotter.add_slider_widget(
    on_Lmin_slider, rng=list(L_RANGE), value=L_MIN,
    title="Actuator MIN (cm)",
    pointa=(0.55, 0.30), pointb=(0.95, 0.30),  # top row, right side
    style='modern'
)
s_max = plotter.add_slider_widget(
    on_Lmax_slider, rng=list(L_RANGE), value=L_MAX,
    title="Actuator MAX (cm)",
    pointa=(0.55, 0.40), pointb=(0.95, 0.40),  # just above MIN
    style='modern'
)

phi_slider = plotter.add_slider_widget(
    on_phi, rng=[0.0, 85.0], value=phi,
    title="phi",
    pointa=(0.55, 0.10), pointb=(0.95, 0.10),
    style='modern'
)

separation_slider = plotter.add_slider_widget(
    on_Ld, rng=list(LD_RANGE), value=L_d,
    title="Mount separation L_d (cm)",
    pointa=(0.55, 0.20), pointb=(0.95, 0.20),
    style='modern'
)


# left column
theta_slider = plotter.add_slider_widget(
    on_theta, rng=[-90.0, 90.0], value=theta,
    title="theta",       
    pointa=(0.05, 0.10), pointb=(0.45, 0.10),
    style='modern'
)

base_slider = plotter.add_slider_widget(
    on_base_x, rng=list(BASE_X_RANGE), value=base_x,
    title="Base X offset (cm)",
    pointa=(0.05, 0.20), pointb=(0.45, 0.20),
    style='modern'
)

mount_slider = plotter.add_slider_widget(
    on_h, rng=list(H_RANGE), value=h,
    title="Mount height h (cm)",
    pointa=(0.05, 0.30), pointb=(0.45, 0.30),
    style='modern'
)

slider_ba = plotter.add_slider_widget(
    on_ba, rng=list(B_A_RANGE), value=b_a,
    title="Attachment height b_a (cm)",
    pointa=(0.05, 0.40), pointb=(0.45, 0.40),  
    style='modern'
)

# make sliders slimmer / cleaner (VTK representation)
for s in (s_min, s_max, theta_slider, phi_slider, base_slider, separation_slider, mount_slider, slider_ba):
    rep = s.GetRepresentation()
    rep.SetTubeWidth(0.004)     # track thickness
    rep.SetSliderWidth(0.015)   # handle width
    rep.SetSliderLength(0.02)   # handle length
    rep.SetEndCapWidth(0.010)
    rep.SetEndCapLength(0.006)
    rep.SetTitleHeight(0.015)   # label size
    rep.SetLabelHeight(0.015)

plotter.show()
