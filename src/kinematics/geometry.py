"""
Geometric calculations and constants for the barrel linear actuator simulator.

This module contains:
- Geometric constants and parameters
- 3D vector calculations and transformations
- PyVista mesh generation functions
- Attachment point calculations
"""

import numpy as np
import pyvista as pv

# --------------------------- Geometry Constants (cm) ---------------------------
b_a = 18.0      # distance pivot -> actuator attach along barrel
b_l = 60.0      # barrel length
L_d = 20.0      # distance between actuator base pivots
h   = 18.0      # actuator base height
base_x = -14.5  # actuator base offset from pivot

pivot  = np.array([0.0, 0.0, 0.0])
mount1 = np.array([base_x,  L_d/2, h])
mount2 = np.array([base_x, -L_d/2, h])

# actuator length constraints
L_MIN = 15.5
L_MAX = L_MIN + 5.0
L_HARD_MIN, L_HARD_MAX = 0.0, 150.0
EPS = 0.1

BARREL_RADIUS = 2.0
LUG_ANGLE_DEG = 0.0

# angle state
ANGLE_SCALE = 20.0
theta = 0.0    # deg (azimuth around +Z)
phi   = 0.0     # deg (tilt from vertical: 0 = +Z)

# ranges for UI (you can tweak)
BASE_X_RANGE = (-60.0, 10.0)
LD_RANGE     = (  5.0, 100.0)
H_RANGE      = (  5.0, 120.0)
B_A_RANGE    = (  5.0,  80.0)

# list of params to save
SETUP_KEYS = [
    "lug_angle",
    "theta", "phi",
    "base_x", "L_d", "h", "b_a",
    "L_MIN", "L_MAX",
    "barrel_radius", "barrel_length",
]


# --------------------------- Math / Builders ---------------------------
def u_from_angles_deg(theta_deg, phi_deg):
    """phi measured from vertical (+Z): 0 up, 90 horizontal."""
    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(phi_deg)
    cph, sph = np.cos(ph), np.sin(ph)
    cth, sth = np.cos(th), np.sin(th)
    return np.array([sph*cth, sph*sth, cph])


def make_barrel(u, radius=BARREL_RADIUS, length=b_l):
    """Create a barrel mesh (requires PyVista)."""
    center = pivot + 0.5 * length * u
    return pv.Cylinder(center=center, direction=u, radius=radius, height=length, resolution=64)


def make_actuator(p0, p1, radius=0.6):
    """Create an actuator mesh (requires PyVista)."""
    return pv.Line(p0, p1).tube(radius=radius, n_sides=24)


# attachment point math
def orthonormal_frame(u):
    """
    Given unit vector u (barrel axis), return two unit vectors v, w so that
    {u, v, w} is an orthonormal basis. v, w are perpendicular to u.
    """
    # pick a reference not nearly parallel to u
    if abs(u[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0])
    else:
        a = np.array([0.0, 1.0, 0.0])
    v = np.cross(u, a)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-9:
        # extreme fallback
        a = np.array([0.0, 0.0, 1.0])
        v = np.cross(u, a)
        v_norm = np.linalg.norm(v)
    v = v / v_norm
    w = np.cross(u, v)  # already unit if u, v are unit and orthogonal
    return v, w


def lug_directions(u, psi_deg):
    """
    psi_deg is the 'clock' angle around u. Returns two opposite unit normals n1, n2.
    n1 = cos(psi)*v + sin(psi)*w, n2 = -n1
    """
    v, w = orthonormal_frame(u)
    psi = np.deg2rad(psi_deg)
    n1 = np.cos(psi)*v + np.sin(psi)*w
    n2 = -n1
    return n1, n2


def fixed_opposite_attach_points(u, ba, radius, psi_deg):
    """
    Returns two fixed (opposite) surface points around the barrel centerline point.
    They do NOT depend on where the mounts are, so they won't slide.
    """
    attach_center = pivot + ba * u
    n1, n2 = lug_directions(u, psi_deg)
    p1 = attach_center + radius * n1
    p2 = attach_center + radius * n2
    return p1, p2


def update_mounts_arrays(base_x_val, ld_val, h_val):
    """Update mount positions based on UI values."""
    mount1 = np.array([base_x_val,  ld_val/2.0, h_val])
    mount2 = np.array([base_x_val, -ld_val/2.0, h_val])
    return mount1, mount2
