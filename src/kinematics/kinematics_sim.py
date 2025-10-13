import configparser, os, math
import sys
import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QLabel, QDoubleSpinBox, QFrame, QSlider, QPushButton, QFileDialog, QComboBox,
    QInputDialog, QMessageBox, QSplitter
)
from pyvistaqt import BackgroundPlotter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


DEFAULT_SETUP_PATH = os.path.join(os.path.dirname(__file__), "actuator_setups", "default_setups.ini")

# --------------------------- Geometry / Globals (cm) ---------------------------
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
    center = pivot + 0.5 * length * u
    return pv.Cylinder(center=center, direction=u, radius=radius, height=length, resolution=64)

def make_actuator(p0, p1, radius=0.6):
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


# --------------------------- App Widget ---------------------------
class RigApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Barrel Linear Actuator Simulator")
        self.resize(1250, 800)

        # --- Layouts
        main = QHBoxLayout(self)
        self.plotter = BackgroundPlotter(show=False, auto_update=True)
        self.plotter.set_background("white")
        self.plotter.add_axes()

        # Left: VTK Interactor
        interactor = self.plotter.interactor
        interactor.setMinimumWidth(820)
        # main.addWidget(interactor, stretch=2)

        # Right: Controls panel
        panel = QWidget()
        panel.setMinimumWidth(200)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.setSpacing(5)
        panel.setMaximumWidth(2000)

        # dynamic resizer
        self.splitter = QSplitter(Qt.Horizontal)


        # loading configs
        load_row = QHBoxLayout()
        self.load_btn = QPushButton("Load setups…")
        self.save_btn = QPushButton("Save setup…")     
        self.setup_combo = QComboBox()
        self.setup_combo.setEnabled(False)

        load_row.addWidget(self.load_btn, stretch=0)
        load_row.addWidget(self.save_btn, stretch=0)   
        load_row.addWidget(self.setup_combo, stretch=1)
        panel_layout.addLayout(load_row)

        # storage
        self.setups = {}
        self.current_setups_path = None   # remember last opened/saved file  <-- NEW

        # connect signals
        self.load_btn.clicked.connect(self._on_load_setups)
        self.save_btn.clicked.connect(self._on_save_setup)       
        self.setup_combo.currentTextChanged.connect(self._on_select_setup)

        # Status label (top)
        self.status = QLabel("")
        self.status.setWordWrap(False)
        panel_layout.addWidget(self.status)

        # Thin line separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        panel_layout.addWidget(line)

        # Form with controls
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignTop)
        form.setVerticalSpacing(15)

        # ---- Spin boxes (numeric inputs) ----
        self.theta_box = self._spin(-90, 90, 1.0, theta, "deg")
        self.phi_box   = self._spin(   0,  90, 1.0, phi,   "deg")  # tilt from vertical

        self.theta_slider = QSlider(Qt.Horizontal)
        self.theta_slider.setRange(int(-90*ANGLE_SCALE), int(90*ANGLE_SCALE))
        self.theta_slider.setSingleStep(1)   # 0.1 deg per tick
        self.theta_slider.setPageStep(5)     # 0.5 deg per page step
        self.theta_slider.setTracking(True)  # emit while dragging
        self.theta_slider.setValue(int(self.theta_box.value()*ANGLE_SCALE))

        self.phi_slider = QSlider(Qt.Horizontal)
        self.phi_slider.setRange(int(0*ANGLE_SCALE), int(90*ANGLE_SCALE))
        self.phi_slider.setSingleStep(1)     # 0.1 deg per tick
        self.phi_slider.setPageStep(5)       # 0.5 deg per page step
        self.phi_slider.setTracking(True)
        self.phi_slider.setValue(int(self.phi_box.value()*ANGLE_SCALE))

        # Add to the form just below the spin boxes
        form.addRow("θ slider", self.theta_slider)
        form.addRow("φ slider", self.phi_slider)

        self.base_x_box = self._spin(BASE_X_RANGE[0], BASE_X_RANGE[1], 0.1, base_x, "cm")
        self.ld_box     = self._spin(LD_RANGE[0], LD_RANGE[1], 0.1, L_d, "cm")
        self.h_box      = self._spin(H_RANGE[0],  H_RANGE[1],  0.1, h,   "cm")
        self.ba_box     = self._spin(B_A_RANGE[0], B_A_RANGE[1], 0.1, b_a, "cm")

        self.lmin_box = self._spin(L_HARD_MIN, L_HARD_MAX, 0.1, L_MIN, "cm")
        self.lmax_box = self._spin(L_HARD_MIN, L_HARD_MAX, 0.1, L_MAX, "cm")
        self.lug_box = self._spin(-180.0, 180.0, 1.0, LUG_ANGLE_DEG, "deg")
        self.radius_box = self._spin(0.1, 20.0, 0.1, BARREL_RADIUS, "cm")   
        self.length_box = self._spin(5.0, 300.0, 1.0, b_l, "cm")  

        # side mounting lugs
        form.addRow("θ (yaw / azimuth)", self.theta_box)
        form.addRow("φ (tilt from vertical)", self.phi_box)
        form.addRow("Lug clock angle", self.lug_box)
        form.addRow("Base X offset", self.base_x_box)
        form.addRow("Mount separation", self.ld_box)
        form.addRow("Mount height h", self.h_box)
        form.addRow("Attachment height", self.ba_box)
        form.addRow("Actuator MIN", self.lmin_box)
        form.addRow("Actuator MAX", self.lmax_box)

        form.addRow("Barrel radius", self.radius_box)
        form.addRow("Barrel length", self.length_box)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setSingleStep(1)
        self.opacity_slider.setPageStep(5)
        self.opacity_slider.setValue(80) 
        form.addRow("Barrel opacity", self.opacity_slider)

        # analysis button and plot area
        self.analyze_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("Analyze max φ vs θ")
        self.analyze_btn.clicked.connect(self._on_analyze_click)

        self.save_plot_btn = QPushButton("Save Plot")
        self.save_plot_btn.clicked.connect(self._on_save_plot_click)

        self.analyze_layout.addWidget(self.analyze_btn)
        self.analyze_layout.addWidget(self.save_plot_btn)

        self.analyze_btn_row = QWidget()
        self.analyze_btn_row.setLayout(self.analyze_layout)

        self.polar_fig = Figure(figsize=(3.8, 3.8), constrained_layout=True)
        self.polar_ax = self.polar_fig.add_subplot(111, projection="polar")
        self.polar_canvas = FigureCanvas(self.polar_fig)
        self.polar_canvas.setMinimumHeight(800)

        self.polar_info_label = QLabel("No analysis yet.")
        self.polar_info_label.setWordWrap(True)

        self.polar_ax.set_title("Max φ vs θ", va="bottom", fontsize=10)
        self.polar_ax.set_theta_zero_location("E")   # 0° at +X (optional)
        self.polar_ax.set_theta_direction(-1)        # clockwise angles (optional)
        self.polar_ax.set_rmax(90)                   # φ ∈ [0, 90]
        self.polar_ax.set_rticks([0, 30, 60, 90])


        panel_layout.addLayout(form)
        panel_layout.addWidget(self.analyze_btn_row)

        panel_layout.addWidget(self.polar_canvas)
        panel_layout.addWidget(self.polar_info_label)

        self.splitter.setSizes([1100, 360])
        self.splitter.setStretchFactor(0, 4)          # left grows more
        self.splitter.setStretchFactor(1, 1)    

        self.splitter.addWidget(interactor)
        self.splitter.addWidget(panel)
        panel_layout.addStretch(1)
        main.addWidget(self.splitter)

        # --- Scene primitives (created once) ---
        self._build_scene()

        # --- Wire up callbacks ---
        self.theta_box.valueChanged.connect(self._on_angles)
        self.phi_box.valueChanged.connect(self._on_angles)

        self.theta_slider.valueChanged.connect(self._on_theta_slider)
        self.phi_slider.valueChanged.connect(self._on_phi_slider)

        self.base_x_box.valueChanged.connect(self._on_mounts)
        self.ld_box.valueChanged.connect(self._on_mounts)
        self.h_box.valueChanged.connect(self._on_mounts)

        self.ba_box.valueChanged.connect(self._on_ba)

        self.lmin_box.valueChanged.connect(self._on_limits)
        self.lmax_box.valueChanged.connect(self._on_limits)
        self.lug_box.valueChanged.connect(self._on_angles)

        self.radius_box.valueChanged.connect(self._on_barrel_params_changed)
        self.length_box.valueChanged.connect(self._on_barrel_params_changed)
        self.opacity_slider.valueChanged.connect(self._on_opacity_slider)

        # load initial setup
        # Try to load default setups file if it exists
        if os.path.exists(DEFAULT_SETUP_PATH):
            try:
                setups = self._read_setups_ini(DEFAULT_SETUP_PATH)
                self.setups = setups
                self.current_setups_path = DEFAULT_SETUP_PATH
                self.setup_combo.addItems(sorted(self.setups.keys()))
                self.setup_combo.setEnabled(True)

                # Apply default section automatically if available
                if "Default" in self.setups:
                    self._apply_setup(self.setups["Default"])
                    self.setup_combo.setCurrentText("Default")
                    self.status.setText(f"Loaded default setup from {DEFAULT_SETUP_PATH}")
                else:
                    first = next(iter(self.setups))
                    self._apply_setup(self.setups[first])
                    self.setup_combo.setCurrentText(first)
                    self.status.setText(f"Loaded setup [{first}] from {DEFAULT_SETUP_PATH}")
            except Exception as e:
                self.status.setText(f"Failed to load default setups: {e}")
        else:
            self.status.setText(f"No default setups file found at {DEFAULT_SETUP_PATH}")

        # initial refresh
        self._apply_pose_update()

    # ---------- UI helpers ----------
    def _spin(self, lo, hi, step, val, suffix=None):
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setDecimals(3)
        sb.setSingleStep(step)
        sb.setValue(float(val))
        if suffix:
            sb.setSuffix(" " + suffix)
        sb.setAlignment(Qt.AlignRight)
        sb.setKeyboardTracking(True)
        return sb

    # ---------- Scene setup ----------
    def _build_scene(self):
        # Camera
        self.plotter.camera.position = (120, 120, 200)
        self.plotter.camera.focal_point = (0, 0, 0)
        self.plotter.camera.up = (0, 0, 1)

        # axis lines
        y_line = pv.Line(pointa=(0, -200, 0), pointb=(0, 200, 0))
        self.plotter.add_mesh(y_line, color="green", line_width=3, name="y_axis_line")
        z_line = pv.Line(pointa=(0, 0, -200), pointb=(0, 0, 200))
        self.plotter.add_mesh(z_line, color="blue", line_width=3, name="z_axis_line")
        x_line = pv.Line(pointa=(-200, 0, 0), pointb=(200, 0, 0))
        self.plotter.add_mesh(x_line, color="red", line_width=3, name="x_axis_line")

        # XY grid plane
        plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1),
                         i_size=200, j_size=200, i_resolution=20, j_resolution=20)
        self.plotter.add_mesh(plane, style='wireframe', color='lightgray', opacity=0.8, pickable=False)

        # YZ wall at x=0
        Y, Z = np.meshgrid(np.linspace(-2*L_d, 2*L_d, 2), np.linspace(-20, 60, 2))
        X = np.zeros_like(Y)
        wall = pv.StructuredGrid(X, Y, Z).extract_geometry()
        self.plotter.add_mesh(wall, color="lightgray", opacity=0.3, smooth_shading=False)

        # Initial geometry
        self._update_mounts_arrays()
        u0 = u_from_angles_deg(self.theta_box.value(), self.phi_box.value())
        r0 = self.radius_box.value()
        l0 = self.length_box.value()
        psi0 = self.lug_box.value() if hasattr(self, "lug_box") else LUG_ANGLE_DEG

        attach1_0, attach2_0 = fixed_opposite_attach_points(u0, self.ba_box.value(), r0, psi0)
        barrel0 = make_barrel(u0, r0, l0)
        act1_0 = make_actuator(self.mount1, attach1_0)
        act2_0 = make_actuator(self.mount2, attach2_0)

        # Persistent polydata
        self.barrel_poly = barrel0
        self.act1_poly = act1_0
        self.act2_poly = act2_0

        self.pts_poly = pv.PolyData(np.vstack([attach1_0, attach2_0, self.mount1, self.mount2]))

        # Actors (once)
        self.barrel_actor = self.plotter.add_mesh(self.barrel_poly, color="black", smooth_shading=True, opacity=self.opacity_slider.value()/100.0)
        self.act1_actor = self.plotter.add_mesh(self.act1_poly, color="blue")
        self.act2_actor = self.plotter.add_mesh(self.act2_poly, color="green")
        self.plotter.add_points(pivot[None, :], color="black", point_size=12, render_points_as_spheres=True)
        self.pts_actor = self.plotter.add_mesh(self.pts_poly, color="green", point_size=12, render_points_as_spheres=True)

        # store last-valid per-attach
        self.attach1_last_valid = attach1_0
        self.attach2_last_valid = attach2_0

        self.colors_ok = True

        # show window now
        self.plotter.show()

        # do an initial update to ensure HUD is correct
        self._apply_pose_update()

    # ---------- Callbacks ----------
    def _on_angles(self, *_):
        th = self.theta_box.value()
        ph = self.phi_box.value()

        self.theta_slider.blockSignals(True)
        self.theta_slider.setValue(int(th * ANGLE_SCALE))
        self.theta_slider.blockSignals(False)

        self.phi_slider.blockSignals(True)
        self.phi_slider.setValue(int(ph * ANGLE_SCALE))
        self.phi_slider.blockSignals(False)

        # then update the scene
        self._apply_pose_update()

        self._apply_pose_update()

    def _on_mounts(self, *_):
        self._update_mounts_arrays()
        self._apply_pose_update()

    def _on_ba(self, *_):
        self._apply_pose_update()

    def _on_limits(self, *_):
        # keep min < max
        if self.lmin_box.value() > self.lmax_box.value() - EPS:
            self.lmin_box.setValue(self.lmax_box.value() - EPS)
        self._apply_pose_update()

    def _on_theta_slider(self, val_int):
        """Slider (int) -> spin box (float) -> update scene."""
        angle = val_int / ANGLE_SCALE
        # sync spinbox without recursive signal
        self.theta_box.blockSignals(True)
        self.theta_box.setValue(angle)
        self.theta_box.blockSignals(False)
        # apply update once
        self._apply_pose_update()

    def _on_phi_slider(self, val_int):
        angle = val_int / ANGLE_SCALE
        self.phi_box.blockSignals(True)
        self.phi_box.setValue(angle)
        self.phi_box.blockSignals(False)
        self._apply_pose_update()

    def _on_barrel_params_changed(self, *_):
        # Recompute geometry with new radius/length
        self._apply_pose_update()

    def _on_opacity_slider(self, val):
        if hasattr(self, "barrel_actor"):
            self.barrel_actor.prop.opacity = val / 100.0
            self.plotter.render()

    def _on_save_plot_click(self):
        """Export the current polar plot as an image file."""

        # Ask the user for a file path
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot As Image",
            "polar_plot.png",
            "PNG Image (*.png);;JPEG Image (*.jpg);;SVG Vector Image (*.svg);;All Files (*)"
        )

        if not path:
            return  # user canceled

        try:
            # Save the current figure from your canvas
            self.polar_canvas.figure.savefig(path, dpi=300, bbox_inches="tight")
            self.status.setText(f"Plot saved to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Save Failed", f"Could not save plot:\n{e}")

    # ---------- Core update ----------
    def _apply_pose_update(self):
        # Read UI
        th = self.theta_box.value()
        ph = self.phi_box.value()
        ba = self.ba_box.value()
        r = self.radius_box.value()
        l = self.length_box.value()

        # Direction and attach
        u_req = u_from_angles_deg(th, ph)
        psi = self.lug_box.value() if hasattr(self, "lug_box") else LUG_ANGLE_DEG
        attach1_req, attach2_req = fixed_opposite_attach_points(u_req, ba, r, psi)
        # Lengths
        L1 = float(np.linalg.norm(attach1_req - self.mount1))
        L2 = float(np.linalg.norm(attach2_req - self.mount2))

        Lmin = self.lmin_box.value()
        Lmax = self.lmax_box.value()
        within = (Lmin <= L1 <= Lmax) and (Lmin <= L2 <= Lmax)

        # Update status HUD
        self._update_status(th, ph, L1, L2, Lmin, Lmax, within)

        if within:
            # update colors back if needed
            if not self.colors_ok:
                self.act1_actor.prop.color = "blue"
                self.act2_actor.prop.color = "green"
                self.colors_ok = True

            # rebuild meshes in memory and shallow_copy
            new_barrel = make_barrel(u_req, r, l)
            new_act1   = make_actuator(self.mount1, attach1_req)
            new_act2   = make_actuator(self.mount2, attach2_req)

            self.barrel_poly.shallow_copy(new_barrel)
            self.act1_poly.shallow_copy(new_act1)
            self.act2_poly.shallow_copy(new_act2)

            # points: [attach, mount1, mount2]
            self.pts_poly.points = np.vstack([attach1_req, attach2_req, self.mount1, self.mount2])

            # remember last valid
            self.attach1_last_valid = attach1_req
            self.attach2_last_valid = attach2_req

        else:
            # invalid: tint and keep last valid geometry, but still move mount points
            if self.colors_ok:
                self.act1_actor.prop.color = "crimson"
                self.act2_actor.prop.color = "crimson"
                self.colors_ok = False

            self.pts_poly.points = np.vstack([self.attach1_last_valid, self.attach2_last_valid, self.mount1, self.mount2])

        self.barrel_actor.prop.opacity = self.opacity_slider.value() / 100.0
        self.plotter.render()

    def _update_status(self, th, ph, L1, L2, Lmin, Lmax, within):
        if within:
            msg = f"θ={th:.1f}°, φ={ph:.1f}°  |  L1={L1:.2f} cm, L2={L2:.2f} cm"
        else:
            msg = f"⚠ OUT OF RANGE  |  L1={L1:.2f} cm, L2={L2:.2f} cm"
        self.status.setText(msg)

    def _update_mounts_arrays(self):
        # refresh mount1/mount2 from UI values
        bx = self.base_x_box.value()
        ld = self.ld_box.value()
        hh = self.h_box.value()
        self.mount1 = np.array([bx,  ld/2.0, hh])
        self.mount2 = np.array([bx, -ld/2.0, hh])

    # ---------- Analysis ----------
    def _is_pose_feasible(self, theta_deg, phi_deg):
        """
        Returns True if both actuator lengths are within [L_MIN, L_MAX]
        for the given angles, using current UI state (mounts, b_a, radius, lug clock).
        """
        # Ensure mounts reflect current UI
        # (safe even if already up-to-date; cheap to call)
        self._update_mounts_arrays()

        # Read current params from UI
        Lmin = float(self.lmin_box.value())
        Lmax = float(self.lmax_box.value())
        ba   = float(self.ba_box.value())
        r    = float(self.radius_box.value())
        psi  = float(self.lug_box.value()) if hasattr(self, "lug_box") else 0.0

        u = u_from_angles_deg(theta_deg, phi_deg)

        # FIXED opposite lugs on the barrel surface (no sliding)
        attach1, attach2 = fixed_opposite_attach_points(u, ba, r, psi)

        # Distances mount -> corresponding lug
        L1 = float(np.linalg.norm(attach1 - self.mount1))
        L2 = float(np.linalg.norm(attach2 - self.mount2))

        return (Lmin <= L1 <= Lmax) and (Lmin <= L2 <= Lmax)


    def _max_phi_for_theta(self, theta_deg, phi_hi=90.0, coarse_step=1.0, tol=1e-3):
        """
        For a fixed theta, find the maximum feasible phi in [0, phi_hi].
        Strategy: coarse scan to bracket boundary, then binary refine.
        Returns float (degrees) or np.nan if nothing is feasible even at phi=0.
        """

        # 1) Coarse scan
        last_feasible = None
        last_phi = 0.0
        phi = 0.0
        while phi <= phi_hi + 1e-9:
            if self._is_pose_feasible(theta_deg, phi):
                last_feasible = phi
                last_phi = phi
                phi += coarse_step
            else:
                break  # first infeasible; bracket is [last_feasible, phi]
        # If even phi=0 is infeasible:
        if last_feasible is None:
            return float("nan")

        # If everything was feasible up to phi_hi, refine toward the top anyway
        lo = last_feasible
        hi = min(last_phi + coarse_step, phi_hi)

        # 2) Binary refine in [lo, hi]
        # We assume feasibility becomes false after some point (usually true);
        # If it stays feasible, we’ll return hi.
        def feasible(x): return self._is_pose_feasible(theta_deg, x)

        # If we somehow landed in a region where hi is feasible too, try to widen
        if feasible(hi):
            # widen as far as we can (safeguarded)
            while hi < phi_hi - 1e-9 and feasible(min(phi_hi, hi + coarse_step)):
                hi = min(phi_hi, hi + coarse_step)

        # Now refine to boundary
        for _ in range(40):  # enough for sub-1e-6 deg, but we'll stop at tol
            mid = 0.5 * (lo + hi)
            if hi - lo <= tol:
                break
            if feasible(mid):
                lo = mid
            else:
                hi = mid

        return lo


    def analyze_max_phi_curve(self, theta_min=-180.0, theta_max=180.0, theta_step=1.0,
                            phi_hi=90.0, coarse_step=1.0, tol=1e-3):
        """
        Compute max phi for each theta in [theta_min, theta_max] using current UI state.
        Returns a (N, 2) ndarray: columns = [theta_deg, max_phi_deg_or_nan]
        """
        # Ensure mounts reflect current UI before looping
        self._update_mounts_arrays()

        thetas = np.arange(theta_min, theta_max + 1e-9, theta_step)
        out = np.empty((len(thetas), 2), dtype=float)
        for i, th in enumerate(thetas):
            max_phi = self._max_phi_for_theta(th, phi_hi=phi_hi, coarse_step=coarse_step, tol=tol)
            out[i, 0] = th
            out[i, 1] = max_phi

        return out
    
    def _on_analyze_click(self):
        """Compute max φ for each θ and update the polar plot (plus special thetas)."""

        # Analysis range and resolution
        theta_min, theta_max, theta_step = -90.0, 90.0, 1.0
        phi_hi, coarse_step, tol = 90.0, 0.5, 1e-3

        # Full curve for the plot
        curve = self.analyze_max_phi_curve(theta_min=theta_min,
                                           theta_max=theta_max,
                                           theta_step=theta_step,
                                           phi_hi=phi_hi,
                                           coarse_step=coarse_step,
                                           tol=tol)

        # --- Compute max φ at θ = 0, 45, θ_max (clamped to range)
        special_thetas = []
        def clamp_th(x): return max(theta_min, min(theta_max, x))
        for th in (0.0, 45.0, theta_max):
            th_c = clamp_th(th)
            if th_c not in special_thetas:
                special_thetas.append(th_c)

        specials = []
        for th in special_thetas:
            phi_star = self._max_phi_for_theta(th, phi_hi=phi_hi,
                                            coarse_step=coarse_step, tol=tol)
            lbl = f"φ_max(θ={th:.0f}°)={phi_star:.1f}°" if not np.isnan(phi_star) else f"φ_max(θ={th:.0f}°)=—"
            specials.append((th, phi_star, lbl))

        # --- Update plot (with annotations for the three points)
        self._update_polar_plot(curve, annotations=specials)

        # --- Text summary
        valid_mask = ~np.isnan(curve[:, 1])

        if np.any(valid_mask):
            best_idx = np.nanargmax(curve[:, 1])
            best_theta = curve[best_idx, 0]
            best_phi   = curve[best_idx, 1]

            # Build a compact multi-line summary
            lines = [f"Maximum φ over range: {best_phi:.2f}° at θ = {best_theta:.1f}°"]
            for th, ph, _ in specials:
                lines.append(f"θ={th:.0f}° → φ_max = {('%.2f°' % ph) if not np.isnan(ph) else '—'}")
            text = "\n".join(lines)

            self.polar_info_label.setText(text)
        else:
            self.polar_info_label.setText("No feasible φ found.")

    def _collect_analysis_info_text(self, curve, theta_min=-90.0, theta_max=90.0,
                                    phi_hi=90.0, coarse_step=0.5, tol=1e-3):
        th = curve[:, 0]
        ph = curve[:, 1]
        valid = ~np.isnan(ph)

        lines = ["Analysis"]

        # Global max over the curve we just plotted
        if np.any(valid):
            i = np.nanargmax(ph)
            lines.append(f"Global max φ  {ph[i]:6.2f}° @ θ={th[i]:5.1f}°")
        else:
            lines.append("Global max φ      —")

        # Helper to clamp target θ to the analysis range
        clamp = lambda t: max(theta_min, min(theta_max, t))

        # Targets: 0°, 45°, 90° (clamped to range)
        for target in (0.0, 45.0, 90.0):
            tt = clamp(target)
            phi_star = self._max_phi_for_theta(tt, phi_hi=phi_hi,
                                            coarse_step=coarse_step, tol=tol)
            if np.isnan(phi_star):
                lines.append(f"φ_max(θ={tt:3.0f}°)      —")
            else:
                lines.append(f"φ_max(θ={tt:3.0f}°)  {phi_star:6.2f}°")

        # Fixed-width feel to match your Setup box
        return "\n".join(lines)

    def _collect_setup_info_text(self):
        # pull straight from your UI boxes
        vals = {
            "Lug clock angle": f"{self.lug_box.value():.1f} deg",
            "Base X offset":   f"{self.base_x_box.value():.3f} cm",
            "Mount separation":f"{self.ld_box.value():.3f} cm",
            "Mount height h":  f"{self.h_box.value():.3f} cm",
            "Attachment height": f"{self.ba_box.value():.3f} cm",
            "Actuator MIN":    f"{self.lmin_box.value():.3f} cm",
            "Actuator MAX":    f"{self.lmax_box.value():.3f} cm",
            "Barrel radius":   f"{self.radius_box.value():.3f} cm",
            "Barrel length":   f"{self.length_box.value():.3f} cm",
        }
        # tidy, fixed-width block
        lines = [f"{k:<18} {v:>10}" for k, v in vals.items()]

        return "Setup\n" + "\n".join(lines)

    def _update_polar_plot(self, curve, annotations=None):
        """
        curve: (N,2) array with columns [theta_deg, max_phi_deg_or_nan]
        annotations: optional list of (theta_deg, phi_deg, label)
        """

        self.polar_fig.clf()
        ax = self.polar_fig.add_subplot(111, projection="polar")
        self.polar_ax = ax

        # Basic polar setup
        th_deg = curve[:, 0]
        r_phi = curve[:, 1]
        th_rad = np.deg2rad(th_deg)
        ax.set_title("Max φ vs θ", va="bottom", fontsize=10)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        ax.set_thetamin(-90)
        ax.set_thetamax(90)
        ax.set_rmax(90)
        ax.grid(True, alpha=0.4)

        valid = ~np.isnan(r_phi)
        if np.any(valid):
            order = np.argsort(th_rad[valid])
            th_sorted = th_rad[valid][order]
            phi_sorted = r_phi[valid][order]
            ax.plot(th_sorted, phi_sorted, linewidth=2)

            # global max marker
            i = np.nanargmax(phi_sorted)
            ax.plot([th_sorted[i]], [phi_sorted[i]], 'ro')

        # extra annotations (e.g., θ=0, 45, θ_max)
        if annotations:
            for th_d, phi_d, label in annotations:
                if np.isnan(phi_d): continue
                th_r = np.deg2rad(th_d)
                ax.plot([th_r], [phi_d], 'ko', markersize=5)

        # analysis text
        analysis_text = self._collect_analysis_info_text(
            curve,
            theta_min=-90.0, theta_max=90.0,   # match your plotted range
            phi_hi=90.0, coarse_step=0.5, tol=1e-3
        )
        ax.text(1.05, 0.95, analysis_text,      # lower than the Setup box
                transform=ax.transAxes, ha='left', va='top',
                fontsize=9, family='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85),
                clip_on=False)
        
        # setup text
        info = self._collect_setup_info_text()
        ax.text(1.05, 0.55, info,
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=9, family='monospace',
                bbox=dict(boxstyle='round,pad=0.4',
                        facecolor='white', alpha=0.8),
                clip_on=False)
        


        self.polar_canvas.draw_idle()


    # ---------- Setup load/save ----------
    def _on_load_setups(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open setups file", "", "INI files (*.ini);;All files (*)")
        if not path:
            return
        try:
            setups = self._read_setups_ini(path)
        except Exception as e:
            self.status.setText(f"Failed to load setups: {e}")
            return

        self.setups = setups
        self.current_setups_path = path
        self.setup_combo.blockSignals(True)
        self.setup_combo.clear()
        self.setup_combo.addItems(sorted(self.setups.keys()))
        self.setup_combo.setEnabled(True)
        self.setup_combo.blockSignals(False)

        # auto-apply first (or "Default" if present)
        initial_key = "Default" if "Default" in self.setups else self.setup_combo.itemText(0)
        if initial_key:
            self._apply_setup(self.setups[initial_key])

    def _on_select_setup(self, name):
        if not name or name not in self.setups:
            return
        self._apply_setup(self.setups[name])

    def _apply_setup(self, d):
        """
        d: dict possibly containing theta, phi, base_x, L_d, h, b_a, L_MIN, L_MAX (floats)
        Only keys present are applied; others remain as-is.
        """
        def set_box(box, key):
            if key in d and d[key] is not None:
                box.blockSignals(True)
                box.setValue(float(d[key]))
                box.blockSignals(False)

        set_box(self.lug_box,   "lug_angle")
        set_box(self.theta_box, "theta")
        set_box(self.phi_box,   "phi")
        set_box(self.base_x_box, "base_x")
        set_box(self.ld_box,     "L_d")
        set_box(self.h_box,      "h")
        set_box(self.ba_box,     "b_a")
        set_box(self.lmin_box,   "L_MIN")
        set_box(self.lmax_box,   "L_MAX")
        set_box(self.radius_box, "barrel_radius")
        set_box(self.length_box, "barrel_length")

        # sync angle sliders
        self.theta_slider.blockSignals(True)
        self.theta_slider.setValue(int(self.theta_box.value() * ANGLE_SCALE))
        self.theta_slider.blockSignals(False)

        self.phi_slider.blockSignals(True)
        self.phi_slider.setValue(int(self.phi_box.value() * ANGLE_SCALE))
        self.phi_slider.blockSignals(False)

        # recompute mounts + geometry with new values
        self._update_mounts_arrays()
        self._apply_pose_update()

    def _read_setups_ini(self, path):
        """
        Parse an INI file with multiple sections.
        Returns a dict: {section_name: {key: float, ...}, ...}
        Unrecognized keys are ignored; comments allowed.
        """
        cp = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
        cp.optionxform = str  # preserve case

        if not cp.read(path, encoding="utf-8"):
            raise RuntimeError("Unable to read file or empty file")

        valid_keys = set(SETUP_KEYS)
        setups = {}
        for section in cp.sections():
            vals = {}
            for k in valid_keys:
                if cp.has_option(section, k):
                    try:
                        vals[k] = float(cp.get(section, k))
                    except ValueError:
                        raise ValueError(f"Section [{section}] key '{k}' must be a number")
            if vals:
                setups[section] = vals

        if not setups:
            raise RuntimeError("No valid sections/keys found")
        return setups
    
    def _collect_current_setup(self):
        """Grab current UI state as a setup dict of floats."""
        d = {
            "lug_angle": float(self.lug_box.value()),
            "theta":     float(self.theta_box.value()),
            "phi":       float(self.phi_box.value()),
            "base_x":    float(self.base_x_box.value()),
            "L_d":       float(self.ld_box.value()),
            "h":         float(self.h_box.value()),
            "b_a":       float(self.ba_box.value()),
            "L_MIN":     float(self.lmin_box.value()),
            "L_MAX":     float(self.lmax_box.value()),
            "barrel_radius": float(self.radius_box.value()),
            "barrel_length": float(self.length_box.value()),
        }
        return d
    
    def _on_save_setup(self):
        """
        Save the current UI as a named setup (section) into an INI file.
        If a setups file is already loaded, ask whether to save there or to a new file.
        """
        # Ask for a setup name
        name, ok = QInputDialog.getText(self, "Save setup", "Setup name:")
        if not ok or not name.strip():
            return
        name = name.strip()

        # Update in-memory dict
        self.setups[name] = self._collect_current_setup()

        # Refresh combo box
        if self.setup_combo.findText(name) == -1:
            self.setup_combo.addItem(name)
            self.setup_combo.setEnabled(True)
        self.setup_combo.setCurrentText(name)

        # Decide target file
        target_path = None
        if self.current_setups_path:
            # Ask the user: save to the currently loaded file or export to another file?
            msg = QMessageBox(self)
            msg.setWindowTitle("Save setups")
            msg.setText(f"Save to existing file?\n\n{self.current_setups_path}")
            msg.setInformativeText("Choose “Save” to write to this file, or “Export…” to pick a different file.")
            save_btn = msg.addButton("Save", QMessageBox.AcceptRole)
            export_btn = msg.addButton("Export…", QMessageBox.ActionRole)
            cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)
            msg.setDefaultButton(save_btn)
            msg.exec_()
            clicked = msg.clickedButton()

            if clicked == save_btn:
                target_path = self.current_setups_path
            elif clicked == export_btn:
                target_path, _ = QFileDialog.getSaveFileName(self, "Export setups to INI", "rig_setups.ini", "INI files (*.ini);;All files (*)")
            else:
                return
        else:
            # No file loaded yet—ask for a path
            target_path, _ = QFileDialog.getSaveFileName(self, "Export setups to INI", "rig_setups.ini", "INI files (*.ini);;All files (*)")

        if not target_path:
            return

        # Write all setups to the chosen file
        try:
            self._write_setups_ini(target_path, self.setups)
        except Exception as e:
            self.status.setText(f"Failed to save: {e}")
            return

        # Remember path for next time
        self.current_setups_path = target_path
        self.status.setText(f"Saved setup [{name}] to {target_path}")

    def _write_setups_ini(self, path, setups_dict):
        """
        Write all setups to an INI file.
        setups_dict: {section_name: {key: float, ...}, ...}
        Unknown keys are ignored. Sections with no valid keys are skipped.
        """

        valid_keys = SETUP_KEYS  # use the unified list

        cp = configparser.ConfigParser()
        cp.optionxform = str  # preserve key case

        # Merge with existing file (don’t drop unrelated sections)
        if os.path.exists(path):
            try:
                cp.read(path, encoding="utf-8")
            except Exception:
                cp = configparser.ConfigParser()
                cp.optionxform = str

        for section, vals in setups_dict.items():
            if section not in cp.sections():
                cp.add_section(section)
            for k in valid_keys:
                if k in vals and vals[k] is not None:
                    cp.set(section, k, str(float(vals[k])))

        with open(path, "w", encoding="utf-8") as f:
            cp.write(f)



# --------------------------- Main ---------------------------
if __name__ == "__main__":
    pv.global_theme.smooth_shading = True
    app = QApplication(sys.argv)
    w = RigApp()
    w.show()
    sys.exit(app.exec_())
