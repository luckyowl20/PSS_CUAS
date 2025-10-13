import os
import sys
import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QLabel, QDoubleSpinBox, QFrame, QSlider, QPushButton, QFileDialog, QComboBox,
    QInputDialog, QMessageBox
)
from pyvistaqt import BackgroundPlotter

DEFAULT_SETUP_PATH = os.path.join(os.path.dirname(__file__), "actuator_setups", "default_setups.ini")

# --------------------------- Geometry / Globals (cm) ---------------------------
b_a = 18.0      # distance pivot -> actuator attach along barrel
b_l = 60.0      # barrel length
L_d = 20.0      # distance between actuator base pivots
h   = 18.0      # actuator base height
base_x = -12.0  # actuator base offset from pivot

pivot  = np.array([0.0, 0.0, 0.0])
mount1 = np.array([base_x,  L_d/2, h])
mount2 = np.array([base_x, -L_d/2, h])

# actuator length constraints
L_MIN = 15.5
L_MAX = L_MIN + 5.0
L_HARD_MIN, L_HARD_MAX = 0.0, 150.0
EPS = 0.1

# angle state
ANGLE_SCALE = 20.0
theta = 45.0     # deg (azimuth around +Z)
phi   = 10.0     # deg (tilt from vertical: 0 = +Z)

# ranges for UI (you can tweak)
BASE_X_RANGE = (-60.0, 10.0)
LD_RANGE     = (  5.0, 100.0)
H_RANGE      = (  5.0, 120.0)
B_A_RANGE    = (  5.0,  80.0)

# --------------------------- Math / Builders ---------------------------
def u_from_angles_deg(theta_deg, phi_deg):
    """phi measured from vertical (+Z): 0 up, 90 horizontal."""
    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(phi_deg)
    cph, sph = np.cos(ph), np.sin(ph)
    cth, sth = np.cos(th), np.sin(th)
    return np.array([sph*cth, sph*sth, cph])

def make_barrel(u):
    center = pivot + 0.5*b_l*u
    return pv.Cylinder(center=center, direction=u, radius=2.0, height=b_l, resolution=64)

def make_actuator(p0, p1, radius=0.6):
    return pv.Line(p0, p1).tube(radius=radius, n_sides=24)

# load default setup:


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
        main.addWidget(interactor, stretch=2)

        # Right: Controls panel
        panel = QWidget()
        panel.setMinimumWidth(200)
        panel.setMaximumWidth(800)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(5, 5, 5, 5)
        panel_layout.setSpacing(5)

        # loading configs
        load_row = QHBoxLayout()
        self.load_btn = QPushButton("Load setups…")
        self.save_btn = QPushButton("Save setup…")     # <-- NEW
        self.setup_combo = QComboBox()
        self.setup_combo.setEnabled(False)

        load_row.addWidget(self.load_btn, stretch=0)
        load_row.addWidget(self.save_btn, stretch=0)   # <-- NEW
        load_row.addWidget(self.setup_combo, stretch=1)
        panel_layout.addLayout(load_row)

        # storage
        self.setups = {}
        self.current_setups_path = None   # remember last opened/saved file  <-- NEW

        # connect signals
        self.load_btn.clicked.connect(self._on_load_setups)
        self.save_btn.clicked.connect(self._on_save_setup)       # <-- NEW
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
        self.theta_box = self._spin(-180, 180, 1.0, theta, "deg")
        self.phi_box   = self._spin(   0,  90, 1.0, phi,   "deg")  # tilt from vertical

        self.theta_slider = QSlider(Qt.Horizontal)
        self.theta_slider.setRange(int(-180*ANGLE_SCALE), int(180*ANGLE_SCALE))
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

        self.base_x_box = self._spin(BASE_X_RANGE[0], BASE_X_RANGE[1], 0.5, base_x, "cm")
        self.ld_box     = self._spin(LD_RANGE[0], LD_RANGE[1], 0.5, L_d, "cm")
        self.h_box      = self._spin(H_RANGE[0],  H_RANGE[1],  0.5, h,   "cm")
        self.ba_box     = self._spin(B_A_RANGE[0], B_A_RANGE[1], 0.5, b_a, "cm")

        self.lmin_box = self._spin(L_HARD_MIN, L_HARD_MAX, 0.1, L_MIN, "cm")
        self.lmax_box = self._spin(L_HARD_MIN, L_HARD_MAX, 0.1, L_MAX, "cm")

        form.addRow("θ (yaw / azimuth)", self.theta_box)
        form.addRow("φ (tilt from vertical)", self.phi_box)
        form.addRow("Base X offset", self.base_x_box)
        form.addRow("Mount separation L_d", self.ld_box)
        form.addRow("Mount height h", self.h_box)
        form.addRow("Attachment distance b_a", self.ba_box)
        form.addRow("Actuator MIN", self.lmin_box)
        form.addRow("Actuator MAX", self.lmax_box)

        panel_layout.addLayout(form)
        panel_layout.addStretch(1)
        main.addWidget(panel, stretch=1)

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
        attach0 = pivot + self.ba_box.value() * u0
        barrel0 = make_barrel(u0)
        act1_0 = make_actuator(self.mount1, attach0)
        act2_0 = make_actuator(self.mount2, attach0)

        # Persistent polydata
        self.barrel_poly = barrel0
        self.act1_poly = act1_0
        self.act2_poly = act2_0
        self.pts_poly = pv.PolyData(np.vstack([attach0, self.mount1, self.mount2]))

        # Actors (once)
        self.barrel_actor = self.plotter.add_mesh(self.barrel_poly, color="black", smooth_shading=True)
        self.act1_actor = self.plotter.add_mesh(self.act1_poly, color="blue")
        self.act2_actor = self.plotter.add_mesh(self.act2_poly, color="green")
        self.plotter.add_points(pivot[None, :], color="black", point_size=12, render_points_as_spheres=True)
        self.pts_actor = self.plotter.add_mesh(self.pts_poly, color="green", point_size=12, render_points_as_spheres=True)

        # State
        self.colors_ok = True
        self.attach_last_valid = attach0

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

    # ---------- Core update ----------
    def _apply_pose_update(self):
        # Read UI
        th = self.theta_box.value()
        ph = self.phi_box.value()
        ba = self.ba_box.value()

        # Direction and attach
        u_req = u_from_angles_deg(th, ph)
        attach_req = pivot + ba * u_req

        # Lengths
        L1 = float(np.linalg.norm(attach_req - self.mount1))
        L2 = float(np.linalg.norm(attach_req - self.mount2))

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
            new_barrel = make_barrel(u_req)
            new_act1   = make_actuator(self.mount1, attach_req)
            new_act2   = make_actuator(self.mount2, attach_req)

            self.barrel_poly.shallow_copy(new_barrel)
            self.act1_poly.shallow_copy(new_act1)
            self.act2_poly.shallow_copy(new_act2)

            # points: [attach, mount1, mount2]
            self.pts_poly.points = np.vstack([attach_req, self.mount1, self.mount2])

            # remember last valid
            self.attach_last_valid = attach_req

        else:
            # invalid: tint and keep last valid geometry, but still move mount points
            if self.colors_ok:
                self.act1_actor.prop.color = "crimson"
                self.act2_actor.prop.color = "crimson"
                self.colors_ok = False

            self.pts_poly.points = np.vstack([self.attach_last_valid, self.mount1, self.mount2])

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
        # helper to set a spinbox safely
        def set_box(box, key):
            if key in d and d[key] is not None:
                box.blockSignals(True)
                box.setValue(float(d[key]))
                box.blockSignals(False)

        set_box(self.theta_box, "theta")
        set_box(self.phi_box,   "phi")
        set_box(self.base_x_box, "base_x")
        set_box(self.ld_box,     "L_d")
        set_box(self.h_box,      "h")
        set_box(self.ba_box,     "b_a")
        set_box(self.lmin_box,   "L_MIN")
        set_box(self.lmax_box,   "L_MAX")

        # sync sliders with angles
        self.theta_slider.blockSignals(True)
        self.theta_slider.setValue(int(self.theta_box.value() * ANGLE_SCALE))
        self.theta_slider.blockSignals(False)

        self.phi_slider.blockSignals(True)
        self.phi_slider.setValue(int(self.phi_box.value() * ANGLE_SCALE))
        self.phi_slider.blockSignals(False)

        # recompute mount points + geometry
        self._update_mounts_arrays()
        self._apply_pose_update()

    def _read_setups_ini(self, path):
        """
        Parse an INI file with multiple sections.
        Returns a dict: {section_name: {key: float, ...}, ...}
        Unrecognized keys are ignored; comments allowed.
        """
        import configparser
        cp = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
        # preserve case of keys
        cp.optionxform = str
        if not cp.read(path, encoding="utf-8"):
            raise RuntimeError("Unable to read file or empty file")

        valid_keys = {"theta", "phi", "base_x", "L_d", "h", "b_a", "L_MIN", "L_MAX"}
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
        return {
            "theta": float(self.theta_box.value()),
            "phi":   float(self.phi_box.value()),
            "base_x": float(self.base_x_box.value()),
            "L_d":    float(self.ld_box.value()),
            "h":      float(self.h_box.value()),
            "b_a":    float(self.ba_box.value()),
            "L_MIN":  float(self.lmin_box.value()),
            "L_MAX":  float(self.lmax_box.value()),
        }
    
    def _on_save_setup(self):
        """
        Save the current UI as a named setup (section) into an INI file.
        If a setups file is already loaded, ask whether to save there or to a new file.
        """
        # Ask for a setup name
        name, ok = QInputDialog.getText(self, "Save setup", "Setup name (section):")
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
        import configparser, os

        valid_keys = ["theta", "phi", "base_x", "L_d", "h", "b_a", "L_MIN", "L_MAX"]

        cp = configparser.ConfigParser()
        cp.optionxform = str  # preserve key case

        # If file exists, try to load and then merge/update (so we don't drop unrelated sections)
        if os.path.exists(path):
            try:
                cp.read(path, encoding="utf-8")
            except Exception:
                # If read fails, we’ll just overwrite with new content
                cp = configparser.ConfigParser()
                cp.optionxform = str

        # Write/merge sections
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
