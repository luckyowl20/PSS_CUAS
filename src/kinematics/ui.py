"""
User interface components for the barrel linear actuator simulator.

This module contains:
- PyQt5 GUI widgets and layouts
- User interaction handlers
- 3D visualization setup
- Plot generation and display
"""

import os
import numpy as np

# Optional PyVista import for 3D visualization
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None
from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QLabel, QDoubleSpinBox, QFrame, QSlider, QPushButton, QFileDialog, QComboBox,
    QInputDialog, QMessageBox, QSplitter, QScrollArea, QTabWidget, QTableWidget,
    QTableWidgetItem, QLineEdit, QHeaderView, QSizePolicy
)
from PyQt5.QtGui import QDoubleValidator
from pyvistaqt import BackgroundPlotter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Handle both relative and direct imports
from .geometry import (
    pivot, BARREL_RADIUS, b_l, LUG_ANGLE_DEG, ANGLE_SCALE,
    BASE_X_RANGE, LD_RANGE, H_RANGE, B_A_RANGE, L_HARD_MIN, L_HARD_MAX, EPS,
    u_from_angles_deg, make_barrel, make_actuator, fixed_opposite_attach_points,
    update_mounts_arrays
)
from .kinematics import KinematicsAnalyzer, calculate_actuator_lengths
from .config import ConfigManager
from .analysis import AnalysisManager
from .visualization import VisualizationManager
from .setup_manager import SetupManager
from .optimization import OptimizationManager
from .optimization_worker import OptimizationWorker


class RigApp(QWidget):
    """Main application window for the barrel linear actuator simulator."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Barrel Linear Actuator Simulator")
        self.resize(1250, 800)
        
        # Initialize managers
        self.config_manager = ConfigManager()
        self.kinematics_analyzer = KinematicsAnalyzer()
        self.analysis_manager = AnalysisManager(self)
        self.visualization_manager = VisualizationManager(self)
        self.setup_manager = SetupManager(self)
        self.optimization_manager = OptimizationManager(self)

        # optimization threads
        self.thread_pool = QThreadPool()
        self._opt_cancel = {"cancel": False}
        
        # Setup UI
        self._setup_layouts()
        self._setup_controls()
        self._setup_scene()
        self._connect_signals()
        self._load_default_setup()
        
        # Initial refresh
        self._apply_pose_update()
    
    def _setup_layouts(self):
        """Setup the main layout structure."""
        main = QHBoxLayout(self)

        # --- Plotter / left side ---
        self.plotter = BackgroundPlotter(show=False, auto_update=True)
        self.plotter.set_background("white")
        self.plotter.add_axes()

        interactor = self.plotter.interactor
        interactor.setMinimumWidth(820)

        # --- Right side: tabs ---
        self.tabs = QTabWidget()

        # Build tabs via helpers (keeps this method tidy)
        setup_page = self.build_setup_config()
        self.tabs.addTab(setup_page, "Setup Config")

        opt_page = self.build_optimization()
        self.tabs.addTab(opt_page, "Optimization")

        # --- Splitter wiring ---
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setSizes([800, 360])
        self.splitter.setStretchFactor(0, 4)  # left grows more
        self.splitter.setStretchFactor(1, 1)

        self.splitter.addWidget(interactor)
        self.splitter.addWidget(self.tabs)

        main.addWidget(self.splitter)
    
    def build_setup_config(self):
        """
        Build the 'Setup Config' tab contents and return the tab QWidget.
        Exposes:
            self.panel_scroll    - QScrollArea that wraps the setup content
            self._panel_content  - QWidget root inside the scroll
            self._panel_layout   - QVBoxLayout to which the rest of the code adds widgets
            self.form            - QFormLayout with your controls
            self.status          - QLabel for status line
            self.load_btn, self.save_btn, self.setup_combo
        """
        # --- Tab shell ---
        setup_page = QWidget()
        setup_page_layout = QVBoxLayout(setup_page)
        setup_page_layout.setContentsMargins(0, 0, 0, 0)
        setup_page_layout.setSpacing(0)

        # --- Scrollable content ---
        self.panel_scroll = QScrollArea()
        self.panel_scroll.setWidgetResizable(True)

        self._panel_content = QWidget()
        self._panel_content.setMinimumWidth(200)

        self._panel_layout = QVBoxLayout(self._panel_content)
        self._panel_layout.setContentsMargins(12, 12, 12, 12)
        self._panel_layout.setSpacing(5)

        self.panel_scroll.setWidget(self._panel_content)
        setup_page_layout.addWidget(self.panel_scroll)

        # --- Header row: load/save/combo ---
        load_row = QHBoxLayout()
        self.load_btn = QPushButton("Load setups…")
        self.save_btn = QPushButton("Save setup…")
        self.setup_combo = QComboBox()
        self.setup_combo.setEnabled(False)

        load_row.addWidget(self.load_btn, stretch=0)
        load_row.addWidget(self.save_btn, stretch=0)
        load_row.addWidget(self.setup_combo, stretch=1)
        self._panel_layout.addLayout(load_row)

        # --- Status label ---
        self.status = QLabel("")
        self.status.setWordWrap(False)
        self._panel_layout.addWidget(self.status)

        # --- Separator ---
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self._panel_layout.addWidget(line)

        # --- Form with controls ---
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignTop)
        form.setVerticalSpacing(15)
        self.form = form
        self._panel_layout.addLayout(form)

        # --- Analysis section (plot, etc.) ---
        self._setup_analysis_section(self._panel_layout)

        # --- Spacer so content hugs top nicely when short ---
        self._panel_layout.addStretch(1)

        return setup_page

    def build_optimization(self):
        """
        Build the 'Optimization' tab contents and return the tab QWidget.
        Exposes:
            self._opt_scroll, self._opt_content, self._opt_layout
            self.opt_table          - QTableWidget with min/max entries
            self._opt_field_edits   - dict[str, QLineEdit] for easy access
        """
        opt_page = QWidget()
        opt_page_layout = QVBoxLayout(opt_page)
        opt_page_layout.setContentsMargins(0, 0, 0, 0)
        opt_page_layout.setSpacing(0)

        self._opt_scroll = QScrollArea()
        self._opt_scroll.setWidgetResizable(True)

        self._opt_content = QWidget()
        self._opt_layout = QVBoxLayout(self._opt_content)
        self._opt_layout.setContentsMargins(12, 12, 12, 12)
        self._opt_layout.setSpacing(10)

        # --- Table: Parameter | Min | Max ---
        self.opt_table = QTableWidget(5, 3, self._opt_content)
        self.opt_table.setHorizontalHeaderLabels(["Parameter", "Min", "Max"])
        self.opt_table.verticalHeader().setVisible(False)
        self.opt_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.opt_table.setAlternatingRowColors(True)

        # Build rows
        rows = [
            ("Mount height",              "min_mount_height",        "max_mount_height"),
            ("Base X offset",             "min_base_x_offset",       "max_base_x_offset"),
            ("Attachment height",         "min_attachment_height",   "max_attachment_height"),
            ("Mount separation distance", "min_mount_separation",    "max_mount_separation"),
            ("Lug offset angle",           "min_lug_angle",           "max_lug_angle")
        ]

        # Keep references to line edits for easy access
        self._opt_field_edits = {}

        # Numeric validator (floats allowed; you can set bounds if you like)
        validator = QDoubleValidator(self)
        validator.setNotation(QDoubleValidator.StandardNotation)

        for r, (label, min_key, max_key) in enumerate(rows):
            # Column 0: parameter label (read-only item)
            item = QTableWidgetItem(label)
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            self.opt_table.setItem(r, 0, item)

            # Column 1: Min (QLineEdit)
            min_edit = QLineEdit(self.opt_table)
            min_edit.setValidator(validator)
            min_edit.setPlaceholderText("min")
            self.opt_table.setCellWidget(r, 1, min_edit)
            self._opt_field_edits[min_key] = min_edit

            # Column 2: Max (QLineEdit)
            max_edit = QLineEdit(self.opt_table)
            max_edit.setValidator(validator)
            max_edit.setPlaceholderText("max")
            self.opt_table.setCellWidget(r, 2, max_edit)
            self._opt_field_edits[max_key] = max_edit

        self.opt_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        def _fit_opt_table_height():
            h = (self.opt_table.horizontalHeader().height()
                + self.opt_table.verticalHeader().length()
                + 2 * self.opt_table.frameWidth())
            self.opt_table.setMaximumHeight(h)

        # Call once after populating:
        _fit_opt_table_height()
        self._opt_layout.addWidget(self.opt_table)

        # Keep content top-aligned when short
        self._opt_layout.addStretch(1)

        self._opt_scroll.setWidget(self._opt_content)
        opt_page_layout.addWidget(self._opt_scroll)

        # hook up the run button
        btn_row = QHBoxLayout()
        self.run_opt_btn = QPushButton("Run Optimization")
        self.cancel_opt_btn = QPushButton("Cancel")
        self.cancel_opt_btn.setEnabled(False)
        btn_row.addWidget(self.run_opt_btn)
        btn_row.addWidget(self.cancel_opt_btn)
        self._opt_layout.addLayout(btn_row)

        self.opt_progress = QLabel("")  # light-weight progress readout
        self._opt_layout.addWidget(self.opt_progress)

        def _run_opt_clicked():
            # try:
            self._opt_cancel["cancel"] = False
            self.run_opt_btn.setEnabled(False)
            self.cancel_opt_btn.setEnabled(True)
            self.status.setText("Optimization running...")

            # snapshot GUI state and bounds on GUI thread
            ctx = self.optimization_manager.snapshot_context()
            mins, maxs = self.optimization_manager.read_bounds_arrays()

            worker = OptimizationWorker(ctx, mins, maxs, self._opt_cancel)

            # connect signals -> GUI thread slots
            worker.signals.progress.connect(self._on_opt_progress)
            worker.signals.best_params.connect(self._on_opt_best_params)
            worker.signals.error.connect(self._on_opt_error)
            worker.signals.finished.connect(self._on_opt_finished)
            worker.signals.cancelled.connect(self._on_opt_cancelled)

            self.thread_pool.start(worker)
            # except Exception as e:
            # self.status.setText(f"Optimization error: {e}")

        def _cancel_opt_clicked():
            self._opt_cancel["cancel"] = True

        self.run_opt_btn.clicked.connect(_run_opt_clicked)
        self.cancel_opt_btn.clicked.connect(_cancel_opt_clicked)

        # return the full page
        return opt_page

    def _setup_controls(self):
        """Setup all the control widgets."""
        # Angle controls
        self.theta_box = self._spin(-90, 90, 1.0, 0.0, "deg")
        self.phi_box   = self._spin(   0,  90, 1.0, 0.0,   "deg")  # tilt from vertical
        
        # Angle sliders
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
        
        # Add sliders to form
        self.form.addRow("θ slider", self.theta_slider)
        self.form.addRow("φ slider", self.phi_slider)
        
        # Geometry controls
        self.base_x_box = self._spin(BASE_X_RANGE[0], BASE_X_RANGE[1], 0.1, -14.5, "cm")
        self.ld_box     = self._spin(LD_RANGE[0], LD_RANGE[1], 0.1, 20.0, "cm")
        self.h_box      = self._spin(H_RANGE[0],  H_RANGE[1],  0.1, 18.0,   "cm")
        self.ba_box     = self._spin(B_A_RANGE[0], B_A_RANGE[1], 0.1, 18.0, "cm")
        
        # Actuator limits
        self.lmin_box = self._spin(L_HARD_MIN, L_HARD_MAX, 0.1, 15.5, "cm")
        self.lmax_box = self._spin(L_HARD_MIN, L_HARD_MAX, 0.1, 20.5, "cm")
        
        # Other parameters
        self.lug_box = self._spin(-180.0, 180.0, 1.0, LUG_ANGLE_DEG, "deg")
        self.radius_box = self._spin(0.1, 20.0, 0.1, BARREL_RADIUS, "cm")   
        self.length_box = self._spin(5.0, 300.0, 1.0, b_l, "cm")  
        
        # Barrel Opacity control
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setSingleStep(1)
        self.opacity_slider.setPageStep(5)
        self.opacity_slider.setValue(80) 

        # Ray opacity slider
        self.ray_opacity_slider = QSlider(Qt.Horizontal)
        self.ray_opacity_slider.setRange(0, 100)
        self.ray_opacity_slider.setSingleStep(1)
        self.ray_opacity_slider.setPageStep(5)
        self.ray_opacity_slider.setValue(80) 
        
        # Add controls to form
        self.form.addRow("θ (yaw / azimuth)", self.theta_box)
        self.form.addRow("φ (tilt from vertical)", self.phi_box)
        self.form.addRow("Lug clock angle", self.lug_box)
        self.form.addRow("Base X offset", self.base_x_box)
        self.form.addRow("Mount separation", self.ld_box)
        self.form.addRow("Mount height h", self.h_box)
        self.form.addRow("Attachment height", self.ba_box)
        self.form.addRow("Actuator MIN", self.lmin_box)
        self.form.addRow("Actuator MAX", self.lmax_box)
        self.form.addRow("Barrel radius", self.radius_box)
        self.form.addRow("Barrel length", self.length_box)
        self.form.addRow("Barrel opacity", self.opacity_slider)
        self.form.addRow("ROM cone opacity", self.ray_opacity_slider)
    
    def _setup_analysis_section(self, panel_layout):
        """Setup the analysis section with plot and buttons."""
        # Analysis button and plot area
        self.analyze_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("Analyze max φ vs θ")
        self.save_plot_btn = QPushButton("Save Plot")
        
        self.analyze_layout.addWidget(self.analyze_btn)
        self.analyze_layout.addWidget(self.save_plot_btn)
        
        self.analyze_btn_row = QWidget()
        self.analyze_btn_row.setLayout(self.analyze_layout)
        
        # Polar plot
        self.polar_fig = Figure(figsize=(3.8, 3.8), constrained_layout=True)
        self.polar_ax = self.polar_fig.add_subplot(111, projection="polar")
        self.polar_canvas = FigureCanvas(self.polar_fig)
        self.polar_canvas.setMinimumHeight(600)
        self.polar_canvas.setMinimumWidth(600)
        
        self.polar_info_label = QLabel("No analysis yet.")
        self.polar_info_label.setWordWrap(True)
        
        self.polar_ax.set_title("Max φ vs θ", va="bottom", fontsize=10)
        self.polar_ax.set_theta_zero_location("E")   # 0° at +X (optional)
        self.polar_ax.set_theta_direction(-1)        # clockwise angles (optional)
        self.polar_ax.set_rmax(90)                   # φ ∈ [0, 90]
        self.polar_ax.set_rticks([0, 30, 60, 90])
        
        panel_layout.addWidget(self.analyze_btn_row)
        panel_layout.addWidget(self.polar_canvas)
        panel_layout.addWidget(self.polar_info_label)
    
    def _setup_scene(self):
        """Setup the 3D scene and initial geometry."""
        # Camera
        self.plotter.camera.position = (120, 120, 200)
        self.plotter.camera.focal_point = (0, 0, 0)
        self.plotter.camera.up = (0, 0, 1)
        
        # ROM cone for analysis
        self.cone_actor = None
        self.cone_surface_actor = None
        
        # Axis lines
        y_line = pv.Line(pointa=(0, -200, 0), pointb=(0, 200, 0))
        self.plotter.add_mesh(y_line, color="green", line_width=3, name="y_axis_line")
        z_line = pv.Line(pointa=(0, 0, -200), pointb=(0, 0, 200))
        self.plotter.add_mesh(z_line, color="blue", line_width=3, name="z_axis_line")
        x_line = pv.Line(pointa=(-200, 0, 0), pointb=(200, 0, 0))
        self.plotter.add_mesh(x_line, color="red", line_width=3, name="x_axis_line")
        
        # YZ wall at x=0
        Y, Z = np.meshgrid(np.linspace(-40, 40, 2), np.linspace(-20, 60, 2))
        X = np.zeros_like(Y)
        wall = pv.StructuredGrid(X, Y, Z).extract_geometry()
        self.plotter.add_mesh(wall, color="lightgray", opacity=0.3, smooth_shading=False)
        
        # Initial geometry
        self._update_mounts_arrays()
        u0 = u_from_angles_deg(self.theta_box.value(), self.phi_box.value())
        r0 = self.radius_box.value()
        l0 = self.length_box.value()
        psi0 = self.lug_box.value()
        
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
        
        # Store last-valid per-attach
        self.attach1_last_valid = attach1_0
        self.attach2_last_valid = attach2_0
        
        self.colors_ok = True
        
        # Show window now
        self.plotter.show()
        
        # Do an initial update to ensure HUD is correct
        self._apply_pose_update()
    
    def _connect_signals(self):
        """Connect all the signal handlers."""
        # Config buttons
        self.load_btn.clicked.connect(self.setup_manager.on_load_setups)
        self.save_btn.clicked.connect(self.setup_manager.on_save_setup)       
        self.setup_combo.currentTextChanged.connect(self.setup_manager.on_select_setup)
        
        # Angle controls
        self.theta_box.valueChanged.connect(self._on_angles)
        self.phi_box.valueChanged.connect(self._on_angles)
        self.theta_slider.valueChanged.connect(self._on_theta_slider)
        self.phi_slider.valueChanged.connect(self._on_phi_slider)
        
        # Geometry controls
        self.base_x_box.valueChanged.connect(self._on_mounts)
        self.ld_box.valueChanged.connect(self._on_mounts)
        self.h_box.valueChanged.connect(self._on_mounts)
        self.ba_box.valueChanged.connect(self._on_ba)
        
        # Limits
        self.lmin_box.valueChanged.connect(self._on_limits)
        self.lmax_box.valueChanged.connect(self._on_limits)
        self.lug_box.valueChanged.connect(self._on_angles)
        
        # Barrel parameters
        self.radius_box.valueChanged.connect(self._on_barrel_params_changed)
        self.length_box.valueChanged.connect(self._on_barrel_params_changed)
        self.opacity_slider.valueChanged.connect(self._on_opacity_slider)
        
        # ROM ray parameters
        self.ray_opacity_slider.valueChanged.connect(self.visualization_manager._ray_opacity_slider)

        # Analysis
        self.analyze_btn.clicked.connect(self.analysis_manager.on_analyze_click)
        self.save_plot_btn.clicked.connect(self._on_save_plot_click)
    
    def _load_default_setup(self):
        """Load default setup if available."""
        default_setup_path = os.path.join(os.path.dirname(__file__), "actuator_setups", "default_setups.ini")
        
        if os.path.exists(default_setup_path):
            try:
                setups = self.config_manager.load_setups(default_setup_path)
                self.setup_combo.addItems(sorted(setups.keys()))
                self.setup_combo.setEnabled(True)
                
                # Apply default section automatically if available
                if "Default" in setups:
                    self.setup_manager.apply_setup(setups["Default"])
                    self.setup_combo.setCurrentText("Default")
                    self.status.setText(f"Loaded default setup from {default_setup_path}")
                else:
                    first = next(iter(setups))
                    self.setup_manager.apply_setup(setups[first])
                    self.setup_combo.setCurrentText(first)
                    self.status.setText(f"Loaded setup [{first}] from {default_setup_path}")
            except Exception as e:
                self.status.setText(f"Failed to load default setups: {e}")
        else:
            self.status.setText(f"No default setups file found at {default_setup_path}")
    
    # ---------- UI helpers ----------
    def _spin(self, lo, hi, step, val, suffix=None):
        """Create a spin box widget."""
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
    
    # ---------- Callbacks ----------
    def _on_angles(self, *_):
        """Handle angle changes."""
        th = self.theta_box.value()
        ph = self.phi_box.value()
        
        self.theta_slider.blockSignals(True)
        self.theta_slider.setValue(int(th * ANGLE_SCALE))
        self.theta_slider.blockSignals(False)
        
        self.phi_slider.blockSignals(True)
        self.phi_slider.setValue(int(ph * ANGLE_SCALE))
        self.phi_slider.blockSignals(False)
        
        self._apply_pose_update()
    
    def _on_mounts(self, *_):
        """Handle mount parameter changes."""
        self._update_mounts_arrays()
        self._apply_pose_update()
    
    def _on_ba(self, *_):
        """Handle attachment height changes."""
        self._apply_pose_update()
    
    def _on_limits(self, *_):
        """Handle actuator limit changes."""
        # Keep min < max
        if self.lmin_box.value() > self.lmax_box.value() - EPS:
            self.lmin_box.setValue(self.lmax_box.value() - EPS)
        self._apply_pose_update()
    
    def _on_theta_slider(self, val_int):
        """Handle theta slider changes."""
        angle = val_int / ANGLE_SCALE
        self.theta_box.blockSignals(True)
        self.theta_box.setValue(angle)
        self.theta_box.blockSignals(False)
        self._apply_pose_update()
    
    def _on_phi_slider(self, val_int):
        """Handle phi slider changes."""
        angle = val_int / ANGLE_SCALE
        self.phi_box.blockSignals(True)
        self.phi_box.setValue(angle)
        self.phi_box.blockSignals(False)
        self._apply_pose_update()
    
    def _on_barrel_params_changed(self, *_):
        """Handle barrel parameter changes."""
        self._apply_pose_update()
    
    def _on_opacity_slider(self, val):
        """Handle opacity slider changes."""
        if hasattr(self, "barrel_actor"):
            self.barrel_actor.prop.opacity = val / 100.0
            self.plotter.render()
    
    def _on_save_plot_click(self):
        """Export the current polar plot as an image file."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot As Image",
            "polar_plot.png",
            "PNG Image (*.png);;JPEG Image (*.jpg);;SVG Vector Image (*.svg);;All Files (*)"
        )
        
        if not path:
            return  # user canceled
        
        try:
            self.polar_canvas.figure.savefig(path, dpi=300, bbox_inches="tight")
            self.status.setText(f"Plot saved to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Save Failed", f"Could not save plot:\n{e}")
    
    # ---------- Core update ----------
    def _apply_pose_update(self):
        """Update the 3D scene with current parameters."""
        # Read UI
        th = self.theta_box.value()
        ph = self.phi_box.value()
        ba = self.ba_box.value()
        r = self.radius_box.value()
        l = self.length_box.value()
        
        # Direction and attach
        u_req = u_from_angles_deg(th, ph)
        psi = self.lug_box.value()
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
            # Update colors back if needed
            if not self.colors_ok:
                self.act1_actor.prop.color = "blue"
                self.act2_actor.prop.color = "green"
                self.colors_ok = True
            
            # Rebuild meshes in memory and shallow_copy
            new_barrel = make_barrel(u_req, r, l)
            new_act1   = make_actuator(self.mount1, attach1_req)
            new_act2   = make_actuator(self.mount2, attach2_req)
            
            self.barrel_poly.shallow_copy(new_barrel)
            self.act1_poly.shallow_copy(new_act1)
            self.act2_poly.shallow_copy(new_act2)
            
            # Points: [attach, mount1, mount2]
            self.pts_poly.points = np.vstack([attach1_req, attach2_req, self.mount1, self.mount2])
            
            # Remember last valid
            self.attach1_last_valid = attach1_req
            self.attach2_last_valid = attach2_req
        
        else:
            # Invalid: tint and keep last valid geometry, but still move mount points
            if self.colors_ok:
                self.act1_actor.prop.color = "crimson"
                self.act2_actor.prop.color = "crimson"
                self.colors_ok = False
            
            self.pts_poly.points = np.vstack([self.attach1_last_valid, self.attach2_last_valid, self.mount1, self.mount2])
        
        self.barrel_actor.prop.opacity = self.opacity_slider.value() / 100.0
        self.plotter.render()
    
    def _update_status(self, th, ph, L1, L2, Lmin, Lmax, within):
        """Update the status display."""
        if within:
            msg = f"θ={th:.1f}°, φ={ph:.1f}°  |  L1={L1:.2f} cm, L2={L2:.2f} cm"
        else:
            msg = f"⚠ OUT OF RANGE  |  L1={L1:.2f} cm, L2={L2:.2f} cm"
        self.status.setText(msg)
    
    def _update_mounts_arrays(self):
        """Update mount positions from UI values."""
        bx = self.base_x_box.value()
        ld = self.ld_box.value()
        hh = self.h_box.value()
        self.mount1, self.mount2 = update_mounts_arrays(bx, ld, hh)
    
    # optimization callbacks and functions
    def apply_kinematic_params(self, params: dict):
        """
        Update the 4 design variables through the same path your analysis uses.
        Keys in params: base_x_offset, mount_separation, mount_height, attachment_height
        """
        if "base_x_offset" in params:
            self.base_x_box.setValue(float(params["base_x_offset"]))
        if "mount_separation" in params:
            self.ld_box.setValue(float(params["mount_separation"]))
        if "mount_height" in params:
            self.h_box.setValue(float(params["mount_height"]))
        if "attachment_height" in params:
            self.ba_box.setValue(float(params["attachment_height"]))
        if "lug_offset_angle" in params:
            self.lug_box.setValue(float(params["lug_offset_angle"]))

        # Recompute mounts from UI values (updates self.mount1, self.mount2)
        self._update_mounts_arrays()

        # Push full parameter set into analyzer, same way your analysis does
        self.kinematics_analyzer.update_parameters(
            self.mount1, self.mount2,
            self.lmin_box.value(), self.lmax_box.value(),  # actuator min/max
            self.ba_box.value(), self.radius_box.value(),
            self.lug_box.value()
        )
    
    # callbacks -- move these later into some sort of utils file
    def _on_opt_progress(self, percent, best_area):
        self.opt_progress.setText(f"Progress: {percent}% | best area: {best_area:.6f} rad²")

    def _on_opt_best_params(self, params, best_area):
        """
        Called often; runs on the GUI thread. It MAY update widgets/scene safely.
        Throttle if desired (e.g., update only every Nth call) for smoothness.
        """
        # Apply candidate to GUI & re-run analysis to refresh the polar plot / cone
        self.apply_kinematic_params(params)
        # optional: refresh your polar plot/3D viz
        self.analysis_manager.on_analyze_click()

    def _on_opt_error(self, tb_text):
        self.status.setText("Optimization crashed. See log.")
        # You could also show a dialog, or log tb_text somewhere visible:
        print(tb_text)

    def _on_opt_finished(self, result):
        self.run_opt_btn.setEnabled(True)
        self.cancel_opt_btn.setEnabled(False)
        self.status.setText("Optimization finished.")
        if result["best_params"]:
            # Apply and draw the final best
            self.apply_kinematic_params(result["best_params"])
            self.analysis_manager.on_analyze_click()

    def _on_opt_cancelled(self):
        self.run_opt_btn.setEnabled(True)
        self.cancel_opt_btn.setEnabled(False)
        self.status.setText("Optimization cancelled.")
