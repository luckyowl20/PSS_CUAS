"""
Setup management functionality for the barrel linear actuator simulator.

This module contains functions for:
- Loading setups from files
- Selecting and applying setups
- Saving current configurations
- Collecting current setup data
"""

from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMessageBox
from .geometry import ANGLE_SCALE


class SetupManager:
    """Manages setup operations for the UI."""
    
    def __init__(self, ui_widget):
        """Initialize with reference to the UI widget."""
        self.ui = ui_widget
    
    def on_load_setups(self):
        """Load setups from a file."""
        path, _ = QFileDialog.getOpenFileName(self.ui, "Open setups file", "", "INI files (*.ini);;All files (*)")
        if not path:
            return
        
        try:
            setups = self.ui.config_manager.load_setups(path)
        except Exception as e:
            self.ui.status.setText(f"Failed to load setups: {e}")
            return
        
        self.ui.setup_combo.blockSignals(True)
        self.ui.setup_combo.clear()
        self.ui.setup_combo.addItems(sorted(setups.keys()))
        self.ui.setup_combo.setEnabled(True)
        self.ui.setup_combo.blockSignals(False)
        
        # Auto-apply first (or "Default" if present)
        initial_key = "Default" if "Default" in setups else self.ui.setup_combo.itemText(0)
        if initial_key:
            self.apply_setup(setups[initial_key])
    
    def on_select_setup(self, name):
        """Handle setup selection from combo box."""
        if not name or name not in self.ui.config_manager.setups:
            return
        self.apply_setup(self.ui.config_manager.setups[name])
    
    def apply_setup(self, d):
        """Apply a setup dictionary to the UI."""
        def set_box(box, key):
            if key in d and d[key] is not None:
                box.blockSignals(True)
                box.setValue(float(d[key]))
                box.blockSignals(False)
        
        set_box(self.ui.lug_box,   "lug_angle")
        set_box(self.ui.theta_box, "theta")
        set_box(self.ui.phi_box,   "phi")
        set_box(self.ui.base_x_box, "base_x")
        set_box(self.ui.ld_box,     "L_d")
        set_box(self.ui.h_box,      "h")
        set_box(self.ui.ba_box,     "b_a")
        set_box(self.ui.lmin_box,   "L_MIN")
        set_box(self.ui.lmax_box,   "L_MAX")
        set_box(self.ui.radius_box, "barrel_radius")
        set_box(self.ui.length_box, "barrel_length")
        
        # Sync angle sliders
        self.ui.theta_slider.blockSignals(True)
        self.ui.theta_slider.setValue(int(self.ui.theta_box.value() * ANGLE_SCALE))
        self.ui.theta_slider.blockSignals(False)
        
        self.ui.phi_slider.blockSignals(True)
        self.ui.phi_slider.setValue(int(self.ui.phi_box.value() * ANGLE_SCALE))
        self.ui.phi_slider.blockSignals(False)
        
        # Recompute mounts + geometry with new values
        self.ui._update_mounts_arrays()
        self.ui._apply_pose_update()
    
    def on_save_setup(self):
        """Save the current UI as a named setup."""
        # Ask for a setup name
        name, ok = QInputDialog.getText(self.ui, "Save setup", "Setup name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        
        # Update in-memory dict
        current_setup = self.collect_current_setup()
        self.ui.config_manager.add_setup(name, current_setup)
        
        # Refresh combo box
        if self.ui.setup_combo.findText(name) == -1:
            self.ui.setup_combo.addItem(name)
            self.ui.setup_combo.setEnabled(True)
        self.ui.setup_combo.setCurrentText(name)
        
        # Decide target file
        target_path = None
        if self.ui.config_manager.current_setups_path:
            # Ask the user: save to the currently loaded file or export to another file?
            msg = QMessageBox(self.ui)
            msg.setWindowTitle("Save setups")
            msg.setText(f"Save to existing file?\n\n{self.ui.config_manager.current_setups_path}")
            msg.setInformativeText("Choose \"Save\" to write to this file, or \"Export...\" to pick a different file.")
            save_btn = msg.addButton("Save", QMessageBox.AcceptRole)
            export_btn = msg.addButton("Export...", QMessageBox.ActionRole)
            cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)
            msg.setDefaultButton(save_btn)
            msg.exec_()
            clicked = msg.clickedButton()
            
            if clicked == save_btn:
                target_path = self.ui.config_manager.current_setups_path
            elif clicked == export_btn:
                target_path, _ = QFileDialog.getSaveFileName(self.ui, "Export setups to INI", "rig_setups.ini", "INI files (*.ini);;All files (*)")
            else:
                return
        else:
            # No file loaded yet—ask for a path
            target_path, _ = QFileDialog.getSaveFileName(self.ui, "Export setups to INI", "rig_setups.ini", "INI files (*.ini);;All files (*)")
        
        if not target_path:
            return
        
        # Write all setups to the chosen file
        try:
            self.ui.config_manager.save_setups(target_path)
        except Exception as e:
            self.ui.status.setText(f"Failed to save: {e}")
            return
        
        self.ui.status.setText(f"Saved setup [{name}] to {target_path}")
    
    def collect_current_setup(self):
        """Collect current UI state as a setup dictionary."""
        d = {
            "lug_angle": float(self.ui.lug_box.value()),
            "theta":     float(self.ui.theta_box.value()),
            "phi":       float(self.ui.phi_box.value()),
            "base_x":    float(self.ui.base_x_box.value()),
            "L_d":       float(self.ui.ld_box.value()),
            "h":         float(self.ui.h_box.value()),
            "b_a":       float(self.ui.ba_box.value()),
            "L_MIN":     float(self.ui.lmin_box.value()),
            "L_MAX":     float(self.ui.lmax_box.value()),
            "barrel_radius": float(self.ui.radius_box.value()),
            "barrel_length": float(self.ui.length_box.value()),
        }
        return d
