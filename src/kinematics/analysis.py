"""
Analysis functionality for the barrel linear actuator simulator.

This module contains functions for:
- Running kinematic analysis
- Generating polar plots
- Collecting analysis information
"""

import numpy as np
from matplotlib.figure import Figure


class AnalysisManager:
    """Manages analysis operations for the UI."""
    
    def __init__(self, ui_widget):
        """Initialize with reference to the UI widget."""
        self.ui = ui_widget
    
    def on_analyze_click(self):
        """Compute max φ for each θ and update the polar plot."""
        # Update kinematics analyzer with current parameters
        self.ui._update_mounts_arrays()
        self.ui.kinematics_analyzer.update_parameters(
            self.ui.mount1, self.ui.mount2,
            self.ui.lmin_box.value(), self.ui.lmax_box.value(),
            self.ui.ba_box.value(), self.ui.radius_box.value(),
            self.ui.lug_box.value()
        )
        
        # get ray opacity
        ray_opacity = self.ui.ray_opacity_slider.value() / 100.0

        # Analysis range and resolution
        theta_min, theta_max, theta_step = -90.0, 90.0, 1.0
        phi_hi, coarse_step, tol = 90.0, 0.5, 1e-3
        
        # Full curve for the plot
        curve = self.ui.kinematics_analyzer.analyze_max_phi_curve(
            theta_min=theta_min, theta_max=theta_max, theta_step=theta_step,
            phi_hi=phi_hi, coarse_step=coarse_step, tol=tol
        )
        
        # Create the ROM cone on the 3D view
        self.ui.visualization_manager.update_cone_side_surface_via_max(
            theta_min=-90.0, theta_max=90.0, theta_step=1.0,
            n_side=24, phi_hi=90.0,
            ray_len=250.0, start_at="pivot",
            opacity=ray_opacity,
            cap_samples_angle=181,
            cap_samples_radial=3,
            cap_opacity=ray_opacity
        )
        
        # Compute max φ at θ = 0, 45, θ_max (clamped to range)
        special_thetas = []
        def clamp_th(x): return max(theta_min, min(theta_max, x))
        for th in (0.0, 45.0, theta_max):
            th_c = clamp_th(th)
            if th_c not in special_thetas:
                special_thetas.append(th_c)
        
        specials = []
        for th in special_thetas:
            phi_star = self.ui.kinematics_analyzer.max_phi_for_theta(
                th, phi_hi=phi_hi, coarse_step=coarse_step, tol=tol
            )
            lbl = f"φ_max(θ={th:.0f}°)={phi_star:.1f}°" if not np.isnan(phi_star) else f"φ_max(θ={th:.0f}°)=—"
            specials.append((th, phi_star, lbl))
        
        # Update plot (with annotations for the three points)
        self.update_polar_plot(curve, annotations=specials)
        
        # Text summary
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
            
            self.ui.polar_info_label.setText(text)
        else:
            self.ui.polar_info_label.setText("No feasible φ found.")
    
    def update_polar_plot(self, curve, annotations=None):
        """Update the polar plot with analysis results."""
        self.ui.polar_fig.clf()
        ax = self.ui.polar_fig.add_subplot(111, projection="polar")
        self.ui.polar_ax = ax
        
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
            
            # Global max marker
            i = np.nanargmax(phi_sorted)
            ax.plot([th_sorted[i]], [phi_sorted[i]], 'ro')
        
        # Extra annotations (e.g., θ=0, 45, θ_max)
        if annotations:
            for th_d, phi_d, label in annotations:
                if np.isnan(phi_d): continue
                th_r = np.deg2rad(th_d)
                ax.plot([th_r], [phi_d], 'ko', markersize=5)
        
        # Analysis text
        analysis_text = self.collect_analysis_info_text(
            curve,
            theta_min=-90.0, theta_max=90.0,
            phi_hi=90.0, coarse_step=0.5, tol=1e-3
        )
        ax.text(1.05, 0.95, analysis_text,
                transform=ax.transAxes, ha='left', va='top',
                fontsize=9, family='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85),
                clip_on=False)
        
        # Setup text
        info = self.collect_setup_info_text()
        ax.text(1.05, 0.55, info,
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=9, family='monospace',
                bbox=dict(boxstyle='round,pad=0.4',
                        facecolor='white', alpha=0.8),
                clip_on=False)
        
        self.ui.polar_canvas.draw_idle()
    
    def collect_analysis_info_text(self, curve, theta_min=-90.0, theta_max=90.0,
                                  phi_hi=90.0, coarse_step=0.5, tol=1e-3):
        """Collect analysis information for display."""
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
            phi_star = self.ui.kinematics_analyzer.max_phi_for_theta(
                tt, phi_hi=phi_hi, coarse_step=coarse_step, tol=tol
            )
            if np.isnan(phi_star):
                lines.append(f"φ_max(θ={tt:3.0f}°)      —")
            else:
                lines.append(f"φ_max(θ={tt:3.0f}°)  {phi_star:6.2f}°")
        
        return "\n".join(lines)
    
    def collect_setup_info_text(self):
        """Collect setup information for display."""
        vals = {
            "Lug clock angle": f"{self.ui.lug_box.value():.1f} deg",
            "Base X offset":   f"{self.ui.base_x_box.value():.3f} cm",
            "Mount separation":f"{self.ui.ld_box.value():.3f} cm",
            "Mount height h":  f"{self.ui.h_box.value():.3f} cm",
            "Attachment height": f"{self.ui.ba_box.value():.3f} cm",
            "Actuator MIN":    f"{self.ui.lmin_box.value():.3f} cm",
            "Actuator MAX":    f"{self.ui.lmax_box.value():.3f} cm",
            "Barrel radius":   f"{self.ui.radius_box.value():.3f} cm",
            "Barrel length":   f"{self.ui.length_box.value():.3f} cm",
        }
        lines = [f"{k:<18} {v:>10}" for k, v in vals.items()]
        return "Setup\n" + "\n".join(lines)
