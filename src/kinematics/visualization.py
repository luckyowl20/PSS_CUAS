"""
3D visualization functionality for the barrel linear actuator simulator.

This module contains functions for:
- Updating ROM cone visualization
- Building cone side surfaces
- Building YZ cap surfaces
"""

import numpy as np
import pyvista as pv
from .geometry import pivot, u_from_angles_deg


class VisualizationManager:
    """Manages 3D visualization operations for the UI."""
    
    def __init__(self, ui_widget):
        """Initialize with reference to the UI widget."""
        self.ui = ui_widget
    
    def update_cone_side_surface_via_max(self, theta_min=-90.0, theta_max=90.0, theta_step=1.0,
                                        n_side=24, phi_hi=90.0, ray_len=180.0, start_at="pivot",
                                        opacity=0.35, cap_samples_angle=181, cap_samples_radial=3, cap_opacity=0.35):
        """Update the ROM cone visualization."""
        # Remove old
        if hasattr(self.ui, "cone_surface_actor") and self.ui.cone_surface_actor is not None:
            try: 
                self.ui.plotter.remove_actor(self.ui.cone_surface_actor, reset_camera=False)
            except Exception: 
                pass
            self.ui.cone_surface_actor = None
        
        for name in ("movement_cone_surface", "yz_cap_full"):
            try: 
                self.ui.plotter.remove_actor(name)
            except Exception: 
                pass
        
        # Side surface (smooth bottom)
        pts, faces = self.build_cone_side_surface_via_max(
            theta_min=theta_min, theta_max=theta_max, theta_step=theta_step,
            n_side=n_side, phi_hi=phi_hi, ray_len=ray_len, start_at=start_at
        )
        
        if pts is None:
            self.ui.plotter.render()
            self.ui.status.setText("No feasible side surface in the requested θ window.")
            return
        
        mesh = pv.PolyData(pts, faces)
        self.ui.cone_surface_actor = self.ui.plotter.add_mesh(
            mesh, color="orange", opacity=opacity, name="movement_cone_surface",
            smooth_shading=True, pickable=False,
        )
        
        # Full YZ cap (smooth inner edge, extends into −Y)
        cap_pts, cap_faces = self.build_full_yz_cap_via_max(
            n_angle=cap_samples_angle, n_radial=cap_samples_radial,
            phi_hi=phi_hi, ray_len=ray_len, start_at=start_at
        )
        
        if cap_pts is not None:
            cap = pv.PolyData(cap_pts, cap_faces)
            self.ui.cone_surface_cap_actor = self.ui.plotter.add_mesh(cap, color="red", opacity=cap_opacity,
                                              name="yz_cap_full", smooth_shading=False, pickable=False)
        
        self.ui.plotter.render()
    
    def build_cone_side_surface_via_max(self, theta_min=-90.0, theta_max=90.0, theta_step=1.0,
                                       n_side=24, phi_hi=90.0, ray_len=180.0, start_at="pivot"):
        """Build the side surface of the ROM cone."""
        # Sample rim directions (θ, φ_max(θ))
        thetas = np.arange(theta_min, theta_max + 1e-9, theta_step, dtype=float)
        phi_max = np.array([self.ui.kinematics_analyzer.max_phi_for_theta(float(th), phi_hi=phi_hi)
                            for th in thetas], dtype=float)
        valid = (~np.isnan(phi_max)) & (phi_max > 0.0)
        if not np.any(valid):
            return None, None
        
        thetas = thetas[valid]
        phi_max = phi_max[valid]
        T = thetas.size
        
        # Boundary directions u_bdry(θ)
        u_bdry = np.vstack([u_from_angles_deg(th, ph) for th, ph in zip(thetas, phi_max)])  # (T,3)
        
        # Spherical inner radius (pivot-centered)
        barrel_length = float(self.ui.length_box.value())
        
        # Guard: ensure we can actually render something outside the barrel tip
        if ray_len <= barrel_length + 1e-9:
            ray_len = barrel_length * 1.05  # nudge so we have an outer ring
        
        # Spherical inner radius (pivot-centered) at exactly barrel_length
        s_inner = np.clip(barrel_length / float(ray_len), 0.0, 1.0)
        
        # Build rings so ring 0 is exactly the sphere (smooth rounded bottom)
        S = n_side + 1
        t_vals = np.linspace(0.0, 1.0, S)
        s_grid = s_inner + (1.0 - s_inner) * t_vals[:, None]
        
        # Points: ring-major (s first), then θ
        points = (pivot[None, None, :] + (s_grid[..., None] * ray_len) * u_bdry[None, :, :]).reshape(S*T, 3)
        
        # Faces: connect adjacent rings/columns (no θ wrap — open window)
        def vid(si, ti): return si * T + ti
        faces = []
        for si in range(S - 1):
            for ti in range(T - 1):
                v00 = vid(si,     ti)
                v01 = vid(si,     ti + 1)
                v10 = vid(si + 1, ti)
                v11 = vid(si + 1, ti + 1)
                faces.extend([3, v00, v10, v11])
                faces.extend([3, v00, v11, v01])
        
        return points, np.asarray(faces, dtype=np.int64)
    
    def build_full_yz_cap_via_max(self, theta_pos=+90.0, theta_neg=-90.0,
                                 n_angle=181, n_radial=3, phi_hi=90.0,
                                 ray_len=180.0, start_at="pivot"):
        """Build the YZ cap for the ROM cone."""
        # Max φ on the two YZ edges
        phi_max_pos = self.ui.kinematics_analyzer.max_phi_for_theta(theta_pos, phi_hi=phi_hi)
        phi_max_neg = self.ui.kinematics_analyzer.max_phi_for_theta(theta_neg, phi_hi=phi_hi)
        ok_pos = (isinstance(phi_max_pos, (float, np.floating)) and np.isfinite(phi_max_pos) and phi_max_pos > 0.0)
        ok_neg = (isinstance(phi_max_neg, (float, np.floating)) and np.isfinite(phi_max_neg) and phi_max_neg > 0.0)
        if not (ok_pos or ok_neg):
            return None, None
        
        # Signed in-plane angle α over the YZ plane
        a_min = -float(phi_max_neg) if ok_neg else 0.0
        a_max = +float(phi_max_pos) if ok_pos else 0.0
        if a_max <= a_min + 1e-9:
            return None, None
        
        alphas = np.linspace(a_min, a_max, int(n_angle), dtype=float)
        theta_edges = np.where(alphas >= 0.0, theta_pos, theta_neg)
        phis = np.abs(alphas)
        
        # Directions in YZ plane
        u_cols = np.vstack([u_from_angles_deg(float(th), float(ph)) for th, ph in zip(theta_edges, phis)])
        u_cols[:, 0] = 0.0
        norms = np.linalg.norm(u_cols, axis=1, keepdims=True)
        u_cols = u_cols / np.clip(norms, 1e-12, None)
        N = u_cols.shape[0]
        
        # Spherical inner radius (same as side surface)
        barrel_length = float(self.ui.length_box.value())
        
        # Match side surface: inner edge at exactly barrel_length from pivot
        if ray_len <= barrel_length + 1e-9:
            ray_len = barrel_length * 1.05
        
        s_inner = np.clip(barrel_length / float(ray_len), 0.0, 1.0)
        
        # Build rings so ring 0 is the spherical intersection curve
        S = n_radial + 1
        t_vals = np.linspace(0.0, 1.0, S)[:, None]
        s_grid = s_inner + (1.0 - s_inner) * t_vals
        
        # Points: ring-major in the YZ plane, snapped to x=0
        pts = pivot[None, None, :] + (s_grid[..., None] * ray_len) * u_cols[None, :, :]
        pts = pts.reshape(S * N, 3)
        pts[:, 0] = 0.0  # exact YZ plane
        
        # Faces (no wrap in α)
        def vid(si, ci): return si * N + ci
        faces = []
        for si in range(S - 1):
            for ci in range(N - 1):
                v00 = vid(si,     ci)
                v01 = vid(si,     ci + 1)
                v10 = vid(si + 1, ci)
                v11 = vid(si + 1, ci + 1)
                faces.extend([3, v00, v10, v11])
                faces.extend([3, v00, v11, v01])
        
        return pts, np.asarray(faces, dtype=np.int64)


    # ray opacity slider code
    def _ray_opacity_slider(self, val):
        """Handle cone opacity slider changes"""
        side = getattr(self.ui, "cone_surface_actor", None)
        opacity = val/100.0
        if side is not None:
            try:
                side.GetProperty().SetOpacity(float(opacity))
            except Exception:
                try:
                    side.prop.opacity = float(opacity)
                except Exception:
                    pass

        # Cap actor (store a handle when creating it; see below)
        cap_actor = getattr(self.ui, "cone_surface_cap_actor", None)
        if cap_actor is not None:
            try:
                cap_actor.GetProperty().SetOpacity(float(opacity))
            except Exception:
                try:
                    cap_actor.prop.opacity = float(opacity)
                except Exception:
                    pass

        try:
            self.ui.plotter.render()
        except Exception:
            pass
        return
    