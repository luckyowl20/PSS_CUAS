"""
Kinematic analysis and calculations for the barrel linear actuator simulator.

This module contains:
- Pose feasibility checking
- Range of motion analysis
- Maximum phi calculations for given theta values
- Analysis curve generation
"""

import numpy as np

# Import geometry functions directly to avoid relative import issues
from .geometry import (
    u_from_angles_deg, fixed_opposite_attach_points, update_mounts_arrays
)


class KinematicsAnalyzer:
    """Handles kinematic analysis and feasibility calculations."""
    
    def __init__(self):
        self.mount1 = None
        self.mount2 = None
        self.lmin = None
        self.lmax = None
        self.ba = None
        self.radius = None
        self.psi = None
    
    def update_parameters(self, mount1, mount2, lmin, lmax, ba, radius, psi):
        """Update the kinematic parameters for analysis."""
        self.mount1 = mount1
        self.mount2 = mount2
        self.lmin = lmin
        self.lmax = lmax
        self.ba = ba
        self.radius = radius
        self.psi = psi
    
    def is_pose_feasible(self, theta_deg, phi_deg):
        """
        Returns True if both actuator lengths are within [L_MIN, L_MAX]
        for the given angles, using current parameters.
        """
        u = u_from_angles_deg(theta_deg, phi_deg)
        
        # FIXED opposite lugs on the barrel surface (no sliding)
        attach1, attach2 = fixed_opposite_attach_points(u, self.ba, self.radius, self.psi)
        
        # Distances mount -> corresponding lug
        L1 = float(np.linalg.norm(attach1 - self.mount1))
        L2 = float(np.linalg.norm(attach2 - self.mount2))
        
        return (self.lmin <= L1 <= self.lmax) and (self.lmin <= L2 <= self.lmax)
    
    def max_phi_for_theta(self, theta_deg, phi_hi=90.0, coarse_step=1.0, tol=1e-3):
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
            if self.is_pose_feasible(theta_deg, phi):
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
        # If it stays feasible, we'll return hi.
        def feasible(x): return self.is_pose_feasible(theta_deg, x)
        
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
        Compute max phi for each theta in [theta_min, theta_max] using current parameters.
        Returns a (N, 2) ndarray: columns = [theta_deg, max_phi_deg_or_nan]
        """
        thetas = np.arange(theta_min, theta_max + 1e-9, theta_step)
        out = np.empty((len(thetas), 2), dtype=float)
        for i, th in enumerate(thetas):
            max_phi = self.max_phi_for_theta(th, phi_hi=phi_hi, coarse_step=coarse_step, tol=tol)
            out[i, 0] = th
            out[i, 1] = max_phi
        
        return out


def calculate_actuator_lengths(theta_deg, phi_deg, mount1, mount2, ba, radius, psi):
    """Calculate actuator lengths for given pose and parameters."""
    u = u_from_angles_deg(theta_deg, phi_deg)
    attach1, attach2 = fixed_opposite_attach_points(u, ba, radius, psi)
    
    L1 = float(np.linalg.norm(attach1 - mount1))
    L2 = float(np.linalg.norm(attach2 - mount2))
    
    return L1, L2, attach1, attach2


def check_pose_feasibility(theta_deg, phi_deg, mount1, mount2, lmin, lmax, ba, radius, psi):
    """Check if a pose is feasible given the constraints."""
    L1, L2, _, _ = calculate_actuator_lengths(theta_deg, phi_deg, mount1, mount2, ba, radius, psi)
    return (lmin <= L1 <= lmax) and (lmin <= L2 <= lmax)
