import numpy as np
from CameraClassifier import Camera
from dataclasses import dataclass
from typing import Optional, Union, List
from itertools import product
import pandas as pd

@dataclass
class ConstraintSet:
    # Each field may be a single float or a list of floats (multiple constraint values to scan)
    sigma_z: Optional[Union[float, List[float]]] = None          # error in depth estimate
    sigma_d: Optional[Union[float, List[float]]] = None          # error in disparity
    sigma_x: Optional[Union[float, List[float]]] = None          # error in object center
    disparity_min: Optional[Union[float, List[float]]] = None    # minimum disparity between cameras' images
    s_min_px: Optional[Union[float, List[float]]] = None         # apparent size

class StereoCamera(Camera):
    def __init__(self, name, Nx, p_um, f_mm, baseline, constraints: Optional[ConstraintSet] = None, p_max: Optional[float] = 5.0):
        super().__init__(name, Nx, p_um, f_mm, p_max)
        self.baseline = baseline
        # If no ConstraintSet is provided, construct an empty/default one
        self.constraints = constraints if constraints is not None else ConstraintSet()

    def __str__(self):
        return f"Stereo Camera: {self.name}, Nx: {self.Nx}, p_um: {self.p_um}, f_mm: {self.f_mm}, p_max: {self.p_max}, B: {self.baseline}"

    def expected_disparity(self, range_m):
        """calculate expected disparity at a given range, Z"""
        return (self.f_px * self.baseline) / range_m
    
    def depth_from_disparity(self, disparity):
        """calculate the depth (m) from a disparity between images (px)"""
        return (self.f_px * self.baseline) / disparity
    
    def zmax_from_apparent_size(self, object_size, s_min_px):
        """
        Max range to keep an object of determined size S at least s_min_px wide/tall
        """
        return (self.f_px * object_size) / s_min_px
    
    def sigma_d_from_sigma_x(self, sigma_x):
        """
        Disparity standard deviation from per camera center
        sigma_d = sqrt{2} * sigma_x
        sigma_x = standard deviation of detected object center in one image 
        (pixel undertainty from object centroid)
        """
        return np.sqrt(2.0) * sigma_x
    
    def sigmaZ_from_sigma_x(self, sigma_x, range_m):
        """
        Depth standard deviation (meters) from per camera center standard deviation in pixels
        sigma_Z = (Z^2 / (f_px * B)) * sigma_d  with sigma_d = sqrt(2) * sigma_x
        """
        sigma_d = self.sigma_d_from_sigma_x(sigma_x)
        return (range_m**2 / (self.f_px * self.baseline)) * sigma_d
    
    def baseline_for_target_sigmaZ(self, range_m, sigma_z, sigma_x):
        """
        Required camera separation (baseline) (m) to reach target depth and within depth error (sigma_z)
        B = (Z^2 / f_px) * (sqrt(2) * sigma_x / sigma_Z), range = Z here
        """
        return (range_m**2 / self.f_px) * (np.sqrt(2.0) * sigma_x / sigma_z)
    
    # range limiting functions
    # 3 sources of error that bound our detection range
    # 1. Not able to resolve any differences between the two cameras (drone looks like it came from 1 image)
    # 2. Max range where the depth error equals or exceeds our target error in depth estimation
    # 3. Apparent size limit, the object becomes too small in the image 

    def zmax_from_disparity(self, min_disparity):
        return (self.f_px * self.baseline) / min_disparity

    def zmax_from_error(self, sigma_x, sigmaZ_target):
        sigma_d = self.sigma_d_from_sigma_x(sigma_x)
        return np.sqrt( (sigmaZ_target * self.f_px * self.baseline) / sigma_d )

    def zmax_from_apparent_size(self, object_size, s_min_px):
        return (self.f_px * object_size) / s_min_px

    def run_analysis(self, range_m, object_size_m, print_results=False):
        # analysis method that handles scalar or list constraints and sweeps combinations
        # Normalize constraint values to lists (None -> [None])
        def _ensure_list(x):
            if x is None:
                return [None]
            if isinstance(x, (list, tuple, np.ndarray)):
                return [float(v) for v in x]
            return [float(x)]

        sigma_z_list = _ensure_list(self.constraints.sigma_z)
        sigma_x_list = _ensure_list(self.constraints.sigma_x)
        disparity_min_list = _ensure_list(self.constraints.disparity_min)
        s_min_px_list = _ensure_list(self.constraints.s_min_px)

        results = {}
        # iterate over image sizes and all combinations of constraint values
        for Nx in self.Nx:
            hfov = self.get_hfov(Nx)
            _, apparent_size_px = self.calculate_gsd(range_m, object_size_m)

            for sigma_z_val, sigma_x_val, disparity_min_val, s_min_px_val in product(
                sigma_z_list, sigma_x_list, disparity_min_list, s_min_px_list
            ):

                # compute zmax for each constraint (None => non-limiting)
                zmax_disp = self.zmax_from_disparity(disparity_min_val) if disparity_min_val is not None else np.inf
                zmax_error = self.zmax_from_error(sigma_x_val, sigma_z_val) if (sigma_x_val is not None and sigma_z_val is not None) else np.inf
                zmax_app = self.zmax_from_apparent_size(object_size_m, s_min_px_val) if s_min_px_val is not None else np.inf

                z_candidates = {'disparity': zmax_disp, 'depth_error': zmax_error, 'apparent_size': zmax_app}
                z_values = {k: (v if (v is not None and v > 0) else np.inf) for k, v in z_candidates.items()}
                limiting_constraint = min(z_values, key=z_values.get)
                zmax = z_values[limiting_constraint]

                if np.isfinite(zmax):
                    disparity_at_z = self.expected_disparity(zmax)
                    sigma_z_at_z = self.sigmaZ_from_sigma_x(sigma_x_val if sigma_x_val is not None else 0.0, zmax) if sigma_x_val is not None else np.nan
                else:
                    disparity_at_z = 0.0
                    sigma_z_at_z = np.nan

                key = (Nx, sigma_z_val, sigma_x_val, disparity_min_val, s_min_px_val)
                results[key] = (
                    round(zmax, 3),
                    round(disparity_at_z, 3),
                    round(sigma_z_at_z, 4) if not np.isnan(sigma_z_at_z) else np.nan,
                    round(hfov, 2),
                    round(apparent_size_px, 2),
                    limiting_constraint
                )

        df = pd.DataFrame.from_dict(
            results,
            orient='index',
            columns=['Zmax (m)', 'Disparity at Zmax (px)', 'SigmaZ at Zmax (m)', 'HFOV', 'Apparent Size (px)', 'Limiting Factor']
        )
        # set MultiIndex names for clarity
        df.index = pd.MultiIndex.from_tuples(df.index)
        df.index.set_names(['Nx', 'sigma_z', 'sigma_x', 'disparity_min', 's_min_px'], inplace=True)

        if print_results:
            print(f"Analysis for: {self.name} at {self.f_mm} mm lens")
            print(df)
        return df
    
    def baseline_optimization(self, range_m):
        """
        Builds on self.baseline_for_target_sigmaZ and runs this calcualtion for all constraints
        in self.constraints
        """
        results = {}
        for Nx in self.Nx:
            hfov = round(self.get_hfov(Nx), 2)
            for sigma_z, sigma_x in product(self.constraints.sigma_x, self.constraints.sigma_z):
                key = ((Nx, hfov), sigma_z, sigma_x)
                results[key] = round(self.baseline_for_target_sigmaZ(range_m, sigma_z, sigma_x), 3)
        df = pd.DataFrame.from_dict(
            results,
            orient='index',
            columns=['Optimal Baseline (m)']
        )

        df.index = pd.MultiIndex.from_tuples(df.index)
        df.index.set_names(['Nx, HFOV', 'sigma_z', 'sigma_x'], inplace=True)

        return df


