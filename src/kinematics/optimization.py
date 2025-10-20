# optimization.py
import math
import numpy as np

class OptimizationManager:
    """
    Maximize the ROM area by tuning 4 mounting parameters.
    Integrates with existing UI/analyzer paths (no new geometry code).
    """

    # Parameter keys in a fixed order (table/UI must match these)
    PARAM_KEYS = (
        "mount_height",        # -> ui.h_box
        "base_x_offset",       # -> ui.base_x_box
        "attachment_height",   # -> ui.ba_box
        "mount_separation",    # -> ui.ld_box
    )

    # Keys for reading bounds from the Optimization tab table
    MIN_KEYS = (
        "min_mount_height",
        "min_base_x_offset",
        "min_attachment_height",
        "min_mount_separation",
    )
    MAX_KEYS = (
        "max_mount_height",
        "max_base_x_offset",
        "max_attachment_height",
        "max_mount_separation",
    )

    def __init__(self, ui):
        self.ui = ui

    # ---------- Cost function (area under your existing curve) ----------
    def compute_rom_area_via_curve(self,
                                   theta_min=-90.0, theta_max=90.0, theta_step=1.0,
                                   phi_hi=90.0, coarse_step=0.5, tol=1e-3) -> float:
        """
        Uses the same analyzer path your analysis uses to generate the curve,
        then computes the polar area:  A = 1/2 ∫ (φ(θ)_rad)^2 dθ.
        """
        # Ask the analyzer for the same curve the plot uses
        curve = self.ui.kinematics_analyzer.analyze_max_phi_curve(
            theta_min=theta_min, theta_max=theta_max, theta_step=theta_step,
            phi_hi=phi_hi, coarse_step=coarse_step, tol=tol
        )
        # curve[:, 0] = θ(deg), curve[:, 1] = φ_max(deg)
        valid = ~np.isnan(curve[:, 1])
        if not valid.any():
            return 0.0
        th_rad = np.deg2rad(curve[valid, 0])
        r_rad  = np.deg2rad(curve[valid, 1])
        integrand = 0.5 * (r_rad ** 2)
        return float(np.trapz(integrand, x=th_rad))

    # ---------- Apply params then score ----------
    def evaluate_params(self, params: dict,
                        theta_window=(-90.0, 90.0), theta_step=1.0, phi_hi=90.0) -> float:
        """
        Apply a candidate 4-tuple to the UI/analyzer and return ROM area score.
        """
        self.ui.apply_kinematic_params(params)  # defined in ui.py (see below)
        return self.compute_rom_area_via_curve(
            theta_min=theta_window[0],
            theta_max=theta_window[1],
            theta_step=theta_step,
            phi_hi=phi_hi,
        )

    # ---------- Bounds from your Optimization table ----------
    def _read_bounds_from_ui(self):
        """
        Pull min/max floats from self.ui._opt_field_edits[...] (populated in build_optimization()).
        Returns (mins, maxs) numpy arrays aligned with PARAM_KEYS, or None if invalid.
        """
        edits = getattr(self.ui, "_opt_field_edits", {})
        def _get_float(key):
            w = edits.get(key)
            if not w: return None
            t = w.text().strip()
            if not t: return None
            try:
                return float(t)
            except ValueError:
                return None

        mins, maxs = [], []
        for kmin, kmax in zip(self.MIN_KEYS, self.MAX_KEYS):
            vmin, vmax = _get_float(kmin), _get_float(kmax)
            if vmin is None or vmax is None or vmax < vmin:
                return None
            mins.append(vmin); maxs.append(vmax)
        return np.array(mins, float), np.array(maxs, float)

    # ---------- A simple, robust optimizer (random + local refine) ----------
    def run_optimization(self,
                         max_samples=200,
                         local_refine_samples=80,
                         theta_window=(-90.0, 90.0),
                         theta_step=1.0,
                         phi_hi=90.0,
                         rng_seed=42):
        """
        1) Random uniform sampling in the box (global).
        2) Gaussian local refinement around the best candidate.
        Returns {"best_params": {...}, "best_area": float}.
        """
        bounds = self._read_bounds_from_ui()
        if bounds is None:
            raise ValueError("Invalid/incomplete bounds in Optimization table.")
        mins, maxs = bounds

        rng = np.random.default_rng(rng_seed)

        def sample_uniform(n):
            u = rng.random((n, len(self.PARAM_KEYS)))
            return mins + (maxs - mins) * u

        # Global search
        best_area = -math.inf
        best_vec = None
        for vec in sample_uniform(max_samples):
            params = {k: float(v) for k, v in zip(self.PARAM_KEYS, vec)}
            area = self.evaluate_params(params,
                                        theta_window=theta_window,
                                        theta_step=theta_step,
                                        phi_hi=phi_hi)
            if area > best_area:
                best_area, best_vec = area, vec

        # Local refinement
        if best_vec is not None:
            span = (maxs - mins)
            std = span * 0.10
            for _ in range(local_refine_samples):
                cand = best_vec + rng.normal(scale=std)
                cand = np.clip(cand, mins, maxs)
                params = {k: float(v) for k, v in zip(self.PARAM_KEYS, cand)}
                area = self.evaluate_params(params,
                                            theta_window=theta_window,
                                            theta_step=theta_step,
                                            phi_hi=phi_hi)
                if area > best_area:
                    best_area, best_vec = area, cand

        best_params = {k: float(v) for k, v in zip(self.PARAM_KEYS, best_vec)} if best_vec is not None else {}
        return {"best_params": best_params, "best_area": float(best_area)}

    # ---------- UI-friendly runner ----------
    def run_from_ui(self):
        """
        Reads bounds from the table, finds best params, applies them to the UI,
        updates status text, and returns the result dict.
        """
        result = self.run_optimization()
        if result["best_params"]:
            self.ui.apply_kinematic_params(result["best_params"])
        if hasattr(self.ui, "status") and self.ui.status:
            self.ui.status.setText(
                f"Best ROM area: {result['best_area']:.6f} rad² | "
                f"h={result['best_params'].get('mount_height'):.3f}, "
                f"bx={result['best_params'].get('base_x_offset'):.3f}, "
                f"ba={result['best_params'].get('attachment_height'):.3f}, "
                f"ld={result['best_params'].get('mount_separation'):.3f}"
            )
        return result
