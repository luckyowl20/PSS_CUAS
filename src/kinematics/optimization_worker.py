# worker_opt.py
from PyQt5.QtCore import QObject, pyqtSignal, QRunnable, pyqtSlot
import numpy as np
import math
import traceback, sys

class OptSignals(QObject):
    progress = pyqtSignal(int, float)           # (percent, best_area)
    best_params = pyqtSignal(dict, float)       # (params, best_area)
    error = pyqtSignal(str)
    finished = pyqtSignal(dict)                 # final {"best_params":..., "best_area":...}
    cancelled = pyqtSignal()

class OptimizationWorker(QRunnable):
    """
    Pure-compute optimizer that NEVER touches Qt widgets or pyvista.
    It uses a snapshot 'ctx' and bounds provided by the GUI thread.
    """
    def __init__(self, ctx: dict, mins: np.ndarray, maxs: np.ndarray, cancel_flag: dict):
        super().__init__()
        self.signals = OptSignals()
        self.ctx = ctx              # immutable runtime config & constants
        self.mins = mins
        self.maxs = maxs
        self.cancel_flag = cancel_flag

    @pyqtSlot()
    def run(self):
        try:
            rng = np.random.default_rng(self.ctx.get("rng_seed", 42))
            keys = self.ctx["param_keys"]
            max_samples = int(self.ctx.get("max_samples", 200))
            local_refine_samples = int(self.ctx.get("local_refine_samples", 80))
            theta_window = tuple(self.ctx.get("theta_window", (-90.0, 90.0)))
            theta_step = float(self.ctx.get("theta_step", 1.0))
            phi_hi = float(self.ctx.get("phi_hi", 90.0))

            # local import to avoid circulars; expected in ctx:
            KinematicsAnalyzer = self.ctx["KinematicsAnalyzer"]
            update_mounts_arrays = self.ctx["update_mounts_arrays"]
            fixed_opposite_attach_points = self.ctx["fixed_opposite_attach_points"]

            def rom_area_for_params(vec):
                # Build a fresh analyzer (thread-local) to avoid sharing state with GUI
                analyzer = KinematicsAnalyzer()

                # Unpack constants snapped from GUI
                lmin = self.ctx["lmin"]; lmax = self.ctx["lmax"]
                radius = self.ctx["radius"]; lug = self.ctx["lug"]

                # Params -> mounts / attachment
                params = {k: float(v) for k, v in zip(keys, vec)}
                bx = params["base_x_offset"]
                ld = params["mount_separation"]
                hh = params["mount_height"]
                ba = params["attachment_height"]

                # compute mounts (pure)
                mount1, mount2 = update_mounts_arrays(bx, ld, hh)

                # tell analyzer current geometry/limits
                analyzer.update_parameters(mount1, mount2, lmin, lmax, ba, radius, lug)

                # generate same curve your analysis uses, then integrate
                curve = analyzer.analyze_max_phi_curve(theta_min=theta_window[0],
                                                       theta_max=theta_window[1],
                                                       theta_step=theta_step,
                                                       phi_hi=phi_hi)
                valid = ~np.isnan(curve[:, 1])
                if not valid.any():
                    return -math.inf

                th_rad = np.deg2rad(curve[valid, 0])
                r_rad  = np.deg2rad(curve[valid, 1])
                integrand = 0.5 * (r_rad ** 2)
                return float(np.trapz(integrand, x=th_rad))

            def sample_uniform(n):
                u = rng.random((n, len(keys)))
                return self.mins + (self.maxs - self.mins) * u

            # --- Global search ---
            best_area = -math.inf
            best_vec = None
            global_batch = sample_uniform(max_samples)
            for i, vec in enumerate(global_batch, 1):
                if self.cancel_flag.get("cancel"):
                    self.signals.cancelled.emit()
                    return
                area = rom_area_for_params(vec)
                if area > best_area:
                    best_area, best_vec = area, vec.copy()
                    self.signals.best_params.emit({k: float(v) for k, v in zip(keys, best_vec)}, best_area)
                self.signals.progress.emit(int(100 * i / (max_samples + local_refine_samples)), best_area)

            # --- Local refine ---
            if best_vec is not None:
                span = (self.maxs - self.mins)
                std = span * 0.10
                for j in range(local_refine_samples):
                    if self.cancel_flag.get("cancel"):
                        self.signals.cancelled.emit()
                        return
                    cand = best_vec + rng.normal(scale=std)
                    cand = np.clip(cand, self.mins, self.maxs)
                    area = rom_area_for_params(cand)
                    if area > best_area:
                        best_area, best_vec = area, cand.copy()
                        self.signals.best_params.emit({k: float(v) for k, v in zip(keys, best_vec)}, best_area)
                    self.signals.progress.emit(int(100 * (max_samples + j + 1) / (max_samples + local_refine_samples)), best_area)

            result = {
                "best_params": {k: float(v) for k, v in zip(keys, best_vec)} if best_vec is not None else {},
                "best_area": float(best_area),
            }
            self.signals.finished.emit(result)
        except Exception:
            tb = "".join(traceback.format_exception(*sys.exc_info()))
            self.signals.error.emit(tb)
