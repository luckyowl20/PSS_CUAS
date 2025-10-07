import numpy as np
import pandas as pd

# analysis for entire camera given spec sheet

class Camera:
    def __init__(self, name: str, Nx: list[int], p_um: float, f_mm: float, p_max: float):
        self.name = name
        if isinstance(Nx, int):
            self.Nx = [Nx]
        else:
            self.Nx = list(Nx)
        self.p_um = p_um
        self.f_mm = f_mm
        self.p_max = p_max

    def __str__(self):
        return f"Camera: {self.name}, Nx: {self.Nx}, p_um: {self.p_um}, f_mm: {self.f_mm}, p_max: {self.p_max}"

    def get_f_px(self, binning=1, downscale=1):
        p_mm = self.p_um / 1000.0
        f_px = (self.f_mm / p_mm) / (binning * downscale)
        return f_px 
    
    def get_hfov(self, Nx):
        f_px = self.get_f_px()
        return np.degrees(2 * np.arctan(Nx/(2*f_px)))
    
    def calc_fps_rates(self, velocity, Nx, range):
        # does not use self.Nx so that we can calculate for different sizes

        f_px = self.get_f_px(self.f_mm, self.p_um)
        hfov = self.get_hfov(Nx)
        # print(f"HFOV: {hfov}")
        hfov_rad = np.radians(hfov)
        omega = velocity / range
        p_dot = omega * (Nx / hfov_rad)
        fps = p_dot / self.p_max
        
        return fps
    
    def calc_exposure_times(self, velocity, Nx, range):
        # does not use self.Nx so that we can calculate for different sizes

        f_px = self.get_f_px()
        hfov = self.get_hfov(Nx)
        # print(f"HFOV: {hfov}")
        hfov_rad = np.radians(hfov)
        omega = velocity / range
        p_dot = omega * (Nx / hfov_rad)
        exposure_time = 1 / p_dot
        return exposure_time * 1000 # convert to ms
    

    def calculate_gsd(self, range_m, object_size_m):
        # gsd = R*p / f
        # Convert to meters
        focal_length_m = self.f_mm / 1000.0
        pixel_size_m = self.p_um * 1e-6

        # GSD formula
        gsd_m = (range_m * pixel_size_m) / focal_length_m

        # Pixels across the object
        apparent_size_px = object_size_m / gsd_m

        return gsd_m, apparent_size_px

    def run_analysis(self, velocity, range, object_size_m, print_results=False):
        results = {}
        for Nx in self.Nx:
            min_fps = self.calc_fps_rates(velocity, Nx, range)
            min_exposure_time = self.calc_exposure_times(velocity, Nx, range)
            gsd, apparent_size_px = self.calculate_gsd(range, object_size_m)
            results[Nx] = (round(min_fps, 2), round(min_exposure_time, 2), round(self.get_hfov(Nx), 2),
                           round(apparent_size_px, 2))
        
        df = pd.DataFrame.from_dict(results, orient='index', columns=['FPS', 'Exposure Time (ms)', 'HFOV', 'Apparent Size (px)'])
        df.index.name = 'Size (px)'

        if print_results:
            print(f"Analysis for: {self.name} at {self.f_mm} mm lens")
            print(df)
        return df