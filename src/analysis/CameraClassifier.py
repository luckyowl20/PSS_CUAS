import numpy as np
import pandas as pd

# analysis for entire camera given spec sheet

class Camera:
    def __init__(self, name: str, Nx: list[int], p_um: float, f_mm: float, p_max_percentage: float):
        self.name = name
        if isinstance(Nx, int):
            self.Nx = [Nx]
        else:
            self.Nx = list(Nx)
        self.p_um = p_um
        self.f_mm = f_mm
        self.p_max_percentage = p_max_percentage
        self.binning = 1
        self.downscale = 1
        self.f_px = self.get_f_px()
        # f_px is left undefined so we can use binning later when we have that info

    def update_binning(self, new_bin: float, new_downscale: float) -> None:
        self.binning = new_bin
        self.downscale = new_downscale
        self.f_px = self.get_f_px(binning=new_bin, downscale=new_downscale)

    def __str__(self):
        return f"Camera: {self.name}, Nx: {self.Nx}, p_um: {self.p_um}, f_mm: {self.f_mm}, p_max: {self.p_max_percentage}"

    def get_f_px(self, binning: float=1.0, downscale: float=1.0 ) -> float:
        p_mm = self.p_um / 1000.0
        f_px = (self.f_mm / p_mm) / (binning * downscale)
        return f_px 
    
    def pixel_size_m_eff(self):
        # effective pixel size to include binning
        return self.p_um * 1e-6 * self.binning * self.downscale
    
    def get_hfov(self, Nx: float) -> float:
        return np.degrees(2 * np.arctan(Nx/(2*self.f_px)))
    
    def get_hfov_rad(self, Nx: float) -> float:
        return 2*np.arctan(Nx/(2*self.f_px))

    def calc_fps_rates(self, velocity, Nx, range_m, p_max_px):
        # does not use self.Nx so that we can calculate for different sizes
        # print(f"HFOV: {hfov}")
        hfov_rad = self.get_hfov_rad(Nx)
        omega = velocity / range_m
        p_dot = omega * (Nx / hfov_rad)
        fps = p_dot / p_max_px
        
        return fps
    
    def calc_exposure_times(self, velocity, Nx, range_m, p_blur_max_px):
        # does not use self.Nx so that we can calculate for different sizes

        hfov_rad = self.get_hfov_rad(Nx)
        px_per_rad = Nx / hfov_rad
        omega = velocity / range_m
        p_dot = omega * px_per_rad
        return (p_blur_max_px / p_dot) * 1000.0 # convert to ms
    
    def calculate_gsd(self, range_m, object_size_m):
        # gsd = R*p / f
        # Convert to meters
        focal_length_m = self.f_mm / 1000.0
        pixel_size_m = self.pixel_size_m_eff()

        # GSD formula
        gsd_m = (range_m * pixel_size_m) / focal_length_m

        # Pixels across the object
        apparent_size_px = object_size_m / gsd_m

        return gsd_m, apparent_size_px

    def calc_focal_lenghth_for_s_min_px(self, s_min_px, range_m, object_size):
        # f = srp/object size
        p_m = self.p_um * 1e-6 * self.binning * self.downscale  # effective pixel size
        f_m = (s_min_px * range_m * p_m) / object_size
        return f_m * 1000.0  # mm

    def get_p_max(self, apparent_size_px):
        # returns p_max as a percentage of apparent size
        return apparent_size_px * self.p_max_percentage

    # analysis functions for camera class

    def analysis_title(self):
        return f"Analysis for: {self.name} at {self.f_mm} mm lens"

    def run_fps_analysis(self, velocity, range_m, object_size_m, p_blur_max_px, print_results=False):
        results = {}
        for Nx in self.Nx:
            gsd, apparent_size_px = self.calculate_gsd(range_m, object_size_m)
            p_max_px = self.get_p_max(apparent_size_px)
            min_fps = self.calc_fps_rates(velocity, Nx, range_m, p_max_px)
            min_exposure_time = self.calc_exposure_times(velocity, Nx, range_m, p_blur_max_px)
            results[Nx] = (round(min_fps, 2), round(min_exposure_time, 2), round(self.get_hfov(Nx), 2),
                           round(apparent_size_px, 2), round(p_max_px, 2))
        
        df = pd.DataFrame.from_dict(results, orient='index', columns=['FPS', 'Exposure Time (ms)', 'HFOV', 'Apparent Size (px)', 'p_max_px'])
        df.index.name = 'Size (px)'

        if print_results:
            print(self.analysis_title())
            print(df)
        return df
    
    def run_apparent_size_analysis(self, ranges_m, object_size_m, print_results=False):
        results = {}
        for Nx in self.Nx:
            hfov = round(self.get_hfov(Nx), 2)

            for range_m in ranges_m:
                _, apparent_size_px = self.calculate_gsd(range_m, object_size_m)
                key = ((Nx, hfov), range_m)
                results[key] = (round(apparent_size_px, 2))

            
        df = pd.DataFrame.from_dict(results, orient='index', columns=['Apparent Size (px)'])
        df.index = pd.MultiIndex.from_tuples(df.index)
        df.index.set_names(['Nx, HFOV,', 'Range (m)'], inplace=True)

        if print_results:
            print(self.analysis_title())
            print(df)
        return df
    
    def optimize_apparent_size(self, object_size_m, target_s_min_px, ranges_m):
        results = {}
        for Nx in self.Nx:
            hfov = round(self.get_hfov(Nx), 2)

            for range_m in ranges_m:
                focal_length_mm = self.calc_focal_lenghth_for_s_min_px(target_s_min_px, range_m)
                key = ((Nx, hfov), range_m)
                results[key] = round(focal_length_mm, 2)
            
        df = pd.DataFrame.from_dict(results, orient='index', columns=['Focal Length (mm)'])
        df.index = pd.MultiIndex.from_tuples(df.index)
        df.index.set_names(['Nx, HFOV,', 'Range (m)'], inplace=True)

        print(f"Focal Lengths to achieve {target_s_min_px} px across {object_size_m} m object")
        print(df)
        return df