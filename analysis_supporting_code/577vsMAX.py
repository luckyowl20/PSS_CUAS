import math
from dataclasses import dataclass
from typing import Tuple, List

# ---------------- core math ----------------

def rectilinear_fov(sensor_w_mm: float, sensor_h_mm: float, f_mm: float) -> Tuple[float, float]:
    """HFOV,VFOV (deg) for a rectilinear lens."""
    hfov = 2 * math.degrees(math.atan((sensor_w_mm / 2) / f_mm))
    vfov = 2 * math.degrees(math.atan((sensor_h_mm / 2) / f_mm))
    return hfov, vfov

def crop_fov(hfov_full_deg: float, vfov_full_deg: float,
             crop_ratio_w: float, crop_ratio_h: float) -> Tuple[float, float]:
    """Apply a central crop to FOVs. ratios are (cropped_px / full_px)."""
    hfov = 2 * math.degrees(math.atan(crop_ratio_w * math.tan(math.radians(hfov_full_deg / 2))))
    vfov = 2 * math.degrees(math.atan(crop_ratio_h * math.tan(math.radians(vfov_full_deg / 2))))
    return hfov, vfov

def range_for_px_density(n_horizontal_px: int, hfov_deg: float, px_per_meter: float = 50.0) -> float:
    """
    Max distance (m) to achieve the target horizontal density (px/m):
      D = ((Nh/ppm)/2) / tan(HFOV/2)
    """
    half_scene_m = (n_horizontal_px / px_per_meter) / 2.0
    return half_scene_m / math.tan(math.radians(hfov_deg / 2.0))

def two_cam_fov(hfov_deg: float, vfov_deg: float) -> Tuple[float, float]:
    """Return (vertical_deg_for_two_cams, horizontal_deg) = (2*VFOV, HFOV)."""
    return 2.0 * vfov_deg, hfov_deg

def fmt_line(title: str, rng_m: float, fps: int, two_v_deg: float, h_deg: float) -> str:
    return (f"{title}\n"
            f"Effective range: {round(rng_m):d} m\n"
            f"FPS: {fps}\n"
            f"FOV: {round(two_v_deg):d} × {round(h_deg):d}\n")

# ---------------- camera specs ----------------

@dataclass
class IMX577Spec:
    name: str = "IMX577"
    w_px: int = 4056
    h_px: int = 3040
    pitch_um: float = 1.55  # pixel pitch

    @property
    def sensor_w_mm(self) -> float:
        return self.w_px * self.pitch_um * 1e-3

    @property
    def sensor_h_mm(self) -> float:
        return self.h_px * self.pitch_um * 1e-3

@dataclass
class OAK1MaxSpec:
    """
    OAK-1 MAX with fixed lens.
    Spec FOV (no crop): DFOV/HFOV/VFOV = 71°/45°/55°.
    Full-res: 5312 x 6000 (≈32 MP)
    4K:       3840 x 2160 (UHD)
    We support:
      - spec orientation: HFOV=45°, VFOV=55°, horizontal=5312
      - rotated 90°:      HFOV=55°, VFOV=45°, horizontal=6000
      - 4K same-FOV (legacy assumption)
      - 4K crop (true crop reduces FOV)
    """
    name: str = "OAK-1 MAX"
    hfov_deg: float = 45.0
    vfov_deg: float = 55.0
    dfov_deg: float = 71.0

    full_w_px: int = 5312
    full_h_px: int = 6000

    # 4K (UHD) active pixels
    uhd_w_px: int = 3840
    uhd_h_px: int = 2160

    fps_full: int = 10
    fps_uhd: int = 30

    def spec_orientation_params(self):
        """HFOV,VFOV and horizontal/vertical pixel counts for spec orientation."""
        return (self.hfov_deg, self.vfov_deg,
                self.full_w_px, self.full_h_px,
                self.uhd_w_px,  self.uhd_h_px)

    def rotated_orientation_params(self):
        """HFOV,VFOV and pixel counts when rotated 90° (swap roles of width/height at full-res)."""
        return (self.vfov_deg, self.hfov_deg,   # swap FOVs
                self.full_h_px, self.full_w_px, # horizontal=6000, vertical=5312
                self.uhd_w_px,  self.uhd_h_px)  # 4K buffer stays 3840x2160 (horizontal=3840)

# ---------------- example run ----------------

if __name__ == "__main__":
    PX_PER_M = 50.0  # target density

    # ---- IMX577 + lenses ----
    imx = IMX577Spec()
    lens_list_mm = [2.8, 3.6, 4.0, 6.0, 8.0, 12.0, 16.0, 25.0]
    print("IMX577 (all full res 12.3MP)\n")
    for f_mm in lens_list_mm:
        hfov, vfov = rectilinear_fov(imx.sensor_w_mm, imx.sensor_h_mm, f_mm)
        rng = range_for_px_density(imx.w_px, hfov, PX_PER_M)
        two_v, h = two_cam_fov(hfov, vfov)
        print(fmt_line(f"Lens {f_mm:.1f} mm", rng, 60, two_v, h))

    # ---- OAK-1 MAX ----
    oak = OAK1MaxSpec()

    def dump_oak_block(title: str, hfov_deg: float, vfov_deg: float,
                       full_w: int, full_h: int, uhd_w: int, uhd_h: int):
        print(title + "\n")

        # Full-res (no crop)
        rng_full = range_for_px_density(full_w, hfov_deg, PX_PER_M)
        two_v, h = two_cam_fov(hfov_deg, vfov_deg)
        print(fmt_line("Full res mode", rng_full, oak.fps_full, two_v, h))

        # 4K SAME-FOV (legacy assumption: keep FOV, only pixels change)
        rng_uhd_same = range_for_px_density(uhd_w, hfov_deg, PX_PER_M)
        print(fmt_line("4K mode (same FOV)", rng_uhd_same, oak.fps_uhd, two_v, h))

        # 4K CROP (true crop: FOV shrinks by pixel ratios)
        crop_w = uhd_w / full_w
        crop_h = uhd_h / full_h
        hfov_uhd, vfov_uhd = crop_fov(hfov_deg, vfov_deg, crop_w, crop_h)
        rng_uhd_crop = range_for_px_density(uhd_w, hfov_uhd, PX_PER_M)
        two_v_crop, h_crop = two_cam_fov(hfov_uhd, vfov_uhd)
        print(fmt_line("4K mode (CROPPED FOV)", rng_uhd_crop, oak.fps_uhd, two_v_crop, h_crop))

    # Spec orientation
    params = oak.spec_orientation_params()
    dump_oak_block("OAK-1 MAX (spec orientation: HFOV=45°, VFOV=55°)", *params)

    # Rotated orientation
    params_r = oak.rotated_orientation_params()
    dump_oak_block("OAK-1 MAX (rotated 90°: HFOV=55°, VFOV=45°)", *params_r)
