#!/usr/bin/env python3
"""
Render tomography isodensity surfaces from a saved NPZ solution.

This script intentionally draws only:
  - tomography isodensity surfaces loaded from NPZ density data
  - an optional solar reference sphere
  - X/Y/Z arrows

It does not draw GCS, Spheroid, or PFSS overlays.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pyvista as pv


@dataclass
class SphericalGrid:
    """Minimal spherical grid container for NPZ-based tomography rendering."""

    r_edges: np.ndarray
    th_edges: np.ndarray
    ph_edges: np.ndarray

    @property
    def nr(self) -> int:
        return int(self.r_edges.size - 1)

    @property
    def nth(self) -> int:
        return int(self.th_edges.size - 1)

    @property
    def nph(self) -> int:
        return int(self.ph_edges.size - 1)

    @property
    def nvox(self) -> int:
        return int(self.nr * self.nth * self.nph)

    def voxel_centers_sph(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = 0.5 * (self.r_edges[:-1] + self.r_edges[1:])
        th = 0.5 * (self.th_edges[:-1] + self.th_edges[1:])
        ph = 0.5 * (self.ph_edges[:-1] + self.ph_edges[1:])
        return np.meshgrid(r, th, ph, indexing="ij")


@dataclass
class SimpleObserver:
    """Carrington observer lon/lat in degrees, used for camera and observer axes."""

    lonlat_deg: Tuple[float, float]


@dataclass
class TomographyNPZ:
    grid: SphericalGrid
    ne: np.ndarray
    npz_path: Path
    density_key: str
    harmonic: Optional[int]
    freq_mhz_from_npz: Optional[np.ndarray]
    observer: Optional[SimpleObserver]
    target_time: str


def _decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _scalar_from_npz(npz, keys: Tuple[str, ...], default=None):
    for key in keys:
        if key not in npz.files:
            continue
        arr = np.asarray(npz[key])
        if arr.size == 0:
            continue
        return _decode_scalar(arr.reshape(-1)[0])
    return default


def _bool_from_value(value) -> bool:
    value = _decode_scalar(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _npz_get_first(npz, keys: Tuple[str, ...]):
    for key in keys:
        if key in npz.files:
            return npz[key], key
    raise KeyError(f"None of the required keys were found in NPZ: {keys}. Available keys: {npz.files}")


def _optional_float_array(npz, keys: Tuple[str, ...]) -> Optional[np.ndarray]:
    for key in keys:
        if key not in npz.files:
            continue
        arr = np.asarray(npz[key])
        if arr.size == 0:
            continue
        try:
            out = arr.astype(np.float64, copy=False).reshape(-1)
        except Exception:
            continue
        out = out[np.isfinite(out)]
        if out.size > 0:
            return out
    return None


def _observer_from_npz(npz) -> Optional[SimpleObserver]:
    render_lon = _scalar_from_npz(npz, ("render_camera_lon_deg",), default=None)
    render_lat = _scalar_from_npz(npz, ("render_camera_lat_deg",), default=None)
    render_is_earth = _scalar_from_npz(npz, ("render_camera_is_earth_view",), default=False)
    if render_lon is not None and render_lat is not None and _bool_from_value(render_is_earth):
        return SimpleObserver(lonlat_deg=(float(render_lon) % 360.0, float(render_lat)))

    render_lonlat = _optional_float_array(npz, ("render_camera_lonlat_deg",))
    if render_lonlat is not None and render_lonlat.size >= 2:
        return SimpleObserver(lonlat_deg=(float(render_lonlat[0]) % 360.0, float(render_lonlat[1])))

    lon = _scalar_from_npz(
        npz,
        ("obs_lon_deg", "observer_lon_deg", "crln_obs", "CRLN_OBS", "lon_obs_deg"),
        default=None,
    )
    lat = _scalar_from_npz(
        npz,
        ("obs_lat_deg", "observer_lat_deg", "crlt_obs", "CRLT_OBS", "lat_obs_deg"),
        default=None,
    )
    if lon is not None and lat is not None:
        return SimpleObserver(lonlat_deg=(float(lon) % 360.0, float(lat)))

    obs_lonlat = _optional_float_array(npz, ("obs_lonlat_deg", "observer_lonlat_deg", "lonlat_deg"))
    if obs_lonlat is not None and obs_lonlat.size >= 2:
        pairs = obs_lonlat.reshape((-1, 2)) if obs_lonlat.size % 2 == 0 else obs_lonlat[:2].reshape((1, 2))
        for pair in pairs:
            if np.all(np.isfinite(pair)):
                return SimpleObserver(lonlat_deg=(float(pair[0]) % 360.0, float(pair[1])))

    return None


def sun_to_observer_unit_vector(observer: SimpleObserver) -> np.ndarray:
    lon_deg, lat_deg = observer.lonlat_deg
    lon = np.deg2rad(float(lon_deg))
    lat = np.deg2rad(float(lat_deg))
    vec = np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        dtype=np.float64,
    )
    norm = np.linalg.norm(vec)
    if (not np.isfinite(norm)) or norm <= 0:
        raise ValueError(f"Invalid observer lon/lat: {observer.lonlat_deg}")
    return vec / norm


def load_tomography_npz(npz_path: Path) -> TomographyNPZ:
    npz_path = Path(npz_path).expanduser().resolve()
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    print("[INFO] Loading tomography NPZ:")
    print(f"       {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        ne_arr, density_key = _npz_get_first(data, ("ne", "ne_scaled", "electron_density", "density", "ne_cm3"))
        r_edges, _ = _npz_get_first(data, ("r_edges",))
        th_edges, _ = _npz_get_first(data, ("th_edges", "theta_edges"))
        ph_edges, _ = _npz_get_first(data, ("ph_edges", "phi_edges"))

        grid = SphericalGrid(
            r_edges=np.asarray(r_edges, dtype=np.float64).reshape(-1).copy(),
            th_edges=np.asarray(th_edges, dtype=np.float64).reshape(-1).copy(),
            ph_edges=np.asarray(ph_edges, dtype=np.float64).reshape(-1).copy(),
        )

        ne_arr = np.asarray(ne_arr, dtype=np.float64)
        if ne_arr.size != grid.nvox:
            raise ValueError(
                f"NPZ density array {density_key!r} has size {ne_arr.size}, "
                f"but grid.nvox={grid.nvox}; grid=({grid.nr}, {grid.nth}, {grid.nph}), "
                f"shape={ne_arr.shape}."
            )
        ne = ne_arr.reshape(-1, order="C").copy()

        harmonic_value = _scalar_from_npz(data, ("harmonic", "HARMONIC"), default=None)
        harmonic = int(harmonic_value) if harmonic_value is not None else None
        freq_mhz = _optional_float_array(data, ("freq_mhz_list", "frequency_mhz_list", "freq_mhz", "frequency_mhz"))
        observer = _observer_from_npz(data)
        target_time = str(
            _scalar_from_npz(
                data,
                ("render_camera_time_utc", "observer_time_iso", "target_time_iso", "target_time", "TARGET_TIME"),
                default="",
            )
        )

    positive = np.isfinite(ne) & (ne > 0)
    if np.any(positive):
        print(
            f"[INFO] Loaded density key={density_key!r}, grid=({grid.nr}, {grid.nth}, {grid.nph}), "
            f"ne range={np.nanmin(ne[positive]):.3e}..{np.nanmax(ne[positive]):.3e} cm^-3"
        )
    else:
        print(f"[WARN] Loaded density key={density_key!r}, but no positive finite density values were found.")

    if observer is not None:
        print(
            "[INFO] Observer/camera lonlat loaded from NPZ: "
            f"lon={observer.lonlat_deg[0]:.6f} deg, lat={observer.lonlat_deg[1]:.6f} deg"
        )
    else:
        print("[WARN] No observer/camera lonlat was found in NPZ; using default isometric camera and Cartesian axes.")

    return TomographyNPZ(
        grid=grid,
        ne=ne,
        npz_path=npz_path,
        density_key=density_key,
        harmonic=harmonic,
        freq_mhz_from_npz=freq_mhz.copy() if freq_mhz is not None else None,
        observer=observer,
        target_time=target_time,
    )


def build_tomography_structured_grid(grid: SphericalGrid, ne_1d: np.ndarray) -> pv.StructuredGrid:
    rr, tt, pp = grid.voxel_centers_sph()
    ne3 = np.asarray(ne_1d, dtype=np.float64).reshape((grid.nr, grid.nth, grid.nph), order="C")

    pp2 = np.concatenate([pp, pp[:, :, :1] + 2.0 * np.pi], axis=2)
    rr2 = np.concatenate([rr, rr[:, :, :1]], axis=2)
    tt2 = np.concatenate([tt, tt[:, :, :1]], axis=2)
    ne2 = np.concatenate([ne3, ne3[:, :, :1]], axis=2)

    xx = rr2 * np.sin(tt2) * np.cos(pp2)
    yy = rr2 * np.sin(tt2) * np.sin(pp2)
    zz = rr2 * np.cos(tt2)

    sg = pv.StructuredGrid(xx, yy, zz)
    sg["ne"] = ne2.ravel(order="F")
    return sg


def ne_cm3_from_fp_mhz(fp_mhz: float, harmonic: int) -> float:
    return (float(fp_mhz) * 1.0e6 / (8980.0 * float(harmonic))) ** 2


def fp_mhz_from_ne_cm3(ne_cm3: np.ndarray, harmonic: int) -> np.ndarray:
    return float(harmonic) * 8980.0 * np.sqrt(np.asarray(ne_cm3, dtype=np.float64)) / 1.0e6


def parse_float_list(text: Optional[str]) -> Optional[list[float]]:
    if text is None:
        return None
    cleaned = str(text).replace(",", " ").strip()
    if not cleaned:
        return None
    return [float(part) for part in cleaned.split()]


def colors_for_count(n: int, colors_text: Optional[str]) -> list[str]:
    if colors_text:
        base = [part.strip() for part in colors_text.replace(",", " ").split() if part.strip()]
    else:
        base = ["gold", "cyan", "tomato", "deepskyblue", "limegreen", "violet", "orange"]
    if not base:
        raise ValueError("At least one color is required.")
    return (base * ((int(n) + len(base) - 1) // len(base)))[: int(n)]


def resolve_isodensity_levels(
    *,
    ne_cm3: Optional[Sequence[float]],
    freq_mhz: Optional[Sequence[float]],
    npz_freq_mhz: Optional[np.ndarray],
    harmonic: Optional[int],
) -> tuple[list[float], list[str], int]:
    if ne_cm3 is not None and freq_mhz is not None:
        raise ValueError("Specify either density levels (--ne-cm3) or frequency levels (--freq-mhz), not both.")

    if ne_cm3 is not None:
        levels = [float(v) for v in ne_cm3]
        labels = [f"ne={v:.3e} cm^-3" for v in levels]
        return levels, labels, int(harmonic) if harmonic is not None else 1

    freq_values = list(freq_mhz) if freq_mhz is not None else None
    if freq_values is None and npz_freq_mhz is not None and npz_freq_mhz.size > 0:
        freq_values = [float(v) for v in npz_freq_mhz]

    if freq_values is None:
        raise ValueError(
            "No isodensity level was specified. Provide --ne-cm3, provide --freq-mhz, "
            "or use an NPZ that contains freq_mhz_list/freq_mhz."
        )

    if harmonic is None:
        raise ValueError("Frequency-based isodensity surfaces require --harmonic or a harmonic value in the NPZ.")

    h = int(harmonic)
    levels = [ne_cm3_from_fp_mhz(float(f), h) for f in freq_values]
    labels = [f"f={float(f):.3g} MHz (H={h})" for f in freq_values]
    return levels, labels, h


def add_isodensity_surfaces(
    plotter: pv.Plotter,
    sg: pv.StructuredGrid,
    levels_ne_cm3: Sequence[float],
    labels: Sequence[str],
    colors: Sequence[str],
    *,
    opacity: float,
) -> int:
    if "ne" not in sg.array_names:
        raise ValueError("StructuredGrid must contain scalar array 'ne'.")

    sg.set_active_scalars("ne")
    ne_all = np.asarray(sg["ne"], dtype=np.float64)
    ne_pos = ne_all[np.isfinite(ne_all) & (ne_all > 0)]
    if ne_pos.size == 0:
        print("[WARN] No positive density in reconstruction.")
        return 0

    ne_min = float(np.nanmin(ne_pos))
    ne_max = float(np.nanmax(ne_pos))
    print(f"[INFO] StructuredGrid ne range={ne_min:.3e}..{ne_max:.3e} cm^-3")

    rendered = 0
    for level, label, color in zip(levels_ne_cm3, labels, colors):
        level = float(level)
        print(f"[INFO] Requested isodensity surface: {label}; ne={level:.3e} cm^-3")
        if (not np.isfinite(level)) or level < ne_min or level > ne_max:
            print(
                f"[WARN] Skipping {label}: ne={level:.3e} cm^-3 is outside "
                f"the reconstructed range {ne_min:.3e}..{ne_max:.3e} cm^-3."
            )
            continue

        surf = sg.contour(isosurfaces=[level], scalars="ne")
        if surf.n_points == 0:
            print(f"[WARN] Empty contour for {label}.")
            continue

        plotter.add_mesh(surf, color=color, opacity=float(opacity), smooth_shading=True)
        rendered += 1

    if rendered == 0:
        print("[WARN] No requested isodensity surface was rendered.")

    print(f"[INFO] Isodensity surfaces rendered: {rendered}")
    return rendered


def add_xyz_arrows(
    plotter: pv.Plotter,
    *,
    observer: Optional[SimpleObserver],
    axis_mode: str,
    axis_len: float,
    origin_rsun: float,
    shaft_radius: float,
    tip_radius: float,
    tip_length: float,
    label_font_size: int,
) -> None:
    axis_mode = axis_mode.lower()
    z_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    if axis_mode == "observer" and observer is not None:
        x_hat = sun_to_observer_unit_vector(observer)
        y_hat = np.cross(z_hat, x_hat)
        yn = np.linalg.norm(y_hat)
        if (not np.isfinite(yn)) or yn <= 0:
            y_hat = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            y_hat = y_hat / yn
        origin = x_hat * float(origin_rsun)
        labels = ["X (Sun-Earth)", "Y (West/right)", "Z (North)"]
    elif axis_mode == "cartesian" or (axis_mode == "observer" and observer is None):
        x_hat = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        y_hat = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        origin = np.zeros(3, dtype=np.float64)
        labels = ["X", "Y", "Z"]
    else:
        raise ValueError("axis_mode must be 'observer' or 'cartesian'.")

    directions = [x_hat, y_hat, z_hat]
    colors = ["crimson", "seagreen", "royalblue"]

    for direction, color in zip(directions, colors):
        arrow = pv.Arrow(
            start=origin,
            direction=direction,
            scale=float(axis_len),
            tip_length=float(tip_length),
            tip_radius=float(tip_radius),
            shaft_radius=float(shaft_radius),
        )
        plotter.add_mesh(arrow, color=color)

    tips = np.vstack([origin + direction * float(axis_len) for direction in directions])
    plotter.add_point_labels(
        tips,
        labels,
        point_size=0,
        font_size=int(label_font_size),
        text_color="black",
        always_visible=True,
        shape=None,
    )


def set_camera(plotter: pv.Plotter, observer: Optional[SimpleObserver], distance_rsun: float) -> None:
    if observer is None:
        plotter.view_isometric()
        try:
            plotter.reset_camera()
            plotter.reset_camera_clipping_range()
        except Exception:
            pass
        return

    cam_dir = sun_to_observer_unit_vector(observer)
    bounds = np.asarray(plotter.bounds, dtype=np.float64)
    if np.all(np.isfinite(bounds)):
        dist = max(float(distance_rsun), 3.0 * float(np.max(np.abs(bounds))), 20.0)
    else:
        dist = float(distance_rsun)

    cam_pos = cam_dir * dist
    try:
        print(f"[DEBUG] set_camera: cam_dir={cam_dir}, dist={dist}, cam_pos={cam_pos}, bounds={bounds}")
    except Exception:
        pass

    try:
        plotter.camera_position = [cam_pos.tolist(), [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    except Exception:
        pass

    try:
        cam = getattr(plotter, "camera", None)
        if cam is not None:
            cam.SetPosition(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
            cam.SetFocalPoint(0.0, 0.0, 0.0)
            cam.SetViewUp(0.0, 0.0, 1.0)
            try:
                plotter.render()
            except Exception:
                pass
    except Exception:
        pass

    try:
        plotter.camera.clipping_range = (0.01, max(1000.0, 4.0 * dist))
    except Exception:
        pass

    try:
        plotter.reset_camera_clipping_range()
    except Exception:
        pass

    try:
        plotter.enable_trackball_style()
    except Exception:
        pass

    try:
        import vtk  # noqa

        if getattr(plotter, "iren", None) is not None and getattr(plotter.iren, "interactor", None) is not None:
            plotter.iren.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    except Exception:
        pass


def render_npz_tomography(
    *,
    npz_path: Path,
    output_png: Optional[Path],
    freq_mhz: Optional[Sequence[float]],
    ne_cm3: Optional[Sequence[float]],
    harmonic_override: Optional[int],
    colors_text: Optional[str],
    opacity: float,
    show_sun: bool,
    axis_mode: str,
    axis_len: float,
    show_gui: bool,
    save_png: bool,
    camera_distance_rsun: float,
) -> Optional[Path]:
    tomo = load_tomography_npz(npz_path)
    harmonic = int(harmonic_override) if harmonic_override is not None else tomo.harmonic
    levels, labels, harmonic = resolve_isodensity_levels(
        ne_cm3=ne_cm3,
        freq_mhz=freq_mhz,
        npz_freq_mhz=tomo.freq_mhz_from_npz,
        harmonic=harmonic,
    )
    colors = colors_for_count(len(levels), colors_text)

    sg = build_tomography_structured_grid(tomo.grid, tomo.ne)
    print(f"[INFO] StructuredGrid bounds: {sg.bounds}")

    off_screen = not bool(show_gui)
    if show_gui and not os.environ.get("DISPLAY"):
        print("[WARN] DISPLAY is not set; forcing off-screen rendering.")
        off_screen = True
        try:
            pv.start_xvfb()
        except Exception as exc:
            print(f"[WARN] pv.start_xvfb failed: {exc}")

    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background("white")
    try:
        plotter.enable_depth_peeling()
    except Exception:
        pass
    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:
        pass

    if show_sun:
        plotter.add_mesh(
            pv.Sphere(radius=1.0, theta_resolution=72, phi_resolution=72),
            color="lightgray",
            opacity=0.20,
        )

    add_isodensity_surfaces(
        plotter,
        sg,
        levels,
        labels,
        colors,
        opacity=float(opacity),
    )
    add_xyz_arrows(
        plotter,
        observer=tomo.observer,
        axis_mode=axis_mode,
        axis_len=float(axis_len),
        origin_rsun=1.0,
        shaft_radius=0.04,
        tip_radius=0.08,
        tip_length=0.25,
        label_font_size=12,
    )

    set_camera(plotter, tomo.observer, distance_rsun=float(camera_distance_rsun))

    saved_path = None
    if save_png:
        if output_png is None:
            output_png = tomo.npz_path.with_name(f"{tomo.npz_path.stem}_main_npz_tomo.png")
        output_png = Path(output_png).expanduser().resolve()
        output_png.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Saving PNG: {output_png}")
        plotter.show(screenshot=str(output_png), auto_close=True)
        saved_path = output_png
    else:
        plotter.show(auto_close=True)

    return saved_path



def main(argv: Optional[Sequence[str]] = None) -> Optional[Path]:
    TARGET_TIME = "20220613_030000"
    SEARCH_WINDOW_DAYS = 5.0
    FREQ = 33
    HARMONIC = 2
    # OTHER_TAG = "no-weight"
    OTHER_TAG = ""

    DATA_DIR = "/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/ne_npz"
    OUTPUT_DIR = "/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/output/multi-tomo"
    WINDOW_TAG = f"pm{int(SEARCH_WINDOW_DAYS)}d"
    OTHER_SUFFIX = f"_{OTHER_TAG}" if OTHER_TAG else ""
    DATA_PATH = f"ne3d_solution_{TARGET_TIME}_{WINDOW_TAG}_{FREQ}MHz{OTHER_SUFFIX}.npz"
    OUTPUT_PATH = f"tomo_{TARGET_TIME}_{WINDOW_TAG}_{FREQ}MHz{OTHER_SUFFIX}.png"

    return render_npz_tomography(
        npz_path=Path(DATA_DIR) / DATA_PATH,
        output_png=Path(OUTPUT_DIR) / OUTPUT_PATH,
        freq_mhz=[FREQ],
        ne_cm3=None,
        harmonic_override=HARMONIC,
        colors_text=None,
        opacity=0.2,
        show_sun=True,
        axis_mode="observer",
        axis_len=1.6,
        show_gui=True,
        save_png=True,
        camera_distance_rsun=4.0,
    )


if __name__ == "__main__":
    main()
