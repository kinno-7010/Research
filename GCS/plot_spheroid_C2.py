"""plot_spheroid_C2_clean.py

K-COR + LASCO-C2 (and AIA193 background) composite difference image produced by
`integrated_analysis.create_single_diff_image()` with a 3D spheroid dome overlay.

Design notes (aligned with `aia_gcs_plot.py`):
- Spheroid is a center-of-symmetry prolate spheroid (a,b,b) whose symmetry axis is radial.
- The axis direction is fixed by (center_lon_deg, center_lat_deg) in HGS.
- The apex is at r = center_r_rsun + a_rsun.
- Optional clipping:
    * only_above_surface: keep r >= 1
    * only_visible: keep points on the visible hemisphere (approx. by observer vector dot-product)
- If wireframe becomes empty with only_visible=True, automatically retry with only_visible=False.

This script intentionally contains NO GCS code.
"""

from __future__ import annotations

import os, sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, CartesianRepresentation

import sunpy.map
from sunpy.coordinates import frames as sunpy_frames

# --- integrated_analysis import (prefer local) ---
sys.path.append("/mnt/d/wsl/home/kinno-7010/Research/SDO_Mk4_SOHO/py_folder")
try:
    from integrated_analysis import create_single_diff_image, create_single_diff_from_time_image
except Exception as exc:
    raise ImportError(
        "Failed to import create_single_diff_image from integrated_analysis.py. "
        "Place integrated_analysis.py next to this script or make it importable via PYTHONPATH."
    ) from exc


# ==========================================================
# Spheroid parameters (same model as in aia_gcs_plot.py)
# ==========================================================

@dataclass
class SpheroidDome3DParams:
    """Center-of-symmetry prolate spheroid (a,b,b) in HGS.

    center_lon_deg / center_lat_deg / center_r_rsun:
        Center position in HGS: radius `center_r_rsun` along the direction (lon,lat).

    a_rsun:
        Semi-major axis along the radial direction (center -> apex).

    b_rsun:
        Semi-minor axis (transverse), rotationally symmetric.
    """

    center_lon_deg: float
    center_lat_deg: float
    center_r_rsun: float
    a_rsun: float
    b_rsun: float

    n_meridians: int = 12
    n_parallels: int = 7
    n_line_pts: int = 240

    only_above_surface: bool = True
    only_visible: bool = True

    @property
    def apex_r_rsun(self) -> float:
        return float(self.center_r_rsun + self.a_rsun)

    @property
    def apex_height_rsun(self) -> float:
        return float(self.apex_r_rsun - 1.0)

    def legend_label(self) -> str:
        return (
            f"Spheroid: a={self.a_rsun:.2f} R$_\\odot$, "
            f"b={self.b_rsun:.2f} R$_\\odot$, "
            f"center r={self.center_r_rsun:.2f} R$_\\odot$, "
            f"(lon,lat)=({self.center_lon_deg:.1f},{self.center_lat_deg:.1f})$^\\circ$"
        )


# ==========================================================
# Geometry helpers
# ==========================================================

def _hgs_unit_vectors(lon_deg: float, lat_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit vectors at (lon,lat) in HGS cartesian basis.

    Returns
    -------
    r_hat : radial unit vector
    e_lon : unit vector in increasing longitude direction
    e_lat : unit vector in increasing latitude direction
    """
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)

    cosl, sinl = np.cos(lon), np.sin(lon)
    cosb, sinb = np.cos(lat), np.sin(lat)

    r_hat = np.array([cosb * cosl, cosb * sinl, sinb], dtype=float)
    e_lon = np.array([-sinl, cosl, 0.0], dtype=float)
    e_lat = np.array([-sinb * cosl, -sinb * sinl, cosb], dtype=float)

    r_hat /= np.linalg.norm(r_hat)
    e_lon /= np.linalg.norm(e_lon)
    e_lat /= np.linalg.norm(e_lat)

    return r_hat, e_lon, e_lat


def _split_skycoord_by_mask(coords: SkyCoord, mask: np.ndarray) -> list[SkyCoord]:
    """Split SkyCoord into contiguous True segments to avoid line bridging."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []

    cuts = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate([[0], cuts + 1])
    ends = np.concatenate([cuts + 1, [idx.size]])

    segs: list[SkyCoord] = []
    for s, e in zip(starts, ends):
        seg_idx = idx[s:e]
        if seg_idx.size >= 2:
            segs.append(coords[seg_idx])
    return segs


# ==========================================================
# Spheroid -> HPC sampling (same logic as aia_gcs_plot.py)
# ==========================================================

def spheroid_dome_apex_hpc(params: SpheroidDome3DParams, reference_map: sunpy.map.Map) -> SkyCoord:
    """Apex = center + a * r_hat."""
    r_hat, _, _ = _hgs_unit_vectors(params.center_lon_deg, params.center_lat_deg)
    center = params.center_r_rsun * r_hat
    apex = center + params.a_rsun * r_hat

    rep = CartesianRepresentation(apex[0] * u.R_sun, apex[1] * u.R_sun, apex[2] * u.R_sun)
    apex_hgs = SkyCoord(rep, frame=sunpy_frames.HeliographicStonyhurst, obstime=reference_map.date)
    return apex_hgs.transform_to(reference_map.coordinate_frame)


def spheroid_axis_footpoint_hpc(params: SpheroidDome3DParams, reference_map: sunpy.map.Map) -> SkyCoord:
    """Intersection of the spheroid symmetry axis with r=1 (diagnostic marker)."""
    r_hat, _, _ = _hgs_unit_vectors(params.center_lon_deg, params.center_lat_deg)
    fp = 1.0 * r_hat

    rep = CartesianRepresentation(fp[0] * u.R_sun, fp[1] * u.R_sun, fp[2] * u.R_sun)
    fp_hgs = SkyCoord(rep, frame=sunpy_frames.HeliographicStonyhurst, obstime=reference_map.date)
    return fp_hgs.transform_to(reference_map.coordinate_frame)


def spheroid_footprint_angular_radius_deg(params: SpheroidDome3DParams) -> float | None:
    """Angular radius ψ [deg] of the photospheric footprint (intersection with r=1)."""
    a = float(params.a_rsun)
    b = float(params.b_rsun)
    cr = float(params.center_r_rsun)

    A = (a * a - b * b)
    B = 2.0 * cr * a
    C = (cr * cr + b * b - 1.0)

    candidates: list[float] = []
    if abs(A) < 1e-12:
        if abs(B) < 1e-12:
            return None
        candidates = [(-C / B)]
    else:
        disc = B * B - 4.0 * A * C
        if disc < 0:
            return None
        sdisc = float(np.sqrt(disc))
        candidates = [(-B + sdisc) / (2.0 * A), (-B - sdisc) / (2.0 * A)]

    valid = [c for c in candidates if np.isfinite(c) and (-1.0 <= c <= 1.0)]
    if not valid:
        return None

    best_c = max(valid, key=lambda cc: np.sqrt(max(0.0, 1.0 - cc * cc)))

    cos_psi = cr + a * best_c
    cos_psi = float(np.clip(cos_psi, -1.0, 1.0))
    psi = float(np.arccos(cos_psi))
    return float(np.rad2deg(psi))


def sample_spheroid_footprint_hpc(params: SpheroidDome3DParams, reference_map: sunpy.map.Map) -> list[SkyCoord]:
    r_hat, e_lon, e_lat = _hgs_unit_vectors(params.center_lon_deg, params.center_lat_deg)

    center_r = float(params.center_r_rsun)
    center = center_r * r_hat

    a = float(params.a_rsun)
    b = float(params.b_rsun)
    cr = center_r

    # Solve (cr + a c)^2 + b^2 (1-c^2) = 1 for c=cos(alpha)
    A = (a * a - b * b)
    B = 2.0 * cr * a
    C = (cr * cr + b * b - 1.0)

    candidates: list[float] = []
    if abs(A) < 1e-12:
        if abs(B) < 1e-12:
            return []
        candidates = [(-C / B)]
    else:
        disc = B * B - 4.0 * A * C
        if disc < 0:
            return []
        sdisc = float(np.sqrt(disc))
        candidates = [(-B + sdisc) / (2.0 * A), (-B - sdisc) / (2.0 * A)]

    valid = [c for c in candidates if np.isfinite(c) and (-1.0 <= c <= 1.0)]
    if not valid:
        return []

    best_c = max(valid, key=lambda cc: np.sqrt(max(0.0, 1.0 - cc * cc)))
    alpha0 = float(np.arccos(best_c))
    sin_a0 = float(np.sin(alpha0))
    cos_a0 = float(np.cos(alpha0))

    bet = np.linspace(0.0, 2.0 * np.pi, params.n_line_pts, endpoint=True)
    sin_b = np.sin(bet)
    cos_b = np.cos(bet)
    dir_latlon = (cos_b[None, :] * e_lon[:, None] + sin_b[None, :] * e_lat[:, None])

    cart = center[:, None] + a * cos_a0 * r_hat[:, None] + b * sin_a0 * dir_latlon
    rep = CartesianRepresentation(cart[0] * u.R_sun, cart[1] * u.R_sun, cart[2] * u.R_sun)
    coords_hgs = SkyCoord(rep, frame=sunpy_frames.HeliographicStonyhurst, obstime=reference_map.date)

    def _visible_mask(coords: SkyCoord) -> np.ndarray:
        if not params.only_visible:
            return np.ones(coords.shape, dtype=bool)
        try:
            obs_vec = reference_map.observer_coordinate.cartesian.xyz.to_value(u.R_sun)
            pt_vec = coords.cartesian.xyz.to_value(u.R_sun)
            s = np.sign(obs_vec[0]) if obs_vec[0] != 0 else 1.0
            dot_sum = np.sum(obs_vec[:, None] * pt_vec, axis=0)
            return (dot_sum * s > 0)
        except Exception:
            return np.ones(coords.shape, dtype=bool)

    mask = _visible_mask(coords_hgs)
    coords_hpc = coords_hgs.transform_to(reference_map.coordinate_frame)
    return _split_skycoord_by_mask(coords_hpc, mask)


def sample_spheroid_dome_wireframe_hpc(params: SpheroidDome3DParams, reference_map: sunpy.map.Map) -> list[SkyCoord]:
    r_hat, e_lon, e_lat = _hgs_unit_vectors(params.center_lon_deg, params.center_lat_deg)

    center_r = float(params.center_r_rsun)
    center = center_r * r_hat

    lines_hpc: list[SkyCoord] = []

    def _above_surface_mask_cart(cart_rsun: np.ndarray) -> np.ndarray:
        if not params.only_above_surface:
            return np.ones(cart_rsun.shape[1], dtype=bool)
        rr = np.sqrt(np.sum(cart_rsun ** 2, axis=0))
        return rr >= 1.0

    def _visible_mask(coords: SkyCoord) -> np.ndarray:
        if not params.only_visible:
            return np.ones(coords.shape, dtype=bool)
        try:
            obs_vec = reference_map.observer_coordinate.cartesian.xyz.to_value(u.R_sun)
            pt_vec = coords.cartesian.xyz.to_value(u.R_sun)
            s = np.sign(obs_vec[0]) if obs_vec[0] != 0 else 1.0
            dot_sum = np.sum(obs_vec[:, None] * pt_vec, axis=0)
            return (dot_sum * s > 0)
        except Exception as e:
            print(f"[WARN] visibility mask disabled (visible check failed): {e}")
            return np.ones(coords.shape, dtype=bool)

    # meridians (beta fixed)
    alpha = np.linspace(0.0, np.pi, params.n_line_pts)
    sin_a = np.sin(alpha)
    cos_a = np.cos(alpha)

    betas = np.linspace(0.0, 2.0 * np.pi, params.n_meridians, endpoint=False)
    for beta in betas:
        dir_latlon = (np.cos(beta) * e_lon + np.sin(beta) * e_lat)
        cart = (
            center[:, None]
            + params.a_rsun * cos_a[None, :] * r_hat[:, None]
            + params.b_rsun * sin_a[None, :] * dir_latlon[:, None]
        )

        rep = CartesianRepresentation(cart[0] * u.R_sun, cart[1] * u.R_sun, cart[2] * u.R_sun)
        coords_hgs = SkyCoord(rep, frame=sunpy_frames.HeliographicStonyhurst, obstime=reference_map.date)

        mask = _above_surface_mask_cart(cart) & _visible_mask(coords_hgs)
        coords_hpc = coords_hgs.transform_to(reference_map.coordinate_frame)
        lines_hpc.extend(_split_skycoord_by_mask(coords_hpc, mask))

    # parallels (alpha fixed)
    bet = np.linspace(0.0, 2.0 * np.pi, params.n_line_pts)
    sin_b = np.sin(bet)
    cos_b = np.cos(bet)

    alphas = np.linspace(0.15 * np.pi, 0.95 * np.pi, params.n_parallels)
    for a0 in alphas:
        sin_a0 = np.sin(a0)
        cos_a0 = np.cos(a0)

        dir_latlon = (cos_b[None, :] * e_lon[:, None] + sin_b[None, :] * e_lat[:, None])
        cart = (
            center[:, None]
            + params.a_rsun * cos_a0 * r_hat[:, None]
            + params.b_rsun * sin_a0 * dir_latlon
        )

        rep = CartesianRepresentation(cart[0] * u.R_sun, cart[1] * u.R_sun, cart[2] * u.R_sun)
        coords_hgs = SkyCoord(rep, frame=sunpy_frames.HeliographicStonyhurst, obstime=reference_map.date)

        mask = _above_surface_mask_cart(cart) & _visible_mask(coords_hgs)
        coords_hpc = coords_hgs.transform_to(reference_map.coordinate_frame)
        lines_hpc.extend(_split_skycoord_by_mask(coords_hpc, mask))

    return lines_hpc


# ==========================================================
# Plot helpers (HPC -> relative pixels in composite axes)
# ==========================================================

def _hpc_to_rel_pix(coords_hpc: SkyCoord, rsun_arcsec: float, px_per_rsun: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert HPC Tx/Ty [arcsec] into composite plot coordinates [pixels]."""
    x_arcsec = np.asarray(coords_hpc.Tx.to_value(u.arcsec), dtype=float)
    y_arcsec = np.asarray(coords_hpc.Ty.to_value(u.arcsec), dtype=float)
    x_px = (x_arcsec / float(rsun_arcsec)) * float(px_per_rsun)
    y_px = (y_arcsec / float(rsun_arcsec)) * float(px_per_rsun)
    return x_px, y_px


def overlay_spheroid_on_coronagraph_axes(
    ax: "plt.Axes",
    reference_map: sunpy.map.Map,
    params_lasco: dict,
    spheroid_params: SpheroidDome3DParams,
    *,
    color: str = "#00FF00",
    lw_wire: float = 1.0,
    lw_footprint: float = 2.2,
    alpha_wire: float = 0.85,
    alpha_footprint: float = 0.95,
    zorder_wire: int = 6,
    zorder_markers: int = 8,
    verbose: bool = True,
) -> SpheroidDome3DParams:
    """Overlay spheroid dome (wireframe + footprint + markers) on the composite axes."""

    rsun_arcsec = float(reference_map.rsun_obs.to_value(u.arcsec))
    px_per_rsun = float(params_lasco["px_per_rsun"])

    if verbose:
        psi_deg = spheroid_footprint_angular_radius_deg(spheroid_params)
        if psi_deg is None:
            print("[INFO] footprint ψ = N/A (no photospheric intersection)")
        else:
            print(f"[INFO] footprint ψ = {psi_deg:.2f} deg")

    wire_lines_hpc = sample_spheroid_dome_wireframe_hpc(spheroid_params, reference_map)
    if (len(wire_lines_hpc) == 0) and spheroid_params.only_visible:
        if verbose:
            print("[WARN] spheroid wireframe is empty; retry with only_visible=False")
        spheroid_params = replace(spheroid_params, only_visible=False)
        wire_lines_hpc = sample_spheroid_dome_wireframe_hpc(spheroid_params, reference_map)

    for ln in wire_lines_hpc:
        x_px, y_px = _hpc_to_rel_pix(ln, rsun_arcsec, px_per_rsun)
        ax.plot(x_px, y_px, color=color, linewidth=lw_wire, alpha=alpha_wire, zorder=zorder_wire)

    footprint_lines_hpc = sample_spheroid_footprint_hpc(spheroid_params, reference_map)
    for fp in footprint_lines_hpc:
        x_px, y_px = _hpc_to_rel_pix(fp, rsun_arcsec, px_per_rsun)
        ax.plot(x_px, y_px, color=color, linewidth=lw_footprint, alpha=alpha_footprint, zorder=zorder_wire + 1)

    # axis-footpoint marker
    try:
        anchor_hpc = spheroid_axis_footpoint_hpc(spheroid_params, reference_map)
        x0, y0 = _hpc_to_rel_pix(anchor_hpc, rsun_arcsec, px_per_rsun)
        ax.scatter(
            [float(np.atleast_1d(x0)[0])],
            [float(np.atleast_1d(y0)[0])],
            marker="*",
            s=90,
            facecolor="yellow",
            edgecolors="black",
            linewidths=0.7,
            zorder=zorder_markers,
            label=(
                f"axis surface intersection (lon,lat)=({spheroid_params.center_lon_deg:.1f},{spheroid_params.center_lat_deg:.1f})°"
            ),
        )
    except Exception as exc:
        if verbose:
            print(f"[WARN] axis-footpoint marker skipped: {exc}")

    # apex marker
    try:
        apex_hpc = spheroid_dome_apex_hpc(spheroid_params, reference_map)
        x1, y1 = _hpc_to_rel_pix(apex_hpc, rsun_arcsec, px_per_rsun)
        apex_label = (
            f"3D spheroid apex (a={spheroid_params.a_rsun:.3f} R$_\\odot$, "
            f"b={spheroid_params.b_rsun:.3f} R$_\\odot$, "
            f"r={spheroid_params.apex_r_rsun:.3f} R$_\\odot$)"
        )
        ax.scatter(
            [float(np.atleast_1d(x1)[0])],
            [float(np.atleast_1d(y1)[0])],
            s=70,
            facecolor="orange",
            edgecolors="black",
            linewidths=0.7,
            zorder=zorder_markers,
            label=apex_label,
        )
    except Exception as exc:
        if verbose:
            print(f"[WARN] apex marker skipped: {exc}")

    # dummy handle for legend
    ax.plot([], [], color=color, lw=3, alpha=0.7, label=spheroid_params.legend_label())

    return spheroid_params


# ==========================================================
# Main
# ==========================================================

def main(
    target_time_str: str,
    spheroid_params: SpheroidDome3DParams,
    *,
    out_png: str | Path | None = None,
    spheroid_color: str = "#00FF00",
    delta_time_min: int = 10,
):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    print(f"[INFO] Building K-COR+LASCO composite for {target_time_str}")
    res = create_single_diff_from_time_image(ax, target_time_str, delta_time_min, mk4_inner=1.4, mk4_outer_lasco_inner=3.0, lasco_outer=6.0, xlim_min=-250, xlim_max=0, ylim_min=-100, ylim_max=200)

    params_lasco = res["params_lasco"]
    lasco_map = res["lasco_map"]

    overlay_spheroid_on_coronagraph_axes(
        ax,
        lasco_map,
        params_lasco,
        spheroid_params,
        color=spheroid_color,
    )

    ax.set_aspect("equal")
    ax.set_xlabel("X [pixels]")
    ax.set_ylabel("Y [pixels]")
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()

    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_png}")

    # Avoid warnings/errors in non-interactive backends (e.g., FigureCanvasAgg).
    import matplotlib
    from matplotlib.backends import BackendFilter, backend_registry

    backend = matplotlib.get_backend().lower()
    interactive_backends = {b.lower() for b in backend_registry.list_builtin(BackendFilter.INTERACTIVE)}
    if backend in interactive_backends:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    # Example: parameterization consistent with aia_gcs_plot.py
    # Set apex radius, then center_r = apex_r - a.
    target_time = "2022-06-13T03:20:00"

    apex_rsun = 2.9
    a_rsun = (apex_rsun-1)/2
    # a_rsun = 2.5
    b_rsun = 0.75
    center_r_rsun = apex_rsun - a_rsun

    spheroid = SpheroidDome3DParams(
        center_lon_deg=-44.0,
        center_lat_deg=+10.0,
        center_r_rsun=float(center_r_rsun),
        a_rsun=float(a_rsun),
        b_rsun=float(b_rsun),
        n_meridians=24,
        n_parallels=12,
        n_line_pts=240,
        only_above_surface=True,
        only_visible=True,
    )

    output_path = f"/mnt/d/wsl/home/kinno-7010/Research/GCS/output/C2_spheroid_{target_time.replace(":","")}.png"
    main(target_time_str=target_time, spheroid_params=spheroid, out_png=output_path)
