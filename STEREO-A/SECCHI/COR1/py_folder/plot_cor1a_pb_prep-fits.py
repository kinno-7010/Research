#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_cor1a_pb_from_fits.py
==========================

Plot a calibrated STEREO-A/SECCHI/COR1 pB FITS file made by make_cor1a_pb.pro.

Input file format:
    /mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/COR1/pB/Rawdata/
        COR1A_pb_pre_<YYYYMMDD>_<hhmmss>.fits

This script does not require secchi_prep.py, cor1_quickpol.py, or density inversion modules.
It only reads the already-created pB FITS file and plots it.
"""

from __future__ import annotations

import os
import math
import re
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


DEFAULT_PB_DIR = "/mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/COR1/pB/Rawdata"
DEFAULT_PNG_DIR = "/mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/COR1/pB/plot"


COR1A_PB_PATTERN = re.compile(r"^COR1A_pb_pre_(\d{8})_(\d{6})\.fits$", re.IGNORECASE)


def _parse_target_time(yyyymmdd: str, hhmmss: str) -> datetime:
    """Parse YYYYMMDD and HHMMSS strings into a timezone-aware UTC datetime."""
    return datetime.strptime(yyyymmdd + hhmmss, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def _parse_cor1a_pb_time_from_name(filename: str) -> Optional[datetime]:
    """Parse COR1A_pb_pre_YYYYMMDD_hhmmss.fits into a timezone-aware UTC datetime."""
    match = COR1A_PB_PATTERN.match(os.path.basename(filename))
    if match is None:
        return None
    yyyymmdd, hhmmss = match.groups()
    return _parse_target_time(yyyymmdd, hhmmss)


def find_nearest_cor1a_pb_fits(pb_dir: str, target_time: datetime) -> Tuple[str, datetime, float]:
    """Find the COR1A_pb_pre_*.fits file whose filename time is nearest to target_time."""
    if not os.path.isdir(pb_dir):
        raise FileNotFoundError(f"pB directory not found: {pb_dir}")

    candidates = []
    for name in sorted(os.listdir(pb_dir)):
        obs_time = _parse_cor1a_pb_time_from_name(name)
        if obs_time is None:
            continue
        path = os.path.join(pb_dir, name)
        if not os.path.isfile(path):
            continue
        delta_seconds = abs((obs_time - target_time).total_seconds())
        candidates.append((delta_seconds, obs_time, path))

    if not candidates:
        raise FileNotFoundError(f"No COR1A_pb_pre_YYYYMMDD_hhmmss.fits files found in {pb_dir}")

    delta_seconds, obs_time, path = min(candidates, key=lambda item: (item[0], item[1]))
    return path, obs_time, float(delta_seconds)


def _read_fits(path: str) -> Tuple[np.ndarray, dict]:
    """Read a FITS image and return (data, uppercase-header-dict).

    Important:
      Older COR1A_pb_pre_*.fits files may still contain stale integer-image
      scaling keywords inherited from the original COR1 raw header, especially
      BZERO=32768 and BLANK=-32768.  For a derived floating-point pB image,
      those keywords are not appropriate.  Reading with Astropy's default
      scaling would add BZERO to the physical pB values and make the image look
      nearly empty or incorrectly scaled.

      Therefore this reader intentionally disables FITS image scaling and uses
      the stored floating-point array directly.  The proper long-term fix is to
      remove BZERO/BSCALE/BLANK when make_cor1a_pb.pro writes the pB FITS.
    """
    with fits.open(path, memmap=False, do_not_scale_image_data=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        hdr = {str(k).upper(): v for k, v in hdul[0].header.items()}

    return data, hdr


def _params_from_header(hdr: dict) -> Dict[str, float]:
    """Extract COR1 image geometry from the FITS header."""
    nx = int(hdr.get("NAXIS1"))
    ny = int(hdr.get("NAXIS2"))

    # FITS CRPIX is 1-based.  Convert to Python 0-based pixel coordinates.
    cx = float(hdr.get("CRPIX1", (nx + 1.0) / 2.0)) - 1.0
    cy = float(hdr.get("CRPIX2", (ny + 1.0) / 2.0)) - 1.0

    scale = abs(float(hdr.get("CDELT1", hdr.get("CDELT2", 15.0))))  # arcsec/pixel

    # SECCHI/COR1 headers often have RSUN in arcsec after secchi_prep.
    # Prefer RSUN_OBS if available, then RSUN, then solar angular radius fallback.
    rsun_arcsec = float(hdr.get("RSUN_OBS", hdr.get("RSUN", 959.2)))
    px_per_rsun = rsun_arcsec / scale

    return dict(
        nx=nx,
        ny=ny,
        cx=cx,
        cy=cy,
        scale=scale,
        arcsec_per_pix=scale,
        rsun_arcsec=rsun_arcsec,
        px_per_rsun=px_per_rsun,
    )


def _radius_map(shape: Tuple[int, int], cx: float, cy: float, px_per_rsun: float) -> np.ndarray:
    """Return radius map in solar radii."""
    ny, nx = shape
    y, x = np.indices((ny, nx))
    r_pix = np.hypot(x - cx, y - cy)
    return r_pix / float(px_per_rsun)


def _mask_cor1_pb(
    pB: np.ndarray,
    r_map: np.ndarray,
    r_keep_min: float = 1.4,
    r_keep_max: float = 4.0,
) -> np.ndarray:
    """
    Keep pB only in [r_keep_min, r_keep_max] R_sun; mask elsewhere with NaN.

    Same convention as density_2D_map_sta._mask_cor1_pb.
    """
    out = np.array(pB, dtype=float, copy=True)
    m = (r_map < float(r_keep_min)) | (r_map > float(r_keep_max))
    out[m] = np.nan
    return out


def _pb_plot_limits(pb: np.ndarray, r_map: np.ndarray, r_min: float, r_max: float) -> Tuple[float, float]:
    """Choose robust positive LogNorm limits for pB."""
    valid = np.isfinite(pb) & (pb > 0.0) & (r_map >= r_min) & (r_map <= r_max)
    if np.count_nonzero(valid) < 10:
        return 1e-12, 1e-9

    vals = pb[valid]
    # vmin = float(np.nanpercentile(vals, 1.0))
    # vmax = float(np.nanpercentile(vals, 99.5))
    vmin = 1e-9
    vmax = 1e-7
    # if not np.isfinite(vmin) or vmin <= 0.0:
    #     vmin = float(np.nanmin(vals[vals > 0.0]))
    # if not np.isfinite(vmax) or vmax <= vmin:
    #     vmax = vmin * 100.0
    return vmin, vmax


def plot_cor1a_pb_fits(
    fits_path: str,
    *,
    savepath: Optional[str] = None,
    r_keep_min: float = 1.1,
    r_keep_max: float = 4.0,
    cmap: str = "inferno",
    show: bool = True,
):
    """
    Plot an already-calibrated COR1-A pB FITS file.

    Parameters
    ----------
    fits_path:
        Path to COR1A_pb_pre_<YYYYMMDD>_<hhmmss>.fits.
    savepath:
        Output PNG path.  If None, a PNG is written next to DEFAULT_PNG_DIR.
    r_keep_min, r_keep_max:
        Radius range shown in color.  Pixels outside this range are masked.
    cmap:
        Matplotlib colormap name.
    show:
        If True, call plt.show().
    """
    if not os.path.exists(fits_path):
        raise FileNotFoundError(f"Input FITS not found: {fits_path}")

    pb, hdr = _read_fits(fits_path)
    finite0 = np.isfinite(pb)
    if np.count_nonzero(finite0) > 0:
        print(
            "[INFO] Raw stored pB stats: "
            f"min={np.nanmin(pb):.3e}, "
            f"median={np.nanmedian(pb):.3e}, "
            f"max={np.nanmax(pb):.3e}, "
            f"positive={np.count_nonzero(finite0 & (pb > 0.0))}/{pb.size}"
        )
    params = _params_from_header(hdr)
    r_map = _radius_map(pb.shape, params["cx"], params["cy"], params["px_per_rsun"])

    pb_plot = _mask_cor1_pb(pb, r_map, r_keep_min=r_keep_min, r_keep_max=r_keep_max)
    bad = ~np.isfinite(pb_plot) | (pb_plot <= 0.0)
    pb_plot[bad] = np.nan

    ny, nx = pb_plot.shape
    cx = float(params["cx"])
    cy = float(params["cy"])
    s_arc = float(params["arcsec_per_pix"])

    extent = [(-cx) * s_arc, (nx - cx) * s_arc, (-cy) * s_arc, (ny - cy) * s_arc]
    yy, xx = np.indices((ny, nx))
    x_arcsec = (xx - cx) * s_arc
    y_arcsec = (yy - cy) * s_arc

    valid_plot = np.isfinite(pb_plot) & (pb_plot > 0.0)
    print(f"[INFO] Plotted positive pixels: {np.count_nonzero(valid_plot)} / {pb_plot.size}")
    vmin, vmax = _pb_plot_limits(pb_plot, r_map, r_keep_min, r_keep_max)
    print(f"[INFO] LogNorm range: vmin={vmin:.3e}, vmax={vmax:.3e}")

    date_obs = str(hdr.get("DATE-OBS", ""))
    date_avg = str(hdr.get("DATE-AVG", ""))
    title_time = date_avg if date_avg else date_obs
    title = f"STEREO-A/COR1 pB\n{title_time} UT"

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(
        pb_plot,
        origin="lower",
        extent=extent,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap=cmap,
        interpolation="nearest",
    )

    # Integer-Rsun reference circles.
    levels = np.arange(1, int(math.floor(r_keep_max)) + 1, dtype=float)
    if levels.size > 0:
        ax.contour(
            x_arcsec,
            y_arcsec,
            r_map,
            levels=levels,
            colors="white",
            linewidths=0.8,
            linestyles="--",
            alpha=0.7,
        )

    # Analysis/view range boundaries.
    for radius, color, label in [
        (r_keep_min, "cyan", f"{r_keep_min:.1f} R$_\\odot$"),
        (r_keep_max, "magenta", f"{r_keep_max:.1f} R$_\\odot$"),
    ]:
        if radius <= np.nanmax(r_map):
            ax.contour(
                x_arcsec,
                y_arcsec,
                r_map,
                levels=[radius],
                colors=[color],
                linewidths=1.2,
                linestyles="-.",
            )
            ax.plot([], [], color=color, linestyle="-.", label=label)

    ax.plot(0.0, 0.0, "+", color="black", markersize=10, markeredgewidth=1.5)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("X [arcsec from Sun center]", fontsize=14)
    ax.set_ylabel("Y [arcsec from Sun center]", fontsize=14)
    ax.set_aspect("equal")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("pB [Bsun]", fontsize=14)
    cb.ax.tick_params(labelsize=12)

    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    if savepath is None:
        os.makedirs(DEFAULT_PNG_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(fits_path))[0]
        savepath = os.path.join(DEFAULT_PNG_DIR, base + ".png")
    else:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)

    fig.savefig(savepath, dpi=300)
    print(f"[save] Wrote {savepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Edit only this block for another observation time.
    # The nearest available file is selected from:
    #   COR1A_pb_pre_<YYYYMMDD>_<hhmmss>.fits
    # ------------------------------------------------------------
    TARGET_YYYYMMDD = "20220613"
    TARGET_HHMMSS = "030000"

    PB_DIR = DEFAULT_PB_DIR
    PNG_DIR = DEFAULT_PNG_DIR

    target_time = _parse_target_time(TARGET_YYYYMMDD, TARGET_HHMMSS)
    input_fits, selected_time, delta_seconds = find_nearest_cor1a_pb_fits(PB_DIR, target_time)

    selected_base = os.path.splitext(os.path.basename(input_fits))[0]
    output_png = os.path.join(PNG_DIR, selected_base + ".png")

    print(f"[INFO] Target time   : {target_time:%Y-%m-%dT%H:%M:%S} UT")
    print(f"[INFO] Selected file : {input_fits}")
    print(f"[INFO] Selected time : {selected_time:%Y-%m-%dT%H:%M:%S} UT")
    print(f"[INFO] Time offset   : {delta_seconds:.1f} s")

    plot_cor1a_pb_fits(
        input_fits,
        savepath=output_png,
        r_keep_min=1.1,
        r_keep_max=4.0,
        cmap="plasma",
        show=True,
    )
