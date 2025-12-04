#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_sta_tB.py
==============
Plot a 2-D total brightness (tB) image from STEREO-A / SECCHI-COR1 using three
polarization frames (0°, 120°, 240°). The processing options mirror plot_sta_pB.py.

Processing chain:
  1) First-stage calibration & optional background subtraction (secchi_prep.py)
  2) Fast Stokes solving for tB, pB, μ (cor1_quickpol.py)
  3) Matplotlib visualization of tB in log scale with axes in arcsec
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Ensure sibling helper modules are importable
sys.path.append(str(Path(__file__).resolve().parent))

from secchi_prep import (
    first_stage_calibration_and_background,
)
from cor1_quickpol import cor1_quickpol


def _read_and_prepare(path: str,
                      calimg=None,
                      bkgimg=None,
                      auto_bkg: bool=False,
                      secchi_bkg_dir: str|None=None,
                      rectify: bool=True,
                      discri: bool|tuple|None=None,
                      sebip_off: bool=False,
                      silent: bool=True):
    """Run first-stage calibration for a single FITS frame; return (img, hdr)."""
    img, hdr, _ = first_stage_calibration_and_background(
        path,
        calimg=calimg,
        bkgimg=bkgimg,
        exptime_off=False,
        bias_off=False,
        calfac_off=False,
        calimg_off=(calimg is None),
        bkgimg_off=(bkgimg is None) and (not auto_bkg),
        rectify=rectify,
        auto_bkg=auto_bkg,
        secchi_bkg_dir=secchi_bkg_dir,
        discri_pobj_on=discri,
        sebip_off=sebip_off,
        silent=silent
    )
    return img, hdr


def _order_by_polar(files: list[str]) -> tuple[list[str], list[dict]]:
    """Read headers to order files into 0°, 120°, 240° (240≡60 mod 180)."""
    from astropy.io import fits
    meta = []
    for p in files:
        with fits.open(p, memmap=False) as hdul:
            h = {k.upper(): v for k, v in hdul[0].header.items()}
        pol = float(h.get('POLAR', 0.0))
        poln = pol % 180.0
        meta.append((p, h, pol, poln))
    # buckets: 0 -> i0, 120 -> i120, 60 (≡240) -> i240
    i0 = min(meta, key=lambda t: abs(t[3]-0.0))
    i120 = min(meta, key=lambda t: abs(t[3]-120.0))
    i60 = min(meta, key=lambda t: abs(t[3]-60.0))
    ordered = [i0, i120, i60]
    return [o[0] for o in ordered], [o[1] for o in ordered]


def _rsun_pix_from_header(h: dict) -> float:
    """Compute R_sun in pixels using RSUN (arcsec) / CDELT1 (arcsec/pix)."""
    rsun_arc = h.get('RSUN')
    cd = h.get('CDELT1')
    if (rsun_arc is None) or (cd is None) or (cd == 0):
        return np.nan
    try:
        return float(rsun_arc) / float(cd)
    except Exception:
        return np.nan


def _extent_in_arcsec(h: dict, nx: int, ny: int) -> tuple[float, float, float, float]:
    """Return imshow extent in arcsec centered on solar disk using CRPIX and CDELT."""
    cd = float(h.get('CDELT1', 0.0)) or 1.0  # arcsec/pixel
    cx = float(h.get('CRPIX1', (nx - 1) / 2))
    cy = float(h.get('CRPIX2', (ny - 1) / 2))
    x = (np.array([0, nx]) - cx) * cd
    y = (np.array([0, ny]) - cy) * cd
    return (x[0], x[1], y[0], y[1])


def _draw_rsun_circles(ax, h: dict, extent: tuple[float, float, float, float]) -> None:
    """
    Draw concentric dotted circles at 1,2,3,... Rs on arcsec axes.
    - 1 Rs: black dotted
    - >=2 Rs: white dotted
    The maximum N is limited so the full circle fits within the current extent.
    """
    rsun_arc = float(h.get('RSUN', 0.0))
    if not np.isfinite(rsun_arc) or rsun_arc <= 0:
        return

    xmin, xmax, ymin, ymax = extent
    rx = max(abs(xmin), abs(xmax))
    ry = max(abs(ymin), abs(ymax))
    rlim = min(rx, ry)  # full circle must fit in both X and Y

    nmax = int(np.floor(rlim / rsun_arc))
    if nmax < 1:
        return

    th = np.linspace(0, 2*np.pi, 721)

    # 1 Rs: black dotted
    r = 1.0 * rsun_arc
    ax.plot(r*np.cos(th), r*np.sin(th), linestyle=":", color="k", linewidth=1.0)

    # 2..N Rs: white dotted
    for k in range(2, nmax + 1):
        r = k * rsun_arc
        ax.plot(r*np.cos(th), r*np.sin(th), linestyle=":", color="w", linewidth=1.0, alpha=0.9)


def plot_tB(f0: str, f120: str, f240: str,
            *, occultr_rsun: float=1.4,
            rmax_rsun: float=4.0,
            auto_bkg: bool=False,
            secchi_bkg_dir: str|None=None,
            save_png: bool=True,
            discri: bool|tuple|None=None,
            sebip_off: bool=False,
            tb_min_value: float=1.0):
    """High-level plotting routine for total brightness (tB)."""
    # First-stage calibration for each polarized frame
    im0, h0 = _read_and_prepare(f0, auto_bkg=auto_bkg, secchi_bkg_dir=secchi_bkg_dir, discri=discri, sebip_off=sebip_off)
    im1, h1 = _read_and_prepare(f120, auto_bkg=auto_bkg, secchi_bkg_dir=secchi_bkg_dir, discri=discri, sebip_off=sebip_off)
    im2, h2 = _read_and_prepare(f240, auto_bkg=auto_bkg, secchi_bkg_dir=secchi_bkg_dir, discri=discri, sebip_off=sebip_off)

    # Ensure shapes match
    if not (im0.shape == im1.shape == im2.shape):
        raise ValueError("Input images do not have identical shapes. Check trimming/rectify.")

    # Build cube in 0,120,240 order for quickpol
    cube = np.stack([im0, im1, im2], axis=-1)
    headers = [h0, h1, h2]

    # Solve for tB, pB, μ
    tB, pB, mu = cor1_quickpol(cube, header=headers)
    tB = tB[..., 0] if tB.ndim == 3 else tB

    # Geometry and masks
    ny, nx = tB.shape
    rsun_pix = _rsun_pix_from_header(h0)
    cx = float(h0.get('CRPIX1', (nx-1)/2))
    cy = float(h0.get('CRPIX2', (ny-1)/2))
    yy, xx = np.indices((ny, nx))
    r_pix = np.hypot(xx - cx, yy - cy)

    if rsun_pix == rsun_pix:  # finite
        mask_inner = (r_pix < occultr_rsun * rsun_pix)
        mask_outer = (r_pix > rmax_rsun * rsun_pix) if (rmax_rsun is not None and rmax_rsun > 0) else np.zeros_like(r_pix, bool)
        mask = np.logical_or(mask_inner, mask_outer)
    else:
        mask = np.zeros_like(r_pix, dtype=bool)

    # Also mask very small/invalid values (noise floor)
    tB_plot = np.where(np.logical_or(mask, (tB <= tb_min_value)), np.nan, tB)

    # Axes extent in arcsec
    extent = _extent_in_arcsec(h0, nx, ny)

    # Robust log scale limits
    finite_pos = np.isfinite(tB_plot) & (tB_plot > 0)
    if not finite_pos.any():
        raise RuntimeError("No positive finite tB values to plot after masking.")
    vmin = 1e7
    vmax = 1e8

    # Plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    # Colorbar size
    cbar_fraction = 0.025            # thickness
    cbar_pad = 0.01                  # gap to axes
    cbar_shrink = 0.50               # length scale
    cbar_aspect = 80                 # aspect ratio
    im = ax.imshow(tB_plot, origin='lower', extent=extent,
                   norm=LogNorm(vmin=vmin, vmax=vmax), cmap='plasma', aspect='equal')

    # Colorbar with adjustable size
    cb = plt.colorbar(im, ax=ax, fraction=cbar_fraction, pad=cbar_pad, shrink=cbar_shrink, aspect=cbar_aspect)
    cb.set_label(f"total brightness [B$_\\odot$] (tB $\\geq$ {tb_min_value:.1e})", fontsize=14)
    cb.ax.tick_params(labelsize=12)

    # Concentric Rs circles (1 Rs black dotted, >=2 Rs white dotted)
    _draw_rsun_circles(ax, h0, extent)

    # Labels & title
    timestr = h0.get('DATE-OBS', h0.get('DATE_OBS', 'Unknown'))
    pretty_t = timestr.split('.')[0].replace('T', ' ')
    ax.set_title(f"STEREO-A / SECCHI-COR1 tB ({pretty_t} UT)", fontsize=16)
    ax.set_xlabel("X [arcsec]")
    ax.set_ylabel("Y [arcsec]")

    plt.tight_layout()

    # Save or show
    if save_png:
        outdir = Path(f0).resolve().parent.parent  # .../COR1/Rawdata -> .../COR1
        out = outdir / f"plot_sta_tB_{timestr.replace(':','')}.png"
        fig.savefig(out, dpi=200)
        print(f"✓ tB plot saved: {out}")
        plt.show()
    else:
        plt.show()


if __name__ == "__main__":
    # ========================= Editable parameters ==========================
    # Paths to the three polarized frames (~0°, ~120°, ~240°).
    f0   = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata/20220613_030100_n4c1A.fts"
    f120 = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata/20220613_030118_n4c1A.fts"
    f240 = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata/20220613_030136_n4c1A.fts"

    # First-stage options
    auto_bkg = True                 # True → try to auto-find background
    secchi_bkg_dir = None            # e.g., "/path/to/SECCHI_BKG"
    sebip_off = False                # True → disable SEBIP correction
    discri = None                    # None, True, or (threshold, bias)

    # Plot options
    occ = 1.4                        # occulter mask radius [R_sun]
    rmax = 4.0                       # outer mask radius [R_sun] (None or <=0 to disable)
    tb_min_value = 1.0               # mask pixels with tB <= this value
    save = True                      # True → save, False → show



    # =======================================================================
    # Sanity: reorder by POLAR for safety (in case files were swapped)
    files, _ = _order_by_polar([f0, f120, f240])
    f0, f120, f240 = files

    # Execute
    plot_tB(f0, f120, f240,
            occultr_rsun=occ,
            rmax_rsun=rmax,
            auto_bkg=auto_bkg,
            secchi_bkg_dir=secchi_bkg_dir,
            save_png=save,
            discri=discri,
            sebip_off=sebip_off,
            tb_min_value=tb_min_value,
            )
