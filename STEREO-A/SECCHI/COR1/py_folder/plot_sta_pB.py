
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_sta_pB.py
==============
Plot a 2-D polarized brightness (pB) image from STEREO-A / SECCHI-COR1 using three
polarization frames (0°, 120°, 240°).

This version lets you change all parameters in the `if __name__ == "__main__":` block
at the bottom of the file (no command-line parsing required).

Processing chain:
  1) First-stage calibration & optional background subtraction (secchi_prep.py)
  2) Fast Stokes solving for tB, pB, μ (cor1_quickpol.py)
  3) Matplotlib visualization of pB in log scale with axes in R_sun
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Make sure we can import the sibling helper modules when the script sits next to them.
sys.path.append(str(Path(__file__).resolve().parent))

from secchi_prep import (
    first_stage_calibration_and_background,
    read_fits,
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
    # target buckets: 0 -> i0, 120 -> i120, 60 (≡240) -> i240
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


def _extent_in_rsun(h: dict, nx: int, ny: int) -> tuple[float,float,float,float]:
    """Return imshow extent in solar radii centered on solar disk using CRPIX and RSUN/CDELT."""
    rsun_arc = float(h.get('RSUN', 0.0))
    cd = float(h.get('CDELT1', 0.0)) or 1.0
    cx = float(h.get('CRPIX1', (nx-1)/2))
    cy = float(h.get('CRPIX2', (ny-1)/2))
    # arcsec per pixel offsets
    x = (np.array([0, nx]) - cx) * cd
    y = (np.array([0, ny]) - cy) * cd
    # convert arcsec -> R_sun
    scale = (1.0 / rsun_arc) if rsun_arc > 0 else 1.0
    return (x[0]*scale, x[1]*scale, y[0]*scale, y[1]*scale)


def plot_pB(f0: str, f120: str, f240: str,
            *, occultr_rsun: float=1.4,
            auto_bkg: bool=False,
            secchi_bkg_dir: str|None=None,
            save_png: str|None=None,
            show_tb: bool=False,
            discri: bool|tuple|None=None,
            sebip_off: bool=False):
    """High-level plotting routine."""
    # First-stage calibration
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
    pB = pB[..., 0] if pB.ndim == 3 else pB

    ny, nx = pB.shape
    # Occulter masking in R_sun
    rsun_pix = _rsun_pix_from_header(h0)
    cx = float(h0.get('CRPIX1', (nx-1)/2))
    cy = float(h0.get('CRPIX2', (ny-1)/2))
    yy, xx = np.indices((ny, nx))
    r_pix = np.hypot(xx - cx, yy - cy)
    mask = (rsun_pix == rsun_pix) and (r_pix < occultr_rsun * rsun_pix)  # nan-safe

    pB_plot = np.where(mask, np.nan, pB)
    tB_plot = np.where(mask, np.nan, tB)

    # Extent in solar radii (R_sun) for axes
    extent = _extent_in_rsun(h0, nx, ny)

    # Choose log scale limits robustly
    vmin = np.nanpercentile(pB_plot[pB_plot > 0], 5) if np.isfinite(pB_plot).any() else None
    vmax = np.nanpercentile(pB_plot, 99.7) if np.isfinite(pB_plot).any() else None
    vmin = max(vmin, 1e-6) if vmin else 1e-6

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(pB_plot, origin='lower', extent=extent,
                   norm=LogNorm(vmin=vmin, vmax=vmax))
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("pB (arb. units)")

    # Decorate
    ax.set_title("STEREO-A / SECCHI-COR1 pB")
    ax.set_xlabel("X [R$_\\odot$]")
    ax.set_ylabel("Y [R$_\\odot$]")
    # Draw solar limb at 1 R_sun
    if rsun_pix == rsun_pix:
        th = np.linspace(0, 2*np.pi, 360)
        ax.plot(np.cos(th), np.sin(th))

    plt.tight_layout()

    if show_tb:
        vmin_t = np.nanpercentile(tB_plot[tB_plot > 0], 5) if np.isfinite(tB_plot).any() else None
        vmax_t = np.nanpercentile(tB_plot, 99.7) if np.isfinite(tB_plot).any() else None
        vmin_t = max(vmin_t, 1e-6) if vmin_t else 1e-6
        fig2 = plt.figure(figsize=(6, 6))
        ax2 = fig2.add_subplot(111)
        im2 = ax2.imshow(tB_plot, origin='lower', extent=extent,
                         norm=LogNorm(vmin=vmin_t, vmax=vmax_t))
        cb2 = plt.colorbar(im2, ax=ax2)
        cb2.set_label("tB (arb. units)")
        ax2.set_title("STEREO-A / SECCHI-COR1 tB")
        ax2.set_xlabel("X [R$_\\odot$]")
        ax2.set_ylabel("Y [R$_\\odot$]")
        th = np.linspace(0, 2*np.pi, 360)
        ax2.plot(np.cos(th), np.sin(th))
        plt.tight_layout()

    if save_png:
        out = Path(save_png).expanduser()
        fig.savefig(out, dpi=200)
        if show_tb:
            out2 = out.with_name(out.stem + "_tB.png")
            plt.figure(fig2.number)
            fig2.savefig(out2, dpi=200)
        print(f"[plot_sta_pB] Saved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    # ========================= Editable parameters ==========================
    # Paths to the three polarized frames (~0°, ~120°, ~240°).
    f0   = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata/20220613_030100_n4c1A.fts"
    f120 = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata/20220613_030118_n4c1A.fts"
    f240 = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata/20220613_030136_n4c1A.fts"

    # First-stage options
    auto_bkg = False                 # True → try to auto-find background
    secchi_bkg_dir = None            # e.g., "/path/to/SECCHI_BKG"
    sebip_off = False                # True → disable SEBIP correction
    discri = None                    # None, True, or (threshold, bias)

    # Plot options
    occ = 1.4                        # occulter mask radius [R_sun]
    show_tb = False                  # also render tB
    save = None                      # e.g., "sta_cor1_pB.png" to save instead of show

    # =======================================================================
    # Sanity: reorder by POLAR for safety (in case files were swapped)
    files, hdrs = _order_by_polar([f0, f120, f240])
    f0, f120, f240 = files

    # Execute
    plot_pB(f0, f120, f240,
            occultr_rsun=occ,
            auto_bkg=auto_bkg,
            secchi_bkg_dir=secchi_bkg_dir,
            save_png=save,
            show_tb=show_tb,
            discri=discri,
            sebip_off=sebip_off)
