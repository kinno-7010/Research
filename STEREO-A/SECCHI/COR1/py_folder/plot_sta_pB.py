
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

import os, sys
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


def _normalize_path(p: str) -> str:
    # 例: D:\wsl\home\... → /mnt/d/wsl/home/...
    if os.name != "nt" and len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/")
        return f"/mnt/{drive}/{rest.lstrip('/')}"
    return p

def _token_for_degree(deg: float) -> str:
    """
    COR1 daily_med のファイルは p000 / p120 / p240 の3種が原則。
    任意のdegを、最も近いバケット {0, 120, 60(≡240)} に丸めて
    ファイル接尾辞 'p000' / 'p120' / 'p240' を返す。
    """
    dn = float(deg) % 180.0
    # 近いものに丸める（240°は60°と同一扱い）
    buckets = [(0.0, "p000"), (120.0, "p120"), (60.0, "p240")]
    return min(buckets, key=lambda b: abs(dn - b[0]))[1]


def _order_by_polar_to_targets(files: list[str], target_degrees=(0, 120, 240)):
    """
    与えられた3枚（またはN枚だが通常3枚）を、ヘッダ POLAR を読み取り、
    指定 target_degrees に最も近い順で並べ替える。
    戻り値: (ordered_files, ordered_headers)
      - ordered_files[i] は target_degrees[i] に対応するファイル
    """
    from astropy.io import fits
    meta = []
    for p in files:
        with fits.open(p, memmap=False) as hdul:
            h = {k.upper(): v for k, v in hdul[0].header.items()}
        pol = float(h.get('POLAR', 0.0))
        poln = pol % 180.0
        meta.append((p, h, poln))

    ordered_files = []
    ordered_headers = []
    used = set()
    for td in target_degrees:
        # それぞれのターゲット角に最も近い未使用フレームを選ぶ
        best = None
        best_err = 1e9
        for i, (p, h, poln) in enumerate(meta):
            if i in used:
                continue
            err = abs(poln - (td % 180.0))
            if err < best_err:
                best_err = err
                best = (i, p, h)
        if best is None:
            raise RuntimeError("POLAR並べ替えで対応ファイルが見つかりませんでした。")
        used.add(best[0])
        ordered_files.append(best[1])
        ordered_headers.append(best[2])

    return ordered_files, ordered_headers



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

def _draw_rsun_circles(ax, h: dict, extent: tuple[float, float, float, float]) -> None:
    """
    Draw concentric dotted circles at 1,2,3,... Rs on arcsec axes.
    - 1 Rs: black dotted
    - >=2 Rs: white dotted
    The maximum N is limited so the full circle fits within the current extent.
    """
    import numpy as np

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


def _extent_in_arcsec(h: dict, nx: int, ny: int) -> tuple[float, float, float, float]:
    """Return imshow extent in arcsec centered on solar disk using CRPIX and CDELT."""
    cd = float(h.get('CDELT1', 0.0)) or 1.0  # arcsec/pixel
    cx = float(h.get('CRPIX1', (nx - 1) / 2))
    cy = float(h.get('CRPIX2', (ny - 1) / 2))
    x = (np.array([0, nx]) - cx) * cd
    y = (np.array([0, ny]) - cy) * cd
    return (x[0], x[1], y[0], y[1])

def load_daily_med_backgrounds(p000_path: str,
                               degrees=(0, 120, 240),
                               p120_path: str | None = None,
                               p240_path: str | None = None):
    """
    daily_med の p000 を基準に、必要なら p120/p240 を同ディレクトリから探す。
    さらに p120_path / p240_path が明示指定された場合はそれを優先して使う。
      - いずれか欠けて読めなければ p000 をフォールバック。
      - 戻り値: dict {deg: (img, hdr)} （deg は引数そのままの数値）
    """
    p000_path = _normalize_path(p000_path)
    p120_path = _normalize_path(p120_path) if p120_path else None
    p240_path = _normalize_path(p240_path) if p240_path else None

    dirn, base = os.path.split(p000_path)

    def _try(path):
        try:
            im, hd = read_fits(path)
            return im, hd
        except Exception as e:
            print(f"[daily_med] Could not read: {path} ({e})")
            return None, None

    # まず p000（フォールバック用に必須）
    im000, h000 = _try(p000_path)
    if im000 is None:
        raise RuntimeError("p000 daily_med が読めませんでした。パスを確認してください。")

    # p120 候補
    if p120_path:
        im120, h120 = _try(p120_path)
    else:
        cand120 = os.path.join(dirn, base.replace("p000", "p120"))
        im120, h120 = _try(cand120)

    # p240 候補
    if p240_path:
        im240, h240 = _try(p240_path)
    else:
        # p000→p240 で推定
        cand240 = os.path.join(dirn, base.replace("p000", "p240"))
        im240, h240 = _try(cand240)

    # フォールバック処理
    if im120 is None:
        print("[daily_med] p120 が無いので p000 を流用します。")
        im120, h120 = im000, h000
    if im240 is None:
        print("[daily_med] p240 が無いので p000 を流用します。")
        im240, h240 = im000, h000

    out = {}
    for d in degrees:
        dn = float(d) % 180.0
        if abs(dn - 0.0) <= 1.0:
            out[d] = (im000, h000)
        elif abs(dn - 120.0) <= 1.0:
            out[d] = (im120, h120)
        else:
            # 60°(≡240°) 扱い
            out[d] = (im240, h240)
    return out


def proprocess_cor1_with_daily_med_background(degree: int, f_path: str, bkg_map: dict):
    """
    Daily-med backgroundを用いて、COR1データを前処理する。
    """
    bimg, bh = bkg_map[degree]
    im, hdr, _ = first_stage_calibration_and_background(
        f_path,
        exptime_off=False, bias_off=False, calfac_off=False, calimg_off=True,
        bkgimg_off=False, bkgimg=bimg, auto_bkg=False, sebip_off=False, silent=False
    )
    return im, hdr

def plot_pB(f0: str, f120: str, f240: str,
            *, occultr_rsun: float=1.4,
            auto_bkg: bool=False,
            secchi_bkg_dir: str|None=None,
            save_png: bool=True,
            discri: bool|tuple|None=None,
            sebip_off: bool=False,
            bkg_map: dict=None):
    """High-level plotting routine."""
    # First-stage calibration
    # 例：plot_pB(...) の先頭3行だけ差し替え
    im0, h0 = _read_and_prepare(f0,   bkgimg=bkg_map[0][0],   auto_bkg=False, sebip_off=sebip_off)
    im1, h1 = _read_and_prepare(f120, bkgimg=bkg_map[120][0], auto_bkg=False, sebip_off=sebip_off)
    im2, h2 = _read_and_prepare(f240, bkgimg=bkg_map[240][0], auto_bkg=False, sebip_off=sebip_off)


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
    # Mask inside occulter (occ * Rs) and also beyond 4 Rs

    mask_inner = (r_pix < occultr_rsun * rsun_pix)
    mask_outer = (r_pix > 4.0 * rsun_pix)
    mask = np.logical_or(mask_inner, mask_outer)


    # threshold = 7e3
    threshold = 0
    pB_plot = np.where(mask | (pB < threshold), np.nan, pB)

    # Extent in solar radii (R_sun) for axes
    extent = _extent_in_arcsec(h0, nx, ny)

    # Choose log scale limits robustly
    vmin = np.nanpercentile(pB_plot[pB_plot > 0], 5) if np.isfinite(pB_plot).any() else None
    vmax = np.nanpercentile(pB_plot, 99.7) if np.isfinite(pB_plot).any() else None
    vmin = max(vmin, 1e-6) if vmin else 1e-6

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    im = ax.imshow(pB_plot, origin='lower', extent=extent,
                   norm=LogNorm(vmin=vmin, vmax=vmax), cmap='plasma', aspect='equal')
    cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.01, aspect=80)
    cb.set_label(f"polarized brightness [B$_\\odot$] (pB $\\geq$ {threshold:.1e})", fontsize=14)
    cb.ax.tick_params(labelsize=12)
    
    _draw_rsun_circles(ax, h0, extent)


    TIME_STR = h0.get('DATE-OBS', h0.get('DATE_OBS', 'Unknown'))

    # Decorate
    ax.set_title(f"STEREO-A / SECCHI-COR1 pB ({TIME_STR.split('.')[0].replace('T', ' ')} UT)", fontsize=16)
    ax.set_xlabel("X [arcsec]")
    ax.set_ylabel("Y [arcsec]")
    # Draw solar limb at 1 R_sun
    if rsun_pix == rsun_pix:
        th = np.linspace(0, 2*np.pi, 360)
        ax.plot(np.cos(th), np.sin(th))

    plt.tight_layout()
    
    output_path = f"/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/plot_sta_pB_{TIME_STR.replace(':', '')}.png"
    fig.savefig(output_path, dpi=200)
    print(f"✓ pB plot saved: {output_path}")
    plt.show()


if __name__ == "__main__":
    # ========================= Editable parameters ==========================
    # Paths to the three polarized frames (~0°, ~120°, ~240°).
    f0   = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata/20220613_030100_n4c1A.fts"
    f120 = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata/20220613_030118_n4c1A.fts"
    f240 = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata/20220613_030136_n4c1A.fts"
    
    daily_med_p000 = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata/dc1A_p000_220613.fts"
    # 追加：ユーザー指定の p120 を明示
    daily_med_p120 = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata/dc1A_p120_220613.fts"
    # p240 は無ければ自動推定、あるなら明示で OK
    daily_med_p240 = daily_med_p000.replace("p000", "p240")

    bkg_map = load_daily_med_backgrounds(daily_med_p000)  # {0:(im,hdr),120:(im,hdr),240:(im,hdr)}


    # First-stage options
    auto_bkg = False             # True → try to auto-find background
    secchi_bkg_dir = None            # e.g., "/path/to/SECCHI_BKG"
    sebip_off = False                # True → disable SEBIP correction
    discri = None                    # None, True, or (threshold, bias)

    # Plot options
    occ = 1.4                        # occulter mask radius [R_sun]
    save = True                      # True → save instead of show

    # =======================================================================
    # Sanity: reorder by POLAR for safety (in case files were swapped)
    # ここだけ自由に変更可能（例： (0, 120, 240) → (0, 60, 120) など）
    target_degrees = (0, 120, 240)

    # ヘッダ POLAR を読んで target_degrees の順に並べ替え
    files, hdrs = _order_by_polar_to_targets([f0, f120, f240], target_degrees)
    # 角度ごとの daily_med 背景を取得
    bkg_map = load_daily_med_backgrounds(daily_med_p000, degrees=target_degrees, p120_path=daily_med_p120, p240_path=daily_med_p240)

    # 以降は既存の plot_pB をそのまま使う場合、引数が f0/f120/f240 のままなので、
    # 並べ替え後の files を対応して渡す（plot_pB 内部はそのまま）
    f0, f120, f240 = files


    # Execute
    # 実行部（__main__）の最後の呼び出しをこれに置き換え
    plot_pB(
    f0, f120, f240,
    occultr_rsun=occ,
    auto_bkg=False,             # ← 明示的に False（自動探索は使わない）
    secchi_bkg_dir=secchi_bkg_dir,
    save_png=save,
    discri=discri,
    sebip_off=sebip_off,
    bkg_map=bkg_map             # ← ここを必ず渡す
    )
