# -*- coding: utf-8 -*-
"""
2D_magnetic_field_map.py

Build a 2D magnetic-field map on the *same* grid as the 2D density map,
using PFSS and the plane-of-sky (POS) sampling in the spirit of
Zucca et al. (2014, A&A 564, A47).

Inputs:
  - LASCO pB (and optionally Mk4/K-Cor pB) FITS used for your density map
  - SDO/HMI magnetogram FITS for PFSS
Outputs:
  - 2D map of |B| (or Br) on the LASCO grid (same pixel count/positions as density map)
  - PNG figure
  - optional CSV dump of (x_pix, y_pix, r[Rsun], PA[deg], Br[G], |B|[G])

Dependencies: numpy, matplotlib, scipy, sunpy, pfsspy, astropy
Also reuses your local helpers:
  - io_and_processing.load_and_prepare_instrument_data, combine_corona_data
  - plotting_utils (for annuli overlays)
  - plot_hmi_pfss_overlay.prepare_hmi_for_pfss, compute_pfss_solution
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator
import astropy.units as u
from astropy.coordinates import SkyCoord
import sunpy.map
from sunpy.coordinates import frames
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- import your existing helpers (assume this file sits alongside them or PYTHONPATH is set) ---
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_SDO_HELPER_DIR = os.path.abspath(
    os.path.join(_BASE_DIR, "..", "..", "SDO_Mk4_SOHO", "pB", "py_folder")
)
for _extra_path in (_SDO_HELPER_DIR, _BASE_DIR):
    if _extra_path not in sys.path:
        sys.path.insert(0, _extra_path)
from io_and_processing import load_and_prepare_instrument_data, combine_corona_data
from plotting_utils import add_radial_guides_on_ax
import numpy as np
# PFSS helpers you already have
from plot_hmi_pfss_overlay import prepare_hmi_for_pfss, compute_pfss_solution

import warnings
# -----------------------------
# Grid construction (identical to density-map grid)
# -----------------------------
# --- PFSS compatibility helpers (add to 2D_magnetic_field_map.py) ---
# 置換：pfsspy.Output.get_bvec を SkyCoord 経由で呼ぶ版
# 追加：磁場の単位として妥当かを判定し、妥当ならその単位、なければ None を返す
def call_add_radial_guides(ax, r_map, r_ranges, params_lasco, extent_pixels):
    """
    plotting_utils.add_radial_guides_on_ax を安全に呼び出し、
    失敗したら自前で半径ガイド（mk4_inner / mk4_outer_lasco_inner / lasco_outer）を描く。
    """
    try:
        params_for_guides = _ensure_params_for_guides(params_lasco, extent_pixels, r_map)
        add_radial_guides_on_ax(ax, r_map, r_ranges, params_for_guides)
        return
    except Exception as e:
        print(f"[plotting] add_radial_guides_on_ax failed ({e}). Drawing fallback guides.")

    # --- フォールバック描画 ---
    legend_lines = []
    legend_info = {
        r_ranges['mk4_inner']: (f"{r_ranges['mk4_inner']:.1f} $R_\\odot$ (Mk4 inner)", 'magenta'),
        r_ranges['mk4_outer_lasco_inner']: (f"{r_ranges['mk4_outer_lasco_inner']:.1f} $R_\\odot$ (Mk4/LASCO)", 'green'),
        r_ranges['lasco_outer']: (f"{r_ranges['lasco_outer']:.1f} $R_\\odot$ (LASCO outer)", 'blue'),
    }
    for level_val, (label_text, color) in legend_info.items():
        try:
            ax.contour(r_map, levels=[level_val], colors=[color],
                       linewidths=1.2, linestyles='-.', extent=extent_pixels)
            proxy = plt.Line2D([0], [0], linestyle='-.', color=color, linewidth=1.2, label=label_text)
            legend_lines.append(proxy)
        except Exception as _:
            pass
    if legend_lines:
        ax.legend(handles=legend_lines, loc='upper right', fontsize=10)

def _ensure_params_for_guides(params_candidate, extent_pixels, r_map):
    """
    plotting_utils.add_radial_guides_on_ax が期待する
    {'cx','cy','nx','ny','px_per_rsun'} を必ず揃えて返す。
    """
    out = {}
    if isinstance(params_candidate, dict):
        out.update(params_candidate)

    # nx, ny
    if ('nx' not in out) or ('ny' not in out):
        if extent_pixels is not None and len(extent_pixels) == 4:
            out['nx'] = int(round(extent_pixels[1] - extent_pixels[0]))
            out['ny'] = int(round(extent_pixels[3] - extent_pixels[2]))
        else:
            ny, nx = r_map.shape
            out['nx'] = int(nx); out['ny'] = int(ny)

    # cx, cy
    if ('cx' not in out) or ('cy' not in out):
        if extent_pixels is not None and len(extent_pixels) == 4:
            out['cx'] = float(-extent_pixels[0])
            out['cy'] = float(-extent_pixels[2])
        else:
            out['cx'] = out['nx'] / 2.0
            out['cy'] = out['ny'] / 2.0

    # px_per_rsun を r_map から推定（中心行で r≈1 の点を探す粗い近似）
    if 'px_per_rsun' not in out or not np.isfinite(out.get('px_per_rsun', np.nan)):
        ny, nx = r_map.shape
        cyi = int(np.clip(round(out['cy']), 0, ny-1))
        row = r_map[cyi, :]
        valid = np.isfinite(row)
        if np.any(valid):
            idx = np.nanargmin(np.abs(row[valid] - 1.0))
            pos = np.arange(nx)[valid][idx]
            denom = max(row[valid][idx], 1e-6)
            px = abs(pos - out['cx']) / denom
            if np.isfinite(px) and px > 0:
                out['px_per_rsun'] = float(px)
            else:
                out['px_per_rsun'] = float(nx / (2 * np.nanmax(r_map)))
        else:
            out['px_per_rsun'] = float(nx / (2 * np.nanmax(r_map)))

    return out


def _mag_unit_or_none(unit_candidate):
    try:
        if unit_candidate is None:
            return None
        uobj = u.Unit(unit_candidate)
        if uobj.is_equivalent(u.G) or uobj.is_equivalent(u.T):
            return uobj
    except Exception:
        pass
    return None

def evaluate_pfss_via_get_bvec(pfss_solution, r_arr, th_arr, ph_arr, component="Babs"):
    """
    pfsspy.Output.get_bvec を用いて、(r,theta,phi) の各点で B を直接評価。
    返り値が無次元でも bunit（磁場単位）または Gauss を強制付与 → Gauss に統一。
    """
    obj = _unwrap_pfss(pfss_solution)
    get_bvec = _get_attr_or_key(obj, "get_bvec")
    if get_bvec is None:
        raise RuntimeError("PFSS object has no get_bvec; cannot fallback.")

    # POS → Stonyhurst
    lat = (np.pi/2.0) - np.asarray(th_arr, dtype=float)   # 緯度
    lon = np.asarray(ph_arr, dtype=float)
    rad = np.asarray(r_arr, dtype=float)

    lat1 = lat.ravel() * u.rad
    lon1 = lon.ravel() * u.rad
    rad1 = rad.ravel() * u.Rsun

    obstime = _get_attr_or_key(obj, "dtime")
    if hasattr(obstime, "isot") or hasattr(obstime, "jd"):
        coord = SkyCoord(lon=lon1, lat=lat1, radius=rad1,
                         frame=frames.HeliographicStonyhurst(obstime=obstime))
    else:
        coord = SkyCoord(lon=lon1, lat=lat1, radius=rad1,
                         frame=frames.HeliographicStonyhurst())

    # ベクトル場取得
    vec = get_bvec(coord)  # Quantity か ndarray の想定

    # 単位の正規化：bunit が磁場単位なら採用、ダメなら Gauss を付与
    bunit_attr = _get_attr_or_key(obj, "bunit")
    bunit = _mag_unit_or_none(bunit_attr)

    if isinstance(vec, u.Quantity):
        if vec.unit == u.one:
            vec = vec.value * (bunit if bunit is not None else u.G)
        elif not (vec.unit.is_equivalent(u.G) or vec.unit.is_equivalent(u.T)):
            warnings.warn(f"[PFSS] get_bvec returned non-magnetic unit ({vec.unit}); forcing Gauss.")
            vec = vec.value * (bunit if bunit is not None else u.G)
    else:
        vec = np.asarray(vec) * (bunit if bunit is not None else u.G)

    # Gauss に統一
    arr = vec.to(u.G).value
    arr = np.asarray(arr)

    # 形状正規化
    if arr.ndim >= 2 and arr.shape[0] == 3:
        V0, V1, V2 = arr[0], arr[1], arr[2]
    elif arr.ndim >= 2 and arr.shape[-1] == 3:
        V0, V1, V2 = arr[..., 0].ravel(), arr[..., 1].ravel(), arr[..., 2].ravel()
    else:
        raise RuntimeError("Unexpected shape from get_bvec; need (...,3) or (3,...)")

    # フレーム判定
    frame = str(_get_attr_or_key(obj, "coordinate_frame") or "").lower()
    if "spher" in frame:
        Br = V0; Bt = V1; Bp = V2
        Babs = np.sqrt(Br**2 + Bt**2 + Bp**2)
    else:
        # Cartesian → e_r 投影で Br
        Bx, By, Bz = V0, V1, V2
        lat1v = lat1.value; lon1v = lon1.value
        erx = np.cos(lat1v) * np.cos(lon1v)
        ery = np.cos(lat1v) * np.sin(lon1v)
        erz = np.sin(lat1v)
        Br = Bx * erx + By * ery + Bz * erz
        Babs = np.sqrt(Bx**2 + By**2 + Bz**2)

    return (Br if component.lower() == "br" else Babs).reshape(np.asarray(r_arr).shape)

def _as_value(arr, unit=None):
    """
    Astropy Quantity -> numpy 値。
    unit を指定:
      - arr がその単位に変換可能なら .to(unit).value
      - 変換不可能かつ arr が無次元なら「指定単位を付与」して値を返す
    unit 未指定: arr.value もしくは np.asarray(arr)
    """
    if hasattr(arr, "to") and hasattr(arr, "value"):
        if unit is not None:
            try:
                return arr.to(unit).value
            except Exception:
                # 無次元などで変換不可 → 指定単位を付与して値を返す（スケール変換はできないがクラッシュは防ぐ）
                try:
                    return (arr.value * unit).to(unit).value
                except Exception:
                    return np.asarray(arr.value)
        else:
            return arr.value
    return np.asarray(arr)

def _unwrap_pfss(obj):
    """
    pfsspy.Output / ラッパー / dict の可能性を考慮し、実体を取り出す。
    .solution / .output / .pfss / ['solution'] などを順に辿る。
    """
    tried = set()
    while True:
        if obj is None or id(obj) in tried:
            return obj
        tried.add(id(obj))

        # dict: よくあるキー名
        if isinstance(obj, dict):
            for k in ("solution", "output", "pfss", "out"):
                if k in obj:
                    obj = obj[k]
                    break
            else:
                return obj
            continue

        # オブジェクト: よくある属性名
        for attr in ("solution", "output", "pfss", "out"):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
                break
        else:
            return obj

def _get_attr_or_key(container, *names):
    "属性 or dict キーで最初に見つかったものを返す（なければ None）。"
    if container is None:
        return None
    for n in names:
        if hasattr(container, n):
            return getattr(container, n)
        if isinstance(container, dict) and n in container:
            return container[n]
    return None


# 置換：既存の resolve_pfss_axes をこの定義に差し替え
def resolve_pfss_axes(pfss_solution, fallback: dict | None = None):
    """
    (r[Rsun], theta[colat, rad], phi[rad]) の 1D 軸を返す。
    まず grid から取得。無ければ配列の形状 or fallback（rss など）から復元。
    """
    obj = _unwrap_pfss(pfss_solution)

    def _q(arr, unit):  # 量なら単位変換、素ならそのまま
        return _as_value(arr, unit) if hasattr(arr, "to") else (np.asarray(arr) if arr is not None else None)

    # 1) grid から直接
    g = _get_attr_or_key(obj, "grid")
    if g is not None:
        r_arr = _get_attr_or_key(g, "r", "rs", "radius", "r_grid")
        th_arr = _get_attr_or_key(g, "theta", "colatitude", "th")
        ph_arr = _get_attr_or_key(g, "phi", "longitude", "lon")
        if r_arr is not None and th_arr is not None and ph_arr is not None:
            return _q(r_arr, u.Rsun), _q(th_arr, u.rad), _q(ph_arr, u.rad)

    # 2) grid が不十分 → 形状だけ推定
    #    コンポーネントをフル抽出せず、存在チェックだけして shape を得る
    shapes_to_try = [
        _get_attr_or_key(obj, "br", "Br", "b_r", "br_rtp"),
        _get_attr_or_key(obj, "bt", "Bt", "b_t", "btheta", "btheta_rtp"),
        _get_attr_or_key(obj, "bp", "Bp", "b_p", "bphi", "bphi_rtp"),
        _get_attr_or_key(obj, "b", "B", "b_sph", "B_sph", "b_rtp", "B_rtp"),
    ]
    field_arr = None
    for arr in shapes_to_try:
        if arr is None:
            continue
        a = np.asarray(_as_value(arr))
        # (nr,nt,np) or (3,nr,nt,np) or (nr,nt,np,3)
        if a.ndim == 3:
            field_arr = a
            break
        if a.ndim >= 4:
            if a.shape[0] == 3:
                field_arr = a[0]
                break
            if a.shape[-1] == 3:
                field_arr = a[..., 0]
                break
    if field_arr is None:
        # 最終フォールバック：fallback のみで作る
        if not fallback or "rss" not in fallback:
            raise AttributeError("PFSS grid axes not found; please provide fallback={'rss':2.5, ...}.")
        nr = int(fallback.get("nrho", 50))
        ntheta = int(fallback.get("ntheta", 180))
        nphi = int(fallback.get("nphi", 360))
    else:
        nr, ntheta, nphi = field_arr.shape

    # rss を決める
    rss_obj = _get_attr_or_key(obj, "rss", "source_surface_radius", "ss")
    rss = _q(rss_obj, u.Rsun) if rss_obj is not None else None
    if rss is None and fallback:
        rss = float(fallback.get("rss", 2.5))
    if rss is None:
        rss = 2.5  # デフォルト

    r = np.linspace(1.0, float(rss), int(nr), dtype=float)
    th = np.linspace(0.0, np.pi, int(ntheta), dtype=float)
    ph = np.linspace(0.0, 2.0*np.pi, int(nphi), endpoint=False, dtype=float)
    return r, th, ph

def resolve_pfss_components(pfss_solution):
    """
    どんな入れ物でも (Br, Bt, Bp) を numpy 配列（Gauss）で返す。
    想定候補名を総当たりし、タプル/配列(3,...) や (...,3) も吸収。
    """
    obj = _unwrap_pfss(pfss_solution)

    # 0) pfsspy.Output: bg（球座標）→ そのまま (Br, Bt, Bp)、bc（直交座標）→ 球座標に変換
    bg = _get_attr_or_key(obj, "bg")
    if bg is not None:
        # bg は通常 (3, nr, nt, np) または (nr, nt, np, 3) の配列/Quantity
        bunit_attr = _get_attr_or_key(obj, "bunit")
        bunit = _mag_unit_or_none(bunit_attr) or u.G
        try:
            arr = _as_value(bg, u.G)
        except Exception:
            # 次善策：無次元なら bunit を付与して Gauss へ
            try:
                arr = (np.asarray(getattr(bg, "value", bg)) * bunit).to(u.G).value
            except Exception:
                arr = np.asarray(getattr(bg, "value", bg))
        arr = np.asarray(arr)
        if arr.ndim >= 4:
            if arr.shape[0] == 3:
                return arr[0], arr[1], arr[2]
            if arr.shape[-1] == 3:
                return arr[..., 0], arr[..., 1], arr[..., 2]
        # 形が想定外なら一般ルートへ継続

    bc = _get_attr_or_key(obj, "bc")
    if bc is not None:
        bunit_attr = _get_attr_or_key(obj, "bunit")
        bunit = _mag_unit_or_none(bunit_attr) or u.G
        try:
            arr = _as_value(bc, u.G)
        except Exception:
            try:
                arr = (np.asarray(getattr(bc, "value", bc)) * bunit).to(u.G).value
            except Exception:
                arr = np.asarray(getattr(bc, "value", bc))
        arr = np.asarray(arr)

        # (3, nr, nt, np) or (nr, nt, np, 3) を解釈
        Bx = By = Bz = None
        if arr.ndim >= 4:
            if arr.shape[0] == 3:
                Bx, By, Bz = arr[0], arr[1], arr[2]
            elif arr.shape[-1] == 3:
                Bx, By, Bz = arr[..., 0], arr[..., 1], arr[..., 2]
        if Bx is not None:
            # 軸取得して球座標成分へ変換
            r_ax, th_ax, ph_ax = resolve_pfss_axes(obj, fallback=None)
            TH, PH = np.meshgrid(th_ax, ph_ax, indexing="ij")  # (nt, np)
            sin_th = np.sin(TH)[None, ...]  # (1, nt, np) → broadcast
            cos_th = np.cos(TH)[None, ...]
            sin_ph = np.sin(PH)[None, ...]
            cos_ph = np.cos(PH)[None, ...]

            Br = Bx * (sin_th * cos_ph) + By * (sin_th * sin_ph) + Bz * (cos_th)
            Bt = Bx * (cos_th * cos_ph) + By * (cos_th * sin_ph) - Bz * (sin_th)
            Bp = -Bx * (sin_ph) + By * (cos_ph)
            return Br, Bt, Bp

    # 1) 代表的な属性名（pfsspy.Output 以外で Br, Bt, Bp を別々に持つケース）
    cand_sets = [
        ("br", "bt", "bp"),
        ("br", "btheta", "bphi"),
        ("Br", "Bt", "Bp"),
        ("b_r", "b_t", "b_p"),
        ("b_r", "b_theta", "b_phi"),
        ("Br_rtp", "Btheta_rtp", "Bphi_rtp"),
        ("br_rtp", "btheta_rtp", "bphi_rtp"),
    ]
    for names in cand_sets:
        Br = _get_attr_or_key(obj, names[0])
        Bt = _get_attr_or_key(obj, names[1])
        Bp = _get_attr_or_key(obj, names[2])
        if Br is not None and Bt is not None and Bp is not None:
            Br = _as_value(Br, u.G)
            Bt = _as_value(Bt, u.G)
            Bp = _as_value(Bp, u.G)
            return Br, Bt, Bp

    # 2) ベクトルをまとめて持つケース（タプル/list、shape が 3×… または …×3）
    pack = _get_attr_or_key(obj, "b", "B", "b_sph", "B_sph", "b_rtp", "B_rtp")
    if pack is not None:
        arr = _as_value(pack, u.G)
        arr = np.asarray(arr)
        if arr.ndim >= 4:
            if arr.shape[0] == 3:
                Br, Bt, Bp = arr[0], arr[1], arr[2]
                return Br, Bt, Bp
            if arr.shape[-1] == 3:
                Br, Bt, Bp = arr[..., 0], arr[..., 1], arr[..., 2]
                return Br, Bt, Bp

    # 3) dict や tuple など、名前付きコンテナ
    if isinstance(obj, (tuple, list)) and len(obj) == 3:
        Br, Bt, Bp = (np.asarray(_as_value(x, u.G)) for x in obj)
        return Br, Bt, Bp
    if isinstance(obj, dict):
        for keys in (
            ("Br", "Bt", "Bp"),
            ("br", "bt", "bp"),
            ("b_r", "b_t", "b_p"),
            ("b_r", "b_theta", "b_phi"),
        ):
            if all(k in obj for k in keys):
                Br = _as_value(obj[keys[0]], u.G)
                Bt = _as_value(obj[keys[1]], u.G)
                Bp = _as_value(obj[keys[2]], u.G)
                return Br, Bt, Bp

    # 4) 取得失敗：型と属性のヒントを出す
    public_attrs = [a for a in dir(obj) if not a.startswith("_")]
    msg = (
        "Unsupported PFSS object for components. "
        f"type={type(obj)} attrs={public_attrs[:30]}"
    )
    raise TypeError(msg)

def ensure_ascending_axes_and_reorder(r_ax, th_ax, ph_ax, Br, Bt, Bp):
    """
    RegularGridInterpolator 用に軸を昇順へ。必要に応じて配列を反転。
    """
    if r_ax[0] > r_ax[-1]:
        r_ax = r_ax[::-1]
        Br = Br[::-1, :, :]
        Bt = Bt[::-1, :, :]
        Bp = Bp[::-1, :, :]
    if th_ax[0] > th_ax[-1]:
        th_ax = th_ax[::-1]
        Br = Br[:, ::-1, :]
        Bt = Bt[:, ::-1, :]
        Bp = Bp[:, ::-1, :]
    if ph_ax[0] > ph_ax[-1]:
        ph_ax = ph_ax[::-1]
        Br = Br[:, :, ::-1]
        Bt = Bt[:, :, ::-1]
        Bp = Bp[:, :, ::-1]
    return r_ax, th_ax, ph_ax, Br, Bt, Bp

def make_lasco_grid_and_final_image(
    filename_lasco: str,
    filename_mk4: str | None,
    r_ranges: dict,
):
    """
    Replicates the exact grid used by the 2D density map:
    - uses LASCO header as reference grid
    - optionally stitches Mk4/K-Cor into the inner annulus
    Returns:
        final_image        : 2D pB (stitched) on LASCO grid (for plotting extent/masks only)
        r_map_lasco        : 2D heliocentric distance in Rsun at each pixel (LASCO grid)
        theta_map_lasco    : 2D position angle [rad], measured from +X to +Y (same as your density code)
        params_lasco       : dict with LASCO geometry
    """
    # Load LASCO
    data_lasco, params_lasco = load_and_prepare_instrument_data(filename_lasco, "SOHO/LASCO", is_lasco=True)

    # Build r, theta on LASCO grid
    y_idx, x_idx = np.indices((params_lasco["ny"], params_lasco["nx"]))
    x_norm = (x_idx - params_lasco["cx"]) / params_lasco["px_per_rsun"]
    y_norm = (y_idx - params_lasco["cy"]) / params_lasco["px_per_rsun"]
    r_map_lasco = np.hypot(x_norm, y_norm)              # [Rsun]
    theta_map_lasco = np.arctan2(y_norm, x_norm)        # [rad], 0 along +X, CCW

    # Optionally stitch Mk4/K-Cor to replicate density-grid content (ensures identical masks)
    if filename_mk4 is not None and os.path.exists(filename_mk4):
        data_mk4, params_mk4 = load_and_prepare_instrument_data(filename_mk4, "Mk4/K-Cor", is_lasco=False)
        final_image = combine_corona_data(
            data_lasco, params_lasco, data_mk4, params_mk4, r_map_lasco, r_ranges
        )
    else:
        final_image = data_lasco.copy()
        final_image[r_map_lasco < r_ranges.get("mk4_outer_lasco_inner", 2.2)] = np.nan
        final_image[r_map_lasco > r_ranges.get("lasco_outer", 7.0)] = np.nan

    return final_image, r_map_lasco, theta_map_lasco, params_lasco


# -----------------------------
# PFSS sampling on the POS grid
# -----------------------------
# 既存の build_pfss_interpolators(...) を下で置き換え
from scipy.interpolate import RegularGridInterpolator

# 置換：既存の build_pfss_interpolators をこの定義に差し替え
def build_pfss_interpolators(pfss_solution, fallback: dict | None = None):
    r_ax, th_ax, ph_ax = resolve_pfss_axes(pfss_solution, fallback=fallback)
    Br, Bt, Bp = resolve_pfss_components(pfss_solution)
    r_ax, th_ax, ph_ax, Br, Bt, Bp = ensure_ascending_axes_and_reorder(
        r_ax, th_ax, ph_ax, Br, Bt, Bp
    )
    axes = (r_ax, th_ax, ph_ax)
    interp_Br = RegularGridInterpolator(axes, Br, bounds_error=False, fill_value=np.nan)
    interp_Bt = RegularGridInterpolator(axes, Bt, bounds_error=False, fill_value=np.nan)
    interp_Bp = RegularGridInterpolator(axes, Bp, bounds_error=False, fill_value=np.nan)
    return {"Br": interp_Br, "Bt": interp_Bt, "Bp": interp_Bp, "axes": axes}


def pos_to_stonyhurst_latlon(
    theta_map: np.ndarray,
    # PA definition: your theta_map is angle from +X, CCW. Convert to PA from solar north:
    b0_deg: float = 0.0,
):
    """
    Convert your position-angle definition to heliographic latitude (approx).
    We adopt the POS approximation like Zucca+2014: sample along the plane-of-sky.
    For limb cases this is robust. For disk-center events, consider full HPC->HGS transforms.

    theta_map: 0 at +X (east), CCW.  (np.arctan2(y, x))
    PA from solar north (counterclockwise) is: PA = 90° - theta_deg.
    Then approx heliographic latitude = PA - B0 (tilt correction).

    Returns: lat_map [rad], with range [-pi/2, +pi/2]
    """
    theta_deg = np.degrees(theta_map)
    pa_from_north = 90.0 - theta_deg
    lat_deg = pa_from_north - b0_deg
    lat_rad = np.radians(np.clip(lat_deg, -90.0, 90.0))
    return lat_rad


def sample_pfss_on_pos_grid(
    pfss_solution,
    r_map_rsun: np.ndarray,
    theta_map: np.ndarray,
    pos_longitude_deg: float = 90.0,
    component: str = "Br",
    b0_deg: float = 0.0,
    pfss_fallback: dict | None = None,
):
    """
    POS グリッド上で PFSS をサンプリング。
    まず RegularGridInterpolator による補間を試み、ダメなら get_bvec で直接評価。
    """
    # Stonyhurst 緯度（B0 補正つき）→ PFSS の colat = pi/2 - lat
    lat_rad = pos_to_stonyhurst_latlon(theta_map, b0_deg=b0_deg)
    colat_rad = (np.pi / 2.0) - lat_rad
    lon_rad = np.radians((pos_longitude_deg % 360.0))

    rr = r_map_rsun
    tt = colat_rad
    pp = np.full_like(rr, lon_rad)

    # 1) 通常経路：Br/Bt/Bp の補間器
    try:
        interps = build_pfss_interpolators(pfss_solution, fallback=pfss_fallback)
        pts = np.stack([rr, tt, pp], axis=-1)  # (...,3)
        Br = interps["Br"](pts)
        if component.lower() == "br":
            return Br
        Bt = interps["Bt"](pts)
        Bp = interps["Bp"](pts)
        return np.sqrt(Br**2 + Bt**2 + Bp**2)
    except Exception as e:
        print(f"[PFSS] Interpolator path failed ({e}). Falling back to get_bvec...")

    # 2) フォールバック：get_bvec で直接評価
    return evaluate_pfss_via_get_bvec(pfss_solution, rr, tt, pp, component=component)


# -----------------------------
# Plotting
# -----------------------------
def plot_B_map(
    B_map: np.ndarray,
    r_map: np.ndarray,
    params_lasco: dict,
    r_ranges: dict,
    title: str = "2D Magnetic Field Map from PFSS (POS sampling)",
    out_png: str | None = None,
    xlim_pix: tuple[float, float] | None = None,
    ylim_pix: tuple[float, float] | None = None,
    vmin: float = 0.01,
    vmax: float = 1.0,
):
    """
    Show |B| (or Br) on the LASCO grid, with the same look-and-feel as your density map.

    Axes coordinates:
        X, Y = pixels from Sun center (0,0).
    Parameters
    ----------
    xlim_pix, ylim_pix : (min, max) in pixels from Sun center (optional)
        e.g., xlim_pix = (-700, 700), ylim_pix = (0, 1400)
    vmin, vmax : float
        Color scale range for B [G] in LogNorm.
    """
    from matplotlib.colors import LogNorm

    ny, nx = B_map.shape
    extent_pixels = [
        -params_lasco["cx"],
        params_lasco["nx"] - params_lasco["cx"],
        -params_lasco["cy"],
        params_lasco["ny"] - params_lasco["cy"],
    ]

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(
        B_map,
        origin="lower",
        cmap="plasma",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        extent=extent_pixels,
        aspect="equal",
    )

    # Overplot integer Rsun contours for guidance
    int_levels = np.arange(1, int(np.floor(r_ranges["lasco_outer"])) + 1)
    ax.contour(
        r_map,
        levels=int_levels,
        colors="white",
        linewidths=1.0,
        linestyles="--",
        alpha=0.5,
        extent=extent_pixels,
    )

    # Inner/outer annuli same as density-figure
    call_add_radial_guides(ax, r_map, r_ranges, params_lasco, extent_pixels)

    # 太陽中心マーク
    ax.plot(0, 0, "+", color="black", markersize=12, markeredgewidth=1.5)

    # ★ ここで表示範囲をピクセル単位で指定 ★
    if xlim_pix is not None:
        ax.set_xlim(xlim_pix)
    if ylim_pix is not None:
        ax.set_ylim(ylim_pix)
    # -----------------------------------------

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("X [pixels from Sun center]")
    ax.set_ylabel("Y [pixels from Sun center]")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.1)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label("Magnetic Field Strength [Gauss]", fontsize=14)

    plt.tight_layout()
    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {out_png}")
    plt.show()

    return fig, ax


# -----------------------------
# CSV dump (optional)
# -----------------------------
def export_B_csv(
    out_csv: str,
    B_map: np.ndarray,
    r_map: np.ndarray,
    theta_map: np.ndarray,
    params_lasco: dict,
):
    """
    Save (x_pix, y_pix, r[Rsun], PA[deg], B[G]) per valid pixel.
    """
    ny, nx = B_map.shape
    y_idx, x_idx = np.indices((ny, nx))
    mask = np.isfinite(B_map)

    x_pix = x_idx[mask]
    y_pix = y_idx[mask]
    r_val = r_map[mask]
    pa_deg = np.degrees(theta_map[mask])
    B_val = B_map[mask]

    arr = np.column_stack([x_pix, y_pix, r_val, pa_deg, B_val])
    header = "x_pix,y_pix,r_Rsun,PA_deg,B_G"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    np.savetxt(out_csv, arr, delimiter=",", header=header, comments="")
    print(f"✓ CSV exported: {out_csv}  ({arr.shape[0]} points)")


# -----------------------------
# Main
# -----------------------------
def main():
    # ---- Paths (edit to your environment) ----
    # pB / density-map side (same as you used for 2D density)
    filename_mk4 = r"/mnt/d/wsl/home/kinno-7010/Research_data/MK4_coronagraph/MK4_coronagraph_KCOR/pB/20220613_025810_kcor_l2.fts"
    filename_lasco = r"/mnt/d/wsl/home/kinno-7010/Research_data/SOHO/pB/C2-PB-20220613_0258.fts"

    # PFSS / HMI input
    hmi_file = r"/mnt/d/wsl/home/kinno-7010/Research_data/SDO/HMI/Rawdata/hmi.M_720s.20110922_090000_TAI.fits"

    # Same annulus as density figure
    r_ranges = {"mk4_inner": 1.0, "mk4_outer_lasco_inner": 2.2, "lasco_outer": 7.0}

    # ---- Step 1: replicate LASCO grid (same as density) ----
    final_image, r_map, theta_map, params_lasco = make_lasco_grid_and_final_image(
        filename_lasco, filename_mk4, r_ranges
    )

    # ---- Step 2: PFSS solution from HMI ----
    # Use your helper to prepare HMI and compute PFSS
    # ---- Step 2: PFSS solution from HMI ----
    # Use your helper to prepare HMI and compute PFSS
    hmi_data = prepare_hmi_for_pfss(hmi_file)

    # HMI フルディスク SunPy Map（PFSS の下端境界）
    hmi_map = hmi_data["full_map"]          # ← 追加

    # PFSS グリッド設定（Zucca+2014 相当：rss ~ 2.5 R⊙、nrho は 50 程度）
    nrho = 50                               # ← 追加
    rss = 2.5

    # pfsspy.Output を直接受け取る（辞書アクセスしない）
    pfss_solution = compute_pfss_solution(hmi_map, nrho=nrho, rss=rss)

    r_ax, _, _ = resolve_pfss_axes(pfss_solution, fallback={"rss": rss, "nrho": nrho})
    print(f"PFSS ready: rss={rss} Rs, nrho={nrho}, grid r∈[{np.nanmin(r_ax):.2f},{np.nanmax(r_ax):.2f}] R⊙")

    # ---- Step 3: sample PFSS on the POS grid (same pixels) ----
    # For a limb event use pos_longitude_deg ~ 90. For disk center, ~0 deg.
    B_map = sample_pfss_on_pos_grid(
        pfss_solution,
        r_map_rsun=r_map,
        theta_map=theta_map,
        pos_longitude_deg=90.0,   # East/West limb POS slice
        component="Babs",         # "Br" or "Babs"
        b0_deg=float(hmi_data["full_map"].meta.get("CRLT_OBS", 0.0)),  # solar B0 tilt
        pfss_fallback={"rss": rss, "nrho": nrho},
    )

    # Mask outside the valid annulus (exactly same as density map used)
    B_map = np.where((r_map >= r_ranges["mk4_inner"]) & (r_map <= r_ranges["lasco_outer"]), B_map, np.nan)
    
        # Output directory (adjust if needed)
    out_dir = r"/mnt/d/wsl/home/kinno-7010/Research_data/PFSS"
    # out_png = os.path.join(out_dir, f"2D_magnetic_field_map_rss={rss}.png")
    # out_csv = os.path.join(out_dir, f"2D_magnetic_field_map_rss={rss}.csv")
    out_png = os.path.join(out_dir, f"2D_magnetic_field_map_rss={rss}_20110922_0900.png")
    out_csv = os.path.join(out_dir, f"2D_magnetic_field_map_rss={rss}_20110922_0900.csv")

    # ---- Step 4: plot ----
    plot_B_map(
        B_map,
        r_map,
        params_lasco,
        r_ranges,
        # title="2D Magnetic Field Map from PFSS (POS, $R_{\\mathrm{ss}}$="+f"{rss}"+" $R_\\odot$)",
        title="2D Magnetic Field Map from PFSS (POS, $R_{\\mathrm{ss}}$="+f"{rss}"+" $R_\\odot$)\n2011-09-22 09:00:00 UT",
        out_png=out_png,
        # xlim_pix=(-150, 0),
        # ylim_pix=(-100, 150),
        vmin=0.01,
        vmax=10.0,
    )

    # ---- Step 5: optional CSV ----
    export_B_csv(out_csv, B_map, r_map, theta_map, params_lasco)


if __name__ == "__main__":
    main()
