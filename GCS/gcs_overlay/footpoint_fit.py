# gcs_overlay/footpoint_fit.py
from dataclasses import replace
import numpy as np
from astropy import units as u
from typing import Tuple, Dict, List

from .gcs_geometry import GCSParams  # 既存の dataclass をそのまま利用

# ────────────────────────────────────────────────────────────────────────────
# 角度/ベクトルユーティリティ
# ────────────────────────────────────────────────────────────────────────────
def _deg2rad(d): return np.deg2rad(float(d))
def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0.0, 1.0, n)

def _lonlat_to_unitvec(lon_deg: float, lat_deg: float) -> np.ndarray:
    lon = _deg2rad(lon_deg); lat = _deg2rad(lat_deg)
    return np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)], dtype=float)

def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    u = _unit(u); v = _unit(v)
    c = np.clip(np.sum(u*v, axis=-1), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(c)))

# ────────────────────────────────────────────────────────────────────────────
# HGS 基底（ex, ey, ez）を作る：ez＝軸方向（lon,lat）、ex,ey は tilt で回転
# （gcs_geometry.py の実装と整合するよう再実装）
# ────────────────────────────────────────────────────────────────────────────
def _hgs_axes_from_axis(lon_rad: float, lat_rad: float, tilt_rad: float):
    ez = np.array([
        np.cos(lat_rad)*np.cos(lon_rad),
        np.cos(lat_rad)*np.sin(lon_rad),
        np.sin(lat_rad)
    ], dtype=float)
    ez = _unit(ez)

    z_world = np.array([0.0, 0.0, 1.0])
    ref = z_world if abs(np.dot(ez, z_world)) <= 0.98 else np.array([1.0, 0.0, 0.0])
    t0 = _unit(np.cross(ez, ref))
    b0 = _unit(np.cross(ez, t0))

    c, s = np.cos(tilt_rad), np.sin(tilt_rad)
    ex = _unit(c * t0 + s * b0)
    ey = _unit(np.cross(ez, ex))
    return ex, ey, ez

# ────────────────────────────────────────────────────────────────────────────
# 脚（leg）の円錐：軸＝ ±y_b（ボディ座標），開角＝alpha
# 円錐上の任意方位 phi の方向ベクトル（ボディ）を HGS に回す
# ────────────────────────────────────────────────────────────────────────────
def _cone_directions_hgs(alpha_rad: float, lon_rad: float, lat_rad: float, tilt_rad: float,
                         n_phi: int = 360) -> Tuple[np.ndarray, np.ndarray]:
    """
    2つの円錐（±y_b を軸）の方向ベクトル（HGS, 単位ベクトル）を返す。
    返り値: (dirs_plus, dirs_minus) それぞれ shape = (n_phi, 3)
    """
    ex, ey, ez = _hgs_axes_from_axis(lon_rad, lat_rad, tilt_rad)

    # ボディ座標の円錐軸（±y_b）
    ax_plus  = np.array([0.0,  1.0, 0.0], dtype=float)
    ax_minus = np.array([0.0, -1.0, 0.0], dtype=float)

    def basis_around(axis):
        axis = _unit(axis)
        ref = np.array([1.0,0.0,0.0]) if abs(axis[0]) < 0.9 else np.array([0.0,0.0,1.0])
        u = _unit(np.cross(axis, ref))
        v = _unit(np.cross(axis, u))
        return axis, u, v

    axp, up, vp = basis_around(ax_plus)
    axm, um, vm = basis_around(ax_minus)

    phi = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)

    # 円錐上の方向（ボディ座標）
    # dir_b = cos(alpha)*axis + sin(alpha)*(cos(phi)*u + sin(phi)*v)
    cosA, sinA = np.cos(alpha_rad), np.sin(alpha_rad)
    cp, sp = np.cos(phi), np.sin(phi)

    dir_b_plus  = _unit(cosA*axp[None,:] + sinA*(cp[:,None]*up[None,:] + sp[:,None]*vp[None,:]))
    dir_b_minus = _unit(cosA*axm[None,:] + sinA*(cp[:,None]*um[None,:] + sp[:,None]*vm[None,:]))

    # ボディ→HGS 回転（列 [ex ey ez]）
    R = np.stack([ex, ey, ez], axis=1)  # 3x3
    dirs_plus_hgs  = dir_b_plus  @ R.T
    dirs_minus_hgs = dir_b_minus @ R.T
    return _unit(dirs_plus_hgs), _unit(dirs_minus_hgs)

# ────────────────────────────────────────────────────────────────────────────
# 1点発生源：最適 tilt を探索
# ────────────────────────────────────────────────────────────────────────────
def find_best_tilt_for_source(params: GCSParams,
                              lon_src_deg: float, lat_src_deg: float,
                              tilt_search_deg: Tuple[float,float]=(-180.0, 180.0),
                              tilt_step_deg: float = 1.0,
                              n_phi: int = 360) -> Dict:
    """
    発生源 (lon_src, lat_src) に最も近い脚（左右どちらか）の円錐方位を選び，
    その最小角距離が最小になる tilt を粗探索で求める。
    返り値: {
      'tilt_deg': 最良tilt,
      'min_angle_deg': その時の最小角距離,
      'which_leg': '+y' or '-y',
      'footpoint_lonlat_deg': (lon_fp, lat_fp)
    }
    """
    # 発生源ユニットベクトル
    s = _lonlat_to_unitvec(lon_src_deg, lat_src_deg)

    lon_rad = _deg2rad(params.lon_deg)
    lat_rad = _deg2rad(params.lat_deg)
    alpha   = _deg2rad(params.alpha_deg)

    best = {'tilt_deg': None, 'min_angle_deg': 1e9, 'which_leg': None, 'footpoint_lonlat_deg': (None, None)}

    lo, hi = tilt_search_deg
    grid = np.arange(lo, hi + 1e-9, tilt_step_deg, dtype=float)
    for tilt in grid:
        tilt_rad = _deg2rad(tilt)
        dirs_p, dirs_m = _cone_directions_hgs(alpha, lon_rad, lat_rad, tilt_rad, n_phi=n_phi)

        # 角距離（円錐上の最接近点を探す）
        # plus cone
        ang_p = np.rad2deg(np.arccos(np.clip(dirs_p @ s, -1.0, 1.0)))
        idx_p = int(np.argmin(ang_p)); val_p = float(ang_p[idx_p]); fp_p = dirs_p[idx_p]

        # minus cone
        ang_m = np.rad2deg(np.arccos(np.clip(dirs_m @ s, -1.0, 1.0)))
        idx_m = int(np.argmin(ang_m)); val_m = float(ang_m[idx_m]); fp_m = dirs_m[idx_m]

        # どちらか近い方
        if val_p <= val_m:
            val = val_p; fp = fp_p; leg = '+y'
        else:
            val = val_m; fp = fp_m; leg = '-y'

        if val < best['min_angle_deg']:
            # footpoint 方向ベクトル → lon,lat
            x,y,z = fp
            lon_fp = float(np.rad2deg(np.arctan2(y, x)))
            lat_fp = float(np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0))))
            best = {
                'tilt_deg': float(tilt),
                'min_angle_deg': val,
                'which_leg': leg,
                'footpoint_lonlat_deg': (lon_fp, lat_fp)
            }
    return best

# ────────────────────────────────────────────────────────────────────────────
# 2点発生源（左右脚のペア）：最適 tilt を探索
# ────────────────────────────────────────────────────────────────────────────
def find_best_tilt_for_two_sources(params: GCSParams,
                                   src1_lonlat_deg: Tuple[float,float],
                                   src2_lonlat_deg: Tuple[float,float],
                                   tilt_search_deg: Tuple[float,float]=(-180.0, 180.0),
                                   tilt_step_deg: float = 1.0,
                                   n_phi: int = 360) -> Dict:
    """
    2つの発生源（左右脚を意図）に対し，
    円錐（±y_b）上の2点を選んで**最小の総角距離**になる tilt を選ぶ。
    2点の割り当て（どちらが +y/-y）も同時に最小化。
    """
    s1 = _lonlat_to_unitvec(*src1_lonlat_deg)
    s2 = _lonlat_to_unitvec(*src2_lonlat_deg)

    lon_rad = _deg2rad(params.lon_deg)
    lat_rad = _deg2rad(params.lat_deg)
    alpha   = _deg2rad(params.alpha_deg)

    best = {'tilt_deg': None, 'min_total_angle_deg': 1e9,
            'assign': None, # ('+y','-y') など
            'footpoints_lonlat_deg': (None, None)}

    lo, hi = tilt_search_deg
    grid = np.arange(lo, hi + 1e-9, tilt_step_deg, dtype=float)
    for tilt in grid:
        tilt_rad = _deg2rad(tilt)
        dirs_p, dirs_m = _cone_directions_hgs(alpha, lon_rad, lat_rad, tilt_rad, n_phi=n_phi)

        # それぞれの cone 上で s1, s2 に最も近い方位を選ぶ
        def nearest_on(dirs, s):
            ang = np.rad2deg(np.arccos(np.clip(dirs @ s, -1.0, 1.0)))
            i = int(np.argmin(ang))
            v = float(ang[i]); d = dirs[i]
            lon = float(np.rad2deg(np.arctan2(d[1], d[0])))
            lat = float(np.rad2deg(np.arcsin(np.clip(d[2], -1.0, 1.0))))
            return v, (lon, lat)

        # 割り当て 2 通りを試す
        # (s1→+y, s2→-y) と (s1→-y, s2→+y)
        v11, fp1 = nearest_on(dirs_p, s1)
        v12, fp2 = nearest_on(dirs_m, s2)
        sumA = v11 + v12

        v21, fp1b = nearest_on(dirs_m, s1)
        v22, fp2b = nearest_on(dirs_p, s2)
        sumB = v21 + v22

        if sumA <= sumB:
            total = sumA; assign = ('+y','-y'); fps = (fp1, fp2)
        else:
            total = sumB; assign = ('-y','+y'); fps = (fp1b, fp2b)

        if total < best['min_total_angle_deg']:
            best = {'tilt_deg': float(tilt),
                    'min_total_angle_deg': float(total),
                    'assign': assign,
                    'footpoints_lonlat_deg': fps}
    return best
