"""
AIA RGB running-difference + GCS wireframe overlay

前提:
- aia_diff_plot_analysis.py 内の AIA 用ユーティリティ
  (BASE_DATA_DIR, parse_datetime_str, normalize_log_stretch, get_dn_per_s)
- plot_GCS.py 内で import されている GCS 関係
  (GCSParams, sample_gcs_wireframe_points)

をそのまま import して利用する。
"""
from __future__ import annotations

from pathlib import Path
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames as sunpy_frames
import sunpy.map

# --- 既存コードから必要なものを import ---
# モジュール名は実際のファイル名に合わせて変更してください
import sys
sys.path.append("/home/kinno-7010/Research_code/SDO/AIA")
from aia_diff_plot_analysis import (
    BASE_DATA_DIR,
    parse_datetime_str,
    normalize_log_stretch,
    get_dn_per_s,
    _format_time_str,
)


# 太陽中心＋Rsun 円を描く関数（既存と整合）
from aia_MGN_diff_plot import add_center_and_rsun

sys.path.append("/home/kinno-7010/Research_code/GCS/gcs_overlay")

# GCS パラメータ & ワイヤーフレーム生成
# --- 追加import（既存の SkyCoord import 行を差し替え） ---
from astropy.coordinates import SkyCoord, CartesianRepresentation

from dataclasses import dataclass, replace

@dataclass
class SpheroidDome3DParams:
    """    Prolate spheroid dome (a,b,b).

    Axis geometry can be specified in two ways:

    (A) Legacy center-based (radial axis):
        center_lon_deg, center_lat_deg, center_r_rsun

    (B) Two-point axis (requested; start and apex lon/lat can differ):
        anchor_lon_deg, anchor_lat_deg : axis-surface intersection on r=1
        apex_lon_deg, apex_lat_deg     : apex direction (HGS lon/lat)
        apex_r_rsun or apex_height_rsun

    If both are provided, (B) takes precedence.
    """

    # --- shape ---
    a_rsun: float
    b_rsun: float

    # --- (B) Two-point axis specification ---
    anchor_lon_deg: float | None = None
    anchor_lat_deg: float | None = None
    apex_lon_deg: float | None = None
    apex_lat_deg: float | None = None
    apex_r_rsun: float | None = None
    apex_height_rsun: float | None = None  # convenience: apex_r_rsun = 1 + apex_height_rsun

    # --- (A) Legacy center-based specification (radial axis) ---
    center_lon_deg: float | None = None
    center_lat_deg: float | None = None
    center_r_rsun: float | None = None

    # --- drawing options ---
    n_meridians: int = 12
    n_parallels: int = 7
    n_line_pts: int = 240

    only_above_surface: bool = True
    only_visible: bool = True

    def _has_two_point_axis(self) -> bool:
        return (
            (self.anchor_lon_deg is not None)
            and (self.anchor_lat_deg is not None)
            and (self.apex_lon_deg is not None)
            and (self.apex_lat_deg is not None)
            and ((self.apex_r_rsun is not None) or (self.apex_height_rsun is not None))
        )

    def _has_center_axis(self) -> bool:
        return (
            (self.center_lon_deg is not None)
            and (self.center_lat_deg is not None)
            and (self.center_r_rsun is not None)
        )

    def __post_init__(self) -> None:
        # Resolve apex_r_rsun <-> apex_height_rsun.
        if self.apex_r_rsun is None and self.apex_height_rsun is not None:
            self.apex_r_rsun = float(1.0 + self.apex_height_rsun)
        elif self.apex_r_rsun is not None and self.apex_height_rsun is None:
            self.apex_height_rsun = float(self.apex_r_rsun - 1.0)

        # Select specification mode.
        if self._has_two_point_axis():
            # Ensure center_lon/lat exist for display/debug (use anchor as default).
            if self.center_lon_deg is None:
                self.center_lon_deg = float(self.anchor_lon_deg)  # type: ignore[arg-type]
            if self.center_lat_deg is None:
                self.center_lat_deg = float(self.anchor_lat_deg)  # type: ignore[arg-type]

            # center_r_rsun is optional in this mode; compute it once for convenience.
            if self.center_r_rsun is None:
                anchor = _cart_rsun_from_lonlat(float(self.anchor_lon_deg), float(self.anchor_lat_deg), 1.0)
                apex = _cart_rsun_from_lonlat(float(self.apex_lon_deg), float(self.apex_lat_deg), float(self.apex_r_rsun))
                u_axis = _unit_vec(apex - anchor)
                center = apex - float(self.a_rsun) * u_axis
                self.center_r_rsun = float(np.linalg.norm(center))

            return


        # Fallback to legacy center-based mode.
        if not self._has_center_axis():
            raise ValueError(
                'SpheroidDome3DParams: provide either (A) center_lon/lat + center_r_rsun, '
                'or (B) anchor_lon/lat + apex_lon/lat + apex_height/apex_r.'
            )

        # In center-based mode, define anchor/apex lon/lat to be identical to center lon/lat.
        if self.anchor_lon_deg is None:
            self.anchor_lon_deg = float(self.center_lon_deg)  # type: ignore[arg-type]
        if self.anchor_lat_deg is None:
            self.anchor_lat_deg = float(self.center_lat_deg)  # type: ignore[arg-type]
        if self.apex_lon_deg is None:
            self.apex_lon_deg = float(self.center_lon_deg)  # type: ignore[arg-type]
        if self.apex_lat_deg is None:
            self.apex_lat_deg = float(self.center_lat_deg)  # type: ignore[arg-type]

        # If apex_r/height were not specified explicitly, derive from center_r + a.
        if self.apex_r_rsun is None:
            self.apex_r_rsun = float(self.center_r_rsun + self.a_rsun)  # type: ignore[operator]
        if self.apex_height_rsun is None:
            self.apex_height_rsun = float(self.apex_r_rsun - 1.0)  # type: ignore[operator]

    def legend_label(self) -> str:
        if self._has_two_point_axis():
            return (
                f"Spheroid: a={self.a_rsun:.2f} R$_\\odot$, "
                f"b={self.b_rsun:.2f} R$_\\odot$, "
                # f"anchor(lon,lat)=({float(self.anchor_lon_deg):.1f},{float(self.anchor_lat_deg):.1f})$^\\circ$, "
                f"apex(lon,lat)=({float(self.apex_lon_deg):.1f},{float(self.apex_lat_deg):.1f})$^\\circ$, "
                f"r_apex={float(self.apex_r_rsun):.2f} R$_\\odot$"
            )

        # Legacy center-based display
        return (
            f"Spheroid: a={self.a_rsun:.2f} R$_\\odot$, "
            f"b={self.b_rsun:.2f} R$_\\odot$, "
            f"center r={float(self.center_r_rsun):.2f} R$_\\odot$, "
            f"(lon,lat)=({float(self.center_lon_deg):.1f},{float(self.center_lat_deg):.1f})$^\\circ$"
        )

def _unit_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n <= 0:
        return v
    return v / n


def _cart_rsun_from_lonlat(lon_deg: float, lat_deg: float, r_rsun: float) -> np.ndarray:
    r_hat, _, _ = _hgs_unit_vectors(lon_deg, lat_deg)
    return float(r_rsun) * r_hat


def _orthonormal_basis_from_axis(axis_u: np.ndarray, ref_vec: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    axis_u に直交する 2 ベクトル e1,e2 を作る（回転対称面の基底）。
    ref_vec を与えると、その方向に近い安定した e1 を作る。
    """
    u = _unit_vec(axis_u)

    if ref_vec is None:
        ref_vec = np.array([0.0, 0.0, 1.0], dtype=float)

    ref = _unit_vec(ref_vec)

    e1 = np.cross(u, ref)
    if np.linalg.norm(e1) < 1e-8:
        # u と ref がほぼ平行 → 代替
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
        e1 = np.cross(u, ref)
        if np.linalg.norm(e1) < 1e-8:
            ref = np.array([1.0, 0.0, 0.0], dtype=float)
            e1 = np.cross(u, ref)

    e1 = _unit_vec(e1)
    e2 = _unit_vec(np.cross(u, e1))
    return e1, e2


def _spheroid_axis_geometry_rsun(params: SpheroidDome3DParams) -> dict[str, np.ndarray]:
    """
    spheroid の幾何（Rsun単位）を返す。

    Returns:
      center (3,), axis_u (3,), e1 (3,), e2 (3,), anchor (3,), apex (3,)
    """
    if params._has_two_point_axis():
        anchor = _cart_rsun_from_lonlat(float(params.anchor_lon_deg), float(params.anchor_lat_deg), 1.0)
        apex = _cart_rsun_from_lonlat(float(params.apex_lon_deg), float(params.apex_lat_deg), float(params.apex_r_rsun))
        axis_u = _unit_vec(apex - anchor)
        center = apex - float(params.a_rsun) * axis_u

        # 基底の向きは anchor 方向で安定化
        e1, e2 = _orthonormal_basis_from_axis(axis_u, ref_vec=anchor)
        return {"center": center, "axis_u": axis_u, "e1": e1, "e2": e2, "anchor": anchor, "apex": apex}

    # legacy (radial axis)
    r_hat, e_lon, e_lat = _hgs_unit_vectors(float(params.center_lon_deg), float(params.center_lat_deg))
    axis_u = r_hat
    center = float(params.center_r_rsun) * r_hat
    anchor = 1.0 * r_hat
    apex = center + float(params.a_rsun) * r_hat
    return {"center": center, "axis_u": axis_u, "e1": e_lon, "e2": e_lat, "anchor": anchor, "apex": apex}


def _sample_footprint_cart_rsun(params: SpheroidDome3DParams, n_beta: int | None = None) -> np.ndarray:
    """
    一般配置（two-point axis含む）でも動くように、数値的に
    spheroid と r=1 球の交線（フットプリント）をサンプルする。

    Returns:
      pts (3,N)  (N は得られた点数)
    """
    geom = _spheroid_axis_geometry_rsun(params)
    center = geom["center"]
    u = geom["axis_u"]
    e1 = geom["e1"]
    e2 = geom["e2"]

    if n_beta is None:
        n_beta = int(params.n_line_pts)

    betas = np.linspace(0.0, 2.0 * np.pi, n_beta, endpoint=False)
    alpha_grid = np.linspace(0.0, np.pi, 1201)

    pts = []

    for beta in betas:
        dir_perp = np.cos(beta) * e1 + np.sin(beta) * e2  # (3,)

        # x(alpha) を一括計算
        ca = np.cos(alpha_grid)
        sa = np.sin(alpha_grid)
        cart = (
            center[:, None]
            + float(params.a_rsun) * ca[None, :] * u[:, None]
            + float(params.b_rsun) * sa[None, :] * dir_perp[:, None]
        )
        rr = np.linalg.norm(cart, axis=0)
        f = rr - 1.0

        # root candidates
        roots = []

        # exact hits
        hit = np.where(np.isclose(f, 0.0, atol=1e-5))[0]
        for idx in hit:
            roots.append(float(alpha_grid[idx]))

        # sign changes
        s = np.sign(f)
        for i in range(len(alpha_grid) - 1):
            if s[i] == 0 or s[i + 1] == 0:
                continue
            if f[i] * f[i + 1] < 0:
                a0 = float(alpha_grid[i] + (0.0 - f[i]) * (alpha_grid[i + 1] - alpha_grid[i]) / (f[i + 1] - f[i]))
                roots.append(a0)

        if not roots:
            continue

        # choose the “outer” intersection similarly to legacy: maximize sin(alpha)
        alpha0 = max(roots, key=lambda a: np.sin(a))

        p = center + float(params.a_rsun) * np.cos(alpha0) * u + float(params.b_rsun) * np.sin(alpha0) * dir_perp
        # Keep only near-photosphere solutions
        if abs(np.linalg.norm(p) - 1.0) < 5e-3:
            pts.append(p)

    if not pts:
        return np.zeros((3, 0), dtype=float)

    return np.stack(pts, axis=1)  # (3,N)


def _hgs_unit_vectors(lon_deg: float, lat_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    HGS のデカルト基底で、(lon,lat) における
    r_hat（放射）, e_lon（経度増加方向）, e_lat（緯度増加方向）を返す。
    """
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)

    cosl = np.cos(lon)
    sinl = np.sin(lon)
    cosb = np.cos(lat)
    sinb = np.sin(lat)

    r_hat = np.array([cosb * cosl, cosb * sinl, sinb], dtype=float)
    e_lon = np.array([-sinl, cosl, 0.0], dtype=float)
    e_lat = np.array([-sinb * cosl, -sinb * sinl, cosb], dtype=float)

    # 念のため正規化
    r_hat /= np.linalg.norm(r_hat)
    e_lon /= np.linalg.norm(e_lon)
    e_lat /= np.linalg.norm(e_lat)

    return r_hat, e_lon, e_lat

def _visible_mask(
    coords_hgs: SkyCoord,
    reference_map: sunpy.map.Map | None = None,
    *,
    only_visible: bool = True,
) -> np.ndarray:
    """Return a boolean mask for points visible from the observer.

    Far-side culling based on the sign of the dot product between:
    (Sun-center -> observer) and (Sun-center -> point).
    """
    if (not only_visible) or (reference_map is None):
        return np.ones(coords_hgs.shape, dtype=bool)

    try:
        obs_hgs = reference_map.observer_coordinate.transform_to(
            sunpy_frames.HeliographicStonyhurst(obstime=reference_map.date)
        )
        obs_vec = obs_hgs.cartesian.xyz.to_value(u.R_sun)  # (3,)

        pt_vec = coords_hgs.cartesian.xyz.to_value(u.R_sun)  # (3,N) or (3,)
        if pt_vec.ndim == 1:
            pt_vec = pt_vec.reshape(3, 1)

        dot = (obs_vec.reshape(3, 1) * pt_vec).sum(axis=0)

        # Robust against sign conventions: keep the side that retains more points.
        mask_pos = dot > 0
        mask_neg = dot < 0
        return mask_pos if mask_pos.sum() >= mask_neg.sum() else mask_neg

    except Exception:
        return np.ones(coords_hgs.shape, dtype=bool)


def _split_skycoord_by_mask(coords: SkyCoord, mask: np.ndarray) -> list[SkyCoord]:
    """
    mask=True の連続区間ごとに SkyCoord を分割して返す。
    （mask の False 区間をまたいで線がつながるのを防ぐ）
    """
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []

    # 連続でないところで切る
    cuts = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate([[0], cuts + 1])
    ends = np.concatenate([cuts + 1, [idx.size]])

    segs: list[SkyCoord] = []
    for s, e in zip(starts, ends):
        seg_idx = idx[s:e]
        if seg_idx.size >= 2:
            segs.append(coords[seg_idx])
    return segs


def spheroid_dome_apex_hpc(params: SpheroidDome3DParams, reference_map: sunpy.map.Map) -> SkyCoord:
    geom = _spheroid_axis_geometry_rsun(params)
    cart_rsun = geom["apex"]
    rep = CartesianRepresentation(
        cart_rsun[0] * u.R_sun,
        cart_rsun[1] * u.R_sun,
        cart_rsun[2] * u.R_sun,
    )
    coord_hgs = SkyCoord(rep, frame=sunpy_frames.HeliographicStonyhurst, obstime=reference_map.date)
    return coord_hgs.transform_to(reference_map.coordinate_frame)


def spheroid_axis_footpoint_hpc(params: SpheroidDome3DParams, reference_map: sunpy.map.Map) -> SkyCoord:
    geom = _spheroid_axis_geometry_rsun(params)
    cart_rsun = geom["anchor"]  # r=1 の軸交点
    rep = CartesianRepresentation(
        cart_rsun[0] * u.R_sun,
        cart_rsun[1] * u.R_sun,
        cart_rsun[2] * u.R_sun,
    )
    coord_hgs = SkyCoord(rep, frame=sunpy_frames.HeliographicStonyhurst, obstime=reference_map.date)
    return coord_hgs.transform_to(reference_map.coordinate_frame)


def spheroid_footprint_angular_radius_deg(params: SpheroidDome3DParams) -> float | None:
    """
    r=1 の交線が存在する場合に「代表的な角半径 ψ」を返す。
    - legacy radial axis: 解析式
    - two-point axis: 数値サンプルの中央値（anchor からの角距離）
    """
    if not params._has_two_point_axis():
        # ---- legacy analytic ----
        center_r = float(params.center_r_rsun)
        a = float(params.a_rsun)
        b = float(params.b_rsun)

        # (r=1) intersection condition: (center_r - a cosα)^2 + (b sinα)^2 = 1
        A = a * a - b * b
        B = -2.0 * center_r * a
        C = center_r * center_r + b * b - 1.0

        disc = B * B - 4.0 * A * C
        if disc < 0:
            return None

        sdisc = np.sqrt(disc)
        x1 = (-B + sdisc) / (2.0 * A)  # cosα
        x2 = (-B - sdisc) / (2.0 * A)

        candidates = []
        for x in (x1, x2):
            if -1.0 <= x <= 1.0:
                alpha = np.arccos(x)
                # As in your original code: choose larger sin(alpha)
                candidates.append((np.sin(alpha), alpha))

        if not candidates:
            return None

        _, alpha_star = max(candidates, key=lambda t: t[0])
        # ψ = angle between r_hat and point on sphere at (r=1)
        # In radial case, footprint is circle of constant colatitude: ψ = arccos(x)
        psi = float(np.rad2deg(alpha_star))
        return psi

    # ---- two-point numeric ----
    pts = _sample_footprint_cart_rsun(params)
    if pts.shape[1] < 8:
        return None

    geom = _spheroid_axis_geometry_rsun(params)
    anchor = _unit_vec(geom["anchor"])
    vv = _unit_vec(pts)  # normalize each column (broadcast-safe)
    # unit normalize columns
    vv = vv / np.linalg.norm(vv, axis=0, keepdims=True)

    cosang = np.clip(anchor @ vv, -1.0, 1.0)
    ang = np.arccos(cosang)
    return float(np.rad2deg(np.median(ang)))


def sample_spheroid_footprint_hpc(params: SpheroidDome3DParams, reference_map: sunpy.map.Map) -> list[SkyCoord]:
    """
    r=1 sphere との交線（フットプリント）をHPCで返す。
    - legacy radial axis: 解析式で直接作る
    - two-point axis: 数値的に交線をサンプルして作る
    """
    # two-point -> numeric
    if params._has_two_point_axis():
        pts = _sample_footprint_cart_rsun(params)
        if pts.shape[1] == 0:
            return []

        rep = CartesianRepresentation(
            pts[0, :] * u.R_sun,
            pts[1, :] * u.R_sun,
            pts[2, :] * u.R_sun,
        )
        coords_hgs = SkyCoord(rep, frame=sunpy_frames.HeliographicStonyhurst, obstime=reference_map.date)
        coords_hpc = coords_hgs.transform_to(reference_map.coordinate_frame)

        if params.only_visible:
            mask = _visible_mask(coords_hgs, reference_map)
            return _split_skycoord_by_mask(coords_hpc, mask)

        return [coords_hpc]

    # legacy analytic (radial)
    center_r = float(params.center_r_rsun)
    a = float(params.a_rsun)
    b = float(params.b_rsun)

    A = a * a - b * b
    B = -2.0 * center_r * a
    C = center_r * center_r + b * b - 1.0

    disc = B * B - 4.0 * A * C
    if disc < 0:
        return []

    sdisc = np.sqrt(disc)
    x1 = (-B + sdisc) / (2.0 * A)
    x2 = (-B - sdisc) / (2.0 * A)

    candidates = []
    for x in (x1, x2):
        if -1.0 <= x <= 1.0:
            alpha = np.arccos(x)
            candidates.append((np.sin(alpha), alpha))

    if not candidates:
        return []

    _, alpha_star = max(candidates, key=lambda t: t[0])

    betas = np.linspace(0.0, 2.0 * np.pi, params.n_line_pts, endpoint=True)
    r_hat, e_lon, e_lat = _hgs_unit_vectors(float(params.center_lon_deg), float(params.center_lat_deg))

    dir_latlon = np.cos(betas)[None, :] * e_lon[:, None] + np.sin(betas)[None, :] * e_lat[:, None]
    cart = 1.0 * r_hat[:, None]  # points lie on r=1
    cart = np.cos(alpha_star) * r_hat[:, None] + np.sin(alpha_star) * dir_latlon

    rep = CartesianRepresentation(
        cart[0, :] * u.R_sun,
        cart[1, :] * u.R_sun,
        cart[2, :] * u.R_sun,
    )
    coords_hgs = SkyCoord(rep, frame=sunpy_frames.HeliographicStonyhurst, obstime=reference_map.date)
    coords_hpc = coords_hgs.transform_to(reference_map.coordinate_frame)

    if params.only_visible:
        mask = _visible_mask(coords_hgs, reference_map)
        return _split_skycoord_by_mask(coords_hpc, mask)

    return [coords_hpc]


def sample_spheroid_dome_wireframe_hpc(params: SpheroidDome3DParams, reference_map: sunpy.map.Map) -> list[SkyCoord]:
    """
    spheroid dome のワイヤーフレーム（経線・緯線）をHPCで返す。
    two-point axis では (anchor, apex) から軸を定義し、任意方向に傾いた spheroid を描ける。
    """
    geom = _spheroid_axis_geometry_rsun(params)
    center = geom["center"][:, None]  # (3,1)
    u_axis = geom["axis_u"][:, None]  # (3,1)
    e1 = geom["e1"][:, None]
    e2 = geom["e2"][:, None]

    lines_hpc: list[SkyCoord] = []

    # ---- meridians (vary alpha, fixed beta) ----
    alphas = np.linspace(0.0, np.pi, params.n_line_pts, endpoint=True)
    ca = np.cos(alphas)[None, :]
    sa = np.sin(alphas)[None, :]

    betas = np.linspace(0.0, 2.0 * np.pi, params.n_meridians, endpoint=False)
    for beta in betas:
        dir_perp = (np.cos(beta) * e1 + np.sin(beta) * e2)  # (3,1)
        cart = center + float(params.a_rsun) * ca * u_axis + float(params.b_rsun) * sa * dir_perp  # (3,N)

        rr = np.linalg.norm(cart, axis=0)
        mask = np.ones(rr.shape, dtype=bool)
        if params.only_above_surface:
            mask &= (rr >= 1.0)

        rep = CartesianRepresentation(
            cart[0, :] * u.R_sun,
            cart[1, :] * u.R_sun,
            cart[2, :] * u.R_sun,
        )
        coords_hgs = SkyCoord(rep, frame=sunpy_frames.HeliographicStonyhurst, obstime=reference_map.date)
        coords_hpc = coords_hgs.transform_to(reference_map.coordinate_frame)

        if params.only_visible:
            vis = _visible_mask(coords_hgs, reference_map)
            mask &= vis

        lines_hpc.extend(_split_skycoord_by_mask(coords_hpc, mask))

    # ---- parallels (fixed alpha, vary beta) ----
    alpha_list = np.linspace(0.0, np.pi, params.n_parallels + 2)[1:-1]
    betas_line = np.linspace(0.0, 2.0 * np.pi, params.n_line_pts, endpoint=True)

    cb = np.cos(betas_line)[None, :]
    sb = np.sin(betas_line)[None, :]

    for alpha0 in alpha_list:
        ca0 = float(np.cos(alpha0))
        sa0 = float(np.sin(alpha0))

        dir_perp = cb * e1 + sb * e2  # (3,N)
        cart = center + float(params.a_rsun) * ca0 * u_axis + float(params.b_rsun) * sa0 * dir_perp  # (3,N)

        rr = np.linalg.norm(cart, axis=0)
        mask = np.ones(rr.shape, dtype=bool)
        if params.only_above_surface:
            mask &= (rr >= 1.0)

        rep = CartesianRepresentation(
            cart[0, :] * u.R_sun,
            cart[1, :] * u.R_sun,
            cart[2, :] * u.R_sun,
        )
        coords_hgs = SkyCoord(rep, frame=sunpy_frames.HeliographicStonyhurst, obstime=reference_map.date)
        coords_hpc = coords_hgs.transform_to(reference_map.coordinate_frame)

        if params.only_visible:
            vis = _visible_mask(coords_hgs, reference_map)
            mask &= vis

        lines_hpc.extend(_split_skycoord_by_mask(coords_hpc, mask))

    return lines_hpc


# =========================================================
# GCS ワイヤーフレームを AIA WCS（Helioprojective）上に投影するヘルパー
# =========================================================


# =========================================================
# AIA RGB ランニング差分 + GCS ワイヤーフレームのメイン関数
# =========================================================
def plot_sdo_aia_rgb_diff_with_spheroid(
    datetime_str: str,
    # AIA 3ch
    channel_r_str: str = "211",
    channel_g_str: str = "193",
    channel_b_str: str = "171",
    delta_minutes: int = 2,
    # spheroid フットプリントパラメータ
    spheroid_params: SpheroidDome3DParams | None = None,
    a_rsun: float = 1.3,
    b_rsun: float = 1.3,
    phi_deg: float = 0.0,
    x0_arcsec: float = 0.0,
    y0_arcsec: float = 0.0,
    # 表示系
    xlim_arcsec: tuple[float, float] | None = None,
    ylim_arcsec: tuple[float, float] | None = None,
    vmax_gray: float = 0.015,
    save_path: str | Path | None = None,
    n_foot_points: int = 360,
):
    """
    aia_diff_plot_analysis.py の処理で AIA 差分マップを描き，
    その上に「spheroid（ショック / EUV wave ドーム）の投影フットプリント」
    を重ねる。

    ここでの spheroid は 3D の軸対称楕円体を想定し，
    その POS (HPC 平面) 上での見かけの楕円輪郭をパラメータ化している。
    """

    # -----------------------------
    # 1. 日時の処理（現在時刻 & delta 分前）
    # -----------------------------
    dt_cur = parse_datetime_str(datetime_str)
    if dt_cur is None:
        return

    dt_cur = dt_cur.replace(second=0, microsecond=0)
    dt_prev = dt_cur - timedelta(minutes=delta_minutes)

    date_cur = dt_cur.strftime("%Y%m%d")
    time_cur_tag = dt_cur.strftime("%H%M")
    date_prev = dt_prev.strftime("%Y%m%d")
    time_prev_tag = dt_prev.strftime("%H%M")

    channels = {"r": channel_r_str, "g": channel_g_str, "b": channel_b_str}

    def _candidate_base_dirs(base_dir: Path) -> list[Path]:
        base_dir = Path(base_dir)
        candidates = [base_dir]
        base_str = str(base_dir)
        if base_str.startswith("F:/wsl/home") or base_str.startswith("F:\\wsl\\home"):
            alt = base_str.replace("F:/wsl/home", "/mnt/d/wsl/home").replace("F:\\wsl\\home", "/mnt/d/wsl/home")
            candidates.append(Path(alt))
        elif base_str.startswith("/mnt/d/wsl/home"):
            alt = base_str.replace("/mnt/d/wsl/home", "F:/wsl/home")
            candidates.append(Path(alt))

        uniq: list[Path] = []
        seen: set[str] = set()
        for p in candidates:
            p_str = str(p)
            if p_str not in seen:
                uniq.append(p)
                seen.add(p_str)
        return uniq

    base_dir_candidates = _candidate_base_dirs(BASE_DATA_DIR)

    def _load_map_with_fallback(file_name: str, ch_str: str, date_str: str, time_str: str, color: str):
        last_error: Exception | None = None
        for base_dir in base_dir_candidates:
            file_path = base_dir / ch_str / file_name
            print(
                f"[{date_str} {time_str}] 読み込み試行: "
                f"{color.upper()} ({ch_str}Å) - {file_path}"
            )
            try:
                return sunpy.map.Map(file_path)
            except Exception as e:
                last_error = e
                continue
        raise last_error if last_error is not None else FileNotFoundError(file_name)

    # ------------------------------------------------
    # 2. 各時刻ごとに、3波長のMapオブジェクトをロード
    # ------------------------------------------------
    def load_maps_for_time(date_str: str, time_str: str):
        maps = {}
        loaded = 0
        for color, ch_str in channels.items():
            wavelength_part_in_fname = ch_str.zfill(4)
            filename = f"AIA{date_str}_{time_str}_{wavelength_part_in_fname}.fits"
            try:
                maps[color] = _load_map_with_fallback(filename, ch_str, date_str, time_str, color)
                print(f"  成功: {ch_str}Å")
                loaded += 1
            except Exception as e:
                print(f"  失敗: {ch_str}Å のファイル読み込みエラー: {e}")
                maps[color] = None
        return maps, loaded

    maps_cur, loaded_cur = load_maps_for_time(date_cur, time_cur_tag)
    maps_prev, loaded_prev = load_maps_for_time(date_prev, time_prev_tag)

    if loaded_cur < 3 or loaded_prev < 3:
        print("エラー: 3波長すべて読み込めませんでした。")
        return

    reference_map = (
        maps_cur["b"] if maps_cur["b"] else maps_cur["g"] if maps_cur["g"] else maps_cur["r"]
    )
    if not reference_map:
        print("エラー: 基準となるMapオブジェクトがありません。")
        return

    wcs_info = reference_map.wcs

    def get_map_time(maps_dict):
        for key in ("b", "g", "r"):
            m = maps_dict.get(key)
            if m is not None:
                return m.date
        return None

    time_cur = get_map_time(maps_cur)
    time_prev = get_map_time(maps_prev)
    time_cur_str = _format_time_str(time_cur)
    time_prev_str = _format_time_str(time_prev)

    # ------------------------------------------------
    # 3. 各時刻・各チャンネルのデータを正規化 → RGB画像に変換
    # ------------------------------------------------
    def make_rgb_image(maps_dict):
        try:
            red_channel_data = normalize_log_stretch(maps_dict["r"].data)
            green_channel_data = normalize_log_stretch(maps_dict["g"].data)
            blue_channel_data = normalize_log_stretch(maps_dict["b"].data)
        except Exception as e_norm:
            print(f"データ正規化中にエラー: {e_norm}")
            return None

        def scale_to_01(data):
            d_min = np.nanmin(data)
            d_max = np.nanmax(data)
            if d_max == d_min:
                return np.zeros_like(data)
            return (data - d_min) / (d_max - d_min)

        red_final = scale_to_01(red_channel_data)
        green_final = scale_to_01(green_channel_data)
        blue_final = scale_to_01(blue_channel_data)

        return np.stack([red_final, green_final, blue_final], axis=-1)

    rgb_cur = make_rgb_image(maps_cur)
    rgb_prev = make_rgb_image(maps_prev)

    if rgb_cur is None or rgb_prev is None:
        print("エラー: RGB画像の生成に失敗しました。")
        return

    # ------------------------------------------------
    # 4. RGB差分画像 → 1チャンネルのスカラー差分に潰す
    # ------------------------------------------------
    diff_211 = get_dn_per_s(maps_cur["r"]) - get_dn_per_s(maps_prev["r"])
    diff_193 = get_dn_per_s(maps_cur["g"]) - get_dn_per_s(maps_prev["g"])
    diff_171 = get_dn_per_s(maps_cur["b"]) - get_dn_per_s(maps_prev["b"])

    diff_scalar = (diff_211 + diff_193 + diff_171) / 3.0

    if isinstance(diff_scalar, np.ma.MaskedArray):
        diff_scalar = diff_scalar.filled(np.nan)

    finite = np.isfinite(diff_scalar)
    if np.any(finite):
        lo, hi = np.nanpercentile(diff_scalar[finite], [10, 90])
        vmax = max(abs(lo), abs(hi))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = np.nanmax(np.abs(diff_scalar[finite]))
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1e-3
        vmin = -vmax
        print("vmin", vmin, "vmax", vmax)
    else:
        diff_scalar = np.zeros_like(diff_scalar, dtype=float)
        vmin, vmax = -1.0, 1.0

    # ------------------------------------------------
    # 5. プロット準備
    # ------------------------------------------------
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection=wcs_info)

    im = ax.imshow(
        diff_scalar,
        origin="lower",
        aspect="equal",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    # 太陽リムとグリッド
    try:
        reference_map.draw_limb(
            axes=ax, color="red", linestyle="dashed", linewidth=3, zorder=1
        )
        reference_map.draw_grid(
            axes=ax,
            grid_spacing=15 * u.deg,
            color="red",
            linestyle="dotted",
            linewidth=2,
            alpha=0.7,
            zorder=1,
        )
        add_center_and_rsun(ax, reference_map)
    except Exception as e_draw:
        print(f"警告: 太陽リム/グリッド/円の描画に失敗しました: {e_draw}")

    # 描画範囲（arcsec指定）
    cdelt1 = reference_map.meta.get("cdelt1")
    cdelt2 = reference_map.meta.get("cdelt2")
    crval1 = reference_map.meta.get("crval1", 0.0)
    crval2 = reference_map.meta.get("crval2", 0.0)
    crpix1 = reference_map.meta.get("crpix1", reference_map.data.shape[1] / 2.0)
    crpix2 = reference_map.meta.get("crpix2", reference_map.data.shape[0] / 2.0)

    if (
        cdelt1 is not None
        and cdelt2 is not None
        and cdelt1 != 0
        and cdelt2 != 0
    ):
        if xlim_arcsec is not None:
            x1_arc, x2_arc = xlim_arcsec
            x1_pix = (x1_arc - crval1) / cdelt1 + crpix1
            x2_pix = (x2_arc - crval1) / cdelt1 + crpix1
            ax.set_xlim(x1_pix, x2_pix)

        if ylim_arcsec is not None:
            y1_arc, y2_arc = ylim_arcsec
            y1_pix = (y1_arc - crval2) / cdelt2 + crpix2
            y2_pix = (y2_arc - crval2) / cdelt2 + crpix2
            ax.set_ylim(y1_pix, y2_pix)
    else:
        print("警告: CDELT が取得できないため、arcsec での描画範囲指定は無効です。")

    # タイトル & 軸ラベル
    title_str_parts = [
        f"SDO/AIA RGB Running Difference: ({channel_r_str}+{channel_g_str}+{channel_b_str}Å)\n",
        f"{time_prev_str} → {time_cur_str} UT",
    ]
    ax.set_title("\n".join(title_str_parts), fontsize=12, pad=5)

    ax.coords[0].set_axislabel("Solar X (arcsec)")
    ax.coords[1].set_axislabel("Solar Y (arcsec)")
    ax.coords[0].set_format_unit(u.arcsec)
    ax.coords[1].set_format_unit(u.arcsec)
    ax.tick_params(axis="both", which="major", labelsize=10, direction="in")

    # ------------------------------------------------
    # 6. 3D spheroid（軸を太陽表面に固定）オーバーレイ
    # ------------------------------------------------
    # 重要:
    # - (anchor_lon_deg, anchor_lat_deg) は「ドーム対称軸が太陽表面と交わる点」で固定する。
    # - 時系列で “広がり” を表現したい場合は、lon/lat を固定したまま
    #   apex_height_rsun, a_rsun, b_rsun を時間とともに増加させる。
    #
    # 可視判定 only_visible が環境差で厳しすぎる場合があるため、
    # ワイヤーフレームが 0 本になったら only_visible=False で自動再試行する。

    if spheroid_params is None:
        # 旧(anchor)の (lon,lat,h,a) を保ったまま center に直すなら：
        # center_r = 1 + h - a
        spheroid_params = SpheroidDome3DParams(
            center_lon_deg=-44.0,
            center_lat_deg=+21.0,
            center_r_rsun=1.0 + 0.20 - 0.20,  # = 1.0
            a_rsun=0.20,
            b_rsun=0.10,
            n_meridians=12,
            n_parallels=7,
            n_line_pts=240,
            only_above_surface=True,
            only_visible=True,
        )

    psi_deg = spheroid_footprint_angular_radius_deg(spheroid_params)
    if psi_deg is None:
        print("ψ = N/A (no photospheric footprint)")
    else:
        print(f"ψ = {psi_deg:.2f}°")

    wire_lines_hpc = sample_spheroid_dome_wireframe_hpc(spheroid_params, reference_map)

    if (len(wire_lines_hpc) == 0) and spheroid_params.only_visible:
        print("[WARN] spheroid wireframe is empty; retry with only_visible=False")
        spheroid_params = SpheroidDome3DParams(
            anchor_lon_deg=spheroid_params.anchor_lon_deg,
            anchor_lat_deg=spheroid_params.anchor_lat_deg,
            apex_height_rsun=spheroid_params.apex_height_rsun,
            a_rsun=spheroid_params.a_rsun,
            b_rsun=spheroid_params.b_rsun,
            n_meridians=spheroid_params.n_meridians,
            n_parallels=spheroid_params.n_parallels,
            n_line_pts=spheroid_params.n_line_pts,
            only_above_surface=spheroid_params.only_above_surface,
            only_visible=False,
        )
        wire_lines_hpc = sample_spheroid_dome_wireframe_hpc(spheroid_params, reference_map)

    for ln in wire_lines_hpc:
        ax.plot_coord(
            ln,
            color="#00FF00",
            linewidth=1.0,
            alpha=0.85,
            zorder=6,
        )

    # 太陽表面で固定する “始まりの点”（対称軸の footpoint）
    anchor_hpc = spheroid_axis_footpoint_hpc(spheroid_params, reference_map)
    ax.plot_coord(
        anchor_hpc,
        marker="*",
        linestyle="None",
        markerfacecolor="yellow",
        markeredgecolor="black",
        markeredgewidth=0.7,
        markersize=20.0,
        zorder=8,
        label=f"axis surface intersection (lon,lat)=({spheroid_params.center_lon_deg:.1f},{spheroid_params.center_lat_deg:.1f})°",

    )

    # 光球 (r=1) との交線＝フットプリント（“広がり” の指標）
    footprint_lines_hpc = sample_spheroid_footprint_hpc(spheroid_params, reference_map)
    for fp in footprint_lines_hpc:
        ax.plot_coord(
            fp,
            color="#00FF00",
            linewidth=2.2,
            alpha=0.95,
            zorder=7,
        )

    # apex マーカー
    apex_hpc = spheroid_dome_apex_hpc(spheroid_params, reference_map)
    apex_label = (
        f"3D spheroid apex (a={spheroid_params.a_rsun:.3f} $R_\\odot$, "
        f"b={spheroid_params.b_rsun:.3f} $R_\\odot$, "
        f"r={spheroid_params.apex_r_rsun:.3f} $R_\\odot$)"
    )
    ax.plot_coord(
        apex_hpc,
        marker="o",
        linestyle="None",
        markerfacecolor="orange",
        markeredgecolor="black",
        markeredgewidth=0.7,
        markersize=10.0,
        zorder=8,
        label=apex_label,
    )

    # 凡例
    ax.plot([], [], color="#00FF00", lw=3, alpha=0.7, label=spheroid_params.legend_label())
    ax.plot([], [], color="#00FF00", lw=3, alpha=0.7)
    ax.legend(loc="upper right", fontsize=10)

    fig.subplots_adjust(bottom=0.20)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved AIA+spheroid figure to {save_path}")

    return fig, ax

# =========================================================
# スクリプトとして実行されたときの例
# =========================================================
if __name__ == "__main__":
    # 例: 2022-06-13 03:33 に対して 2 分ランニング差分 + GCS
    target_time_str = "2022-06-13 02:50"
    target_time_str_no_colon = target_time_str.replace(':', '')

    # 表示範囲 (Zucca イベントで使っている範囲の一例)
    # xlim_arcsec = (-1240.0, 200.0)
    # ylim_arcsec = (-500.0, 1240.0)
    xlim_arcsec = (-1240.0, -100.0)
    ylim_arcsec = (-300.0, 1240.0)
    out_png = (
        Path("/mnt/d/wsl/home/kinno-7010/Research_data/GCS/output")
        / "AIA_GCS"
        / f"aiaRGB_diff_spheroid_{target_time_str_no_colon.replace(' ', '_')}.png"
    )
    
    apex_rsun = 2.5
    a_rsun = (apex_rsun-1)/2
    # a_rsun = 2.5
    b_rsun = 0.5
    center_r_rsun = apex_rsun - a_rsun

    plot_sdo_aia_rgb_diff_with_spheroid(
        datetime_str=target_time_str,
        channel_r_str="193",
        channel_g_str="193",
        channel_b_str="193",
        delta_minutes=10,
        spheroid_params=SpheroidDome3DParams(
            a_rsun=a_rsun,
            b_rsun=b_rsun,
            # ここが「anchor（軸の太陽表面交点）」の指定
            anchor_lon_deg=-30.0,
            anchor_lat_deg=+19.0,
            # ここが「apex（頂点方向）」の指定（anchorと別にできる）
            apex_lon_deg=-50.0,
            apex_lat_deg=+22,
            # apex の半径（または apex_height_rsun を指定）
            apex_r_rsun=apex_rsun,          # 例: 1.8
            # apex_height_rsun=0.8,         # こちらでもOK（apex_r_rsun=1+height）
            n_meridians=12,
            n_parallels=7,
            n_line_pts=240,
            only_above_surface=True,
            only_visible=True,
        ),
        xlim_arcsec=xlim_arcsec,
        ylim_arcsec=ylim_arcsec,
        save_path=out_png,
    )

    plt.show()
