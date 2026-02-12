#!/usr/bin/env python3
"""
cme_radio_height_time_plot.py

Plot CME apex height together with model radio-source heights
for type II upper/lower bands using Saito (1977) density model
and constant radial speeds.

- X axis: Time [UT]
- Y axis: Distance from the solar center [R_sun]

Contents:
  * CME apex height from apex_height_*.csv (as in apex_height_time_plot.py)
  * Additional GCS reference line between 01:10:09 and 03:00:00
  * For the upper band segments:
      - Constant-speed track: 7 × Saito 1977, v = 545 km/s
      - Constant-speed track: 3 × Saito 1977, v = 440 km/s
    Error bars at segment endpoint times span between these two tracks.
  * For the lower band segments: same procedure as for the upper band.

Assumptions:
  - The observed metric type II frequencies (upper/lower bands) are on
    the second harmonic branch, so the inversion uses branch="H".
"""

from __future__ import annotations

import os
import sys
import math
from pathlib import Path
from typing import Sequence, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ----------------------------------------------------------------------
# Project-path setup (same style as predict_type2_const_speed.py)
# ----------------------------------------------------------------------
_CURRENT_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ----------------------------------------------------------------------
# Imports from existing scripts
# ----------------------------------------------------------------------
# CME apex loader (from your existing script)
from apex_height_time_plot import (
    _discover_csv_files,
    _load_apex_time_series,
)

combine_dir = os.path.abspath(os.path.join(_CURRENT_DIR, "..", "RadioData", "combine"))
if combine_dir not in sys.path:
    sys.path.insert(0, combine_dir)

# Saito (1977) density model (cm^-3)
from wind_hf_assa_dynamic_spectrum import Saito1977

# ----------------------------------------------------------------------
# Physical constants & density / frequency helpers
# ----------------------------------------------------------------------
RS_KM = 6.957e5          # solar radius in km (same as in predict_type2_const_speed.py)
kappa = 8.9786628e-3  # MHz * cm^{3/2}
           # plasma frequency coefficient (MHz * cm^{3/2})


def ne_saito_factor(r_rs: float | np.ndarray, factor: float = 6.0) -> np.ndarray:
    """
    Electron density model: n_e(r) = factor × Saito1977(r) [cm^-3].
    """
    r_rs = np.asarray(r_rs, dtype=float)
    return factor * Saito1977(r_rs)


def f_model_from_r(r_rs: float | np.ndarray,
                   branch: str = "F",
                   factor: float = 6.0) -> np.ndarray:
    """
    Plasma emission frequency for a given radius using factor × Saito 1977.

    Parameters
    ----------
    r_rs : float or array
        Radius in solar radii.
    branch : {"F", "H"}
        "F" = fundamental, "H" = harmonic (×2).
    factor : float
        Multiplicative factor for the Saito (1977) density.

    Returns
    -------
    f_mhz : ndarray
        Frequency in MHz (linear scale).
    """
    ne = ne_saito_factor(r_rs, factor=factor)  # cm^-3
    f_fund = kappa * np.sqrt(ne)              # MHz
    if branch.upper() == "H":
        return 2.0 * f_fund
    return f_fund


def invert_r_from_f(f_mhz: float,
                    branch: str = "F",
                    factor: float = 6.0,
                    r_lo: float = 1.2,
                    r_hi: float = 30.0,
                    max_iter: int = 120) -> float:
    """
    Given frequency in MHz, find radius r [R_sun] such that
    f_model_from_r(r) ≈ f_mhz, using a robust bisection.

    This is essentially the same logic as in predict_type2_const_speed.py.
    """
    target = float(f_mhz)
    if branch.upper() == "H":
        target *= 0.5  # convert harmonic to fundamental for inversion

    lo, hi = float(r_lo), float(r_hi)

    # Try to ensure bracketing (expand slowly if needed)
    for _ in range(12):
        f_lo = kappa * math.sqrt(float(ne_saito_factor(lo, factor)))
        f_hi = kappa * math.sqrt(float(ne_saito_factor(hi, factor)))
        if f_lo >= target >= f_hi:
            break
        lo = max(1.05, 0.9 * lo)
        hi *= 1.2

    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = kappa * math.sqrt(float(ne_saito_factor(mid, factor)))
        if f_mid > target:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) < 1e-6:
            break

    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------
# Radio segment utilities
# ----------------------------------------------------------------------

# Upper and lower harmonic-band segments (time in ISO, frequency in MHz)
UPPER_SEGMENTS: List[Tuple[str, str, float, float]] = [
    ("2022-06-13T03:25:40", "2022-06-13T03:26:40", 35.7, 36.5),
    ("2022-06-13T03:26:40", "2022-06-13T03:29:55", 36.5, 31.0),
    ("2022-06-13T03:29:55", "2022-06-13T03:31:15", 31.0, 30.7),
    ("2022-06-13T03:31:15", "2022-06-13T03:31:35", 30.7, 29.7),
    ("2022-06-13T03:31:35", "2022-06-13T03:32:40", 29.7, 29.2),
]

LOWER_SEGMENTS: List[Tuple[str, str, float, float]] = [
    ("2022-06-13T03:25:40", "2022-06-13T03:26:10", 33.0, 31.8),
    ("2022-06-13T03:26:10", "2022-06-13T03:28:15", 31.8, 29.7),
    ("2022-06-13T03:28:15", "2022-06-13T03:28:48", 29.7, 30.2),
    ("2022-06-13T03:28:45", "2022-06-13T03:29:05", 30.2, 29.1),
    ("2022-06-13T03:29:05", "2022-06-13T03:31:00", 29.1, 26.9),
    ("2022-06-13T03:31:00", "2022-06-13T03:31:20", 26.9, 26.0),
    ("2022-06-13T03:31:20", "2022-06-13T03:32:40", 26, 25.3)
]


# GCS プロットと同様に描画する参照ライン（時間範囲）
GCS_REFERENCE_LINE_START = "2022-06-13T01:10:09"
GCS_REFERENCE_LINE_END = "2022-06-13T03:00:00"


def pick_seed_from_segments(segments: Sequence[Tuple[str, str, float, float]]
                            ) -> Tuple[pd.Timestamp, float]:
    """
    Pick a seed point (t_seed, f_seed) from a list of segments.
    Here we simply choose the earliest segment's start time and its start frequency.

    This seed is then used to anchor the constant-speed track at that
    radius inferred from Saito (1977).
    """
    if not segments:
        raise ValueError("Empty segment list")

    # Sort by start time
    sorted_segments = sorted(segments, key=lambda s: s[0])
    start_str, _, f_start, _ = sorted_segments[0]
    t_seed = pd.Timestamp(start_str)
    f_seed = float(f_start)
    return t_seed, f_seed


def segment_endpoint_times(segments: Sequence[Tuple[str, str, float, float]]
                           ) -> List[pd.Timestamp]:
    """
    Collect all unique segment endpoint times (start & end).
    These are used as the x-positions for the radio error bars.
    """
    times: List[pd.Timestamp] = []
    for start_str, end_str, _, _ in segments:
        times.append(pd.Timestamp(start_str))
        times.append(pd.Timestamp(end_str))
    # unique & sorted
    times_sorted = sorted(set(times))
    return times_sorted


def constant_speed_radius_times(
    times: Sequence[pd.Timestamp],
    t_seed: pd.Timestamp,
    f_seed_mhz: float,
    speed_kms: float,
    factor: float,
    branch: str = "H",
) -> np.ndarray:
    """
    Evaluate the constant-speed model radius r(t) [R_sun] at given times.

    - r_seed is determined from Saito (1977) × factor at (t_seed, f_seed_mhz).
    - Then r(t) = r_seed + v * (t - t_seed), with v = speed_kms / RS_KM [R_sun/s].
    """
    t_seed = pd.Timestamp(t_seed)
    r_seed = invert_r_from_f(f_seed_mhz, branch=branch, factor=factor)

    times = pd.to_datetime(times)
    dt_sec = (times - t_seed).total_seconds().astype(float)
    v_rs_s = float(speed_kms) / RS_KM
    r = r_seed + v_rs_s * dt_sec
    return r


def build_track_on_grid(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    t_seed: pd.Timestamp,
    f_seed_mhz: float,
    speed_kms: float,
    factor: float,
    branch: str = "H",
    step_s: float = 10.0,
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Build a (time_grid, r_grid) track over [start_time, end_time] with constant speed.

    Track is anchored at (t_seed, f_seed_mhz), whose radius is determined by
    Saito (1977) × factor.
    """
    time_grid = pd.date_range(start_time, end_time, freq=f"{step_s:.0f}s")
    r_grid = constant_speed_radius_times(
        time_grid, t_seed=t_seed, f_seed_mhz=f_seed_mhz,
        speed_kms=speed_kms, factor=factor, branch=branch
    )
    return time_grid, r_grid


# ----------------------------------------------------------------------
# Main plotting function
# ----------------------------------------------------------------------
def plot_cme_and_radio_height(
    start_time_str: str = "2022-06-13T03:19:00",
    end_time_str: str = "2022-06-13T03:50:00",
    search_root: Path | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot CME apex height and radio-source heights derived directly
    from the upper/lower band segments using Saito (1977) density model.

    - No constant-speed assumption is used.
    - For each segment, the (start_time, start_frequency) and
      (end_time, end_frequency) are connected by a straight line in time–frequency space.
      Frequencies along that line are converted to radius via:
         f_pe -> n_e  (using f_pe[MHz] = kappa * sqrt(n_e[cm^-3]))
         n_e  -> r    (inverse of factor × Saito1977).
    - For each density scaling (7×, 3×), upper/lower tracks are plotted with
      the same color and linestyle, and their separation is represented as
      vertical error bars at key times.
    - For GCS, 7× Saito, and 3× Saito, 1st- and 2nd-order fit velocities
      (converted to km/s) are written into the legend.
    """

    # -----------------------------
    # 1. CME apex height (GCS)
    # -----------------------------
    if search_root is None:
        search_root = Path(__file__).resolve().parent.parent  # search from repo root

    csv_paths = _discover_csv_files(search_root)
    if not csv_paths:
        raise FileNotFoundError(f"No apex_height_*.csv found under {search_root}")

    data = _load_apex_time_series(csv_paths)
    if data.empty:
        raise ValueError("No apex height data were loaded.")

    data["datetime"] = pd.to_datetime(data["datetime"])
    data.sort_values("datetime", inplace=True)

    start_dt = pd.to_datetime(start_time_str)
    end_dt = pd.to_datetime(end_time_str)

    filtered = data[(data["datetime"] >= start_dt) & (data["datetime"] <= end_dt)].copy()
    if filtered.empty:
        raise ValueError("No apex-height points within the requested time range.")

    filtered.sort_values("datetime", inplace=True)
    filtered.reset_index(drop=True, inplace=True)

    # 早期 GCS apex ライン（01:10:09～03:00:00）
    ref_start_dt = pd.Timestamp(GCS_REFERENCE_LINE_START)
    ref_end_dt = pd.Timestamp(GCS_REFERENCE_LINE_END)
    ref_range_label = f"{ref_start_dt.strftime('%H:%M:%S')}–{ref_end_dt.strftime('%H:%M:%S')}"
    gcs_reference = data[
        (data["datetime"] >= ref_start_dt) & (data["datetime"] <= ref_end_dt)
    ].copy()
    gcs_reference.sort_values("datetime", inplace=True)
    gcs_reference.reset_index(drop=True, inplace=True)

    # -----------------------------
    # 2. Radio segments -> r(t)
    # -----------------------------
    def build_tracks_for_segments(
        segments: Sequence[Tuple[str, str, float, float]],
        factor: float,
        branch: str = "H",
        step_s: float = 5.0,
    ) -> Tuple[pd.DatetimeIndex, np.ndarray]:
        """
        For a list of segments (start_t, end_t, f_start, f_end), build
        a time series of radius r(t) by:
          - constructing a linear f(t) between each (t_start, f_start) and (t_end, f_end),
          - converting f(t) -> r(t) using inverse of factor × Saito1977.

        Returns
        -------
        times   : DatetimeIndex
        radii_r : ndarray of r [R_sun]
        """
        all_times: list[pd.DatetimeIndex] = []
        all_radii: list[np.ndarray] = []

        for (t_start_str, t_end_str, f_start, f_end) in segments:
            t_start = pd.to_datetime(t_start_str)
            t_end = pd.to_datetime(t_end_str)

            # 可視範囲との共通部分のみ
            seg_start = max(t_start, start_dt)
            seg_end = min(t_end, end_dt)
            if seg_end <= seg_start:
                continue

            # 時間グリッド
            if step_s > 0:
                time_grid = pd.date_range(seg_start, seg_end, freq=f"{step_s:.0f}s")
                # 1点だけになる場合は両端を含めるように補正
                if len(time_grid) == 1 and seg_start != seg_end:
                    time_grid = pd.DatetimeIndex([seg_start, seg_end])
            else:
                time_grid = pd.DatetimeIndex([seg_start, seg_end])

            if len(time_grid) == 0:
                continue

            # 線形補間で f(t)
            total_sec = (t_end - t_start).total_seconds()
            if total_sec == 0:
                freqs = np.full(len(time_grid), float(f_start))
            else:
                frac = (time_grid - t_start).total_seconds() / total_sec
                freqs = f_start + (f_end - f_start) * frac

            # f -> r に変換
            radii = np.array(
                [invert_r_from_f(f, branch=branch, factor=factor) for f in freqs],
                dtype=float,
            )

            # 直前セグメントと境界が重なる場合は 1 点削る
            if all_times:
                last_times = all_times[-1]
                if len(last_times) > 0 and time_grid[0] == last_times[-1]:
                    time_grid = time_grid[1:]
                    radii = radii[1:]

            if len(time_grid) == 0:
                continue

            all_times.append(time_grid)
            all_radii.append(radii)

        if not all_times:
            return pd.DatetimeIndex([], name="datetime"), np.array([], dtype=float)

        times_concat = pd.DatetimeIndex(
            np.concatenate([idx.values for idx in all_times])
        )
        radii_concat = np.concatenate(all_radii)

        # 念のため時刻でソート
        order = np.argsort(times_concat.values)
        times_sorted = times_concat[order]
        radii_sorted = radii_concat[order]

        return times_sorted, radii_sorted

    def r_at_time_from_segments(
        t: pd.Timestamp,
        segments: Sequence[Tuple[str, str, float, float]],
        factor: float,
        branch: str = "H",
    ) -> float:
        """
        ある時刻 t に対して、その t を含む segment を探し、
        f(t) を線形補間して r(t) を返す。範囲外なら NaN。
        """
        t = pd.to_datetime(t)

        for (t_start_str, t_end_str, f_start, f_end) in segments:
            t_start = pd.to_datetime(t_start_str)
            t_end = pd.to_datetime(t_end_str)
            if t_start <= t <= t_end:
                total_sec = (t_end - t_start).total_seconds()
                if total_sec == 0:
                    f = float(f_start)
                else:
                    frac = (t - t_start).total_seconds() / total_sec
                    f = f_start + (f_end - f_start) * frac
                return float(invert_r_from_f(f, branch=branch, factor=factor))

        return float("nan")

    def build_error_series_for_factor(
        factor: float,
        branch: str,
    ) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
        """
        ある密度倍率 factor (= 7× or 3×) について、
        upper_segments と lower_segments の r(t) を
        共通の時刻で評価し、その最小–最大をエラーバーとして返す。

        Returns
        -------
        times_err      : DatetimeIndex
        r_mid          : 中央値 r_mid(t)
        yerr_minus     : r_mid - r_min
        yerr_plus      : r_max - r_mid
        """
        # upper / lower 両方の start/end 時刻をすべて集める
        all_times: list[pd.Timestamp] = []
        for segs in (UPPER_SEGMENTS, LOWER_SEGMENTS):
            for t_start_str, t_end_str, _, _ in segs:
                all_times.append(pd.to_datetime(t_start_str))
                all_times.append(pd.to_datetime(t_end_str))

        # 重複を除いてソート
        all_times = sorted(set(all_times))

        valid_times: list[pd.Timestamp] = []
        r_mid_list: list[float] = []
        yminus_list: list[float] = []
        yplus_list: list[float] = []

        for t in all_times:
            # 表示範囲外は無視
            if not (start_dt <= t <= end_dt):
                continue

            ru = r_at_time_from_segments(t, UPPER_SEGMENTS, factor=factor, branch=branch)
            rl = r_at_time_from_segments(t, LOWER_SEGMENTS, factor=factor, branch=branch)

            if not (np.isfinite(ru) and np.isfinite(rl)):
                continue

            rmin = min(ru, rl)
            rmax = max(ru, rl)
            rmid = 0.5 * (rmin + rmax)
            valid_times.append(t)
            r_mid_list.append(rmid)
            yminus_list.append(rmid - rmin)
            yplus_list.append(rmax - rmid)

        if not valid_times:
            return (
                pd.DatetimeIndex([], name="datetime"),
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=float),
            )

        times_err = pd.DatetimeIndex(valid_times)
        r_mid = np.array(r_mid_list, dtype=float)
        yerr_minus = np.array(yminus_list, dtype=float)
        yerr_plus = np.array(yplus_list, dtype=float)

        return times_err, r_mid, yerr_minus, yerr_plus

    # モデル因子（密度倍率）
    factor_high = 7.0   # 7 × Saito1977
    factor_low = 3.0    # 3 × Saito1977

    # Upper/Lower トラック（7×）
    t_upper_high, r_upper_high = build_tracks_for_segments(
        UPPER_SEGMENTS, factor=factor_high, branch="H", step_s=5.0
    )
    t_lower_high, r_lower_high = build_tracks_for_segments(
        LOWER_SEGMENTS, factor=factor_high, branch="H", step_s=5.0
    )

    # Upper/Lower トラック（3×）
    t_upper_low, r_upper_low = build_tracks_for_segments(
        UPPER_SEGMENTS, factor=factor_low, branch="H", step_s=5.0
    )
    t_lower_low, r_lower_low = build_tracks_for_segments(
        LOWER_SEGMENTS, factor=factor_low, branch="H", step_s=5.0
    )

    # エラーバー用データ（upper & lower の間）
    times_err_high, r_mid_high, yminus_high, yplus_high = build_error_series_for_factor(
        factor=factor_high, branch="H"
    )
    times_err_low, r_mid_low, yminus_low, yplus_low = build_error_series_for_factor(
        factor=factor_low, branch="H"
    )

    # -----------------------------
    # 3. 速度指標（全区間を1次/2次多項式で近似）→ Legend 用ラベル
    # -----------------------------
    def build_velocity_label_poly(times, r_rsun, base_label: str) -> str:
        """
        times と r[R_sun] から

          - 1st order:
              全区間を r(t) = a1 t + b1 でフィットし、
              v = a1 [km/s], std_v = sqrt(cov[0,0]) を求める。
          - 2nd order:
              全区間を r(t) = a2 t^2 + b2 t + c2 でフィットし、
              v0 = b2 [km/s], a = 2 a2 [km/s^2] とし、
              それぞれの標準偏差を cov から計算する。

        を行い、Legend 用ラベル文字列を返す。
        """
        times = pd.to_datetime(times)
        if len(times) < 2:
            return base_label

        # numpy datetime64 → 秒
        times_np = times.to_numpy()  # datetime64[ns]
        t0 = times_np[0]
        t_sec = (times_np - t0) / np.timedelta64(1, "s")  # [s]

        # r [R_sun] → [km]
        r_km = np.asarray(r_rsun, dtype=float) * RS_KM

        # 有限値のみ
        finite = np.isfinite(t_sec) & np.isfinite(r_km)
        if finite.sum() < 2:
            return base_label

        t_sec = t_sec[finite]
        r_km = r_km[finite]

        legend_parts: list[str] = []

        # ---------- 1st order: r(t) = a1 t + b1 ----------
        try:
            coeff1, cov1 = np.polyfit(t_sec, r_km, 1, cov=True)
            a1, b1 = coeff1           # a1: km/s, b1: km
            v_lin = a1
            std_v_lin = 0.0
            if cov1 is not None and np.shape(cov1) == (2, 2) and np.isfinite(cov1[0, 0]):
                std_v_lin = float(np.sqrt(max(cov1[0, 0], 0.0)))  # km/s

            legend_parts.append(
                r"$v={:.1f}\pm{:.1f}\,\mathrm{{km\,s^{{-1}}}}$"
                .format(v_lin, std_v_lin)
            )
        except Exception:
            pass

        # ---------- 2nd order: r(t) = a2 t^2 + b2 t + c2 ----------
        if t_sec.size >= 3:
            try:
                coeff2, cov2 = np.polyfit(t_sec, r_km, 2, cov=True)
                a2, b2, c2 = coeff2          # a2: km/s^2, b2: km/s, c2: km
                v0 = b2                      # 初期速度 [km/s]
                a_const = 2.0 * a2           # 加速度 [km/s^2]

                std_v0 = 0.0
                std_a = 0.0
                if cov2 is not None and np.shape(cov2) == (3, 3):
                    # b2 の分散 → v0
                    if np.isfinite(cov2[1, 1]):
                        std_v0 = float(np.sqrt(max(cov2[1, 1], 0.0)))
                    # a2 の分散 → 2 a2
                    if np.isfinite(cov2[0, 0]):
                        std_a = float(2.0 * np.sqrt(max(cov2[0, 0], 0.0)))

                # legend_parts.append(
                #     r"2nd order: $v_0={:.1f}\pm{:.1f}\,\mathrm{{km\,s^{{-1}}}}, "
                #     r"a={:.2f}\pm{:.2f}\,\mathrm{{km\,s^{{-2}}}}$"
                #     .format(v0, std_v0, a_const, std_a)
                # )
            except Exception:
                pass


        if legend_parts:
            return base_label + " (" + "; ".join(legend_parts) + ")"
        return base_label

    # GCS apex：Apex height の全区間でポリフィット
    label_gcs = build_velocity_label_poly(
        filtered["datetime"],
        filtered["apex_height"],
        "CME apex height (GCS)",
    )

    if not gcs_reference.empty:
        label_gcs_reference = build_velocity_label_poly(
            gcs_reference["datetime"],
            gcs_reference["apex_height"],
            f"CME apex height ({ref_range_label})",
        )
    else:
        label_gcs_reference = f"CME apex height ({ref_range_label})"

    # 7× Saito 1977：upper/lower 中点 r_mid_high を代表高さとして使用
    if len(times_err_high) > 1:
        label_7x = build_velocity_label_poly(
            times_err_high,
            r_mid_high,
            "7× Saito 1977",
        )
    else:
        label_7x = "7× Saito 1977"

    # 3× Saito 1977
    if len(times_err_low) > 1:
        label_3x = build_velocity_label_poly(
            times_err_low,
            r_mid_low,
            "3× Saito 1977",
        )
    else:
        label_3x = "3× Saito 1977"

    # -----------------------------
    # 4. Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(18, 6))

    # CME apex height
    ax.plot(
        filtered["datetime"],
        filtered["apex_height"],
        marker="o",
        linestyle="-",
        color="black",
        label=label_gcs,
        zorder=3,
    )

    if not gcs_reference.empty:
        ax.plot(
            gcs_reference["datetime"],
            gcs_reference["apex_height"],
            marker="o",
            linestyle="-",
            color="dimgray",
            label=label_gcs_reference,
            zorder=2,
        )

    # モデルごとの色・ラインスタイル
    color7, ls7 = "tab:red", "--"
    color3, ls3 = "tab:blue", "-."

    # 7× Saito 1977: upper & lower を同じ色・linestyle で
    if len(t_upper_high) > 0:
        ax.plot(
            t_upper_high,
            r_upper_high,
            color=color7,
            linestyle=ls7,
            linewidth=1.5,
            label="_nolegend_",
            zorder=2,
        )
    if len(t_lower_high) > 0:
        ax.plot(
            t_lower_high,
            r_lower_high,
            color=color7,
            linestyle=ls7,
            linewidth=1.5,
            label="_nolegend_",
            zorder=2,
        )

    # 3× Saito 1977: upper & lower を同じ色・linestyle で
    if len(t_upper_low) > 0:
        ax.plot(
            t_upper_low,
            r_upper_low,
            color=color3,
            linestyle=ls3,
            linewidth=1.5,
            label="_nolegend_",
            zorder=2,
        )
    if len(t_lower_low) > 0:
        ax.plot(
            t_lower_low,
            r_lower_low,
            color=color3,
            linestyle=ls3,
            linewidth=1.5,
            label="_nolegend_",
            zorder=2,
        )

    # 7× モデルの upper/lower の間をエラーバーで
    if len(times_err_high) > 0:
        ax.errorbar(
            times_err_high,
            r_mid_high,
            yerr=[yminus_high, yplus_high],
            fmt="o",
            mfc="none",
            mec=color7,
            ecolor=color7,
            elinewidth=1.2,
            capsize=3,
            label=label_7x,
            zorder=4,
        )

    # 3× モデルの upper/lower の間をエラーバーで
    if len(times_err_low) > 0:
        ax.errorbar(
            times_err_low,
            r_mid_low,
            yerr=[yminus_low, yplus_low],
            fmt="s",
            mfc="none",
            mec=color3,
            ecolor=color3,
            elinewidth=1.2,
            capsize=3,
            label=label_3x,
            zorder=4,
        )

    # 軸範囲
    x_min = start_dt
    x_max = end_dt
    if not gcs_reference.empty:
        x_min = min(x_min, gcs_reference["datetime"].min())
        x_max = max(x_max, gcs_reference["datetime"].max())
    ax.set_xlim(x_min, x_max)

    y_candidates = [filtered["apex_height"].min(), filtered["apex_height"].max()]
    if not gcs_reference.empty:
        y_candidates.extend(
            [gcs_reference["apex_height"].min(), gcs_reference["apex_height"].max()]
        )
    for arr in (r_upper_high, r_lower_high, r_upper_low, r_lower_low):
        if len(arr) > 0:
            y_candidates.extend([arr.min(), arr.max()])
    for arr in (r_mid_high, r_mid_low):
        if len(arr) > 0:
            y_candidates.extend([arr.min(), arr.max()])

    y_min = min(y_candidates)
    y_max = max(y_candidates)
    ax.set_ylim(y_min - 0.3, y_max + 0.3)
    ax.set_xlim("2022-06-13T01:08:00", "2022-06-13T03:53:00")

    ax.set_xlabel("Time [UT]", fontsize=14)
    ax.set_ylabel("Distance from the solar center [$R_\\odot$]", fontsize=14)
    ax.set_title("CME apex height and type II radio-source heights (Saito 1977)", fontsize=16)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.tick_params(axis="x", labelrotation=0, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    # -----------------------------
    # 5. 縦線 + テキスト（apex_height_time_plot.py と同じ書き方）
    # -----------------------------
    base_date = filtered["datetime"].iloc[0].date()
    vline_info = [
        ("03:25:40",  "03:25:40\nSRBII start"),
        ("03:28:45",  "03:28:45\ntransition time"),
        ("03:31:20",  "03:31:20\ncleaving start"),
        ("03:34:00",  "03:34:00\nHB start"),
        ("03:45:00",  "03:45:00\nSRB II (Harmonic) end"),
    ]
    ax.grid(True, linestyle="--", alpha=0.3)
    for time_str, label in vline_info:
        if time_str == "03:25:40" or time_str == "03:45:00":
            vline_color = "red"; text_color = "red"
        else:
            vline_color = "black"; text_color = "black"
        vline_dt = pd.to_datetime(f"{base_date} {time_str}")
        ax.axvline(x=vline_dt, color=vline_color, linestyle="--", linewidth=0.5, alpha=0.8)
        # ax.text(
        #     vline_dt,
        #     ax.get_ylim()[0],
        #     label,
        #     color=text_color,
        #     fontsize=12,
        #     ha="right",
        #     va="bottom",
        #     alpha=0.8,
        # )

    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

# ----------------------------------------------------------------------
# Script entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    START = "2022-06-13T03:23:00"
    END = "2022-06-13T03:50:00"

    script_dir = Path(__file__).resolve().parent
    # same convention as apex_height_time_plot.py (search from repo root)
    search_root = script_dir.parent

    out_png = script_dir / "output" / "cme_radio_height_time_plot.png"

    plot_cme_and_radio_height(
        start_time_str=START,
        end_time_str=END,
        search_root=search_root,
        save_path=out_png,
        show=True,
    )
