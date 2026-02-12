#!/usr/bin/env python3
"""
predict_type2_const_speed.py

Constant-speed prediction curve for a Type II burst (Saito1977 × Factor).

- Uses Saito1977 from wind_hf_assa_dynamic_spectrum.py and multiplies by `factor` (default 6).
- y-data for the predicted curve are linear MHz; display with ax.set_yscale("log") if desired.
- Provides high-level `plot_type2_prediction(...)` that assembles the spectrum (optional),
  overlays user-specified point(s), and draws the prediction curve.

Now supports drawing the prediction **across the entire plot window** (not a half-line),
clipped to [start_time, end_time] × [min_frequency, max_frequency].
"""

from __future__ import annotations

import math
import datetime as dt
from typing import Tuple, List, Sequence, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Ensure project root is on sys.path for absolute imports like `RadioData.*`
import os
import sys
_CURRENT_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---- Import from your existing script ----
from wind_hf_assa_dynamic_spectrum import (
    Saito1977,
    # optional imports used when with_spectrum=True:
    WIND_CDF_PATH, HF_CDF_PATH, ASSA_FITS_PATHS,
    load_wind_rad2, load_hf, load_callisto,
    create_dataframe, resample_to_grid, normalize_by_median, combine_spectra,
)
# from RadioData.HF_plot.py_folder.spectrum_plot import plot_removed_dynamic_spectrum


from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.dates import SecondLocator

RS_KM = 6.957e5  # solar radius in km
R_SCALE = 2.0
# Plasma frequency conversion constant (MHz · cm^{3/2})
# Matches the value used in `wind_hf_assa_dynamic_spectrum.py`
_e = 1.60217663e-19 # C
_m_e = 9.1093837015e-31
_eps0 = 8.8541878128e-12 # F/m
kappa = (1/(2*np.pi) * np.sqrt((_e)**2 / (_eps0 * _m_e)) * 1e-3)  # exact factor to MHz

def density_fitting_line(r_rs: np.ndarray | float) -> np.ndarray | float:
    # IMPORTANT: the fitted coefficients were obtained using x = r / R_SCALE as the independent variable.
    r = np.asarray(r_rs, dtype=float) /2
    # A_parameter_150 = np.asarray([
    #     19239.472617578933,
    #     12127176.400519352,
    #     627656.5985874196,
    #     587.7192272207918,
    #     114.5211038844094,
    #     1301.3283062767775,
    # ], dtype=float)
    # p_parameter_150 = np.asarray([
    #     0,
    #     -4.647288398527815,
    #     -4.647485853430121,
    #     -4.647820396098652,
    #     -4.650570662126402,
    #     -4.647545731832596,
    # ], dtype=float)
    A_parameter_160 = np.asarray([12131.9958654713, 10877098.90924386, 1187716.8483792453, 99092.27846300564, 163656.4070882954, 141772.35089987706], dtype=float)
    p_parameter_160 = np.asarray([0, -4.377095779844041, -4.37975092178961, -4.377200497543038, -4.377206155735901, -4.377202726835257], dtype=float)

    # 形状をフラット化して A0*r**p0 + ... + A5*r**p5 を計算し、元の形状に戻す
    if r.ndim == 0:
        r_safe = float(max(r, 1e-6))
        return float(np.sum(A_parameter_160 * r_safe**p_parameter_160))

    r_flat = r.ravel()
    r_safe = np.clip(r_flat, 1e-6, None)
    ne_flat = np.sum(A_parameter_160[:, None] * r_safe[None, :] ** p_parameter_160[:, None], axis=0)
    return ne_flat.reshape(r.shape)

# ---------- Density model wrapper (Factor × Saito1977) ----------
def ne_saito_factor(r_rs: np.ndarray | float, factor: float = 1.0) -> np.ndarray | float:
    """Electron density model: n_e(r) = Factor × Saito1977(r) in cm^-3."""
    return factor * Saito1977(np.asarray(r_rs))

def ne_density_fitting_line(r_rs: np.ndarray | float, factor: float = 1.0) -> np.ndarray | float:
    """Electron density model: n_e(r) = Factor × density_fitting_line(r) in cm^-3."""
    # return factor * density_fitting_line(r_rs)
    return factor * Saito1977(r_rs)



def f_model_from_r(r_rs: np.ndarray | float, branch: str = "F", factor: float = 1.0) -> np.ndarray | float:
    """Map radius to plasma emission frequency (F/H)."""
    # ne = ne_saito_factor(r_rs, factor=factor)
    ne = ne_density_fitting_line(r_rs, factor=factor)
    f_f = kappa * np.sqrt(ne)  # MHz
    return 2.0 * f_f if branch.upper() == "H" else f_f


def invert_r_from_f(f_mhz: float, branch: str = "F", factor: float = 1.0,
                    r_lo: float = 1.1, r_hi: float = 30.0, max_iter: int = 120) -> float:
    """Given frequency in MHz, find radius r [R_s] such that f_model_from_r(r) ≈ f_mhz (robust bisection)."""
    target = float(f_mhz) * (0.5 if branch.upper() == "H" else 1.0)
    lo, hi = float(r_lo), float(r_hi)

    # Try to ensure bracketing by gentle expansion
    for _ in range(12):
        # f_lo = kappa * math.sqrt(float(ne_saito_factor(lo, factor)))
        f_lo = kappa * math.sqrt(float(ne_density_fitting_line(lo, factor)))
        f_hi = kappa * math.sqrt(float(ne_density_fitting_line(hi, factor)))
        if f_lo >= target >= f_hi:
            break
        lo = max(1.05, 0.9 * lo)
        hi *= 1.2

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        # f_mid = kappa * math.sqrt(float(ne_saito_factor(mid, factor)))
        f_mid = kappa * math.sqrt(float(ne_density_fitting_line(mid, factor)))
        if f_mid > target:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) < 1e-6:
            break
    return 0.5 * (lo + hi)


def frequency_to_r_saito_factor(
    f_mhz: np.ndarray | float, branch: str = "F", factor: float = 1.0
) -> np.ndarray | float:
    """Map frequency (MHz) to r [R_s] using Factor × Saito1977 (vectorized)."""
    f_arr = np.asarray(f_mhz, dtype=float)
    vec = np.vectorize(lambda val: invert_r_from_f(float(val), branch=branch, factor=factor))
    return vec(f_arr)


def r_to_frequency_saito_factor(
    r_rs: np.ndarray | float, branch: str = "F", factor: float = 1.0
) -> np.ndarray | float:
    """Map r [R_s] back to frequency (MHz) using Factor × Saito1977."""
    return f_model_from_r(r_rs, branch=branch, factor=factor)


# ---------- Constant-speed prediction (half-line generator) ----------
def predict_curve_const_speed(
    t_end: pd.Timestamp | dt.datetime,
    f_end_mhz: float,
    speed_kms: float,
    branch: str = "F",
    factor: float = 6.0,
    duration_s: float = 45 * 60,
    dt_s: float = 5.0,
) -> Tuple[List[pd.Timestamp], np.ndarray]:
    """
    Build a prediction curve f(t) for t >= t_end assuming constant radial speed.
    Returns (times, freqs_MHz) with y in *linear MHz*.
    """
    t_end = pd.Timestamp(t_end)
    r_end = invert_r_from_f(f_end_mhz, branch=branch, factor=factor)
    v_rs_per_s = float(speed_kms) / RS_KM  # convert to R_s per second

    # Time array (including t_end at 0)
    ts = np.arange(0.0, float(duration_s) + 1e-9, float(dt_s))
    r = r_end + v_rs_per_s * ts  # R_s; constant-speed radial motion
    r = np.maximum(r, 1.05)      # guard

    f_lin = f_model_from_r(r, branch=branch, factor=factor)  # MHz, linear
    times = [t_end + pd.to_timedelta(s, unit="s") for s in ts]
    return times, np.asarray(f_lin, dtype=float)


# ---------- Full-span overlay across the plot window ----------
def overlay_prediction_fullspan(
    ax: plt.Axes,
    start_time: pd.Timestamp | dt.datetime,
    end_time: pd.Timestamp | dt.datetime,
    min_frequency: float,
    max_frequency: float,
    t_seed: pd.Timestamp | dt.datetime,
    f_seed_mhz: float,
    speed_kms: float,
    branch: str = "F",
    factor: float = 6.0,
    dt_s: float = 2.0,
    **plot_kwargs,
):
    """
    Draw the constant-speed prediction across the entire time window [start_time, end_time],
    clipped vertically to [min_frequency, max_frequency]. Returns the Line2D handle.

    表示上は常に「赤い点線、alpha=0.8」をデフォルトにします（必要なら plot_kwargs で上書き可）。
    """
    # --- 時刻グリッド（Matplotlib の "days" スケール） ---
    t0 = pd.Timestamp(start_time)
    t1 = pd.Timestamp(end_time)
    tseed = pd.Timestamp(t_seed)

    t0_num = mdates.date2num(t0.to_pydatetime())
    t1_num = mdates.date2num(t1.to_pydatetime())
    tseed_num = mdates.date2num(tseed.to_pydatetime())

    if t1_num <= t0_num:
        return None

    dt_days = float(dt_s) / 86400.0
    n = int(np.floor((t1_num - t0_num) / dt_days)) + 1
    tnums = t0_num + np.arange(n, dtype=float) * dt_days  # shape (n,)

    # --- 半径 r(t) と周波数 f(t) ---
    r_seed = invert_r_from_f(f_seed_mhz, branch=branch, factor=factor)
    v_rs_per_s = float(speed_kms) / RS_KM  # Rs / s
    dt_sec = (tnums - tseed_num) * 86400.0
    r = r_seed + v_rs_per_s * dt_sec
    r = np.maximum(r, 1.02)  # guard
    f = f_model_from_r(r, branch=branch, factor=factor).astype(float)

    # --- 可視範囲の抽出と上下境界の交点補間 ---
    inside = (f >= min_frequency) & (f <= max_frequency)
    if not np.any(inside):
        return None

    idx = np.where(inside)[0]
    i0, i1 = idx[0], idx[-1]

    # 下側境界との交点
    if i0 > 0 and (f[i0-1] < min_frequency) and (f[i0] >= min_frequency):
        t_cross = tnums[i0-1] + (tnums[i0] - tnums[i0-1]) * (min_frequency - f[i0-1]) / (f[i0] - f[i0-1])
        tnums = np.insert(tnums, i0, t_cross)
        f = np.insert(f, i0, min_frequency)
        inside = np.insert(inside, i0, True)
        idx = np.where(inside)[0]
        i0 = idx[0]

    # 上側境界との交点
    if i1 < len(f) - 1 and (f[i1] <= max_frequency) and (f[i1+1] > max_frequency):
        t_cross = tnums[i1] + (tnums[i1+1] - tnums[i1]) * (max_frequency - f[i1]) / (f[i1+1] - f[i1])
        tnums = np.insert(tnums, i1 + 1, t_cross)
        f = np.insert(f, i1 + 1, max_frequency)
        inside = np.insert(inside, i1 + 1, True)
        idx = np.where(inside)[0]
        i1 = idx[-1]

    # 可視区間（連続 1 本）を取得
    tnums_in = tnums[i0:i1+1]
    f_in = f[i0:i1+1]

    # --- プロット（赤い点線、alpha=0.8 をデフォルトに） ---
    style = dict(color="red", linestyle="--", lw=2.0, alpha=0.8, zorder=6)
    style.update(plot_kwargs or {})
    style.setdefault("label", f"{factor}× Saito 1977 ({branch.upper()})")
    (line,) = ax.plot(tnums_in, f_in, **style)
    return line


# ---------- Dynamic spectrum assembly (optional background) ----------
def assemble_dynamic_spectrum(
    start_time: pd.Timestamp, end_time: pd.Timestamp, cadence: str = "0.5s"
) -> pd.DataFrame:
    """Assemble the combined spectrum using your existing helpers (Wind/HF/ASSA)."""
    # Load raw
    wind_times, wind_freqs, wind_values = load_wind_rad2(WIND_CDF_PATH)
    hf_times, hf_freqs, hf_values = load_hf(HF_CDF_PATH, polarization="RH")
    assa_times, assa_freqs, assa_values = load_callisto(ASSA_FITS_PATHS)

    # Create per-instrument DataFrames and normalize
    extended_range = (pd.Timestamp(start_time) - pd.Timedelta(seconds=30),
                      pd.Timestamp(end_time) + pd.Timedelta(seconds=30))
    target_index = pd.date_range(extended_range[0], extended_range[1], freq=cadence)

    wind_df = create_dataframe(wind_times, wind_freqs, wind_values, extended_range)
    wind_df = resample_to_grid(wind_df, target_index, cadence)
    wind_df = normalize_by_median(wind_df)

    hf_df = create_dataframe(hf_times, hf_freqs, hf_values, extended_range)
    hf_df = resample_to_grid(hf_df, target_index, cadence)
    hf_df = normalize_by_median(hf_df)

    assa_df = create_dataframe(assa_times, assa_freqs, assa_values, extended_range)
    assa_df = resample_to_grid(assa_df, target_index, cadence)
    assa_df = normalize_by_median(assa_df)

    combined = combine_spectra([wind_df, hf_df, assa_df])
    # Crop to requested time window only
    combined = combined.loc[(combined.index >= start_time) & (combined.index <= end_time)]
    return combined


# ---------- High-level plotting function ----------
def plot_type2_prediction(
    start_time: pd.Timestamp | str,
    end_time: pd.Timestamp | str,
    min_frequency: float,
    max_frequency: float,
    points: Sequence[Tuple[pd.Timestamp | str, float]] | None,
    speed_kms: float,
    branch: str = "F",
    factor: float = 6.0,
    with_spectrum: bool = True,
    yscale: str = "log",
    dt_s: float = 2.0,
    outfile: str | None = None,
    figsize: Tuple[float, float] = (16, 8),
    cmap: str = "viridis",
    clim: Tuple[float, float] = (0.0, 2.5),
    seed_point: Tuple[pd.Timestamp | str, float] | None = None,
    extra_predictions: Sequence[Dict[str, Any]] | None = None,
):
    """
    Make a figure, optionally draw the combined dynamic spectrum, overlay Type II points,
    and draw the constant-speed prediction **across the entire time window**.

    - If `seed_point` is provided, it overrides the last element of `points` as the primary prediction seed.
    - `extra_predictions` lets you add additional curves (e.g., harmonic branch) with their own seed/time.

    wind_hf_assa_dynamic_spectrum.py の図示・体裁に合わせた設定。
    """
    from matplotlib.ticker import MultipleLocator, FuncFormatter

    # Normalize and parse inputs
    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)
    pts = [(pd.Timestamp(t), float(f)) for (t, f) in (points or [])]
    log_scale = (yscale.lower() == "log")

    fig, ax = plt.subplots(figsize=figsize)

    if with_spectrum:
    # removedコードここから
        # plot_removed_dynamic_spectrum(
        #     fig, ax,
        #     start_time, end_time,
        #     min_frequency, max_frequency,
        #     time_tick_sec=1*60, freq_tick_mhz=5,
        #     med_filter_size=(1,1), vmin=0, vmax=10,
        #     background_method='median_time',
        #     return_data=False,
        # )
    # removedコードここまで
    #------------------------------------------------------------
    # combineコードここから
    # Combine spectra from Wind/RAD2, HF, ASSA using existing helpers
        spectrum = assemble_dynamic_spectrum(start_time, end_time, cadence="0.5s")
        time_axis = spectrum.index.to_pydatetime()
        freq_axis = spectrum.columns.to_numpy()
        values = spectrum.to_numpy().T  # (freq, time)
        mesh = ax.pcolormesh(
            mdates.date2num(time_axis),
            freq_axis,
            values,
            shading="auto",
            cmap=cmap,
            vmin=1, vmax=1.1,
        )

        ax.set_ylabel("Frequency [MHz]", fontsize=16)
        ax.set_yscale("log" if log_scale else "linear")
        ax.set_xlabel("Time [UT]", fontsize=16)
        ax.set_title(
            "Dynamic Spectrum (Wind/RAD2 (1-14 MHz) + HF (15-40 MHz) + ASSA (40-85 MHz)) and Density Fitting Line",
            fontsize=18,
        )
        ax.set_xlim(mdates.date2num(time_axis[0]), mdates.date2num(time_axis[-1]))
        ax.set_ylim(min_frequency, max_frequency)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=10*60))
        ax.yaxis.set_major_locator(MultipleLocator(5.0))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.0f}"))
        ax.tick_params(axis="x", rotation=0, labelrotation=0, labelsize=14)
        ax.tick_params(axis="y", labelsize=12)

        # cbar = fig.colorbar(mesh, ax=ax, pad=0.01, shrink=0.5)
        # cbar.set_label(
        #     "Intensity normalized to per-frequency median",
        #     fontsize=14,
        # )
    # combineコードここまで
    #------------------------------------------------------------
    # Type II の点
    points_handle = None
    if pts:
        xs = [mdates.date2num(t) for t, _ in pts]
        ys = [f for _, f in pts]
        (points_handle,) = ax.plot(
            xs, ys, "o", color="magenta", ms=10, mec="k", mew=1, alpha=0.95, label="Type II points"
        )

    def resolve_seed_point(
        sp: Tuple[pd.Timestamp | str, float] | None,
    ) -> Tuple[pd.Timestamp, float]:
        if sp is not None:
            return pd.Timestamp(sp[0]), float(sp[1])
        if pts:
            t_last, f_last = pts[-1]
            return pd.Timestamp(t_last), float(f_last)
        default_time = start_time + (end_time - start_time) / 2
        default_freq = 0.5 * (min_frequency + max_frequency)
        return pd.Timestamp(default_time), float(default_freq)

    prediction_handles: List[Any] = []

    primary_seed_time, primary_seed_freq = resolve_seed_point(seed_point)
    primary_style: Dict[str, Any] = {}
    if branch.upper() == "H":
        primary_style.setdefault("color", "red")
        primary_style.setdefault("linestyle", "--")
        primary_style.setdefault("alpha", 0.7)

    else:
        primary_style.setdefault("color", "#A0E4FF")
        primary_style.setdefault("linestyle", "--")
        primary_style.setdefault("alpha", 0.7)

    primary_style.setdefault("label", f"{factor}× Saito 1977 ({branch.upper()})")
    # primary_style.setdefault("label", f"{factor}× Density Fitting Line ({branch.upper()})")

    line = overlay_prediction_fullspan(
        ax,
        start_time=start_time,
        end_time=end_time,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        t_seed=primary_seed_time,
        f_seed_mhz=primary_seed_freq,
        speed_kms=speed_kms,
        branch=branch,
        factor=factor,
        dt_s=dt_s,
        **primary_style,
    )
    if line is not None:
        prediction_handles.append(line)

    for spec in extra_predictions or []:
        pred_branch = str(spec.get("branch", branch)).upper()
        pred_factor = float(spec.get("factor", factor))
        pred_speed = float(spec.get("speed_kms", speed_kms))
        pred_dt = float(spec.get("dt_s", dt_s))
        pred_seed = spec.get("seed_point", seed_point)
        seed_time, seed_freq = resolve_seed_point(pred_seed)

        plot_kwargs = dict(spec.get("plot_kwargs", {}))
        if pred_branch == "H":
            plot_kwargs.setdefault("color", "red")
            plot_kwargs.setdefault("linestyle", "--")
            plot_kwargs.setdefault("alpha", 0.8)
        else:
            plot_kwargs.setdefault("color", "blue")
            plot_kwargs.setdefault("linestyle", "-")
            plot_kwargs.setdefault("alpha", 0.9)

        plot_kwargs.setdefault("label", f"{pred_factor}× Saito 1977 ({pred_branch})")
        # plot_kwargs.setdefault("label", f"{pred_factor}× Density Fitting Line ({pred_branch})")

        line_extra = overlay_prediction_fullspan(
            ax,
            start_time=start_time,
            end_time=end_time,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            t_seed=seed_time,
            f_seed_mhz=seed_freq,
            speed_kms=pred_speed,
            branch=pred_branch,
            factor=pred_factor,
            dt_s=pred_dt,
            **plot_kwargs,
        )
        if line_extra is not None:
            prediction_handles.append(line_extra)

    legend_handles: List[Any] = []
    if points_handle is not None:
        points_handle.set_label(f"Type II points\nconst speed = {speed_kms:.0f} km/s")
        legend_handles.append(points_handle)
    legend_handles.extend(prediction_handles)
    if legend_handles:
        ax.legend(
            legend_handles,
            [handle.get_label() for handle in legend_handles],
            loc="lower right",
            fontsize=12,
        )
    ax.set_xlim(mdates.date2num(start_time), mdates.date2num(end_time))
    ax.set_ylim(min_frequency, max_frequency)

    # 第2軸: 周波数→r (5× Saito 1977, Fundamental)
    secax = ax.secondary_yaxis(
        "right",
        functions=(
            lambda f_mhz: frequency_to_r_saito_factor(f_mhz, branch="F", factor=factor),
            lambda r_rs: r_to_frequency_saito_factor(r_rs, branch="F", factor=factor),
        ),
    )
    secax.set_ylabel(f"Radial distance (F)[R$_\\odot$] ({factor}× Saito 1977)", fontsize=14)
    secax.tick_params(axis="y", labelsize=12)
    # 第2軸（右軸）の目盛は 1.5–6.0 Rs のみに固定（主軸の周波数レンジは維持）
    rmin, rmax = 1.5, 6.0

    secax.set_yticks(np.arange(rmin, rmax + 1e-9, 0.5))            # 例：0.5 Rs刻み（必要なら変更）
    secax.yaxis.set_minor_locator(MultipleLocator(0.1))            # 例：副目盛 0.1 Rs
    secax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))

    
    ax.axhline(14.0, color="white", linestyle=":", linewidth=1)
    ax.text(mdates.date2num(end_time), 14.0, "Wind/RAD2", color="white", fontsize=14, ha="right", va="top", fontweight="bold")
    ax.axhline(40.0, color="white", linestyle=":", linewidth=1)
    ax.text(mdates.date2num(end_time), 40.0, "Iitate HF antenna", color="white", fontsize=14, ha="right", va="top", fontweight="bold")
    ax.text(mdates.date2num(end_time), 85.0, "Australia-ASSA", color="white", fontsize=14, ha="right", va="top", fontweight="bold")
    
    CME_time = mdates.date2num(pd.Timestamp("2022-06-13 03:12:00"))
    flare_time = mdates.date2num(pd.Timestamp("2022-06-13 04:07:00"))
    SRB_start_time = mdates.date2num(pd.Timestamp("2022-06-13 03:25:40"))
    SRB_end_time = mdates.date2num(pd.Timestamp("2022-06-13 03:31:20"))
    SRB_end_time_harmonic = mdates.date2num(pd.Timestamp("2022-06-13 03:50:00"))
    ax.axvline(CME_time, color="white", linestyle="--", linewidth=1)
    ax.text(CME_time, 5.7, " Main CME\n erupted\n 03:12 UT", color="white", fontsize=14, ha="left", va="bottom", fontweight="bold")
    # ax.axvline(flare_time, color="white", linestyle="--", linewidth=1)
    # ax.text(flare_time, 85, " M3.4 flare peaked\n 04:07 UT", color="white", fontsize=14, ha="left", va="top", fontweight="bold")
    # ax.axvline(SRB_start_time, color="red", linestyle="--", linewidth=1)
    # ax.text(SRB_start_time, 85, " SRB II start\n 03:25:40 UT", color="red", fontsize=14, ha="left", va="top", fontweight="bold")
    # ax.axvline(SRB_end_time, color="red", linestyle="--", linewidth=1)
    # ax.text(SRB_end_time, 85, " SRB II end\n 03:31:20 UT", color="red", fontsize=14, ha="left", va="top", fontweight="bold")
    # ax.axvline(SRB_end_time_harmonic, color="red", linestyle="--", linewidth=1)
    # ax.text(SRB_end_time_harmonic, 85, " SRB II (Harmonic) end\n 03:50:00 UT", color="red", fontsize=14, ha="left", va="top", fontweight="bold")
    
    # 2.2 Rsの位置にホリゾンタルラインを追加
    # ax.axhline(y=r_to_frequency_saito_factor(2.2, branch="F", factor=factor), color="#A0E4FF", linestyle=":", linewidth=1.5)
    # ax.text(mdates.date2num(end_time), r_to_frequency_saito_factor(2.2, branch="F", factor=factor), "2.2 $R_\\odot$ \n(Fundamental) ",
    #     color="#A0E4FF", fontsize=14, ha="right", va="top", fontweight="bold")
    # ax.axhline(y=r_to_frequency_saito_factor(2.736, branch="H", factor=factor), color="white", linestyle=":", linewidth=1.5)
    # ax.text(mdates.date2num(end_time), r_to_frequency_saito_factor(2.736, branch="H", factor=factor), "2.736 $R_\\odot$ \n(Harmonic) ",
    #     color="white", fontsize=14, ha="right", va="top", fontweight="bold")
    # ax.axhline(y=r_to_frequency_saito_factor(2.2, branch="H", factor=factor), color="pink", linestyle=":", linewidth=1.5)
    # ax.text(mdates.date2num(end_time), r_to_frequency_saito_factor(2.2, branch="H", factor=factor), "2.2 $R_\\odot$ \n(Harmonic) ",
    #     color="pink", fontsize=14, ha="right", va="top", fontweight="bold")
    
    # ax.text(mdates.date2num(pd.Timestamp("2022-06-13T05:00:00")), 7, "Second Harmonic ", fontsize=14, ha="right", va="bottom", fontweight="bold", color="pink")
    # ax.text(mdates.date2num(pd.Timestamp("2022-06-13T05:00:00")), 4, "Fundamental ", fontsize=14, ha="right", va="top", fontweight="bold", color="#A0E4FF")
    
    # ensure secondary axis range stays fixed after any autoscale adjustments
    # secax.set_ylim(1.5, 6.0)
    
    fig.tight_layout()

    if outfile:
        fig.savefig(outfile, dpi=200)
        print(f"Saved figure to {outfile}")
        plt.show()
    else:
        plt.show()

    return fig, ax

# ---------- Main (configure here) ----------
if __name__ == "__main__":
    # ==== USER CONFIG START ====
    # Plot range
    start_time = "2022-06-13T03:00:00"   
    end_time = "2022-06-13T05:00:00"
    min_frequency = 1.0     # MHz
    max_frequency = 87.0    # MHz
    
    # start_time = "2022-06-13T03:25:00"
    # end_time = "2022-06-13T03:34:00"
    # min_frequency = 25.0     # MHz
    # max_frequency = 38.0    # MHz

    # (Option A) Multiple picked points; the last is used unless seed_point is set.
    start_point_time, end_point_time = "2022-06-13T03:25:30", "2022-06-13T03:31:20"
    start_point_frequency, end_point_frequency = 35, 28
    points = [
        (start_point_time, start_point_frequency),
        (end_point_time, end_point_frequency),
    ]

    # (Option B) Seed points for prediction curve
    seed_point_fundamental, seed_point_harmonic = (start_point_time, start_point_frequency/2), (start_point_time, start_point_frequency)  # Fundamental branch

    # Prediction params (use 2.8 × Saito1977; density_fitting_lineは使用しない)
    speed_kms = 394.0       # km/s (constant speed)
    branch = "F"            # Primary branch ('F' for fundamental)
    factor = 2.8            # Factor × Saito1977
    yscale = "log"           # display scale
    with_spectrum = True
    dt_s = 1.0               # sampling for the line across the window
    # outfile = f"/mnt/d/wsl/home/kinno-7010/Research/RadioData/combine/dynamic_spectra_with_fitting_line_from_pB_convert.png"
    outfile = f"/mnt/d/wsl/home/kinno-7010/Research/RadioData/combine/all_dynamic_spectra_with_fitting_line.png"
    # ==== USER CONFIG END ====

    plot_type2_prediction(
        start_time=start_time,
        end_time=end_time,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        points=points,
        speed_kms=speed_kms,
        branch=branch,
        factor=factor,
        with_spectrum=with_spectrum,
        yscale=yscale,
        dt_s=dt_s,
        outfile=outfile,
        seed_point=seed_point_fundamental,
        extra_predictions=[
            {
                "branch": "H",
                "seed_point": seed_point_harmonic,
                "plot_kwargs": {
                    "color": "red",
                    "linestyle": "--",
                    "alpha": 0.8,
                },
            },
        ],
    )
