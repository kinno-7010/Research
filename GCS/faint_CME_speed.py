#!/usr/bin/env python3
"""
faint_CME_speed.py

追加した GCS 参照ライン（01:10:09〜03:00:00）のうち、
指定範囲 01:08:00〜02:03:00 を切り出して単独でプロットするスクリプト。
データの読み込みや凡例の速度フィット設定は
`radio_source_apex_height_plot.py` と同じ方法に従う。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from apex_height_time_plot import _load_apex_time_series
from radio_source_apex_height_plot import (
    LOWER_SEGMENTS,
    UPPER_SEGMENTS,
    invert_r_from_f,
)

RS_KM = 6.957e5  # solar radius in km
SAITO_FACTOR_HIGH = 7.0
SAITO_FACTOR_LOW = 3.0
TYPEII_BRANCH = "H"
TYPEII_STEP_SECONDS = 5.0

DEFAULT_START = "2022-06-13T01:08:00"
DEFAULT_END = "2022-06-13T03:50:00"


def _load_apex_data(search_root: Path, pattern: str) -> pd.DataFrame:
    """GCS apex CSV を読み込み、日時順に整列した DataFrame を返す。"""
    csv_paths = _collect_csv_paths(search_root, pattern)
    if not csv_paths:
        raise FileNotFoundError(
            f"No {pattern} found under {search_root}"
        )

    data = _load_apex_time_series(csv_paths)
    if data.empty:
        raise ValueError("No apex height data were loaded.")

    data["datetime"] = pd.to_datetime(data["datetime"])
    data.sort_values("datetime", inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data



def _candidate_output_dirs(search_root: Path) -> list[Path]:
    """探索対象の output ディレクトリ候補を取得する。"""
    root = search_root.resolve()
    candidates: list[Path] = []
    preferred = [
        root / "output",
        root / "GCS" / "output",
    ]
    for candidate in preferred:
        if candidate.is_dir():
            candidates.append(candidate)

    if not candidates:
        candidates.append(root)
    return candidates


def _collect_csv_paths(search_root: Path, pattern: str) -> list[Path]:
    """指定パターンの CSV を効率的に探索する。"""
    for base in _candidate_output_dirs(search_root):
        matches = sorted(base.rglob(pattern))
        if matches:
            return matches
    return [
        path
        for path in sorted(search_root.resolve().rglob(pattern))
        if "output" in path.parts
    ]


def _generate_linear_fit_curve(
    times: pd.Series,
    radii: pd.Series,
    x_start: pd.Timestamp,
    x_end: pd.Timestamp,
    num_points: int = 200,
) -> Optional[tuple[pd.Series, np.ndarray, float, float]]:
    """指定区間全体に延長した線形フィット曲線を生成する。"""
    if len(times) < 2:
        return None

    times = pd.to_datetime(times)
    base_time = times.iloc[0]

    t_sec = (times - base_time).dt.total_seconds().to_numpy(dtype=float)
    radii_arr = np.asarray(radii, dtype=float)
    finite = np.isfinite(t_sec) & np.isfinite(radii_arr)
    if finite.sum() < 2:
        return None

    coeff, cov = np.polyfit(t_sec[finite], radii_arr[finite], 1, cov=True)

    t_start = (pd.Timestamp(x_start) - base_time).total_seconds()
    t_end = (pd.Timestamp(x_end) - base_time).total_seconds()
    if not np.isfinite(t_start) or not np.isfinite(t_end):
        return None

    t_min, t_max = (t_start, t_end) if t_start <= t_end else (t_end, t_start)
    t_samples = np.linspace(t_min, t_max, num_points)
    fit_values = np.polyval(coeff, t_samples)
    time_samples = base_time + pd.to_timedelta(t_samples, unit="s")

    slope = float(coeff[0])
    velocity_km_s = slope * RS_KM
    std_velocity = 0.0
    if cov is not None and np.shape(cov) == (2, 2) and np.isfinite(cov[0, 0]):
        std_velocity = float(np.sqrt(max(cov[0, 0], 0.0)) * RS_KM)

    return pd.Series(time_samples), fit_values, velocity_km_s, std_velocity


def _build_radio_track(
    segments: list[tuple[str, str, float, float]],
    factor: float,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    branch: str = TYPEII_BRANCH,
    step_seconds: float = TYPEII_STEP_SECONDS,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """タイプIIセグメントを半径トラックに変換する。"""
    times: list[pd.Timestamp] = []
    radii: list[float] = []

    prev_last: Optional[pd.Timestamp] = None
    for t_start_str, t_end_str, f_start, f_end in segments:
        t_start = pd.Timestamp(t_start_str)
        t_end = pd.Timestamp(t_end_str)
        seg_start = max(t_start, start_dt)
        seg_end = min(t_end, end_dt)
        if seg_end <= seg_start:
            continue

        if step_seconds <= 0:
            time_grid = pd.DatetimeIndex([seg_start, seg_end])
        else:
            freq_str = f"{int(max(step_seconds, 1))}S"
            time_grid = pd.date_range(seg_start, seg_end, freq=freq_str)
            if len(time_grid) == 0:
                time_grid = pd.DatetimeIndex([seg_start, seg_end])
            elif len(time_grid) == 1 and seg_start != seg_end:
                time_grid = pd.DatetimeIndex([seg_start, seg_end])

        total_sec = (t_end - t_start).total_seconds()
        if total_sec == 0:
            freqs = np.full(len(time_grid), float(f_start))
        else:
            frac = (time_grid - t_start).total_seconds() / total_sec
            freqs = f_start + (f_end - f_start) * frac

        radii_seg = np.array(
            [float(invert_r_from_f(freq, branch=branch, factor=factor)) for freq in freqs],
            dtype=float,
        )

        if prev_last is not None and len(time_grid) > 0 and time_grid[0] == prev_last:
            time_grid = time_grid[1:]
            radii_seg = radii_seg[1:]

        if len(time_grid) == 0:
            continue

        times.extend(time_grid.to_list())
        radii.extend(radii_seg.tolist())
        prev_last = time_grid[-1]

    if not times:
        return pd.DatetimeIndex([], name="datetime"), np.array([], dtype=float)

    time_index = pd.DatetimeIndex(times)
    radii_arr = np.asarray(radii, dtype=float)
    return time_index, radii_arr


def _radio_radius_at_time(
    t: pd.Timestamp,
    segments: list[tuple[str, str, float, float]],
    factor: float,
    branch: str = TYPEII_BRANCH,
) -> float:
    """指定時刻におけるタイプIIセグメントの半径を返す。"""
    t = pd.Timestamp(t)
    for t_start_str, t_end_str, f_start, f_end in segments:
        t_start = pd.Timestamp(t_start_str)
        t_end = pd.Timestamp(t_end_str)
        if t_start <= t <= t_end:
            total_sec = (t_end - t_start).total_seconds()
            if total_sec == 0:
                freq = float(f_start)
            else:
                frac = (t - t_start).total_seconds() / total_sec
                freq = f_start + (f_end - f_start) * frac
            return float(invert_r_from_f(freq, branch=branch, factor=factor))
    return float("nan")


def _build_radio_error_series(
    upper_segments: list[tuple[str, str, float, float]],
    lower_segments: list[tuple[str, str, float, float]],
    factor: float,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    branch: str = TYPEII_BRANCH,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """上帯域・下帯域からエラーバー用の中央値と上下幅を生成する。"""
    time_candidates: set[pd.Timestamp] = set()
    for seg_list in (upper_segments, lower_segments):
        for t_start_str, t_end_str, _, _ in seg_list:
            time_candidates.add(pd.Timestamp(t_start_str))
            time_candidates.add(pd.Timestamp(t_end_str))

    valid_times: list[pd.Timestamp] = []
    r_mid: list[float] = []
    y_minus: list[float] = []
    y_plus: list[float] = []

    for t in sorted(time_candidates):
        if not (start_dt <= t <= end_dt):
            continue
        r_upper = _radio_radius_at_time(t, upper_segments, factor, branch)
        r_lower = _radio_radius_at_time(t, lower_segments, factor, branch)
        if not (np.isfinite(r_upper) and np.isfinite(r_lower)):
            continue
        r_min = min(r_upper, r_lower)
        r_max = max(r_upper, r_lower)
        r_center = 0.5 * (r_min + r_max)
        valid_times.append(t)
        r_mid.append(r_center)
        y_minus.append(r_center - r_min)
        y_plus.append(r_max - r_center)

    if not valid_times:
        return (
            pd.DatetimeIndex([], name="datetime"),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    return (
        pd.DatetimeIndex(valid_times),
        np.asarray(r_mid, dtype=float),
        np.asarray(y_minus, dtype=float),
        np.asarray(y_plus, dtype=float),
    )


def _build_velocity_label(times: pd.Series, radii_rsun: pd.Series, base_label: str) -> str:
    """1次/2次フィットの速度指標を凡例に付加する。"""
    times = pd.to_datetime(times)
    if len(times) < 2:
        return base_label

    times_np = times.to_numpy(dtype="datetime64[ns]")
    t0 = times_np[0]
    t_sec = (times_np - t0) / np.timedelta64(1, "s")
    r_km = np.asarray(radii_rsun, dtype=float) * RS_KM

    finite = np.isfinite(t_sec) & np.isfinite(r_km)
    if finite.sum() < 2:
        return base_label

    t_sec = t_sec[finite]
    r_km = r_km[finite]
    legend_parts: list[str] = []

    try:
        coeff1, cov1 = np.polyfit(t_sec, r_km, 1, cov=True)
        v_lin = coeff1[0]
        std_v_lin = 0.0
        if cov1 is not None and np.shape(cov1) == (2, 2) and np.isfinite(cov1[0, 0]):
            std_v_lin = float(np.sqrt(max(cov1[0, 0], 0.0)))
        legend_parts.append(
            r"1st order: $v={:.1f}\pm{:.1f}\,\mathrm{{km\,s^{{-1}}}}$".format(
                v_lin, std_v_lin
            )
        )
    except Exception:
        pass

    if t_sec.size >= 3:
        try:
            coeff2, cov2 = np.polyfit(t_sec, r_km, 2, cov=True)
            a2, b2, _ = coeff2
            v0 = b2
            a_const = 2.0 * a2
            std_v0 = 0.0
            std_a = 0.0
            if cov2 is not None and np.shape(cov2) == (3, 3):
                if np.isfinite(cov2[1, 1]):
                    std_v0 = float(np.sqrt(max(cov2[1, 1], 0.0)))
                if np.isfinite(cov2[0, 0]):
                    std_a = float(2.0 * np.sqrt(max(cov2[0, 0], 0.0)))
            legend_parts.append(
                r"2nd order: $v_0={:.1f}\pm{:.1f}\,\mathrm{{km\,s^{{-1}}}}, "
                r"a={:.2f}\pm{:.2f}\,\mathrm{{km\,s^{{-2}}}}$".format(
                    v0, std_v0, a_const, std_a
                )
            )
        except Exception:
            pass

    if legend_parts:
        return base_label + " (" + "; ".join(legend_parts) + ")"
    return base_label


def plot_faint_cme_speed(
    start_time_str: str = DEFAULT_START,
    end_time_str: str = DEFAULT_END,
    search_root: Optional[Path] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    01:08:00〜02:03:00 の CME apex 高度だけを表示する。
    """
    if search_root is None:
        search_root = Path(__file__).resolve().parent

    data = _load_apex_data(search_root, "aia_faint_apex_height_*.csv")
    reference_data = _load_apex_data(search_root, "faint_apex_height_*.csv")
    cme_data = _load_apex_data(search_root, "apex_height_*.csv")
    start_dt = pd.Timestamp(start_time_str)
    end_dt = pd.Timestamp(end_time_str)

    filtered = data[(data["datetime"] >= start_dt) & (data["datetime"] <= end_dt)].copy()
    if filtered.empty:
        raise ValueError("No apex-height points within the requested time range.")

    filtered.sort_values("datetime", inplace=True)
    filtered.reset_index(drop=True, inplace=True)

    reference_filtered = reference_data[
        (reference_data["datetime"] >= start_dt) & (reference_data["datetime"] <= end_dt)
    ].copy()
    reference_filtered.sort_values("datetime", inplace=True)
    reference_filtered.reset_index(drop=True, inplace=True)

    cme_filtered = cme_data[
        (cme_data["datetime"] >= start_dt) & (cme_data["datetime"] <= end_dt)
    ].copy()
    cme_filtered.sort_values("datetime", inplace=True)
    cme_filtered.reset_index(drop=True, inplace=True)

    upper_high_times, upper_high_r = _build_radio_track(
        UPPER_SEGMENTS,
        SAITO_FACTOR_HIGH,
        start_dt,
        end_dt,
    )
    lower_high_times, lower_high_r = _build_radio_track(
        LOWER_SEGMENTS,
        SAITO_FACTOR_HIGH,
        start_dt,
        end_dt,
    )
    upper_low_times, upper_low_r = _build_radio_track(
        UPPER_SEGMENTS,
        SAITO_FACTOR_LOW,
        start_dt,
        end_dt,
    )
    lower_low_times, lower_low_r = _build_radio_track(
        LOWER_SEGMENTS,
        SAITO_FACTOR_LOW,
        start_dt,
        end_dt,
    )
    err_high = _build_radio_error_series(
        UPPER_SEGMENTS,
        LOWER_SEGMENTS,
        SAITO_FACTOR_HIGH,
        start_dt,
        end_dt,
    )
    err_low = _build_radio_error_series(
        UPPER_SEGMENTS,
        LOWER_SEGMENTS,
        SAITO_FACTOR_LOW,
        start_dt,
        end_dt,
    )

    label = "EUV wave height (SDO/AIA)"

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(
        filtered["datetime"],
        filtered["apex_height"],
        marker="o",
        linestyle="-",
        linewidth=1.5,
        color="dimgray",
        label=label,
    )

    fit_curve = _generate_linear_fit_curve(
        filtered["datetime"],
        filtered["apex_height"],
        start_dt,
        end_dt,
    )
    if fit_curve is not None:
        fit_times, fit_values, fit_velocity, fit_std = fit_curve
        ax.plot(
            fit_times,
            fit_values,
            linestyle="--",
            linewidth=1.2,
            color="firebrick",
            label=f"Linear fit (EUV wave): v={fit_velocity:.1f}±{fit_std:.1f} km/s",
        )

    if not reference_filtered.empty:
        ax.plot(
            reference_filtered["datetime"],
            reference_filtered["apex_height"],
            marker="s",
            linestyle="-",
            linewidth=1.2,
            color="tab:blue",
            label="Faint CME apex height (LASCO-C2)",
        )
        ref_fit_curve = _generate_linear_fit_curve(
            reference_filtered["datetime"],
            reference_filtered["apex_height"],
            start_dt,
            end_dt,
        )
        if ref_fit_curve is not None:
            ref_fit_times, ref_fit_values, ref_velocity, ref_std = ref_fit_curve
            ax.plot(
                ref_fit_times,
                ref_fit_values,
                linestyle="--",
                linewidth=1.2,
                color="navy",
                label=f"Linear fit (GCS reference): v={ref_velocity:.1f}±{ref_std:.1f} km/s",
            )

    if len(upper_high_times) > 0:
        ax.plot(
            upper_high_times,
            upper_high_r,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label="_nolegend_",
        )
    if len(lower_high_times) > 0:
        ax.plot(
            lower_high_times,
            lower_high_r,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label="_nolegend_",
        )
    times_err_high, r_mid_high, yminus_high, yplus_high = err_high
    if len(times_err_high) > 0:
        ax.errorbar(
            times_err_high,
            r_mid_high,
            yerr=[yminus_high, yplus_high],
            fmt="o",
            mfc="none",
            mec="tab:red",
            ecolor="tab:red",
            elinewidth=1.1,
            capsize=3,
            label="Type II band (7× Saito 1977)",
        )

    if len(upper_low_times) > 0:
        ax.plot(
            upper_low_times,
            upper_low_r,
            color="tab:blue",
            linestyle="-.",
            linewidth=1.2,
            label="_nolegend_",
        )
    if len(lower_low_times) > 0:
        ax.plot(
            lower_low_times,
            lower_low_r,
            color="tab:blue",
            linestyle="-.",
            linewidth=1.2,
            label="_nolegend_",
        )
    times_err_low, r_mid_low, yminus_low, yplus_low = err_low
    if len(times_err_low) > 0:
        ax.errorbar(
            times_err_low,
            r_mid_low,
            yerr=[yminus_low, yplus_low],
            fmt="s",
            mfc="none",
            mec="tab:blue",
            ecolor="tab:blue",
            elinewidth=1.1,
            capsize=3,
            label="Type II band (3× Saito 1977)",
        )

    if not cme_filtered.empty:
        ax.plot(
            cme_filtered["datetime"],
            cme_filtered["apex_height"],
            marker="^",
            linestyle="-",
            linewidth=1.2,
            color="tab:green",
            label="Main CME apex height",
        )
        cme_fit_curve = _generate_linear_fit_curve(
            cme_filtered["datetime"],
            cme_filtered["apex_height"],
            start_dt,
            end_dt,
        )
        if cme_fit_curve is not None:
            cme_fit_times, cme_fit_values, cme_velocity, cme_std = cme_fit_curve
            ax.plot(
                cme_fit_times,
                cme_fit_values,
                linestyle="--",
                linewidth=1.2,
                color="darkgreen",
                label=f"Linear fit (CME apex height): v={cme_velocity:.1f}±{cme_std:.1f} km/s",
            )

    ax.set_xlim(start_dt, end_dt)
    y_min = filtered["apex_height"].min()
    y_max = filtered["apex_height"].max()
    ax.set_ylim(1, 6)

    ax.set_xlabel("Time [UT]", fontsize=14)
    ax.set_ylabel("Distance from the solar center [$R_\\odot$]", fontsize=14)
    ax.set_title("CME apex height (01:08–02:03 UT)", fontsize=16)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.tick_params(axis="x", labelrotation=0, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=10)
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


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    search_root = script_dir
    out_png = script_dir / "output" / "faint_cme_speed.png"

    plot_faint_cme_speed(
        start_time_str=DEFAULT_START,
        end_time_str=DEFAULT_END,
        search_root=search_root,
        save_path=out_png,
        show=True,
    )

