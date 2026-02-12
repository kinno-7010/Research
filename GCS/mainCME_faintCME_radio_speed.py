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
from radio_source_apex_height_plot import invert_r_from_f

UPPER_SEGMENTS = [("2022-06-13T03:25:40", 35.7), ("2022-06-13T03:25:50", 35.9), ("2022-06-13T03:26:00", 36.1), ("2022-06-13T03:26:10", 36.2),
                ("2022-06-13T03:26:20", 36.3), ("2022-06-13T03:26:30", 36.4), ("2022-06-13T03:26:40", 36.5), 
                
                ("2022-06-13T03:26:50", 36.3), ("2022-06-13T03:27:00", 36.0), ("2022-06-13T03:27:10", 35.7), ("2022-06-13T03:27:20", 35.4),
                ("2022-06-13T03:27:30", 35.1), ("2022-06-13T03:27:40", 34.8), ("2022-06-13T03:27:50", 34.5), ("2022-06-13T03:28:00", 34.2),
                ("2022-06-13T03:28:10", 33.9), ("2022-06-13T03:28:20", 33.6), ("2022-06-13T03:28:30", 33.3), ("2022-06-13T03:28:40", 33.0),
                ("2022-06-13T03:28:50", 32.7), ("2022-06-13T03:29:00", 32.4), ("2022-06-13T03:29:10", 32.1), ("2022-06-13T03:29:20", 31.8),
                ("2022-06-13T03:29:30", 31.5), ("2022-06-13T03:29:40", 31.4), ("2022-06-13T03:29:50", 31.3),
                
                ("2022-06-13T03:30:00", 31.3), ("2022-06-13T03:30:10", 31.1), ("2022-06-13T03:30:20", 31.0), ("2022-06-13T03:30:30", 30.9),
                ("2022-06-13T03:30:40", 30.8), ("2022-06-13T03:30:50", 30.8), ("2022-06-13T03:31:00", 30.7), ("2022-06-13T03:31:10", 30.6),
                
                ("2022-06-13T03:31:20", 30.3), ("2022-06-13T03:31:30", 30.2), ("2022-06-13T03:31:40", 29.7),
                
                ("2022-06-13T03:31:50", 29.6), ("2022-06-13T03:32:00", 29.5), ("2022-06-13T03:32:10", 29.4), ("2022-06-13T03:32:20", 29.3),
                ("2022-06-13T03:32:30", 29.2), ("2022-06-13T03:32:40", 29.2), 
                
                ("2022-06-13T03:32:50", 29.0), ("2022-06-13T03:33:00", 28.9)
                ]

LOWER_SEGMENTS = [("2022-06-13T03:25:40", 33.2), ("2022-06-13T03:25:50", 32.8), ("2022-06-13T03:26:00", 32.4), ("2022-06-13T03:26:10", 31.9),
                
                ("2022-06-13T03:26:20", 31.7), ("2022-06-13T03:26:30", 31.6), ("2022-06-13T03:26:40", 31.5), ("2022-06-13T03:26:50", 31.0),
                ("2022-06-13T03:27:00", 30.9), ("2022-06-13T03:27:10", 30.8), ("2022-06-13T03:27:20", 30.6), ("2022-06-13T03:27:30", 30.5),
                ("2022-06-13T03:27:40", 30.4), ("2022-06-13T03:27:50", 30.2), ("2022-06-13T03:28:00", 30.0), ("2022-06-13T03:28:10", 29.8),
                ("2022-06-13T03:28:20", 29.7),
                
                ("2022-06-13T03:28:30", 29.8), ("2022-06-13T03:28:40", 30.0), ("2022-06-13T03:28:50", 30.2),
                
                
                ("2022-06-13T03:29:00", 29.7), ("2022-06-13T03:29:10", 29.1), 
                
                ("2022-06-13T03:29:20", 29.1), ("2022-06-13T03:29:30", 28.8), ("2022-06-13T03:29:40", 28.5), ("2022-06-13T03:29:50", 28.2),
                ("2022-06-13T03:30:00", 27.9), ("2022-06-13T03:30:10", 27.6), ("2022-06-13T03:30:20", 27.3), ("2022-06-13T03:31:00", 26.9), 
                
                
                ("2022-06-13T03:31:10", 26.4), ("2022-06-13T03:31:20", 26.0),
                
                ("2022-06-13T03:31:30", 25.9), ("2022-06-13T03:31:40", 25.8), ("2022-06-13T03:31:50", 25.7), ("2022-06-13T03:32:00", 25.6),
                ("2022-06-13T03:32:10", 25.5), ("2022-06-13T03:32:20", 25.4), ("2022-06-13T03:32:30", 25.3), ("2022-06-13T03:32:40", 25.2)
                ]

split_upper_lane = [(t, f + 8.3) for t, f in UPPER_SEGMENTS]
split_lower_lane = [("2022-06-13T03:25:40", 40.2), ("2022-06-13T03:25:50", 39.8), ("2022-06-13T03:26:00", 39.4), ("2022-06-13T03:26:10", 38.9),
            
            ("2022-06-13T03:26:20", 38.5),
            
            ("2022-06-13T03:26:30", 38), ("2022-06-13T03:26:40", 38), ("2022-06-13T03:26:50", 38.0),
            ("2022-06-13T03:27:00", 38.5), ("2022-06-13T03:27:10", 38.5), ("2022-06-13T03:27:20", 38.5),
            
            ("2022-06-13T03:27:30", 38.5),
            ("2022-06-13T03:27:40", 38.5), ("2022-06-13T03:27:50", 38.5), ("2022-06-13T03:28:00", 39.0), ("2022-06-13T03:28:10", 39),
            ("2022-06-13T03:28:20", 39.5),
            
            ("2022-06-13T03:28:30", 39.5), ("2022-06-13T03:28:40", 39.5), ("2022-06-13T03:28:50", 39.5),
            ]


RS_KM = 6.957e5  # solar radius in km
SAITO_FACTOR = 2.8
TYPEII_BRANCH = "H"
TYPEII_STEP_SECONDS = 5.0
kappa = 8.9786628e-3  # MHz * cm^{3/2}

DEFAULT_START = "2022-06-13T01:08:00"
DEFAULT_END = "2022-06-13T03:50:00"


def density_fitting_line(r_rs: np.ndarray | float) -> np.ndarray | float:
    # IMPORTANT: the fitted coefficients were obtained using x = r / R_SCALE as the independent variable.
    r = np.asarray(r_rs, dtype=float) /2
    # A_parameter_150 = np.asarray([19239.472617578933, 12127176.400519352, 627656.5985874196, 587.7192272207918, 114.5211038844094, 1301.3283062767775], dtype=float)
    # p_parameter_150 = np.asarray([0, -4.647288398527815, -4.647485853430121, -4.647820396098652, -4.650570662126402, -4.647545731832596], dtype=float)
    
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


def _freq_from_density_fit(r_rs: float | np.ndarray, branch: str = "H") -> np.ndarray:
    """Plasma emission frequency [MHz] from density_fitting_line (fundamental/harmonic)."""
    ne = np.asarray(density_fitting_line(r_rs), dtype=float)  # cm^-3
    f_fund = kappa * np.sqrt(ne)
    if branch.upper() == "H":
        return 2.0 * f_fund
    return f_fund


def invert_r_from_f_density(
    f_mhz: float,
    branch: str = "H",
    r_lo: float = 1.0,
    r_hi: float = 30.0,
    max_iter: int = 120,
) -> float:
    """
    Invert frequency to radius using density_fitting_line (160 deg case).
    """
    target = float(f_mhz)
    if branch.upper() == "H":
        target *= 0.5  # convert harmonic to fundamental

    lo, hi = float(r_lo), float(r_hi)
    # ensure bracket: frequency decreases with r, so f(lo)>target>f(hi)
    for _ in range(12):
        f_lo = float(_freq_from_density_fit(lo, branch="F"))
        f_hi = float(_freq_from_density_fit(hi, branch="F"))
        if f_lo >= target >= f_hi:
            break
        lo = max(1.05, 0.9 * lo)
        hi *= 1.2

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = float(_freq_from_density_fit(mid, branch="F"))
        if f_mid > target:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) < 1e-6:
            break
    return 0.5 * (lo + hi)

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

    times = pd.to_datetime(pd.Series(times))
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
    segments: list[tuple[str, float]],
    factor: float,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    branch: str = TYPEII_BRANCH,
    step_seconds: float = TYPEII_STEP_SECONDS,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """タイプIIの時間‐周波数点列を、そのままの時刻で半径トラックに変換する。"""
    if not segments:
        return pd.DatetimeIndex([], name="datetime"), np.array([], dtype=float)

    times: list[pd.Timestamp] = []
    radii: list[float] = []

    # 与えられた時刻の順番で、そのまま結ぶ（再サンプリングなし）
    for t_str, freq in segments:
        t = pd.Timestamp(t_str)
        if not (start_dt <= t <= end_dt):
            continue
        times.append(t)
        radii.append(float(invert_r_from_f(freq, branch=branch, factor=factor)))

    if not times:
        return pd.DatetimeIndex([], name="datetime"), np.array([], dtype=float)

    time_index = pd.DatetimeIndex(times)
    radii_arr = np.asarray(radii, dtype=float)
    return time_index, radii_arr


def _build_radio_track_density(
    segments: list[tuple[str, float]],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    branch: str = TYPEII_BRANCH,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """density_fitting_line に基づき周波数→半径に変換したトラック。"""
    if not segments:
        return pd.DatetimeIndex([], name="datetime"), np.array([], dtype=float)

    times: list[pd.Timestamp] = []
    radii: list[float] = []
    for t_str, freq in segments:
        t = pd.Timestamp(t_str)
        if not (start_dt <= t <= end_dt):
            continue
        times.append(t)
        r_val = invert_r_from_f_density(float(freq), branch=branch)
        radii.append(r_val)

    if not times:
        return pd.DatetimeIndex([], name="datetime"), np.array([], dtype=float)

    time_index = pd.DatetimeIndex(times)
    radii_arr = np.asarray(radii, dtype=float)
    return time_index, radii_arr


def _radio_radius_at_time(
    t: pd.Timestamp,
    segments: list[tuple[str, float]],
    factor: float,
    branch: str = TYPEII_BRANCH,
) -> float:
    """指定時刻におけるタイプII点列の半径を返す（必要に応じて隣接点で線形補間）。"""
    t = pd.Timestamp(t)
    if len(segments) == 0:
        return float("nan")

    # 時刻順にソートして探索
    sorted_segments = sorted(segments, key=lambda x: pd.Timestamp(x[0]))
    times = [pd.Timestamp(ts) for ts, _ in sorted_segments]
    freqs = [float(f) for _, f in sorted_segments]

    # ぴったり一致
    for ts, freq in zip(times, freqs):
        if ts == t:
            return float(invert_r_from_f(freq, branch=branch, factor=factor))

    # 範囲外
    if t < times[0] or t > times[-1]:
        return float("nan")

    # 隣接点で線形補間（与えられた時刻間のみ）
    for (t0, f0), (t1, f1) in zip(zip(times, freqs), zip(times[1:], freqs[1:])):
        if t0 <= t <= t1:
            total_sec = (t1 - t0).total_seconds()
            if total_sec == 0:
                freq = f0
            else:
                frac = (t - t0).total_seconds() / total_sec
                freq = f0 + (f1 - f0) * frac
            return float(invert_r_from_f(freq, branch=branch, factor=factor))

    return float("nan")


def _build_radio_error_series(
    upper_segments: list[tuple[str, float]],
    lower_segments: list[tuple[str, float]],
    factor: float,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    branch: str = TYPEII_BRANCH,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """上帯域・下帯域からエラーバー用の中央値と上下幅を生成する。"""
    time_candidates: set[pd.Timestamp] = set()
    for seg_list in (upper_segments, lower_segments):
        for t_str, _ in seg_list:
            time_candidates.add(pd.Timestamp(t_str))

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


def _radio_radius_at_time_density(
    t: pd.Timestamp,
    segments: list[tuple[str, float]],
    branch: str = TYPEII_BRANCH,
) -> float:
    """density_fitting_line を用いた指定時刻の半径（線形補間）。"""
    t = pd.Timestamp(t)
    if len(segments) == 0:
        return float("nan")

    sorted_segments = sorted(segments, key=lambda x: pd.Timestamp(x[0]))
    times = [pd.Timestamp(ts) for ts, _ in sorted_segments]
    freqs = [float(f) for _, f in sorted_segments]

    for ts, freq in zip(times, freqs):
        if ts == t:
            return float(invert_r_from_f_density(freq, branch=branch))

    if t < times[0] or t > times[-1]:
        return float("nan")

    for (t0, f0), (t1, f1) in zip(zip(times, freqs), zip(times[1:], freqs[1:])):
        if t0 <= t <= t1:
            total_sec = (t1 - t0).total_seconds()
            if total_sec == 0:
                freq = f0
            else:
                frac = (t - t0).total_seconds() / total_sec
                freq = f0 + (f1 - f0) * frac
            return float(invert_r_from_f_density(freq, branch=branch))

    return float("nan")


def _build_radio_error_series_density(
    upper_segments: list[tuple[str, float]],
    lower_segments: list[tuple[str, float]],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    branch: str = TYPEII_BRANCH,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """density_fitting_line を用いた中央値と上下幅（Type II）。"""
    time_candidates: set[pd.Timestamp] = set()
    for seg_list in (upper_segments, lower_segments):
        for t_str, _ in seg_list:
            time_candidates.add(pd.Timestamp(t_str))

    valid_times: list[pd.Timestamp] = []
    r_mid: list[float] = []
    y_minus: list[float] = []
    y_plus: list[float] = []

    for t in sorted(time_candidates):
        if not (start_dt <= t <= end_dt):
            continue
        r_upper = _radio_radius_at_time_density(t, upper_segments, branch)
        r_lower = _radio_radius_at_time_density(t, lower_segments, branch)
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


def _build_velocity_label(times: pd.Series, radii_rsun: pd.Series, base_label=None) -> str:
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
            r"$v={:.1f}\pm{:.1f}\,\mathrm{{km\,s^{{-1}}}}$".format(
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
                f"2nd order: $v_0=$"+f"{v0:.2f}"+"$\pm$"+f"{std_v0:.2f}"+"$\\mathrm{{km/s}}$", 
                f"$a=$"+f"{a_const:.2f}"+"$\pm$"+f"{std_a:.2f}"+"$\\mathrm{{km/s^2}}$"
            )
        except Exception:
            pass

    if legend_parts:
        return "; ".join(legend_parts)
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

    start_dt = pd.Timestamp(start_time_str)
    end_dt = pd.Timestamp(end_time_str)

    # Faint CME EUV wave height from SDO/AIA
    EUV_wave_data = [("2022-06-13T01:10:09", 1.505), ("2022-06-13T01:20:09", 1.56), ("2022-06-13T01:30:09", 1.602), ("2022-06-13T01:40:09", 1.62), ("2022-06-13T02:00:09", 1.681)]
    
    EUV_wave_data = pd.DataFrame(EUV_wave_data, columns=["datetime", "apex_height"])
    EUV_wave_data["datetime"] = pd.to_datetime(EUV_wave_data["datetime"])
    EUV_wave_data.sort_values("datetime", inplace=True)
    EUV_wave_data.reset_index(drop=True, inplace=True)
    
    filtered = EUV_wave_data[
        (EUV_wave_data["datetime"] >= start_dt)
        & (EUV_wave_data["datetime"] <= end_dt)
    ].copy()
    
    # Faint CME apex height from LASCO-C2
    
    faint_cme_data = [("2022-06-13T01:48:05", 3.2), ("2022-06-13T02:00:05", 3.5), ("2022-06-13T02:12:05", 3.9), ("2022-06-13T02:24:05", 4.1), ("2022-06-13T02:36:07", 4.1), ("2022-06-13T02:48:05", 4.1)]
    faint_cme_data = pd.DataFrame(faint_cme_data, columns=["datetime", "apex_height"])
    faint_cme_data["datetime"] = pd.to_datetime(faint_cme_data["datetime"])
    faint_cme_data.sort_values("datetime", inplace=True)
    faint_cme_data.reset_index(drop=True, inplace=True)
    
    faint_cme_filtered = faint_cme_data[
        (faint_cme_data["datetime"] >= start_dt) & (faint_cme_data["datetime"] <= end_dt)
    ].copy()
    faint_cme_filtered.sort_values("datetime", inplace=True)
    faint_cme_filtered.reset_index(drop=True, inplace=True)


    # Main CME apex height from LASCO-C2
    main_cme_data = _load_apex_data(search_root, "apex_height_*.csv")
    
    main_cme_filtered = main_cme_data[
        (main_cme_data["datetime"] >= start_dt) & (main_cme_data["datetime"] <= end_dt)
    ].copy()
    main_cme_filtered.sort_values("datetime", inplace=True)
    main_cme_filtered.reset_index(drop=True, inplace=True)

    
    # Radio track for Type II band (2.8× Saito 1977)
    radio_track_upper_times, radio_track_upper_r = _build_radio_track(
        UPPER_SEGMENTS,
        SAITO_FACTOR,
        start_dt,
        end_dt,
    )
    radio_track_lower_times, radio_track_lower_r = _build_radio_track(
        LOWER_SEGMENTS,
        SAITO_FACTOR,
        start_dt,
        end_dt,
    )
    radio_track_err = _build_radio_error_series(
        UPPER_SEGMENTS,
        LOWER_SEGMENTS,
        SAITO_FACTOR,
        start_dt,
        end_dt,
    )
    radio_split_err = _build_radio_error_series(
        split_upper_lane,
        split_lower_lane,
        SAITO_FACTOR,
        start_dt,
        end_dt,
    )
    # Type II (density fitting line, 160 deg) track
    radio_track_upper_times_dense, radio_track_upper_r_dense = _build_radio_track_density(
        UPPER_SEGMENTS,
        start_dt,
        end_dt,
        branch=TYPEII_BRANCH,
    )
    radio_track_lower_times_dense, radio_track_lower_r_dense = _build_radio_track_density(
        LOWER_SEGMENTS,
        start_dt,
        end_dt,
        branch=TYPEII_BRANCH,
    )
    radio_track_err_dense = _build_radio_error_series_density(
        UPPER_SEGMENTS,
        LOWER_SEGMENTS,
        start_dt,
        end_dt,
        branch=TYPEII_BRANCH,
    )

    faint_cme_label = "Faint CME apex height (LASCO-C2)"
    main_cme_label = "Main CME apex height (LASCO-C2)"
    EUV_wave_label = "EUV wave height (SDO/AIA)"
    tomo_label = "Tomography"
    faint_cme_fit_label: Optional[str] = None
    main_cme_fit_label: Optional[str] = None
    tomo_fit_label: Optional[str] = None
    radio_label = "Type II (2.8× Saito 1977)"
    radio_fit_label: Optional[str] = None
    radio_split_label = "Type II (band-split)"
    radio_dense_label = "Type II (density-fit 160deg)"
    radio_dense_fit_label: Optional[str] = None

    # Tomography points (3点、誤差±0.05 Rsun)
    tomo_points = [
        ("2022-06-13T03:25:29", 2.95),
        ("2022-06-13T03:28:46", 3.08),
        ("2022-06-13T03:31:17", 3.30),
    ]
    tomo_df = pd.DataFrame(tomo_points, columns=["datetime", "apex_height"])
    tomo_df["datetime"] = pd.to_datetime(tomo_df["datetime"])
    tomo_df.sort_values("datetime", inplace=True)
    tomo_df.reset_index(drop=True, inplace=True)
    tomo_filtered = tomo_df[
        (tomo_df["datetime"] >= start_dt) & (tomo_df["datetime"] <= end_dt)
    ]

    
    # plot
    fig, ax = plt.subplots(figsize=(10,5))
    
    # EUV wave height（指定時間帯に点がある場合のみプロット・フィット）
    if not filtered.empty:
        ax.plot(
            filtered["datetime"],
            filtered["apex_height"],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            color="dimgray",
            label=EUV_wave_label,
        )

        EUV_wave_fit_curve = _generate_linear_fit_curve(
            filtered["datetime"],
            filtered["apex_height"],
            start_dt,
            end_dt,
        )
        if EUV_wave_fit_curve is not None:
            (
                EUV_wave_fit_times,
                EUV_wave_fit_values,
                EUV_wave_fit_velocity,
                EUV_wave_fit_std,
            ) = EUV_wave_fit_curve
            ax.plot(
                EUV_wave_fit_times,
                EUV_wave_fit_values,
                linestyle="--",
                linewidth=1.2,
                color="dimgray",
                label=_build_velocity_label(
                    filtered["datetime"], filtered["apex_height"]
                ),
            )

    # Tomography (3点を線で結び、±0.05 Rsun のエラーバー付き)
    tomo_fit = _generate_linear_fit_curve(
        tomo_filtered["datetime"],
        tomo_filtered["apex_height"],
        start_dt,
        end_dt,
        )
    if tomo_fit is not None:
        (
            tomo_fit_times,
            tomo_fit_values,
            tomo_fit_velocity,
            tomo_fit_std,
        ) = tomo_fit
        tomo_fit_label = (
            f"Tomography: v={tomo_fit_velocity:.1f}±{tomo_fit_std:.1f} km/s"
        )
        if not tomo_filtered.empty:
            ax.errorbar(
                tomo_filtered["datetime"],
                tomo_filtered["apex_height"],
                yerr=0.05,
                fmt="d",
                mfc="white",
                mec="tab:purple",
                ecolor="tab:purple",
                elinewidth=1.0,
                capsize=3,
                linestyle="-",
                color="tab:purple",
                label=tomo_fit_label,
            )
        # 線形フィットで速度推定
        # tomo_fit = _generate_linear_fit_curve(
        #     tomo_filtered["datetime"],
        #     tomo_filtered["apex_height"],
        #     start_dt,
        #     end_dt,
        # )
        # if tomo_fit is not None:
        #     (
        #         tomo_fit_times,
        #         tomo_fit_values,
        #         tomo_fit_velocity,
        #         tomo_fit_std,
        #     ) = tomo_fit
        #     tomo_fit_label = (
        #         f"v={tomo_fit_velocity:.1f}±{tomo_fit_std:.1f} km/s"
        #     )
            # ax.plot(
            #     tomo_fit_times,
            #     tomo_fit_values,
            #     linestyle="--",
            #     linewidth=1.1,
            #     color="tab:purple",
            #     label=tomo_fit_label,
            # )

    # Faint CME apex height
    if not faint_cme_filtered.empty:
        ax.plot(
            faint_cme_filtered["datetime"],
            faint_cme_filtered["apex_height"],
            marker="s",
            linestyle="-",
            linewidth=1.2,
            color="black",
            label=faint_cme_label,
        )
        faint_cme_fit_curve = _generate_linear_fit_curve(
            faint_cme_filtered["datetime"],
            faint_cme_filtered["apex_height"],
            start_dt,
            end_dt,
        )
        if faint_cme_fit_curve is not None:
            faint_cme_fit_times, faint_cme_fit_values, faint_cme_fit_velocity, faint_cme_fit_std = faint_cme_fit_curve
            faint_cme_fit_label = f"v={faint_cme_fit_velocity:.1f}±{faint_cme_fit_std:.1f} km/s"
            ax.plot(
                faint_cme_fit_times,
                faint_cme_fit_values,
                linestyle="--",
                linewidth=1.2,
                color="black",
                label=faint_cme_fit_label,
            )

    # Radio track for Type II band (2.8× Saito 1977)
    times_err_low, r_mid_low, yminus_low, yplus_low = radio_track_err
    typeII_fit = _generate_linear_fit_curve(
        times_err_low,
        r_mid_low,
        start_dt,
        end_dt,
    )
    if typeII_fit is not None:
        (
            typeII_fit_times,
            typeII_fit_values,
            typeII_fit_velocity,
            typeII_fit_std,
        ) = typeII_fit
        radio_fit_label = f"v={typeII_fit_velocity:.1f}±{typeII_fit_std:.1f} km/s"
        radio_label = (
            f"Type II (2.8×Saito1977): v={typeII_fit_velocity:.1f}±{typeII_fit_std:.1f} km/s"
        )
        if len(radio_track_upper_times) > 0:
            ax.plot(
                radio_track_upper_times,
                radio_track_upper_r,
                color="lightgreen",
                linestyle="--",
                linewidth=1.2,
                label="_nolegend_",
            )
        if len(radio_track_lower_times) > 0:
            ax.plot(
                radio_track_lower_times,
                radio_track_lower_r,
                color="lightgreen",
                linestyle="--",
                linewidth=1.2,
                label="_nolegend_",
            )
    times_err_high, r_mid_high, yminus_high, yplus_high = radio_track_err
    if len(times_err_high) > 0:
        ax.errorbar(
            times_err_high,
            r_mid_high,
            yerr=[yminus_high, yplus_high],
            fmt="o",
            mfc="none",
            mec="lightgreen",
            ecolor="lightgreen",
            elinewidth=1.1,
            capsize=3,
            # label="Type II band (2.8× Saito 1977)",
        )

    if len(radio_track_upper_times) > 0:
        ax.plot(
            times_err_high,
            r_mid_high,
            color="lightgreen",
            linestyle="-.",
            linewidth=1.2,
            label="_nolegend_",
        )
    if len(radio_track_lower_times) > 0:
        ax.plot(
            radio_track_lower_times,
            radio_track_lower_r,
            color="lightgreen",
            linestyle="-.",
            linewidth=1.2,
            label="_nolegend_",
        )
    times_err_low, r_mid_low, yminus_low, yplus_low = radio_track_err
    if len(times_err_low) > 0:
        ax.errorbar(
            times_err_low,
            r_mid_low,
            yerr=[yminus_low, yplus_low],
            fmt="s",
            mfc="none",
            mec="lightgreen",
            ecolor="lightgreen",
            elinewidth=1.1,
            capsize=3,
            label=radio_label,
        )
    times_err_split, r_mid_split, yminus_split, yplus_split = radio_split_err
    if len(times_err_split) > 0:
        ax.errorbar(
            times_err_split,
            r_mid_split,
            yerr=[yminus_split, yplus_split],
            fmt="^",
            mfc="none",
            mec="tab:green",
            ecolor="tab:green",
            elinewidth=1.1,
            capsize=3,
            label=radio_split_label,
        )
        # # Type II median track linear fit
        # typeII_fit = _generate_linear_fit_curve(
        #     times_err_low,
        #     r_mid_low,
        #     start_dt,
        #     end_dt,
        # )
        # if typeII_fit is not None:
        #     (
        #         typeII_fit_times,
        #         typeII_fit_values,
        #         typeII_fit_velocity,
        #         typeII_fit_std,
        #     ) = typeII_fit
        #     radio_fit_label = f"v={typeII_fit_velocity:.1f}±{typeII_fit_std:.1f} km/s"
            # ax.plot(
            #     typeII_fit_times,
            #     typeII_fit_values,
            #     linestyle="--",
            #     linewidth=1.1,
            #     color="lightgreen",
            #     label=radio_fit_label,
            # )

    # # Type II (density-fitting line, 160 deg) using same time stamps
    # times_err_low_dense, r_mid_low_dense, _, _ = radio_track_err_dense
    # if len(times_err_low_dense) > 0:
    #     yminus_dense = radio_track_err_dense[2]
    #     yplus_dense = radio_track_err_dense[3]
    #     ax.errorbar(
    #         times_err_low_dense,
    #         r_mid_low_dense,
    #         yerr=[yminus_dense, yplus_dense],
    #         fmt="x",
    #         mfc="none",
    #         mec="green",
    #         ecolor="green",
    #         elinewidth=1.0,
    #         capsize=3,
    #         linestyle="",
    #         label=radio_dense_label,
    #     )
    #     ax.plot(
    #         times_err_low_dense,
    #         r_mid_low_dense,
    #         marker="x",
    #         linestyle="-",
    #         linewidth=1.1,
    #         color="green",
    #         label="_nolegend_",
    #     )
    #     radio_dense_fit = _generate_linear_fit_curve(
    #         times_err_low_dense,
    #         r_mid_low_dense,
    #         start_dt,
    #         end_dt,
    #     )
    #     if radio_dense_fit is not None:
    #         (
    #             radio_dense_fit_times,
    #             radio_dense_fit_values,
    #             radio_dense_fit_velocity,
    #             radio_dense_fit_std,
    #         ) = radio_dense_fit
    #         radio_dense_fit_label = (
    #             f"v={radio_dense_fit_velocity:.1f}±{radio_dense_fit_std:.1f} km/s"
    #         )
            # ax.plot(
            #     radio_dense_fit_times,
            #     radio_dense_fit_values,
            #     linestyle="--",
            #     linewidth=1.0,
            #     color="green",
            #     label=radio_dense_fit_label,
            # )

    # Main CME apex height
    if not main_cme_filtered.empty:
        ax.plot(
            main_cme_filtered["datetime"],
            main_cme_filtered["apex_height"],
            marker="^",
            linestyle="-",
            linewidth=1.2,
            color="red",
            label=main_cme_label,
        )
        main_cme_fit_curve = _generate_linear_fit_curve(
            main_cme_filtered["datetime"],
            main_cme_filtered["apex_height"],
            start_dt,
            end_dt,
        )
        if main_cme_fit_curve is not None:
            main_cme_fit_times, main_cme_fit_values, main_cme_fit_velocity, main_cme_fit_std = main_cme_fit_curve
            main_cme_fit_label = f"v={main_cme_fit_velocity:.1f}±{main_cme_fit_std:.1f} km/s"
            ax.plot(
                main_cme_fit_times,
                main_cme_fit_values,
                linestyle="--",
                linewidth=1.2,
                color="red",
                label=f"v={main_cme_fit_velocity:.1f}±{main_cme_fit_std:.1f} km/s",
            )
    
    y_min = 1
    y_max = 5
    
    # 03:25:40 UT, UPPER_SEGMENTS (開始時の最低高度)
    typeii_ref_r = _radio_radius_at_time(
        pd.to_datetime("2022-06-13T03:25:40"),
        UPPER_SEGMENTS,
        SAITO_FACTOR,
        TYPEII_BRANCH,
    )
    if np.isfinite(typeii_ref_r):
        ax.hlines(
            y=typeii_ref_r,
            xmin=start_dt,
            xmax=end_dt,
            color="black",
            linestyle=":",
            linewidth=1,
        )
        ax.text(pd.to_datetime("2022-06-13T03:40:00"), typeii_ref_r, f"{typeii_ref_r:.3f} R$_\\odot$", color="black", fontsize=12, ha="right", va="center")
        
    # 03:25:40 UT, split_upper_lane (Splittingの最低高度)
    typeii_split_upper_r = _radio_radius_at_time(
        pd.to_datetime("2022-06-13T03:25:40"),
        split_upper_lane,
        SAITO_FACTOR,
        TYPEII_BRANCH,
    )
    if np.isfinite(typeii_split_upper_r):
        ax.hlines(
            y=typeii_split_upper_r,
            xmin=start_dt,
            xmax=end_dt,
            color="black",
            linestyle=":",
            linewidth=1,
        )
        ax.text(pd.to_datetime("2022-06-13T03:40:00"), typeii_split_upper_r, f"{typeii_split_upper_r:.3f} R$_\\odot$", color="black", fontsize=12, ha="right", va="top")
    
    # 03:28:45 UT, UPPER_SEGMENTS (Transition時の最低高度)
    typeii_r_upper = _radio_radius_at_time(
        pd.to_datetime("2022-06-13T03:28:45"),
        UPPER_SEGMENTS,
        SAITO_FACTOR,
        TYPEII_BRANCH,
    )
    typeii_r_lower = _radio_radius_at_time(
        pd.to_datetime("2022-06-13T03:28:45"),
        LOWER_SEGMENTS,
        SAITO_FACTOR,
        TYPEII_BRANCH,
    )
    typeii_r_candidates = [r for r in (typeii_r_upper, typeii_r_lower) if np.isfinite(r)]
    if typeii_r_candidates:
        ax.hlines(
            y=max(typeii_r_candidates),
            xmin=start_dt,
            xmax=end_dt,
            color="black",
            linestyle=":",
            linewidth=1,
        )
        ax.text(pd.to_datetime("2022-06-13T03:40:00"), max(typeii_r_candidates), f"{max(typeii_r_candidates):.3f} R$_\\odot$", color="black", fontsize=12, ha="right", va="bottom")
        
    ax.vlines(
        x=pd.to_datetime("2022-06-13T03:28:45"),
        ymin=y_min,
        ymax=y_max,
        color="black",
        linestyle="--",
        linewidth=1,
    )
    ax.text(pd.to_datetime("2022-06-13T03:28:45"), 5, "Transition time \n03:28:45 UT ", color="black", fontsize=12, ha="right", va="top")
    
    

    # 軸の設定
    ax.set_xlim(start_dt, end_dt)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Time [UT]", fontsize=14)
    ax.set_ylabel("Distance from the solar center [$R_\\odot$]", fontsize=14)
    ax.set_title("CME apex and radio source height", fontsize=16)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.tick_params(axis="x", labelrotation=0, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.3)
    # 凡例の並びを指定: Main CME plot → Main CME fit → Tomography plot → Tomography fit → Radio plot → Radio fit
    handles, labels = ax.get_legend_handles_labels()
    ordered_labels: list[str] = []
    ordered_handles: list[any] = []

    preferred_order = [
        faint_cme_label,
        faint_cme_fit_label,
        main_cme_label,
        main_cme_fit_label,
        tomo_label,
        tomo_fit_label,
        radio_dense_label,
        radio_dense_fit_label,
        radio_label,
        radio_split_label,
        radio_fit_label,
    ]

    for target in preferred_order:
        if target is None:
            continue
        for h, l in zip(handles, labels):
            if l == target and l not in ordered_labels:
                ordered_handles.append(h)
                ordered_labels.append(l)
                break

    for h, l in zip(handles, labels):
        if l not in ordered_labels:
            ordered_handles.append(h)
            ordered_labels.append(l)

    ax.legend(ordered_handles, ordered_labels, loc="best", fontsize=10)
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
    start_time_str = "2022-06-13T03:15:00"
    end_time_str = "2022-06-13T03:40:00"
    script_dir = Path(__file__).resolve().parent
    search_root = script_dir
    out_png = script_dir / "output" / f"mainCME_faintCME_radio_speed_{start_time_str.replace(":", "")}_{end_time_str.replace(":", "")}.png"

    plot_faint_cme_speed(
        start_time_str=start_time_str,
        end_time_str=end_time_str,
        search_root=search_root,
        save_path=out_png,
        show=True,
    )

