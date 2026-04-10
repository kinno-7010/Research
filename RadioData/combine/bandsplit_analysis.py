from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import math
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np
import pandas as pd
from astropy.io import fits
from cdflib import CDF, cdfepoch
from matplotlib.dates import SecondLocator, MinuteLocator
import datetime as dt

from predict_type2_const_speed import f_model_from_r, invert_r_from_f, ne_saito_factor

# Fixed data locations
WIND_CDF_PATH = Path("/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/Wind/Rawdata/wi_l2_wav_rad2_20220613_v01.cdf")
HF_CDF_PATH = Path("/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/HF_plot/Rawdata/it_h1_hf_20220613_v01.cdf")
ASSA_FITS_PATHS = [
    Path("/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/e-Callisto/Rawdata/Australia-ASSA_20220613_031500_62.fit"),
    Path("/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/e-Callisto/Rawdata/Australia-ASSA_20220613_033000_62.fit"),
]

from wind_hf_assa_dynamic_spectrum import *

def _freq_to_r(f_mhz: np.ndarray | float, branch: str = "F", factor: float = 1.0) -> np.ndarray | float:
    f_arr = np.asarray(f_mhz, dtype=float)
    vec = np.vectorize(lambda v: invert_r_from_f(float(v), branch=branch, factor=factor))
    return vec(f_arr)

def _r_to_freq(r_rs: np.ndarray | float, branch: str = "F", factor: float = 1.0) -> np.ndarray | float:
    r_arr = np.asarray(r_rs, dtype=float)
    vec = np.vectorize(lambda v: f_model_from_r(float(v), branch=branch, factor=factor))
    return vec(r_arr)

def _lane_to_series(lane: Sequence[Tuple[str, float]], name: str) -> pd.Series:
    """Convert a (iso-time, freq_MHz) sequence to a time-indexed Series."""
    t = pd.to_datetime([ts for ts, _ in lane])
    y = np.asarray([v for _, v in lane], dtype=float)
    s = pd.Series(y, index=t, name=name)
    return s.sort_index()

def _merge_asof_on_time(
    left: pd.Series,
    right: pd.Series,
    left_name: str,
    right_name: str,
    tolerance: str = "2s",
) -> pd.DataFrame:
    """Time-align two Series using a nearest-neighbor merge with a tolerance."""
    ldf = left.rename(left_name).sort_index().reset_index().rename(columns={"index": "time"})
    rdf = right.rename(right_name).sort_index().reset_index().rename(columns={"index": "time"})
    out = pd.merge_asof(
        ldf,
        rdf,
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )
    out = out.dropna(subset=[left_name, right_name])
    return out

def _compute_shock_speed_from_main_lane(
    main_upper: pd.Series,
    main_lower: pd.Series,
    model_factor: float,
    tolerance: str = "2s",
) -> pd.Series:
    """Estimate radial shock speed [km/s] from the main-lane center frequency.

    Steps:
      1) time-align max/min boundaries
      2) compute center frequency (fundamental)
      3) convert f->r via the 2.8×Saito1977 mapping
      4) differentiate r(t)
    """
    center_df = _merge_asof_on_time(main_upper, main_lower, "main_upper", "main_lower", tolerance=tolerance)
    center_df["f_center"] = 0.5 * (center_df["main_upper"] + center_df["main_lower"])  # MHz (fundamental)

    # r(t) from the density model (fundamental branch)
    r_rs = _freq_to_r(center_df["f_center"].to_numpy(dtype=float), branch="F", factor=model_factor)
    t = pd.to_datetime(center_df["time"]).to_numpy(dtype="datetime64[ns]")
    t_sec = (t.astype("datetime64[ns]") - t[0]).astype("timedelta64[ns]").astype(float) * 1e-9

    # Numerical derivative dr/dt (Rsun/s)
    drdt_rsun_per_s = np.gradient(np.asarray(r_rs, dtype=float), np.asarray(t_sec, dtype=float))

    R_SUN_KM = 695700.0
    v_kms = drdt_rsun_per_s * R_SUN_KM

    return pd.Series(v_kms, index=pd.to_datetime(center_df["time"]), name="v_shock_kms").sort_index()


def _ma_from_compression_ratio(X: np.ndarray, gamma: float = 5.0 / 3.0) -> np.ndarray:
    """Alfvén Mach number from compression ratio for a low-β perpendicular shock.

    Uses: X = ((γ+1) M_A^2) / ((γ-1) M_A^2 + 2)
      => M_A^2 = 2X / ((γ+1) - (γ-1)X)
    """
    X = np.asarray(X, dtype=float)
    # Keep X within the physically admissible range for this formula.
    X = np.clip(X, 1.0 + 1e-6, (gamma + 1.0) / (gamma - 1.0) - 1e-6)
    denom = (gamma + 1.0) - (gamma - 1.0) * X
    ma2 = 2.0 * X / denom
    ma2 = np.where(ma2 > 0.0, ma2, np.nan)
    return np.sqrt(ma2)


def _make_fig2_bandwidth(
    main_upper: pd.Series,
    main_lower: pd.Series,
    split_upper: pd.Series,
    split_lower: pd.Series,
    intermittent_time: pd.Timestamp,
    output_path: Path | None,
) -> plt.Figure:
    """Fig.2: time evolution of relative band width between split and main lanes.

    Definitions (boundary-based and center-based):
      BDW_max(t)    = ( f_split_max(t)    - f_main_max(t)    ) / f_main_max(t)
      BDW_min(t)    = ( f_split_min(t)    - f_main_min(t)    ) / f_main_min(t)
      BDW_center(t) = ( f_split_center(t) - f_main_center(t) ) / f_main_center(t)

    where
      f_main_center  = (f_main_max + f_main_min)/2
      f_split_center = (f_split_max + f_split_min)/2

    Notes:
      - Ratios are invariant under harmonic->fundamental scaling.
      - Here, max/min denote the traced boundaries of each lane.
    """
    # Boundary-based relative band width
    up_df = _merge_asof_on_time(split_upper, main_upper, "split_upper", "main_upper")
    up_df["bdw_upper"] = (up_df["split_upper"] - up_df["main_upper"]) / up_df["main_upper"]

    lo_df = _merge_asof_on_time(split_lower, main_lower, "split_lower", "main_lower")
    lo_df["bdw_lower"] = (lo_df["split_lower"] - lo_df["main_lower"]) / lo_df["main_lower"]

    # Center-based relative band width
    main_df = _merge_asof_on_time(main_upper, main_lower, "main_upper", "main_lower")
    main_df["main_center"] = 0.5 * (main_df["main_upper"] + main_df["main_lower"])

    split_df = _merge_asof_on_time(split_upper, split_lower, "split_upper", "split_lower")
    split_df["split_center"] = 0.5 * (split_df["split_upper"] + split_df["split_lower"])

    main_center_ser = pd.Series(
        main_df["main_center"].to_numpy(dtype=float),
        index=pd.to_datetime(main_df["time"]),
        name="main_center",
    ).sort_index()

    split_center_ser = pd.Series(
        split_df["split_center"].to_numpy(dtype=float),
        index=pd.to_datetime(split_df["time"]),
        name="split_center",
    ).sort_index()

    cen_df = _merge_asof_on_time(split_center_ser, main_center_ser, "split_center", "main_center")
    cen_df["bdw_center"] = (cen_df["split_center"] - cen_df["main_center"]) / cen_df["main_center"]

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(
        up_df["time"],
        up_df["bdw_upper"],
        linestyle='--',
        marker='+',
        label=r"BDW (max): $(f_{\rm split,max}-f_{\rm main,max})/f_{\rm main,max}$",
        color='orange',
    )
    ax2.plot(
        lo_df["time"],
        lo_df["bdw_lower"],
        linestyle=':',
        marker='+',
        label=r"BDW (min): $(f_{\rm split,min}-f_{\rm main,min})/f_{\rm main,min}$",
        color='purple',
    )
    ax2.plot(
        cen_df["time"],
        cen_df["bdw_center"],
        linestyle='-',
        marker='+',
        label=r"BDW (center): $(f_{\rm split,c}-f_{\rm main,c})/f_{\rm main,c}$",
        color='red',
    )

    ax2.axvline(x=intermittent_time.to_pydatetime(), color='k', linestyle='--', linewidth=1.5)

    ax2.set_xlabel("Time [UT]", fontsize=12)
    ax2.set_ylabel("Relative band width", fontsize=12)
    ax2.set_title(r"Relative band width between split and main lanes", fontsize=16)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax2.xaxis.set_major_locator(SecondLocator(interval=60))
    ax2.tick_params(axis="x", rotation=0, labelrotation=0)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    if output_path is not None:
        fig2.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Fig.2 saved to {output_path} \n")

    return fig2

def _make_fig3_compression(
    main_upper: pd.Series,
    main_lower: pd.Series,
    split_upper: pd.Series,
    split_lower: pd.Series,
    intermittent_time: pd.Timestamp,
    output_path: Path | None,
) -> plt.Figure:
    """Fig.3: time evolution of shock compression estimated from split/main frequency ratios.

    Definitions:
      X_max(t)    = ( f_split_max(t)    / f_main_max(t)    )^2
      X_min(t)    = ( f_split_min(t)    / f_main_min(t)    )^2
      X_center(t) = ( f_split_center(t) / f_main_center(t) )^2

    where
      f_main_center  = (f_main_max + f_main_min)/2
      f_split_center = (f_split_max + f_split_min)/2
    """
    # Max-boundary compression
    up_df = _merge_asof_on_time(split_upper, main_upper, "split_upper", "main_upper")
    up_df["X_upper"] = (up_df["split_upper"] / up_df["main_upper"]) ** 2

    # Min-boundary compression
    lo_df = _merge_asof_on_time(split_lower, main_lower, "split_lower", "main_lower")
    lo_df["X_lower"] = (lo_df["split_lower"] / lo_df["main_lower"]) ** 2

    # Center-based compression
    main_df = _merge_asof_on_time(main_upper, main_lower, "main_upper", "main_lower")
    main_df["main_center"] = 0.5 * (main_df["main_upper"] + main_df["main_lower"])

    split_df = _merge_asof_on_time(split_upper, split_lower, "split_upper", "split_lower")
    split_df["split_center"] = 0.5 * (split_df["split_upper"] + split_df["split_lower"])

    main_center_ser = pd.Series(
        main_df["main_center"].to_numpy(dtype=float),
        index=pd.to_datetime(main_df["time"]),
        name="main_center",
    ).sort_index()

    split_center_ser = pd.Series(
        split_df["split_center"].to_numpy(dtype=float),
        index=pd.to_datetime(split_df["time"]),
        name="split_center",
    ).sort_index()

    cen_df = _merge_asof_on_time(split_center_ser, main_center_ser, "split_center", "main_center")
    cen_df["X_center"] = (cen_df["split_center"] / cen_df["main_center"]) ** 2

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(
        up_df["time"],
        up_df["X_upper"],
        linestyle='--',
        marker='+',
        label=r'Compression (max/max): $X=(f_{\rm split}/f_{\rm main})^2$',
        color='orange',
    )
    ax3.plot(
        lo_df["time"],
        lo_df["X_lower"],
        linestyle=':',
        marker='+',
        label=r'Compression (min/min): $X=(f_{\rm split}/f_{\rm main})^2$',
        color='purple',
    )
    ax3.plot(
        cen_df["time"],
        cen_df["X_center"],
        linestyle='-',
        marker='+',
        label=r'Compression (center/center): $X=(f_{\rm split,c}/f_{\rm main,c})^2$',
        color='red',
    )

    ax3.axvline(x=intermittent_time.to_pydatetime(), color='k', linestyle='--', linewidth=1.5)

    ax3.set_xlabel("Time [UT]", fontsize=12)
    ax3.set_ylabel("Compression ratio $X$", fontsize=12)
    ax3.set_title("Shock compression (03:25:40-03:28:45)", fontsize=16)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax3.xaxis.set_major_locator(SecondLocator(interval=60))
    ax3.tick_params(axis="x", rotation=0, labelrotation=0)
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="best")

    if output_path is not None:
        fig3.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Fig.3 saved to {output_path} \n")

    return fig3

def _compute_ma_va_b_dfs(
    main_upper: pd.Series,
    main_lower: pd.Series,
    split_upper: pd.Series,
    split_lower: pd.Series,
    model_factor: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute M_A, V_A, and B series for max/min/center."""
    # Compression ratios (same definitions as Fig.3)
    up_df = _merge_asof_on_time(split_upper, main_upper, "split_upper", "main_upper")
    up_df["X_upper"] = (up_df["split_upper"] / up_df["main_upper"]) ** 2
    lo_df = _merge_asof_on_time(split_lower, main_lower, "split_lower", "main_lower")
    lo_df["X_lower"] = (lo_df["split_lower"] / lo_df["main_lower"]) ** 2

    main_df = _merge_asof_on_time(main_upper, main_lower, "main_upper", "main_lower")
    main_df["main_center"] = 0.5 * (main_df["main_upper"] + main_df["main_lower"])
    split_df = _merge_asof_on_time(split_upper, split_lower, "split_upper", "split_lower")
    split_df["split_center"] = 0.5 * (split_df["split_upper"] + split_df["split_lower"])
    main_center_ser = pd.Series(
        main_df["main_center"].to_numpy(dtype=float),
        index=pd.to_datetime(main_df["time"]),
        name="main_center",
    ).sort_index()
    split_center_ser = pd.Series(
        split_df["split_center"].to_numpy(dtype=float),
        index=pd.to_datetime(split_df["time"]),
        name="split_center",
    ).sort_index()
    cen_df = _merge_asof_on_time(split_center_ser, main_center_ser, "split_center", "main_center")
    cen_df["X_center"] = (cen_df["split_center"] / cen_df["main_center"]) ** 2

    # Shock speed from main-lane center frequency (fundamental)
    v_shock_ser = _compute_shock_speed_from_main_lane(main_upper, main_lower, model_factor=model_factor)

    # Attach v_shock to the X dataframes
    v_df_up = pd.merge_asof(
        up_df.sort_values("time"),
        v_shock_ser.rename("v_shock_kms").sort_index().reset_index().rename(columns={"index": "time"}),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta("5s"),
    )
    v_df_lo = pd.merge_asof(
        lo_df.sort_values("time"),
        v_shock_ser.rename("v_shock_kms").sort_index().reset_index().rename(columns={"index": "time"}),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta("5s"),
    )
    v_df_cen = pd.merge_asof(
        cen_df.sort_values("time"),
        v_shock_ser.rename("v_shock_kms").sort_index().reset_index().rename(columns={"index": "time"}),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta("5s"),
    )

    # M_A
    v_df_up["M_A"] = _ma_from_compression_ratio(v_df_up["X_upper"].to_numpy(dtype=float))
    v_df_lo["M_A"] = _ma_from_compression_ratio(v_df_lo["X_lower"].to_numpy(dtype=float))
    v_df_cen["M_A"] = _ma_from_compression_ratio(v_df_cen["X_center"].to_numpy(dtype=float))

    # V_A [km/s]
    v_df_up["V_A_kms"] = v_df_up["v_shock_kms"] / v_df_up["M_A"]
    v_df_lo["V_A_kms"] = v_df_lo["v_shock_kms"] / v_df_lo["M_A"]
    v_df_cen["V_A_kms"] = v_df_cen["v_shock_kms"] / v_df_cen["M_A"]

    # Density and B [G] along the main-lane center r(t)
    # Use the same r(t) mapping as the speed estimate to keep the inference chain consistent.
    center_df = _merge_asof_on_time(main_upper, main_lower, "main_upper", "main_lower")
    center_df["f_center"] = 0.5 * (center_df["main_upper"] + center_df["main_lower"])  # MHz (fundamental)
    center_df["r_center"] = _freq_to_r(center_df["f_center"].to_numpy(dtype=float), branch="F", factor=model_factor)

    # Map r_center onto the v_df_{up,lo} times
    r_map = pd.merge_asof(
        v_df_up[["time"]].sort_values("time"),
        center_df[["time", "r_center"]].sort_values("time"),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta("5s"),
    )
    r_map_lo = pd.merge_asof(
        v_df_lo[["time"]].sort_values("time"),
        center_df[["time", "r_center"]].sort_values("time"),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta("5s"),
    )
    r_map_cen = pd.merge_asof(
        v_df_cen[["time"]].sort_values("time"),
        center_df[["time", "r_center"]].sort_values("time"),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta("5s"),
    )

    # Electron density [cm^-3] from 2.8×Saito1977. If the local implementation is
    # unavailable or mismatched, fall back to plasma-frequency density.
    def _ne_cm3_from_r_and_f(r_rs_arr: np.ndarray, f_mhz_arr: np.ndarray) -> np.ndarray:
        try:
            ne_cm3 = ne_saito_factor(r_rs_arr, factor=model_factor)
            return np.asarray(ne_cm3, dtype=float)
        except Exception:
            # Plasma frequency: f_p [MHz] = 8.98e-3 * sqrt(n_e [cm^-3])
            return (np.asarray(f_mhz_arr, dtype=float) / 8.98e-3) ** 2

    # Upstream density assumed from main-lane center
    ne_up_cm3 = _ne_cm3_from_r_and_f(r_map["r_center"].to_numpy(dtype=float), v_df_up["main_upper"].to_numpy(dtype=float))
    ne_lo_cm3 = _ne_cm3_from_r_and_f(r_map_lo["r_center"].to_numpy(dtype=float), v_df_lo["main_lower"].to_numpy(dtype=float))
    ne_cen_cm3 = _ne_cm3_from_r_and_f(r_map_cen["r_center"].to_numpy(dtype=float), v_df_cen["main_center"].to_numpy(dtype=float))

    # B from V_A and rho (SI), then convert to Gauss
    MU0 = 4.0e-7 * math.pi
    M_P = 1.67262192369e-27

    def _b_gauss(va_kms: np.ndarray, ne_cm3: np.ndarray) -> np.ndarray:
        va_mps = np.asarray(va_kms, dtype=float) * 1e3
        ne_m3 = np.asarray(ne_cm3, dtype=float) * 1e6
        rho = ne_m3 * M_P
        b_tesla = va_mps * np.sqrt(MU0 * rho)
        return b_tesla * 1.0e4

    v_df_up["B_G"] = _b_gauss(v_df_up["V_A_kms"].to_numpy(dtype=float), ne_up_cm3)
    v_df_lo["B_G"] = _b_gauss(v_df_lo["V_A_kms"].to_numpy(dtype=float), ne_lo_cm3)
    v_df_cen["B_G"] = _b_gauss(v_df_cen["V_A_kms"].to_numpy(dtype=float), ne_cen_cm3)

    return v_df_up, v_df_lo, v_df_cen


def _make_fig4_ma(
    v_df_up: pd.DataFrame,
    v_df_lo: pd.DataFrame,
    v_df_cen: pd.DataFrame,
    intermittent_time: pd.Timestamp,
    output_path: Path | None,
) -> plt.Figure:
    """Fig.4: time evolution of M_A using 2.8×Saito1977."""
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.plot(v_df_up["time"], v_df_up["M_A"], linestyle='--', marker='+', label=r'$M_A$ (max/max)', color='orange')
    ax4.plot(v_df_lo["time"], v_df_lo["M_A"], linestyle=':', marker='+', label=r'$M_A$ (min/min)', color='purple')
    ax4.plot(v_df_cen["time"], v_df_cen["M_A"], linestyle='-', marker='+', label=r'$M_A$ (center/center)', color='red')
    ax4.axvline(x=intermittent_time.to_pydatetime(), color='k', linestyle='--', linewidth=1.5)
    ax4.set_xlabel("Time [UT]", fontsize=12)
    ax4.set_ylabel(r"Alfvén Mach number $M_A$", fontsize=12)
    ax4.set_title(r"Alfvén Mach number from band-splitting (2.8$\times$Saito1977)", fontsize=14)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax4.xaxis.set_major_locator(SecondLocator(interval=60))
    ax4.tick_params(axis="x", rotation=0, labelrotation=0)
    ax4.grid(True, alpha=0.25)
    ax4.legend(loc="best")

    if output_path is not None:
        fig4_path = output_path.with_name(output_path.stem + "_Fig4_MA.png")
        fig4.savefig(fig4_path, dpi=300, bbox_inches="tight")
        print(f"Fig.4 saved to {fig4_path} \n")

    return fig4


def _make_fig5_va_b(
    v_df_up: pd.DataFrame,
    v_df_lo: pd.DataFrame,
    v_df_cen: pd.DataFrame,
    intermittent_time: pd.Timestamp,
    output_path: Path | None,
) -> Tuple[plt.Figure, plt.Figure]:
    """Fig.5: time evolution of V_A and B using 2.8×Saito1977 (separate figures)."""
    v_shock_mean = float(np.nanmean(v_df_up["v_shock_kms"].to_numpy(dtype=float)))
    v_shock_std = float(np.nanstd(v_df_up["v_shock_kms"].to_numpy(dtype=float)))
    fig5_va, ax5a = plt.subplots(figsize=(10, 5))
    ax5a.plot(v_df_up["time"], v_df_up["V_A_kms"], linestyle='--', marker='+', label=r'$V_A$ (max/max)', color='orange')
    ax5a.plot(v_df_lo["time"], v_df_lo["V_A_kms"], linestyle=':', marker='+', label=r'$V_A$ (min/min)', color='purple')
    ax5a.plot(v_df_cen["time"], v_df_cen["V_A_kms"], linestyle='-', marker='+', label=r'$V_A$ (center/center)', color='red')
    ax5a.axvline(x=intermittent_time.to_pydatetime(), color='k', linestyle='--', linewidth=1.5)
    ax5a.set_xlabel("Time [UT]", fontsize=12)
    ax5a.set_ylabel(r"Alfvén speed $V_A$ [km s$^{-1}$]", fontsize=12)
    ax5a.set_title(
        rf"Alfvén speed from band-splitting (2.8$\times$Saito1977), "
        rf"$v_\mathrm{{shock}}={v_shock_mean:.1f}\pm{v_shock_std:.1f}$",
        fontsize=14,
    )
    ax5a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax5a.xaxis.set_major_locator(SecondLocator(interval=60))
    ax5a.tick_params(axis="x", rotation=0, labelrotation=0)
    ax5a.grid(True, alpha=0.25)
    ax5a.legend(loc="best")

    fig5_b, ax5b = plt.subplots(figsize=(10, 5))
    ax5b.plot(v_df_up["time"], v_df_up["B_G"], linestyle='--', marker='+', label=r'$B$ (max/max)', color='orange')
    ax5b.plot(v_df_lo["time"], v_df_lo["B_G"], linestyle=':', marker='+', label=r'$B$ (min/min)', color='purple')
    ax5b.plot(v_df_cen["time"], v_df_cen["B_G"], linestyle='-', marker='+', label=r'$B$ (center/center)', color='red')
    ax5b.axvline(x=intermittent_time.to_pydatetime(), color='k', linestyle='--', linewidth=1.5)
    ax5b.set_xlabel("Time [UT]", fontsize=12)
    ax5b.set_ylabel(r"Magnetic field $B$ [G]", fontsize=12)
    ax5b.set_title(r"Magnetic field from band-splitting (2.8$\times$Saito1977)", fontsize=14)
    ax5b.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax5b.xaxis.set_major_locator(SecondLocator(interval=60))
    ax5b.tick_params(axis="x", rotation=0, labelrotation=0)
    ax5b.grid(True, alpha=0.25)
    ax5b.legend(loc="best")

    if output_path is not None:
        fig5_va_path = output_path.with_name(output_path.stem + "_Fig5_VA.png")
        fig5_b_path = output_path.with_name(output_path.stem + "_Fig5_B.png")
        fig5_va.savefig(fig5_va_path, dpi=300, bbox_inches="tight")
        fig5_b.savefig(fig5_b_path, dpi=300, bbox_inches="tight")
        print(f"Fig.5 (V_A) saved to {fig5_va_path} \n")
        print(f"Fig.5 (B) saved to {fig5_b_path} \n")

    return fig5_va, fig5_b


def plot_dynamic_spectrum_bandsplit(
    spectrum: pd.DataFrame,
    output_path: Path | None,
    title: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    log_scale: bool,
    show: bool,
    start_time, 
    end_time,
    min_frequency,
    max_frequency,
    model_factor: float,
) -> None:
    """Generate and optionally save the combined dynamic spectrum figure."""
    print("----------------figure export----------------")
    if spectrum.empty:
        raise ValueError("Combined spectrum is empty. Check the time range and input files.")

    time_axis = spectrum.index.to_pydatetime()
    freq_axis = spectrum.columns.to_numpy()
    values = spectrum.to_numpy().T  # shape -> (freq, time)

    # fig, ax = plt.subplots(figsize=(18, 8))
    fig, ax = plt.subplots(figsize=(16, 8))
    mesh = ax.pcolormesh(
        mdates.date2num(time_axis),
        freq_axis,
        values,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    
    ######################### max/min lane data ##############################
    upper_lane = [("2022-06-13T03:25:40", 35.7), ("2022-06-13T03:25:50", 35.9), ("2022-06-13T03:26:00", 36.1), ("2022-06-13T03:26:10", 36.2),
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
    
    lower_lane = [("2022-06-13T03:25:40", 33.2), ("2022-06-13T03:25:50", 32.8), ("2022-06-13T03:26:00", 32.4), ("2022-06-13T03:26:10", 31.9),
                  
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

    # 文字列→datetime→mdates数値に変換してから描画（直接文字列を渡さない）
    def _to_mdates_xy(lane):
        t = [dt.datetime.fromisoformat(ts) for ts, _ in lane]
        y = [v for _, v in lane]
        return mdates.date2num(t), y

    upper_x, upper_y = _to_mdates_xy(upper_lane)
    lower_x, lower_y = _to_mdates_xy(lower_lane)

    # ドリフトレート（MHz/s）の平均・標準偏差を算出
    def _drift_rate_stats(x_mdates, y_mhz):
        if len(x_mdates) < 2:
            return None, None
        dt_sec = np.diff(x_mdates) * 86400.0
        df_mhz = np.diff(y_mhz)
        rates = df_mhz / dt_sec
        return float(np.mean(rates)), float(np.std(rates))

    upper_mean, upper_std = _drift_rate_stats(upper_x, upper_y)
    lower_mean, lower_std = _drift_rate_stats(lower_x, lower_y)

    # プロットは線のみ（scatterはコメントアウトのまま）
    ax.scatter(upper_x, upper_y, color='orange', marker='+', s=50, zorder=12)
    ax.scatter(lower_x, lower_y, color='purple', marker='+', s=50, zorder=12)
    upper_label = 'Max Lane'
    lower_label = 'Min Lane'
    # この周波数は 2nd harmonic のため、Fundamental 換算として 1/2 を用いる
    if upper_mean is not None:
        upper_label += f" (FDR={upper_mean/2:.3e}±{upper_std/2:.3e} MHz/s)"
    if lower_mean is not None:
        lower_label += f" (FDR={lower_mean/2:.3e}±{lower_std/2:.3e} MHz/s)"
    ax.plot(upper_x, upper_y, color='orange', linestyle='--', linewidth=2, label=upper_label)
    ax.plot(lower_x, lower_y, color='purple', linestyle='--', linewidth=2, label=lower_label)
    
    ######################### vertical line ##############################
    intermittent_time = pd.Timestamp("2022-06-13T03:28:45")
    ax.axvline(x=mdates.date2num(intermittent_time), color='white', linestyle='--', linewidth=2)
    ax.text(mdates.date2num(intermittent_time), 47, intermittent_time.strftime('%H:%M:%S'), color='white', fontsize=14, ha='right', va='top')


    ######################### band-split data ##############################
    # intermittent_timeより前までのデータのみを抽出してプロット
    shift_upper_frequency = 8.3
    # upper_x, upper_y: mdates数値, y値
    upper_x_np = np.array(upper_x)
    upper_y_np = np.array(upper_y)
    shifted_upper_y_np = upper_y_np + shift_upper_frequency
    cutoff = mdates.date2num(pd.Timestamp("2022-06-13T03:28:45"))
    mask = upper_x_np < cutoff
    ax.plot(upper_x_np[mask], shifted_upper_y_np[mask], color='#FFD580', linestyle=':', linewidth=2, label=f'Band-split Max Lane (max lane +{shift_upper_frequency} MHz)')
    
    
    split_lower_lane = [("2022-06-13T03:25:40", 40.2), ("2022-06-13T03:25:50", 39.8), ("2022-06-13T03:26:00", 39.4), ("2022-06-13T03:26:10", 38.9),
                  
                  ("2022-06-13T03:26:20", 38.5),
                  
                  ("2022-06-13T03:26:30", 38), ("2022-06-13T03:26:40", 38), ("2022-06-13T03:26:50", 38.0),
                  ("2022-06-13T03:27:00", 38.5), ("2022-06-13T03:27:10", 38.5), ("2022-06-13T03:27:20", 38.5),
                  
                  ("2022-06-13T03:27:30", 38.5),
                  ("2022-06-13T03:27:40", 38.5), ("2022-06-13T03:27:50", 38.5), ("2022-06-13T03:28:00", 39.0), ("2022-06-13T03:28:10", 39),
                  ("2022-06-13T03:28:20", 39.5),
                  
                  ("2022-06-13T03:28:30", 39.5), ("2022-06-13T03:28:40", 39.5), ("2022-06-13T03:28:50", 39.5),
                  ]
    split_lower_x, split_lower_y = _to_mdates_xy(split_lower_lane)
    ax.plot(split_lower_x, split_lower_y, color='#C5A3FF', linestyle=':', linewidth=2, label=f'Band-split Min Lane (original)')

    # ------------------ center frequency (max/min lanes) ------------------
    main_max_ser = _lane_to_series(upper_lane, name="main_max")
    main_min_ser = _lane_to_series(lower_lane, name="main_min")

    split_max_times = pd.to_datetime([ts for ts, _ in upper_lane])
    split_max_freq = np.asarray([v for _, v in upper_lane], dtype=float) + float(shift_upper_frequency)
    split_max_ser = pd.Series(split_max_freq, index=split_max_times, name="split_max").sort_index()
    split_max_ser = split_max_ser[split_max_ser.index < intermittent_time]

    split_min_ser = _lane_to_series(split_lower_lane, name="split_min")
    split_min_ser = split_min_ser[split_min_ser.index < intermittent_time]

    center_split_df = _merge_asof_on_time(split_max_ser, split_min_ser, "split_max", "split_min")
    center_split_df["center_split"] = 0.5 * (center_split_df["split_max"] + center_split_df["split_min"])
    center_main_df = _merge_asof_on_time(main_max_ser, main_min_ser, "main_max", "main_min")
    center_main_df["center_main"] = 0.5 * (center_main_df["main_max"] + center_main_df["main_min"])

    center_split_x = mdates.date2num(pd.to_datetime(center_split_df["time"]))
    center_split_y = center_split_df["center_split"].to_numpy(dtype=float)
    center_main_x = mdates.date2num(pd.to_datetime(center_main_df["time"]))
    center_main_y = center_main_df["center_main"].to_numpy(dtype=float)

    center_split_mean, center_split_std = _drift_rate_stats(center_split_x, center_split_y)
    center_main_mean, center_main_std = _drift_rate_stats(center_main_x, center_main_y)

    center_split_label = "Center Split Lane"
    center_main_label = "Center Main Lane"
    # この周波数は 2nd harmonic のため、Fundamental 換算として 1/2 を用いる
    if center_split_mean is not None:
        center_split_label += f" (FDR={center_split_mean/2:.3e}±{center_split_std/2:.3e} MHz/s)"
    if center_main_mean is not None:
        center_main_label += f" (FDR={center_main_mean/2:.3e}±{center_main_std/2:.3e} MHz/s)"

    ax.plot(center_split_x, center_split_y, color='red', linestyle='-', linewidth=2, label=center_split_label)
    ax.plot(center_main_x, center_main_y, color='red', linestyle=':', linewidth=2, label=center_main_label)

    
    
    ######################### plot settings ##############################
    ax.set_ylabel("Frequency [MHz]", fontsize=16)
    if log_scale is not False:
        ax.set_yscale("log")
    else:
        ax.set_yscale("linear")
    ax.set_xlabel("Time [UT]", fontsize=16)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(mdates.date2num(start_time), mdates.date2num(end_time))
    ax.set_ylim(min_frequency, max_frequency)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.xaxis.set_major_locator(SecondLocator(interval=60))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.0f}"))
    ax.tick_params(axis="x", rotation=0, labelrotation=0, labelsize=14)
    ax.tick_params(axis="y", labelsize=12)

    # --- 右軸: Radial distance (2nd harmonic; 2.8×Saito1977) ---
    secax = ax.secondary_yaxis(
        "right",
        functions=(
            lambda f_mhz: _freq_to_r(f_mhz, branch="H", factor=model_factor),
            lambda r_rs: _r_to_freq(r_rs, branch="H", factor=model_factor),
        ),
    )
    secax.set_ylabel(
        f"Radial distance (Harmonic) [R$_\\odot$] ({model_factor}× Saito1977)",
        fontsize=14,
    )
    secax.tick_params(axis="y", labelsize=12)
    secax.yaxis.set_major_locator(MultipleLocator(0.1))
    secax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.1f}"))

    ax.legend(loc="upper right", fontsize=14)

    # ================= Fig.2 / Fig.3 (derived quantities) =================
    # Second harmonic -> fundamental conversion (freq/2) for derived quantities
    harmonic_to_fundamental = 0.5

    # Convert picked segments to time-indexed series (fundamental)
    main_upper_ser = _lane_to_series(upper_lane, name="main_upper") * harmonic_to_fundamental
    main_lower_ser = _lane_to_series(lower_lane, name="main_lower") * harmonic_to_fundamental

    # Band-split segments (use the same time cutoff as the plotted split)
    split_upper_times = pd.to_datetime([ts for ts, _ in upper_lane])
    split_upper_freq = (np.asarray([v for _, v in upper_lane], dtype=float) + float(shift_upper_frequency)) * harmonic_to_fundamental
    split_upper_ser = pd.Series(split_upper_freq, index=split_upper_times, name="split_upper").sort_index()
    split_upper_ser = split_upper_ser[split_upper_ser.index < intermittent_time]

    split_lower_ser = _lane_to_series(split_lower_lane, name="split_lower") * harmonic_to_fundamental
    split_lower_ser = split_lower_ser[split_lower_ser.index < intermittent_time]

    fig2_path = None if output_path is None else output_path.with_name(output_path.stem + "_Fig2_bandwidth.png")
    fig3_path = None if output_path is None else output_path.with_name(output_path.stem + "_Fig3_compression.png")

    fig2 = _make_fig2_bandwidth(
        main_upper=main_upper_ser,
        main_lower=main_lower_ser,
        split_upper=split_upper_ser,
        split_lower=split_lower_ser,
        intermittent_time=intermittent_time,
        output_path=fig2_path,
    )

    fig3 = _make_fig3_compression(
        main_upper=main_upper_ser,
        main_lower=main_lower_ser,
        split_upper=split_upper_ser,
        split_lower=split_lower_ser,
        intermittent_time=intermittent_time,
        output_path=fig3_path,
    )

    # ================= Fig.4 / Fig.5 (M_A, V_A, B) =================
    v_df_up, v_df_lo, v_df_cen = _compute_ma_va_b_dfs(
        main_upper=main_upper_ser,
        main_lower=main_lower_ser,
        split_upper=split_upper_ser,
        split_lower=split_lower_ser,
        model_factor=model_factor,
    )
    fig4 = _make_fig4_ma(
        v_df_up=v_df_up,
        v_df_lo=v_df_lo,
        v_df_cen=v_df_cen,
        intermittent_time=intermittent_time,
        output_path=output_path,
    )
    fig5_va, fig5_b = _make_fig5_va_b(
        v_df_up=v_df_up,
        v_df_lo=v_df_lo,
        v_df_cen=v_df_cen,
        intermittent_time=intermittent_time,
        output_path=output_path,
    )

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {output_path} \n")

    if show:
        plt.show()
    else:
        plt.close(fig)
        # Close derived-quantity figures as well
        try:
            plt.close(fig2)
        except Exception:
            pass
        try:
            plt.close(fig3)
        except Exception:
            pass

        try:
            plt.close(fig4)
        except Exception:
            pass
        try:
            plt.close(fig5_va)
        except Exception:
            pass
        try:
            plt.close(fig5_b)
        except Exception:
            pass




def main(
    start_time,
    end_time,
    min_frequency,
    max_frequency,
    draw_lines: bool = True,
    point_start_time=None,
    point_end_time=None,
    point_start_frequency=None,
    point_end_frequency=None,
    point_start_time_2=None,
    point_end_time_2=None,
    point_start_frequency_2=None,
    point_end_frequency_2=None,
) -> None:
    # ----- Configuration section (edit as needed) -----
    cadence = "0.5s"
    polarization = "RH"  # or "LH"
    show_plot = True
    log_scale = True
    cmap = "viridis"
    vmin = None
    vmax = None
    model_factor = 2.8
    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)
    point_start_time = pd.Timestamp(point_start_time) if point_start_time is not None else None
    point_end_time = pd.Timestamp(point_end_time) if point_end_time is not None else None
    point_start_frequency = float(point_start_frequency) if point_start_frequency is not None else None
    point_end_frequency = float(point_end_frequency) if point_end_frequency is not None else None
    point_start_time_2 = pd.Timestamp(point_start_time_2) if point_start_time_2 is not None else None
    point_end_time_2 = pd.Timestamp(point_end_time_2) if point_end_time_2 is not None else None
    point_start_frequency_2 = float(point_start_frequency_2) if point_start_frequency_2 is not None else None
    point_end_frequency_2 = float(point_end_frequency_2) if point_end_frequency_2 is not None else None

    point_start_time_1 = point_start_time
    point_end_time_1 = point_end_time
    point_start_frequency_1 = point_start_frequency
    point_end_frequency_1 = point_end_frequency
    # -------------------------------------------------

    if start_time >= end_time:
        raise ValueError("Start time must be earlier than end time.")

    wind_times, wind_freqs, wind_values = load_wind_rad2(WIND_CDF_PATH)
    hf_times, hf_freqs, hf_values = load_hf(HF_CDF_PATH, polarization)
    assa_times, assa_freqs, assa_values = load_callisto(ASSA_FITS_PATHS)

    target_index = pd.date_range(start=start_time, end=end_time, freq=cadence)
    if len(target_index) == 0:
        raise ValueError("Target index is empty. Check cadence and time range.")

    time_margin = pd.to_timedelta(cadence)
    extended_range = (start_time - time_margin, end_time + time_margin)

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

    frequency_index = combined.columns.astype(float)
    freq_mask = (frequency_index >= min_frequency) & (frequency_index <= max_frequency)
    combined = combined.loc[:, freq_mask]

    title = "Dynamic Spectrum; Wind/RAD2 (1-14 MHz) + HF antenna (14-40 MHz) + Australia-ASSA (40-85 MHz)"
    if combined.empty:
        raise ValueError("No data remains after applying the frequency bounds.")

    data_output_path = Path(f"/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/combine/wind_hf_assa_dynamic_spectrum_{start_time.strftime('%Y-%m-%d_%H%M%S')}_{end_time.strftime('%H%M%S')}.csv")
    figure_path = Path(f"/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/combine/bandsplit_analysis_{start_time.strftime('%Y-%m-%d_%H%M%S')}_{end_time.strftime('%H%M%S')}.png")
    data_output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    # export_dataframe(combined, data_output_path)
    
    plot_dynamic_spectrum_bandsplit(
        combined,
        figure_path,
        title=title,
        cmap=cmap,
        vmin=1.0,
        vmax=1.1,
        log_scale=log_scale,
        show=show_plot,
        start_time=start_time,
        end_time=end_time,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        model_factor=model_factor,
    )


if __name__ == "__main__":
    start_time = "2022-06-13T03:25:00"
    end_time = "2022-06-13T03:33:00"
    min_frequency = 25
    max_frequency = 47
    main(start_time, end_time, min_frequency, max_frequency)

