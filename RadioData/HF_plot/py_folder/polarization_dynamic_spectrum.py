#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from cdflib import CDF, cdfepoch
from matplotlib.dates import SecondLocator
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator
from scipy.ndimage import median_filter


def _load_cdf(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    cdf = CDF(str(path))
    epoch = cdf.varget("Epoch")
    times = cdfepoch.to_datetime(epoch)
    frequency_hz = cdf.varget("Frequency")
    frequencies_mhz = frequency_hz.astype(np.float64) / 1e6

    polarization_data: Dict[str, np.ndarray] = {}
    for pol in ("RH", "LH"):
        raw = cdf.varget(pol)
        attrs = cdf.varattsget(pol)
        fill_attr = attrs.get("FILLVAL")
        if fill_attr is not None:
            fill_value = np.array(fill_attr, dtype=raw.dtype)
            mask = raw == fill_value
        else:
            mask = np.zeros_like(raw, dtype=bool)
        float_data = raw.astype(np.float32)
        if mask.any():
            float_data = np.where(mask, np.nan, float_data)
        polarization_data[pol] = float_data


    return times, frequencies_mhz, polarization_data


def _slice_time_frequency(
    times: np.ndarray,
    frequencies: np.ndarray,
    data: np.ndarray,
    start_time: str | None,
    end_time: str | None,
    start_frequency: float | None,
    end_frequency: float | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_mask = np.ones(times.shape, dtype=bool)
    if start_time is not None:
        start_dt = np.datetime64(start_time)
        time_mask &= times >= start_dt
    if end_time is not None:
        end_dt = np.datetime64(end_time)
        time_mask &= times <= end_dt

    sliced_times = times[time_mask]
    sliced_data = data[time_mask, :]

    freq_mask = np.ones(frequencies.shape, dtype=bool)
    if start_frequency is not None:
        freq_mask &= frequencies >= float(start_frequency)
    if end_frequency is not None:
        freq_mask &= frequencies <= float(end_frequency)

    return sliced_times, frequencies[freq_mask], sliced_data[:, freq_mask]


def _apply_processing(
    data: np.ndarray,
    median_size: Tuple[int, int],
    background: str,
) -> np.ndarray:
    processed = data.astype(np.float32, copy=True)

    if background == "median_time":
        background_level = np.nanmedian(processed, axis=0, keepdims=True)
        processed = processed - background_level

    if median_size != (1, 1):
        finite_mask = np.isfinite(processed)
        if finite_mask.any():
            fill_value = np.nanmedian(processed[finite_mask])
            filled = np.where(finite_mask, processed, fill_value)
            filtered = median_filter(filled, size=median_size, mode="nearest")
            processed = np.where(finite_mask, filtered, np.nan)

    return processed


def _auto_color_limits(data: np.ndarray) -> Tuple[float, float]:
    finite_values = data[np.isfinite(data)]
    if finite_values.size == 0:
        return 0.0, 1.0
    return (
        float(np.nanpercentile(finite_values, 2)),
        float(np.nanpercentile(finite_values, 98)),
    )


def _plot_dynamic_spectrum(
    ax: plt.Axes,
    times: np.ndarray,
    frequencies_mhz: np.ndarray,
    data: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    time_tick: int | None,
    freq_tick: float | None,
    colorbar_label: str,
) -> None:
    if times.size == 0 or frequencies_mhz.size == 0:
        raise ValueError("時間または周波数の条件でデータが空になりました。")

    extent = [
        mdates.date2num(times[0]),
        mdates.date2num(times[-1]),
        frequencies_mhz[0],
        frequencies_mhz[-1],
    ]

    mesh = ax.imshow(
        data.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )

    locator = SecondLocator(interval=time_tick) if time_tick else mdates.AutoDateLocator()
    formatter = mdates.AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_ylabel("Frequency (MHz)")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}"))
    if freq_tick is not None:
        ax.yaxis.set_minor_locator(MultipleLocator(freq_tick))
    ax.tick_params(axis="both", which="major", labelsize=12)

    cbar = ax.figure.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(colorbar_label)
    cbar.ax.tick_params(labelsize=12)


def _compute_polarization_ratio(lh_db: np.ndarray, rh_db: np.ndarray) -> np.ndarray:
    """
    Compute LH/RH power ratio from dB inputs.
    """
    difference_db = lh_db.astype(np.float64) - rh_db.astype(np.float64)
    ratio = np.power(10.0, difference_db / 10.0)
    invalid_mask = (~np.isfinite(lh_db)) | (~np.isfinite(rh_db)) | (~np.isfinite(ratio))
    ratio = np.where(invalid_mask, np.nan, ratio)
    return ratio.astype(np.float32)


def main(start_time: str, end_time: str, min_frequency: float, max_frequency: float) -> None:
    config = {
        "cdf_path": Path(
            "/mnt/d/wsl/home/kinno-7010/Research_data/RadioData/HF_plot/Rawdata/it_h1_hf_20220613_v01.cdf"
        ),
        "median_size": (1, 1),
        "background": "none",
        "vmin": None,
        "vmax": None,
        "time_tick": None,
        "freq_tick": None,
        "output": None,
        "title": "LH/RH Ratio Dynamic Spectrum",
        "colorbar_label": "LH/RH Power Ratio",
    }

    times, frequencies_mhz, pol_data = _load_cdf(config["cdf_path"])
    ratio = _compute_polarization_ratio(pol_data["LH"], pol_data["RH"])

    fig, ax = plt.subplots(
        figsize=(12, 5),
        constrained_layout=True,
    )

    sliced_times, sliced_freqs, sliced_ratio = _slice_time_frequency(
        times,
        frequencies_mhz,
        ratio,
        start_time,
        end_time,
        min_frequency,
        max_frequency,
    )
    processed = _apply_processing(
        sliced_ratio, tuple(config["median_size"]), config["background"]
    )
    vmin, vmax = (
        (config["vmin"], config["vmax"])
        if config["vmin"] is not None and config["vmax"] is not None
        else _auto_color_limits(processed)
    )
    _plot_dynamic_spectrum(
        ax,
        sliced_times,
        sliced_freqs,
        processed,
        config["title"],
        vmin,
        vmax,
        config["time_tick"],
        config["freq_tick"],
        config["colorbar_label"],
    )

    ax.set_xlabel("Time (UTC)")

    if config["output"]:
        config["output"].parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(config["output"], dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main(
        start_time="2022-06-13T03:20:00",
        end_time="2022-06-13T04:30:00",
        min_frequency=15.0,
        max_frequency=42.0,
    )
