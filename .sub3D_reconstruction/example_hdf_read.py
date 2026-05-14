"""Python port of ``example_hdf_read.pro`` for MAS tomography files."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from tomo_hdf_read import tomo_hdf_read


def _parse_date(text: str) -> datetime:
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {text}")


def _argmin_index(values: Sequence[float], target: float) -> int:
    arr = np.asarray(values, dtype=float)
    return int(np.argmin(np.abs(arr - target)))


def _default_data_file() -> Path:
    return Path(__file__).resolve().parent / "Rawdata" / "rho002.hdf"


def _log_floor(data: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    return np.log10(np.clip(data, floor, None))


def main(args: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Example reader for MAS tomography output")
    parser.add_argument("hdf_file", nargs="?", default=_default_data_file(), type=Path,
                        help="Tomography data file (default: Rawdata/rho002.hdf)")
    parser.add_argument("--date", default=None,
                        help="Date of interest (yyyy/mm/dd). Defaults to file metadata if available.")
    parser.add_argument("--radii", type=float, nargs="*", default=(1.0, 3.0),
                        help="Radii in Rsun for latitude/longitude plots")
    parser.add_argument("--latitudes", type=float, nargs="*", default=(0.0,),
                        help="Latitudes in degrees for longitude/radius plots")
    options = parser.parse_args(args)

    lon, lat, rad, time, volume, misc = tomo_hdf_read(options.hdf_file)

    if options.date is None:
        date_text = misc.get("startingdate") or misc.get("endingdate")
        if not date_text:
            raise RuntimeError("No date information available; please provide --date explicitly")
        target_date = str(date_text)
    else:
        target_date = options.date

    date_interest = _parse_date(target_date)
    tomo_start_text = misc.get("startingdate")
    if tomo_start_text:
        tomo_start = _parse_date(str(tomo_start_text))
        delay = (date_interest - tomo_start).days
    else:
        delay = 0.0
    time_index = _argmin_index(time, delay)

    lon_min, lon_max = float(lon.min()), float(lon.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())

    # Latitude / Longitude maps at selected radii
    for radius in options.radii:
        r_index = _argmin_index(rad, radius)
        slice_data = volume[r_index, :, :, time_index]
        view = _log_floor(slice_data)

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(
            view,
            extent=(lon_min, lon_max, lat_min, lat_max),
            origin="lower",
            aspect="auto",
            vmin=3.5,
            vmax=5.5,
            cmap="gnuplot2",
        )
        ax.set_xlabel("longitude [deg]")
        ax.set_ylabel("latitude [deg]")
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_yticks([-90, -45, 0, 45, 90])
        ax.set_title(f"lat/lon map at {rad[r_index]:.2f} Rsun")

        obscl = np.atleast_1d(misc.get("obscl", np.zeros_like(time)))
        if obscl.size > time_index:
            ax.axvline(float(obscl[time_index]), color="cyan", linestyle="--", linewidth=1.0)

        fig.colorbar(im, ax=ax, label="log10(value)")

    # Longitude / Radius maps at selected latitudes
    rad_scaled = np.asarray(rad, dtype=float) * 40.0
    y_min, y_max = float(rad_scaled.min()), float(rad_scaled.max())

    for latitude in options.latitudes:
        l_index = _argmin_index(lat, latitude)
        slice_data = volume[:, l_index, :, time_index]
        view = _log_floor(slice_data)

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(
            view,
            extent=(lon_min, lon_max, y_min, y_max),
            origin="lower",
            aspect="auto",
            vmin=3.5,
            vmax=5.5,
            cmap="gnuplot2",
        )
        ax.set_xlabel("longitude [deg]")
        ax.set_ylabel("radius * 40")
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_title(f"lon/rad map at latitude {lat[l_index]:.2f} deg")

        fig.colorbar(im, ax=ax, label="log10(value)")

    plt.show()


if __name__ == "__main__":
    main()
