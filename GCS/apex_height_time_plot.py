"""Plot apex height time series from GCS output CSV files.

The script discovers `apex_height_*.csv` files located under directories named
`output` within the repository, combines them into a single time series, and
plots Apex height (in solar radii) against observation time. Use the
`start_time` and `end_time` arguments to limit the plotted range.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

SOLAR_RADIUS_KM = 695700.0

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd



def _discover_csv_files(search_root: Path) -> list[Path]:
    """Return all apex_height CSV files that live in an `output` directory."""
    csv_paths: list[Path] = []
    for path in sorted(search_root.rglob("apex_height_*.csv")):
        if "output" in path.parts:
            csv_paths.append(path)
    return csv_paths


def _load_apex_time_series(csv_paths: Iterable[Path]) -> pd.DataFrame:
    """Load apex height measurements from the provided CSV files."""
    records: list[dict[str, object]] = []
    for csv_path in csv_paths:
        try:
            frame = pd.read_csv(csv_path)
        except Exception as exc:  # pragma: no cover - informative logging only
            print(f"Warning: failed to read {csv_path}: {exc}")
            continue

        if "time" not in frame.columns or "h_apex" not in frame.columns:
            print(f"Warning: skipped {csv_path} (missing required columns)")
            continue

        subset = frame[["time", "h_apex"]].copy()
        subset.dropna(inplace=True)
        if subset.empty:
            continue

        subset["datetime"] = pd.to_datetime(subset["time"], utc=False, errors="coerce")
        subset["apex_height"] = pd.to_numeric(subset["h_apex"], errors="coerce")
        subset.dropna(inplace=True)

        for row in subset.itertuples(index=False):
            records.append(
                {
                    "datetime": row.datetime.to_pydatetime(),
                    "apex_height": float(row.apex_height),
                    "source": str(csv_path),
                }
            )

    data = pd.DataFrame.from_records(records)
    if data.empty:
        return data

    data.sort_values("datetime", inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def _parse_time_label(label: Optional[str], reference_date: datetime.date) -> Optional[datetime]:
    """Convert a user-supplied time label into a concrete datetime."""
    if label is None:
        return None

    text = label.strip()
    if not text:
        return None

    datetime_formats = (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
    )
    for fmt in datetime_formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    time_formats = ("%H:%M:%S", "%H:%M")
    for fmt in time_formats:
        try:
            parsed = datetime.strptime(text, fmt)
            return datetime.combine(reference_date, parsed.time())
        except ValueError:
            continue

    raise ValueError(f"Unsupported time format: {label}")


def plot_apex_height_time_series(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    search_root: Optional[Path] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot Apex height against time within the requested interval.

    Parameters
    ----------
    start_time : Optional[str]
        Inclusive start limit for the plot. Accepts `HH:MM[:SS]` or ISO-8601
        datetime strings. When only a time of day is provided, the date of the
        earliest measurement is used.
    end_time : Optional[str]
        Inclusive end limit for the plot. Same accepted formats as
        `start_time`.
    search_root : Optional[Path]
        Directory under which `output/apex_height_*.csv` files will be
        discovered. Defaults to the parent directory of this script.
    save_path : Optional[Path]
        If provided, the plot is saved to this location instead of only being
        displayed.
    show : bool
        Whether to display the plot with `plt.show()`.
    """
    if search_root is None:
        search_root = Path(__file__).resolve().parent

    if not search_root.exists():
        raise FileNotFoundError(f"Search root does not exist: {search_root}")

    csv_paths = _discover_csv_files(search_root)
    if not csv_paths:
        raise FileNotFoundError(
            f"No apex_height_*.csv files found beneath {search_root}"
        )

    data = _load_apex_time_series(csv_paths)
    if data.empty:
        raise ValueError("No valid apex height data could be loaded.")

    reference_date = data.loc[0, "datetime"].date()
    start_dt = _parse_time_label(start_time, reference_date) if start_time else None
    end_dt = _parse_time_label(end_time, reference_date) if end_time else None

    if start_dt and end_dt and end_dt < start_dt:
        raise ValueError("end_time must be greater than or equal to start_time")

    filtered = data.copy()
    if start_dt:
        filtered = filtered[filtered["datetime"] >= start_dt]
    if end_dt:
        filtered = filtered[filtered["datetime"] <= end_dt]

    if filtered.empty:
        raise ValueError("No data points fall within the requested time range.")

    filtered = filtered.copy()
    filtered["datetime"] = pd.to_datetime(filtered["datetime"])
    filtered.sort_values("datetime", inplace=True)
    filtered.reset_index(drop=True, inplace=True)

    legend_parts: list[str] = []
    if len(filtered) >= 2:
        delta_seconds = filtered["datetime"].diff().dt.total_seconds().iloc[1:]
        delta_height = filtered["apex_height"].diff().iloc[1:]
        valid_mask = delta_seconds != 0
        if valid_mask.any():
            velocities = (delta_height[valid_mask] / delta_seconds[valid_mask]).dropna()
            if not velocities.empty:
                velocities_km_s = velocities * SOLAR_RADIUS_KM
                mean_velocity = velocities_km_s.mean()
                if len(velocities_km_s) > 1:
                    std_velocity = velocities_km_s.std(ddof=1)
                else:
                    std_velocity = 0.0
                legend_parts.append(
                    f"1st order: $v$={mean_velocity:.1f}±{std_velocity:.1f} km/s"
                )

    if len(filtered) >= 3:
        times_sec = (
            filtered["datetime"] - filtered["datetime"].iloc[0]
        ).dt.total_seconds().to_numpy()
        heights_rsun = filtered["apex_height"].to_numpy()

        finite_mask = np.isfinite(times_sec) & np.isfinite(heights_rsun)
        if finite_mask.sum() >= 3 and np.unique(times_sec[finite_mask]).size >= 3:
            try:
                coeffs, cov = np.polyfit(
                    times_sec[finite_mask],
                    heights_rsun[finite_mask],
                    deg=2,
                    cov=True,
                )
            except np.linalg.LinAlgError:
                coeffs, cov = None, None
            if coeffs is not None:
                a, b, _ = coeffs  # h(t) = a*t^2 + b*t + c
                velocity0_km_s = b * SOLAR_RADIUS_KM
                acceleration_km_s2 = (2.0 * a) * SOLAR_RADIUS_KM

                std_v0 = 0.0
                std_acc = 0.0
                if cov is not None and np.shape(cov) == (3, 3):
                    cov = np.asarray(cov, dtype=float)
                    if np.isfinite(cov[1, 1]):
                        std_v0 = float(np.sqrt(max(cov[1, 1], 0.0)) * SOLAR_RADIUS_KM)
                    if np.isfinite(cov[0, 0]):
                        std_acc = float(np.sqrt(max(cov[0, 0], 0.0)) * (2.0 * SOLAR_RADIUS_KM))

                legend_parts.append(
                    f"2nd order: $v_0$={velocity0_km_s:.1f}±{std_v0:.1f} km/s, "
                    f"$a$={acceleration_km_s2:.2f}±{std_acc:.2f}"
                    "$\\mathrm{km/s^2}$"
                )

    legend_label = "\n".join(legend_parts) if legend_parts else None

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        filtered["datetime"],
        filtered["apex_height"],
        marker="o",
        linestyle="-",
        label=legend_label,
    )
    # faint CMEの時間変化
    
    
    ax.set_xlabel("Time [UT]", fontsize=14)
    ax.set_ylabel("Apex height [$R_\\odot$]", fontsize=14)
    # x軸の目盛を斜めにしない（水平にする）
    ax.set_title("Apex height", fontsize=16)
    ax.set_ylim(filtered["apex_height"].min() - 0.5, filtered["apex_height"].max() + 0.1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.tick_params(axis='x', labelsize=12, labelrotation=0)
    ax.tick_params(axis='y', labelsize=12)
    # 03:25:40に縦線を引く
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
        ax.text(vline_dt, ax.get_ylim()[0], label, color=text_color, fontsize=12, ha="right", va="bottom", alpha=0.8)
    
    if legend_label:
        ax.legend(loc="best", fontsize=12)
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


def main(start_time: Optional[str] = None, end_time: Optional[str] = None) -> None:
    # Update these values as needed before running the script.
    script_dir = Path(__file__).resolve().parent
    search_root = script_dir.parent  # search from the repository root by default


    save_path = f"./output/apex_height_time_series_{start_time.replace(':', '')}_{end_time.replace(':', '')}.png"
    try:
        plot_apex_height_time_series(
            start_time=start_time,
            end_time=end_time,
            search_root=search_root,
            show=True,
            save_path=save_path,
        )
    except Exception as exc:  # pragma: no cover - user-facing message
        print(f"Error: {exc}")


if __name__ == "__main__":
    start_time = "2022-06-13T03:23:00"
    end_time = "2022-06-13T03:50:00"
    main(start_time, end_time)
