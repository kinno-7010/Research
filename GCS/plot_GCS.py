"""
plot_GCS.py
-----------
Driver script that builds the MLSO/K-Cor + SOHO/LASCO-C2 composite via
`integrated_analysis.create_single_diff_image` and overlays a GCS wireframe.

Enhancements:
- Dynamic PYTHONPATH discovery for integrated_analysis / PyThea.
- Automatic tilt estimation via `gcs_overlay.footpoint_fit` using
  - CLI-provided sources (`fit ...` arguments)
  - Footpoint catalogue (JSON) pointed by `$GCS_FOOTPOINT_CONFIG` or
    `footpoint_sources.json` placed next to this script
  - Legacy global switches (`FIT_TILT_FROM_SOURCE`, etc.) as fallback.

The first matching strategy wins; failures gracefully fall back to the user
provided `tilt_deg`.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from astropy.time import Time

CANDIDATE_PATHS = [
    r"D:\\wsl\\home\\kinno-7010\\Research\\SDO_Mk4_SOHO\\py_folder",
    "/mnt/d/wsl/home/kinno-7010/Research_data/SDO_Mk4_SOHO/py_folder",
    r"D:\\wsl\\home\\kinno-7010\\Research\\PyThea\\Kouloumvakos_GitHub",
    "/mnt/d/wsl/home/kinno-7010/Research_data/PyThea/Kouloumvakos_GitHub",
]

for path in CANDIDATE_PATHS:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

from gcs_overlay import (
    GCSParams,
    overlay_gcs_on_composite,
    overlay_gcs_wireframe_on_axes,
    footpoint_fit,
)
from gcs_overlay.gcs_geometry import sample_gcs_wireframe_points  # noqa: E402
from gcs_overlay.footpoint_fit import (  # noqa: E402
    find_best_tilt_for_source,
    find_best_tilt_for_two_sources,
)

import importlib

import matplotlib


def _select_interactive_backend() -> str:
    """Pick an interactive backend (Tk preferred, Qt as fallback)."""
    candidates = []
    try:
        import tkinter  # noqa: F401
    except Exception:
        pass
    else:
        candidates.append("TkAgg")

    try:
        importlib.import_module("PyQt5")
    except Exception:
        pass
    else:
        candidates.append("Qt5Agg")

    for backend in candidates:
        try:
            matplotlib.use(backend)
            return backend
        except Exception:
            continue

    raise RuntimeError(
        "No interactive Matplotlib backend could be activated. Install/configure Tkinter or PyQt5 with a working display."
    )


_ACTIVE_BACKEND = _select_interactive_backend()

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

LASCO_GEOM_CACHE = Path(__file__).resolve().with_name("lasco_geometry_cache.json")


def _normalize_json_value(value):
    import numpy as _np

    if isinstance(value, (int, float, str)) or value is None:
        return value
    if isinstance(value, (_np.integer,)):
        return int(value)
    if isinstance(value, (_np.floating,)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_json_value(v) for k, v in value.items()}
    return value


def _compute_extent_from_params(params: Dict[str, object]) -> List[float]:
    cx = float(params.get('cx', 0.0))
    cy = float(params.get('cy', 0.0))
    nx = float(params.get('nx', 0.0))
    ny = float(params.get('ny', 0.0))
    return [-cx, nx - cx, -cy, ny - cy]


def _cache_lasco_geometry(
    params_lasco: Dict[str, object],
    lasco_map,
    timestamp: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    gcs_params: Dict[str, object],
) -> None:
    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from sunpy.coordinates import frames

        params_copy: Dict[str, object] = {str(k): _normalize_json_value(v) for k, v in params_lasco.items()}
        if 'center_px' not in params_copy:
            cx = float(params_copy.get('cx', 0.0))
            cy = float(params_copy.get('cy', 0.0))
            params_copy['center_px'] = [cx, cy]

        extent = params_copy.get('extent')
        if extent is None:
            extent = _compute_extent_from_params(params_copy)
        params_copy['extent'] = _normalize_json_value(extent)

        observer = lasco_map.observer_coordinate.transform_to(
            frames.HeliographicStonyhurst(obstime=lasco_map.date)
        )

        entry = {
            "timestamp": Time(timestamp).iso,
            "rsun_arcsec": float(lasco_map.rsun_obs.to_value(u.arcsec)),
            "observer": {
                "lon_deg": float(observer.lon.to_value(u.deg)),
                "lat_deg": float(observer.lat.to_value(u.deg)),
                "radius_m": float(observer.radius.to_value(u.m)),
            },
            "params_lasco": params_copy,
            "xlim": [float(xlim[0]), float(xlim[1])],
            "ylim": [float(ylim[0]), float(ylim[1])],
            "gcs_params": _normalize_json_value(gcs_params),
        }

        LASCO_GEOM_CACHE.write_text(json.dumps(entry, indent=2))
    except Exception as exc:
        print(f"[WARN] Failed to cache LASCO geometry: {exc}")


def _load_cached_lasco_geometry() -> Tuple[
    Dict[str, object],
    List[float],
    object,
    Tuple[float, float],
    Tuple[float, float],
    Dict[str, object],
]:
    if not LASCO_GEOM_CACHE.exists():
        raise FileNotFoundError(
            "LASCO geometry cache not found. Run with composite enabled once to populate the cache."
        )

    try:
        entry = json.loads(LASCO_GEOM_CACHE.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse LASCO geometry cache: {exc}") from exc

    params_lasco = entry.get('params_lasco') or {}
    extent = params_lasco.get('extent')
    if not extent:
        extent = _compute_extent_from_params(params_lasco)

    from astropy import units as u
    from astropy.time import Time
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import frames

    class _CachedLascoMap:
        def __init__(self, entry_obj):
            self.date = Time(entry_obj["timestamp"])
            self.rsun_obs = u.Quantity(entry_obj["rsun_arcsec"], u.arcsec)
            obs = entry_obj["observer"]
            self._observer = SkyCoord(
                lon=obs["lon_deg"] * u.deg,
                lat=obs["lat_deg"] * u.deg,
                radius=obs["radius_m"] * u.m,
                frame=frames.HeliographicStonyhurst,
                obstime=self.date,
            )

        @property
        def observer_coordinate(self):
            return self._observer

    xlim_entry = entry.get('xlim') or [extent[0], extent[1]]
    ylim_entry = entry.get('ylim') or [extent[2], extent[3]]
    xlim = (float(xlim_entry[0]), float(xlim_entry[1]))
    ylim = (float(ylim_entry[0]), float(ylim_entry[1]))

    cached_params = entry.get('gcs_params') or {}

    lasco_map = _CachedLascoMap(entry)
    return params_lasco, extent, lasco_map, xlim, ylim, cached_params


def _extract_center(params_lasco: Dict[str, object]) -> Optional[Tuple[float, float]]:
    center = params_lasco.get('center_px')
    if center is None:
        cx = params_lasco.get('cx')
        cy = params_lasco.get('cy')
        if cx is not None and cy is not None:
            center = (cx, cy)

    if isinstance(center, (list, tuple)) and len(center) >= 2:
        try:
            return float(center[0]), float(center[1])
        except Exception:
            return None
    return None


# Legacy globals (still honored for backwards compatibility)

SRC1_LONLAT: Optional[Tuple[float, float]] = None 
SRC2_LONLAT: Optional[Tuple[float, float]] = None
TILT_RANGE = (-180.0, 180.0)
TILT_STEP = 1.0


def _event_key(ts: str) -> str:
    """Normalize timestamp to ISO yyyy-mm-ddTHH:MM:SS string."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return ts
    return dt.replace(microsecond=0, tzinfo=None).isoformat()


def _sanitize_timestamp_tag(ts_str: str) -> str:
    normalized = _event_key(ts_str)
    cleaned = normalized.replace(':', '').replace('-', '').replace('T', '_')
    return ''.join(ch if ch.isalnum() or ch in {'_', '-'} else '-' for ch in cleaned)


def _mk4_time_to_iso(map_time: object) -> Optional[str]:
    """Convert MK4 map time to an ISO yyyy-mm-ddTHH:MM:SS string if possible."""
    if map_time is None:
        return None
    try:
        # Handle astropy Time or convertible inputs first for consistency
        time_obj = Time(map_time)
        dt_obj = time_obj.to_datetime()
        return dt_obj.replace(microsecond=0, tzinfo=None).isoformat()
    except Exception:
        pass

    if hasattr(map_time, "strftime"):
        try:
            return map_time.strftime('%Y-%m-%dT%H:%M:%S')
        except Exception:
            pass

    try:
        return _event_key(str(map_time))
    except Exception:
        return None


def _load_catalogue(path: Path) -> Dict[str, Dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse footpoint catalogue {path}: {exc}") from exc

    if isinstance(data, dict):
        return data

    raise ValueError(f"Unsupported footpoint catalogue format in {path}")


def _resolve_catalog_entry(ts: str) -> Optional[Dict[str, object]]:
    candidates: List[Path] = []

    env_path = os.environ.get("GCS_FOOTPOINT_CONFIG")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.append(Path(__file__).resolve().with_name("footpoint_sources.json"))

    for path in candidates:
        if not path.exists():
            continue
        entry = _load_catalogue(path).get(_event_key(ts))
        if entry is not None:
            entry.setdefault("_catalog_path", str(path))
            return entry
    return None


def _parse_sources(raw_sources: Iterable[object]) -> List[Tuple[float, float]]:
    parsed: List[Tuple[float, float]] = []
    for item in raw_sources:
        if isinstance(item, dict):
            lon = float(item["lon_deg"])
            lat = float(item["lat_deg"])
        elif isinstance(item, Sequence) and len(item) >= 2:
            lon, lat = float(item[0]), float(item[1])
        else:
            raise ValueError("Invalid footpoint entry: expected dict or sequence")
        parsed.append((lon, lat))
    return parsed


def _apply_catalogue_tilt(params: GCSParams, ts: str) -> Tuple[GCSParams, Optional[Dict[str, object]]]:
    entry = _resolve_catalog_entry(ts)
    if entry is None:
        return params, None

    try:
        sources = _parse_sources(entry.get("sources", []))
    except ValueError as exc:
        print(f"[WARN] Footpoint catalogue has invalid entry: {exc}")
        return params, None

    if not sources:
        print("[WARN] Footpoint catalogue entry missing 'sources'")
        return params, None

    tilt_range = entry.get("tilt_search_deg", TILT_RANGE)
    if not isinstance(tilt_range, Sequence) or len(tilt_range) < 2:
        tilt_range = TILT_RANGE
    else:
        tilt_range = (float(tilt_range[0]), float(tilt_range[1]))

    tilt_step = float(entry.get("tilt_step_deg", TILT_STEP))
    n_phi = int(entry.get("n_phi", 360))

    try:
        if len(sources) == 1:
            best = find_best_tilt_for_source(
                params, sources[0][0], sources[0][1],
                tilt_search_deg=tilt_range, tilt_step_deg=tilt_step, n_phi=n_phi,
            )
        else:
            best = find_best_tilt_for_two_sources(
                params, sources[0], sources[1],
                tilt_search_deg=tilt_range, tilt_step_deg=tilt_step, n_phi=n_phi,
            )
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] Auto tilt estimation (catalogue) failed: {exc}")
        return params, None

    new_params = params.__class__(**{**asdict(params), "tilt_deg": best["tilt_deg"]})
    return new_params, {
        "source": "catalogue",
        "catalog_path": entry.get("_catalog_path"),
        "sources": sources,
        "result": best,
    }


def _apply_sources_override(
    params: GCSParams, sources: Sequence[Tuple[float, float]],
    tilt_range: Tuple[float, float], tilt_step: float, n_phi: int = 360,
) -> Tuple[GCSParams, Optional[Dict[str, object]]]:
    if not sources:
        return params, None

    try:
        if len(sources) == 1:
            best = find_best_tilt_for_source(
                params, sources[0][0], sources[0][1],
                tilt_search_deg=tilt_range, tilt_step_deg=tilt_step, n_phi=n_phi,
            )
        else:
            best = find_best_tilt_for_two_sources(
                params, sources[0], sources[1],
                tilt_search_deg=tilt_range, tilt_step_deg=tilt_step, n_phi=n_phi,
            )
    except Exception as exc:
        print(f"[WARN] Footpoint-based tilt fit skipped due to error: {exc}")
        return params, None

    return (
        params.__class__(**{**asdict(params), "tilt_deg": best["tilt_deg"]}),
        {"source": "manual", "result": best},
    )


def main(
    ts: str,
    h_apex: float,
    kappa: float,
    alpha_deg: float,
    tilt_deg: float,
    lon_deg: float,
    lat_deg: float,
    auto_tilt: Optional[bool] = None,
    auto_h_apex: Optional[bool] = None,
    use_composite: bool = True,
):
    params = GCSParams(
        h_apex=h_apex,kappa=kappa,alpha_deg=alpha_deg,
        tilt_deg=tilt_deg,lon_deg=lon_deg,lat_deg=lat_deg
    )

    tilt_meta: Optional[Dict[str, object]] = None

    tilt_mode = "auto" if auto_tilt else "manual"
    cli_mode = getattr(main, "_cli_tilt_mode", None)
    if cli_mode in {"auto", "manual"}:
        tilt_mode = cli_mode
    elif auto_tilt is not None:
        tilt_mode = "auto" if auto_tilt else "manual"

    h_apex_mode = "auto" if auto_h_apex else "manual"
    cli_h_apex_mode = getattr(main, "_cli_h_apex_mode", None)
    if isinstance(cli_h_apex_mode, str) and cli_h_apex_mode in {"auto", "manual"}:
        h_apex_mode = cli_h_apex_mode
    auto_h_apex_enabled = h_apex_mode == "auto"

    # 1) CLI override (fit ...)
    cli_sources: Optional[List[Tuple[float, float]]] = getattr(main, "_cli_sources", None)
    cli_range: Tuple[float, float] = getattr(main, "_cli_tilt_range", TILT_RANGE)
    cli_step: float = getattr(main, "_cli_tilt_step", TILT_STEP)
    main_sources: Optional[List[Tuple[float, float]]] = getattr(main, "_main_sources", None)

    if tilt_mode != "auto":
        pass
    elif cli_sources:
        params, tilt_meta = _apply_sources_override(params, cli_sources, cli_range, cli_step)
    elif main_sources:
        params, tilt_meta = _apply_sources_override(params, main_sources, TILT_RANGE, TILT_STEP)
    elif FIT_TILT_FROM_SOURCE and (SRC1_LONLAT is not None):
        override_sources = [SRC1_LONLAT] if SRC2_LONLAT is None else [SRC1_LONLAT, SRC2_LONLAT]
        params, tilt_meta = _apply_sources_override(params, override_sources, TILT_RANGE, TILT_STEP)
    else:
        params, tilt_meta = _apply_catalogue_tilt(params, ts)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    overlay_style = dict(
        n_parallels=8,
        n_meridians=32,
        color='green',
        color_legs='green',
        lw=1,
        alpha=0.8,
        include_legs=True,
        depth_shade=True,
        alpha_near=0.8,
        alpha_far=0.3,
        alpha_far_legs =0.3,
        leg_depth_from_joint=True,
    )

    res: Dict[str, object]
    base_info: Optional[Dict[str, object]] = None
    lasco_map = None
    params_lasco: Dict[str, object] = {}
    mk4_map = None
    params_for_summary = params

    if use_composite:
        res = overlay_gcs_on_composite(
            ax,
            target_time_str=ts,
            gcs_params=params,
            auto_h_apex=auto_h_apex_enabled,
            h_bounds=(2.5, 5.0),  # h_apexの探索範囲
            pa_band_deg=6.0,  # nose PAの帯域幅
            auto_lonlat=False,
            lon_bounds=(-60.0, 60.0),
            lat_bounds=(-30.0, 30.0),
            coarse_step_deg=5.0,
            refine_step_deg=1.0,
            refine_iters=2,
            **overlay_style,
        )

        base_info = res.get('base') if isinstance(res, dict) else None
        lasco_map = base_info.get('lasco_map') if isinstance(base_info, dict) else None
        params_lasco = base_info.get('params_lasco') if isinstance(base_info, dict) else {}
        mk4_map = base_info.get('mk4_map') if isinstance(base_info, dict) else None

        if isinstance(res, dict) and ('gcs_params' in res):
            rp = res['gcs_params']
            if isinstance(rp, dict):
                params_for_summary = params.__class__(**rp)
            else:
                params_for_summary = rp

        if lasco_map is not None and params_lasco:
            _cache_lasco_geometry(
                params_lasco,
                lasco_map,
                ts,
                ax.get_xlim(),
                ax.get_ylim(),
                asdict(params_for_summary),
            )
    else:
        try:
            params_lasco, extent, lasco_map, xlim, ylim, cached_params = _load_cached_lasco_geometry()
        except Exception as exc:
            raise RuntimeError(
                "Composite disabled but no cached LASCO geometry is available. "
                "Run once with composite enabled to create the cache."
            ) from exc

        ax.cla()
        ax.set_facecolor('white')
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])

        use_cached_params = bool(cached_params) and tilt_mode == "auto"
        if use_cached_params:
            try:
                params = params.__class__(**cached_params)
                params_for_summary = params
            except Exception:
                params_for_summary = params
        else:
            params_for_summary = params

        polylines = overlay_gcs_wireframe_on_axes(
            ax,
            lasco_map,
            params_lasco,
            params_for_summary,
            obstime_str=ts,
            **overlay_style,
        )
        px_per_rsun = params_lasco.get('px_per_rsun')
        try:
            scale_px = float(px_per_rsun)
        except Exception:
            scale_px = None

        if scale_px:
            for radius_rsun in (1.1, 3.0):
                circle = Circle(
                    (0.0, 0.0),
                    scale_px * radius_rsun,
                    fill=False,
                    color='gray',
                    linestyle='--',
                    linewidth=0.8,
                )
                ax.add_patch(circle)

        res = {
            'base': None,
            'polylines': polylines,
            'gcs_params': asdict(params_for_summary),
        }

    ax.get_legend()
    ax.set_aspect('equal')
    ax.set_xlabel("X [pixels] ")
    ax.set_ylabel("Y [pixels] ")

    title_time = ts

    if isinstance(res, dict) and ('gcs_params' in res):
        rp = res['gcs_params']
        if isinstance(rp, dict):
            params_for_summary = params.__class__(**rp)
        else:
            params_for_summary = rp

    mk4_obstime_iso = None
    if mk4_map is not None:
        try:
            mk4_obstime_iso = _mk4_time_to_iso(getattr(mk4_map, 'date', None))
        except Exception:
            mk4_obstime_iso = None

    if mk4_obstime_iso:
        title_time = mk4_obstime_iso

    apex_metrics: Optional[Dict[str, object]] = None

    if lasco_map is not None and isinstance(params_lasco, dict):
        try:
            wireframe = sample_gcs_wireframe_points(
                params_for_summary,
                obstime=Time(ts),
                n_parallels=16,
                n_meridians=32,
                include_legs=True,
            )
            stacks: List[np.ndarray] = []
            for key in ("parallels", "meridians", "legs"):
                curves = wireframe.get(key, []) if isinstance(wireframe, dict) else []
                for curve in curves:
                    if isinstance(curve, np.ndarray) and curve.size:
                        stacks.append(curve)

            if stacks:
                points = np.vstack(stacks)
                radii = np.linalg.norm(points, axis=1)
                if radii.size:
                    idx_max = int(np.argmax(radii))
                    apex_point = points[idx_max]
                    apex_r = float(radii[idx_max])
                    if apex_r > 0:
                        z_norm = np.clip(apex_point[2] / apex_r, -1.0, 1.0)
                    else:
                        z_norm = 0.0
                    phi_rad = float(np.arctan2(apex_point[1], apex_point[0]))
                    theta_rad = float(np.arcsin(z_norm))
                    phi_deg = float(np.degrees(phi_rad))
                    theta_deg = float(np.degrees(theta_rad))

                    from astropy import units as u
                    from astropy.coordinates import SkyCoord
                    from sunpy.coordinates import frames

                    apex_coord = SkyCoord(
                        lon=phi_rad * u.rad,
                        lat=theta_rad * u.rad,
                        radius=apex_r * u.R_sun,
                        frame=frames.HeliographicStonyhurst,
                        obstime=lasco_map.date,
                    )
                    hpc_apex = apex_coord.transform_to(
                        frames.Helioprojective(
                            observer=lasco_map.observer_coordinate,
                            obstime=lasco_map.date,
                        )
                    )

                    rsun_arcsec = lasco_map.rsun_obs.to_value(u.arcsec)
                    px_per_rsun = params_lasco.get('px_per_rsun')
                    if rsun_arcsec and px_per_rsun:
                        x_arcsec = hpc_apex.Tx.to_value(u.arcsec)
                        y_arcsec = hpc_apex.Ty.to_value(u.arcsec)
                        x_px = float((x_arcsec / rsun_arcsec) * px_per_rsun)
                        y_px = float((y_arcsec / rsun_arcsec) * px_per_rsun)
                        apex_metrics = {
                            "time": mk4_obstime_iso or ts,
                            "radius_rsun": apex_r,
                            "phi_deg": phi_deg,
                            "theta_deg": theta_deg,
                            "x_px": x_px,
                            "y_px": y_px,
                        }
                        updated_params = params_for_summary.__class__(
                            **{**asdict(params_for_summary), "h_apex": apex_r}
                        )
                        params_for_summary = updated_params
                                               # --- 表示ラベルを「推定後の lon/lat」に変更 ---
                        apex_label = (
                            f"Apex height (r={apex_r:.3f} $R_\\odot$, "
                            f"lon={params_for_summary.lon_deg:.1f}$^\\circ$, "
                            f"lat={params_for_summary.lat_deg:.1f}$^\\circ$)"
                        )
                        ax.scatter(
                            [x_px], [y_px],
                            facecolor='orange',
                            edgecolors='black',
                            linewidths=0.7,
                            s=70,
                            zorder=7,
                            label=apex_label,
                        )

                        handles, labels = ax.get_legend_handles_labels()
                        if handles and labels:
                            updated_labels: List[str] = []
                            override_label = params_for_summary.legend_label()
                            for lbl in labels:
                                if "$h_{\\mathrm{apex}}$" in lbl:
                                    updated_labels.append(override_label)
                                else:
                                    updated_labels.append(lbl)
                            ax.legend(handles, updated_labels)
                        apex_time_for_print = (mk4_obstime_iso or ts)
                        print(
                            f"Apex height at {apex_time_for_print}: "
                            f"r={apex_r:.3f} Rsun, "
                            f"kappa={params_for_summary.kappa:.2f}, "
                            f"alpha={params_for_summary.alpha_deg:.1f} deg, "
                            f"tilt={params_for_summary.tilt_deg:.1f} deg, "
                            f"lon={params_for_summary.lon_deg:.2f} deg, "
                            f"lat={params_for_summary.lat_deg:.2f} deg"
                        )

        except Exception as exc:
            print(f"[WARN] Apex height computation failed: {exc}")

    if mk4_map is not None and not mk4_obstime_iso:
        try:
            title_time = mk4_map.date.strftime('%Y-%m-%dT%H:%M:%S')
        except Exception:
            title_time = str(mk4_map.date)
    ax.set_title(title_time, fontsize=14)

    output_dir = Path(__file__).resolve().with_name("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"GCS_{title_time.replace(':', '')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")

    if tilt_meta is not None:
        print(f"Tilt auto-fit info: {tilt_meta}")

    plt.tight_layout()
    plt.show()

    return {
        "timestamp": mk4_obstime_iso or ts,
        "input_timestamp": ts,
        "gcs_params": asdict(params_for_summary),
        "apex": apex_metrics,
        "tilt_meta": tilt_meta,
        "figure_path": str(save_path),
    }


def run_gcs_over_times(
    times: Sequence[str],
    h_apex: float,
    kappa: float,
    alpha_deg: float,
    tilt_deg: float,
    lon_deg: float,
    lat_deg: float,
    auto_tilt: Optional[bool] = None,
    auto_h_apex: Optional[bool] = None,
    write_csv: bool = True,
    use_composite: bool = True,
) -> Dict[str, object]:
    """Run the GCS overlay for multiple timestamps and optionally save apex heights.

    Parameters mirror :func:`main`, except that ``times`` is a sequence of timestamps
    (ISO strings). The function returns a bundle containing per-run metadata and the
    apex-height time series.
    """

    normalized_times = [str(ts).strip() for ts in times if str(ts).strip()]
    if not normalized_times:
        return {"results": [], "apex_series": [], "csv_path": None}

    results: List[Dict[str, object]] = []
    apex_series: List[Dict[str, object]] = []

    for ts_entry in normalized_times:
        run_result = main(
            ts_entry,
            h_apex,
            kappa,
            alpha_deg,
            tilt_deg,
            lon_deg,
            lat_deg,
            auto_tilt=auto_tilt,
            auto_h_apex=auto_h_apex,
            use_composite=use_composite,
        )
        if isinstance(run_result, dict):
            results.append(run_result)
            apex_info = run_result.get("apex")
            if isinstance(apex_info, dict):
                radius = apex_info.get("radius_rsun")
                phi_val = apex_info.get("phi_deg")
                theta_val = apex_info.get("theta_deg")
                if radius is not None and phi_val is not None and theta_val is not None:
                    apex_time_str = apex_info.get("time") or ts_entry
                    # 最終パラメータ（自動化後）を取得
                    p = run_result.get("gcs_params") or {}
                    if not isinstance(p, dict):
                        try:
                            from dataclasses import asdict as _asdict
                            p = _asdict(p)
                        except Exception:
                            p = {}
                    apex_series.append(
                        {
                            "time": str(apex_time_str),            # Mk4 の時刻（または ts）
                            "radius_rsun": float(radius),          # 幾何から得た apex 実測値
                            "phi_deg": float(phi_val),
                            "theta_deg": float(theta_val),
                            # ここからパラメータ（最終値）
                            "h_apex": float(p.get("h_apex", radius)),
                            "kappa": float(p.get("kappa")) if p.get("kappa") is not None else None,
                            "alpha_deg": float(p.get("alpha_deg")) if p.get("alpha_deg") is not None else None,
                            "tilt_deg": float(p.get("tilt_deg")) if p.get("tilt_deg") is not None else None,
                            "lon_deg": float(p.get("lon_deg")) if p.get("lon_deg") is not None else None,
                            "lat_deg": float(p.get("lat_deg")) if p.get("lat_deg") is not None else None,
                        })

        else:
            results.append({"timestamp": ts_entry, "result": run_result})

    csv_path: Optional[Path] = None
    if write_csv and apex_series:
        output_dir = Path(__file__).resolve().with_name("output")
        output_dir.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "time", "radius_rsun", "phi_deg", "theta_deg",
            "h_apex", "kappa", "alpha_deg", "tilt_deg", "lon_deg", "lat_deg",
        ]

        csv_paths: List[Path] = []
        for row in apex_series:
            time_tag = _sanitize_timestamp_tag(row["time"])
            candidate = output_dir / f"apex_height_{time_tag}.csv"
            with candidate.open('w', newline='', encoding='utf-8') as fp:
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            csv_paths.append(candidate)
            print(f"Saved apex height series to {candidate}")

        if csv_paths:
            csv_path = csv_paths[-1]

    return {
        "results": results,
        "apex_series": apex_series,
        "csv_path": str(csv_path) if csv_path is not None else None,
    }


def _parse_cli(argv: List[str]):
    global FIT_TILT_FROM_SOURCE, SRC1_LONLAT, SRC2_LONLAT, TILT_RANGE, TILT_STEP

    idx = 8
    if len(argv) > idx:
        mode_candidate = argv[idx].lower()
        if mode_candidate in ("auto", "manual"):
            main._cli_tilt_mode = mode_candidate
            idx += 1

    if len(argv) > idx:
        h_apex_mode_candidate = argv[idx].lower()
        if h_apex_mode_candidate in ("auto", "manual"):
            main._cli_h_apex_mode = h_apex_mode_candidate
            idx += 1

    if len(argv) <= idx:
        return

    if str(argv[idx]).lower() not in ("fit", "1", "true", "yes"):
        return

    FIT_TILT_FROM_SOURCE = True

    cli_sources: List[Tuple[float, float]] = []

    if len(argv) > idx + 2:
        SRC1_LONLAT = (float(argv[idx + 1]), float(argv[idx + 2]))
        cli_sources.append(SRC1_LONLAT)
    if len(argv) > idx + 4:
        SRC2_LONLAT = (float(argv[idx + 3]), float(argv[idx + 4]))
        cli_sources.append(SRC2_LONLAT)

    if len(cli_sources) == 1:
        main._cli_sources = cli_sources
    elif len(cli_sources) >= 2:
        main._cli_sources = cli_sources[:2]

    if len(argv) > idx + 7:
        TILT_RANGE = (float(argv[idx + 5]), float(argv[idx + 6]))
        TILT_STEP = float(argv[idx + 7])
        main._cli_tilt_range = TILT_RANGE
        main._cli_tilt_step = TILT_STEP


if __name__ == "__main__":
    """
    - h_apex: 先端の半径（外側表面までの 3D 高さ）。投影にも効く。
        → 前縁位置の合わせに使う（幅そのものより“どこまで伸びているか”）
    - kappa: 前面トーラスの肉厚（丸み）。見かけの“先端の幅感”を増せる。
        → apex を“太く・広く見せる”には κ を上げる（0.35–0.5 など）
    - alpha_deg: 横方向の広がり（肩の張り）と脚の開きを同時に増減。
        → 足を狭くしたいなら α を小さく。
    - tilt_deg: 軸回りの回転。**脚の方位（見かけの間隔）**を大きく変える。
        → 見かけの footpoint 間隔を圧縮したいときは、脚が視線方向に並ぶ向きへ tilt を回す。
        (+: 反時計回り，-: 時計回り)
    - (lon, lat)：nose の向き → 投影を通じて apex の“見かけの幅”と脚の“見かけの間隔”を同時に変える。
        → apex を広く見せたいなら POS（面内）に寄せる、脚は 視線方向に寄せると投影で狭く見える。
    """
    # h_apex=3.81; kappa=0.12; alpha_deg=23; tilt_deg=-85; lon_deg=-44; lat_deg=10
    h_apex=3.39; kappa=0.10; alpha_deg=22; tilt_deg=87; lon_deg=-44; lat_deg=10

    ts = "2022-06-13T03:25:29"
    # Tilt mode for direct execution (CLI 未指定時に適用)
    # "auto" にすると自動推定を行う
    # "manual" にすると自動推定を行わない
    tilt_mode_at_main = "manual"
    # tilt_mode_at_main = "auto"

    # h_apex 自動推定の ON/OFF（"auto" で有効化、"manual" で無効化）
    h_apex_mode_at_main = "manual"
    # h_apex_mode_at_main = "auto"

    # Composite 表示の ON/OFF を切り替える（"off" で背景を無効化）
    composite_mode_at_main = "on"
    # composite_mode_at_main = "off"

    FIT_TILT_FROM_SOURCE = False  # True にするとグローバル設定でフィット
    # "True" にするとグローバル設定でフィット
    # "False" にするとグローバル設定でフィットしない

    # tilt_mode_at_main を "auto" にした際にここで発生源の緯度経度を指定可能
    """
    source1 … 片側の発生源 (EUV ダイミング/AR/リボン等の片方)の (lon_deg, lat_deg)
    source2 … 反対側の発生源 (もう片方の脚の着地点) (lon_deg, lat_deg)
    1点だけ分かるときは source2 を**空（未指定）**で構いません。
    """
    source1_lat_deg, source1_lon_deg = 21, -44
    source2_lat_deg, source2_lon_deg = None, None

    # -----------------------------------------------------------------------------------
    main_sources_override: List[Tuple[float, float]] = []
    if tilt_mode_at_main == "auto":
        if source1_lon_deg is not None and source1_lat_deg is not None:
            main_sources_override.append((float(source1_lon_deg), float(source1_lat_deg)))
        if source2_lon_deg is not None and source2_lat_deg is not None:
            main_sources_override.append((float(source2_lon_deg), float(source2_lat_deg)))

    if main_sources_override:
        main._main_sources = main_sources_override
        SRC1_LONLAT = main_sources_override[0]
        SRC2_LONLAT = main_sources_override[1] if len(main_sources_override) > 1 else None
        FIT_TILT_FROM_SOURCE = True


    _parse_cli(sys.argv)

    def _normalize_times(raw_items: Iterable[str]) -> List[str]:
        return [item for item in (entry.strip() for entry in raw_items) if item]

    def _extract_times(arg: str) -> List[str]:
        candidate_path: Optional[Path] = None
        if arg.startswith('@'):
            candidate_path = Path(arg[1:]).expanduser()
        else:
            try:
                candidate_path = Path(arg)
            except Exception:
                candidate_path = None
        if candidate_path is not None and candidate_path.exists() and candidate_path.is_file():
            with candidate_path.open('r', encoding='utf-8') as fp:
                return _normalize_times(fp.readlines())
        times = _normalize_times(arg.split(','))
        return times if times else [arg.strip()]

    if len(sys.argv) > 1:
        ts_candidates = _extract_times(sys.argv[1])
    else:
        ts_candidates = [ts]
    if not ts_candidates:
        ts_candidates = [ts]

    h_apex_param = float(sys.argv[2]) if len(sys.argv) > 2 else h_apex
    kappa_param = float(sys.argv[3]) if len(sys.argv) > 3 else kappa
    alpha_deg_param = float(sys.argv[4]) if len(sys.argv) > 4 else alpha_deg
    tilt_deg_param = float(sys.argv[5]) if len(sys.argv) > 5 else tilt_deg
    lon_deg_param = float(sys.argv[6]) if len(sys.argv) > 6 else lon_deg
    lat_deg_param = float(sys.argv[7]) if len(sys.argv) > 7 else lat_deg

    auto_flag = getattr(main, "_cli_tilt_mode", None)
    if auto_flag is None and tilt_mode_at_main in ("auto", "manual"):
        auto_flag = tilt_mode_at_main

    auto_bool = None
    if auto_flag in ("auto", "manual"):
        auto_bool = auto_flag == "auto"

    h_apex_flag = getattr(main, "_cli_h_apex_mode", None)
    if h_apex_flag is None and h_apex_mode_at_main in ("auto", "manual"):
        h_apex_flag = h_apex_mode_at_main

    auto_h_apex_bool = None
    if h_apex_flag in ("auto", "manual"):
        auto_h_apex_bool = h_apex_flag == "auto"

    use_composite_bool = composite_mode_at_main.lower() != "off"

    run_gcs_over_times(
        ts_candidates,
        h_apex_param,
        kappa_param,
        alpha_deg_param,
        tilt_deg_param,
        lon_deg_param,
        lat_deg_param,
        auto_tilt=auto_bool,
        auto_h_apex=auto_h_apex_bool,
        write_csv=True,
        use_composite=use_composite_bool,
    )
