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

import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

CANDIDATE_PATHS = [
    r"D:\\wsl\\home\\kinno-7010\\Research\\SDO_Mk4_SOHO\\py_folder",
    "/mnt/d/wsl/home/kinno-7010/Research/SDO_Mk4_SOHO/py_folder",
    r"D:\\wsl\\home\\kinno-7010\\Research\\PyThea\\Kouloumvakos_GitHub",
    "/mnt/d/wsl/home/kinno-7010/Research/PyThea/Kouloumvakos_GitHub",
]

for path in CANDIDATE_PATHS:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

from gcs_overlay import GCSParams, overlay_gcs_on_composite  # noqa: E402
from gcs_overlay.footpoint_fit import (  # noqa: E402
    find_best_tilt_for_source,
    find_best_tilt_for_two_sources,
)

import matplotlib.pyplot as plt  # noqa: E402


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


def main(ts: str, h_apex: float, kappa: float, alpha_deg: float, tilt_deg: float, lon_deg: float, lat_deg: float, auto_tilt: Optional[bool] = None):
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

    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    res = overlay_gcs_on_composite(
        ax, target_time_str=ts, gcs_params=params,
        n_parallels=8, n_meridians=32,
        color='green', color_legs='green', lw=1, alpha=0.8,
        include_legs=True,
        depth_shade=True,
        alpha_near=1.0,
        alpha_far=0.3,
        alpha_far_legs=0.3,
        leg_depth_from_joint=True,
    )
    ax.get_legend(); ax.set_aspect('equal')
    ax.set_xlabel("X [pixels] "); ax.set_ylabel("Y [pixels] ")

    title_time = ts
    base_info = res.get('base') if isinstance(res, dict) else None
    mk4_map = base_info.get('mk4_map') if isinstance(base_info, dict) else None
    if mk4_map is not None:
        try:
            title_time = mk4_map.date.strftime('%Y-%m-%dT%H:%M:%S')
        except Exception:
            title_time = str(mk4_map.date)
    ax.set_title(title_time)

    output_dir = Path(__file__).resolve().with_name("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"GCS_{title_time.replace(':', '')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")

    if tilt_meta is not None:
        print(f"Tilt auto-fit info: {tilt_meta}")

    plt.tight_layout()
    plt.show()


def _parse_cli(argv: List[str]):
    global FIT_TILT_FROM_SOURCE, SRC1_LONLAT, SRC2_LONLAT, TILT_RANGE, TILT_STEP

    idx = 8
    if len(argv) > idx:
        mode_candidate = argv[idx].lower()
        if mode_candidate in ("auto", "manual"):
            main._cli_tilt_mode = mode_candidate
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
    h_apex=3.05; kappa=0.35; alpha_deg=5; tilt_deg=-10; lon_deg=-40; lat_deg=10


    ts = "2022-06-13T03:20:00"
    # Tilt mode for direct execution (CLI 未指定時に適用)
    # "auto" にすると自動推定を行う
    # "manual" にすると自動推定を行わない
    # tilt_mode_at_main = "manual"
    tilt_mode_at_main = "auto"

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

    h_apex_param = float(sys.argv[2]) if len(sys.argv) > 2 else h_apex
    kappa_param = float(sys.argv[3]) if len(sys.argv) > 3 else kappa
    alpha_deg_param = float(sys.argv[4]) if len(sys.argv) > 4 else alpha_deg
    tilt_deg_param = float(sys.argv[5]) if len(sys.argv) > 5 else tilt_deg
    lon_deg_param = float(sys.argv[6]) if len(sys.argv) > 6 else lon_deg
    lat_deg_param = float(sys.argv[7]) if len(sys.argv) > 7 else lat_deg
    ts_param = sys.argv[1] if len(sys.argv) > 1 else ts

    auto_flag = getattr(main, "_cli_tilt_mode", None)
    if auto_flag is None and tilt_mode_at_main in ("auto", "manual"):
        auto_flag = tilt_mode_at_main

    auto_bool = None
    if auto_flag in ("auto", "manual"):
        auto_bool = auto_flag == "auto"

    main(
        ts_param, h_apex_param, kappa_param, alpha_deg_param, tilt_deg_param,
        lon_deg_param, lat_deg_param, auto_tilt=auto_bool,
    )
