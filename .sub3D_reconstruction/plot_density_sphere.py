"""球面グリッド上の電子密度を3次元で可視化するユーティリティ.

`tomo_hdf_read.py` が提供する HDF 読み込み関数から電子密度データを取得し、
全半径の球面格子を PyVista で 3D 描画する。

描画値は常に ``log10(ne)`` [1/cc] とし、背景色を白に設定する。
PyVista が導入されていない環境では、実行時にインストールを促すメッセージを表示して終了する。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from tomo_hdf_read import tomo_hdf_read

try:
    import pyvista as pv
except Exception as exc:  # pragma: no cover - ランタイム環境に依存
    raise SystemExit(
        "PyVistaが見つかりません。`pip install pyvista` などでインストールしてから再実行してください。\n"
        f"詳細: {exc}"
    )

try:
    from matplotlib import colormaps, colors
except Exception as exc:  # pragma: no cover - matplotlib が無い環境ではそのまま伝える
    raise SystemExit(
        "Matplotlibが見つかりません。`pip install matplotlib` でインストールしてください。\n"
        f"詳細: {exc}"
    )
def _parse_arguments(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="電子密度を球面上に3D表示")
    parser.add_argument(
        "hdf_file",
        nargs="?",
        default=Path(__file__).resolve().parent / "Rawdata" / "rho002.hdf",
        help="読み込むHDFファイル (既定: Rawdata/rho002.hdf)",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=0,
        help="時系列のインデックス (既定: 0)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="指定するとスクリーンショットを保存し、対話表示は行わない",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="使用するカラーマップ名 (matplotlib互換、既定: viridis)",
    )
    return parser.parse_args(args)


def _spherical_surface(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon_rad = np.deg2rad(lon_deg)
    lat_rad = np.deg2rad(lat_deg)
    lon_grid, lat_grid, rad_grid = np.meshgrid(
        lon_rad, lat_rad, rad, indexing="xy"
    )

    cos_lat = np.cos(lat_grid)
    x = rad_grid * cos_lat * np.cos(lon_grid)
    y = rad_grid * cos_lat * np.sin(lon_grid)
    z = rad_grid * np.sin(lat_grid)
    return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)


def _prepare_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    rad: np.ndarray,
    values: np.ndarray,
) -> pv.StructuredGrid:
    x, y, z = _spherical_surface(lon, lat, rad)
    grid = pv.StructuredGrid(x, y, z)
    grid["values"] = values.ravel(order="F")
    return grid


def _lighten_colormap(name: str, factor: float = 0.3) -> colors.ListedColormap:
    base = colormaps.get_cmap(name)
    sample = base(np.linspace(0, 1, 256))
    sample[:, :3] = sample[:, :3] * (1.0 - factor) + factor  # 白に寄せて淡い色へ
    return colors.ListedColormap(sample)


def main(args: Iterable[str] | None = None) -> None:
    options = _parse_arguments(args)
    lon, lat, rad, time, volume, misc = tomo_hdf_read(options.hdf_file)

    time_index = np.clip(options.time_index, 0, volume.shape[-1] - 1)
    density_volume = volume[:, :, :, time_index].astype(np.float32)  # (rad, lat, lon)
    masked_density = np.where(
        (density_volume >= 1e5) & (density_volume <= 1e10), density_volume, np.nan
    )
    with np.errstate(invalid="ignore"):
        log_density = np.log10(masked_density)
    log_density = log_density.astype(np.float32)

    # StructuredGridの形状に合わせて (lon, lat, rad) へ並べ替え
    data_for_grid = np.asfortranarray(np.transpose(log_density, (2, 1, 0)))
    if not np.isfinite(data_for_grid).any():
        raise RuntimeError("指定条件では log10(ne) がすべて NaN です (閾値を下げるなど調整してください)")

    grid = _prepare_grid(lon, lat, rad, data_for_grid)
    grid["log_density"] = grid["values"]
    grid.point_data.pop("values")

    plotter = pv.Plotter()
    plotter.add_mesh(
        grid,
        scalars="log_density",
        cmap=_lighten_colormap(options.cmap, factor=0.35),
        smooth_shading=True,
        show_edges=False,
        opacity=0.85,
        nan_opacity=0.0,
        clim=[5.0, 10.0],
    )
    plotter.add_scalar_bar(title="log10(ne [/cc])")
    plotter.show_axes()
    plotter.add_text(
        f"log10(ne) /cc, 時刻 index {time_index}",
        position="upper_left",
        font_size=10,
    )
    plotter.set_background("white")

    if options.save:
        plotter.show(screenshot=str(options.save), auto_close=True)
    else:
        plotter.show()


if __name__ == "__main__":
    main()
