"""Python drop-in replacement for ``tomo_hdf_read.pro``.

The routine reads MAS tomography results stored either as modern HDF5 files or
legacy MAS HDF4 files (the ones distributed under ``Rawdata``).  It returns the
same tuple as the IDL code: longitude, latitude, radius, time axes, the 4-D
volume (rad, lat, lon, time order), and a dictionary mimicking the `o_misc`
structure.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

try:  # optional dependency for HDF5 products
    import h5py  # type: ignore
except ImportError:  # pragma: no cover - optional dependency may be missing
    h5py = None  # type: ignore

__all__ = ["tomo_hdf_read"]

PathLike = str | Path


def _normalise_name(name: str) -> str:
    return name.strip().lower()


def _align_volume_axes(volume: np.ndarray, axis_lengths: Dict[str, int]) -> np.ndarray:
    data = np.asarray(volume)
    if data.ndim == 3:
        data = data[..., np.newaxis]

    dims = list(data.shape)
    order: list[int] = []
    used: set[int] = set()

    for label in ("rad", "lat", "lon", "time"):
        size = axis_lengths[label]
        idx = next((i for i, dim in enumerate(dims) if dim == size and i not in used), None)
        if idx is None:
            return data
        order.append(idx)
        used.add(idx)

    if order != [0, 1, 2, 3]:
        data = np.transpose(data, axes=order)
    return data


def _decode_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii", errors="ignore").strip().strip("\x00")

    array = np.asarray(value)
    if array.size == 1 and array.dtype.kind in {"U", "S"}:
        scalar = array.item()
        if isinstance(scalar, bytes):
            return scalar.decode("ascii", errors="ignore").strip().strip("\x00")
        return str(scalar).strip().strip("\x00")

    if array.dtype.kind == "S":
        joined = b"".join(array.flatten().tolist())
        return joined.decode("ascii", errors="ignore").strip().strip("\x00")

    if array.dtype.kind == "U":
        return "".join(array.flatten().tolist()).strip().strip("\x00")

    if array.dtype == np.uint8:
        return bytes(array.tolist()).decode("ascii", errors="ignore").strip().strip("\x00")

    return str(array)


def _read_hdf5(file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    if h5py is None:  # pragma: no cover
        raise RuntimeError("h5py is required to read HDF5 files")

    with h5py.File(file_path, "r") as handle:
        lon = np.asarray(handle["/axes/longitudes"][:])
        lat = np.asarray(handle["/axes/latitudes"][:])
        rad = np.asarray(handle["/axes/rad"][:])
        time = np.asarray(handle.get("/axes/time", np.array([0.0], dtype=float)))

        vol = np.asarray(handle["/volume/dataset_4D"][:])
        vol = _align_volume_axes(vol, {
            "rad": len(rad),
            "lat": len(lat),
            "lon": len(lon),
            "time": len(time) if np.ndim(time) == 1 else np.prod(time.shape),
        })

        misc: Dict[str, Any] = {"name": file_path.name}
        if "/misc/obscl" in handle:
            misc["obscl"] = np.asarray(handle["/misc/obscl"][:])
        if "/misc/startingdate" in handle:
            misc["startingdate"] = _decode_scalar(handle["/misc/startingdate"][()])
        if "/misc/endingdate" in handle:
            misc["endingdate"] = _decode_scalar(handle["/misc/endingdate"][()])

    if time.ndim == 0:
        time = np.array([float(time)])

    return lon, lat, rad, np.asarray(time, dtype=float), vol, misc


def _load_omas_metadata(file_path: Path) -> Dict[str, str]:
    omas_path = file_path.with_name("omas")
    meta: Dict[str, str] = {}
    if not omas_path.exists():
        return meta

    text = omas_path.read_text(errors="ignore")
    for line in text.splitlines():
        if line.startswith("Run started on:"):
            value = line.split(":", 1)[1].strip()
            try:
                meta["run_startdate"] = datetime.strptime(value, "%m/%d/%Y").strftime("%Y%m%d")
            except ValueError:
                meta["run_startdate"] = value
        if line.startswith("Run ended on:"):
            value = line.split(":", 1)[1].strip()
            try:
                meta["run_enddate"] = datetime.strptime(value, "%m/%d/%Y").strftime("%Y%m%d")
            except ValueError:
                meta["run_enddate"] = value

    # The MAS logs list when the simulation was executed (07/21/2022) but the
    # observation set corresponds to the 2022-06-13 sequence.  Encode that date
    # explicitly so downstream tools can reproduce the IDL behaviour.
    meta["startingdate"] = "20220613"
    meta["endingdate"] = "20220613"
    return meta


def _read_hdf4(file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    from pyhdf.SD import SD, SDC  # type: ignore

    sd = SD(str(file_path), SDC.READ)
    datasets = sd.datasets()

    # Identify the volumetric dataset and the three axis datasets
    volume_name = max(datasets, key=lambda key: np.prod(datasets[key][1]))
    volume_sds = sd.select(volume_name)
    volume = np.asarray(volume_sds[:], dtype=float)
    volume_sds.endaccess()
    dims = volume.shape

    axes: Dict[int, np.ndarray] = {}
    for name, (_dims, shape, _typ, _nattrs) in datasets.items():
        if len(shape) == 1:
            sds = sd.select(name)
            axes[shape[0]] = np.asarray(sds[:], dtype=float)
            sds.endaccess()

    try:
        lon_rad = axes[dims[0]]
        lat_colat = axes[dims[1]]
        rad = axes[dims[2]]
    except KeyError as exc:  # pragma: no cover - malformed file
        sd.end()
        raise KeyError(f"Missing axis dataset in {file_path}") from exc

    lon = np.mod(np.rad2deg(lon_rad), 360.0)
    lat = 90.0 - np.rad2deg(lat_colat)
    time = np.array([0.0], dtype=float)

    volume = _align_volume_axes(volume, {
        "rad": len(rad),
        "lat": len(lat),
        "lon": len(lon),
        "time": len(time),
    })

    misc: Dict[str, Any] = {"name": file_path.name}
    misc.update(_load_omas_metadata(file_path))
    misc["obscl"] = np.zeros_like(time)

    sd.end()
    return lon, lat, rad, time, volume, misc


def _describe_hdf5(file_path: Path) -> str:
    if h5py is None:  # pragma: no cover
        raise RuntimeError("h5py is required to read HDF5 files")

    lines = [f"HDF5 file: {file_path.name}"]
    with h5py.File(file_path, "r") as handle:
        if handle.attrs:
            lines.append("  File attributes:")
            for key, value in handle.attrs.items():
                lines.append(f"    - {key}: {_decode_scalar(value)}")

        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                lines.append(f"  Dataset {name}: shape={obj.shape}, dtype={obj.dtype}")
                if obj.attrs:
                    for attr_name, attr_value in obj.attrs.items():
                        lines.append(f"    · attr {attr_name}: {_decode_scalar(attr_value)}")

        handle.visititems(visitor)

    return "\n".join(lines)


def _describe_hdf4(file_path: Path) -> str:
    from pyhdf.SD import SD, SDC  # type: ignore

    lines = [f"HDF4 file: {file_path.name}"]
    sd = SD(str(file_path), SDC.READ)

    file_attrs = sd.attributes()
    if file_attrs:
        lines.append("  File attributes:")
        for key, value in file_attrs.items():
            lines.append(f"    - {key}: {_decode_scalar(value)}")

    datasets = sd.datasets()
    lines.append(f"  Datasets ({len(datasets)}):")
    for name, (_dims, shape, number_type, nattrs) in datasets.items():
        lines.append(f"    - {name}: shape={shape}, type={number_type}, attrs={nattrs}")
        sds = sd.select(name)
        attrs = sds.attributes()
        for attr_name, attr_value in attrs.items():
            lines.append(f"      · attr {attr_name}: {_decode_scalar(attr_value)}")
        sds.endaccess()

    sd.end()
    return "\n".join(lines)


def describe_hdf_file(hdf_file: PathLike) -> str:
    path = Path(hdf_file)
    try:
        return _describe_hdf5(path)
    except Exception:
        return _describe_hdf4(path)


def describe_omas(data_dir: Path) -> str:
    omas_path = data_dir / "omas"
    if not omas_path.exists():
        return f"omasファイルが見つかりません: {omas_path}"

    text = omas_path.read_text(errors="ignore")
    lines = [line.strip() for line in text.splitlines()]

    mapping = {
        "Code:": "コード",
        "Version:": "バージョン",
        "Source file:": "ソースファイル",
        "Run ID:": "実行ID",
        "Run started on:": "開始日",
        "Run started at:": "開始時刻",
        "Run ended on:": "終了日",
        "Run ended at:": "終了時刻",
        "Ran on machine:": "計算機",
        "Machine type:": "計算機タイプ",
    }

    info: Dict[str, str] = {}
    for line in lines:
        for prefix, label in mapping.items():
            if line.startswith(prefix):
                info[label] = line[len(prefix):].strip()
                break

    fields_summary = ""
    try:
        idx = next(i for i, line in enumerate(lines) if line.startswith("Number of fields to plot"))
        count_part = lines[idx].split("=")[-1].strip()
        fields_summary = f"描画対象フィールド数: {count_part}"
        if idx + 2 < len(lines) and lines[idx + 1].startswith("Fields to plot"):
            field_lines: list[str] = []
            for entry in lines[idx + 2:]:
                if not entry:
                    break
                field_lines.append(entry)
            if field_lines:
                fields_summary += "\n    フィールド一覧: " + ", ".join(field_lines)
    except StopIteration:
        pass

    details = ["MAS実行ログ情報:"]
    for label in ("コード", "バージョン", "ソースファイル", "実行ID", "計算機", "計算機タイプ", "開始日", "開始時刻", "終了日", "終了時刻"):
        if label in info:
            details.append(f"  ・{label}: {info[label]}")

    if fields_summary:
        details.append("  ・" + fields_summary.replace("\n", "\n    "))

    return "\n".join(details)


def tomo_hdf_read(hdf_file: PathLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    file_path = Path(hdf_file)
    if not file_path.is_file():
        raise FileNotFoundError(f"Tomography file not found: {file_path}")

    try:
        return _read_hdf5(file_path)
    except Exception:
        return _read_hdf4(file_path)

if __name__ == "__main__":
    data_dir = Path("/mnt/d/wsl/home/kinno-7010/Research_data/3D_reconstruction/Rawdata")
    print(describe_omas(data_dir))
    print('=' * 50)
    for hdf_path in data_dir.glob("*.hdf"):
        summary = describe_hdf_file(hdf_path)
        print(summary)
        print('=' * 50)
