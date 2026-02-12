#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from astropy.visualization import ImageNormalize, PowerStretch
from astropy.io import fits
import sunpy.map


# ===== ユーザー環境に合わせて調整 =====
BASE_DATA_DIR = Path(r"/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/EUVI/Rawdata")
OUT_DIR = Path(r"/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/EUVI/diff_output")


# ===== データ構造 =====
@dataclass(frozen=True)
class EUVIFile:
    path: Path
    obs_time: datetime
    wavelength: Optional[int]  # filenameから確定できない場合があるのでOptional


# ===== ユーティリティ =====
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def get_euvi_search_dirs(base_dir: Path, wavelength_angstrom: Optional[int]) -> List[Path]:
    """
    EUVI Rawdata の探索対象ディレクトリを返す。

    期待する配置例:
      Rawdata/0195/20220613_014500_195eu_R.fts

    wavelength_angstrom が指定されていれば、Rawdata/<WL4桁> を優先する。
    見つからなければ base_dir 自体も探索対象に含める（保険）。
    """
    dirs: List[Path] = []

    if wavelength_angstrom is not None:
        wl4 = f"{int(wavelength_angstrom):04d}"  # 195 -> "0195"
        d1 = base_dir / wl4
        if d1.exists():
            dirs.append(d1)

        # まれに "195" のような命名の場合もあるので保険
        d2 = base_dir / f"{int(wavelength_angstrom):03d}"
        if d2.exists() and d2 not in dirs:
            dirs.append(d2)

    # 最後に base_dir 自体も探索（フォールバック）
    if base_dir.exists() and base_dir not in dirs:
        dirs.append(base_dir)

    return dirs


def find_nearest_euvi_file(
    files: List[EUVIFile],
    target_utc: datetime,
    max_abs_diff: Optional[timedelta] = None,
) -> EUVIFile:
    """
    files の中から target_utc に最も近い EUVIFile を返す。
    max_abs_diff を指定した場合、それを超えるとエラー。
    """
    if len(files) == 0:
        raise RuntimeError("No EUVI files available to select from.")

    best = min(files, key=lambda f: abs((f.obs_time - target_utc).total_seconds()))
    if max_abs_diff is not None:
        if abs(best.obs_time - target_utc) > max_abs_diff:
            raise RuntimeError(
                "Nearest file is too far from target time.\n"
                f"target={target_utc}, nearest={best.obs_time}, diff={best.obs_time - target_utc}"
            )
    return best

def read_obs_time_from_header(fp: Path) -> Optional[datetime]:
    """
    ファイル名から時刻が取れない場合の保険として、FITS headerの DATE-OBS 等から観測時刻を読む。
    EUVIで一般的な DATE-OBS を優先し、無ければ他候補も試す。
    """
    try:
        hdr = fits.getheader(fp, 0)
    except Exception:
        return None

    for key in ("DATE-OBS", "DATEOBS", "DATE_OBS", "TIME_OBS"):
        if key in hdr:
            val = str(hdr[key]).strip()
            # 典型例: '2022-06-13T03:00:00.008'
            try:
                # 小数秒があってもOK
                return datetime.fromisoformat(val.replace("Z", ""))
            except Exception:
                pass

    return None


def make_target_running_difference(
    target_utc: datetime,
    wavelength_angstrom: int = 195,
    dt_minutes: int = 5,
    search_window_minutes: int = 30,
    max_time_error_seconds: int = 240,
):
    """
    target_utc の画像と、(target_utc - dt_minutes) の画像を選び、
    diff = I(target) - I(target - dt) を1枚だけ作って保存する。

    - search_window_minutes: 対象ファイル探索の余裕（±分）
    - max_time_error_seconds: 最近傍採用時の許容誤差（秒）
      （例：観測の時刻がきっちり 5分刻みでない場合に備える）
    """
    dt = timedelta(minutes=dt_minutes)
    t_prev_req = target_utc - dt
    t_cur_req = target_utc

    # 余裕を持ってファイルを収集（間引きはしない：最近傍選択を正確にする）
    start_utc = t_prev_req - timedelta(minutes=search_window_minutes)
    end_utc = t_cur_req + timedelta(minutes=search_window_minutes)

    files = collect_euvi_files_in_range(
        start_utc=start_utc,
        end_utc=end_utc,
        wavelength_angstrom=wavelength_angstrom,
        step_minutes=None,
    )
    if len(files) < 2:
        raise RuntimeError(f"Not enough files around target time. Found {len(files)} files.")

    tol = timedelta(seconds=max_time_error_seconds)
    prev_file = find_nearest_euvi_file(files, t_prev_req, max_abs_diff=tol)
    cur_file = find_nearest_euvi_file(files, t_cur_req, max_abs_diff=tol)

    if prev_file.path == cur_file.path:
        raise RuntimeError(
            "Selected the same file for target and target-dt. "
            "Increase search_window_minutes or relax max_time_error_seconds."
        )

    prev_map = load_map(prev_file.path)
    cur_map = load_map(cur_file.path)

    if cur_map.data.shape != prev_map.data.shape:
        raise RuntimeError(
            f"Shape mismatch: prev={prev_map.data.shape} vs cur={cur_map.data.shape}\n"
            f"prev={prev_file.obs_time}, cur={cur_file.obs_time}"
        )

    diff = cur_map.data - prev_map.data

    # 出力先
    out_base = ensure_dir(OUT_DIR / f"EUVI_{wavelength_angstrom:03d}" / "target_running_diff")
    out_png = out_base / f"diff_{safe_timestr(prev_file.obs_time)}_{safe_timestr(cur_file.obs_time)}.png"

    title = (
        f"STEREO-A EUVI {wavelength_angstrom} Å  target running-diff (dt={dt_minutes} min)\n"
        f"target={target_utc.strftime('%Y-%m-%d %H:%M:%S')}  "
        f"used: {prev_file.obs_time.strftime('%H:%M:%S')} -> {cur_file.obs_time.strftime('%H:%M:%S')}"
    )

    # WCSは通常同一なので cur_map を基準にプロット
    plot_and_save_diff(cur_map, diff, title, out_png)

    print("[INFO] requested times:")
    print(f"  prev_req = {t_prev_req}  cur_req = {t_cur_req}")
    print("[INFO] selected files:")
    print(f"  prev_use = {prev_file.obs_time}  ({prev_file.path.name})")
    print(f"  cur_use  = {cur_file.obs_time}  ({cur_file.path.name})")
    print(f"[DONE] saved image: {out_png}")


def parse_dt(s: str) -> datetime:
    """
    "YYYY-MM-DD HH:MM" もしくは "YYYY-MM-DD HH:MM:SS" を想定（UTCとして扱う）。
    """
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"Invalid datetime string: {s}. Use 'YYYY-MM-DD HH:MM' (or with :SS).")


def safe_timestr(t: datetime) -> str:
    return t.strftime("%Y%m%d_%H%M%S")


def infer_time_and_wavelength_from_filename(fp: Path) -> Optional[Tuple[datetime, Optional[int]]]:
    """
    EUVIのFITS名は環境や取得元(VSO等)で揺れるので、re.matchではなくre.searchで柔軟に拾う。
    時刻が取れない場合は header fallback を使う。

    例として拾いたい形:
      - 20220613_020000_195eu_R.fts
      - 20220613_010000_n4euA.fts / .fits / .fts.gz / .fits.gz
      - prefix付き: euvi_20220613_010000_n4euA.fts など
    """
    name = fp.name

    # まずファイル名から YYYYMMDD_HHMMSS を探す（どこにあってもよい）
    m = re.search(r"(\d{8})_(\d{6})", name)
    obs_time: Optional[datetime] = None
    if m:
        dstr, tstr = m.group(1), m.group(2)
        try:
            obs_time = datetime.strptime(f"{dstr}_{tstr}", "%Y%m%d_%H%M%S")
        except Exception:
            obs_time = None

    # 波長がファイル名に含まれる場合（_195eu など）
    wl: Optional[int] = None
    m_wl = re.search(r"_(\d{3})eu", name)
    if m_wl:
        try:
            wl = int(m_wl.group(1))
        except Exception:
            wl = None

    # ファイル名で時刻が取れない場合は header fallback
    if obs_time is None:
        obs_time = read_obs_time_from_header(fp)

    if obs_time is None:
        return None

    return obs_time, wl


def read_wavelength_from_header(fp: Path) -> Optional[int]:
    """
    EUVIファイル名から波長が確定できない場合に、FITS headerから推定する。
    よくあるキーを順に試す。
    """
    try:
        hdr = fits.getheader(fp, 0)
    except Exception:
        return None

    for key in ("WAVELNTH", "WAVELN", "WAVE_LEN", "WAVELENGTH"):
        if key in hdr:
            try:
                return int(round(float(hdr[key])))
            except Exception:
                pass

    # instrumentにより FILTER が入る場合があるが、ここでは無理に解釈しない
    return None


def collect_euvi_files_in_range(
    start_utc: datetime,
    end_utc: datetime,
    wavelength_angstrom: Optional[int] = 195,
    step_minutes: Optional[int] = None,
) -> List[EUVIFile]:
    """
    Rawdataから、指定時間範囲に入るEUVIファイルを列挙（時刻順）。

    修正点:
    - Rawdata直下ではなく Rawdata/<WL4桁>（例: 0195）配下を優先して探索する
    - サブディレクトリも含めて再帰探索する（rglob）
    """
    if not BASE_DATA_DIR.exists():
        raise FileNotFoundError(f"Rawdata directory not found: {BASE_DATA_DIR}")

    search_dirs = get_euvi_search_dirs(BASE_DATA_DIR, wavelength_angstrom)

    def is_fits_like(p: Path) -> bool:
        if not p.is_file():
            return False
        suf = [s.lower() for s in p.suffixes]  # 例: ['.fts', '.gz']
        if len(suf) == 0:
            return False
        # 末尾が .fts / .fits
        if suf[-1] in (".fts", ".fits"):
            return True
        # 末尾が .gz で、その1つ前が .fts / .fits
        if suf[-1] == ".gz" and len(suf) >= 2 and suf[-2] in (".fts", ".fits"):
            return True
        return False

    files: List[EUVIFile] = []
    n_scanned = 0

    for root in search_dirs:
        # root直下だけでなく、配下を再帰的に探索
        for fp in root.rglob("*"):
            if not is_fits_like(fp):
                continue
            n_scanned += 1

            parsed = infer_time_and_wavelength_from_filename(fp)
            if parsed is None:
                continue

            obs_time, wl = parsed

            if not (start_utc <= obs_time <= end_utc):
                continue

            # 波長がfilenameで取れない場合はheaderから読む
            if wl is None:
                wl = read_wavelength_from_header(fp)

            # 波長フィルタ
            if wavelength_angstrom is not None:
                if wl is None or int(wl) != int(wavelength_angstrom):
                    continue

            files.append(EUVIFile(path=fp, obs_time=obs_time, wavelength=wl))

    if len(files) == 0:
        # デバッグしやすいように探索場所を明示
        scanned_dirs = ", ".join(str(d) for d in search_dirs)
        raise RuntimeError(
            "No EUVI files found in the requested time range.\n"
            f"searched_dirs=[{scanned_dirs}]\n"
            f"scanned_candidates={n_scanned}\n"
            f"time_range={start_utc}..{end_utc}, wavelength={wavelength_angstrom}"
        )

    files.sort(key=lambda x: x.obs_time)

    # 間引き（step_minutes）
    if step_minutes is not None and step_minutes > 0 and len(files) > 0:
        kept: List[EUVIFile] = [files[0]]
        last = files[0].obs_time
        step = timedelta(minutes=step_minutes)
        for f in files[1:]:
            if (f.obs_time - last) >= step:
                kept.append(f)
                last = f.obs_time
        files = kept

    return files


def load_map(fp: Path) -> sunpy.map.Map:
    m = sunpy.map.Map(fp)
    # 差分計算のためfloat化（符号付きにする）
    m = sunpy.map.Map(m.data.astype(np.float32), m.meta)
    return m


def _robust_sym_vmax(diff: np.ndarray, q: float = 99.0) -> float:
    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        return 1.0
    vmax = np.nanpercentile(np.abs(finite), q)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = np.nanmax(np.abs(finite))
    return float(vmax) if vmax > 0 else 1.0


def plot_and_save_diff(
    base_map: sunpy.map.Map,
    diff_data: np.ndarray,
    title: str,
    out_png: Path,
    cmap: str = "gray",
    grid: bool = True,
    limb: bool = True,
):
    ensure_dir(out_png.parent)

    vmax = _robust_sym_vmax(diff_data, q=99.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection=base_map.wcs)

    im = ax.imshow(diff_data, origin="lower", cmap=cmap, norm=norm)

    if limb:
        try:
            base_map.draw_limb(axes=ax, color="red", linestyle="dashed", linewidth=1.0)
        except Exception:
            pass
    if grid:
        try:
            base_map.draw_grid(
                axes=ax,
                grid_spacing=15 * u.deg,
                color="red",
                linestyle="dotted",
                linewidth=0.8,
                alpha=0.7,
            )
        except Exception:
            pass

    ax.set_title(title, fontsize=11, pad=12)
    ax.set_xlim(-800, 100)
    ax.set_ylim(-300, 800)
    ax.coords[0].set_axislabel("Solar X (arcsec)")
    ax.coords[1].set_axislabel("Solar Y (arcsec)")

    cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.04)
    # cbar.set_label("Difference (DN)", rotation=270, labelpad=14)

    plt.tight_layout()
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)


def make_difference_images(
    start_utc: datetime,
    end_utc: datetime,
    wavelength_angstrom: int = 195,
    mode: str = "running",              # "running" or "base"
    base_time_utc: Optional[datetime] = None,  # mode="base" で任意指定
    sample_minutes: Optional[int] = 5,
):
    """
    指定時間範囲の差分画像を作る。

    mode="running":
      I(t_i) - I(t_{i-1})
    mode="base":
      I(t_i) - I(t_base)
      base_time_utc を指定しない場合は範囲内の最初の画像を基準にする。
    """
    files = collect_euvi_files_in_range(
        start_utc=start_utc,
        end_utc=end_utc,
        wavelength_angstrom=wavelength_angstrom,
        step_minutes=sample_minutes,
    )

    if len(files) < 2:
        raise RuntimeError(f"Not enough files in range. Found {len(files)} files.")

    # 出力先ディレクトリ
    out_base = ensure_dir(OUT_DIR / f"EUVI_{wavelength_angstrom:03d}" / f"{mode}_diff")

    # 基準画像
    if mode == "base":
        if base_time_utc is None:
            base_file = files[0]
        else:
            # base_timeに最も近いファイルを選ぶ
            base_file = min(files, key=lambda f: abs((f.obs_time - base_time_utc).total_seconds()))
        base_map = load_map(base_file.path)
        base_label = safe_timestr(base_file.obs_time)
        print(f"[INFO] base file: {base_file.path.name}  ({base_file.obs_time})")
    else:
        base_map = None
        base_label = ""

    prev_map = None
    prev_time = None

    for i, f in enumerate(files):
        cur_map = load_map(f.path)

        if mode == "running":
            if prev_map is None:
                prev_map = cur_map
                prev_time = f.obs_time
                continue

            # 形状が違う場合は差分不能なので明示的に止める（必要ならreprojectを実装してください）
            if cur_map.data.shape != prev_map.data.shape:
                raise RuntimeError(
                    f"Shape mismatch: {prev_map.data.shape} vs {cur_map.data.shape}\n"
                    f"prev={prev_map.date}, cur={cur_map.date}"
                )

            diff = cur_map.data - prev_map.data
            title = (
                f"STEREO-A EUVI {wavelength_angstrom} Å  running-diff\n"
                f"{prev_time.strftime('%Y-%m-%d %H:%M:%S')} -> {f.obs_time.strftime('%H:%M:%S')}"
            )
            out_png = out_base / f"diff_{safe_timestr(prev_time)}_{safe_timestr(f.obs_time)}.png"
            plot_and_save_diff(prev_map, diff, title, out_png)

            prev_map = cur_map
            prev_time = f.obs_time

        elif mode == "base":
            if cur_map.data.shape != base_map.data.shape:
                raise RuntimeError(
                    f"Shape mismatch: base={base_map.data.shape} vs cur={cur_map.data.shape}\n"
                    f"base={base_map.date}, cur={cur_map.date}"
                )

            diff = cur_map.data - base_map.data
            title = (
                f"STEREO-A EUVI {wavelength_angstrom} Å  base-diff (base={base_label})\n"
                f"{f.obs_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            out_png = out_base / f"diff_base{base_label}_{safe_timestr(f.obs_time)}.png"
            plot_and_save_diff(base_map, diff, title, out_png)

        else:
            raise ValueError("mode must be 'running' or 'base'.")

    print(f"[DONE] saved image: {out_png}")


def main():
    # ===== ここを調整してください（UTC）=====
    # target_time = parse_dt("2022-06-13 03:00")  # 例：この時刻の差分画像を作る
    # start_time, end_time, delta_minを調整してください
    start_time = parse_dt("2022-06-13 01:00")
    end_time = parse_dt("2022-06-13 04:00")
    delta_min = 5

    time_list = []
    t = start_time
    while t <= end_time:
        time_list.append(t)
        t += timedelta(minutes=delta_min)

    wl = 195

    ensure_dir(OUT_DIR)

    # 各時刻ごとに差分画像を作成
    for target_time in time_list:
        print(f"[INFO] target_time = {target_time}")
        make_target_running_difference(
            target_utc=target_time,
            wavelength_angstrom=wl,
            dt_minutes=delta_min,
            search_window_minutes=30,
            max_time_error_seconds=240,  # 必要に応じて調整
        )
    wl = 195

    ensure_dir(OUT_DIR)

    # target_time と 5分前の差分（I(target) - I(target-5min)）を1枚作る
    make_target_running_difference(
        target_utc=target_time,
        wavelength_angstrom=wl,
        dt_minutes=5,
        search_window_minutes=30,
        max_time_error_seconds=240,  # 必要に応じて調整（例：300=±5分）
    )


if __name__ == "__main__":
    main()
