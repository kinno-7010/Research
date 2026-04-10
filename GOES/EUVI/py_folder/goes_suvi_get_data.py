"""
GOES (XRS / optional SUVI) と STEREO/SECCHI EUVI のデータを取得するスクリプト。

依存:
  pip install sunpy astropy parfive pandas

参考:
  - SunPy: Fido / XRS / SUVI / VSO の取得方法
    https://docs.sunpy.org/ (Fido, XRS example, SUVIClient docs)
"""

from __future__ import annotations

import datetime as dt
import os
import pathlib
from dataclasses import dataclass
from typing import Optional

import astropy.units as u
from sunpy.net import Fido, attrs as a
from sunpy import timeseries as ts


# =========================
# ユーザー設定（必要に応じて変更）
# =========================

@dataclass
class TimeWindow:
    center_utc: dt.datetime
    minutes_before: int
    minutes_after: int

    @property
    def start(self) -> dt.datetime:
        return self.center_utc - dt.timedelta(minutes=self.minutes_before)

    @property
    def end(self) -> dt.datetime:
        return self.center_utc + dt.timedelta(minutes=self.minutes_after)


def parse_utc(timestr: str) -> dt.datetime:
    """
    'YYYY-MM-DDTHH:MM:SS' を UTC datetime に変換
    """
    t = dt.datetime.strptime(timestr, "%Y-%m-%dT%H:%M:%S")
    return t.replace(tzinfo=dt.timezone.utc)


def ensure_dir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    if not os.access(str(path), os.W_OK):
        raise PermissionError(f"書き込み権限がありません: {path}")
    return path


def fido_fetch_to(
    result,
    base_dir: pathlib.Path,
    path_template: str,
    max_conn: int = 1,
):
    """
    Fido.fetch() を指定ディレクトリへ保存。
    path_template には Fido の {file} や {instrument} などを使える。
    """
    ensure_dir(base_dir)
    save_pattern = str(base_dir / path_template)
    downloaded = Fido.fetch(result, path=save_pattern, max_conn=max_conn)
    return downloaded


# =========================
# GOES XRS (軟X線時系列) 取得
# =========================

def download_goes_xrs(
    tw: TimeWindow,
    out_dir: pathlib.Path,
    satellite_number: Optional[int] = None,
    to_csv: bool = True,
) -> list[pathlib.Path]:
    """
    GOES XRS を取得して保存。
    - satellite_number を指定しない場合: 指定時間に利用可能な GOES XRS を全て取得
    - to_csv=True の場合: 取得したファイルを読み込み、CSV も出力（1ファイルにマージ）
    """
    print("=== GOES XRS download ===")
    print(f"Time (UTC): {tw.start.isoformat()} -> {tw.end.isoformat()}")

    query = [a.Time(tw.start, tw.end), a.Instrument("XRS")]
    if satellite_number is not None:
        # XRSClient でも SatelliteNumber 列が付与されるためフィルタ可能なことが多い
        query.append(a.goes.SatelliteNumber(satellite_number))

    result = Fido.search(*query)
    print(result)

    downloaded = fido_fetch_to(
        result,
        out_dir,
        path_template="GOES/XRS/{start_time:%Y%m%d}/{file}",
        max_conn=1,
    )
    files = [pathlib.Path(p) for p in downloaded]
    print(f"Downloaded {len(files)} files.")

    if to_csv and files:
        # 複数衛星・複数ファイルでも TimeSeries 側でまとめて読めることが多い
        try:
            goes_ts = ts.TimeSeries(files)
            df = goes_ts.to_dataframe()
            csv_path = ensure_dir(out_dir / "GOES" / "XRS") / f"goes_xrs_{tw.start:%Y%m%dT%H%M%S}_{tw.end:%Y%m%dT%H%M%S}.csv"
            df.to_csv(csv_path)
            print(f"Saved CSV: {csv_path}")
        except Exception as e:
            print(f"[WARN] TimeSeries->CSV 変換に失敗しました（FITS/NetCDFの形式差等の可能性）: {e}")

    return files




# =========================
# 任意: GOES SUVI (EUV画像) 取得（SunPy SUVIClient）
# =========================

def download_goes_suvi(
    tw: TimeWindow,
    out_dir: pathlib.Path,
    satellite_number: int = 16,        # GOES-16/17/18...
    wavelength_angstrom: int = 195,    # 94, 131, 171, 195, 284, 304
    level: int = 1,                    # 1 or 2 (SunPy attrsは a.Level.one / two)
    sample_minutes: Optional[int] = None,
) -> list[pathlib.Path]:
    """
    GOES SUVI を NOAA アーカイブから取得（SunPy SUVIClient）。
    """
    print("=== GOES SUVI download ===")
    print(f"Time (UTC): {tw.start.isoformat()} -> {tw.end.isoformat()}")
    print(f"GOES-{satellite_number} SUVI {wavelength_angstrom} Å Level-{level}")

    lvl_attr = a.Level.one if level == 1 else a.Level.two

    query = [
        a.Time(tw.start, tw.end),
        a.Instrument("suvi"),
        lvl_attr,
        a.goes.SatelliteNumber(satellite_number),
        a.Wavelength(wavelength_angstrom * u.angstrom),
    ]
    if sample_minutes is not None:
        query.append(a.Sample(sample_minutes * u.min))

    result = Fido.search(*query)
    print(result)

    downloaded = fido_fetch_to(
        result,
        out_dir,
        path_template=f"GOES/SUVI/G{satellite_number:02d}/L{level}/{wavelength_angstrom:04d}/{{start_time:%Y%m%d}}/{{file}}",
        max_conn=1,
    )
    files = [pathlib.Path(p) for p in downloaded]
    print(f"Downloaded {len(files)} files.")
    return files


# =========================
# 実行例
# =========================

def main():
    # 例: あなたの AIA synoptic 取得コードと同じ中心時刻・時間窓
    target_time_str = "2022-06-13T01:00:00"
    tw = TimeWindow(
        center_utc=parse_utc(target_time_str),
        minutes_before=0,
        minutes_after=180,
    )

    # 出力先（Windows パスでもOK）
    out_dir = ensure_dir(pathlib.Path(r"/mnt/d/wsl/home/kinno-7010/Research_data/GOES_EUVI"))

    # # GOES XRS（軟X線時系列）
    # download_goes_xrs(
    #     tw=tw,
    #     out_dir=out_dir,
    #     satellite_number=None,  # 例: 16 を指定すると GOES-16 を優先
    #     to_csv=True,
    # )

    # # STEREO-A EUVI（例: 195Å、5分間引き）
    # download_stereo_euvi(
    #     tw=tw,
    #     out_dir=out_dir,
    #     spacecraft="A",
    #     wavelength_angstrom=195,
    #     sample_minutes=5,
    # )

    # 任意: GOES-16 SUVI（例: 195Å, Level-1）
    # 195 (10min), 171(2.5min), 284(20min), 304(10min)
    download_goes_suvi(
        tw=tw,
        out_dir=out_dir,
        satellite_number=15,
        wavelength_angstrom=195,
        level=2,
        sample_minutes=None,
    )


if __name__ == "__main__":
    main()
