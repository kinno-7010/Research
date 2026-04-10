
from __future__ import annotations

import datetime as dt
import os
import pathlib
from dataclasses import dataclass
from typing import Optional

import astropy.units as u
from sunpy.net import Fido, attrs as a
from sunpy import timeseries as ts
import re
import gzip
import shutil
import urllib.request
import urllib.error


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

def build_jhuapl_euvi_r_url(obs_time_utc: dt.datetime, wavelength_angstrom: int, spacecraft: str = "A") -> str:
    """
    JHUAPLのEUVI R版URLを構築する。

    URL例:
    https://solar.jhuapl.edu/secchi/wavelets/fits/202206/13/171_A/20220613_020930_171eu_R.fts.gz
    """
    t = obs_time_utc.astimezone(dt.timezone.utc)
    yyyymm = t.strftime("%Y%m")
    dd = t.strftime("%d")
    yyyymmdd = t.strftime("%Y%m%d")
    hhmmss = t.strftime("%H%M%S")
    sc = spacecraft.upper()
    wl3 = f"{int(wavelength_angstrom):03d}"
    return f"https://solar.jhuapl.edu/secchi/wavelets/fits/{yyyymm}/{dd}/{wl3}_{sc}/{yyyymmdd}_{hhmmss}_{wl3}eu_R.fts.gz"


def download_url(url: str, out_path: pathlib.Path, timeout_sec: int = 60) -> bool:
    """
    url を out_path に保存する。
    - 成功: True
    - 404等で存在しない: False
    - その他エラー: 例外
    """
    ensure_dir(out_path.parent)

    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as r, open(out_path, "wb") as f:
            shutil.copyfileobj(r, f)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise
    except urllib.error.URLError:
        # ネットワーク断などは「存在しない」とは別なので例外扱い
        raise


def gunzip_file(gz_path: pathlib.Path, out_path: pathlib.Path, remove_gz: bool = True) -> None:
    """
    .gz を解凍して out_path（通常 .fts）を作る。
    """
    ensure_dir(out_path.parent)
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    if remove_gz:
        try:
            gz_path.unlink()
        except Exception:
            pass


def select_euvi_r_records(vso_table, wavelength_angstrom: int):
    """
    VSOQueryResponseTable から EUVI の R 版（..._###eu_R.fts）に該当する行だけを抽出して返す。
    fileid 列があればそれを優先して判定し、無ければ文字列化した行全体から判定する。
    """
    # R版の典型的なファイル名: 20220613_020000_195eu_R.fts
    pat = re.compile(rf"_{wavelength_angstrom:03d}eu_R\.fts(\.gz)?$", re.IGNORECASE)

    colnames = list(getattr(vso_table, "colnames", []))

    # まずは fileid を最優先で見る（VSOでは表示上hiddenでも列として存在することが多い）
    fileid_col = None
    for c in colnames:
        if c.lower() == "fileid":
            fileid_col = c
            break

    if fileid_col is not None:
        ids = [str(x) for x in vso_table[fileid_col]]
        mask = [bool(pat.search(s)) for s in ids]
        return vso_table[mask]

    # fileid が無い/取れない場合のフォールバック：行全体を文字列化して末尾マッチで判定
    mask = []
    for i in range(len(vso_table)):
        row_str = str(vso_table[i])
        mask.append(bool(pat.search(row_str)))
    return vso_table[mask]


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
    path_template には {file} などを使えるが、SunPy/Parfiveの組み合わせによっては
    {start_time:%Y%m%d} のような datetime 書式が Astropy Time で失敗することがある。
    その場合は {file} のみに切り替えてリトライする。
    """
    ensure_dir(base_dir)

    save_pattern = str(base_dir / path_template)

    try:
        downloaded = Fido.fetch(result, path=save_pattern, max_conn=max_conn)
        return downloaded
    except Exception as e:
        msg = str(e)
        if "Time.__format__" in msg or "unsupported format string" in msg:
            print("[WARN] path_template に datetime 書式が含まれており、Astropy Time のフォーマットで失敗しました。")
            print("[WARN] {file} のみの保存パターンでリトライします。")
            fallback_pattern = str(base_dir / "{file}")
            downloaded = Fido.fetch(result, path=fallback_pattern, max_conn=max_conn)
            return downloaded
        raise

# =========================
# STEREO/SECCHI EUVI (EUV画像) 取得（VSO 経由）
# =========================

def download_stereo_euvi(
    tw: TimeWindow,
    out_dir: pathlib.Path,
    spacecraft: str = "A",
    wavelength_angstrom: int = 195,
    sample_minutes: Optional[float] = 5,
    require_r: bool = True,  # 互換のため残す（本実装は常にR版のみ）
    remove_gz: bool = True,
    try_time_offsets_sec: tuple[int, ...] = (0, -30, 30, -60, 60, -90, 90, -120, 120),
) -> list[pathlib.Path]:
    """
    JHUAPLの固定URLから EUVI R版（...eu_R.fts.gz）を指定範囲でダウンロードし、解凍して .fts を作る。

    - sample_minutes: 生成する時刻列のステップ（分）。例: 5 → 5分ごとにURLを作る。
      ※ JHUAPL側の実際の時刻が 02:09:30 のように 30秒ずれるケースがあるため、
         try_time_offsets_sec により近傍候補も試す。
    - try_time_offsets_sec: 各時刻 t に対して t+offset のURLも順番に試す（最初に見つかったものを採用）。
    - remove_gz: 解凍後に .gz を消すかどうか
    """
    print("=== STEREO EUVI download (direct JHUAPL R files) ===")
    print(f"Time (UTC): {tw.start.isoformat()} -> {tw.end.isoformat()}")
    print(f"SC: {spacecraft}, Wavelength: {wavelength_angstrom} Å, Step: {sample_minutes} min")
    print("Mode: download *.fts.gz (R) then gunzip to *.fts")

    if sample_minutes is None:
        raise ValueError("sample_minutes must be provided for direct-URL download.")

    step_sec = int(round(float(sample_minutes) * 60.0))
    if step_sec <= 0:
        raise ValueError("sample_minutes must be > 0.")

    out_subdir = ensure_dir(out_dir / f"{int(wavelength_angstrom):04d}")

    t = tw.start.astimezone(dt.timezone.utc)
    end = tw.end.astimezone(dt.timezone.utc)

    downloaded_fts: list[pathlib.Path] = []
    n_try = 0
    n_ok = 0
    n_miss = 0

    while t <= end:
        # この時刻に対応する「候補URL群」を試す（秒ずれ対策）
        got_this_time = False

        for off in try_time_offsets_sec:
            t_try = t + dt.timedelta(seconds=int(off))
            url = build_jhuapl_euvi_r_url(t_try, wavelength_angstrom=wavelength_angstrom, spacecraft=spacecraft)

            fname_gz = pathlib.Path(url).name  # 例: 20220613_020930_171eu_R.fts.gz
            gz_path = out_subdir / fname_gz
            fts_path = out_subdir / fname_gz.replace(".gz", "")

            # 既に解凍済みならスキップ
            if fts_path.exists() and fts_path.stat().st_size > 0:
                downloaded_fts.append(fts_path)
                got_this_time = True
                break

            # まず .gz をダウンロード
            n_try += 1
            ok = download_url(url, gz_path)
            if not ok:
                # 404なら次候補へ
                continue

            # ダウンロードできたので解凍
            gunzip_file(gz_path, fts_path, remove_gz=remove_gz)
            downloaded_fts.append(fts_path)
            n_ok += 1
            got_this_time = True
            print(f"[OK] {fts_path.name}  (from {t_try.strftime('%H:%M:%S')}Z)")
            break

        if not got_this_time:
            n_miss += 1
            # どれも404なら「その時刻付近は存在しない」扱い
            print(f"[MISS] around {t.strftime('%Y-%m-%d %H:%M:%S')}Z  (no R file found within offsets)")

        t = t + dt.timedelta(seconds=step_sec)

    print("=== Summary ===")
    print(f"tries={n_try}, downloaded_new={n_ok}, miss_slots={n_miss}, total_fts={len(downloaded_fts)}")

    if require_r and len(downloaded_fts) == 0:
        raise RuntimeError("No EUVI R files were downloaded in the requested time range.")

    # 時刻順に並べ替え（ファイル名ソートで概ねOK）
    downloaded_fts = sorted(set(downloaded_fts), key=lambda p: p.name)
    return downloaded_fts




def main():
    # 例: あなたの AIA synoptic 取得コードと同じ中心時刻・時間窓
    target_time_str = "2022-06-13T00:50:00"
    tw = TimeWindow(
        center_utc=parse_utc(target_time_str),
        minutes_before=0,
        minutes_after=180,
    )

    # 出力先（Windows パスでもOK）
    out_dir = ensure_dir(pathlib.Path(r"/mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/EUVI/Rawdata"))

    # STEREO-A EUVI（例: 195Å、5分間引き）
    # 195 , 171(2.5min), 284(20min), 304(10min)
    download_stereo_euvi(
        tw=tw,
        out_dir=out_dir,
        spacecraft="A",
        wavelength_angstrom=195,
        sample_minutes=1,
    )
    
if __name__ == "__main__":
    main()