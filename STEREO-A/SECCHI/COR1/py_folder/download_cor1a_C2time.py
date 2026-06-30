#!/usr/bin/env python3
"""
Download STEREO-A/SECCHI/COR1-A science FITS files.

Two download modes are supported.

1. download_mode = "lasco_pb"
   Scan local LASCO-C2 pB files named
       C2-PB-<YYYYMMDD>_<hhmm>.fts
   and, for each LASCO-C2 pB time, search COR1-A files named
       <YYYYMMDD>_<HHMMSS>_n4c1A.fts
   within +/- half_window_min.

   The +/- window is only a search range.  The script downloads only one
   nearest complete COR1-A minute sequence for each LASCO-C2 pB time, i.e.,
   the three files belonging to the closest sequence such as
       20220613_030100_n4c1A.fts
       20220613_030118_n4c1A.fts
       20220613_030136_n4c1A.fts

2. download_mode = "time_range"
   Download all COR1-A *_n4c1A.fts files whose observation times are
   between start_time and end_time.

SECCHI monthly-minimum background FITS (mc1A_*, mc2A_*, etc.) are also
downloaded for the same UTC time windows, filtered by YYMMDD in each
filename, into DEFAULT_MONTHLY_MIN_DIR/<YYYYMM>/.

Default inputs/outputs are set for Kinno's WSL directory layout.
"""

from __future__ import annotations

import argparse
import gzip
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

LASCO_PB_PATTERN = re.compile(r"^C2-PB-(\d{8})_(\d{4})\.fts(?:\.gz)?$", re.IGNORECASE)
COR1A_PATTERN = re.compile(r"(\d{8})_(\d{6})_n4c1A\.fts(?:\.gz)?", re.IGNORECASE)
BKG_DATE_PATTERN = re.compile(r"_(\d{6})\.fts(?:\.gz)?$", re.IGNORECASE)

DEFAULT_LASCO_PB_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research_data/SOHO/pB")
DEFAULT_COR1_RAW_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/COR1/Rawdata")

# NRL SECCHI directory generated from https://secchi.nrl.navy.mil/get_data
DEFAULT_BASE_URLS = (
    "https://secchi.nrl.navy.mil/postflight/cor1/L0/a/seq/",
    # Fallback mirror. This is slower for bulk use, but useful if NRL is unavailable.
    "https://stereo-ssc.nascom.nasa.gov/data/ins_data/secchi/L0/a/seq/cor1/",
)

MONTHLY_MIN_BASE_URL = (
    "https://soho.nascom.nasa.gov/sdb/stereo/secchi/backgrounds/a/monthly_min/"
)
DEFAULT_MONTHLY_MIN_DIR = Path(
    "/home/kinno-7010/sswdb/stereo/secchi/backgrounds/a/monthly_min"
)

HREF_PATTERN = re.compile(r'<a\s+href="([^"]+)"', re.IGNORECASE)


@dataclass(frozen=True)
class RemoteCor1File:
    name: str
    url: str
    obs_time: datetime


@dataclass(frozen=True)
class LascoNearestTriplet:
    lasco_time: datetime
    sequence_minute: datetime
    sequence_center_time: datetime
    delta_seconds: float
    files: tuple[RemoteCor1File, ...]


def parse_lasco_pb_time(path: Path) -> datetime | None:
    """Parse C2-PB-YYYYMMDD_hhmm.fts into a timezone-aware UTC datetime."""
    match = LASCO_PB_PATTERN.match(path.name)
    if match is None:
        return None
    ymd, hm = match.groups()
    return datetime.strptime(ymd + hm, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)


def parse_cor1a_time(name: str) -> datetime | None:
    """Parse YYYYMMDD_HHMMSS_n4c1A.fts(.gz) into a timezone-aware UTC datetime."""
    match = COR1A_PATTERN.search(name)
    if match is None:
        return None
    ymd, hms = match.groups()
    return datetime.strptime(ymd + hms, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def parse_utc_datetime(value: str | datetime | None, *, name: str) -> datetime:
    """
    Parse a user-specified time into a timezone-aware UTC datetime.

    Accepted string examples:
        2022-06-13T02:00:00
        2022-06-13 02:00:00
        2022-06-13T02:00:00Z
    """
    if value is None:
        raise ValueError(f"{name} must be specified.")

    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            raise ValueError(f"{name} must not be empty.")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(
                f"Could not parse {name}={value!r}. "
                "Use a format like '2022-06-13T02:00:00'."
            ) from exc

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_download_mode(mode: str) -> str:
    """Normalize mode strings used in the direct settings block or command line."""
    normalized = mode.strip().lower().replace("-", "_")
    if normalized in {"lasco", "lasco_pb", "lasco_c2_pb"}:
        return "lasco_pb"
    if normalized in {"range", "time_range", "timerange"}:
        return "time_range"
    raise ValueError("download_mode must be 'lasco_pb' or 'time_range'.")


def collect_lasco_pb_times(lasco_pb_dir: Path) -> list[datetime]:
    """Return sorted unique LASCO-C2 pB times found in lasco_pb_dir."""
    times: set[datetime] = set()
    for path in sorted(lasco_pb_dir.glob("C2-PB-*.fts*")):
        t = parse_lasco_pb_time(path)
        if t is not None:
            times.add(t)
    return sorted(times)


def merge_time_windows(times: Iterable[datetime], half_width: timedelta) -> list[tuple[datetime, datetime]]:
    """Build merged +/- half_width windows around each time."""
    raw = sorted((t - half_width, t + half_width) for t in times)
    if not raw:
        return []

    merged: list[tuple[datetime, datetime]] = [raw[0]]
    for start, end in raw[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def build_time_range_window(start_time: str | datetime, end_time: str | datetime) -> list[tuple[datetime, datetime]]:
    """Build one fixed time window from start_time to end_time."""
    start = parse_utc_datetime(start_time, name="start_time")
    end = parse_utc_datetime(end_time, name="end_time")
    if end < start:
        raise ValueError("end_time must be later than or equal to start_time.")
    return [(start, end)]


def build_lasco_search_windows(
    lasco_times: Iterable[datetime], half_width: timedelta
) -> list[tuple[datetime, datetime]]:
    """
    Build unmerged search windows around LASCO-C2 pB reference times.

    These windows are used only to determine which remote days must be listed.
    The final file selection in lasco_pb mode is done per LASCO-C2 pB time by
    select_nearest_complete_triplets().
    """
    return sorted((t - half_width, t + half_width) for t in lasco_times)


def iter_days_covering(windows: Iterable[tuple[datetime, datetime]]) -> list[str]:
    """Return YYYYMMDD strings for all UTC days touched by windows."""
    days: set[str] = set()
    for start, end in windows:
        day = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
        last_day = datetime(end.year, end.month, end.day, tzinfo=timezone.utc)
        while day <= last_day:
            days.add(day.strftime("%Y%m%d"))
            day += timedelta(days=1)
    return sorted(days)


def months_covering_windows(windows: Iterable[tuple[datetime, datetime]]) -> list[int]:
    """Return YYYYMM integers for all UTC months touched by windows."""
    return sorted({int(day[:6]) for day in iter_days_covering(windows)})


def parse_background_file_date(name: str) -> datetime | None:
    """Parse YYMMDD embedded in monthly-min background FITS names."""
    match = BKG_DATE_PATTERN.search(name)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%y%m%d").replace(tzinfo=timezone.utc)


def background_file_in_windows(
    name: str, windows: Iterable[tuple[datetime, datetime]]
) -> bool:
    """True if the background file's embedded date falls inside any window (UTC days)."""
    file_day = parse_background_file_date(name)
    if file_day is None:
        return False
    day = file_day.date()
    return any(start.date() <= day <= end.date() for start, end in windows)


def time_in_windows(t: datetime, windows: Iterable[tuple[datetime, datetime]]) -> bool:
    return any(start <= t <= end for start, end in windows)


def read_url_text(url: str, timeout: float = 60.0) -> str:
    request = Request(url, headers={"User-Agent": "cor1a-downloader/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def extract_cor1a_filenames(directory_html: str) -> list[str]:
    """Extract unique *_n4c1A.fts or *_n4c1A.fts.gz names from an Apache-style index."""
    names = sorted(set(match.group(0) for match in COR1A_PATTERN.finditer(directory_html)))
    return names


def list_remote_day(day_yyyymmdd: str, base_urls: tuple[str, ...]) -> list[RemoteCor1File]:
    """List remote COR1-A n4 science files for one day from the first reachable archive."""
    last_error: Exception | None = None
    for base_url in base_urls:
        day_url = urljoin(base_url.rstrip("/") + "/", day_yyyymmdd + "/")
        try:
            html = read_url_text(day_url)
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            continue

        files: list[RemoteCor1File] = []
        for name in extract_cor1a_filenames(html):
            obs_time = parse_cor1a_time(name)
            if obs_time is None:
                continue
            files.append(RemoteCor1File(name=name, url=urljoin(day_url, name), obs_time=obs_time))
        return sorted(files, key=lambda item: item.obs_time)

    if last_error is not None:
        print(f"[WARN] Could not open remote directory for {day_yyyymmdd}: {last_error}", file=sys.stderr)
    return []


def cor1a_final_name(name: str, *, decompress_gz: bool = True) -> str:
    """Return the final local filename for a remote COR1-A file name."""
    if name.lower().endswith(".gz") and decompress_gz:
        return name[:-3]
    return name


def deduplicate_remote_files(files: Iterable[RemoteCor1File]) -> list[RemoteCor1File]:
    """
    Deduplicate remote files by their final FITS name.

    If both .fts and .fts.gz are present for the same observation, prefer .fts.
    """
    by_final_name: dict[str, RemoteCor1File] = {}
    for item in files:
        key = cor1a_final_name(item.name, decompress_gz=True)
        previous = by_final_name.get(key)
        if previous is None:
            by_final_name[key] = item
            continue
        previous_is_gz = previous.name.lower().endswith(".gz")
        current_is_gz = item.name.lower().endswith(".gz")
        if previous_is_gz and not current_is_gz:
            by_final_name[key] = item
    return sorted(by_final_name.values(), key=lambda item: item.obs_time)


def sequence_minute_key(obs_time: datetime) -> datetime:
    """Return the minute key for a COR1-A polarization sequence."""
    return obs_time.replace(second=0, microsecond=0)


def select_nearest_complete_triplets(
    *,
    lasco_times: Iterable[datetime],
    remote_files: Iterable[RemoteCor1File],
    half_window_min: float,
    files_per_sequence: int = 3,
) -> tuple[list[RemoteCor1File], list[LascoNearestTriplet]]:
    """
    Select one nearest complete COR1-A triplet for each LASCO-C2 pB time.

    The +/- half_window_min interval is used only as a search range.  Within
    that interval, remote files are grouped by their minute key, e.g.
    03:01:00, 03:01:18, 03:01:36 -> 03:01.  The selected group is the complete
    group whose center time is closest to the LASCO-C2 pB reference time.
    """
    half_width = timedelta(minutes=half_window_min)
    all_files = deduplicate_remote_files(remote_files)

    selected_reports: list[LascoNearestTriplet] = []
    selected_files: list[RemoteCor1File] = []

    for lasco_time in sorted(lasco_times):
        start = lasco_time - half_width
        end = lasco_time + half_width
        candidates = [item for item in all_files if start <= item.obs_time <= end]

        groups: dict[datetime, list[RemoteCor1File]] = {}
        for item in candidates:
            groups.setdefault(sequence_minute_key(item.obs_time), []).append(item)

        complete_groups: list[tuple[datetime, tuple[RemoteCor1File, ...], datetime, float]] = []
        for minute_key, group_files in groups.items():
            unique_group = deduplicate_remote_files(group_files)
            if len(unique_group) < files_per_sequence:
                continue
            triplet = tuple(sorted(unique_group, key=lambda item: item.obs_time)[:files_per_sequence])
            center_time = triplet[len(triplet) // 2].obs_time
            delta_seconds = abs((center_time - lasco_time).total_seconds())
            complete_groups.append((minute_key, triplet, center_time, delta_seconds))

        if not complete_groups:
            print(
                "[WARN] No complete COR1-A triplet found within "
                f"+/- {half_window_min:g} min of LASCO-C2 pB time "
                f"{lasco_time:%Y-%m-%dT%H:%M:%S} UT.",
                file=sys.stderr,
            )
            continue

        minute_key, triplet, center_time, delta_seconds = min(
            complete_groups,
            key=lambda item: (item[3], item[0]),
        )
        selected_reports.append(
            LascoNearestTriplet(
                lasco_time=lasco_time,
                sequence_minute=minute_key,
                sequence_center_time=center_time,
                delta_seconds=delta_seconds,
                files=triplet,
            )
        )
        selected_files.extend(triplet)

    selected_by_final_name = {
        cor1a_final_name(item.name, decompress_gz=True): item for item in selected_files
    }
    unique_selected = sorted(selected_by_final_name.values(), key=lambda item: item.obs_time)
    return unique_selected, selected_reports


def download_one(remote: RemoteCor1File, out_dir: Path, *, decompress_gz: bool, overwrite: bool) -> Path | None:
    """Download one remote file. Return local path, or None if skipped/failed."""
    out_dir.mkdir(parents=True, exist_ok=True)

    remote_name = remote.name
    final_name = remote_name[:-3] if remote_name.lower().endswith(".gz") and decompress_gz else remote_name
    final_path = out_dir / final_name
    temp_path = out_dir / remote_name

    if final_path.exists() and not overwrite:
        print(f"[SKIP] Exists: {final_path}")
        return final_path

    try:
        print(f"[GET] {remote.url}")
        request = Request(remote.url, headers={"User-Agent": "cor1a-downloader/1.0"})
        with urlopen(request, timeout=120.0) as response, open(temp_path, "wb") as handle:
            shutil.copyfileobj(response, handle)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        print(f"[WARN] Download failed: {remote.url} ({exc})", file=sys.stderr)
        return None

    if remote_name.lower().endswith(".gz") and decompress_gz:
        try:
            with gzip.open(temp_path, "rb") as src, open(final_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            temp_path.unlink()
        except OSError as exc:
            print(f"[WARN] Could not decompress {temp_path}: {exc}", file=sys.stderr)
            return None

    print(f"[OK] {final_path}")
    return final_path


def parse_base_urls(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_BASE_URLS
    return tuple(url.rstrip("/") + "/" for url in values)


def extract_directory_filenames(directory_html: str) -> list[str]:
    """Extract downloadable file names from an Apache-style directory index."""
    names: set[str] = set()
    for match in HREF_PATTERN.finditer(directory_html):
        href = match.group(1).strip()
        if not href or href in (".", ".."):
            continue
        if href.startswith("?"):
            continue
        if href.endswith("/"):
            continue
        base = href.rstrip("/").split("/")[-1]
        if not base or base.lower().startswith("index.html"):
            continue
        names.add(base)
    return sorted(names)


def download_binary_url(url: str, dest: Path, *, overwrite: bool) -> bool:
    """Download one binary file to dest. Return True on success or skip."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        print(f"[SKIP] Exists: {dest}")
        return True
    try:
        print(f"[GET] {url}")
        request = Request(url, headers={"User-Agent": "cor1a-downloader/1.0"})
        with urlopen(request, timeout=120.0) as response, open(dest, "wb") as handle:
            shutil.copyfileobj(response, handle)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        print(f"[WARN] Download failed: {url} ({exc})", file=sys.stderr)
        return False
    print(f"[OK] {dest}")
    return True


def download_monthly_min_month(
    month: int,
    out_dir: Path,
    *,
    time_windows: Iterable[tuple[datetime, datetime]] | None = None,
    base_url: str = MONTHLY_MIN_BASE_URL,
    dry_run: bool = False,
    overwrite: bool = False,
    sleep_sec: float = 0.1,
) -> tuple[int, int]:
    """Download files listed under monthly_min/<month>/ into out_dir/<month>/."""
    month_url = urljoin(base_url.rstrip("/") + "/", f"{month}/")
    local_month_dir = out_dir / str(month)
    try:
        html = read_url_text(month_url)
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"[WARN] Could not list {month_url}: {exc}", file=sys.stderr)
        return 0, 1

    names = extract_directory_filenames(html)
    if time_windows is not None:
        names = [name for name in names if background_file_in_windows(name, time_windows)]
    if not names:
        print(f"[WARN] No matching files under {month_url}", file=sys.stderr)
        return 0, 0

    ok = failed = 0
    for name in names:
        file_url = urljoin(month_url, name)
        dest = local_month_dir / name
        if dry_run:
            print(f"[DRY] Would download: {file_url} -> {dest}")
            ok += 1
            continue
        if download_binary_url(file_url, dest, overwrite=overwrite):
            ok += 1
        else:
            failed += 1
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    return ok, failed


def download_monthly_min_backgrounds(
    *,
    time_windows: Iterable[tuple[datetime, datetime]],
    months: Iterable[int] | None = None,
    out_dir: Path = DEFAULT_MONTHLY_MIN_DIR,
    base_url: str = MONTHLY_MIN_BASE_URL,
    dry_run: bool = False,
    overwrite: bool = False,
    sleep_sec: float = 0.1,
) -> int:
    """
    Download SECCHI monthly-minimum background FITS files for a time range.

    ``time_windows`` defines which YYMMDD-tagged background files are kept.
    ``months`` defaults to all YYYYMM values touched by those windows.
    """
    windows = list(time_windows)
    if not windows:
        print("[WARN] No time windows for monthly backgrounds.", file=sys.stderr)
        return 0

    month_list = sorted(months) if months is not None else months_covering_windows(windows)
    out_dir = Path(out_dir)
    total_ok = total_failed = 0
    for month in month_list:
        print(f"[INFO] Monthly backgrounds: month={month}")
        ok, failed = download_monthly_min_month(
            month,
            out_dir,
            time_windows=windows,
            base_url=base_url,
            dry_run=dry_run,
            overwrite=overwrite,
            sleep_sec=sleep_sec,
        )
        total_ok += ok
        total_failed += failed
    print(
        f"[INFO] Monthly backgrounds finished. ok={total_ok} failed={total_failed} -> {out_dir}"
    )
    return 1 if total_failed else 0


def run_download(
    *,
    download_mode: str,
    start_time: str | datetime | None = None,
    end_time: str | datetime | None = None,
    lasco_pb_dir: Path = DEFAULT_LASCO_PB_DIR,
    out_dir: Path = DEFAULT_COR1_RAW_DIR,
    half_window_min: float = 60.0,
    base_urls: tuple[str, ...] = DEFAULT_BASE_URLS,
    download_monthly_min: bool = True,
    monthly_min_dir: Path = DEFAULT_MONTHLY_MIN_DIR,
    dry_run: bool = False,
    overwrite: bool = False,
    keep_gz: bool = False,
    sleep_sec: float = 0.1,
) -> int:
    """
    Download COR1-A files using either LASCO-C2 pB reference times or a fixed time range.

    In lasco_pb mode, +/- half_window_min is only the search range.  The final
    selection is one nearest complete 3-file COR1-A sequence per LASCO-C2 pB
    time.  In time_range mode, all files within the fixed time range are
    selected.
    """
    try:
        mode = normalize_download_mode(download_mode)
        if mode == "time_range":
            if start_time is None or end_time is None:
                raise ValueError("start_time and end_time are required when download_mode='time_range'.")
            selection_windows = build_time_range_window(start_time, end_time)
            lasco_times: list[datetime] = []
            reference_count = 1
            search_windows = selection_windows
        else:
            lasco_times = collect_lasco_pb_times(lasco_pb_dir)
            if not lasco_times:
                raise FileNotFoundError(f"No C2-PB-YYYYMMDD_hhmm.fts files found in {lasco_pb_dir}")
            reference_count = len(lasco_times)
            half_width = timedelta(minutes=half_window_min)
            search_windows = build_lasco_search_windows(lasco_times, half_width)
            selection_windows = merge_time_windows(lasco_times, half_width)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    days = iter_days_covering(search_windows)

    print("------------------------------------------------------------")
    print("[INFO] Download mode:", mode)
    if mode == "lasco_pb":
        print("[INFO] LASCO-C2 pB directory:", lasco_pb_dir)
        print("[INFO] Number of LASCO-C2 pB reference times:", reference_count)
        print("[INFO] Search half-window around each LASCO-C2 pB time [min]:", half_window_min)
        print("[INFO] Selection rule: nearest complete 3-file COR1-A minute sequence per LASCO-C2 pB time")
    else:
        print("[INFO] Fixed time range mode")
        print("[INFO] Selection rule: all COR1-A files in the fixed time range")
    print("[INFO] COR1-A output directory:", out_dir)
    print("[INFO] Merged search windows:", len(selection_windows))
    for start, end in selection_windows:
        print(f"       {start:%Y-%m-%dT%H:%M:%S} -- {end:%Y-%m-%dT%H:%M:%S} UT")
    print("[INFO] Remote days:", ", ".join(days))
    if download_monthly_min:
        bkg_months = months_covering_windows(selection_windows)
        print("[INFO] Monthly-min background directory:", monthly_min_dir)
        print("[INFO] Monthly-min months:", ", ".join(str(m) for m in bkg_months))
        print("[INFO] Monthly-min filter: YYMMDD in selection time windows")
    print("[INFO] Archive base URLs:")
    for url in base_urls:
        print("      ", url)
    print("------------------------------------------------------------")

    remote_files: list[RemoteCor1File] = []
    for day in days:
        day_files = list_remote_day(day, base_urls)
        remote_files.extend(day_files)
        if mode == "time_range":
            matched_count = sum(1 for item in day_files if time_in_windows(item.obs_time, selection_windows))
            print(f"[INFO] {day}: remote n4c1A={len(day_files)}, selected-in-range={matched_count}")
        else:
            search_count = sum(1 for item in day_files if time_in_windows(item.obs_time, search_windows))
            print(f"[INFO] {day}: remote n4c1A={len(day_files)}, in-search-window={search_count}")

    remote_files = deduplicate_remote_files(remote_files)

    if mode == "time_range":
        selected = [item for item in remote_files if time_in_windows(item.obs_time, selection_windows)]
        triplet_reports: list[LascoNearestTriplet] = []
    else:
        selected, triplet_reports = select_nearest_complete_triplets(
            lasco_times=lasco_times,
            remote_files=remote_files,
            half_window_min=half_window_min,
            files_per_sequence=3,
        )

    print("------------------------------------------------------------")
    if mode == "lasco_pb":
        print("[INFO] Selected nearest COR1-A sequences:", len(triplet_reports), "/", reference_count)
        for report in triplet_reports:
            file_names = ", ".join(item.name for item in report.files)
            print(
                f"[PAIR] LASCO {report.lasco_time:%Y-%m-%dT%H:%M:%S} UT -> "
                f"COR1A {report.sequence_minute:%Y-%m-%dT%H:%M} "
                f"center={report.sequence_center_time:%H:%M:%S} "
                f"dt={report.delta_seconds:.1f} s :: {file_names}"
            )
    print("[INFO] Total selected COR1-A files:", len(selected))

    exit_code = 0

    if dry_run:
        for item in selected:
            print(f"[DRY] {item.obs_time:%Y-%m-%dT%H:%M:%S}  {item.name}")
    else:
        downloaded = 0
        for item in selected:
            local = download_one(
                item,
                out_dir,
                decompress_gz=not keep_gz,
                overwrite=overwrite,
            )
            if local is not None:
                downloaded += 1
            if sleep_sec > 0:
                time.sleep(sleep_sec)

        print("------------------------------------------------------------")
        print(f"[INFO] COR1-A finished. Downloaded/skipped-valid: {downloaded} / {len(selected)}")

    if download_monthly_min:
        print("------------------------------------------------------------")
        monthly_code = download_monthly_min_backgrounds(
            time_windows=selection_windows,
            out_dir=monthly_min_dir,
            dry_run=dry_run,
            overwrite=overwrite,
            sleep_sec=sleep_sec,
        )
        exit_code = max(exit_code, monthly_code)

    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download STEREO-A/SECCHI/COR1-A *_n4c1A.fts files. "
            "Use --mode lasco_pb for nearest triplets around LASCO-C2 pB times, "
            "or --mode time_range for all files in a fixed time range."
        )
    )
    parser.add_argument(
        "--mode",
        default="lasco_pb",
        help="Download mode: 'lasco_pb' or 'time_range'. Default: lasco_pb.",
    )
    parser.add_argument("--start-time", default=None, help="UTC start time for --mode time_range.")
    parser.add_argument("--end-time", default=None, help="UTC end time for --mode time_range.")
    parser.add_argument("--lasco-pb-dir", type=Path, default=DEFAULT_LASCO_PB_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_COR1_RAW_DIR)
    parser.add_argument(
        "--half-window-min",
        type=float,
        default=60.0,
        help=(
            "Search half-window in minutes around each LASCO-C2 pB time. "
            "In lasco_pb mode, this is not the number of files to download; "
            "only the nearest complete 3-file COR1-A sequence is selected."
        ),
    )
    parser.add_argument("--base-url", action="append", help="Archive base URL. Can be specified more than once.")
    parser.add_argument("--dry-run", action="store_true", help="Only show files that would be downloaded.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing local files.")
    parser.add_argument("--keep-gz", action="store_true", help="Keep .fts.gz as-is if the archive provides gzipped FITS.")
    parser.add_argument("--sleep", type=float, default=0.1, help="Sleep seconds between downloads.")
    parser.add_argument(
        "--monthly-min-dir",
        type=Path,
        default=DEFAULT_MONTHLY_MIN_DIR,
        help="Output directory for SECCHI monthly-minimum background FITS.",
    )
    parser.add_argument(
        "--skip-monthly-min",
        action="store_true",
        help="Do not download monthly-minimum background FITS for the same time range.",
    )
    args = parser.parse_args()

    return run_download(
        download_mode=args.mode,
        start_time=args.start_time,
        end_time=args.end_time,
        lasco_pb_dir=args.lasco_pb_dir,
        out_dir=args.out_dir,
        half_window_min=args.half_window_min,
        base_urls=parse_base_urls(args.base_url),
        download_monthly_min=not args.skip_monthly_min,
        monthly_min_dir=args.monthly_min_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        keep_gz=args.keep_gz,
        sleep_sec=args.sleep,
    )


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Direct settings block.
    #
    # If you run this script without command-line arguments,
    # the settings below are used.
    #
    # DOWNLOAD_MODE:
    #   "lasco_pb"   : for each local C2-PB-YYYYMMDD_hhmm.fts time,
    #                  search within +/- HALF_WINDOW_MIN and download only
    #                  the nearest complete 3-file COR1-A minute sequence.
    #   "time_range" : download all *_n4c1A.fts files from START_TIME
    #                  to END_TIME.
    #
    # Monthly-minimum backgrounds are downloaded for the same UTC time
    # windows (START_TIME..END_TIME or LASCO-C2 pB times +/- HALF_WINDOW_MIN).
    #
    # If command-line arguments are supplied, argparse is used instead.
    # Examples:
    #   python3 download_cor1a_pb.py --mode lasco_pb --dry-run
    #   python3 download_cor1a_pb.py --mode time_range \
    #       --start-time 2022-06-13T02:00:00 --end-time 2022-06-13T04:00:00
    # ------------------------------------------------------------------

    DOWNLOAD_MODE = "lasco_pb"  # "lasco_pb" or "time_range"

    START_TIME = "2022-06-13T02:00:00"
    END_TIME = "2022-06-13T04:00:00"

    LASCO_PB_DIR = DEFAULT_LASCO_PB_DIR
    OUT_DIR = DEFAULT_COR1_RAW_DIR

    HALF_WINDOW_MIN = 60.0
    DRY_RUN = False
    OVERWRITE = False
    KEEP_GZ = False
    SLEEP_SEC = 0.1
    DOWNLOAD_MONTHLY_MIN = True
    MONTHLY_MIN_DIR = DEFAULT_MONTHLY_MIN_DIR

    if len(sys.argv) > 1:
        raise SystemExit(main())

    raise SystemExit(
        run_download(
            download_mode=DOWNLOAD_MODE,
            start_time=START_TIME,
            end_time=END_TIME,
            lasco_pb_dir=LASCO_PB_DIR,
            out_dir=OUT_DIR,
            half_window_min=HALF_WINDOW_MIN,
            base_urls=DEFAULT_BASE_URLS,
            download_monthly_min=DOWNLOAD_MONTHLY_MIN,
            monthly_min_dir=MONTHLY_MIN_DIR,
            dry_run=DRY_RUN,
            overwrite=OVERWRITE,
            keep_gz=KEEP_GZ,
            sleep_sec=SLEEP_SEC,
        )
    )
