#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_kcor_lasco_pb.py

Standalone preparation script for Earth-view pB inputs used by tomography.

This file completes the entire LASCO-C2/K-Cor preparation step by itself:
  1) discover/download LASCO-C2 pB files in target_time +/- window_days,
  2) discover/download only K-Cor pB/pBavg files close to each LASCO-C2 pB time,
  3) reproject K-Cor onto the LASCO grid,
  4) write tomography-ready pB_Kcor_LASCO_axi_*.fits products.

main_multi_tomo.py does not call this script. Run this file first when the
K-Cor/LASCO pB products need to be created or refreshed, then run
main_multi_tomo.py to perform tomography using the prepared FITS files.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import gzip
import re
import shutil
import tarfile
import urllib.request
import urllib.parse
import http.cookiejar
from urllib.error import HTTPError, URLError

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


# Set by download_url() when a remote server returns a registration/auth/error HTML page.
LAST_DOWNLOAD_AUTH_BLOCKED = False


def parse_target_datetime(value: str | datetime) -> datetime:
    """
    Parse a target time used for the rotational-tomography data window.

    Accepted string formats:
      - YYYYMMDD_HHMM
      - YYYYMMDD_HHMMSS
      - YYYY-MM-DD HH:MM
      - YYYY-MM-DD HH:MM:SS
      - YYYY-MM-DDTHH:MM
      - YYYY-MM-DDTHH:MM:SS
    """
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)

    text = str(value).strip()
    formats = (
        "%Y%m%d_%H%M",
        "%Y%m%d_%H%M%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%dT%H:%M:%S",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError(
        f"Cannot parse target_time={value!r}. Use e.g. '20220613_0300', "
        f"'20220613_030000', or '2022-06-13T03:00:00'."
    )

def iter_dates_in_window(target_time: str | datetime, window_days: float) -> List[datetime]:
    """Return midnight datetimes for all calendar dates intersecting target_time +/- window_days."""
    target_dt = parse_target_datetime(target_time)
    t0 = target_dt - timedelta(days=float(window_days))
    t1 = target_dt + timedelta(days=float(window_days))
    d = datetime(t0.year, t0.month, t0.day)
    dates = []
    while d <= datetime(t1.year, t1.month, t1.day):
        dates.append(d)
        d += timedelta(days=1)
    return dates

def parse_pb_filename_datetime(path: Path) -> Optional[datetime]:
    """
    Extract observation time from supported pB FITS filenames.

    Supported patterns:
      - pB_Kcor_LASCO_axi_<YYYYMMDD>_<HHMM>.fits
      - COR1A_pb_pre_<YYYYMMDD>_<HHMMSS>.fits
      - C2-PB-<YYYYMMDD>_<HHMM>.fts / .fits
      - pB_LASCO_C2_only_<YYYYMMDD>_<HHMM>.fits
      - <YYYYMMDD>_<HHMMSS>_kcor_l2_pb.fts / .fits
      - <YYYYMMDD>_<HHMMSS>_kcor_l2_pbavg.fts / .fits
      - <YYYYMMDD>_<HHMMSS>_kcor_l2_pb_avg.fts / .fits
    """
    name = Path(path).name

    m = re.fullmatch(r"pB_Kcor_LASCO_axi_(\d{8})_(\d{4})\.fits", name)
    if m:
        return datetime.strptime(m.group(1) + "_" + m.group(2), "%Y%m%d_%H%M")

    m = re.fullmatch(r"pB_Kcor_LASCO_edge_smooth_(\d{8})_(\d{4})\.fits", name)
    if m:
        return datetime.strptime(m.group(1) + "_" + m.group(2), "%Y%m%d_%H%M")

    m = re.fullmatch(r"COR1A_pb_pre_(\d{8})_(\d{6})\.fits", name)
    if m:
        return datetime.strptime(m.group(1) + "_" + m.group(2), "%Y%m%d_%H%M%S")

    m = re.fullmatch(r"C2-PB-(\d{8})_(\d{4})\.(?:fts|fits)(?:\.gz)?", name)
    if m:
        return datetime.strptime(m.group(1) + "_" + m.group(2), "%Y%m%d_%H%M")

    m = re.fullmatch(r"pB_LASCO_C2_only_(\d{8})_(\d{4})\.fits", name)
    if m:
        return datetime.strptime(m.group(1) + "_" + m.group(2), "%Y%m%d_%H%M")

    m = re.fullmatch(r"(\d{8})_(\d{6})_kcor_l2_pb(?:_?avg)?\.(?:fts|fits)(?:\.gz)?", name)
    if m:
        return datetime.strptime(m.group(1) + "_" + m.group(2), "%Y%m%d_%H%M%S")

    return None

def lasco_c2_pb_url(obs_dt: datetime) -> str:
    """Build the LASCO-C2 pB URL for a given observation datetime."""
    return (
        "https://lasco-www.nrl.navy.mil/lz/polarize/"
        f"{obs_dt:%Y_%m}/vig/c2/C2-PB-{obs_dt:%Y%m%d}_{obs_dt:%H%M}.fts"
    )

def lasco_c2_month_listing_url(year: int, month: int) -> str:
    """Directory URL used to discover available LASCO-C2 pB files for a month."""
    return f"https://lasco-www.nrl.navy.mil/lz/polarize/{year:04d}_{month:02d}/vig/c2/"

def kcor_pb_url(obs_dt: datetime, kcor_product: str = "pbavg") -> str:
    """Build the HAO/MLSO K-Cor level-2 pB/pBavg download URL for a given observation datetime."""
    product = str(kcor_product).strip().lower()
    if product not in ("pb", "pbavg"):
        raise ValueError(f"Unsupported K-Cor product: {kcor_product!r}. Use 'pb' or 'pbavg'.")

    yyyy = f"{obs_dt:%Y}"
    mm = f"{obs_dt:%m}"
    dd = f"{obs_dt:%d}"
    ymd = f"{obs_dt:%Y%m%d}"
    hms = f"{obs_dt:%H%M%S}"
    # MLSO/HAO K-Cor averaged pB files are commonly named with an underscore: pb_avg.
    # Keep the public query parameter as proc=pbavg, but request the actual file suffix pb_avg.
    file_product = "pb_avg" if product == "pbavg" else product
    file_path = f"/hao/acos/{yyyy}/{mm}/{dd}/{ymd}_{hms}_kcor_l2_{file_product}.fts.gz"
    referer = kcor_listing_url(obs_dt, kcor_product=product)
    query = urllib.parse.urlencode({
        "file": file_path,
        "referer": referer,
        "instrument": "kcor",
    })
    return f"https://registration.hao.ucar.edu/hao-reg_file-deliver.php?{query}"

def kcor_listing_url(obs_dt: datetime, kcor_product: str = "pbavg") -> str:
    """Build the MLSO K-Cor L2 pB/pBavg listing URL for one calendar date."""
    product = str(kcor_product).strip().lower()
    if product not in ("pb", "pbavg"):
        raise ValueError(f"Unsupported K-Cor product: {kcor_product!r}. Use 'pb' or 'pbavg'.")

    query = urllib.parse.urlencode({
        "date1": f"{obs_dt:%Y-%m-%d}",
        "inst": "kcor",
        "level": "l2",
        "qual": "all",
        "proc": product,
    })
    return f"https://mlso.hao.ucar.edu/mlso_data_get.php?{query}"

def extract_kcor_download_links_from_html(html: str, base_url: str, kcor_product: str = "pbavg") -> List[Tuple[datetime, str]]:
    """Extract K-Cor pB/pBavg file times and download URLs from an MLSO/HAO HTML page."""
    if not html:
        return []

    product = str(kcor_product).strip().lower()
    if product not in ("pb", "pbavg"):
        raise ValueError(f"Unsupported K-Cor product: {kcor_product!r}. Use 'pb' or 'pbavg'.")

    results: List[Tuple[datetime, str]] = []
    seen = set()
    # The MLSO query uses proc=pbavg, but the filename may be either pbavg or pb_avg.
    product_re = r"pb_?avg" if product == "pbavg" else re.escape(product)

    # 1) Extract links that already contain the target filename or the HAO delivery endpoint.
    for href in re.findall(r'''href=[\"']([^\"']+)[\"']''', html, flags=re.IGNORECASE):
        href_unescaped = href.replace("&amp;", "&")
        full_url = urllib.parse.urljoin(base_url, href_unescaped)
        decoded_url = urllib.parse.unquote(full_url)
        m = re.search(rf"(\d{{8}})_(\d{{6}})_kcor_l2_({product_re})\.(?:fts|fits)(?:\.gz)?", decoded_url, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            obs_dt = datetime.strptime(m.group(1) + "_" + m.group(2), "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        key = (obs_dt, full_url)
        if key not in seen:
            results.append((obs_dt, full_url))
            seen.add(key)

    # 2) Fallback: extract bare filenames from the page and construct HAO delivery URLs.
    for ymd, hms, found_product in re.findall(rf"(\d{{8}})_(\d{{6}})_kcor_l2_({product_re})\.(?:fts|fits)(?:\.gz)?", html, flags=re.IGNORECASE):
        try:
            obs_dt = datetime.strptime(f"{ymd}_{hms}", "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        found_product = found_product.lower()
        yyyy, mm, dd = ymd[:4], ymd[4:6], ymd[6:8]
        file_path = f"/hao/acos/{yyyy}/{mm}/{dd}/{ymd}_{hms}_kcor_l2_{found_product}.fts.gz"
        query = urllib.parse.urlencode({
            "file": file_path,
            "referer": base_url,
            "instrument": "kcor",
        })
        full_url = f"https://registration.hao.ucar.edu/hao-reg_file-deliver.php?{query}"
        key = (obs_dt, full_url)
        if key not in seen:
            results.append((obs_dt, full_url))
            seen.add(key)

    results.sort(key=lambda item: item[0])
    return results

def list_remote_kcor_pb_links_for_day(day_dt: datetime, timeout: float = 60.0, kcor_product: str = "pbavg") -> List[Tuple[datetime, str]]:
    """
    Read the MLSO K-Cor L2 pB/pBavg daily listing and return available download links.

    This avoids blindly probing non-existent K-Cor timestamps. The links returned by the
    MLSO page can still pass through HAO's registration gateway, so a later download step
    may still fail unless the gateway returns a binary file or the user provides a valid
    cookie file.
    """
    day_dt = datetime(day_dt.year, day_dt.month, day_dt.day)
    url = kcor_listing_url(day_dt, kcor_product=kcor_product)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            html = response.read().decode("utf-8", errors="ignore")
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        print(f"[WARN] Could not read K-Cor daily listing: {url} ({exc})")
        return []

    links = extract_kcor_download_links_from_html(html, base_url=url, kcor_product=kcor_product)
    if not links:
        print(f"[WARN] No K-Cor {kcor_product} links found in MLSO listing for {day_dt:%Y-%m-%d}.")
    return links

def _html_looks_like_registration_or_error(html: str) -> bool:
    """Return True when an HTML payload is likely an HAO registration/error page."""
    h = html.lower()
    keywords = (
        "registration", "register", "terms", "agreement", "captcha",
        "login", "not found", "error", "unavailable", "file not found",
        "hao-reg", "mlso",
    )
    return any(k in h for k in keywords)

def _save_html_debug(payload: bytes, output_path: Path, url: str) -> Path:
    """Save an unexpected HTML response next to the requested output for debugging."""
    debug_path = output_path.with_name(output_path.name + ".html")
    try:
        text = payload.decode("utf-8", errors="ignore")
        debug_path.write_text(f"<!-- URL: {url} -->\n" + text, encoding="utf-8")
    except Exception:
        debug_path.write_bytes(payload)
    return debug_path

def _extract_followable_binary_links(html: str, base_url: str) -> List[str]:
    """
    Extract possible binary-file links from an HTML wrapper.

    This handles simple cases where the server returns an intermediate HTML page containing
    a direct .fts/.fits/.gz/.tar link. It intentionally does not submit registration forms.
    """
    urls: List[str] = []
    seen = set()
    patterns: List[str] = []
    patterns.extend(re.findall(r'''href=[\"\']([^\"\']+)[\"\']''', html, flags=re.IGNORECASE))
    patterns.extend(re.findall(r'''src=[\"\']([^\"\']+)[\"\']''', html, flags=re.IGNORECASE))
    patterns.extend(re.findall(r'''URL=([^\"'<>\s]+)''', html, flags=re.IGNORECASE))
    for href in patterns:
        href = href.replace("&amp;", "&").strip()
        full = urllib.parse.urljoin(base_url, href)
        decoded = urllib.parse.unquote(full)
        if (
            re.search(r"\.(?:fts|fits)(?:\.gz)?(?:$|[?&#])", decoded, flags=re.IGNORECASE)
            or re.search(r"\.(?:tar|tgz|tar\.gz)(?:$|[?&#])", decoded, flags=re.IGNORECASE)
            or "hao-reg_file-deliver.php" in decoded
        ):
            if full not in seen:
                urls.append(full)
                seen.add(full)
    return urls

def _load_cookie_header_from_text_file(cookie_file: Path) -> Optional[str]:
    """
    Load a simple browser Cookie header from a text file.

    This is a fallback for files that are not in Netscape cookies.txt format. The file may
    contain either a raw 'name=value; name2=value2' string or a line beginning with 'Cookie:'.
    """
    try:
        text = Path(cookie_file).read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None
    if not text:
        return None

    # Prefer an explicit Cookie: line when present.
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.lower().startswith("cookie:"):
            val = s.split(":", 1)[1].strip()
            return val or None

    # Netscape cookies.txt has tab-separated fields; raw cookie strings have semicolon-separated pairs.
    if ";" in text and "=" in text and "\t" not in text:
        return text.replace("\n", " ").strip()
    return None

def download_url(
    url: str,
    output_path: Path,
    referer: Optional[str] = None,
    timeout: float = 60.0,
    cookie_file: Optional[Path] = None,
    follow_html_once: bool = True,
    _visited: Optional[set] = None,
) -> bool:
    """
    Download a URL to output_path. Return False for missing/unavailable/auth-blocked files.

    HAO's registration gateway can return HTML instead of the requested FITS/tar payload.
    In that case, this function saves a sidecar *.html file for inspection and, at most once,
    follows an embedded binary-looking link if present. It does not attempt to submit HAO
    registration forms.

    cookie_file accepts either:
      - Netscape cookies.txt format, or
      - a raw browser Cookie header string such as 'A=B; C=D'.
    """
    global LAST_DOWNLOAD_AUTH_BLOCKED

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _visited is None:
        LAST_DOWNLOAD_AUTH_BLOCKED = False
        _visited = set()
    if url in _visited:
        return False
    _visited.add(url)

    headers = {"User-Agent": "Mozilla/5.0"}
    if referer:
        headers["Referer"] = referer

    cookie_file = Path(cookie_file).expanduser() if cookie_file else None
    cookie_header = None
    opener = None
    if cookie_file:
        try:
            cj = http.cookiejar.MozillaCookieJar(str(cookie_file))
            cj.load(ignore_discard=True, ignore_expires=True)
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
        except Exception:
            cookie_header = _load_cookie_header_from_text_file(cookie_file)
            if cookie_header:
                headers["Cookie"] = cookie_header
            else:
                print(f"[WARN] Could not load cookie file as Netscape or raw Cookie header: {cookie_file}")

    req = urllib.request.Request(url, headers=headers)

    try:
        if opener is not None:
            with opener.open(req, timeout=timeout) as response:
                payload = response.read()
        else:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                payload = response.read()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        print(f"[SKIP] Download failed: {url} ({exc})")
        return False

    head = payload[:512].lower().lstrip()
    is_html = head.startswith(b"<html") or head.startswith(b"<!doctype html") or b"<html" in head[:256]
    if is_html:
        html = payload.decode("utf-8", errors="ignore")
        debug_path = _save_html_debug(payload, output_path, url)
        links = _extract_followable_binary_links(html, base_url=url)
        for link in links:
            if link != url and link not in _visited and follow_html_once:
                print(f"[INFO] HTML wrapper returned; trying embedded data link: {link}")
                if download_url(
                    link,
                    output_path,
                    referer=url,
                    timeout=timeout,
                    cookie_file=cookie_file,
                    follow_html_once=False,
                    _visited=_visited,
                ):
                    return True

        if _html_looks_like_registration_or_error(html):
            LAST_DOWNLOAD_AUTH_BLOCKED = True
            print(
                "[AUTH] HAO returned an HTML registration/error page instead of K-Cor FITS/tar data. "
                f"Saved HTML debug page: {debug_path}"
            )
        else:
            print(f"[SKIP] Server returned HTML instead of FITS/tar data. Saved HTML debug page: {debug_path}")
        return False

    output_path.write_bytes(payload)
    return True

def list_remote_lasco_c2_times_for_month(year: int, month: int, timeout: float = 60.0) -> List[datetime]:
    """Try to parse the LASCO monthly directory listing and return available C2 pB times."""
    url = lasco_c2_month_listing_url(year, month)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            html = response.read().decode("utf-8", errors="ignore")
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        print(f"[WARN] Could not read LASCO directory listing: {url} ({exc})")
        return []

    times = []
    for ymd, hhmm in re.findall(r"C2-PB-(\d{8})_(\d{4})\.fts", html):
        try:
            times.append(datetime.strptime(f"{ymd}_{hhmm}", "%Y%m%d_%H%M"))
        except ValueError:
            pass
    return sorted(set(times))

def download_lasco_c2_pb(obs_dt: datetime, lasco_raw_dir: Path, overwrite: bool = False) -> Optional[Path]:
    """Download one LASCO-C2 pB FITS file. Missing files are skipped."""
    lasco_raw_dir = Path(lasco_raw_dir).expanduser()
    out = lasco_raw_dir / f"C2-PB-{obs_dt:%Y%m%d}_{obs_dt:%H%M}.fts"
    if out.exists() and not overwrite:
        return out

    url = lasco_c2_pb_url(obs_dt)
    ok = download_url(url, out, timeout=60.0)
    if not ok:
        if out.exists() and out.stat().st_size == 0:
            out.unlink()
        return None
    print(f"[OK] LASCO-C2 pB: {out}")
    return out

def find_lasco_c2_pb_files(lasco_raw_dir: Path, target_time: str | datetime, window_days: float) -> List[Path]:
    """Find local LASCO-C2 pB files within the requested time window."""
    lasco_raw_dir = Path(lasco_raw_dir).expanduser()
    if not lasco_raw_dir.exists():
        return []
    target_dt = parse_target_datetime(target_time)
    t0 = target_dt - timedelta(days=float(window_days))
    t1 = target_dt + timedelta(days=float(window_days))
    selected = []
    for path in lasco_raw_dir.glob("C2-PB-????????_????.fts"):
        obs_dt = parse_pb_filename_datetime(path)
        if obs_dt is not None and t0 <= obs_dt <= t1:
            selected.append((obs_dt, path))
    selected.sort(key=lambda item: (item[0], item[1].name))
    return [path for _, path in selected]

def download_lasco_c2_pb_window(
    target_time: str | datetime,
    window_days: float,
    lasco_raw_dir: Path,
    overwrite: bool = False,
    fallback_hhmm_list: Optional[List[str]] = None,
) -> List[Path]:
    """
    Download all discoverable LASCO-C2 pB files within target_time +/- window_days.

    The preferred mode is to read the NRL monthly directory listing. If the listing is not
    available, the fallback HHMM list is used.
    """
    target_dt = parse_target_datetime(target_time)
    t0 = target_dt - timedelta(days=float(window_days))
    t1 = target_dt + timedelta(days=float(window_days))

    months = sorted({(d.year, d.month) for d in iter_dates_in_window(target_dt, window_days)})
    remote_times = []
    for year, month in months:
        remote_times.extend(list_remote_lasco_c2_times_for_month(year, month))
    remote_times = sorted(set(t for t in remote_times if t0 <= t <= t1))

    if not remote_times:
        if fallback_hhmm_list is None:
            fallback_hhmm_list = ["0006", "0606", "1206", "1806"]
        print("[WARN] LASCO directory listing yielded no files; using fallback HHMM candidates.")
        for d in iter_dates_in_window(target_dt, window_days):
            for hhmm in fallback_hhmm_list:
                try:
                    cand = datetime.strptime(f"{d:%Y%m%d}_{hhmm}", "%Y%m%d_%H%M")
                except ValueError:
                    continue
                if t0 <= cand <= t1:
                    remote_times.append(cand)
        remote_times = sorted(set(remote_times))

    paths = []
    for obs_dt in remote_times:
        path = download_lasco_c2_pb(obs_dt, lasco_raw_dir=lasco_raw_dir, overwrite=overwrite)
        if path is not None:
            paths.append(path)
    return paths

def _decompress_gzip_file(gz_path: Path, output_path: Optional[Path] = None) -> Path:
    """Decompress a .gz file and return the decompressed path."""
    gz_path = Path(gz_path)
    if output_path is None:
        output_path = gz_path.with_suffix("")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(gz_path, "rb") as fin, output_path.open("wb") as fout:
        shutil.copyfileobj(fin, fout)
    return output_path

def normalize_kcor_download_payload(payload_path: Path, kcor_raw_dir: Path) -> List[Path]:
    """
    Normalize a K-Cor download payload into decompressed FITS files in kcor_raw_dir.

    HAO delivery may return a tar archive, a .fts.gz file, or occasionally a raw FITS file.
    Both K-Cor pB and pBavg filename patterns are accepted.
    """
    payload_path = Path(payload_path)
    kcor_raw_dir = Path(kcor_raw_dir)
    kcor_raw_dir.mkdir(parents=True, exist_ok=True)
    out_paths: List[Path] = []
    kcor_name_re = r"\d{8}_\d{6}_kcor_l2_pb(?:_?avg)?\.(?:fts|fits)(?:\.gz)?"

    try:
        if tarfile.is_tarfile(payload_path):
            with tarfile.open(payload_path, "r:*") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    name = Path(member.name).name
                    if not re.fullmatch(kcor_name_re, name):
                        continue
                    extracted = kcor_raw_dir / name
                    with tar.extractfile(member) as fin, extracted.open("wb") as fout:
                        if fin is not None:
                            shutil.copyfileobj(fin, fout)
                    if extracted.suffix == ".gz":
                        out_paths.append(_decompress_gzip_file(extracted))
                    else:
                        out_paths.append(extracted)
            return out_paths
    except tarfile.TarError as exc:
        print(f"[WARN] Could not read K-Cor tar payload {payload_path}: {exc}")

    name = payload_path.name
    if re.fullmatch(r"\d{8}_\d{6}_kcor_l2_pb(?:_?avg)?\.(?:fts|fits)\.gz", name):
        copied = kcor_raw_dir / name
        if payload_path != copied:
            shutil.copy2(payload_path, copied)
        out_paths.append(_decompress_gzip_file(copied))
    elif re.fullmatch(r"\d{8}_\d{6}_kcor_l2_pb(?:_?avg)?\.(?:fts|fits)", name):
        copied = kcor_raw_dir / name
        if payload_path != copied:
            shutil.copy2(payload_path, copied)
        out_paths.append(copied)
    else:
        # Last-resort naming for direct raw response.
        guess = kcor_raw_dir / payload_path.name.replace(".download", ".fts")
        shutil.copy2(payload_path, guess)
        out_paths.append(guess)

    return out_paths

def download_kcor_pb(
    obs_dt: datetime,
    kcor_raw_dir: Path,
    overwrite: bool = False,
    source_url: Optional[str] = None,
    cookie_file: Optional[Path] = None,
    kcor_product: str = "pbavg",
) -> List[Path]:
    """Download one K-Cor level-2 pB/pBavg file. Missing/auth-blocked files are skipped."""
    product = str(kcor_product).strip().lower()
    if product not in ("pb", "pbavg"):
        raise ValueError(f"Unsupported K-Cor product: {kcor_product!r}. Use 'pb' or 'pbavg'.")

    kcor_raw_dir = Path(kcor_raw_dir).expanduser()
    file_product = "pb_avg" if product == "pbavg" else product
    out = kcor_raw_dir / f"{obs_dt:%Y%m%d}_{obs_dt:%H%M%S}_kcor_l2_{file_product}.fts"
    if out.exists() and not overwrite:
        return [out]

    # Backward-compatible check: older code may have saved pbavg without underscore.
    if product == "pbavg":
        legacy_out = kcor_raw_dir / f"{obs_dt:%Y%m%d}_{obs_dt:%H%M%S}_kcor_l2_pbavg.fts"
        if legacy_out.exists() and not overwrite:
            return [legacy_out]

    tmp = kcor_raw_dir / f"{obs_dt:%Y%m%d}_{obs_dt:%H%M%S}_kcor_l2_{file_product}.download"
    url = source_url if source_url else kcor_pb_url(obs_dt, kcor_product=product)
    referer = kcor_listing_url(obs_dt, kcor_product=product)
    ok = download_url(url, tmp, referer=referer, timeout=60.0, cookie_file=cookie_file)
    if not ok:
        if tmp.exists():
            tmp.unlink()
        return []

    paths = normalize_kcor_download_payload(tmp, kcor_raw_dir)
    try:
        tmp.unlink()
    except OSError:
        pass

    valid = []
    for path in paths:
        dt = parse_pb_filename_datetime(path)
        if dt is not None:
            valid.append(path)
            print(f"[OK] K-Cor {product}: {path}")
    return valid

def find_kcor_pb_files(
    kcor_raw_dir: Path,
    target_time: str | datetime,
    window_days: float,
    kcor_product: str = "pbavg",
) -> List[Path]:
    """Find local K-Cor pB/pBavg FITS files within the requested time window."""
    kcor_raw_dir = Path(kcor_raw_dir).expanduser()
    if not kcor_raw_dir.exists():
        return []

    product = str(kcor_product).strip().lower()
    if product not in ("pb", "pbavg", "both"):
        raise ValueError(f"Unsupported K-Cor product: {kcor_product!r}. Use 'pb', 'pbavg', or 'both'.")

    target_dt = parse_target_datetime(target_time)
    t0 = target_dt - timedelta(days=float(window_days))
    t1 = target_dt + timedelta(days=float(window_days))
    selected = []

    products = ("pb", "pbavg") if product == "both" else (product,)
    patterns = []
    for prod in products:
        suffixes = ("pbavg", "pb_avg") if prod == "pbavg" else (prod,)
        for suffix in suffixes:
            patterns.extend((
                f"????????_??????_kcor_l2_{suffix}.fts",
                f"????????_??????_kcor_l2_{suffix}.fits",
            ))

    for pat in patterns:
        for path in kcor_raw_dir.glob(pat):
            obs_dt = parse_pb_filename_datetime(path)
            if obs_dt is not None and t0 <= obs_dt <= t1:
                selected.append((obs_dt, path))
    selected.sort(key=lambda item: (item[0], item[1].name))
    return [path for _, path in selected]

def nearest_file_by_time(paths: List[Path], target_dt: datetime, max_delta_seconds: float) -> Optional[Path]:
    """Return the file with timestamp nearest to target_dt within max_delta_seconds."""
    best = None
    best_delta = None
    for path in paths:
        obs_dt = parse_pb_filename_datetime(path)
        if obs_dt is None:
            continue
        delta = abs((obs_dt - target_dt).total_seconds())
        if delta <= max_delta_seconds and (best_delta is None or delta < best_delta):
            best = path
            best_delta = delta
    return best

def download_kcor_near_lasco_times(
    lasco_paths: List[Path],
    kcor_raw_dir: Path,
    max_time_delta_minutes: float = 60.0,
    search_step_minutes: float = 5.0,
    overwrite: bool = False,
    cookie_file: Optional[Path] = None,
    kcor_product: str = "pbavg",
) -> List[Path]:
    """
    Download only K-Cor pB files close to LASCO-C2 pB times.

    The preferred mode is to query the MLSO daily listing and use actual available K-Cor
    timestamps. This avoids probing non-existent exact-minute URLs such as HHMM00, which
    often return an HTML page rather than a FITS/tar payload. If the MLSO listing cannot be
    parsed, the function falls back to local files and then to a sparse exact-time probe.
    """
    if not lasco_paths:
        return []

    kcor_raw_dir = Path(kcor_raw_dir).expanduser()
    kcor_raw_dir.mkdir(parents=True, exist_ok=True)

    lasco_times = [parse_pb_filename_datetime(p) for p in lasco_paths]
    lasco_times = [t for t in lasco_times if t is not None]
    if not lasco_times:
        return []

    t_mid = lasco_times[len(lasco_times) // 2]
    window_days = max(abs((t - t_mid).total_seconds()) for t in lasco_times) / 86400.0 + 1.0
    product = str(kcor_product).strip().lower()
    if product not in ("pb", "pbavg"):
        raise ValueError(f"Unsupported K-Cor product: {kcor_product!r}. Use 'pb' or 'pbavg'.")

    local_kcor = find_kcor_pb_files(kcor_raw_dir, t_mid, window_days=window_days, kcor_product=product)

    max_delta = float(max_time_delta_minutes) * 60.0
    remote_cache: dict[datetime, List[Tuple[datetime, str]]] = {}
    remote_auth_blocked = False
    warned_remote_auth_blocked = False

    def _day_key(dt: datetime) -> datetime:
        return datetime(dt.year, dt.month, dt.day)

    def _remote_links_for_day(dt: datetime) -> List[Tuple[datetime, str]]:
        key = _day_key(dt)
        if key not in remote_cache:
            remote_cache[key] = list_remote_kcor_pb_links_for_day(key, kcor_product=product)
        return remote_cache[key]

    def _nearest_remote_link(target_dt: datetime) -> Optional[Tuple[datetime, str]]:
        links: List[Tuple[datetime, str]] = []
        # Include neighboring dates because the +/- window can cross midnight.
        for dd in (-1, 0, 1):
            links.extend(_remote_links_for_day(target_dt + timedelta(days=dd)))
        best = None
        best_delta = None
        for obs_dt, url in links:
            delta = abs((obs_dt - target_dt).total_seconds())
            if delta <= max_delta and (best_delta is None or delta < best_delta):
                best = (obs_dt, url)
                best_delta = delta
        return best

    # First use local files; then use actual remote-listing times; only then fall back to exact probes.
    for lasco_dt in lasco_times:
        if nearest_file_by_time(local_kcor, lasco_dt, max_delta) is not None:
            continue

        if remote_auth_blocked:
            if not warned_remote_auth_blocked:
                print(
                    "[AUTH] K-Cor remote download is blocked by the HAO registration gateway. "
                    "Skipping remaining K-Cor remote attempts and using local K-Cor files only. "
                    "Set KCOR_COOKIE_FILE to a valid browser-exported cookie file, or manually place "
                    "K-Cor FITS files in KCOR_RAW_DIR."
                )
                warned_remote_auth_blocked = True
            continue

        remote = _nearest_remote_link(lasco_dt)
        if remote is not None:
            obs_dt, source_url = remote
            new_paths = download_kcor_pb(
                obs_dt,
                kcor_raw_dir=kcor_raw_dir,
                overwrite=overwrite,
                source_url=source_url,
                cookie_file=cookie_file,
                kcor_product=product,
            )
            if new_paths:
                local_kcor.extend(new_paths)
                continue
            # A valid remote listing was found, but HAO returned HTML instead of data.
            # Do not probe many synthetic timestamps; that only repeats the same gateway response.
            if LAST_DOWNLOAD_AUTH_BLOCKED:
                remote_auth_blocked = True
                print(
                    f"[AUTH] K-Cor listed near LASCO time {lasco_dt}, but HAO returned a registration/auth HTML page. "
                    "Remote K-Cor download will be disabled for the rest of this run."
                )
            else:
                print(
                    f"[SKIP] K-Cor listed near LASCO time {lasco_dt}, but automatic download failed. "
                    "Use a valid HAO cookie file or place the K-Cor FITS in KCOR_RAW_DIR."
                )
            continue

        # Fallback: exact-time sparse probing. This is intentionally used only when the daily listing
        # could not provide any nearby candidate.
        step = max(1, int(round(float(search_step_minutes))))
        start = lasco_dt - timedelta(minutes=float(max_time_delta_minutes))
        end = lasco_dt + timedelta(minutes=float(max_time_delta_minutes))
        candidates = []
        nstep = int(np.floor(float(max_time_delta_minutes) / float(step)))
        for k in range(nstep + 1):
            for sign in ((0,) if k == 0 else (-1, 1)):
                cand = lasco_dt + timedelta(minutes=float(sign * k * step))
                cand = cand.replace(second=0, microsecond=0)
                if start <= cand <= end:
                    candidates.append(cand)
        candidates = sorted(set(candidates), key=lambda t: abs((t - lasco_dt).total_seconds()))

        for cand in candidates:
            new_paths = download_kcor_pb(
                cand,
                kcor_raw_dir=kcor_raw_dir,
                overwrite=overwrite,
                cookie_file=cookie_file,
                kcor_product=product,
            )
            if new_paths:
                local_kcor.extend(new_paths)
                break
            if LAST_DOWNLOAD_AUTH_BLOCKED:
                remote_auth_blocked = True
                print(
                    "[AUTH] HAO registration/auth HTML was returned during fallback probing. "
                    "Stopping further K-Cor remote attempts."
                )
                break

        if nearest_file_by_time(local_kcor, lasco_dt, max_delta) is None:
            print(f"[SKIP] No K-Cor pB found within +/-{max_time_delta_minutes:g} min of LASCO time {lasco_dt}.")

    # Return unique K-Cor paths that are actually close to at least one LASCO pB time.
    selected = []
    seen = set()
    for lasco_dt in lasco_times:
        kcor = nearest_file_by_time(local_kcor, lasco_dt, max_delta)
        if kcor is not None and kcor not in seen:
            selected.append(kcor)
            seen.add(kcor)
    selected.sort(key=lambda p: parse_pb_filename_datetime(p) or datetime.min)
    return selected


# ----------------------------
# K-Cor/LASCO pB combination helpers
# ----------------------------
def _wcs_unit_to_arcsec(values: np.ndarray, unit: str) -> np.ndarray:
    unit = str(unit).lower()
    if "deg" in unit:
        return values * 3600.0
    return values

def _arcsec_to_wcs_unit(values: np.ndarray, unit: str) -> np.ndarray:
    unit = str(unit).lower()
    if "deg" in unit:
        return values / 3600.0
    return values

def sample_image_bilinear_safe(
    img: np.ndarray,
    xpix: np.ndarray,
    ypix: np.ndarray,
) -> np.ndarray:
    """
    Bilinearly sample an image at floating-point pixel coordinates.

    Points outside the source image become NaN.

    When only some of the four bilinear neighbours are finite, the finite
    interpolation weights are renormalized. This prevents one NaN neighbour
    from unnecessarily turning a valid boundary pixel into NaN.

    At a general sub-pixel position, at least two finite contributing
    neighbours are required. At an exact source-pixel position, one finite
    neighbour is sufficient.
    """
    img = np.asarray(img, dtype=np.float64)
    xpix = np.asarray(xpix, dtype=np.float64)
    ypix = np.asarray(ypix, dtype=np.float64)

    xpix, ypix = np.broadcast_arrays(xpix, ypix)
    out = np.full(xpix.shape, np.nan, dtype=np.float64)

    finite_coord = (
        np.isfinite(xpix)
        & np.isfinite(ypix)
    )
    if not np.any(finite_coord):
        return out

    flat_coord = np.flatnonzero(finite_coord)

    x = xpix.ravel()[flat_coord]
    y = ypix.ravel()[flat_coord]

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    inside = (
        (x0 >= 0)
        & (y0 >= 0)
        & (x1 < img.shape[1])
        & (y1 < img.shape[0])
    )
    if not np.any(inside):
        return out

    flat_out = flat_coord[inside]

    x = x[inside]
    y = y[inside]
    x0 = x0[inside]
    y0 = y0[inside]
    x1 = x1[inside]
    y1 = y1[inside]

    wx = x - x0
    wy = y - y0

    values = np.stack(
        [
            img[y0, x0],
            img[y0, x1],
            img[y1, x0],
            img[y1, x1],
        ],
        axis=0,
    )

    weights = np.stack(
        [
            (1.0 - wx) * (1.0 - wy),
            wx * (1.0 - wy),
            (1.0 - wx) * wy,
            wx * wy,
        ],
        axis=0,
    )

    valid_value = np.isfinite(values)

    finite_weights = np.where(
        valid_value,
        weights,
        0.0,
    )

    contributing = (
        valid_value
        & (weights > 1e-12)
    )

    n_contributing = np.count_nonzero(
        contributing,
        axis=0,
    )

    weight_sum = np.sum(
        finite_weights,
        axis=0,
    )

    # At an exact integer pixel position, one neighbour can carry almost all
    # the weight and should be accepted.
    exact_single_pixel = (
        np.max(finite_weights, axis=0)
        >= 1.0 - 1e-12
    )

    usable = (
        (weight_sum > 1e-12)
        & (
            (n_contributing >= 2)
            | exact_single_pixel
        )
    )

    sampled = np.full(
        x.shape,
        np.nan,
        dtype=np.float64,
    )

    if np.any(usable):
        numerator = np.sum(
            np.where(
                valid_value,
                values,
                0.0,
            )
            * finite_weights,
            axis=0,
        )

        sampled[usable] = (
            numerator[usable]
            / weight_sum[usable]
        )

    out.ravel()[flat_out] = sampled

    return out

def _header_unit_to_arcsec_scale(unit: str) -> float:
    """Return multiplicative scale from a FITS WCS angular unit to arcsec."""
    unit = str(unit or "").strip().lower()
    if "deg" in unit:
        return 3600.0
    if "arcmin" in unit:
        return 60.0
    # Default for LASCO/K-Cor headers used here is arcsec or ARCSEC.
    return 1.0


def _linear_wcs_matrix_arcsec_per_pixel(hdr: fits.Header) -> np.ndarray:
    """Return the 2x2 linear FITS WCS matrix in arcsec per pixel.

    The matrix maps 0-based numpy pixel offsets measured from CRPIX to
    helioprojective-like world-coordinate offsets.  It honors, in order,
    a full CD matrix, a PC matrix with CDELT, and the legacy CROTA keywords.
    This avoids a subtle radial/rotational mismatch when LASCO headers carry
    CROTA but not complete HPLN/HPLT CTYPE keywords.
    """
    cdelt1 = float(hdr.get("CDELT1", 1.0))
    cdelt2 = float(hdr.get("CDELT2", 1.0))

    if all(k in hdr for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")):
        mat = np.array(
            [[float(hdr["CD1_1"]), float(hdr["CD1_2"])],
             [float(hdr["CD2_1"]), float(hdr["CD2_2"])]],
            dtype=np.float64,
        )
    elif any(k in hdr for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2")):
        pc11 = float(hdr.get("PC1_1", 1.0))
        pc12 = float(hdr.get("PC1_2", 0.0))
        pc21 = float(hdr.get("PC2_1", 0.0))
        pc22 = float(hdr.get("PC2_2", 1.0))
        mat = np.array(
            [[pc11 * cdelt1, pc12 * cdelt2],
             [pc21 * cdelt1, pc22 * cdelt2]],
            dtype=np.float64,
        )
    elif "CROTA2" in hdr or "CROTA1" in hdr:
        theta = np.deg2rad(float(hdr.get("CROTA2", hdr.get("CROTA1", 0.0))))
        c = np.cos(theta)
        s = np.sin(theta)
        mat = np.array(
            [[cdelt1 * c, -cdelt2 * s],
             [cdelt1 * s,  cdelt2 * c]],
            dtype=np.float64,
        )
    else:
        mat = np.array([[cdelt1, 0.0], [0.0, cdelt2]], dtype=np.float64)

    unit_scale = np.array(
        [_header_unit_to_arcsec_scale(str(hdr.get("CUNIT1", "arcsec"))),
         _header_unit_to_arcsec_scale(str(hdr.get("CUNIT2", "arcsec")))],
        dtype=np.float64,
    )
    return mat * unit_scale[:, None]


def _linear_pixel_to_arcsec(hdr: fits.Header, xpix: np.ndarray, ypix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback FITS-linear pixel -> helioprojective arcsec conversion.

    FITS CRPIX is 1-based, whereas numpy/astropy pixel coordinates are 0-based.
    This routine deliberately honors CD/PC/CROTA when present, because LASCO-C2
    pB headers can have only linear WCS keywords plus CROTA, while K-Cor headers
    usually have complete HPLN/HPLT WCS keywords.
    """
    crpix1 = float(hdr.get("CRPIX1", (int(hdr.get("NAXIS1", 1)) + 1) / 2.0))
    crpix2 = float(hdr.get("CRPIX2", (int(hdr.get("NAXIS2", 1)) + 1) / 2.0))
    crval1 = float(hdr.get("CRVAL1", 0.0)) * _header_unit_to_arcsec_scale(str(hdr.get("CUNIT1", "arcsec")))
    crval2 = float(hdr.get("CRVAL2", 0.0)) * _header_unit_to_arcsec_scale(str(hdr.get("CUNIT2", "arcsec")))

    dx = np.asarray(xpix, dtype=np.float64) + 1.0 - crpix1
    dy = np.asarray(ypix, dtype=np.float64) + 1.0 - crpix2
    mat = _linear_wcs_matrix_arcsec_per_pixel(hdr)
    xw = crval1 + mat[0, 0] * dx + mat[0, 1] * dy
    yw = crval2 + mat[1, 0] * dx + mat[1, 1] * dy
    return xw, yw


def _linear_arcsec_to_pixel(hdr: fits.Header, x_arcsec: np.ndarray, y_arcsec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback helioprojective arcsec -> FITS-linear pixel conversion."""
    crpix1 = float(hdr.get("CRPIX1", (int(hdr.get("NAXIS1", 1)) + 1) / 2.0))
    crpix2 = float(hdr.get("CRPIX2", (int(hdr.get("NAXIS2", 1)) + 1) / 2.0))
    crval1 = float(hdr.get("CRVAL1", 0.0)) * _header_unit_to_arcsec_scale(str(hdr.get("CUNIT1", "arcsec")))
    crval2 = float(hdr.get("CRVAL2", 0.0)) * _header_unit_to_arcsec_scale(str(hdr.get("CUNIT2", "arcsec")))

    rhs0 = np.asarray(x_arcsec, dtype=np.float64) - crval1
    rhs1 = np.asarray(y_arcsec, dtype=np.float64) - crval2
    mat = _linear_wcs_matrix_arcsec_per_pixel(hdr)
    try:
        inv = np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(mat)
    dx = inv[0, 0] * rhs0 + inv[0, 1] * rhs1
    dy = inv[1, 0] * rhs0 + inv[1, 1] * rhs1
    return dx + crpix1 - 1.0, dy + crpix2 - 1.0


def _has_complete_celestial_wcs(hdr: fits.Header) -> bool:
    """Return True if the header looks safe for astropy WCS celestial transforms."""
    ctype1 = str(hdr.get("CTYPE1", "")).upper()
    ctype2 = str(hdr.get("CTYPE2", "")).upper()
    return bool(ctype1 and ctype2 and ("HPL" in ctype1 or "RA" in ctype1) and ("HPL" in ctype2 or "DEC" in ctype2))


def reproject_image_to_header(src_img: np.ndarray, src_hdr: fits.Header, dst_hdr: fits.Header, dst_shape: Tuple[int, int]) -> np.ndarray:
    """Reproject src_img onto dst_hdr/dst_shape using WCS, with a robust FITS-linear fallback.

    The fallback is essential for LASCO pB files that contain CRPIX/CDELT/CRVAL/CUNIT but
    may not contain complete HPLN/HPLT CTYPE keywords.  Without this fallback, astropy WCS
    can silently return pixel-like coordinates and the K-Cor image is sampled outside its
    valid field, producing an output that is accidentally identical to the LASCO input.
    """
    yy, xx = np.mgrid[0:dst_shape[0], 0:dst_shape[1]]

    use_astropy_wcs = _has_complete_celestial_wcs(src_hdr) and _has_complete_celestial_wcs(dst_hdr)
    if use_astropy_wcs:
        try:
            w_dst = WCS(dst_hdr)
            w_src = WCS(src_hdr)
            xw_dst, yw_dst = w_dst.pixel_to_world_values(xx, yy)
            dst_u1 = str(dst_hdr.get("CUNIT1", getattr(w_dst.wcs, "cunit", ["", ""])[0])).lower()
            dst_u2 = str(dst_hdr.get("CUNIT2", getattr(w_dst.wcs, "cunit", ["", ""])[1])).lower()
            src_u1 = str(src_hdr.get("CUNIT1", getattr(w_src.wcs, "cunit", ["", ""])[0])).lower()
            src_u2 = str(src_hdr.get("CUNIT2", getattr(w_src.wcs, "cunit", ["", ""])[1])).lower()

            x_arc = _wcs_unit_to_arcsec(np.asarray(xw_dst, dtype=np.float64), dst_u1)
            y_arc = _wcs_unit_to_arcsec(np.asarray(yw_dst, dtype=np.float64), dst_u2)
            x_src_world = _arcsec_to_wcs_unit(x_arc, src_u1)
            y_src_world = _arcsec_to_wcs_unit(y_arc, src_u2)
            x_src_pix, y_src_pix = w_src.world_to_pixel_values(x_src_world, y_src_world)
            out = sample_image_bilinear_safe(src_img, x_src_pix, y_src_pix)
            if np.count_nonzero(np.isfinite(out)) > 0:
                return out
            print("[WARN] Astropy-WCS reprojection produced no finite K-Cor pixels; falling back to linear CRPIX/CDELT mapping.")
        except Exception as exc:
            print(f"[WARN] Astropy-WCS reprojection failed ({exc}); falling back to linear CRPIX/CDELT mapping.")

    x_arc, y_arc = _linear_pixel_to_arcsec(dst_hdr, xx, yy)
    x_src_pix, y_src_pix = _linear_arcsec_to_pixel(src_hdr, x_arc, y_arc)
    return sample_image_bilinear_safe(src_img, x_src_pix, y_src_pix)


def _pb_unit_kind(hdr: fits.Header) -> str:
    """Classify pB units for sanity checks."""
    bunit = str(hdr.get("BUNIT", "")).strip().lower()
    if bunit in ("dn", "adu", "counts") or "dn" == bunit:
        return "raw_dn"
    if "msb" in bunit or "mean solar brightness" in bunit or bunit == "pb":
        return "msb_like"
    return "unknown"


def _robust_positive_scale(reference: np.ndarray, target: np.ndarray, mask: np.ndarray, min_pixels: int = 100) -> Tuple[float, int]:
    """Return median scale s so that s*target approximately matches reference over mask."""
    m = mask & np.isfinite(reference) & np.isfinite(target) & (reference > 0) & (target > 0)
    n = int(np.count_nonzero(m))
    if n < int(min_pixels):
        return 1.0, n
    ratio = np.asarray(reference[m] / target[m], dtype=np.float64)
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if ratio.size < int(min_pixels):
        return 1.0, int(ratio.size)
    lo, hi = np.nanpercentile(ratio, [5.0, 95.0])
    clipped = ratio[(ratio >= lo) & (ratio <= hi)]
    if clipped.size < int(min_pixels):
        clipped = ratio
    scale = float(np.nanmedian(clipped))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return scale, int(clipped.size)

def read_fits_image(path: Path) -> Tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float64)
        hdr = hdul[0].header
    return data, hdr


def _first_finite_header_float(hdr: fits.Header, keys: Tuple[str, ...]) -> Optional[float]:
    """Return the first finite floating-point value found in a FITS header."""
    for key in keys:
        if key not in hdr:
            continue
        try:
            val = float(hdr[key])
        except Exception:
            continue
        if np.isfinite(val):
            return val
    return None


def _parse_header_datetime(hdr: fits.Header, fallback: Optional[datetime] = None) -> Optional[datetime]:
    """Parse an observation datetime from common FITS time keywords."""
    for key in ("DATE-AVG", "DATE-OBS", "DATE-BEG", "DATE-END", "DATE"):
        if key not in hdr:
            continue
        text = str(hdr[key]).strip()
        if not text:
            continue
        text = text.replace("Z", "")
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            pass
        for fmt in ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y%m%d_%H%M%S", "%Y%m%d_%H%M"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                pass
    return fallback


def _earth_observer_carrington_lonlat_sunpy(obs_dt: Optional[datetime]) -> Optional[Tuple[float, float]]:
    """
    Compute Earth/SOHO/K-Cor observer Carrington lon/lat using SunPy when available.

    This is intentionally optional.  If SunPy is not installed, the preparation script
    still works and simply relies on observer-coordinate keywords already present in
    the input FITS headers.
    """
    if obs_dt is None:
        return None

    try:
        import astropy.units as u
        from astropy.time import Time
        from sunpy.coordinates import frames

        try:
            from sunpy.coordinates import get_body_heliographic_stonyhurst
        except Exception:
            from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst

        obstime = Time(obs_dt)
        earth_hgs = get_body_heliographic_stonyhurst("earth", obstime)
        earth_hgc = earth_hgs.transform_to(frames.HeliographicCarrington(observer="earth", obstime=obstime))
        lon = float(earth_hgc.lon.to_value(u.deg)) % 360.0
        lat = float(earth_hgc.lat.to_value(u.deg))
        if np.isfinite(lon) and np.isfinite(lat):
            return lon, lat
    except Exception:
        return None
    return None


def add_observer_geometry_keywords(
    out_hdr: fits.Header,
    source_hdr: fits.Header,
    obs_dt: Optional[datetime] = None,
    source_label: str = "KCOR",
) -> None:
    """
    Add observer heliographic/Carrington coordinates to a tomography-ready FITS header.

    main_multi_tomo.py determines the LOS/camera direction from CRLN_OBS/CRLT_OBS
    first.  Therefore the combined K-Cor/LASCO products must carry these keywords;
    otherwise every Earth-view file falls back to DEFAULT_LONLAT=(0,0), even when
    +/- one week of data are supplied.

    The priority is:
      1) Carrington observer coordinates already present in the source header;
      2) common central-meridian Carrington keywords such as SOLAR_L0/L0;
      3) optional SunPy Earth ephemeris computed from the observation time.
    """
    label = str(source_label).strip().upper()[:4] or "SRC"
    obs_dt = _parse_header_datetime(source_hdr, fallback=obs_dt)

    # Carrington observer longitude/latitude candidates.
    # SOLAR_L0/L0 are commonly used for the Carrington longitude of disk center,
    # which is the sub-observer Carrington longitude for Earth-view coronagraphs.
    carr_lon = _first_finite_header_float(
        source_hdr,
        (
            "CRLN_OBS", "CRLN", "SOLAR_L0", "OBS_L0", "L0",
            "CARR_LON", "CARRLONG", "CMLON", "CMLON_OB",
        ),
    )
    carr_lat = _first_finite_header_float(
        source_hdr,
        (
            "CRLT_OBS", "CRLT", "SOLAR_B0", "OBS_B0", "B0",
            "CARR_LAT", "CARRLAT", "CMLAT", "CMLAT_OB",
        ),
    )

    source = "HEADER"
    if carr_lon is None or carr_lat is None:
        computed = _earth_observer_carrington_lonlat_sunpy(obs_dt)
        if computed is not None:
            carr_lon, carr_lat = computed
            source = "SUNPY"

    # Stonyhurst heliographic observer coordinates, if available.  These are kept
    # for traceability, but CRLN_OBS/CRLT_OBS remain the coordinates used by tomography.
    hgs_lon = _first_finite_header_float(source_hdr, ("HGLN_OBS", "HGLN", "HG_LON", "OBSLON"))
    hgs_lat = _first_finite_header_float(source_hdr, ("HGLT_OBS", "HGLT", "HG_LAT", "OBSLAT"))

    if carr_lon is not None and carr_lat is not None:
        carr_lon = float(carr_lon) % 360.0
        carr_lat = float(carr_lat)
        out_hdr["CRLN_OBS"] = (carr_lon, "Carrington observer longitude [deg]")
        out_hdr["CRLT_OBS"] = (carr_lat, "Carrington observer latitude [deg]")
        out_hdr[f"{label}CLON"] = (carr_lon, f"{label} Carrington obs lon [deg]")
        out_hdr[f"{label}CLAT"] = (carr_lat, f"{label} Carrington obs lat [deg]")
        out_hdr[f"{label}OSRC"] = (source, f"{label} observer coord source")

    if hgs_lon is not None and hgs_lat is not None:
        out_hdr["HGLN_OBS"] = (float(hgs_lon), "Stonyhurst observer longitude [deg]")
        out_hdr["HGLT_OBS"] = (float(hgs_lat), "Stonyhurst observer latitude [deg]")
        out_hdr[f"{label}HLON"] = (float(hgs_lon), f"{label} Stonyhurst obs lon [deg]")
        out_hdr[f"{label}HLAT"] = (float(hgs_lat), f"{label} Stonyhurst obs lat [deg]")

    if obs_dt is not None:
        out_hdr[f"{label}ODAT"] = (obs_dt.isoformat(timespec="seconds"), f"{label} observer time")

    if carr_lon is None or carr_lat is None:
        out_hdr["HISTORY"] = (
            f"WARNING: {label} observer Carrington lon/lat not found; "
            "CRLN_OBS/CRLT_OBS were not added."
        )

def xy_rsun_for_rebinned_image(
    hdr,
    orig_n: int,
    out_n: int,
    rsun_arcsec_override: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Return pixel-center coordinates on the rebinned image in solar-radius units.

    The radial masks used during K-COR/LASCO integration are calculated with
    the explicit FITS linear image-plane transformation:

        CRPIX / CRVAL / CDELT / CD / PC / CROTA

    This is intentionally the same local image-plane convention used by the
    plotting and tomography routines. It avoids celestial-longitude wrapping
    and small inconsistencies between TAN-projected and linearly plotted
    coordinates.
    """
    if (
        rsun_arcsec_override is not None
        and np.isfinite(float(rsun_arcsec_override))
        and float(rsun_arcsec_override) > 0
    ):
        rsun_arcsec = float(
            rsun_arcsec_override
        )
    else:
        rsun_arcsec = float(
            hdr.get(
                "RSUN_OBS",
                hdr.get("RSUN", 959.63),
            )
        )

    if (
        not np.isfinite(rsun_arcsec)
        or rsun_arcsec <= 0
    ):
        rsun_arcsec = 959.63

    orig_n = int(orig_n)
    out_n = int(out_n)

    if orig_n <= 0 or out_n <= 0:
        raise ValueError(
            "orig_n and out_n must be positive; "
            f"got orig_n={orig_n}, out_n={out_n}."
        )

    scale = (
        float(orig_n)
        / float(out_n)
    )

    # Pixel centers on the rebinned grid expressed in the original,
    # zero-based FITS pixel-coordinate system.
    yy, xx = np.mgrid[
        0:out_n,
        0:out_n,
    ]

    xpix = (
        (xx + 0.5) * scale
        - 0.5
    )
    ypix = (
        (yy + 0.5) * scale
        - 0.5
    )

    # Always use the local linear image-plane coordinates so that the radial
    # mask and the plotted guide circles use the same coordinate convention.
    x_arcsec, y_arcsec = _linear_pixel_to_arcsec(
        hdr,
        xpix,
        ypix,
    )

    x_map_rsun = (
        np.asarray(
            x_arcsec,
            dtype=np.float64,
        )
        / rsun_arcsec
    )
    y_map_rsun = (
        np.asarray(
            y_arcsec,
            dtype=np.float64,
        )
        / rsun_arcsec
    )

    return (
        x_map_rsun,
        y_map_rsun,
        float(rsun_arcsec),
    )



def _wcs_rotation_angle_deg(hdr: fits.Header) -> float:
    """Estimate the image-plane WCS rotation angle in degrees.

    The returned value is mainly diagnostic.  It is derived from CROTA first,
    then from the first column of the CD/PC/CDELT matrix.  A positive value
    means the input x-axis is rotated counter-clockwise in the projected
    helioprojective plane.
    """
    for key in ("CROTA2", "CROTA1"):
        if key in hdr:
            try:
                val = float(hdr[key])
            except Exception:
                continue
            if np.isfinite(val):
                return float(val)

    try:
        mat = _linear_wcs_matrix_arcsec_per_pixel(hdr)
        angle = np.degrees(np.arctan2(mat[1, 0], mat[0, 0]))
        if np.isfinite(angle):
            return float(angle)
    except Exception:
        pass
    return 0.0


def make_earth_aligned_output_header(src_hdr: fits.Header) -> fits.Header:
    """Return a rotation-free Earth-view output header based on src_hdr.

    LASCO-C2 pB headers can carry a small roll angle through CROTA, PC, or CD
    matrix keywords.  The previous merge kept the LASCO native roll in the
    output grid.  This routine keeps the same Sun center, image size, CDELT
    pixel scale, and apparent solar radius, but removes the in-plane rotation
    so that both LASCO-C2 and K-COR are resampled onto a common Earth-view
    projected x-y grid before merging.
    """
    out_hdr = src_hdr.copy()
    rot_deg = _wcs_rotation_angle_deg(src_hdr)

    try:
        mat_arcsec = _linear_wcs_matrix_arcsec_per_pixel(src_hdr)
        cdelt1_sign = np.sign(float(src_hdr.get("CDELT1", mat_arcsec[0, 0]))) or 1.0
        cdelt2_sign = np.sign(float(src_hdr.get("CDELT2", mat_arcsec[1, 1]))) or 1.0
        cdelt1_arcsec = cdelt1_sign * float(np.hypot(mat_arcsec[0, 0], mat_arcsec[1, 0]))
        cdelt2_arcsec = cdelt2_sign * float(np.hypot(mat_arcsec[0, 1], mat_arcsec[1, 1]))
    except Exception:
        cdelt1_arcsec = float(src_hdr.get("CDELT1", 1.0)) * _header_unit_to_arcsec_scale(str(src_hdr.get("CUNIT1", "arcsec")))
        cdelt2_arcsec = float(src_hdr.get("CDELT2", 1.0)) * _header_unit_to_arcsec_scale(str(src_hdr.get("CUNIT2", "arcsec")))

    unit1_scale = _header_unit_to_arcsec_scale(str(src_hdr.get("CUNIT1", "arcsec")))
    unit2_scale = _header_unit_to_arcsec_scale(str(src_hdr.get("CUNIT2", "arcsec")))
    if not np.isfinite(unit1_scale) or unit1_scale <= 0:
        unit1_scale = 1.0
    if not np.isfinite(unit2_scale) or unit2_scale <= 0:
        unit2_scale = 1.0

    for key in (
        "CD1_1", "CD1_2", "CD2_1", "CD2_2",
        "PC1_1", "PC1_2", "PC2_1", "PC2_2",
        "CROTA1", "CROTA2",
    ):
        if key in out_hdr:
            del out_hdr[key]

    out_hdr["CDELT1"] = (float(cdelt1_arcsec) / unit1_scale, "Earth-aligned CDELT1")
    out_hdr["CDELT2"] = (float(cdelt2_arcsec) / unit2_scale, "Earth-aligned CDELT2")
    out_hdr["CROTA2"] = (0.0, "Earth-aligned image roll angle [deg]")
    out_hdr["WCSALIGN"] = ("EARTH", "Output grid derotated to Earth-view axes")
    out_hdr["DEROTATE"] = (True, "LASCO and K-COR reprojected to WCSALIGN grid")
    out_hdr["LASCOROT"] = (float(rot_deg), "Original LASCO WCS roll angle [deg]")
    return out_hdr

def combine_kcor_lasco_pair(
    lasco_path: Path,
    kcor_path: Path,
    output_dir: Path,
    blend_inner_rsun: float = 2.2,
    blend_outer_rsun: float = 2.5,
    overwrite: bool = False,
    calibrate_kcor_to_lasco: bool = True,
    min_kcor_used_pixels: int = 100,
    max_scale_factor: float = 20.0,
    kcor_use_min_rsun: float = 1.1,
    kcor_use_max_rsun: Optional[float] = None,
    scale_fallback_inner_rsun: float = 2.0,
    scale_fallback_outer_rsun: Optional[float] = None,
) -> Optional[Path]:
    """
    Combine one LASCO-C2 pB image with the nearest K-Cor pB image.

    The LASCO grid/header is used as the output grid. K-Cor is reprojected onto the LASCO
    grid and used preferentially at small heliocentric distances, with a linear blend across
    blend_inner_rsun..blend_outer_rsun. This creates the tomography-ready file:
      pB_Kcor_LASCO_axi_<YYYYMMDD>_<HHMM>.fits

    K-Cor is not allowed to overwrite arbitrary radii: only pixels within
    kcor_use_min_rsun..kcor_use_max_rsun are eligible. This makes the inner K-Cor
    use explicit (for example, 1.1 R_sun) and prevents accidental use of invalid
    occulted/edge pixels.

    At rho <= blend_inner_rsun, LASCO-C2 is never used in the combined image. A valid
    K-Cor value is used when available; otherwise the output pixel is written as NaN.

    This version is deliberately conservative: if the K-Cor reprojection contributes too
    few valid pixels, it does not write a misleading LASCO-identical "combined" file.
    """
    lasco_dt = parse_pb_filename_datetime(lasco_path)
    if lasco_dt is None:
        return None

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"pB_Kcor_LASCO_axi_{lasco_dt:%Y%m%d}_{lasco_dt:%H%M}.fits"
    if out.exists() and not overwrite:
        return out

    try:
        lasco_img, lasco_hdr = read_fits_image(Path(lasco_path))
        kcor_img, kcor_hdr = read_fits_image(Path(kcor_path))

        # LASCO-C2 pB headers used here often lack RSUN_OBS/RSUN, while K-Cor
        # has a reliable apparent solar radius at nearly the same Earth-view time.
        # Copy it before any Rsun-based radial mask or output-header creation.
        working_lasco_hdr = lasco_hdr.copy()
        rsun_for_grid = _first_finite_header_float(
            working_lasco_hdr,
            ("RSUN_OBS", "RSUN"),
        )
        rsun_source = "LASCO"

        if rsun_for_grid is None:
            rsun_for_grid = _first_finite_header_float(
                kcor_hdr,
                ("RSUN_OBS", "RSUN"),
            )
            rsun_source = "KCOR"

        if (
            rsun_for_grid is not None
            and np.isfinite(rsun_for_grid)
            and rsun_for_grid > 0
        ):
            working_lasco_hdr["RSUN_OBS"] = (
                float(rsun_for_grid),
                "Apparent solar radius [arcsec]",
            )
            working_lasco_hdr["RSUN"] = (
                float(rsun_for_grid),
                "Apparent solar radius [arcsec]",
            )

        # Build a rotation-free Earth-view output grid. LASCO-C2 and K-COR are
        # both reprojected to this same grid before scaling/blending.
        lasco_input_hdr = working_lasco_hdr.copy()
        lasco_input_img = np.array(lasco_img, dtype=np.float64, copy=True)

        earth_aligned_hdr = make_earth_aligned_output_header(lasco_input_hdr)

        lasco_img = reproject_image_to_header(
            lasco_input_img,
            lasco_input_hdr,
            earth_aligned_hdr,
            lasco_input_img.shape,
        )
        kcor_on_lasco = reproject_image_to_header(
            kcor_img,
            kcor_hdr,
            earth_aligned_hdr,
            lasco_input_img.shape,
        )
        working_lasco_hdr = earth_aligned_hdr

    except Exception as exc:
        print(
            f"[SKIP] Could not combine {lasco_path.name} "
            f"with {kcor_path.name}: {exc}"
        )
        return None

    lasco_unit = _pb_unit_kind(lasco_hdr)
    kcor_unit = _pb_unit_kind(kcor_hdr)

    if lasco_unit == "raw_dn" or kcor_unit == "raw_dn":
        print(
            f"[SKIP] Raw DN input detected "
            f"(LASCO={lasco_hdr.get('BUNIT')!r}, "
            f"KCOR={kcor_hdr.get('BUNIT')!r}); "
            "refusing to write tomography pB product."
        )
        return None

    if lasco_unit == "unknown" or kcor_unit == "unknown":
        print(
            f"[WARN] Unknown pB unit convention "
            f"(LASCO BUNIT={lasco_hdr.get('BUNIT')!r}, "
            f"KCOR BUNIT={kcor_hdr.get('BUNIT')!r}); "
            "proceeding but check absolute density scale."
        )

    x_map, y_map, rsun_arcsec_for_grid = xy_rsun_for_rebinned_image(
        working_lasco_hdr,
        orig_n=lasco_img.shape[0],
        out_n=lasco_img.shape[0],
        rsun_arcsec_override=rsun_for_grid,
    )
    rho = np.hypot(x_map, y_map)

    kcor_rmin = float(kcor_use_min_rsun)
    kcor_rmax = (
        float(kcor_use_max_rsun)
        if kcor_use_max_rsun is not None
        else float(blend_outer_rsun)
    )

    if not np.isfinite(kcor_rmin) or kcor_rmin < 0:
        kcor_rmin = 0.0

    if not np.isfinite(kcor_rmax) or kcor_rmax <= kcor_rmin:
        kcor_rmax = float(blend_outer_rsun)

    kcor_radius_mask = (
        (rho >= kcor_rmin)
        & (rho <= kcor_rmax)
    )

    has_kcor_any_radius = (
        np.isfinite(kcor_on_lasco)
        & (kcor_on_lasco > 0)
    )
    has_kcor_raw = (
        has_kcor_any_radius
        & kcor_radius_mask
    )
    has_lasco = (
        np.isfinite(lasco_img)
        & (lasco_img > 0)
    )

    finite_kcor = int(np.count_nonzero(has_kcor_raw))
    finite_kcor_all = int(np.count_nonzero(has_kcor_any_radius))

    n_kcor_11_15 = int(
        np.count_nonzero(
            has_kcor_raw
            & (rho >= 1.0)
            & (rho < 1.5)
        )
    )
    n_kcor_15_22 = int(
        np.count_nonzero(
            has_kcor_raw
            & (rho >= 1.5)
            & (rho <= blend_inner_rsun)
        )
    )
    n_kcor_blend = int(
        np.count_nonzero(
            has_kcor_raw
            & (rho > blend_inner_rsun)
            & (rho < blend_outer_rsun)
        )
    )

    print(
        f"[QC] K-Cor reprojected onto LASCO grid: "
        f"positive finite pixels={finite_kcor_all}, "
        f"eligible pixels={finite_kcor} within "
        f"{kcor_rmin:.3f}..{kcor_rmax:.3f} Rsun, "
        f"LASCO positive finite pixels="
        f"{int(np.count_nonzero(has_lasco))}"
    )
    print(
        f"[QC] K-Cor eligible pixels by radius: "
        f"1.0..1.5 Rs={n_kcor_11_15}, "
        f"1.5..{blend_inner_rsun:.2f} Rs={n_kcor_15_22}, "
        f"{blend_inner_rsun:.2f}..{blend_outer_rsun:.2f} Rs="
        f"{n_kcor_blend}"
    )

    if finite_kcor < int(min_kcor_used_pixels):
        print(
            f"[SKIP] K-Cor reprojection contributed only "
            f"{finite_kcor} valid positive pixels "
            f"(< min_kcor_used_pixels={min_kcor_used_pixels}). "
            "Not writing misleading combined file."
        )
        return None

    kcor_scale = 1.0
    scale_npix = 0
    scale_method = "NONE"

    if bool(calibrate_kcor_to_lasco):
        min_scale_pixels = max(
            20,
            int(min_kcor_used_pixels) // 2,
        )

        scale_mask = (
            has_kcor_raw
            & has_lasco
            & (rho >= blend_inner_rsun)
            & (rho <= blend_outer_rsun)
        )

        kcor_scale, scale_npix = _robust_positive_scale(
            lasco_img,
            kcor_on_lasco,
            scale_mask,
            min_pixels=min_scale_pixels,
        )

        if scale_npix >= min_scale_pixels:
            scale_method = "BLEND"
        else:
            fb_inner = float(scale_fallback_inner_rsun)
            fb_outer = (
                float(scale_fallback_outer_rsun)
                if scale_fallback_outer_rsun is not None
                else float(blend_inner_rsun)
            )

            if (
                np.isfinite(fb_inner)
                and np.isfinite(fb_outer)
                and fb_outer > fb_inner
            ):
                fallback_mask = (
                    has_kcor_raw
                    & has_lasco
                    & (rho >= fb_inner)
                    & (rho <= fb_outer)
                )

                fb_scale, fb_npix = _robust_positive_scale(
                    lasco_img,
                    kcor_on_lasco,
                    fallback_mask,
                    min_pixels=min_scale_pixels,
                )

                if fb_npix >= min_scale_pixels:
                    kcor_scale = fb_scale
                    scale_npix = fb_npix
                    scale_method = "FALLBACK"

                    print(
                        "[WARN] No usable K-Cor/LASCO pixels in the "
                        "nominal blend annulus; using fallback scale "
                        f"annulus {fb_inner:.3f}..{fb_outer:.3f} Rsun."
                    )

        if (
            kcor_scale > float(max_scale_factor)
            or kcor_scale < 1.0 / float(max_scale_factor)
        ):
            print(
                f"[WARN] K-Cor/LASCO overlap scale={kcor_scale:.4g} "
                "is outside allowed range "
                f"[1/{max_scale_factor:g}, {max_scale_factor:g}]. "
                "Using scale=1.0."
            )
            kcor_scale = 1.0
            scale_method = "NONE"

        print(
            f"[QC] K-Cor-to-LASCO scale={kcor_scale:.6g} "
            f"(method={scale_method}, pixels={scale_npix})"
        )

    kcor_scaled = kcor_on_lasco * kcor_scale

    has_kcor = (
        np.isfinite(kcor_scaled)
        & (kcor_scaled > 0)
        & kcor_radius_mask
    )

    combined = np.array(
        lasco_img,
        dtype=np.float64,
        copy=True,
    )

    # ---------------------------------------------------------
    # Inner region: rho <= BLENDIN
    #
    # LASCO-C2 is never used in this region.
    #
    # Valid K-COR exists:
    #     combined = scaled K-COR
    #
    # Valid K-COR does not exist:
    #     combined = NaN
    # ---------------------------------------------------------
    c2_excluded_inner = (
        np.isfinite(rho)
        & (rho <= blend_inner_rsun)
    )

    # First remove all LASCO-C2 values inside BLENDIN.
    combined[c2_excluded_inner] = np.nan

    # Restore only valid K-COR values inside BLENDIN.
    inner = (
        has_kcor
        & c2_excluded_inner
    )
    combined[inner] = kcor_scaled[inner]

    # Explicitly retain NaN where no valid K-COR value exists.
    inner_missing_kcor = (
        c2_excluded_inner
        & (~has_kcor)
    )
    combined[inner_missing_kcor] = np.nan

    # ---------------------------------------------------------
    # Blend region: BLENDIN < rho < BLENDOUT
    # ---------------------------------------------------------
    overlap = (
        has_kcor
        & has_lasco
        & (rho > blend_inner_rsun)
        & (rho < blend_outer_rsun)
    )

    if np.any(overlap):
        alpha = (
            rho[overlap] - blend_inner_rsun
        ) / max(
            1e-6,
            blend_outer_rsun - blend_inner_rsun,
        )

        combined[overlap] = (
            (1.0 - alpha) * kcor_scaled[overlap]
            + alpha * lasco_img[overlap]
        )

    # Fill LASCO holes only inside the explicitly allowed K-Cor radial domain.
    fill_lasco_holes = (
        has_kcor
        & (~has_lasco)
    )
    combined[fill_lasco_holes] = kcor_scaled[fill_lasco_holes]

    inner_no_data = (
        np.isfinite(rho)
        & (rho < kcor_rmin)
    )
    bad_geometry = ~np.isfinite(rho)

    combined[
        inner_no_data
        | bad_geometry
    ] = np.nan

    kcor_used = (
        inner
        | overlap
        | fill_lasco_holes
    )

    n_inner_no_data = int(
        np.count_nonzero(inner_no_data)
    )
    n_bad_geometry = int(
        np.count_nonzero(bad_geometry)
    )
    n_c2_inner_excluded = int(
        np.count_nonzero(
            c2_excluded_inner
            & has_lasco
        )
    )
    n_inner_missing_kcor = int(
        np.count_nonzero(inner_missing_kcor)
    )
    n_used = int(
        np.count_nonzero(kcor_used)
    )
    n_inner_used = int(
        np.count_nonzero(inner)
    )
    n_blend_used = int(
        np.count_nonzero(overlap)
    )
    n_hole_used = int(
        np.count_nonzero(fill_lasco_holes)
    )

    finite_difference = np.isfinite(
        combined - lasco_img
    )

    max_diff = (
        float(
            np.nanmax(
                np.abs(combined - lasco_img)
            )
        )
        if np.any(finite_difference)
        else 0.0
    )

    if (
        n_used < int(min_kcor_used_pixels)
        or not np.isfinite(max_diff)
        or max_diff <= 0
    ):
        print(
            "[SKIP] Combined image is effectively LASCO-only "
            f"(K-Cor used pixels={n_used}, "
            f"max_diff={max_diff:.3e}). "
            "Not writing pB_Kcor_LASCO_axi product."
        )
        return None

    hdr = working_lasco_hdr.copy()

    # Add K-Cor/Earth-view observer coordinates needed by main_multi_tomo.py.
    add_observer_geometry_keywords(
        out_hdr=hdr,
        source_hdr=kcor_hdr,
        obs_dt=(
            parse_pb_filename_datetime(kcor_path)
            or lasco_dt
        ),
        source_label="KCOR",
    )

    # Ensure that downstream tomography/plotting uses the same apparent solar
    # radius as the radial masks above.
    if (
        rsun_for_grid is not None
        and np.isfinite(rsun_for_grid)
        and rsun_for_grid > 0
    ):
        hdr["RSUN_OBS"] = (
            float(rsun_for_grid),
            "Apparent solar radius [arcsec]",
        )
        hdr["RSUN"] = (
            float(rsun_for_grid),
            "Apparent solar radius [arcsec]",
        )
        hdr["KCORRSUN"] = (
            float(
                _first_finite_header_float(
                    kcor_hdr,
                    ("RSUN_OBS", "RSUN"),
                )
                or rsun_for_grid
            ),
            "K-Cor apparent solar radius [arcsec]",
        )
        hdr["RSUNSRC"] = (
            str(rsun_source)[:8],
            "Source of RSUN_OBS used for output grid",
        )

    hdr["HISTORY"] = (
        f"Combined LASCO-C2 pB with K-Cor pB: "
        f"{Path(kcor_path).name}"
    )
    hdr["KCORFILE"] = Path(kcor_path).name[:68]
    hdr["LASCOPB"] = Path(lasco_path).name[:68]
    hdr["BLENDIN"] = float(blend_inner_rsun)
    hdr["BLENDOUT"] = float(blend_outer_rsun)

    hdr["KCORSCAL"] = (
        float(kcor_scale),
        "Multiplicative scale applied to reprojected K-Cor pB",
    )
    hdr["KCORSPX"] = (
        int(scale_npix),
        "Pixels used for K-Cor/LASCO overlap scale",
    )
    hdr["KCORSMET"] = (
        str(scale_method)[:8],
        "K-Cor/LASCO scale method: BLEND/FALLBACK/NONE",
    )
    hdr["KCORUPX"] = (
        int(n_used),
        "Pixels where K-Cor contributed to combined pB",
    )
    hdr["KCORINPX"] = (
        int(n_inner_used),
        "K-Cor pixels used inside BLENDIN",
    )
    hdr["KCORBPX"] = (
        int(n_blend_used),
        "Pixels blended between K-Cor and LASCO",
    )
    hdr["KCORHPX"] = (
        int(n_hole_used),
        "LASCO-hole pixels filled by K-Cor",
    )
    hdr["KCORDIFF"] = (
        float(max_diff),
        "Max abs difference between combined and LASCO pB",
    )
    hdr["KCORRMIN"] = (
        float(kcor_rmin),
        "Minimum radius where K-Cor may be used [Rsun]",
    )
    hdr["KCORRMAX"] = (
        float(kcor_rmax),
        "Maximum radius where K-Cor may be used [Rsun]",
    )

    # Record the explicit removal of C2 data at/below BLENDIN.
    hdr["C2NANR"] = (
        float(blend_inner_rsun),
        "LASCO-C2 excluded at/below this radius [Rsun]",
    )
    hdr["C2NANPX"] = (
        int(n_c2_inner_excluded),
        "C2-positive pixels excluded at/below C2NANR",
    )
    hdr["INMISSPX"] = (
        int(n_inner_missing_kcor),
        "Inner pixels NaN because valid K-Cor is absent",
    )

    hdr["NANRMIN"] = (
        float(kcor_rmin),
        "Pixels below this radius are written as NaN [Rsun]",
    )
    hdr["INNERNPX"] = (
        int(n_inner_no_data),
        "Pixels set to NaN below NANRMIN",
    )
    hdr["BADGEOPX"] = (
        int(n_bad_geometry),
        "Pixels set to NaN due to invalid geometry",
    )
    hdr["KCOR11PX"] = (
        int(n_kcor_11_15),
        "Eligible K-Cor pixels in 1.0..1.5 Rsun",
    )
    hdr["KCOR15PX"] = (
        int(n_kcor_15_22),
        "Eligible K-Cor pixels in tomo inner range",
    )
    hdr["KCORBAPX"] = (
        int(n_kcor_blend),
        "Eligible K-Cor pixels in blend annulus",
    )
    hdr["KCORUNIT"] = str(
        kcor_hdr.get("BUNIT", "")
    )[:68]
    hdr["LASCOUNI"] = str(
        lasco_hdr.get("BUNIT", "")
    )[:68]

    fits.writeto(
        out,
        combined.astype(np.float32),
        hdr,
        overwrite=True,
    )

    print(
        f"[OK] Combined K-Cor/LASCO pB: {out} "
        f"(K-Cor used pixels={n_used}, "
        f"inner={n_inner_used}, "
        f"blend={n_blend_used}, "
        f"holes={n_hole_used}, "
        f"C2 inner excluded={n_c2_inner_excluded}, "
        f"inner missing K-Cor={n_inner_missing_kcor}, "
        f"max_diff={max_diff:.3e})"
    )

    return out


def copy_lasco_as_tomography_ready(
    lasco_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> Optional[Path]:
    """
    Copy one LASCO-C2 pB file into the tomography-ready directory when no suitable K-Cor
    pB/pBavg file is available. This preserves the +/-7 day rotational-tomography coverage
    instead of collapsing the inversion to only the few epochs with K-Cor.

    The output is intentionally named differently from the K-Cor/LASCO merged product so
    that later analysis can distinguish LASCO-only constraints from merged constraints.
    """
    lasco_dt = parse_pb_filename_datetime(lasco_path)
    if lasco_dt is None:
        return None

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"pB_LASCO_C2_only_{lasco_dt:%Y%m%d}_{lasco_dt:%H%M}.fits"
    if out.exists() and not overwrite:
        return out

    try:
        data, hdr = read_fits_image(Path(lasco_path))
        input_hdr = hdr.copy()
        output_hdr = make_earth_aligned_output_header(input_hdr)
        data = reproject_image_to_header(data, input_hdr, output_hdr, data.shape)
    except Exception as exc:
        print(f"[SKIP] Could not copy LASCO-only pB {Path(lasco_path).name}: {exc}")
        return None

    hdr = output_hdr.copy()
    # LASCO-only fallback products also need observer coordinates for rotational tomography.
    add_observer_geometry_keywords(
        out_hdr=hdr,
        source_hdr=input_hdr,
        obs_dt=parse_pb_filename_datetime(lasco_path),
        source_label="LASCO",
    )
    hdr["HISTORY"] = "LASCO-C2 pB used without K-Cor merge; no nearby K-Cor pB/pBavg was available."
    hdr["LASCOPB"] = Path(lasco_path).name[:68]
    hdr["KCORFILE"] = "NONE"
    fits.writeto(out, data.astype(np.float32), hdr, overwrite=True)
    print(f"[OK] LASCO-only tomography pB: {out}")
    return out

def prepare_kcor_lasco_pb_window(
    target_time: str | datetime,
    window_days: float,
    tomography_data_dir: Path,
    lasco_raw_dir: Path,
    kcor_raw_dir: Path,
    download_lasco: bool = True,
    download_kcor: bool = True,
    overwrite_downloads: bool = False,
    overwrite_combined: bool = False,
    kcor_max_delta_minutes: float = 60.0,
    kcor_search_step_minutes: float = 5.0,
    blend_inner_rsun: float = 2.2,
    blend_outer_rsun: float = 2.5,
    calibrate_kcor_to_lasco: bool = True,
    min_kcor_used_pixels: int = 100,
    kcor_use_min_rsun: float = 1.1,
    kcor_use_max_rsun: Optional[float] = None,
    scale_fallback_inner_rsun: float = 2.0,
    scale_fallback_outer_rsun: Optional[float] = None,
    lasco_fallback_hhmm_list: Optional[List[str]] = None,
    kcor_cookie_file: str = "",
    kcor_product: str = "pbavg",
    use_lasco_only_when_no_kcor: bool = True,
) -> List[Path]:
    """
    Prepare tomography-ready combined K-Cor/LASCO pB files for target_time +/- window_days.

    Steps:
      1) Download/find LASCO-C2 pB files in the full time window.
      2) Download/find only K-Cor pB files within +/- kcor_max_delta_minutes of LASCO times.
      3) Combine each LASCO-C2 pB with the nearest K-Cor pB into pB_Kcor_LASCO_axi_*.fits.
    """
    tomography_data_dir = Path(tomography_data_dir).expanduser()
    lasco_raw_dir = Path(lasco_raw_dir).expanduser()
    kcor_raw_dir = Path(kcor_raw_dir).expanduser()
    tomography_data_dir.mkdir(parents=True, exist_ok=True)
    lasco_raw_dir.mkdir(parents=True, exist_ok=True)
    kcor_raw_dir.mkdir(parents=True, exist_ok=True)

    if download_lasco:
        download_lasco_c2_pb_window(
            target_time=target_time,
            window_days=window_days,
            lasco_raw_dir=lasco_raw_dir,
            overwrite=overwrite_downloads,
            fallback_hhmm_list=lasco_fallback_hhmm_list,
        )

    lasco_paths = find_lasco_c2_pb_files(lasco_raw_dir, target_time=target_time, window_days=window_days)
    if not lasco_paths:
        print("[WARN] No LASCO-C2 pB files available in the requested time window.")
        return []
    print(f"[INFO] LASCO-C2 pB files in window: {len(lasco_paths)}")

    if download_kcor:
        download_kcor_near_lasco_times(
            lasco_paths=lasco_paths,
            kcor_raw_dir=kcor_raw_dir,
            max_time_delta_minutes=kcor_max_delta_minutes,
            search_step_minutes=kcor_search_step_minutes,
            overwrite=overwrite_downloads,
            cookie_file=Path(kcor_cookie_file).expanduser() if kcor_cookie_file else None,
            kcor_product=str(kcor_product),
        )

    kcor_paths = find_kcor_pb_files(
        kcor_raw_dir,
        target_time=target_time,
        window_days=window_days + 1.0,
        kcor_product=str(kcor_product),
    )
    if not kcor_paths:
        print("[WARN] No K-Cor pB/pBavg files available near the requested LASCO times.")
        if not use_lasco_only_when_no_kcor:
            return []
        lasco_only_paths = []
        for lasco_path in lasco_paths:
            out = copy_lasco_as_tomography_ready(
                lasco_path=lasco_path,
                output_dir=tomography_data_dir,
                overwrite=overwrite_combined,
            )
            if out is not None:
                lasco_only_paths.append(out)
        lasco_only_paths.sort(key=lambda p: parse_pb_filename_datetime(p) or datetime.min)
        print(f"[INFO] LASCO-only tomography pB files prepared: {len(lasco_only_paths)}")
        return lasco_only_paths
    print(f"[INFO] K-Cor {kcor_product} files available near window: {len(kcor_paths)}")

    combined_paths = []
    max_delta = float(kcor_max_delta_minutes) * 60.0
    for lasco_path in lasco_paths:
        lasco_dt = parse_pb_filename_datetime(lasco_path)
        if lasco_dt is None:
            continue
        kcor_path = nearest_file_by_time(kcor_paths, lasco_dt, max_delta)
        if kcor_path is None:
            print(f"[INFO] No K-Cor pB/pBavg within +/-{kcor_max_delta_minutes:g} min of {lasco_path.name}; using LASCO-only pB.")
            if use_lasco_only_when_no_kcor:
                out = copy_lasco_as_tomography_ready(
                    lasco_path=lasco_path,
                    output_dir=tomography_data_dir,
                    overwrite=overwrite_combined,
                )
                if out is not None:
                    combined_paths.append(out)
            continue
        out = combine_kcor_lasco_pair(
            lasco_path=lasco_path,
            kcor_path=kcor_path,
            output_dir=tomography_data_dir,
            blend_inner_rsun=blend_inner_rsun,
            blend_outer_rsun=blend_outer_rsun,
            overwrite=overwrite_combined,
            calibrate_kcor_to_lasco=calibrate_kcor_to_lasco,
            min_kcor_used_pixels=min_kcor_used_pixels,
            kcor_use_min_rsun=kcor_use_min_rsun,
            kcor_use_max_rsun=kcor_use_max_rsun,
            scale_fallback_inner_rsun=scale_fallback_inner_rsun,
            scale_fallback_outer_rsun=scale_fallback_outer_rsun,
        )
        if out is not None:
            combined_paths.append(out)
        elif use_lasco_only_when_no_kcor:
            fallback = copy_lasco_as_tomography_ready(
                lasco_path=lasco_path,
                output_dir=tomography_data_dir,
                overwrite=overwrite_combined,
            )
            if fallback is not None:
                combined_paths.append(fallback)

    combined_paths.sort(key=lambda p: parse_pb_filename_datetime(p) or datetime.min)
    print(f"[INFO] Combined K-Cor/LASCO pB files prepared: {len(combined_paths)}")
    return combined_paths

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Standalone LASCO-C2/K-Cor pB preparation settings.
    # Run this script first, then run main_multi_tomo.py for tomography.
    # ------------------------------------------------------------------
    DATA_DIR = "/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata"
    TARGET_TIME = "20220613_030000"
    SEARCH_WINDOW_DAYS = 7.0

    LASCO_RAW_DIR = "/mnt/d/wsl/home/kinno-7010/Research_data/SOHO/pB"
    KCOR_RAW_DIR = "/mnt/d/wsl/home/kinno-7010/Research_data/MK4_coronagraph/MK4_coronagraph_KCOR/pB/Rawdata"

    DOWNLOAD_LASCO = True
    DOWNLOAD_KCOR = True
    OVERWRITE_DOWNLOADS = False
    OVERWRITE_COMBINED = True

    KCOR_MAX_DELTA_MINUTES = 60.0
    KCOR_SEARCH_STEP_MINUTES = 5.0
    KCOR_COOKIE_FILE = "/home/kinno-7010/Research_code/MK4_coronagraph/MK4_coronagraph_KCOR/pB/hao_cookies.txt"
    KCOR_PRODUCT = "pbavg"

    KCOR_LASCO_BLEND_INNER_RSUN = 2.0
    KCOR_LASCO_BLEND_OUTER_RSUN = 3.0
    CALIBRATE_KCOR_TO_LASCO = True
    MIN_KCOR_USED_PIXELS = 100

    # K-Cor normally has useful inner-corona pB below the LASCO-C2 occulter.
    # These limits make the intended K-Cor radial usage explicit.  Tomography itself
    # will still ignore radii below main_multi_tomo.py's R_USE_MIN/R_MIN.
    KCOR_USE_MIN_RSUN = 1.0
    KCOR_USE_MAX_RSUN = KCOR_LASCO_BLEND_OUTER_RSUN

    # If K-Cor has no valid pixels in the nominal 2.2..2.5 Rs blend annulus, try
    # a conservative inner fallback annulus before leaving KCORSCAL=1.
    KCOR_SCALE_FALLBACK_INNER_RSUN = 2.0
    KCOR_SCALE_FALLBACK_OUTER_RSUN = KCOR_LASCO_BLEND_INNER_RSUN

    LASCO_FALLBACK_HHMM_LIST = ["0006", "0606", "1206", "1806"]
    USE_LASCO_ONLY_WHEN_NO_KCOR = False

    outputs = prepare_kcor_lasco_pb_window(
        target_time=TARGET_TIME,
        window_days=SEARCH_WINDOW_DAYS,
        tomography_data_dir=Path(DATA_DIR),
        lasco_raw_dir=Path(LASCO_RAW_DIR),
        kcor_raw_dir=Path(KCOR_RAW_DIR),
        download_lasco=DOWNLOAD_LASCO,
        download_kcor=DOWNLOAD_KCOR,
        overwrite_downloads=OVERWRITE_DOWNLOADS,
        overwrite_combined=OVERWRITE_COMBINED,
        kcor_max_delta_minutes=KCOR_MAX_DELTA_MINUTES,
        kcor_search_step_minutes=KCOR_SEARCH_STEP_MINUTES,
        blend_inner_rsun=KCOR_LASCO_BLEND_INNER_RSUN,
        blend_outer_rsun=KCOR_LASCO_BLEND_OUTER_RSUN,
        calibrate_kcor_to_lasco=CALIBRATE_KCOR_TO_LASCO,
        min_kcor_used_pixels=MIN_KCOR_USED_PIXELS,
        kcor_use_min_rsun=KCOR_USE_MIN_RSUN,
        kcor_use_max_rsun=KCOR_USE_MAX_RSUN,
        scale_fallback_inner_rsun=KCOR_SCALE_FALLBACK_INNER_RSUN,
        scale_fallback_outer_rsun=KCOR_SCALE_FALLBACK_OUTER_RSUN,
        lasco_fallback_hhmm_list=LASCO_FALLBACK_HHMM_LIST,
        kcor_cookie_file=KCOR_COOKIE_FILE,
        kcor_product=KCOR_PRODUCT,
        use_lasco_only_when_no_kcor=USE_LASCO_ONLY_WHEN_NO_KCOR,
    )

    print(f"[INFO] Prepared K-Cor/LASCO tomography-ready pB files: {len(outputs)}")
    for path in outputs:
        print(f"       {path}")
