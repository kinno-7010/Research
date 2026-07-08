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

def sample_image_bilinear_safe(img: np.ndarray, xpix: np.ndarray, ypix: np.ndarray) -> np.ndarray:
    """Bilinear sample img at floating pixel coordinates. Invalid/outside points become NaN."""
    img = np.asarray(img, dtype=np.float64)
    xpix = np.asarray(xpix, dtype=np.float64)
    ypix = np.asarray(ypix, dtype=np.float64)
    xpix, ypix = np.broadcast_arrays(xpix, ypix)
    out = np.full(xpix.shape, np.nan, dtype=np.float64)

    finite = np.isfinite(xpix) & np.isfinite(ypix)
    if not np.any(finite):
        return out

    x0 = np.floor(xpix[finite]).astype(np.int64)
    y0 = np.floor(ypix[finite]).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    inside = (x0 >= 0) & (y0 >= 0) & (x1 < img.shape[1]) & (y1 < img.shape[0])
    if not np.any(inside):
        return out

    flat_indices = np.flatnonzero(finite)[inside]
    x0 = x0[inside]
    y0 = y0[inside]
    x1 = x1[inside]
    y1 = y1[inside]
    x = xpix.ravel()[flat_indices]
    y = ypix.ravel()[flat_indices]

    wx = x - x0
    wy = y - y0
    vals = (
        (1.0 - wx) * (1.0 - wy) * img[y0, x0]
        + wx * (1.0 - wy) * img[y0, x1]
        + (1.0 - wx) * wy * img[y1, x0]
        + wx * wy * img[y1, x1]
    )
    out.ravel()[flat_indices] = vals
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


def _signed_header_cdelt_arcsec(hdr: fits.Header, axis: int, fallback_sign: float = 1.0) -> float:
    """Return signed CDELT in arcsec/pixel for a FITS axis."""
    try:
        cdelt = float(hdr.get(f"CDELT{axis}", fallback_sign))
    except Exception:
        cdelt = float(fallback_sign)
    unit_scale = _header_unit_to_arcsec_scale(str(hdr.get(f"CUNIT{axis}", "arcsec")))
    val = cdelt * unit_scale
    if not np.isfinite(val) or val == 0.0:
        val = float(fallback_sign)
    return float(val)


def make_earth_aligned_output_header(src_hdr: fits.Header) -> fits.Header:
    """Return a north-up, east/west-aligned output header derived from a LASCO header.

    LASCO-C2 pB headers can carry a small image rotation through CROTA, PC, or CD
    matrix keywords.  The old merge wrote the combined product on the native LASCO
    grid, although the K-Cor-to-LASCO reprojection already honored that rotation.
    This function creates an Earth-view projected grid with the same image size,
    Sun center, apparent solar radius, and pixel scale, but with the rotation terms
    removed.  LASCO and K-Cor are then both reprojected onto this common grid.
    """
    out_hdr = src_hdr.copy()

    mat = _linear_wcs_matrix_arcsec_per_pixel(src_hdr)
    scale1 = float(np.hypot(mat[0, 0], mat[1, 0]))
    scale2 = float(np.hypot(mat[0, 1], mat[1, 1]))
    if not np.isfinite(scale1) or scale1 <= 0:
        scale1 = abs(_signed_header_cdelt_arcsec(src_hdr, 1, fallback_sign=1.0))
    if not np.isfinite(scale2) or scale2 <= 0:
        scale2 = abs(_signed_header_cdelt_arcsec(src_hdr, 2, fallback_sign=1.0))

    # Keep the original axis signs to avoid unintended mirror flips, but remove rotation.
    sign1 = np.sign(_signed_header_cdelt_arcsec(src_hdr, 1, fallback_sign=1.0)) or 1.0
    sign2 = np.sign(_signed_header_cdelt_arcsec(src_hdr, 2, fallback_sign=1.0)) or 1.0

    crval1_arcsec = float(src_hdr.get("CRVAL1", 0.0)) * _header_unit_to_arcsec_scale(str(src_hdr.get("CUNIT1", "arcsec")))
    crval2_arcsec = float(src_hdr.get("CRVAL2", 0.0)) * _header_unit_to_arcsec_scale(str(src_hdr.get("CUNIT2", "arcsec")))

    # Store the output WCS in arcsec with a diagonal matrix.  Remove all legacy
    # rotation terms so downstream plotting does not need to interpret CROTA/PC/CD.
    for key in (
        "CROTA", "CROTA1", "CROTA2",
        "PC1_1", "PC1_2", "PC2_1", "PC2_2",
        "CD1_1", "CD1_2", "CD2_1", "CD2_2",
    ):
        if key in out_hdr:
            del out_hdr[key]

    out_hdr["CUNIT1"] = "arcsec"
    out_hdr["CUNIT2"] = "arcsec"
    out_hdr["CDELT1"] = (float(sign1 * scale1), "Earth-view aligned x pixel scale [arcsec/pixel]")
    out_hdr["CDELT2"] = (float(sign2 * scale2), "Earth-view aligned y pixel scale [arcsec/pixel]")
    out_hdr["CRVAL1"] = (float(crval1_arcsec), "Earth-view aligned x coordinate at CRPIX [arcsec]")
    out_hdr["CRVAL2"] = (float(crval2_arcsec), "Earth-view aligned y coordinate at CRPIX [arcsec]")
    out_hdr["CROTA2"] = (0.0, "Output image de-rotated to Earth-view axes")
    out_hdr["WCSALIGN"] = ("EARTH", "LASCO/K-Cor merged on Earth-view north-up grid")
    out_hdr["DEROTATE"] = (True, "LASCO and K-Cor were reprojected to rotation-free output WCS")

    # Keep a compact record of the original LASCO rotation information.
    if "CROTA2" in src_hdr or "CROTA1" in src_hdr:
        out_hdr["LASCOROT"] = (float(src_hdr.get("CROTA2", src_hdr.get("CROTA1", 0.0))), "Original LASCO CROTA angle [deg]")
    else:
        try:
            theta = float(np.degrees(np.arctan2(mat[1, 0], mat[0, 0])))
        except Exception:
            theta = 0.0
        out_hdr["LASCOROT"] = (theta, "Original LASCO WCS rotation angle [deg]")

    return out_hdr


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
    Return helioprojective x/y coordinate maps for the *rebinned* image in **Rsun units**.

    This codebase frequently uses r_use_min/r_use_max in Rsun. Therefore x_map and y_map must be
    expressed as x/RSUN_OBS and y/RSUN_OBS (dimensionless solar radii), not in arcsec.

    If LASCO-C2 lacks RSUN_OBS/RSUN, pass the nearest K-Cor RSUN_OBS as
    rsun_arcsec_override before applying any radial masks.  Otherwise the code falls
    back to 959.63 arcsec and can shift the apparent 1.1 Rs boundary by about 1--2%.

    Returns
    -------
    x_map_rsun, y_map_rsun : (out_n,out_n)
        Helioprojective coordinates in Rsun (dimensionless), centered at Sun center.
    rsun_arcsec : float
        Apparent solar radius in arcsec (RSUN_OBS/RSUN keyword or override).
    """
    if rsun_arcsec_override is not None and np.isfinite(float(rsun_arcsec_override)) and float(rsun_arcsec_override) > 0:
        rsun_arcsec = float(rsun_arcsec_override)
    else:
        rsun_arcsec = float(hdr.get("RSUN_OBS", hdr.get("RSUN", 959.63)))

    # scale factor from original -> rebinned pixels
    s = float(orig_n) / float(out_n)

    # Pixel centers on rebinned grid
    yy, xx = np.mgrid[0:out_n, 0:out_n]
    xpix = (xx + 0.5) * s - 0.5
    ypix = (yy + 0.5) * s - 0.5

    # Use astropy WCS only when the header has a complete celestial WCS.
    # For LASCO pB headers that only carry CRPIX/CDELT/CROTA, astropy can return
    # a generic linear coordinate without enough semantic checks, so the explicit
    # FITS-linear fallback is safer and easier to audit.
    if _has_complete_celestial_wcs(hdr):
        try:
            w = WCS(hdr)
            xw, yw = w.pixel_to_world_values(xpix, ypix)
            cu = getattr(w.wcs, "cunit", ["", ""])
            u1 = str(hdr.get("CUNIT1", cu[0] if len(cu) > 0 else "arcsec")).lower()
            u2 = str(hdr.get("CUNIT2", cu[1] if len(cu) > 1 else "arcsec")).lower()
            x_map = _wcs_unit_to_arcsec(np.asarray(xw, dtype=np.float64), u1)
            y_map = _wcs_unit_to_arcsec(np.asarray(yw, dtype=np.float64), u2)
        except Exception as exc:
            print(f"[WARN] WCS coordinate-map construction failed ({exc}); using linear CRPIX/CDELT/CD/PC/CROTA mapping.")
            x_map, y_map = _linear_pixel_to_arcsec(hdr, xpix, ypix)
    else:
        x_map, y_map = _linear_pixel_to_arcsec(hdr, xpix, ypix)

    # ---- convert arcsec -> Rsun ----
    if not np.isfinite(rsun_arcsec) or rsun_arcsec <= 0:
        rsun_arcsec = 959.63
    x_map_rsun = x_map / rsun_arcsec
    y_map_rsun = y_map / rsun_arcsec

    return x_map_rsun.astype(np.float64), y_map_rsun.astype(np.float64), rsun_arcsec


def _estimate_rsun_pixel_scale_from_rho(rho: np.ndarray) -> float:
    """Estimate a representative projected pixel scale in Rsun from a rho map."""
    vals = []
    for axis in (0, 1):
        d = np.abs(np.diff(np.asarray(rho, dtype=np.float64), axis=axis))
        d = d[np.isfinite(d) & (d > 0)]
        if d.size:
            vals.append(d)
    if not vals:
        return np.nan
    all_d = np.concatenate(vals)
    # Reject very large radial-gradient values near the plot edge or masked geometry.
    all_d = all_d[(all_d > 0) & (all_d < np.nanpercentile(all_d, 95.0))]
    if all_d.size == 0:
        return np.nan
    return float(np.nanmedian(all_d))


def _nan_box_mean_2d(values: np.ndarray, radius_pix: int) -> np.ndarray:
    """NaN-aware square-box mean using summed-area tables; no scipy dependency."""
    arr = np.asarray(values, dtype=np.float64)
    r = int(max(1, radius_pix))
    pad = np.pad(arr, r, mode="constant", constant_values=np.nan)

    finite = np.isfinite(pad)
    val = np.where(finite, pad, 0.0)
    cnt = finite.astype(np.float64)

    # Integral images with a leading zero row/column.
    val_ii = np.pad(np.cumsum(np.cumsum(val, axis=0), axis=1), ((1, 0), (1, 0)), mode="constant")
    cnt_ii = np.pad(np.cumsum(np.cumsum(cnt, axis=0), axis=1), ((1, 0), (1, 0)), mode="constant")

    w = 2 * r + 1
    sumv = val_ii[w:, w:] - val_ii[:-w, w:] - val_ii[w:, :-w] + val_ii[:-w, :-w]
    sumc = cnt_ii[w:, w:] - cnt_ii[:-w, w:] - cnt_ii[w:, :-w] + cnt_ii[:-w, :-w]

    out = np.full(arr.shape, np.nan, dtype=np.float64)
    ok = sumc > 0
    out[ok] = sumv[ok] / sumc[ok]
    return out


def _edge_transition_boundary_mask(kcor_weight: np.ndarray, good: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Find pixels lying on a K-Cor/C2 contribution boundary.

    kcor_weight=1 means pure K-Cor, kcor_weight=0 means pure LASCO-C2, and
    intermediate values mean a K-Cor/LASCO blend.  The boundary is defined as
    neighboring finite pixels that fall on opposite sides of the chosen K-Cor
    weight threshold.  Both sides of the transition are marked.
    """
    w = np.asarray(kcor_weight, dtype=np.float64)
    good = np.asarray(good, dtype=bool) & np.isfinite(w)
    thr = float(threshold)
    if not np.isfinite(thr):
        thr = 0.5
    thr = float(np.clip(thr, 0.01, 0.99))

    label = np.full(w.shape, -1, dtype=np.int8)
    label[good & (w >= thr)] = 1
    label[good & (w < thr)] = 0

    boundary = np.zeros(w.shape, dtype=bool)

    # Vertical neighbor pairs.
    a = label[:-1, :]
    b = label[1:, :]
    diff = (a >= 0) & (b >= 0) & (a != b)
    boundary[:-1, :] |= diff
    boundary[1:, :] |= diff

    # Horizontal neighbor pairs.
    a = label[:, :-1]
    b = label[:, 1:]
    diff = (a >= 0) & (b >= 0) & (a != b)
    boundary[:, :-1] |= diff
    boundary[:, 1:] |= diff

    return boundary


def _fill_circular_nan_by_interpolation(values: np.ndarray) -> np.ndarray:
    """Fill NaNs in a periodic 1D array by circular linear interpolation."""
    vals = np.asarray(values, dtype=np.float64)
    out = np.array(vals, dtype=np.float64, copy=True)
    n = out.size
    good = np.isfinite(out)
    if n == 0 or not np.any(good):
        return out
    if np.all(good):
        return out

    x_good = np.flatnonzero(good).astype(np.float64)
    y_good = out[good]
    x_ext = np.concatenate([x_good - n, x_good, x_good + n])
    y_ext = np.concatenate([y_good, y_good, y_good])
    x_all = np.arange(n, dtype=np.float64)
    out[~good] = np.interp(x_all[~good], x_ext, y_ext)
    return out


def _radial_boundary_edge_smooth(
    image: np.ndarray,
    rho: np.ndarray,
    x_rsun: np.ndarray,
    y_rsun: np.ndarray,
    kcor_weight: np.ndarray,
    boundary_width_rsun: float,
    log_space: bool = True,
    angular_bin_deg: float = 2.0,
    weight_threshold: float = 0.5,
    smooth_min_rsun: Optional[float] = None,
    smooth_max_rsun: Optional[float] = None,
) -> Tuple[np.ndarray, int, int, float, float, float]:
    """Smooth radially around the actual K-Cor/C2 boundary.

    The boundary is not prescribed as a single circular radius.  It is detected
    from the K-Cor contribution weight map and then converted into a boundary
    radius as a function of polar angle.  Pixels within BOUNDARY_WIDTH_RSUN/2
    of that local boundary radius are replaced by a NaN-aware radial running
    mean computed along the same angular sector.  This suppresses the
    instrument-transition seam without broadly blurring tangential structures.
    """
    arr = np.array(image, dtype=np.float64, copy=True)
    rho = np.asarray(rho, dtype=np.float64)
    x_rsun = np.asarray(x_rsun, dtype=np.float64)
    y_rsun = np.asarray(y_rsun, dtype=np.float64)
    kcor_weight = np.asarray(kcor_weight, dtype=np.float64)

    width = float(boundary_width_rsun)
    if not np.isfinite(width) or width <= 0:
        return arr, 0, 0, np.nan, np.nan, np.nan

    try:
        smooth_min = float(smooth_min_rsun) if smooth_min_rsun is not None else np.nan
        smooth_max = float(smooth_max_rsun) if smooth_max_rsun is not None else np.nan
        use_minmax = np.isfinite(smooth_min) and np.isfinite(smooth_max) and smooth_max > smooth_min
    except Exception:
        smooth_min = np.nan
        smooth_max = np.nan
        use_minmax = False

    good = (
        np.isfinite(arr)
        & (arr > 0)
        & np.isfinite(rho)
        & np.isfinite(x_rsun)
        & np.isfinite(y_rsun)
        & np.isfinite(kcor_weight)
    )
    if not np.any(good):
        return arr, 0, 0, np.nan, np.nan, np.nan

    boundary = _edge_transition_boundary_mask(kcor_weight, good=good, threshold=weight_threshold)
    if use_minmax:
        boundary &= (rho >= smooth_min) & (rho <= smooth_max)

    n_boundary = int(np.count_nonzero(boundary))
    if n_boundary == 0:
        return arr, 0, 0, np.nan, np.nan, np.nan

    dtheta = float(angular_bin_deg)
    if not np.isfinite(dtheta) or dtheta <= 0:
        dtheta = 2.0
    n_theta = int(np.clip(round(360.0 / dtheta), 36, 1440))
    theta = (np.arctan2(y_rsun, x_rsun) + 2.0 * np.pi) % (2.0 * np.pi)
    theta_bin = np.floor(theta / (2.0 * np.pi / float(n_theta))).astype(np.int64)
    theta_bin = np.clip(theta_bin, 0, n_theta - 1)

    boundary_r = np.full(n_theta, np.nan, dtype=np.float64)
    for ibin in range(n_theta):
        m = boundary & (theta_bin == ibin)
        if np.any(m):
            boundary_r[ibin] = float(np.nanmedian(rho[m]))

    boundary_r = _fill_circular_nan_by_interpolation(boundary_r)
    if not np.any(np.isfinite(boundary_r)):
        return arr, 0, n_boundary, np.nan, np.nan, np.nan

    local_boundary_r = boundary_r[theta_bin]
    replace = good & np.isfinite(local_boundary_r) & (np.abs(rho - local_boundary_r) <= 0.5 * width)
    if use_minmax:
        replace &= (rho >= smooth_min) & (rho <= smooth_max)

    n_replace = int(np.count_nonzero(replace))
    if n_replace == 0:
        return arr, 0, n_boundary, float(np.nanmedian(boundary_r)), np.nan, np.nan

    work = np.full_like(arr, np.nan, dtype=np.float64)
    if bool(log_space):
        work[good] = np.log(arr[good])
    else:
        work[good] = arr[good]

    half_width = 0.5 * width
    flat_bin = theta_bin.ravel()
    flat_rho = rho.ravel()
    flat_work = work.ravel()
    flat_replace = replace.ravel()
    flat_out = arr.ravel()

    for ibin in range(n_theta):
        sector = np.flatnonzero((flat_bin == ibin) & np.isfinite(flat_work) & np.isfinite(flat_rho))
        if sector.size < 2:
            continue

        order = np.argsort(flat_rho[sector])
        idx = sector[order]
        r_sorted = flat_rho[idx]
        v_sorted = flat_work[idx]
        target_sorted = flat_replace[idx]
        if not np.any(target_sorted):
            continue

        csum = np.concatenate([[0.0], np.cumsum(v_sorted)])
        ccnt = np.arange(v_sorted.size + 1, dtype=np.float64)

        left = np.searchsorted(r_sorted, r_sorted - half_width, side="left")
        right = np.searchsorted(r_sorted, r_sorted + half_width, side="right")
        count = ccnt[right] - ccnt[left]
        valid = target_sorted & (count > 0)
        if not np.any(valid):
            continue

        smoothed = np.empty_like(v_sorted)
        smoothed[:] = np.nan
        smoothed[valid] = (csum[right[valid]] - csum[left[valid]]) / count[valid]
        if bool(log_space):
            smoothed[valid] = np.exp(smoothed[valid])

        good_assign = valid & np.isfinite(smoothed) & (smoothed > 0)
        flat_out[idx[good_assign]] = smoothed[good_assign]

    actual = replace & np.isfinite(arr) & (arr > 0)
    smooth_min_actual = float(np.nanmin(rho[replace])) if np.any(replace) else np.nan
    smooth_max_actual = float(np.nanmax(rho[replace])) if np.any(replace) else np.nan
    boundary_r_median = float(np.nanmedian(boundary_r[np.isfinite(boundary_r)]))
    return arr, n_replace, n_boundary, boundary_r_median, smooth_min_actual, smooth_max_actual


def _local_mean_edge_smooth(
    image: np.ndarray,
    rho: np.ndarray,
    r_edge: float,
    width_rsun: float,
    log_space: bool = True,
    smooth_min_rsun: Optional[float] = None,
    smooth_max_rsun: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """Smooth a finite radial annulus around a K-Cor/LASCO transition.

    If smooth_min_rsun and smooth_max_rsun are both finite and
    smooth_max_rsun > smooth_min_rsun, the replacement annulus is exactly
    smooth_min_rsun <= rho <= smooth_max_rsun.  Otherwise, the older
    center/width definition is used: |rho-r_edge| <= width_rsun/2.

    The smoothing kernel is a NaN-aware square-box mean.  Its radius is tied
    to the selected smoothing annulus width, not fixed to 3x3 pixels.
    """
    arr = np.array(image, dtype=np.float64, copy=True)

    use_minmax = False
    try:
        smooth_min = float(smooth_min_rsun) if smooth_min_rsun is not None else np.nan
        smooth_max = float(smooth_max_rsun) if smooth_max_rsun is not None else np.nan
        use_minmax = np.isfinite(smooth_min) and np.isfinite(smooth_max) and smooth_max > smooth_min
    except Exception:
        smooth_min = np.nan
        smooth_max = np.nan
        use_minmax = False

    if use_minmax:
        width = smooth_max - smooth_min
        band = (
            np.isfinite(rho)
            & (rho >= smooth_min)
            & (rho <= smooth_max)
            & np.isfinite(arr)
            & (arr > 0)
        )
    else:
        width = float(width_rsun)
        if (not np.isfinite(r_edge)) or (not np.isfinite(width)) or width <= 0:
            return arr, 0
        band = (
            np.isfinite(rho)
            & (rho >= float(r_edge) - 0.5 * width)
            & (rho <= float(r_edge) + 0.5 * width)
            & np.isfinite(arr)
            & (arr > 0)
        )

    if not np.any(band):
        return arr, 0

    pix_scale = _estimate_rsun_pixel_scale_from_rho(rho)
    if not np.isfinite(pix_scale) or pix_scale <= 0:
        radius_pix = 2
    else:
        radius_pix = int(np.ceil(0.5 * width / pix_scale))
    radius_pix = int(np.clip(radius_pix, 2, 80))

    work = np.full_like(arr, np.nan, dtype=np.float64)
    good = np.isfinite(arr) & (arr > 0)
    if bool(log_space):
        work[good] = np.log(arr[good])
    else:
        work[good] = arr[good]

    smooth = _nan_box_mean_2d(work, radius_pix=radius_pix)
    valid = np.isfinite(smooth)
    if bool(log_space):
        smooth[valid] = np.exp(smooth[valid])

    replace = band & np.isfinite(smooth) & (smooth > 0)
    arr[replace] = smooth[replace]
    return arr, int(np.count_nonzero(replace))

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
    edge_smooth_when_no_blend: bool = True,
    edge_smooth_width_rsun: float = 0.03,
    edge_smooth_log_space: bool = True,
    edge_smooth_min_blend_pixels: int = 20,
    edge_smooth_min_blend_fraction: float = 0.10,
    edge_smooth_always: bool = True,
    edge_smooth_min_rsun: Optional[float] = None,
    edge_smooth_max_rsun: Optional[float] = None,
    boundary_smooth: bool = True,
    boundary_width_rsun: float = 0.35,
    boundary_angular_bin_deg: float = 2.0,
    boundary_weight_threshold: float = 0.5,
) -> Optional[Path]:
    """
    Combine one LASCO-C2 pB image with the nearest K-Cor pB image.

    A LASCO-derived Earth-view, rotation-free grid/header is used as the output grid.
    LASCO-C2 and K-Cor are both reprojected onto this grid. K-Cor is used
    preferentially at small heliocentric distances, with a linear blend across
    blend_inner_rsun..blend_outer_rsun. This creates the tomography-ready file:
      pB_Kcor_LASCO_axi_<YYYYMMDD>_<HHMM>.fits

    This version is deliberately conservative: if the K-Cor reprojection contributes too
    few valid pixels, it does not write a misleading LASCO-identical "combined" file.

    If the nominal K-Cor/LASCO overlap blend region is sparse, or if
    edge_smooth_always=True, an optional edge-smoothing step is applied over
    a narrow annulus around the K-Cor/LASCO transition.  This suppresses
    artificial pixel-scale discontinuities, but it is not a substitute for a true
    overlap-based photometric scale calibration.
    """
    lasco_dt = parse_pb_filename_datetime(lasco_path)
    if lasco_dt is None:
        return None

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"pB_Kcor_LASCO_edge_smooth_{lasco_dt:%Y%m%d}_{lasco_dt:%H%M}.fits"
    if out.exists() and not overwrite:
        return out

    try:
        lasco_img, lasco_hdr = read_fits_image(Path(lasco_path))
        kcor_img, kcor_hdr = read_fits_image(Path(kcor_path))

        # LASCO-C2 pB headers used here often lack RSUN_OBS/RSUN, while K-Cor
        # has a reliable apparent solar radius at nearly the same Earth-view time.
        # Copy it before any Rsun-based radial mask or output-header creation.
        working_lasco_hdr = lasco_hdr.copy()
        rsun_for_grid = _first_finite_header_float(working_lasco_hdr, ("RSUN_OBS", "RSUN"))
        rsun_source = "LASCO"
        if rsun_for_grid is None:
            rsun_for_grid = _first_finite_header_float(kcor_hdr, ("RSUN_OBS", "RSUN"))
            rsun_source = "KCOR"
        if rsun_for_grid is not None and np.isfinite(rsun_for_grid) and rsun_for_grid > 0:
            working_lasco_hdr["RSUN_OBS"] = (float(rsun_for_grid), "Apparent solar radius [arcsec]")
            working_lasco_hdr["RSUN"] = (float(rsun_for_grid), "Apparent solar radius [arcsec]")

        # Reproject both instruments onto a common Earth-view, rotation-free grid.
        # The previous implementation corrected the relative K-Cor/LASCO angle when
        # mapping K-Cor to the native LASCO grid, but the output product itself kept
        # the native LASCO tilt.  For tomography and visual comparison, write the
        # merged pB image on a north-up Earth-view grid.
        output_hdr = make_earth_aligned_output_header(working_lasco_hdr)
        lasco_img = reproject_image_to_header(lasco_img, working_lasco_hdr, output_hdr, lasco_img.shape)
        kcor_on_lasco = reproject_image_to_header(kcor_img, kcor_hdr, output_hdr, lasco_img.shape)
        working_lasco_hdr = output_hdr
        print(f"[QC] Reprojected LASCO-C2 and K-Cor onto Earth-view aligned grid (original LASCO rotation={working_lasco_hdr.get('LASCOROT', 0.0)} deg).")
    except Exception as exc:
        print(f"[SKIP] Could not combine {lasco_path.name} with {kcor_path.name}: {exc}")
        return None

    lasco_unit = _pb_unit_kind(lasco_hdr)
    kcor_unit = _pb_unit_kind(kcor_hdr)
    if lasco_unit == "raw_dn" or kcor_unit == "raw_dn":
        print(
            f"[SKIP] Raw DN input detected (LASCO={lasco_hdr.get('BUNIT')!r}, "
            f"KCOR={kcor_hdr.get('BUNIT')!r}); refusing to write tomography pB product."
        )
        return None
    if lasco_unit == "unknown" or kcor_unit == "unknown":
        print(
            f"[WARN] Unknown pB unit convention (LASCO BUNIT={lasco_hdr.get('BUNIT')!r}, "
            f"KCOR BUNIT={kcor_hdr.get('BUNIT')!r}); proceeding but check absolute density scale."
        )

    x_map, y_map, rsun_arcsec_for_grid = xy_rsun_for_rebinned_image(
        working_lasco_hdr,
        orig_n=lasco_img.shape[0],
        out_n=lasco_img.shape[0],
        rsun_arcsec_override=rsun_for_grid,
    )
    rho = np.hypot(x_map, y_map)

    kcor_rmin = float(kcor_use_min_rsun)
    kcor_rmax = float(kcor_use_max_rsun) if kcor_use_max_rsun is not None else float(blend_outer_rsun)
    if not np.isfinite(kcor_rmin) or kcor_rmin < 0:
        kcor_rmin = 0.0
    if not np.isfinite(kcor_rmax) or kcor_rmax <= kcor_rmin:
        kcor_rmax = float(blend_outer_rsun)
    kcor_radius_mask = (rho >= kcor_rmin) & (rho <= kcor_rmax)

    has_kcor_any_radius = np.isfinite(kcor_on_lasco) & (kcor_on_lasco > 0)
    has_kcor_raw = has_kcor_any_radius & kcor_radius_mask
    has_lasco = np.isfinite(lasco_img) & (lasco_img > 0)
    finite_kcor = int(np.count_nonzero(has_kcor_raw))
    finite_kcor_all = int(np.count_nonzero(has_kcor_any_radius))
    print(
        f"[QC] K-Cor reprojected onto LASCO grid: positive finite pixels={finite_kcor_all}, "
        f"eligible pixels={finite_kcor} within {kcor_rmin:.3f}..{kcor_rmax:.3f} Rsun, "
        f"LASCO positive finite pixels={int(np.count_nonzero(has_lasco))}"
    )
    if finite_kcor < int(min_kcor_used_pixels):
        print(
            f"[SKIP] K-Cor reprojection contributed only {finite_kcor} valid positive pixels "
            f"(< min_kcor_used_pixels={min_kcor_used_pixels}). Not writing misleading combined file."
        )
        return None

    kcor_scale = 1.0
    scale_npix = 0
    if bool(calibrate_kcor_to_lasco):
        scale_mask = has_kcor_raw & has_lasco & (rho >= blend_inner_rsun) & (rho <= blend_outer_rsun)
        kcor_scale, scale_npix = _robust_positive_scale(lasco_img, kcor_on_lasco, scale_mask, min_pixels=max(20, min_kcor_used_pixels // 2))
        if kcor_scale > float(max_scale_factor) or kcor_scale < 1.0 / float(max_scale_factor):
            print(
                f"[WARN] K-Cor/LASCO overlap scale={kcor_scale:.4g} is outside allowed range "
                f"[1/{max_scale_factor:g}, {max_scale_factor:g}]. Using scale=1.0."
            )
            kcor_scale = 1.0
        print(f"[QC] K-Cor-to-LASCO scale from overlap annulus={kcor_scale:.6g} (pixels={scale_npix})")

    kcor_scaled = kcor_on_lasco * kcor_scale
    has_kcor = np.isfinite(kcor_scaled) & (kcor_scaled > 0) & kcor_radius_mask

    combined = np.array(lasco_img, dtype=np.float64, copy=True)
    kcor_weight = np.zeros_like(combined, dtype=np.float64)

    inner = has_kcor & (rho <= blend_inner_rsun)
    combined[inner] = kcor_scaled[inner]
    kcor_weight[inner] = 1.0

    overlap = has_kcor & has_lasco & (rho > blend_inner_rsun) & (rho < blend_outer_rsun)
    n_overlap = int(np.count_nonzero(overlap))
    if n_overlap > 0:
        alpha = (rho[overlap] - blend_inner_rsun) / max(1e-6, (blend_outer_rsun - blend_inner_rsun))
        w_kcor = 1.0 - alpha
        combined[overlap] = w_kcor * kcor_scaled[overlap] + alpha * lasco_img[overlap]
        kcor_weight[overlap] = np.maximum(kcor_weight[overlap], w_kcor)

    # Fill LASCO holes only in the K-Cor/LASCO transition domain.  Do not allow K-Cor
    # to overwrite arbitrary large-radius LASCO holes beyond the intended field.
    fill_lasco_holes = has_kcor & (~has_lasco)
    combined[fill_lasco_holes] = kcor_scaled[fill_lasco_holes]
    kcor_weight[fill_lasco_holes] = 1.0

    kcor_used_pre_smooth = inner | overlap | fill_lasco_holes

    edge_smooth_pixels = 0
    edge_smooth_radius = np.nan
    overlap_possible = has_lasco & np.isfinite(rho) & (rho > blend_inner_rsun) & (rho < blend_outer_rsun)
    n_overlap_possible = int(np.count_nonzero(overlap_possible))
    overlap_fraction = (float(n_overlap) / float(n_overlap_possible)) if n_overlap_possible > 0 else 0.0

    # Apply the edge smoother not only when the overlap pixel count is small, but
    # also when the overlap is sparse compared with the available annulus.  This
    # matters when blend_outer_rsun is widened: n_overlap can exceed 20, while the
    # actual K-Cor coverage is still too patchy for a visually smooth transition.
    do_edge_smooth = bool(edge_smooth_when_no_blend) and (
        bool(edge_smooth_always)
        or n_overlap < int(edge_smooth_min_blend_pixels)
        or overlap_fraction < float(edge_smooth_min_blend_fraction)
    )

    boundary_pixels = 0
    boundary_smooth_min_actual = np.nan
    boundary_smooth_max_actual = np.nan
    if do_edge_smooth:
        if bool(boundary_smooth):
            combined_new, edge_smooth_pixels, boundary_pixels, edge_smooth_radius, boundary_smooth_min_actual, boundary_smooth_max_actual = _radial_boundary_edge_smooth(
                combined,
                rho,
                x_map,
                y_map,
                kcor_weight,
                boundary_width_rsun=float(boundary_width_rsun),
                log_space=bool(edge_smooth_log_space),
                angular_bin_deg=float(boundary_angular_bin_deg),
                weight_threshold=float(boundary_weight_threshold),
                smooth_min_rsun=edge_smooth_min_rsun,
                smooth_max_rsun=edge_smooth_max_rsun,
            )
            if edge_smooth_pixels > 0:
                combined = combined_new
                print(
                    f"[QC] Boundary-smoothed K-Cor/LASCO transition: "
                    f"boundary_pixels={boundary_pixels}, "
                    f"boundary_r_median={edge_smooth_radius:.4f} Rs, "
                    f"boundary_width={float(boundary_width_rsun):.4f} Rs, "
                    f"smooth_min_actual={boundary_smooth_min_actual:.4f} Rs, "
                    f"smooth_max_actual={boundary_smooth_max_actual:.4f} Rs, "
                    f"pixels={edge_smooth_pixels}, log_space={bool(edge_smooth_log_space)}, "
                    f"overlap_fraction={overlap_fraction:.4f}"
                )

        if edge_smooth_pixels <= 0:
            # Fallback to the older annular smoother if no K-Cor/C2 boundary was
            # found.  This preserves the previous behavior for unusual images.
            if n_overlap > 0:
                edge_smooth_radius = float(blend_inner_rsun)
            else:
                inner_kcor_for_edge = kcor_used_pre_smooth & np.isfinite(rho) & (rho <= blend_inner_rsun)
                if np.any(inner_kcor_for_edge):
                    edge_smooth_radius = float(np.nanmax(rho[inner_kcor_for_edge]))

            if np.isfinite(edge_smooth_radius):
                combined, edge_smooth_pixels = _local_mean_edge_smooth(
                    combined,
                    rho,
                    r_edge=edge_smooth_radius,
                    width_rsun=float(edge_smooth_width_rsun),
                    log_space=bool(edge_smooth_log_space),
                    smooth_min_rsun=edge_smooth_min_rsun,
                    smooth_max_rsun=edge_smooth_max_rsun,
                )
                if edge_smooth_pixels > 0:
                    try:
                        sm_min = float(edge_smooth_min_rsun) if edge_smooth_min_rsun is not None else edge_smooth_radius - 0.5 * float(edge_smooth_width_rsun)
                        sm_max = float(edge_smooth_max_rsun) if edge_smooth_max_rsun is not None else edge_smooth_radius + 0.5 * float(edge_smooth_width_rsun)
                        sm_mask = np.isfinite(rho) & (rho >= sm_min) & (rho <= sm_max)
                        if np.any(sm_mask):
                            boundary_smooth_min_actual = float(np.nanmin(rho[sm_mask]))
                            boundary_smooth_max_actual = float(np.nanmax(rho[sm_mask]))
                    except Exception:
                        boundary_smooth_min_actual = np.nan
                        boundary_smooth_max_actual = np.nan
                    print(
                        f"[QC] Edge-smoothed K-Cor/LASCO transition: "
                        f"r_edge={edge_smooth_radius:.4f} Rs, "
                        f"width={float(edge_smooth_width_rsun):.4f} Rs, "
                        f"smooth_min={float(edge_smooth_min_rsun) if edge_smooth_min_rsun is not None else np.nan:.4f} Rs, "
                        f"smooth_max={float(edge_smooth_max_rsun) if edge_smooth_max_rsun is not None else np.nan:.4f} Rs, "
                        f"pixels={edge_smooth_pixels}, log_space={bool(edge_smooth_log_space)}, "
                        f"overlap_fraction={overlap_fraction:.4f}"
                    )

    inner_no_data = np.isfinite(rho) & (rho < kcor_rmin)
    bad_geometry = ~np.isfinite(rho)
    combined[inner_no_data | bad_geometry] = np.nan

    kcor_used = kcor_used_pre_smooth
    n_inner_no_data = int(np.count_nonzero(inner_no_data))
    n_bad_geometry = int(np.count_nonzero(bad_geometry))
    n_used = int(np.count_nonzero(kcor_used))
    max_diff = float(np.nanmax(np.abs(combined - lasco_img))) if np.any(np.isfinite(combined - lasco_img)) else 0.0
    if n_used < int(min_kcor_used_pixels) or not np.isfinite(max_diff) or max_diff <= 0:
        print(
            f"[SKIP] Combined image is effectively LASCO-only (K-Cor used pixels={n_used}, max_diff={max_diff:.3e}). "
            "Not writing pB_Kcor_LASCO_axi product."
        )
        return None

    hdr = working_lasco_hdr.copy()
    # Add K-Cor/Earth-view observer coordinates needed by main_multi_tomo.py.
    add_observer_geometry_keywords(
        out_hdr=hdr,
        source_hdr=kcor_hdr,
        obs_dt=parse_pb_filename_datetime(kcor_path) or lasco_dt,
        source_label="KCOR",
    )
    if rsun_for_grid is not None and np.isfinite(rsun_for_grid) and rsun_for_grid > 0:
        hdr["RSUN_OBS"] = (float(rsun_for_grid), "Apparent solar radius [arcsec]")
        hdr["RSUN"] = (float(rsun_for_grid), "Apparent solar radius [arcsec]")
        hdr["KCORRSUN"] = (float(_first_finite_header_float(kcor_hdr, ("RSUN_OBS", "RSUN")) or rsun_for_grid), "K-Cor apparent solar radius [arcsec]")
        hdr["RSUNSRC"] = (str(rsun_source)[:8], "Source of RSUN_OBS used for output grid")

    hdr["HISTORY"] = f"Combined LASCO-C2 pB with K-Cor pB: {Path(kcor_path).name}"
    hdr["KCORFILE"] = Path(kcor_path).name[:68]
    hdr["LASCOPB"] = Path(lasco_path).name[:68]
    hdr["BLENDIN"] = float(blend_inner_rsun)
    hdr["BLENDOUT"] = float(blend_outer_rsun)
    hdr["KCORSCAL"] = (float(kcor_scale), "Multiplicative scale applied to reprojected K-Cor pB")
    hdr["KCORSPX"] = (int(scale_npix), "Pixels used for K-Cor/LASCO overlap scale")
    hdr["KCORBPX"] = (int(n_overlap), "Pixels blended using valid K-Cor and LASCO pB")
    hdr["KCOREDPX"] = (int(edge_smooth_pixels), "Pixels edge-smoothed at K-Cor/LASCO boundary")
    hdr["KCOREDR"] = (float(edge_smooth_width_rsun), "Edge-smoothing width [Rsun]")
    hdr["KCOREDGE"] = (float(edge_smooth_radius) if np.isfinite(edge_smooth_radius) else -1.0, "Edge-smoothing center radius [Rsun]")
    hdr["KCOREDMN"] = (float(edge_smooth_min_rsun) if edge_smooth_min_rsun is not None and np.isfinite(float(edge_smooth_min_rsun)) else -1.0, "Edge-smoothing minimum radius [Rsun]")
    hdr["KCOREDMX"] = (float(edge_smooth_max_rsun) if edge_smooth_max_rsun is not None and np.isfinite(float(edge_smooth_max_rsun)) else -1.0, "Edge-smoothing maximum radius [Rsun]")
    hdr["KCORBDPX"] = (int(boundary_pixels), "Detected K-Cor/C2 boundary pixels")
    hdr["KCORBDW"] = (float(boundary_width_rsun), "Boundary radial smoothing width [Rsun]")
    hdr["KCORBDTH"] = (float(boundary_weight_threshold), "K-Cor weight threshold for boundary detection")
    hdr["KCORBDAD"] = (float(boundary_angular_bin_deg), "Boundary angular bin size [deg]")
    hdr["KCORBDMN"] = (float(boundary_smooth_min_actual) if np.isfinite(boundary_smooth_min_actual) else -1.0, "Actual boundary-smoothing min radius [Rsun]")
    hdr["KCORBDMX"] = (float(boundary_smooth_max_actual) if np.isfinite(boundary_smooth_max_actual) else -1.0, "Actual boundary-smoothing max radius [Rsun]")
    hdr["KCORBFR"] = (float(overlap_fraction), "K-Cor/LASCO overlap coverage fraction")
    hdr["KCORBAPX"] = (int(n_overlap_possible), "Available pixels in nominal blend annulus")
    if edge_smooth_pixels > 0 and boundary_pixels > 0:
        hdr["KCORSMET"] = ("BOUND_SM", "K-Cor/C2 boundary-detected radial smoothing")
    elif edge_smooth_pixels > 0 and n_overlap > 0:
        hdr["KCORSMET"] = ("BLEND+SM", "K-Cor/LASCO blend plus annular smoothing")
    elif edge_smooth_pixels > 0:
        hdr["KCORSMET"] = ("EDGE_SM", "Local edge smoothing applied")
    elif n_overlap > 0:
        hdr["KCORSMET"] = ("BLEND", "K-Cor/LASCO overlap blending applied")
    else:
        hdr["KCORSMET"] = ("NONE", "No adequate overlap; hard switch/no edge smoothing")
    hdr["KCORUPX"] = (int(n_used), "Pixels where K-Cor contributed to combined pB")
    hdr["KCORDIFF"] = (float(max_diff), "Max abs difference between combined and LASCO pB")
    hdr["KCORRMIN"] = (float(kcor_rmin), "Minimum radius where K-Cor may be used [Rsun]")
    hdr["KCORRMAX"] = (float(kcor_rmax), "Maximum radius where K-Cor may be used [Rsun]")
    hdr["NANRMIN"] = (float(kcor_rmin), "Pixels below this radius are written as NaN [Rsun]")
    hdr["INNERNPX"] = (int(n_inner_no_data), "Pixels set to NaN below NANRMIN")
    hdr["BADGEOPX"] = (int(n_bad_geometry), "Pixels set to NaN due to invalid geometry")
    hdr["KCORUNIT"] = str(kcor_hdr.get("BUNIT", ""))[:68]
    hdr["LASCOUNI"] = str(lasco_hdr.get("BUNIT", ""))[:68]
    hdr["HISTORY"] = "LASCO-C2 and K-Cor were reprojected to an Earth-view rotation-free output grid."

    fits.writeto(out, combined.astype(np.float32), hdr, overwrite=True)
    print(f"[OK] Combined K-Cor/LASCO pB: {out} (K-Cor used pixels={n_used}, max_diff={max_diff:.3e})")
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
    except Exception as exc:
        print(f"[SKIP] Could not copy LASCO-only pB {Path(lasco_path).name}: {exc}")
        return None

    hdr = hdr.copy()
    # LASCO-only fallback products also need observer coordinates for rotational tomography.
    add_observer_geometry_keywords(
        out_hdr=hdr,
        source_hdr=hdr,
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
    edge_smooth_when_no_blend: bool = True,
    edge_smooth_width_rsun: float = 0.03,
    edge_smooth_log_space: bool = True,
    edge_smooth_min_blend_pixels: int = 20,
    edge_smooth_min_blend_fraction: float = 0.10,
    edge_smooth_always: bool = True,
    edge_smooth_min_rsun: Optional[float] = None,
    edge_smooth_max_rsun: Optional[float] = None,
    boundary_smooth: bool = True,
    boundary_width_rsun: float = 0.35,
    boundary_angular_bin_deg: float = 2.0,
    boundary_weight_threshold: float = 0.5,
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
            edge_smooth_when_no_blend=edge_smooth_when_no_blend,
            edge_smooth_width_rsun=edge_smooth_width_rsun,
            edge_smooth_log_space=edge_smooth_log_space,
            edge_smooth_min_blend_pixels=edge_smooth_min_blend_pixels,
            edge_smooth_min_blend_fraction=edge_smooth_min_blend_fraction,
            edge_smooth_always=edge_smooth_always,
            edge_smooth_min_rsun=edge_smooth_min_rsun,
            edge_smooth_max_rsun=edge_smooth_max_rsun,
            boundary_smooth=boundary_smooth,
            boundary_width_rsun=boundary_width_rsun,
            boundary_angular_bin_deg=boundary_angular_bin_deg,
            boundary_weight_threshold=boundary_weight_threshold,
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
    KCOR_LASCO_BLEND_OUTER_RSUN = 3
    CALIBRATE_KCOR_TO_LASCO = True
    MIN_KCOR_USED_PIXELS = 100

    # Use the same physically valid K-Cor radial domain as the normal product.
    KCOR_USE_MIN_RSUN = 1.1
    KCOR_USE_MAX_RSUN = KCOR_LASCO_BLEND_OUTER_RSUN

    # If the nominal 2.2--2.5 Rs overlap has too few valid K-Cor pixels,
    # suppress the hard K-Cor/LASCO edge by locally averaging only a narrow annulus.
    EDGE_SMOOTH_WHEN_NO_BLEND = True
    EDGE_SMOOTH_WIDTH_RSUN = 0.2
    # Legacy fallback annulus.  Keep both as None when using boundary-based smoothing.
    EDGE_SMOOTH_MIN_RSUN = None
    EDGE_SMOOTH_MAX_RSUN = None
    EDGE_SMOOTH_LOG_SPACE = True

    # Boundary-based radial smoothing.  The K-Cor/C2 seam is detected from the
    # actual K-Cor contribution weight map, then smoothed radially within
    # BOUNDARY_WIDTH_RSUN around that local boundary.
    BOUNDARY_SMOOTH = True
    BOUNDARY_WIDTH_RSUN = 0.35
    BOUNDARY_ANGULAR_BIN_DEG = 2.0
    BOUNDARY_WEIGHT_THRESHOLD = 0.5
    EDGE_SMOOTH_MIN_BLEND_PIXELS = 20
    EDGE_SMOOTH_MIN_BLEND_FRACTION = 0.10
    EDGE_SMOOTH_ALWAYS = True

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
        edge_smooth_when_no_blend=EDGE_SMOOTH_WHEN_NO_BLEND,
        edge_smooth_width_rsun=EDGE_SMOOTH_WIDTH_RSUN,
        edge_smooth_log_space=EDGE_SMOOTH_LOG_SPACE,
        edge_smooth_min_blend_pixels=EDGE_SMOOTH_MIN_BLEND_PIXELS,
        edge_smooth_min_blend_fraction=EDGE_SMOOTH_MIN_BLEND_FRACTION,
        edge_smooth_always=EDGE_SMOOTH_ALWAYS,
        edge_smooth_min_rsun=EDGE_SMOOTH_MIN_RSUN,
        edge_smooth_max_rsun=EDGE_SMOOTH_MAX_RSUN,
        boundary_smooth=BOUNDARY_SMOOTH,
        boundary_width_rsun=BOUNDARY_WIDTH_RSUN,
        boundary_angular_bin_deg=BOUNDARY_ANGULAR_BIN_DEG,
        boundary_weight_threshold=BOUNDARY_WEIGHT_THRESHOLD,
        lasco_fallback_hhmm_list=LASCO_FALLBACK_HHMM_LIST,
        kcor_cookie_file=KCOR_COOKIE_FILE,
        kcor_product=KCOR_PRODUCT,
        use_lasco_only_when_no_kcor=USE_LASCO_ONLY_WHEN_NO_KCOR,
    )

    print(f"[INFO] Prepared K-Cor/LASCO tomography-ready pB files: {len(outputs)}")
    for path in outputs:
        print(f"       {path}")
