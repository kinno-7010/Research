#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix stale raw-image FITS scaling keywords in COR1A_pb_pre_*.fits files.

Problem fixed
-------------
Some pB FITS files created from COR1 raw headers may inherit integer-image
keywords such as BZERO=32768, BSCALE, and BLANK.  The image itself is already a
floating-point pB array, so those keywords are not appropriate.  Astropy may add
BZERO to the physical pB values when reading the file.

This script rewrites the primary HDU using the stored floating-point array
without applying FITS scaling, removes stale scaling/statistics keywords, and
adds correct pB statistics.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from astropy.io import fits

DEFAULT_PB_DIR = Path("/mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/COR1/pB/Rawdata")

REMOVE_KEYS = [
    "BZERO", "BSCALE", "BLANK",
    "DATAMIN", "DATAMAX", "DATAAVG", "DATASIG",
    "DATAP01", "DATAP10", "DATAP25", "DATAP50", "DATAP75",
    "DATAP90", "DATAP95", "DATAP98", "DATAP99",
    "DATAZER", "DATASAT", "DSATVAL",
]


def fix_one(path: Path, *, backup: bool = True, dry_run: bool = False) -> bool:
    with fits.open(path, memmap=False, do_not_scale_image_data=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        hdr = hdul[0].header.copy()

    old_bzero = hdr.get("BZERO", None)
    old_blank = hdr.get("BLANK", None)
    has_problem_keys = any(key in hdr for key in ("BZERO", "BLANK"))

    for key in REMOVE_KEYS:
        if key in hdr:
            del hdr[key]

    finite = np.isfinite(data)
    if np.count_nonzero(finite) > 0:
        vals = data[finite].astype(np.float64)
        hdr["DATAMIN"] = float(np.nanmin(vals))
        hdr["DATAMAX"] = float(np.nanmax(vals))
        hdr["DATAAVG"] = float(np.nanmean(vals))
        hdr["DATASIG"] = float(np.nanstd(vals))
    hdr["BUNIT"] = "pB"
    hdr.add_history("Removed stale raw-image BZERO/BSCALE/BLANK keywords from derived pB FITS.")

    print(
        f"[FIX] {path.name}: "
        f"old BZERO={old_bzero}, old BLANK={old_blank}, "
        f"data min={np.nanmin(data):.3e}, median={np.nanmedian(data):.3e}, max={np.nanmax(data):.3e}"
    )

    if dry_run:
        return has_problem_keys

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)
    return has_problem_keys


def main() -> int:
    parser = argparse.ArgumentParser(description="Fix COR1A_pb_pre_*.fits pB headers.")
    parser.add_argument("--pb-dir", type=Path, default=DEFAULT_PB_DIR)
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak files.")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be fixed.")
    args = parser.parse_args()

    files = sorted(args.pb_dir.glob("COR1A_pb_pre_*.fits"))
    if not files:
        print(f"[ERROR] No COR1A_pb_pre_*.fits files found in {args.pb_dir}")
        return 1

    n_problem = 0
    for path in files:
        if fix_one(path, backup=not args.no_backup, dry_run=args.dry_run):
            n_problem += 1

    print(f"[INFO] Finished. Files checked: {len(files)}, files with stale scaling keywords: {n_problem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
