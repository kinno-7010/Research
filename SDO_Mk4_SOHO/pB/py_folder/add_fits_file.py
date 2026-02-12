#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inject observer-geometry keywords into a FITS header using SOHO ancillary orbit record.

Target use case:
  pB_Kcor_LASCO_axi_20220613_0300.fits (combined Earth-view pB map) lacks DSUN_OBS/HGLN_OBS/HGLT_OBS etc.

This script:
  - parses the orbit record you provided (2022-06-13 03:00:00 UT)
  - computes DSUN_OBS [m], CRLN_OBS/CRLT_OBS [deg], HGLN_OBS/HGLT_OBS [deg], CAR_ROT
  - writes them into the FITS header (with backup)

Default observer mode: "earth"
  - DSUN_OBS is derived from the GCI Sun vector norm (Earth–Sun distance).
  - HGLN_OBS is set to 0 deg (Stonyhurst definition for Earth viewpoint).
  - HGLT_OBS uses the orbit "Earth heliographic latitude" (rad→deg).

Option observer mode: "soho"
  - DSUN_OBS is derived from HEC S/C position norm (Sun–SOHO distance).
  - CRLN/CRLT use S/C values from the orbit line.
  - HGLN_OBS is computed as (CRLN_SC - CRLN_EARTH) wrapped to [-180, 180] deg.
  - HGLT_OBS uses S/C heliographic latitude (rad→deg).

Notes:
  - This script does NOT alter image WCS keywords (CDELT/CRPIX/CROTA/PC/CD).
  - It adds HISTORY lines recording both Earth and SOHO values for traceability.
"""

import argparse
import shutil
import math
from pathlib import Path

import numpy as np
from astropy.io import fits


# ===== Orbit record values you pasted (2022-06-13 03:00:00.000 UT) =====
# GCI Sun vector (km)  [columns 23-25 in the orbit record]
GCI_SUN_VEC_KM = np.array([21639103.96194, 137961719.95592, 59805297.21122], dtype=float)

# HEC spacecraft (SOHO) position (km) [columns 26-28 in the orbit record]
HEC_SC_POS_KM = np.array([-21031000.00116, -149148164.88366, 79145.79911], dtype=float)

# Carrington rotation numbers (Earth, S/C)
CAR_ROT_EARTH = 2258
CAR_ROT_SC = 2258

# Heliographic lon/lat (Earth, S/C) in radians [last 6 columns in the orbit record]
EARTH_LON_RAD = 2.624
EARTH_LAT_RAD = 0.013
SC_LON_RAD = 2.627
SC_LAT_RAD = 0.014


def norm_km(vec_km: np.ndarray) -> float:
    return float(np.linalg.norm(vec_km))


def rad2deg(x_rad: float) -> float:
    return x_rad * 180.0 / math.pi


def wrap180(deg: float) -> float:
    """Wrap an angle in degrees to [-180, 180)."""
    return (deg + 180.0) % 360.0 - 180.0


def compute_orbit_derived_values():
    """
    Compute observer-relevant values from the orbit record.
    Returns a dict with both Earth and SOHO (S/C) values.
    """
    dsun_earth_km = norm_km(GCI_SUN_VEC_KM)   # Earth–Sun distance
    dsun_soho_km = norm_km(HEC_SC_POS_KM)     # Sun–SOHO distance

    earth_lon_deg = rad2deg(EARTH_LON_RAD)
    earth_lat_deg = rad2deg(EARTH_LAT_RAD)
    soho_lon_deg = rad2deg(SC_LON_RAD)
    soho_lat_deg = rad2deg(SC_LAT_RAD)

    # Stonyhurst lon for SOHO relative to Earth (approx, good for near-Earth spacecraft)
    hglon_soho_deg = wrap180(soho_lon_deg - earth_lon_deg)

    out = {
        "dsun_earth_m": dsun_earth_km * 1000.0,
        "dsun_soho_m": dsun_soho_km * 1000.0,
        "crln_earth_deg": earth_lon_deg,
        "crlt_earth_deg": earth_lat_deg,
        "crln_soho_deg": soho_lon_deg,
        "crlt_soho_deg": soho_lat_deg,
        "car_rot_earth": CAR_ROT_EARTH,
        "car_rot_soho": CAR_ROT_SC,
        "hglon_earth_deg": 0.0,               # definition for Earth viewpoint in Stonyhurst
        "hglt_earth_deg": earth_lat_deg,
        "hglon_soho_deg": hglon_soho_deg,
        "hglt_soho_deg": soho_lat_deg,
    }
    return out


def update_header_inplace(fits_path: Path, observer_mode: str, overwrite_carrington: bool):
    vals = compute_orbit_derived_values()

    # Decide which set becomes the *standard* observer keywords
    if observer_mode.lower() == "earth":
        dsun_obs = vals["dsun_earth_m"]
        hglon_obs = vals["hglon_earth_deg"]
        hglt_obs = vals["hglt_earth_deg"]
        crln_obs = vals["crln_earth_deg"]
        crlt_obs = vals["crlt_earth_deg"]
        car_rot = vals["car_rot_earth"]
        obs_tag = "EARTH"
    elif observer_mode.lower() == "soho":
        dsun_obs = vals["dsun_soho_m"]
        hglon_obs = vals["hglon_soho_deg"]
        hglt_obs = vals["hglt_soho_deg"]
        crln_obs = vals["crln_soho_deg"]
        crlt_obs = vals["crlt_soho_deg"]
        car_rot = vals["car_rot_soho"]
        obs_tag = "SOHO"
    else:
        raise ValueError("observer_mode must be 'earth' or 'soho'.")

    # Backup
    bak = fits_path.with_suffix(fits_path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(fits_path, bak)

    # Update
    with fits.open(fits_path, mode="update") as hdul:
        hdr = hdul[0].header

        # Insert the essential observer keywords
        hdr["DSUN_OBS"] = (float(dsun_obs), "Distance from observer to Sun center [m]")

        # Stonyhurst observer lon/lat (deg)
        hdr["HGLN_OBS"] = (float(hglon_obs), "Obs. heliographic Stonyhurst lon [deg]")
        hdr["HGLT_OBS"] = (float(hglt_obs), "Obs. heliographic Stonyhurst lat [deg]")

        # Carrington rotation number
        hdr["CAR_ROT"] = (int(car_rot), "Carrington rotation number at DATE-OBS/AVG")

        # Carrington lon/lat: overwrite only if user explicitly requests,
        # otherwise keep existing values to avoid unintended inconsistency with prior processing.
        if overwrite_carrington or ("CRLN_OBS" not in hdr):
            hdr["CRLN_OBS"] = (float(crln_obs), "Obs. Carrington heliographic lon [deg]")
        if overwrite_carrington or ("CRLT_OBS" not in hdr):
            hdr["CRLT_OBS"] = (float(crlt_obs), "Obs. Carrington heliographic lat [deg]")

        # Optional: tag the observatory if absent
        if "OBSRVTRY" not in hdr:
            hdr["OBSRVTRY"] = (obs_tag, "Observer tag inserted by orbit-based patch")

        # Traceability: record both Earth and SOHO values
        hdr.add_history("Inserted observer keywords from SOHO ancillary ORBIT predictive record:")
        hdr.add_history("Orbit timestamp: 2022-06-13T03:00:00.000Z (user-provided line)")
        hdr.add_history(
            f"EARTH: DSUN={vals['dsun_earth_m']:.6f} m, CRLN={vals['crln_earth_deg']:.9f} deg, "
            f"CRLT={vals['crlt_earth_deg']:.9f} deg, CAR_ROT={vals['car_rot_earth']}"
        )
        hdr.add_history(
            f"SOHO : DSUN={vals['dsun_soho_m']:.6f} m, CRLN={vals['crln_soho_deg']:.9f} deg, "
            f"CRLT={vals['crlt_soho_deg']:.9f} deg, HGLN~{vals['hglon_soho_deg']:.9f} deg, "
            f"HGLT={vals['hglt_soho_deg']:.9f} deg, CAR_ROT={vals['car_rot_soho']}"
        )
        hdr.add_history(f"Standard observer mode applied: {observer_mode.upper()}")

        hdul.flush()

    # Print a concise report
    print("Updated FITS:", str(fits_path))
    print("Backup saved:", str(bak))
    print("Applied observer mode:", observer_mode.upper())
    print("Written keywords (standard):")
    print(f"  DSUN_OBS = {dsun_obs:.6f}  [m]")
    print(f"  HGLN_OBS = {hglon_obs:.9f} [deg]")
    print(f"  HGLT_OBS = {hglt_obs:.9f} [deg]")
    print(f"  CAR_ROT  = {car_rot:d}")
    if overwrite_carrington:
        print("  (Overwrote CRLN_OBS/CRLT_OBS)")
        print(f"  CRLN_OBS = {crln_obs:.9f} [deg]")
        print(f"  CRLT_OBS = {crlt_obs:.9f} [deg]")
    else:
        print("  (CRLN_OBS/CRLT_OBS left as-is unless missing)")


def main():
    parser = argparse.ArgumentParser(
        description="Inject DSUN_OBS/HGLN_OBS/HGLT_OBS/CAR_ROT into a FITS header using SOHO orbit record."
    )
    parser.add_argument(
        "--fits",
        default="/mnt/d/wsl/home/kinno-7010/Research/Tomography/Rawdata/pB_Kcor_LASCO_axi_20220613_0300.fits",
        help="Target FITS path to update in place."
    )
    parser.add_argument(
        "--observer",
        default="earth",
        choices=["earth", "soho"],
        help="Which observer to write as standard keywords (default: earth)."
    )
    parser.add_argument(
        "--overwrite-carrington",
        action="store_true",
        help="If set, overwrite CRLN_OBS/CRLT_OBS with orbit-derived values (otherwise only fill if missing)."
    )

    args = parser.parse_args()
    fits_path = Path(args.fits)

    if not fits_path.exists():
        raise FileNotFoundError(f"FITS not found: {fits_path}")

    update_header_inplace(
        fits_path=fits_path,
        observer_mode=args.observer,
        overwrite_carrington=args.overwrite_carrington
    )


if __name__ == "__main__":
    main()
