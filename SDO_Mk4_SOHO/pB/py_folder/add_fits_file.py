from astropy.io import fits
import numpy as np

# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------
P_COMBINED = "/mnt/d/wsl/home/kinno-7010/Research/SDO_Mk4_SOHO/pB/Rawdata/pB_Kcor_LASCO_axi_20220613_0300.fits"

# K-COR元データ（観測者CRLN/CRLTをコピーするため）
P_KCOR_RAW = "/mnt/d/wsl/home/kinno-7010/Research/MK4_coronagraph/MK4_coronagraph_KCOR/pB/Rawdata/20220613_025810_kcor_l2.fts"

# （任意）COR1前処理済みファイルと元COR1（CRLT_OBS等の補完用）
P_COR1_PRE = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata/COR1A_pb_pre_20220613_030100.fits"
P_COR1_RAW = "/mnt/d/wsl/home/kinno-7010/Research/STEREO-A/SECCHI/COR1/Rawdata/20220613_030100_n4c1A.fts"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _get_key(hdr, keys, default=None):
    """Return the first existing header keyword value among `keys`."""
    for k in keys:
        if k in hdr:
            return hdr[k]
    return default


def _set_if_missing(hdr, key, value, comment=None):
    """Set only if key is missing or empty."""
    if key not in hdr or hdr[key] in ("", None):
        if comment is None:
            hdr[key] = value
        else:
            hdr[key] = (value, comment)
        return True
    return False


def _force_set(hdr, key, value, comment=None):
    """Always set/overwrite."""
    if comment is None:
        hdr[key] = value
    else:
        hdr[key] = (value, comment)


def _pc_from_crota_deg(crota_deg):
    th = np.deg2rad(float(crota_deg))
    c, s = float(np.cos(th)), float(np.sin(th))
    # WCS PC matrix for a rotation by +theta
    return c, -s, s, c


def patch_combined_header(p_combined, p_kcor_raw):
    """
    Patch pB_Kcor_LASCO_axi_*.fits:
      - add CRLN_OBS/CRLT_OBS from K-COR raw
      - ensure HPC WCS keywords exist: CTYPE, CUNIT
      - ensure PC matrix exists (from CROTA2 if possible)
      - ensure RSUN/RSUN_OBS exist (keep if already present)
    """
    # Read observer lon/lat from K-COR raw
    with fits.open(p_kcor_raw) as hk:
        hk_hdr = hk[0].header
        crln = _get_key(hk_hdr, ["CRLN_OBS"])
        crlt = _get_key(hk_hdr, ["CRLT_OBS"])
        if crln is None or crlt is None:
            raise RuntimeError(
                f"K-COR raw file lacks CRLN_OBS/CRLT_OBS: {p_kcor_raw}"
            )

        # Use K-COR solar apparent radius if present
        rsun = _get_key(hk_hdr, ["RSUN_OBS", "RSUN"])

    with fits.open(p_combined, mode="update") as hdul:
        hdr = hdul[0].header

        changed = []

        # 1) Observer Carrington lon/lat (critical for your tomography code)
        if _set_if_missing(hdr, "CRLN_OBS", float(crln), "Carrington lon of observer [deg]"):
            changed.append("CRLN_OBS")
        if _set_if_missing(hdr, "CRLT_OBS", float(crlt), "Carrington lat of observer [deg]"):
            changed.append("CRLT_OBS")

        # 2) Ensure HPC WCS types exist (many tools require these)
        if _set_if_missing(hdr, "CTYPE1", "HPLN-TAN", "Helioprojective longitude"):
            changed.append("CTYPE1")
        if _set_if_missing(hdr, "CTYPE2", "HPLT-TAN", "Helioprojective latitude"):
            changed.append("CTYPE2")

        # 3) Units: normalize to 'arcsec' (astropy warns on 'ARCSEC')
        #    Overwrite to be safe.
        _force_set(hdr, "CUNIT1", "arcsec", "Axis unit")
        _force_set(hdr, "CUNIT2", "arcsec", "Axis unit")
        changed.append("CUNIT1/2")

        # 4) Rotation: prefer CROTA2 -> PC matrix.
        #    If CROTA2 missing, keep existing PC if present; else identity.
        crota2 = _get_key(hdr, ["CROTA2", "CROTA", "CROTA1"], default=None)

        has_pc = all(k in hdr for k in ["PC1_1", "PC1_2", "PC2_1", "PC2_2"])
        if crota2 is not None:
            pc11, pc12, pc21, pc22 = _pc_from_crota_deg(crota2)
            _force_set(hdr, "PC1_1", pc11, "WCS rotation matrix")
            _force_set(hdr, "PC1_2", pc12, "WCS rotation matrix")
            _force_set(hdr, "PC2_1", pc21, "WCS rotation matrix")
            _force_set(hdr, "PC2_2", pc22, "WCS rotation matrix")
            changed.append("PC* (from CROTA)")
        else:
            if not has_pc:
                _force_set(hdr, "PC1_1", 1.0, "WCS rotation matrix")
                _force_set(hdr, "PC1_2", 0.0, "WCS rotation matrix")
                _force_set(hdr, "PC2_1", 0.0, "WCS rotation matrix")
                _force_set(hdr, "PC2_2", 1.0, "WCS rotation matrix")
                changed.append("PC* (identity)")

        # 5) RSUN / RSUN_OBS: keep if present, otherwise set from K-COR (or fallback 944.95)
        if rsun is None:
            rsun = 944.95  # your known value
        if _set_if_missing(hdr, "RSUN_OBS", float(rsun), "Apparent solar radius [arcsec]"):
            changed.append("RSUN_OBS")
        if _set_if_missing(hdr, "RSUN", float(rsun), "Apparent solar radius [arcsec]"):
            changed.append("RSUN")

        hdr.add_history("Patched header for tomography: added CRLN/CRLT, CTYPE, CUNIT, PC matrix, RSUN.")
        hdul.flush()

    print("[OK] Patched:", p_combined)
    if changed:
        print("     Updated keys:", ", ".join(changed))
    else:
        print("     No changes (all required keys already present).")


def patch_cor1_pre_if_needed(p_cor1_pre, p_cor1_raw):
    """
    Optional safety patch:
    If COR1 preprocessed file lacks CRLT_OBS (or CRLN_OBS), copy from COR1 raw.
    This avoids the situation where code falls back to HGLN/HGLT unintentionally.
    """
    with fits.open(p_cor1_raw) as hr:
        r_hdr = hr[0].header
        crln = _get_key(r_hdr, ["CRLN_OBS"])
        crlt = _get_key(r_hdr, ["CRLT_OBS"])

    if crln is None and crlt is None:
        print("[SKIP] COR1 raw lacks CRLN_OBS/CRLT_OBS; cannot patch pre file.")
        return

    with fits.open(p_cor1_pre, mode="update") as hdul:
        hdr = hdul[0].header
        changed = []
        if crln is not None and _set_if_missing(hdr, "CRLN_OBS", float(crln), "Carrington lon of observer [deg]"):
            changed.append("CRLN_OBS")
        if crlt is not None and _set_if_missing(hdr, "CRLT_OBS", float(crlt), "Carrington lat of observer [deg]"):
            changed.append("CRLT_OBS")

        # RSUN_OBS is optional but improves tool interoperability
        if "RSUN" in hdr and _set_if_missing(hdr, "RSUN_OBS", float(hdr["RSUN"]), "Apparent solar radius [arcsec]"):
            changed.append("RSUN_OBS")

        if changed:
            hdr.add_history("Patched COR1 pre header: ensured CRLN/CRLT (and RSUN_OBS).")
            hdul.flush()
            print("[OK] Patched:", p_cor1_pre)
            print("     Updated keys:", ", ".join(changed))
        else:
            print("[OK] COR1 pre already has CRLN/CRLT (no patch needed).")


if __name__ == "__main__":
    patch_combined_header(P_COMBINED, P_KCOR_RAW)

    # Optional but recommended (safe-guard)
    # Comment out if you do not want to touch COR1 preprocessed FITS.
    patch_cor1_pre_if_needed(P_COR1_PRE, P_COR1_RAW)
