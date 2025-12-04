# limb_dark.py

import numpy as np
# set_u_from_instrument.py

from pathlib import Path
from typing import Literal

def limb_dark(lambda_nm, meanflux=False, allen=False):
    """
    Python version of the SSWIDL LIMB_DARK function.

    Parameters
    ----------
    lambda_nm : float or array_like
        Wavelength in nanometers (nm). Must be within 305–995 nm.
    meanflux : bool, optional
        If True, return the ratio F/I(0) (mean flux to disk-center intensity),
        i.e., the quantity tabulated by Pierce & Allen.
        If False (default), return the limb darkening coefficient
        u = 3*(1 - F/I(0)).
    allen : bool, optional
        If True, use the values from Allen "Astrophysical Quantities"
        (same as IDL keyword /ALLEN).
        If False (default), use the values derived from Pierce & Allen (1977).

    Returns
    -------
    out : float or ndarray
        Limb darkening coefficient u (default) or mean-flux ratio F/I(0)
        if meanflux=True. The shape follows the input `lambda_nm`.

    Raises
    ------
    ValueError
        If any wavelength is outside the tabulated range (305–995 nm).

    Notes
    -----
    Direct translation of SSWIDL limb_dark.pro:

        FUNCTION LIMB_DARK, Lambda, MEANFLUX=meanflux, ALLEN=allen

    with the same tables and interpolation behavior.
    """
    scalar_input = np.isscalar(lambda_nm)
    lam = np.asarray(lambda_nm, dtype=float)

    # Choose table: ALLEN vs Pierce & Allen
    if allen:
        # Allen "Astrophysical Quantities"
        wl = 10.0 * np.array(
            [30, 32, 35, 37, 38, 40, 45, 50, 55, 60, 80, 100],
            dtype=float
        )  # [nm]
        meantocenter = np.array(
            [648, 685, 705, 710, 710, 718, 755, 782, 803, 817, 862, 886],
            dtype=float
        ) / 1000.0  # F/I(0)
    else:
        # Pierce & Allen (1977) Table 2
        wl = 305.0 + 10.0 * np.arange(70, dtype=float)
        u300 = np.array([0.636, 0.657, 0.674, 0.685, 0.694,
                         0.703, 0.691, 0.680, 0.687, 0.698])
        u400 = np.array([0.710, 0.718, 0.726, 0.734, 0.744,
                         0.753, 0.759, 0.765, 0.769, 0.775])
        u500 = np.array([0.779, 0.784, 0.788, 0.793, 0.798,
                         0.802, 0.805, 0.809, 0.812, 0.815])
        u600 = np.array([0.817, 0.820, 0.823, 0.825, 0.828,
                         0.830, 0.834, 0.836, 0.838, 0.841])
        u700 = np.array([0.843, 0.845, 0.847, 0.849, 0.851,
                         0.853, 0.855, 0.857, 0.859, 0.861])
        u800 = np.array([0.863, 0.864, 0.865, 0.867, 0.868,
                         0.869, 0.870, 0.872, 0.873, 0.874])
        u900 = np.array([0.875, 0.876, 0.877, 0.878, 0.879,
                         0.880, 0.881, 0.882, 0.883, 0.884])
        meantocenter = np.concatenate(
            [u300, u400, u500, u600, u700, u800, u900]
        )  # actually F/I(0)

    # Range check
    if np.any((lam < wl[0]) | (lam > wl[-1])):
        raise ValueError(
            f"lambda outside of table range [{wl[0]:.1f}, {wl[-1]:.1f}] nm: {lambda_nm}"
        )

    # Interpolate F/I(0)
    F_over_I0 = np.interp(lam, wl, meantocenter)

    if meanflux:
        out = F_over_I0
    else:
        # Allen's approximate relation:
        #   u = 3 * (1 - F/I(0))
        out = 3.0 * (1.0 - F_over_I0)

    if scalar_input:
        return float(out)
    return out


def get_lambda_eff_for_instrument(
    instrument: Literal["lasco_c2", "kcor"]
) -> float:
    """
    Return an effective wavelength [nm] for each coronagraph.

    Parameters
    ----------
    instrument : {"lasco_c2", "kcor"}
        Which instrument's pB data you are using.

    Returns
    -------
    lambda_eff_nm : float
        Effective wavelength in nm to be fed into limb_dark().
    """
    inst = instrument.lower()

    if inst == "lasco_c2":
        # LASCO C2 orange filter: effective wavelength.
        # Many works assume ~540–550 nm; here we set 540 nm as default.
        lambda_eff_nm = 540.0
    elif inst == "kcor":
        # K-Cor: header says WAVELNTH=735 nm, FWHM=30 nm
        lambda_eff_nm = 735.0
    else:
        raise ValueError(f"Unknown instrument: {instrument}")

    return lambda_eff_nm


def compute_u_for_instrument(
    instrument: Literal["lasco_c2", "kcor"],
    use_allen: bool = False,
) -> float:
    """
    Compute limb-darkening coefficient u for the given coronagraph.

    Parameters
    ----------
    instrument : {"lasco_c2", "kcor"}
        Instrument name.
    use_allen : bool, optional
        If True, use Allen's table (/ALLEN in IDL).
        If False (default), use Pierce & Allen (1977) table.

    Returns
    -------
    u : float
        Limb-darkening coefficient suitable for the instrument.
    """
    lambda_eff = get_lambda_eff_for_instrument(instrument)
    u_val = limb_dark(lambda_eff, meanflux=False, allen=use_allen)
    return u_val


def update_constants_vdh_u(
    constants_path: str | Path,
    instrument: Literal["lasco_c2", "kcor"],
    use_allen: bool = False,
    backup: bool = True,
) -> float:
    """
    Update the `u` value inside constants_vdh.py based on the instrument.

    This function:
      1. Computes u from the instrument's effective wavelength using limb_dark().
      2. Opens constants_vdh.py as plain text.
      3. Replaces the line starting with 'u =' by `u = <new_value>`.
      4. Optionally writes a backup file constants_vdh.py.bak.

    Parameters
    ----------
    constants_path : str or Path
        Path to constants_vdh.py.
    instrument : {"lasco_c2", "kcor"}
        Instrument for which to choose u.
    use_allen : bool, optional
        If True, use Allen-based u. If False, use Pierce & Allen.
    backup : bool, optional
        If True, save a backup of the original file as constants_vdh.py.bak.

    Returns
    -------
    u_new : float
        The new u value that was written to constants_vdh.py.
    """
    constants_path = Path(constants_path)
    text = constants_path.read_text(encoding="utf-8")

    u_new = compute_u_for_instrument(instrument, use_allen=use_allen)

    # Very simple pattern-based replacement: look for a line starting with 'u ='
    lines = text.splitlines()
    new_lines = []
    replaced = False

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("u ") and "=" in stripped:
            # Replace this line with new u
            indent = line[: len(line) - len(stripped)]
            new_line = f"{indent}u = {u_new:.5f}  # limb-darkening coefficient (auto-set for {instrument})"
            new_lines.append(new_line)
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        raise RuntimeError(
            "Could not find a line starting with 'u =' in constants_vdh.py to replace."
        )

    new_text = "\n".join(new_lines)

    if backup:
        backup_path = constants_path.with_suffix(".py.bak")
        backup_path.write_text(text, encoding="utf-8")

    constants_path.write_text(new_text, encoding="utf-8")

    return u_new


if __name__ == "__main__":
    # Example usage: update u for LASCO C2
    import argparse

    parser = argparse.ArgumentParser(
        description="Update limb-darkening coefficient u in constants_vdh.py"
    )
    parser.add_argument(
        "--constants",
        default="constants_vdh.py",
        help="Path to constants_vdh.py",
    )
    parser.add_argument(
        "--instrument",
        choices=["lasco_c2", "kcor"],
        default="lasco_c2",
        help="Instrument for which to choose u",
    )
    parser.add_argument(
        "--allen",
        action="store_true",
        help="Use Allen's table instead of Pierce & Allen",
    )

    args = parser.parse_args()

    u_new = update_constants_vdh_u(
        constants_path=args.constants,
        instrument=args.instrument,
        use_allen=args.allen,
    )

    print(f"Updated u in {args.constants} for {args.instrument}: u = {u_new:.5f}")
