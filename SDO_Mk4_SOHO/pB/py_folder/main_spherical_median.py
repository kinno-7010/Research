# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import importlib.util
import sys
from astropy.io import fits
from matplotlib.colors import LogNorm

from io_and_processing import (
    load_and_prepare_instrument_data, combine_corona_data, extract_pB_profile
)
from plotting_utils import (
    plot_combined_image, generate_ne_profile_plot
)
from constants_vdh import (
    invert_ablation, triple_power, density_from_frequency, set_u_from_instrument, set_u
)


def _fill_nan_1d(arr):
    """1次元配列の NaN を端の最近傍 + 内部線形補間で埋める（pB_multi_line_main.py と同様）。"""
    filled = np.array(arr, copy=True)
    idx = np.arange(filled.size)
    finite_mask = np.isfinite(filled)
    if not np.any(finite_mask):
        return filled
    finite_idx = idx[finite_mask]
    finite_val = filled[finite_mask]
    if finite_idx[0] > 0:
        filled[:finite_idx[0]] = finite_val[0]
    if finite_idx[-1] < filled.size - 1:
        filled[finite_idx[-1] + 1:] = finite_val[-1]
    nan_mask = ~finite_mask
    if np.any(nan_mask):
        filled[nan_mask] = np.interp(idx[nan_mask], finite_idx, finite_val)
    return filled




def extract_azimuthally_averaged_pB_profile(
    image,
    r_map,
    edges,
    cy,
    cx,
    angle_bin_deg=2.0,
    sigma_clip=3.0,
    min_angular_coverage=0.50,
):
    """
    各PA sectorのradial pB profileと、全方位の代表統計量を作る。

    各半径bin・PA sector内ではpixel中央値を採用する。全方位の代表pBは、
    PA-sector値をMAD clippingした後の中央値とする。一方、角度方向の
    最小値・最大値はclipping前の全有効sectorから計算し、実際の方位角差を
    保持する。

    Returns
    -------
    pB_sector_profiles : ndarray, shape (n_angle_bins, n_radial_bins)
        各PA sectorのradial pB profile。
    pB_median : ndarray
        各半径binにおけるPA-sector値のrobust median。
    pB_min : ndarray
        各半径binにおける全有効PA sectorの最小値。
    pB_max : ndarray
        各半径binにおける全有効PA sectorの最大値。
    pB_sector_mad : ndarray
        robust median計算に採用したsector値のscaled MAD。
    angular_coverage : ndarray
        clipping前に有効だったPA sectorの割合（0–1）。
    n_valid_sectors : ndarray
        clipping前に有効だったPA sector数。
    angle_centers_deg : ndarray
        PA sector中心角 [deg]。
    """
    image = np.asarray(image, dtype=np.float64)
    r_map = np.asarray(r_map, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)

    if image.shape != r_map.shape:
        raise ValueError(
            f"image and r_map must have the same shape: {image.shape} != {r_map.shape}"
        )
    if len(edges) < 2:
        raise ValueError("edges must contain at least two radial boundaries.")
    if not np.isfinite(angle_bin_deg) or angle_bin_deg <= 0 or angle_bin_deg > 360:
        raise ValueError(f"Invalid angle_bin_deg={angle_bin_deg}")
    if not 0.0 <= float(min_angular_coverage) <= 1.0:
        raise ValueError(
            f"min_angular_coverage must be in [0, 1], got {min_angular_coverage}"
        )

    yy, xx = np.indices(image.shape, dtype=np.float64)
    pa_deg = (
        np.degrees(np.arctan2(yy - float(cy), xx - float(cx))) + 360.0
    ) % 360.0

    n_angle_bins = max(1, int(np.ceil(360.0 / float(angle_bin_deg))))
    angle_edges = np.linspace(0.0, 360.0, n_angle_bins + 1)
    angle_centers_deg = 0.5 * (angle_edges[:-1] + angle_edges[1:])
    angle_index = np.clip(
        np.digitize(pa_deg, angle_edges, right=False) - 1,
        0,
        n_angle_bins - 1,
    )

    n_radial_bins = len(edges) - 1
    pB_sector_profiles = np.full(
        (n_angle_bins, n_radial_bins), np.nan, dtype=np.float64
    )
    pB_median = np.full(n_radial_bins, np.nan, dtype=np.float64)
    pB_min = np.full(n_radial_bins, np.nan, dtype=np.float64)
    pB_max = np.full(n_radial_bins, np.nan, dtype=np.float64)
    pB_sector_mad = np.full(n_radial_bins, np.nan, dtype=np.float64)
    angular_coverage = np.zeros(n_radial_bins, dtype=np.float64)
    n_valid_sectors = np.zeros(n_radial_bins, dtype=np.int32)

    finite_image = np.isfinite(image)
    for ir in range(n_radial_bins):
        annulus = (
            finite_image
            & (r_map >= edges[ir])
            & (r_map < edges[ir + 1])
        )
        if not np.any(annulus):
            continue

        for ia in range(n_angle_bins):
            values = image[annulus & (angle_index == ia)]
            values = values[np.isfinite(values)]
            if values.size > 0:
                pB_sector_profiles[ia, ir] = float(np.nanmedian(values))

        sector_values = pB_sector_profiles[:, ir]
        sector_values = sector_values[np.isfinite(sector_values)]
        coverage = sector_values.size / float(n_angle_bins)
        angular_coverage[ir] = coverage
        n_valid_sectors[ir] = int(sector_values.size)
        if sector_values.size == 0 or coverage < float(min_angular_coverage):
            continue

        pB_min[ir] = float(np.nanmin(sector_values))
        pB_max[ir] = float(np.nanmax(sector_values))

        center = float(np.nanmedian(sector_values))
        mad_all = 1.4826 * float(np.nanmedian(np.abs(sector_values - center)))
        if (
            np.isfinite(mad_all)
            and mad_all > 0
            and np.isfinite(sigma_clip)
            and sigma_clip > 0
        ):
            keep = np.abs(sector_values - center) <= float(sigma_clip) * mad_all
            clipped = sector_values[keep]
        else:
            clipped = sector_values

        if clipped.size == 0:
            continue

        center_clipped = float(np.nanmedian(clipped))
        pB_median[ir] = center_clipped
        pB_sector_mad[ir] = 1.4826 * float(
            np.nanmedian(np.abs(clipped - center_clipped))
        )

    return (
        pB_sector_profiles,
        pB_median,
        pB_min,
        pB_max,
        pB_sector_mad,
        angular_coverage,
        n_valid_sectors,
        angle_centers_deg,
    )


def _smooth_kcor_density_weight(r_mid, blend_inner, blend_outer):
    """
    Return the K-Cor density weight for a smooth K-Cor/LASCO transition.

    The weight is 1 below blend_inner, 0 above blend_outer, and follows a
    cubic smoothstep inside the overlap.  Both the weight and its first
    derivative are continuous at the overlap boundaries.
    """
    r_mid = np.asarray(r_mid, dtype=np.float64)
    blend_inner = float(blend_inner)
    blend_outer = float(blend_outer)
    if (
        not np.isfinite(blend_inner)
        or not np.isfinite(blend_outer)
        or blend_outer <= blend_inner
    ):
        raise ValueError(
            f"Invalid density blend range: {blend_inner}–{blend_outer} Rsun"
        )

    x = np.clip(
        (r_mid - blend_inner) / (blend_outer - blend_inner),
        0.0,
        1.0,
    )
    lasco_weight = x * x * (3.0 - 2.0 * x)
    return 1.0 - lasco_weight


def invert_azimuthal_sector_profiles(
    pB_sector_profiles,
    r_mid,
    edges,
    r_boundary,
    instrument_kcor,
    instrument_lasco,
    min_profile_coverage=0.50,
    min_valid_sectors_per_radius=10,
    blend_inner=2.0,
    blend_outer=3.0,
):
    """
    Invert each PA-sector pB profile without cutting the Abel inversion at 2.2 Rsun.

    The previous implementation separately inverted the inner profile ending at
    r_boundary and the outer profile beginning immediately outside it.  A van de
    Hulst/Abel inversion is radially non-local: density at radius r depends on pB
    at all projected radii >= r.  Truncating the K-Cor profile at 2.2 Rsun
    therefore creates a terminal inversion artifact at the last inner bins.

    This implementation instead:
      1. fills one complete pB(r) profile for each PA sector;
      2. inverts that complete profile once with the K-Cor limb-darkening
         coefficient and once with the LASCO-C2 coefficient;
      3. combines the two positive density solutions over blend_inner–blend_outer
         using a smooth geometric blend.

    The geometric blend is used because density-scale differences are
    multiplicative.  Below blend_inner the K-Cor solution is used, above
    blend_outer the LASCO solution is used, and no hard numerical cut is made at
    r_boundary.  The r_boundary argument is retained only for backward
    compatibility and for diagnostic output.

    Returns
    -------
    ne_sector_profiles : ndarray
        Smoothly blended PA-sector densities [cm^-3].
    ne_min : ndarray
        Minimum positive density across PA sectors at each radius.
    ne_median : ndarray
        Median positive density across PA sectors at each radius.
    ne_max : ndarray
        Maximum positive density across PA sectors at each radius.
    n_valid_density_sectors : ndarray
        Number of valid PA-sector densities at each radius.
    """
    pB_sector_profiles = np.asarray(pB_sector_profiles, dtype=np.float64)
    r_mid = np.asarray(r_mid, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)

    if pB_sector_profiles.ndim != 2:
        raise ValueError(
            "pB_sector_profiles must be a 2D array with shape "
            "(n_angle_bins, n_radial_bins)."
        )
    if pB_sector_profiles.shape[1] != r_mid.size:
        raise ValueError(
            "pB_sector_profiles and r_mid have inconsistent radial dimensions."
        )
    if edges.size != r_mid.size + 1:
        raise ValueError("edges must have exactly len(r_mid) + 1 elements.")
    if not 0.0 <= float(min_profile_coverage) <= 1.0:
        raise ValueError(
            f"min_profile_coverage must be in [0, 1], got {min_profile_coverage}"
        )

    n_angle_bins, n_radial_bins = pB_sector_profiles.shape
    minimum_count = max(
        5,
        int(np.ceil(float(min_profile_coverage) * n_radial_bins)),
    )

    prepared_profiles = np.full_like(
        pB_sector_profiles,
        np.nan,
        dtype=np.float64,
    )
    valid_sector = np.zeros(n_angle_bins, dtype=bool)
    for ia in range(n_angle_bins):
        raw_profile = pB_sector_profiles[ia]
        if np.count_nonzero(np.isfinite(raw_profile)) < minimum_count:
            continue
        prepared_profiles[ia] = _fill_nan_1d(raw_profile)
        valid_sector[ia] = True

    ne_kcor_full = np.full_like(
        pB_sector_profiles,
        np.nan,
        dtype=np.float64,
    )
    ne_lasco_full = np.full_like(
        pB_sector_profiles,
        np.nan,
        dtype=np.float64,
    )

    set_u_from_instrument(instrument_kcor)
    for ia in np.where(valid_sector)[0]:
        try:
            ne_kcor_full[ia] = invert_ablation(
                prepared_profiles[ia],
                r_mid,
                edges,
                n_radial_bins,
            )
        except Exception as exc:
            print(
                f"[WARN] Full-profile K-Cor-kernel inversion failed "
                f"for PA sector {ia}: {exc}"
            )

    set_u_from_instrument(instrument_lasco)
    for ia in np.where(valid_sector)[0]:
        try:
            ne_lasco_full[ia] = invert_ablation(
                prepared_profiles[ia],
                r_mid,
                edges,
                n_radial_bins,
            )
        except Exception as exc:
            print(
                f"[WARN] Full-profile LASCO-kernel inversion failed "
                f"for PA sector {ia}: {exc}"
            )

    ne_kcor_full[
        ~np.isfinite(ne_kcor_full) | (ne_kcor_full <= 0)
    ] = np.nan
    ne_lasco_full[
        ~np.isfinite(ne_lasco_full) | (ne_lasco_full <= 0)
    ] = np.nan

    kcor_weight = _smooth_kcor_density_weight(
        r_mid,
        blend_inner=blend_inner,
        blend_outer=blend_outer,
    )
    lasco_weight = 1.0 - kcor_weight

    ne_sector_profiles = np.full_like(
        pB_sector_profiles,
        np.nan,
        dtype=np.float64,
    )
    for ia in range(n_angle_bins):
        ne_k = ne_kcor_full[ia]
        ne_l = ne_lasco_full[ia]

        both = (
            np.isfinite(ne_k)
            & (ne_k > 0)
            & np.isfinite(ne_l)
            & (ne_l > 0)
        )
        only_kcor = np.isfinite(ne_k) & (ne_k > 0) & ~both
        only_lasco = np.isfinite(ne_l) & (ne_l > 0) & ~both

        if np.any(both):
            ne_sector_profiles[ia, both] = np.exp(
                kcor_weight[both] * np.log(ne_k[both])
                + lasco_weight[both] * np.log(ne_l[both])
            )
        ne_sector_profiles[ia, only_kcor] = ne_k[only_kcor]
        ne_sector_profiles[ia, only_lasco] = ne_l[only_lasco]

    ne_sector_profiles[
        ~np.isfinite(ne_sector_profiles) | (ne_sector_profiles <= 0)
    ] = np.nan

    print(
        "[INFO] PA-sector density inversion used complete radial profiles; "
        f"K-Cor/LASCO density solutions were smoothly blended over "
        f"{float(blend_inner):.2f}–{float(blend_outer):.2f} Rsun. "
        f"The nominal {float(r_boundary):.2f} Rsun boundary was not used "
        "as an inversion endpoint."
    )

    ne_min = np.full(n_radial_bins, np.nan, dtype=np.float64)
    ne_median = np.full(n_radial_bins, np.nan, dtype=np.float64)
    ne_max = np.full(n_radial_bins, np.nan, dtype=np.float64)
    n_valid_density_sectors = np.zeros(n_radial_bins, dtype=np.int32)

    minimum_sectors = max(1, int(min_valid_sectors_per_radius))
    for ir in range(n_radial_bins):
        values = ne_sector_profiles[:, ir]
        values = values[np.isfinite(values) & (values > 0)]
        n_valid_density_sectors[ir] = int(values.size)
        if values.size < minimum_sectors:
            continue
        ne_min[ir] = float(np.nanmin(values))
        ne_median[ir] = float(np.nanmedian(values))
        ne_max[ir] = float(np.nanmax(values))

    return (
        ne_sector_profiles,
        ne_min,
        ne_median,
        ne_max,
        n_valid_density_sectors,
    )




def _pava_nonincreasing(values, weights=None):
    """Pool-adjacent-violators algorithm for a non-increasing 1D sequence."""
    values = np.asarray(values, dtype=np.float64).ravel()
    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = np.asarray(weights, dtype=np.float64).ravel()
    if values.size != weights.size:
        raise ValueError("values and weights must have the same size.")
    if values.size == 0:
        return values.copy()

    blocks = []
    for i, (value, weight) in enumerate(zip(values, weights)):
        block = {
            "start": i,
            "end": i,
            "weight": float(max(weight, 1.0e-12)),
            "value": float(value),
        }
        blocks.append(block)

        # For a non-increasing sequence, previous >= next must hold.
        while len(blocks) >= 2 and blocks[-2]["value"] < blocks[-1]["value"]:
            right = blocks.pop()
            left = blocks.pop()
            merged_weight = left["weight"] + right["weight"]
            merged_value = (
                left["weight"] * left["value"]
                + right["weight"] * right["value"]
            ) / merged_weight
            blocks.append({
                "start": left["start"],
                "end": right["end"],
                "weight": merged_weight,
                "value": merged_value,
            })

    out = np.empty_like(values)
    for block in blocks:
        out[block["start"]:block["end"] + 1] = block["value"]
    return out




def evaluate_monotonic_loglog_prior(r, fit_result):
    """
    Evaluate the observation-derived prior stored as monotonic log-log PCHIP knots.

    Inside the fitted interval, PCHIP interpolation is used. Outside the interval,
    the endpoint log-log slopes are used for controlled power-law extrapolation.
    """
    from scipy.interpolate import PchipInterpolator

    r = np.asarray(r, dtype=np.float64)
    log_r_knots = np.asarray(fit_result["log_r_knots"], dtype=np.float64)
    log_ne_knots = np.asarray(fit_result["log_ne_knots"], dtype=np.float64)
    if log_r_knots.size < 2 or log_r_knots.size != log_ne_knots.size:
        raise ValueError("Invalid monotonic prior knots.")

    valid_r = np.isfinite(r) & (r > 0)
    log_ne = np.full(r.shape, np.nan, dtype=np.float64)
    if not np.any(valid_r):
        return log_ne

    x = np.log10(r[valid_r])
    interpolator = PchipInterpolator(log_r_knots, log_ne_knots, extrapolate=False)
    y = np.asarray(interpolator(x), dtype=np.float64)

    # Controlled extrapolation. Negative slopes preserve radial decrease.
    slope_inner = (log_ne_knots[1] - log_ne_knots[0]) / (
        log_r_knots[1] - log_r_knots[0]
    )
    slope_outer = (log_ne_knots[-1] - log_ne_knots[-2]) / (
        log_r_knots[-1] - log_r_knots[-2]
    )
    slope_inner = float(np.clip(slope_inner, -30.0, -0.05))
    slope_outer = float(np.clip(slope_outer, -30.0, -0.05))

    inner = x < log_r_knots[0]
    outer = x > log_r_knots[-1]
    if np.any(inner):
        y[inner] = log_ne_knots[0] + slope_inner * (x[inner] - log_r_knots[0])
    if np.any(outer):
        y[outer] = log_ne_knots[-1] + slope_outer * (x[outer] - log_r_knots[-1])

    log_ne[valid_r] = y
    return np.power(10.0, log_ne)




def _normalize_fit_exclude_ranges(exclude_r_ranges):
    """Validate and merge radial intervals excluded from prior fitting."""
    if exclude_r_ranges is None:
        return tuple()

    normalized = []
    for item in exclude_r_ranges:
        if item is None or len(item) != 2:
            raise ValueError(
                "Each excluded fit range must contain exactly two values: "
                "(r_min, r_max)."
            )
        r0, r1 = float(item[0]), float(item[1])
        if not np.isfinite(r0) or not np.isfinite(r1):
            raise ValueError(f"Excluded fit range contains a non-finite value: {item}")
        if r1 <= r0:
            raise ValueError(
                f"Excluded fit range must satisfy r_max > r_min, got {item}"
            )
        normalized.append((r0, r1))

    normalized.sort(key=lambda pair: pair[0])
    merged = []
    for r0, r1 in normalized:
        if not merged or r0 > merged[-1][1]:
            merged.append([r0, r1])
        else:
            merged[-1][1] = max(merged[-1][1], r1)
    return tuple((float(r0), float(r1)) for r0, r1 in merged)

def fit_monotonic_loglog_prior(
    r,
    ne,
    knot_spacing_rsun=0.05,
    median_filter_bins=3,
    outlier_sigma=4.0,
    exclude_r_ranges=None,
):
    """
    Fit a data-driven spherical prior without fixing the Saito radial exponents.

    The observed azimuthal-median density is fitted in log10(r)-log10(ne) space.
    Data inside ``exclude_r_ranges`` are not used to determine the fit.  Radial
    knots are constructed only where retained observations exist; therefore a
    PCHIP segment bridges each excluded interval directly between the valid
    knots on its two sides.  No artificial knot is inserted inside the gap.

    The retained knot values are robust medians, mildly median-filtered within
    each contiguous radial segment, and constrained to be non-increasing by
    PAVA.  A shape-preserving PCHIP interpolation then defines the prior table.

    R^2, RMSE, MAE, and MdAPE are evaluated only from retained observations,
    i.e. the excluded radial intervals do not contribute to these metrics.
    """
    from scipy.ndimage import median_filter

    r = np.asarray(r, dtype=np.float64).ravel()
    ne = np.asarray(ne, dtype=np.float64).ravel()
    if r.size != ne.size:
        raise ValueError("r and ne must have the same size.")

    exclude_ranges = _normalize_fit_exclude_ranges(exclude_r_ranges)

    valid_all = np.isfinite(r) & np.isfinite(ne) & (r > 1.0) & (ne > 0)
    r_valid_all = r[valid_all]
    ne_valid_all = ne[valid_all]
    if r_valid_all.size < 10:
        raise ValueError(
            f"Not enough valid density points for robust fit: "
            f"{r_valid_all.size} < 10"
        )

    excluded = np.zeros(r_valid_all.size, dtype=bool)
    for r0, r1 in exclude_ranges:
        excluded |= (r_valid_all >= r0) & (r_valid_all <= r1)

    r_fit = r_valid_all[~excluded]
    ne_fit = ne_valid_all[~excluded]
    n_excluded = int(np.count_nonzero(excluded))
    if r_fit.size < 10:
        raise ValueError(
            "Too few points remain after applying excluded radial ranges: "
            f"{r_fit.size} < 10"
        )

    order = np.argsort(r_fit)
    r_fit = r_fit[order]
    ne_fit = ne_fit[order]
    log_ne = np.log10(ne_fit)

    # Remove isolated inversion spikes relative to a local median.  Filtering is
    # performed independently on radial segments separated by a large gap, so
    # values on opposite sides of an excluded interval do not contaminate each
    # other's local median.
    local_size = max(3, int(median_filter_bins))
    if local_size % 2 == 0:
        local_size += 1

    keep = np.ones(r_fit.size, dtype=bool)
    if r_fit.size >= 2:
        dr_positive = np.diff(r_fit)
        dr_positive = dr_positive[np.isfinite(dr_positive) & (dr_positive > 0)]
        typical_dr = float(np.nanmedian(dr_positive)) if dr_positive.size else 0.0
        gap_threshold = max(3.0 * typical_dr, 1.0e-6)
        split_indices = np.where(np.diff(r_fit) > gap_threshold)[0] + 1
    else:
        split_indices = np.array([], dtype=int)

    segment_bounds = np.concatenate(([0], split_indices, [r_fit.size]))
    for i0, i1 in zip(segment_bounds[:-1], segment_bounds[1:]):
        segment = log_ne[i0:i1]
        if segment.size == 0:
            continue
        filter_size = min(local_size, segment.size)
        if filter_size % 2 == 0:
            filter_size -= 1
        filter_size = max(filter_size, 1)
        local_median = median_filter(segment, size=filter_size, mode="nearest")
        residual_local = segment - local_median
        local_center = float(np.nanmedian(residual_local))
        local_mad = 1.4826 * float(
            np.nanmedian(np.abs(residual_local - local_center))
        )
        if np.isfinite(local_mad) and local_mad > 0 and outlier_sigma > 0:
            keep[i0:i1] = (
                np.abs(residual_local - local_center)
                <= float(outlier_sigma) * local_mad
            )

    r_used = r_fit[keep]
    ne_used = ne_fit[keep]
    if r_used.size < 10:
        raise ValueError(
            f"Too few points remain after robust clipping: {r_used.size} < 10"
        )

    r_min = float(np.nanmin(r_used))
    r_max = float(np.nanmax(r_used))
    spacing = float(knot_spacing_rsun)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(f"Invalid knot_spacing_rsun={knot_spacing_rsun}")

    edges = np.arange(r_min, r_max + spacing, spacing)
    if edges.size < 3 or edges[-1] < r_max:
        edges = np.append(edges, r_max + spacing)
    knot_r_all = 0.5 * (edges[:-1] + edges[1:])
    knot_log_ne_all = np.full(knot_r_all.size, np.nan, dtype=np.float64)
    knot_weight_all = np.zeros(knot_r_all.size, dtype=np.float64)

    log_ne_used = np.log10(ne_used)
    for i in range(knot_r_all.size):
        mask = (r_used >= edges[i]) & (r_used < edges[i + 1])
        if np.any(mask):
            knot_log_ne_all[i] = float(np.nanmedian(log_ne_used[mask]))
            knot_weight_all[i] = float(np.count_nonzero(mask))

    finite_knots = np.isfinite(knot_log_ne_all)
    if np.count_nonzero(finite_knots) < 4:
        raise ValueError(
            f"Not enough populated radial knots: "
            f"{np.count_nonzero(finite_knots)} < 4"
        )

    # Keep only data-supported knots.  In particular, no synthetic knot is
    # inserted inside 2.4--2.7 Rs (or any other excluded interval).  PCHIP will
    # connect the nearest valid knots across that interval with a continuous
    # first derivative.
    knot_r = knot_r_all[finite_knots]
    knot_log_ne = knot_log_ne_all[finite_knots]
    knot_weight = knot_weight_all[finite_knots]

    # Mildly filter each contiguous knot segment separately.  This preserves the
    # independence of both sides of an excluded radial interval.
    if knot_r.size >= 2:
        split_knots = np.where(np.diff(knot_r) > 1.5 * spacing)[0] + 1
    else:
        split_knots = np.array([], dtype=int)
    knot_bounds = np.concatenate(([0], split_knots, [knot_r.size]))
    filtered_log_ne = knot_log_ne.copy()
    for i0, i1 in zip(knot_bounds[:-1], knot_bounds[1:]):
        segment = knot_log_ne[i0:i1]
        if segment.size < 3:
            continue
        filter_size = min(3, segment.size)
        if filter_size % 2 == 0:
            filter_size -= 1
        filtered_log_ne[i0:i1] = median_filter(
            segment,
            size=max(filter_size, 1),
            mode="nearest",
        )
    knot_log_ne = filtered_log_ne

    # Enforce a physically non-increasing radial prior.  Since the knot grid has
    # no points inside excluded intervals, PAVA does not create an artificial
    # constant block there; the final PCHIP curve bridges the gap directly.
    knot_log_ne = _pava_nonincreasing(knot_log_ne, weights=knot_weight)
    log_r_knots = np.log10(knot_r)

    fit_result = {
        "model_name": "monotonic_loglog_pchip_with_excluded_ranges",
        "equation": "log10(ne)=PCHIP(log10(r); data-supported monotonic knots)",
        "log_r_knots": log_r_knots,
        "log_ne_knots": knot_log_ne,
        "radius_knots_rsun": knot_r,
        "ne_knots_cm3": np.power(10.0, knot_log_ne),
        "r_used": r_used,
        "ne_used": ne_used,
        "n_input": int(r_valid_all.size),
        "n_excluded": n_excluded,
        "n_used": int(r_used.size),
        "exclude_r_ranges": np.asarray(exclude_ranges, dtype=np.float64).reshape(-1, 2),
    }

    ne_pred = evaluate_monotonic_loglog_prior(r_used, fit_result)
    log_obs = np.log10(ne_used)
    log_pred = np.log10(np.maximum(ne_pred, 1.0e-30))
    residual = log_pred - log_obs
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((log_obs - np.mean(log_obs)) ** 2))
    r2_log10 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    rmse_log10 = float(np.sqrt(np.mean(residual ** 2)))
    mae_log10 = float(np.mean(np.abs(residual)))
    mdape_percent = float(
        100.0
        * np.nanmedian(
            np.abs(ne_pred - ne_used) / np.maximum(ne_used, 1.0e-30)
        )
    )

    fit_result.update(
        ne_pred_used=ne_pred,
        r2_log10=r2_log10,
        rmse_log10=rmse_log10,
        mae_log10=mae_log10,
        mdape_percent=mdape_percent,
    )
    return fit_result



# --- Integrated Earth-view pB input directory ---
EARTH_VIEW_PB_DIR = Path(
    r'/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata'
)


def build_earth_view_pb_path(yyyymmdd, hhmm, data_dir=EARTH_VIEW_PB_DIR):
    """Build pB_Kcor_LASCO_axi_<YYYYMMDD>_<HHMM>.fits after validating time text."""
    yyyymmdd = str(yyyymmdd).strip()
    hhmm = str(hhmm).strip()
    try:
        datetime.strptime(f"{yyyymmdd}_{hhmm}", "%Y%m%d_%H%M")
    except ValueError as exc:
        raise ValueError(
            "yyyymmdd and hhmm must be valid strings such as "
            "'20220613' and '0258'."
        ) from exc

    return Path(data_dir).expanduser() / (
        f"pB_Kcor_LASCO_axi_{yyyymmdd}_{hhmm}.fits"
    )


def load_integrated_earth_view_pb(fits_path):
    """
    Load one integrated K-Cor/LASCO-C2 Earth-view pB FITS file.

    The integrated image already shares a common Earth-view image grid. The solar
    center and plate scale are reconstructed from FITS WCS keywords. FITS CRPIX
    values are 1-based, whereas NumPy pixel coordinates are 0-based.
    """
    fits_path = Path(fits_path).expanduser()
    if not fits_path.exists():
        raise FileNotFoundError(f"Integrated Earth-view pB FITS not found: {fits_path}")

    with fits.open(fits_path, memmap=False) as hdul:
        if hdul[0].data is None:
            raise ValueError(f"Primary HDU contains no image data: {fits_path}")
        image = np.asarray(hdul[0].data, dtype=np.float64).squeeze()
        header = hdul[0].header.copy()

    if image.ndim != 2:
        raise ValueError(
            f"Integrated pB data must be 2D after squeeze, got shape={image.shape}"
        )

    ny, nx = image.shape
    cx = float(header.get('CRPIX1', (nx + 1.0) / 2.0)) - 1.0
    cy = float(header.get('CRPIX2', (ny + 1.0) / 2.0)) - 1.0

    cdelt_values = []
    for key in ('CDELT1', 'CDELT2'):
        try:
            value = abs(float(header.get(key)))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            cdelt_values.append(value)

    rsun_arcsec = None
    for key in ('RSUN_OBS', 'RSUN'):
        try:
            value = float(header.get(key))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            rsun_arcsec = value
            break

    px_per_rsun = None
    if rsun_arcsec is not None and cdelt_values:
        arcsec_per_pixel = float(np.mean(cdelt_values))
        px_per_rsun = rsun_arcsec / arcsec_per_pixel
    else:
        for key in ('RSUNPIX', 'R_SUNPIX'):
            try:
                value = float(header.get(key))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0:
                px_per_rsun = value
                break

    if px_per_rsun is None or not np.isfinite(px_per_rsun) or px_per_rsun <= 0:
        raise ValueError(
            "Could not derive pixels per solar radius. The FITS header must contain "
            "RSUN_OBS (or RSUN) together with CDELT1/CDELT2, or RSUNPIX."
        )

    image[~np.isfinite(image)] = np.nan
    params = {
        'nx': int(nx),
        'ny': int(ny),
        'cx': float(cx),
        'cy': float(cy),
        'px_per_rsun': float(px_per_rsun),
        'date_obs': str(header.get('DATE-OBS', '')),
        'bunit': str(header.get('BUNIT', '')),
        'header': header,
    }

    yy, xx = np.indices((ny, nx), dtype=np.float64)
    r_map = np.hypot(
        (xx - params['cx']) / params['px_per_rsun'],
        (yy - params['cy']) / params['px_per_rsun'],
    )

    print(f"[INFO] Loaded integrated Earth-view pB: {fits_path}")
    print(
        "[INFO] FITS geometry: "
        f"shape={image.shape}, center=({params['cx']:.3f}, {params['cy']:.3f}) px, "
        f"scale={params['px_per_rsun']:.4f} px/Rsun, "
        f"DATE-OBS={params['date_obs']!r}, BUNIT={params['bunit']!r}"
    )
    return image, params, r_map


# --- Plotting module supplied by plot_pb_2d_maps.py ---
PB_2D_PLOT_MODULE_CANDIDATES = (
    "plot_pb_2d_maps.py",
    "plot_pb_2d_maps(2).py",
)


def load_pb_2d_plot_module():
    """
    Load the attached pB 2D plotting code and return it as a Python module.

    Both the normal filename and the uploaded ``(2)`` filename are accepted.
    The imported module's ``read_pb_fits_as_xy_map`` and ``plot_single_pb_map``
    functions are used directly for the combined-pB image.
    """
    base_dir = Path(__file__).resolve().parent
    module_path = None
    for filename in PB_2D_PLOT_MODULE_CANDIDATES:
        candidate = base_dir / filename
        if candidate.exists():
            module_path = candidate
            break

    if module_path is None:
        searched = ", ".join(str(base_dir / name) for name in PB_2D_PLOT_MODULE_CANDIDATES)
        raise FileNotFoundError(
            "Could not find the pB 2D plotting module. Searched: " + searched
        )

    module_name = "plot_pb_2d_maps_external"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import specification for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    print(f"[INFO] Loaded combined-pB plotting functions from: {module_path}")
    return module


def density_to_plasma_frequency_mhz(ne_cm3, harmonic=1):
    """Convert electron density [cm^-3] to plasma-emission frequency [MHz]."""
    ne_cm3 = np.asarray(ne_cm3, dtype=np.float64)
    harmonic = float(harmonic)
    if not np.isfinite(harmonic) or harmonic <= 0:
        raise ValueError(f"harmonic must be positive, got {harmonic}")
    return harmonic * 8.98e-3 * np.sqrt(np.maximum(ne_cm3, 0.0))


def plasma_frequency_mhz_to_density(freq_mhz, harmonic=1):
    """Convert plasma-emission frequency [MHz] to electron density [cm^-3]."""
    freq_mhz = np.asarray(freq_mhz, dtype=np.float64)
    harmonic = float(harmonic)
    if not np.isfinite(harmonic) or harmonic <= 0:
        raise ValueError(f"harmonic must be positive, got {harmonic}")
    return np.square(np.maximum(freq_mhz, 0.0) / (harmonic * 8.98e-3))


def build_density_2d_map_from_sector_profiles(
    ne_sector_profiles,
    r_map,
    cx,
    cy,
    radial_edges,
    r_min=1.1,
    r_max=4.0,
):
    """
    Expand PA-sector density profiles onto the original image grid.

    The result is an SSI-derived plane-of-sky density map. Each image pixel is
    assigned the density from its radial bin and PA sector. It is not a true LOS-
    resolved 3D density map.
    """
    ne_sector_profiles = np.asarray(ne_sector_profiles, dtype=np.float64)
    r_map = np.asarray(r_map, dtype=np.float64)
    radial_edges = np.asarray(radial_edges, dtype=np.float64)

    if ne_sector_profiles.ndim != 2:
        raise ValueError("ne_sector_profiles must be a 2D array.")
    if radial_edges.size != ne_sector_profiles.shape[1] + 1:
        raise ValueError(
            "radial_edges must contain one more element than the radial density dimension."
        )

    yy, xx = np.indices(r_map.shape, dtype=np.float64)
    pa_deg = (np.degrees(np.arctan2(yy - float(cy), xx - float(cx))) + 360.0) % 360.0

    n_angle_bins = ne_sector_profiles.shape[0]
    angle_index = np.floor(pa_deg / 360.0 * n_angle_bins).astype(np.int64)
    angle_index = np.clip(angle_index, 0, n_angle_bins - 1)

    radial_index = np.digitize(r_map, radial_edges, right=False) - 1
    valid = (
        np.isfinite(r_map)
        & (r_map >= float(r_min))
        & (r_map <= float(r_max))
        & (radial_index >= 0)
        & (radial_index < ne_sector_profiles.shape[1])
    )

    density_map = np.full(r_map.shape, np.nan, dtype=np.float64)
    density_map[valid] = ne_sector_profiles[
        angle_index[valid],
        radial_index[valid],
    ]
    density_map[~np.isfinite(density_map) | (density_map <= 0)] = np.nan
    return density_map


def plot_density_2d_map(
    pbmap,
    density_map,
    pb_plot_module,
    output_dir,
    yyyymmdd,
    hhmm,
    xy_limit_rsun=4.5,
    dpi=300,
):
    """Plot and save the SSI-derived plane-of-sky electron-density map."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    density_map = np.asarray(density_map, dtype=np.float64)
    positive = density_map[np.isfinite(density_map) & (density_map > 0)]
    if positive.size == 0:
        print("[WARN] Density 2D map contains no positive finite values; plot skipped.")
        return None

    vmin = float(np.nanpercentile(positive, 1.0))
    vmax = float(np.nanpercentile(positive, 99.5))
    if not np.isfinite(vmin) or vmin <= 0:
        vmin = float(np.nanmin(positive))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(positive))
    if vmax <= vmin:
        vmax = vmin * 10.0

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("black")

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(
        density_map,
        origin="lower",
        extent=pbmap.extent,
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation="nearest",
        aspect="equal",
    )

    pb_plot_module._add_radial_guides(ax, [1.0, 2.0, 4.0])

    boundary_points = pb_plot_module._boundary_points_by_angle(
        pbmap,
        boundary_scatter_deg=float(pb_plot_module.BOUNDARY_SCATTER_DEG),
    )
    if boundary_points is not None:
        bx, by = boundary_points
        ax.scatter(
            bx,
            by,
            s=float(pb_plot_module.BOUNDARY_SCATTER_SIZE),
            c="white",
            marker="o",
            linewidths=0.0,
            alpha=0.95,
        )

    ax.axhline(0.0, color="white", linewidth=0.5, alpha=0.7)
    ax.axvline(0.0, color="white", linewidth=0.5, alpha=0.7)
    ax.set_xlabel(r"X [$R_\odot$]")
    ax.set_ylabel(r"Y [$R_\odot$]")
    ax.set_title(
        "SSI-derived plane-of-sky electron density\n"
        f"{pbmap.path.name}\n{yyyymmdd} {hhmm} UT"
    )

    if xy_limit_rsun is not None:
        lim = float(xy_limit_rsun)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.86)
    cbar.set_label(r"Electron density [cm$^{-3}$]")

    ax.text(
        0.02,
        0.02,
        "Each PA sector is independently inverted\n"
        "under local spherical symmetry.",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.78, pad=3.0),
    )

    out_path = output_dir / (
        f"pB_Kcor_LASCO_axi_{yyyymmdd}_{hhmm}_density_2d_map.png"
    )
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved SSI-derived density 2D map: {out_path}")
    print(f"     LogNorm={vmin:.3e}..{vmax:.3e} cm^-3")
    return out_path




def main(
    yyyymmdd,
    hhmm,
    fit_r_min,
    fit_r_max,
    data_dir=EARTH_VIEW_PB_DIR,
    plasma_frequency_harmonic=1,
    fit_exclude_r_ranges=((2.4, 2.7),),
):
    instrument_kcor = "K-Cor"
    instrument_lasco = "SOHO/LASCO"
    fit_exclude_r_ranges = _normalize_fit_exclude_ranges(
        fit_exclude_r_ranges
    )

    # 1) Select and load the integrated Earth-view pB FITS file.
    input_fits = build_earth_view_pb_path(
        yyyymmdd=yyyymmdd,
        hhmm=hhmm,
        data_dir=data_dir,
    )
    final_image, params_lasco, r_map_lasco = load_integrated_earth_view_pb(
        input_fits
    )

    output_dir = Path(
        r'/mnt/d/wsl/home/kinno-7010/Research_data/SDO_Mk4_SOHO/pB'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the attached plot_pb_2d_maps.py functions for the combined-pB image.
    pb_plot_module = load_pb_2d_plot_module()
    pb_plot_module.SHOW_FIGURES = False
    pbmap = pb_plot_module.read_pb_fits_as_xy_map(input_fits)
    pb_plot_module.plot_single_pb_map(
        pbmap,
        label='K-COR + LASCO-C2 combined pB',
        output_dir=output_dir,
    )

    print(
        "[INFO] Each PA-sector profile is inverted under local spherical symmetry; "
        "the final prior is the azimuthal median ne(r)."
    )
    print(
        "[INFO] The fitted prior uses the median of the angle-resolved densities. "
        "Minimum and maximum densities are plotted as asymmetric error bars."
    )

    r_ranges = {
        'kcor_inner': 1.1,
        'kcor_outer_lasco_inner': 2.2,
        'lasco_outer': 7.0,
    }

    # Preserve only the radial domain used by the original integrated-image workflow.
    final_image = np.asarray(final_image, dtype=np.float64).copy()
    final_image[
        (r_map_lasco < r_ranges['kcor_inner'])
        | (r_map_lasco > r_ranges['lasco_outer'])
    ] = np.nan

    # The combined-pB map has already been drawn with plot_single_pb_map()
    # from the attached plot_pb_2d_maps.py code.

    # 2) Extract PA-sector pB profiles and invert each sector separately.
    bin_width = 0.01
    current_plot_r_min = r_ranges.get('kcor_inner', 1.1)
    current_plot_r_max = r_ranges.get('lasco_outer', 4.0)
    edges = np.arange(
        current_plot_r_min,
        current_plot_r_max + bin_width,
        bin_width,
    )
    r_mid = (
        (edges[:-1] + edges[1:]) / 2
        if len(edges) > 1
        else np.array([])
    )
    n_bins = len(r_mid)

    pB_sector_profiles = np.empty((0, n_bins), dtype=np.float64)
    pB_spherical = np.full_like(r_mid, np.nan)
    pB_min = np.full_like(r_mid, np.nan)
    pB_max = np.full_like(r_mid, np.nan)
    pB_sector_mad = np.full_like(r_mid, np.nan)
    angular_coverage = np.zeros_like(r_mid)
    n_valid_sectors = np.zeros_like(r_mid, dtype=np.int32)
    angle_centers_deg = np.array([], dtype=np.float64)

    if n_bins > 0:
        (
            pB_sector_profiles,
            pB_spherical,
            pB_min,
            pB_max,
            pB_sector_mad,
            angular_coverage,
            n_valid_sectors,
            angle_centers_deg,
        ) = extract_azimuthally_averaged_pB_profile(
            image=final_image,
            r_map=r_map_lasco,
            edges=edges,
            cy=params_lasco['cy'],
            cx=params_lasco['cx'],
            angle_bin_deg=2.0,
            sigma_clip=3.0,
            min_angular_coverage=0.50,
        )
        print(
            "[INFO] PA-sector extraction completed: "
            f"n_sectors={pB_sector_profiles.shape[0]}, "
            f"valid median-pB bins="
            f"{np.count_nonzero(np.isfinite(pB_spherical))}/{n_bins}, "
            f"median angular coverage={np.nanmedian(angular_coverage):.3f}"
        )

    r_boundary = r_ranges.get('kcor_outer_lasco_inner', 2.2)
    (
        Ne_sector_profiles,
        Ne_min,
        Ne_median,
        Ne_max,
        n_valid_density_sectors,
    ) = invert_azimuthal_sector_profiles(
        pB_sector_profiles=pB_sector_profiles,
        r_mid=r_mid,
        edges=edges,
        r_boundary=r_boundary,
        instrument_kcor=instrument_kcor,
        instrument_lasco=instrument_lasco,
        min_profile_coverage=0.50,
        min_valid_sectors_per_radius=10,
    )

    density_2d_map = build_density_2d_map_from_sector_profiles(
        ne_sector_profiles=Ne_sector_profiles,
        r_map=r_map_lasco,
        cx=params_lasco['cx'],
        cy=params_lasco['cy'],
        radial_edges=edges,
        r_min=r_ranges['kcor_inner'],
        r_max=4.0,
    )
    plot_density_2d_map(
        pbmap=pbmap,
        density_map=density_2d_map,
        pb_plot_module=pb_plot_module,
        output_dir=output_dir,
        yyyymmdd=yyyymmdd,
        hhmm=hhmm,
        xy_limit_rsun=4.5,
        dpi=300,
    )

    valid_ne_indices = (
        np.isfinite(Ne_median)
        & np.isfinite(Ne_min)
        & np.isfinite(Ne_max)
        & (Ne_median > 0)
        & (Ne_min > 0)
        & (Ne_max >= Ne_median)
        & (Ne_median >= Ne_min)
    )
    r_all_valid_ne = r_mid[valid_ne_indices]
    Ne_all_valid = Ne_median[valid_ne_indices]
    Ne_min_valid = Ne_min[valid_ne_indices]
    Ne_max_valid = Ne_max[valid_ne_indices]

    fit_range_mask = (
        (r_all_valid_ne >= fit_r_min)
        & (r_all_valid_ne <= fit_r_max)
        & (Ne_all_valid > 0)
    )
    r_fit_candidates = r_all_valid_ne[fit_range_mask]
    Ne_fit_candidates = Ne_all_valid[fit_range_mask]

    excluded_candidate_mask = np.zeros(r_fit_candidates.size, dtype=bool)
    for r0, r1 in fit_exclude_r_ranges:
        excluded_candidate_mask |= (
            (r_fit_candidates >= r0) & (r_fit_candidates <= r1)
        )
    r_for_fitting = r_fit_candidates[~excluded_candidate_mask]
    Ne_for_fitting = Ne_fit_candidates[~excluded_candidate_mask]

    if len(r_for_fitting) == 0:
        print(
            f"[WARN] No valid median-Ne points remain in fit range "
            f"{fit_r_min}–{fit_r_max} Rsun after exclusions."
        )
    else:
        exclusion_text = (
            ", ".join(f"{r0:.2f}–{r1:.2f} Rs" for r0, r1 in fit_exclude_r_ranges)
            if fit_exclude_r_ranges
            else "none"
        )
        print(
            f"[INFO] Median-density fitting points: {len(r_for_fitting)} "
            f"in {fit_r_min}–{fit_r_max} Rsun; "
            f"excluded ranges={exclusion_text} "
            f"(fit-range candidates: {len(r_fit_candidates)})"
        )

    # 3) Density bounds for highlighting (14–42 MHz).
    ne_14MHz_limit = density_from_frequency(14)
    ne_42MHz_limit = density_from_frequency(42)
    density_lower_highlight = np.nanmin(
        [ne_14MHz_limit, ne_42MHz_limit]
    )
    density_upper_highlight = np.nanmax(
        [ne_14MHz_limit, ne_42MHz_limit]
    )

    # 4) Fit only the azimuthal-median density profile.
    fit_result = None
    if len(r_for_fitting) > 0:
        try:
            fit_result = fit_monotonic_loglog_prior(
                r_fit_candidates,
                Ne_fit_candidates,
                knot_spacing_rsun=0.05,
                median_filter_bins=5,
                outlier_sigma=4.0,
                exclude_r_ranges=fit_exclude_r_ranges,
            )
            print(
                "[FIT] Observation-derived prior fitted to median Ne(r):\n"
                "      log10(ne) = PCHIP(log10(r); monotonic fitted knots)\n"
                f"[FIT] R2(log10 ne)={fit_result['r2_log10']:.6f}, "
                f"RMSE={fit_result['rmse_log10']:.5f} dex, "
                f"MAE={fit_result['mae_log10']:.5f} dex, "
                f"MdAPE={fit_result['mdape_percent']:.2f}%, "
                f"N={fit_result['n_used']}/{fit_result['n_input']}, "
                f"excluded={fit_result['n_excluded']}"
            )
        except Exception as exc:
            print(
                f"[WARN] Observation-derived prior fitting failed: {exc}"
            )

    # 5) Plot median density with angle-wise min/max asymmetric error bars.
    fig_ne, ax_ne = plt.subplots(figsize=(10, 5))
    r_min_for_plot, r_max_for_plot = 1.1, 4.0

    mask_plot_points = (
        (r_all_valid_ne >= r_min_for_plot)
        & (r_all_valid_ne <= r_max_for_plot)
    )
    r_plot_points = r_all_valid_ne[mask_plot_points]
    Ne_plot_points = Ne_all_valid[mask_plot_points]
    Ne_min_plot = Ne_min_valid[mask_plot_points]
    Ne_max_plot = Ne_max_valid[mask_plot_points]

    if len(r_plot_points) > 0:
        lower_error = np.maximum(Ne_plot_points - Ne_min_plot, 0.0)
        upper_error = np.maximum(Ne_max_plot - Ne_plot_points, 0.0)
        asymmetric_error = np.vstack([lower_error, upper_error])
        ax_ne.errorbar(
            r_plot_points,
            Ne_plot_points,
            yerr=asymmetric_error,
            fmt='o',
            markersize=3.5,
            markerfacecolor='#B3DB7D',
            markeredgecolor='black',
            markeredgewidth=0.7,
            ecolor='gray',
            elinewidth=0.7,
            capsize=0,
            alpha=0.65,
            label='Median $n_e$; error bars = PA-sector min–max',
            zorder=3,
        )

    if fit_result is not None:
        r_plot_line = np.linspace(r_min_for_plot, r_max_for_plot, 500)
        ne_fit_line = evaluate_monotonic_loglog_prior(
            r_plot_line, fit_result
        )
        fit_label = (
            r'Prior fitted to median: '
            r'$\log_{10}n_e=\mathrm{PCHIP}(\log_{10}r)$' '\n'
            f"$R^2_{{\\log}}$={fit_result['r2_log10']:.4f}, "
            f"RMSE={fit_result['rmse_log10']:.3f} dex\n"
            f"MdAPE={fit_result['mdape_percent']:.1f}%, "
            f"N={fit_result['n_used']}, "
            f"excluded={fit_result['n_excluded']}"
        )
        ax_ne.plot(
            r_plot_line,
            ne_fit_line,
            linestyle='-',
            linewidth=3,
            alpha=0.9,
            label=fit_label,
            zorder=4,
        )

    ax_ne.axvspan(
        fit_r_min,
        fit_r_max,
        alpha=0.10,
        label=f'Fit range: {fit_r_min:.1f}–{fit_r_max:.1f} Rs',
    )
    for i_range, (r0, r1) in enumerate(fit_exclude_r_ranges):
        ax_ne.axvspan(
            r0,
            r1,
            alpha=0.18,
            hatch='//',
            label=(
                f'Excluded from fit: {r0:.1f}–{r1:.1f} Rs'
                if i_range == 0
                else None
            ),
        )
    ax_ne.axhspan(
        density_lower_highlight,
        density_upper_highlight,
        alpha=0.10,
        label='14–42 MHz density range',
    )
    ax_ne.axvline(x=2.2, color='gray', linestyle='--', linewidth=1)
    ax_ne.text(
        2.2,
        1.3e5,
        'K-Cor/LASCO boundary\n(2.2 Rs)',
        color='black',
        fontsize=11,
        ha='right',
        va='bottom',
    )

    ax_ne.set_xlim(r_min_for_plot, r_max_for_plot)
    ax_ne.set_ylim(1e5, 1e9)
    ax_ne.set_yscale('log')
    ax_ne.set_xlabel(r'Heliocentric distance [$R_\odot$]')
    ax_ne.set_ylabel(r'Electron density [cm$^{-3}$]')

    harmonic = int(plasma_frequency_harmonic)
    if harmonic < 1:
        raise ValueError(
            f"plasma_frequency_harmonic must be >= 1, got {harmonic}"
        )
    secondary_axis = ax_ne.secondary_yaxis(
        'right',
        functions=(
            lambda ne: density_to_plasma_frequency_mhz(ne, harmonic=harmonic),
            lambda freq: plasma_frequency_mhz_to_density(freq, harmonic=harmonic),
        ),
    )
    if harmonic == 1:
        secondary_axis.set_ylabel('Plasma frequency [MHz] (fundamental)')
    else:
        secondary_axis.set_ylabel(
            f'Plasma emission frequency [MHz] (harmonic={harmonic})'
        )

    ax_ne.set_title(
        f'Spherically symmetric prior from {yyyymmdd} {hhmm} UT\n'
        'Median of PA-sector inversions'
    )
    ax_ne.grid(True, which='both', alpha=0.25)
    ax_ne.legend(fontsize=8)

    output_stem = (
        f'pB_spherical_median_prior_errorbar_{yyyymmdd}_{hhmm}_'
        f'fit{fit_r_min:.1f}-{fit_r_max:.1f}'
    )
    output_png = output_dir / f'{output_stem}.png'
    output_csv = output_dir / f'{output_stem}.csv'
    output_npz = output_dir / f'{output_stem}_prior_model.npz'

    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"✓ Spherical radial-density plot saved: {output_png}")

    fit_curve_all = np.full_like(r_mid, np.nan, dtype=np.float64)
    if fit_result is not None:
        fit_curve_all = evaluate_monotonic_loglog_prior(r_mid, fit_result)

    fit_excluded_flag = np.zeros_like(r_mid, dtype=np.int32)
    for r0, r1 in fit_exclude_r_ranges:
        fit_excluded_flag[(r_mid >= r0) & (r_mid <= r1)] = 1

    output_table = np.column_stack([
        r_mid,
        pB_spherical,
        pB_min,
        pB_max,
        pB_sector_mad,
        angular_coverage,
        n_valid_sectors.astype(np.float64),
        Ne_min,
        Ne_median,
        Ne_max,
        n_valid_density_sectors.astype(np.float64),
        fit_curve_all,
        fit_excluded_flag.astype(np.float64),
    ])
    np.savetxt(
        output_csv,
        output_table,
        delimiter=',',
        header=(
            'radius_rsun,pB_azimuthal_median,pB_azimuthal_min,'
            'pB_azimuthal_max,pB_sector_mad,angular_coverage,'
            'n_valid_pB_sectors,ne_azimuthal_min_cm3,'
            'ne_azimuthal_median_cm3,ne_azimuthal_max_cm3,'
            'n_valid_density_sectors,ne_prior_cm3,'
            'fit_excluded_from_prior'
        ),
        comments='',
    )
    print(f"✓ Spherical radial profile saved: {output_csv}")

    if fit_result is not None:
        np.savez_compressed(
            output_npz,
            model_name=fit_result['model_name'],
            equation=fit_result['equation'],
            input_fits=str(input_fits),
            target_yyyymmdd=str(yyyymmdd),
            target_hhmm=str(hhmm),
            fit_target='azimuthal_median_density',
            fit_r_min=float(fit_r_min),
            fit_r_max=float(fit_r_max),
            fit_exclude_r_ranges=np.asarray(
                fit_exclude_r_ranges, dtype=np.float64
            ).reshape(-1, 2),
            fit_excluded_mask=fit_excluded_flag.astype(bool),
            log_r_knots=np.asarray(
                fit_result['log_r_knots'], dtype=np.float64
            ),
            log_ne_knots=np.asarray(
                fit_result['log_ne_knots'], dtype=np.float64
            ),
            radius_knots_rsun=np.asarray(
                fit_result['radius_knots_rsun'], dtype=np.float64
            ),
            ne_knots_cm3=np.asarray(
                fit_result['ne_knots_cm3'], dtype=np.float64
            ),
            radius_rsun=r_mid,
            ne_prior_cm3=fit_curve_all,
            ne_azimuthal_min_cm3=Ne_min,
            ne_azimuthal_median_cm3=Ne_median,
            ne_azimuthal_max_cm3=Ne_max,
            n_valid_density_sectors=n_valid_density_sectors,
            pa_sector_centers_deg=angle_centers_deg,
            ne_sector_profiles_cm3=Ne_sector_profiles,
            r2_log10=float(fit_result['r2_log10']),
            rmse_log10=float(fit_result['rmse_log10']),
            mae_log10=float(fit_result['mae_log10']),
            mdape_percent=float(fit_result['mdape_percent']),
            n_input=int(fit_result['n_input']),
            n_excluded=int(fit_result['n_excluded']),
            n_used=int(fit_result['n_used']),
        )
        print(
            f"✓ Tomography-ready spherical prior table saved: {output_npz}"
        )

    plt.show()





if __name__ == "__main__":
    # Select the integrated Earth-view pB file here.
    TARGET_YYYYMMDD = "20220613"
    TARGET_HHMM = "0258"

    # 1: fundamental plasma frequency. Set 2 to show second-harmonic emission frequency.
    PLASMA_FREQUENCY_HARMONIC = 1

    # K-Cor/LASCO接続の影響が残る区間はprior fitから除外する。
    # データ点とエラーバーは表示するが、橙色のprior線の決定には使わない。
    FIT_EXCLUDE_R_RANGES = [(2.3, 2.8)]

    # Tomographyの主使用範囲に合わせ、K-CorとLASCOの両方を含む範囲でfitする。
    main(
        yyyymmdd=TARGET_YYYYMMDD,
        hhmm=TARGET_HHMM,
        fit_r_min=1.5,
        fit_r_max=4.0,
        data_dir=EARTH_VIEW_PB_DIR,
        plasma_frequency_harmonic=PLASMA_FREQUENCY_HARMONIC,
        fit_exclude_r_ranges=FIT_EXCLUDE_R_RANGES,
    )
