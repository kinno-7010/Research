"""
gcs_geometry.py
----------------
Utility helpers for generating a GCS "croissant" wireframe in 3D solar coordinates.
This module reuses the PyThea implementation of the GCS model so that the
visualisation matches the well-tested reference geometry employed by PyThea.

Parameters follow the standard GCS conventions:
  - h_apex : float [Rsun]    -> apex (frontmost) height from Sun center
  - kappa  : float [0-1)     -> aspect ratio (minor/major radius) of the torus front (Thernisien 2011)
                               With this convention:
                                    R_major = h_apex / (1 + kappa)
                                    r_minor = kappa * R_major
                                so that R_major + r_minor = h_apex
  - alpha  : float [deg]      -> half-angle controlling lateral extent (used to trim torus + approx legs)
  - tilt   : float [deg]      -> rotation of the wireframe around the CME axis (+CCW looking outward)
  - lon    : float [deg]      -> heliographic Stonyhurst longitude of CME axis direction
  - lat    : float [deg]      -> heliographic Stonyhurst latitude of CME axis direction

NOTE: This is an *approximate* GCS geometry intended for overlay visualization.
      The torus front uses a true torus parameterization; the legs are approximated as
      conical surfaces with opening alpha that smoothly attach visually.

Outputs:
  sample_gcs_wireframe_points(...) returns a dict with lists of 3D points (in Rsun),
  in the HGS world frame (Sun-centered Cartesian: X toward lon=0, Y toward lon=+90°, Z to solar north).

You can adjust sampling density via n_parallels and n_meridians.
"""
from dataclasses import dataclass
import sys
from pathlib import Path
import warnings
import types

import numpy as np

# Silence PyThea version fallback warnings when bundled _version metadata is missing.
warnings.filterwarnings(
    "ignore",
    message="could not determine PyThea package version; this indicates a broken installation",
    category=UserWarning,
    module=r"PyThea\.version",
)

# Provide a minimal seaborn stub if seaborn is not installed; PyThea only needs
# the color_palette helper for plotting defaults, so a lightweight replacement
# keeps the import path working without pulling the full dependency tree.
try:  # pragma: no cover - optional dependency shim
    import seaborn as _seaborn  # noqa: F401
except Exception:  # pragma: no cover
    def _fallback_color_palette(name="colorblind", n_colors=None):
        palettes = {
            "colorblind": [
                (0.0, 0.45, 0.70),
                (0.80, 0.47, 0.67),
                (0.95, 0.90, 0.25),
                (0.35, 0.70, 0.90),
                (0.00, 0.62, 0.45),
                (0.83, 0.37, 0.00),
                (0.94, 0.89, 0.26),
                (0.10, 0.10, 0.10),
            ],
            "deep": [
                (0.30, 0.45, 0.74),
                (0.86, 0.37, 0.34),
                (0.37, 0.72, 0.36),
                (0.55, 0.34, 0.69),
                (0.89, 0.59, 0.28),
                (0.26, 0.64, 0.76),
                (0.60, 0.60, 0.60),
                (0.99, 0.75, 0.44),
            ],
        }
        base = palettes.get(name, palettes["colorblind"]).copy()
        if n_colors is None or n_colors <= len(base):
            return base[:n_colors] if n_colors else base
        repeats = (n_colors + len(base) - 1) // len(base)
        extended = (base * repeats)[:n_colors]
        return extended

    seaborn_stub = types.ModuleType("seaborn")
    seaborn_stub.color_palette = _fallback_color_palette
    sys.modules.setdefault("seaborn", seaborn_stub)

# Ensure the bundled PyThea source is importable so we can reuse its GCS implementation
_PYTHEA_ROOT = None
for base in Path(__file__).resolve().parents:
    candidate = base / "PyThea" / "Kouloumvakos_GitHub"
    if candidate.exists():
        _PYTHEA_ROOT = candidate
        break

if _PYTHEA_ROOT is not None:
    _pythea_path = str(_PYTHEA_ROOT)
    if _pythea_path not in sys.path:
        sys.path.insert(0, _pythea_path)

try:  # pragma: no cover - import guarding for optional dependency
    from PyThea.geometrical_models import gcs as PyTheaGCS  # type: ignore
except Exception:  # pragma: no cover
    PyTheaGCS = None  # will raise at runtime if unavailable

@dataclass
class GCSParams:
    h_apex: float   # [Rsun]
    kappa: float    # [0-1)
    alpha_deg: float
    tilt_deg: float
    lon_deg: float
    lat_deg: float

    def legend_label(self) -> str:
        """Return a formatted legend string summarizing the GCS parameters."""
        return (
            "$h_{\\mathrm{apex}}$ ="+ f"{self.h_apex:.3f}"+"$\\mathrm{R_\\odot}$, "
            "$\\kappa$ ="+ f"{self.kappa:.2f}, "
            "$\\alpha$ ="+ f"{self.alpha_deg:.1f}"+"$\\mathrm{\\degree}$, "
            "$\\mathrm{tilt}$ ="+ f"{self.tilt_deg:.1f}"+"$\\mathrm{\\degree}$, "
            "$\\mathrm{lon}$ ="+ f"{self.lon_deg:.1f}"+"$\\mathrm{\\degree}$, "
            "$\\mathrm{lat}$ ="+ f"{self.lat_deg:.1f}"+"$\\mathrm{\\degree}$"
        )

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def _hgs_axes_from_axis(lon_rad: float, lat_rad: float, tilt_rad: float):
    """
    Build an orthonormal triad (ex, ey, ez) in HGS frame:
      - ez: unit vector along the CME axis (given by lon, lat)
      - ex: unit vector perpendicular to ez (after applying tilt around ez)
      - ey: ez x ex
    HGS world basis is:
      X toward lon=0,lat=0; Y toward lon=+90°,lat=0; Z toward +solar-north.
    """
    import numpy as np
    # Axis direction (ez) from lon, lat in HGS
    ez = np.array([
        np.cos(lat_rad)*np.cos(lon_rad),
        np.cos(lat_rad)*np.sin(lon_rad),
        np.sin(lat_rad)
    ], dtype=float)
    ez = _unit(ez)

    # Reference direction (world Z) to construct a perpendicular
    z_world = np.array([0.0, 0.0, 1.0])
    # If ez is too close to z_world, use X instead to avoid degeneracy
    ref = z_world
    if abs(np.dot(ez, ref)) > 0.98:
        ref = np.array([1.0, 0.0, 0.0])
    t0 = _unit(np.cross(ez, ref))
    b0 = _unit(np.cross(ez, t0))

    # Apply tilt around ez: ex = cos* t0 + sin* b0
    c, s = np.cos(tilt_rad), np.sin(tilt_rad)
    ex = _unit(c * t0 + s * b0)
    ey = _unit(np.cross(ez, ex))
    return ex, ey, ez

def _torus_points(R: float, a: float, theta: np.ndarray, phi: np.ndarray):
    """
    Standard torus parameterization in the "body" frame where the axis is z_b,
    and the apex points along x_b (theta=0).
    Returns points in that body frame (x_b, y_b, z_b).
    """
    import numpy as np
    # Broadcast to a grid
    TT, PP = np.meshgrid(theta, phi, indexing='xy')
    Xb = (R + a*np.cos(PP)) * np.cos(TT)
    Yb = (R + a*np.cos(PP)) * np.sin(TT)
    Zb = a * np.sin(PP)
    return Xb, Yb, Zb

def _rotate_body_to_hgs(Xb, Yb, Zb, ex, ey, ez):
    """
    Convert body-frame coordinates (ex: apex direction, ey: 90° CCW, ez: axis)
    into HGS world coordinates.
    """
    import numpy as np
    # Stack body coords
    pts = np.stack([Xb, Yb, Zb], axis=-1)  # (..., 3)
    # Build 3x3 rotation matrix columns = [ex ey ez]
    Rm = np.stack([ex, ey, ez], axis=1)  # shape (3,3)
    # Apply rotation: world = body * Rm^T  (or pts @ Rm)
    world = pts @ Rm.T  # (...,3)
    Xw, Yw, Zw = world[...,0], world[...,1], world[...,2]
    return Xw, Yw, Zw

def _conical_legs(alpha_rad: float, r_min: float, r_max: float, n_r: int, n_phi: int):
    """
    Approximate two conical "legs" symmetric about the apex direction.
    Cones open with half-angle alpha around axes at ±90° from apex in body-plane.
    Returns list of (Xb,Yb,Zb) for each meridian-like curve.
    """
    import numpy as np
    # Axes for two cones in body frame: +yb and -yb directions
    axes = [(0.0, 1.0, 0.0), (0.0, -1.0, 0.0)]
    leg_curves = []
    r_vals = np.linspace(r_min, r_max, n_r)
    phi_vals = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
    for ax in axes:
        # Direction of the cone axis (unit vector in body frame)
        ax_vec = _unit(np.array(ax, dtype=float))
        # Build an orthonormal basis (u,v,ax) for circular cross section
        ref = np.array([1.0,0.0,0.0]) if abs(ax_vec[0])<0.9 else np.array([0.0,0.0,1.0])
        u = _unit(np.cross(ax_vec, ref))
        v = _unit(np.cross(ax_vec, u))
        # For each phi around the cone, build a "meridian" curve over r
        for ph in phi_vals:
            # direction at alpha from axis:
            dir_vec = _unit(np.cos(alpha_rad)*ax_vec + np.sin(alpha_rad)*(np.cos(ph)*u + np.sin(ph)*v))
            Xb = dir_vec[0]*r_vals
            Yb = dir_vec[1]*r_vals
            Zb = dir_vec[2]*r_vals
            leg_curves.append((Xb, Yb, Zb))
    return leg_curves

def sample_gcs_wireframe_points(
    params: GCSParams,
    obstime,
    n_parallels: int = 8,
    n_meridians: int = 12,
    include_legs: bool = True,
    leg_r_min: float = 1.02,
    clip_front_by_alpha: bool = True  # kept for backwards compatibility, unused in PyThea workflow
):
    """Generate GCS wireframe curves using the PyThea implementation."""
    if PyTheaGCS is None:
        raise ImportError("PyThea GCS geometry is unavailable; ensure PyThea sources are present.")

    import numpy as np
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import frames

    # Convert user parameters to quantities expected by PyThea
    h_apex = float(params.h_apex) * u.R_sun
    kappa = float(params.kappa)
    alpha = float(params.alpha_deg) * u.deg
    tilt = float(params.tilt_deg) * u.deg
    lon = float(params.lon_deg) * u.deg
    lat = float(params.lat_deg) * u.deg

    # Determine the GCS centre radius using PyThea's helper and construct the SkyCoord centre
    r_center = PyTheaGCS.rcenter_(h_apex, alpha, kappa)
    center = SkyCoord(
        lon=lon,
        lat=lat,
        radius=r_center,
        frame=frames.HeliographicStonyhurst,
        obstime=obstime,
        observer="earth",
    )

    # Increase PyThea sampling to maintain smooth curves when down-selecting lines
    nbvertsl = max(10, int(np.ceil(n_meridians * 1.5)))
    nbvertcirc = max(24, int(np.ceil(n_meridians * 2.5)))
    nbvertcircshell = max(60, int(np.ceil(n_parallels * 6)))

    model = PyTheaGCS(
        center,
        h_apex,
        alpha,
        kappa,
        tilt,
        nbvertsl=nbvertsl,
        nbvertcirc=nbvertcirc,
        nbvertcircshell=nbvertcircshell,
    )

    p, r, ca = model.shell_skeleton()
    total_nodes = len(r)
    if total_nodes == 0:
        return {"parallels": [], "meridians": [], "legs": []}

    leg_len = model.nbvertsl
    core_slice = slice(leg_len, total_nodes - leg_len) if total_nodes > 2 * leg_len else slice(None)
    core_indices = np.arange(total_nodes)[core_slice]
    if core_indices.size == 0:
        core_indices = np.arange(total_nodes)

    def _build_body_curve(phi_value: float, indices: np.ndarray) -> np.ndarray:
        cos_phi = np.cos(phi_value)
        sin_phi = np.sin(phi_value)
        x = r[indices] * cos_phi + p[indices, 0]
        y = r[indices] * sin_phi * np.cos(ca[indices]) + p[indices, 1]
        z = r[indices] * sin_phi * np.sin(ca[indices]) + p[indices, 2]
        return np.column_stack([x, y, z])

    def _rotate_body(points_body: np.ndarray) -> np.ndarray:
        xb, yb, zb = points_body.T
        xh, yh, zh = model.rotate(xb * u.R_sun, yb * u.R_sun, zb * u.R_sun)
        return np.column_stack([
            xh.to_value(u.R_sun),
            yh.to_value(u.R_sun),
            zh.to_value(u.R_sun),
        ])

    # Parallels: sweep along the skeleton for a set of azimuth angles
    phi_values = np.linspace(0.0, 2.0 * np.pi, max(1, n_parallels), endpoint=False)
    idx_for_parallels = core_indices if not include_legs else np.arange(total_nodes)
    parallels = []
    for phi in phi_values:
        body_curve = _build_body_curve(phi, idx_for_parallels)
        parallels.append(_rotate_body(body_curve))

    # Meridians: revolve selected skeleton nodes around the CME axis
    theta_samples = np.linspace(0.0, 2.0 * np.pi, max(nbvertcircshell, 120))
    if include_legs:
        idx_candidates = np.arange(total_nodes)
    else:
        idx_candidates = core_indices
    if idx_candidates.size == 0:
        idx_candidates = np.arange(total_nodes)
    meridian_indices = np.unique(
        np.clip(
            np.round(np.linspace(0, idx_candidates.size - 1, max(1, n_meridians))).astype(int),
            0,
            idx_candidates.size - 1,
        )
    )
    meridians = []
    for idx in idx_candidates[meridian_indices]:
        cos_theta = np.cos(theta_samples)
        sin_theta = np.sin(theta_samples)
        x = r[idx] * cos_theta + p[idx, 0]
        y = r[idx] * sin_theta * np.cos(ca[idx]) + p[idx, 1]
        z = r[idx] * sin_theta * np.sin(ca[idx]) + p[idx, 2]
        body_curve = np.column_stack([x, y, z])
        meridians.append(_rotate_body(body_curve))

    legs = []
    if include_legs and leg_len > 1:
        leg_indices = [np.arange(0, leg_len), np.arange(total_nodes - leg_len, total_nodes)]
        leg_phi_values = [0.0, np.pi]
        for indices in leg_indices:
            for phi in leg_phi_values:
                body_curve = _build_body_curve(phi, indices)
                # Avoid duplicating very short legs (close to radial minimum)
                radial_mask = np.linalg.norm(body_curve, axis=1) >= leg_r_min
                if np.count_nonzero(radial_mask) < 2:
                    continue
                legs.append(_rotate_body(body_curve[radial_mask]))

    return {
        "parallels": parallels,
        "meridians": meridians,
        "legs": legs,
    }
