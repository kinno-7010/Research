# -*- coding: utf-8 -*-
"""
Core constants & kernels for van de Hulst / Billings / Hayes white-light pB inversion.

- A(r), B(r): van de Hulst (1950) / Billings (1966) Thomson-scattering geometric kernels.
- u: linear limb-darkening coefficient in I(µ) = I0[(1-u)+uµ] (visible continuum).
- K_CONST: LOS normalization consistent with pB expressed in B_sun units.
           C' = (3/8) * sigma_T * R_SUN / (1 - u/3)  [van de Hulst; Billings; Hayes]
Notes:
- Axisymmetry is NOT assumed in A,B; it is assumed when inverting to N_e(r).
- Units: r_cm in centimeters; r (without suffix) in R_sun when used in outer code.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import fsolve, root_scalar
from scipy.integrate import trapezoid
from set_u_from_instrument import compute_u_for_instrument

# --- Physical constants / configuration ---
R_SUN = 6.96e10                 # cm
sigma_T = 6.6524587321e-25      # cm^2 (Thomson cross section, CODATA-2014/2018 value)

def _compute_K_const(u_val: float) -> float:
    """
    LOS 正規化定数 K_CONST を u から計算する補助関数。

    元々の実装:
        K_CONST = π σ_T (1 - u) / 6
    をそのまま用い、u が変わるたびに再計算する。
    """
    return np.pi * sigma_T * (1.0 - u_val) / 6.0

# ---- 初期値：デフォルトでは LASCO C2 の u を使う ----
# （後から u_from_LASCO / u_from_KCOR / set_u_from_instrument(...) で上書き可能）
u = compute_u_for_instrument("lasco_c2", use_allen=False)
K_CONST = _compute_K_const(u)

def set_u(u_new: float) -> float:
    """
    グローバルな limb-darkening 係数 u と K_CONST を整合的に更新する。

    Parameters
    ----------
    u_new : float
        新しい limb-darkening 係数。

    Returns
    -------
    u : float
        実際に設定された u（float(u_new)）。
    """
    global u, K_CONST
    u = float(u_new)
    K_CONST = _compute_K_const(u)
    return u

def u_from_LASCO(use_allen: bool = False) -> float:
    """
    SOHO/LASCO-C2 用の u と K_CONST を設定し、その u を返す。

    instrument 名は set_u_from_instrument() 側では "lasco", "lasco_c2", "soho/lasco" などで認識される。
    """
    # u_val = compute_u_for_instrument("lasco_c2", use_allen=use_allen)
    u_val_lasco = 0.6134999999999997
    # return set_u(u_val)
    return u_val_lasco

def u_from_KCOR(use_allen: bool = False) -> float:
    """
    MLSO COSMO K-Coronagraph (K-Cor) 用の u と K_CONST を設定し、その u を返す。
    """
    # u_val = compute_u_for_instrument("kcor", use_allen=use_allen)
    u_val_kcor = 0.45300000000000007
    # return set_u(u_val)
    return u_val_kcor

def set_u_from_instrument(instrument: str, use_allen: bool = False) -> float:
    """
    自由形式の instrument 名から u を自動選択して設定する便利関数。

    Examples
    --------
    set_u_from_instrument("SOHO/LASCO C2")
    set_u_from_instrument("LASCO C2")
    set_u_from_instrument("K-Cor")
    set_u_from_instrument("Mk4")
    """
    inst = instrument.lower()

    # LASCO C2 系
    if "lasco" in inst or " c2" in inst or inst == "lasco_c2":
        print(f"u_from_LASCO(use_allen={use_allen}): {u_from_LASCO(use_allen=use_allen)}")
        u_val_lasco = 0.6134999999999997
        set_u(u_val_lasco)
        return u_val_lasco
        

    # K-Cor / Mk4 系
    if "kcor" in inst or "k-cor" in inst or "kcoronagraph" in inst or "mk4" in inst:
        print(f"u_from_KCOR(use_allen={use_allen}): {u_from_KCOR(use_allen=use_allen)}")
        u_val_kcor = 0.45300000000000007
        set_u(u_val_kcor)
        return u_val_kcor
    

    raise ValueError(f"Unknown instrument name for limb darkening: {instrument!r}")

# Plasma constants (for plasma frequency conversions)
_e = 1.60217662e-19     # C
_m_e = 9.10938356e-31   # kg
_eps0 = 8.854187817e-12 # F/m

# --- van de Hulst / Billings geometric kernels ---
def van_de_hulst_A(r_cm: np.ndarray | float) -> np.ndarray | float:
    """
    A(r) = cosΩ * sin^2Ω, with sinΩ = R_sun / r.
    r_cm: heliocentric distance in cm.
    """
    sinO = np.clip(R_SUN / r_cm, 0.0, 1.0)
    cosO = np.sqrt(1.0 - sinO**2)
    return cosO * sinO**2

def van_de_hulst_B(r_cm: np.ndarray | float) -> np.ndarray | float:
    """
    B(r) = -(1/8) * [(1 - 3 sin^2Ω - cos^2Ω)/(1 + 3 sin^2Ω)] * sinΩ * ln((1+sinΩ)/cosΩ).
    """
    sinO = np.clip(R_SUN / r_cm, 0.0, 1.0)
    cosO = np.sqrt(1.0 - sinO**2)
    denom = 1.0 + 3.0 * sinO**2
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    cos_safe = np.where(cosO < 1e-30, 1e-30, cosO)
    logterm = np.log((1.0 + sinO) / cos_safe)
    term1 = 1.0 - 3.0 * sinO**2 - cosO**2  # = -2 sin^2Ω
    return -0.125 * (term1 / denom) * sinO * logterm

def weight_pB(r_cm: np.ndarray | float) -> np.ndarray | float:
    """
    pB angular kernel: (1-u)A + u B.
    This is algebraically equivalent to Hayes' [A - B] notation where A,B absorb u.
    """
    return (1.0 - u) * van_de_hulst_A(r_cm) + u * van_de_hulst_B(r_cm)

# --- 5-term power-law model for Ne(r) ---
def triple_power(r: np.ndarray, A1, p1, A2, p2, A3, p3, A4, p4, A5, p5):
    return A1*r**p1 + A2*r**p2 + A3*r**p3 + A4*r**p4 + A5*r**p5

# --- Discrete "ablation" inversion using concentric shells (same math as original code) ---
def invert_ablation(pb_prof: np.ndarray, r_mid: np.ndarray, edges: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Invert pB (in B_sun units) to electron density using a shell-by-shell subtraction,
    consistent with van de Hulst LOS geometry and the kernel (1-u)A + uB.
    """
    Ne = np.zeros_like(pb_prof)
    rho0 = r_mid[-1] * R_SUN

    # Tail beyond outermost shell ~ r^{-b} with b=3 (original code choice retained)
    tail_b = 3.0
    r_out = rho0 * 1.0001
    r_grid = np.linspace(r_out, 10.0*rho0, 2000)
    Wgrid = weight_pB(r_grid)
    geom = (rho0**2)/(r_grid * np.sqrt(r_grid**2 - rho0**2))
    tail_factor = trapezoid(Wgrid * geom * (rho0/r_grid)**tail_b, r_grid)
    Ne[-1] = pb_prof[-1] / (K_CONST * tail_factor)

    # March inward
    for i in range(n_bins-2, -1, -1):
        rho = r_mid[i] * R_SUN
        pB_rem = pb_prof[i] / K_CONST
        for j in range(i+1, n_bins):
            if Ne[j] <= 0:
                continue
            r_in = max(edges[j], r_mid[i]) * R_SUN
            r_outj = edges[j+1] * R_SUN
            L_j = rho * (
                np.arctan(np.sqrt(r_outj**2 - rho**2)/rho) -
                np.arctan(np.sqrt(max(r_in**2 - rho**2, 0.0))/rho)
            )
            Wj = weight_pB(r_mid[j]*R_SUN)
            pB_rem -= Wj * Ne[j] * L_j
        L_i = rho * np.arctan(np.sqrt((edges[i+1]*R_SUN)**2 - rho**2)/rho)
        Wi = weight_pB(r_mid[i]*R_SUN)
        Ne[i] = pB_rem / (Wi * L_i) if Wi*L_i > 0 else 0.0
    return Ne

# --- Frequency-density utilities ---
def density_from_frequency(freq_mhz: float | np.ndarray) -> float | np.ndarray:
    """freq [MHz] -> n_e [cm^-3]"""
    f_hz_harmonic = freq_mhz * 1e6
    omega_p = 2 * np.pi * f_hz_harmonic
    ne_m3 = _eps0 * _m_e / _e**2 * omega_p**2
    return ne_m3 / 1e6     # [cm^-3]

def frequency_from_density(dens_cm3: float | np.ndarray) -> float | np.ndarray:
    """n_e [cm^-3] -> freq [MHz]"""
    n_m3 = dens_cm3 * 1e6
    f_hz_harmonic = np.sqrt(n_m3 * _e**2 / (_eps0 * _m_e))
    return f_hz_harmonic / (2 * np.pi) / 1e6


def BaumbachAllen(rho):
    return 1e8*(2.99*rho**(-16) + 1.55*rho**(-6))

# --- Empirical coronal density models ---
def Saito1970(rho, phi: float = 0.0):
    sin_phi = np.sin(np.radians(phi))
    return (3.09e8 * (rho) ** -16 * (1 - 0.5 * sin_phi) +
            1.58e8 * rho ** -6 * (1 - 0.95 * sin_phi) +
            0.0251e8 * rho ** -2.5 * (1 - np.sqrt(sin_phi)))

def Saito1977(rho):  # 2.5–5.5 Rs (background component returned)
    C1 = [1.36e6, 5.27e4, 3.15e5]
    C2 = [1.68e8, 3.54e6, 1.60e6]
    d1 = [2.14, 3.30, 4.71]
    d2 = [6.13, 5.80, 3.01]
    background = C1[0]*rho**(-d1[0]) + C2[0]*rho**(-d2[0])
    # eq_hole = C1[1]*rho**(-d1[1]) + C2[1]*rho**(-d2[1])
    # pole_hole = C1[2]*rho**(-d1[2]) + C2[2]*rho**(-d2[2])
    return background

def Newkirk1961(rho):
    return 4.4e4 * 10**(4.32/rho)

# Plasma frequency helper (MHz) for a given n_e (cm^-3)
_K = (1/(2*np.pi) * np.sqrt((_e)**2 / (_eps0 * _m_e)) * 1e-6)  # exact factor to MHz
def f_plasma(n_e_cm3: np.ndarray | float) -> np.ndarray | float:
    return _K * np.sqrt(n_e_cm3)

def rho_at_frequency(f_MHz: float, model, x0: float = 1.5) -> float:
    """Solve model(rho) such that f_p(model(rho)) = f_MHz."""
    func = lambda r: f_plasma(model(r)) - f_MHz
    rho_solution, = fsolve(func, x0=float(x0), xtol=1e-10, maxfev=10000)
    return rho_solution

def find_rho_for_value(func, target, rho_min=1.0, rho_max=4.0):
    g = lambda rho: func(rho) - target
    sol = root_scalar(g, bracket=[rho_min, rho_max], method='brentq')
    if sol.converged:
        return sol.root
    else:
        raise RuntimeError("解の探索に失敗しました。範囲や関数の形を確認してください。")
