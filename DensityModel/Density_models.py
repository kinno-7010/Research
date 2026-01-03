import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import math

# ----------------------- Density model -----------------------
# Baumbach-Allen model
def Baumbach(rho):
    return 1e8*(2.99*rho**(-16) + 1.55*rho**(-6) + 0.036*rho**(-3/2))

def Baumbach_Allen(rho):
    return 1e8*(2.99*rho**(-16) + 1.55*rho**(-6))

# def Saito1970(rho, phi: float=0):
#     # initial_latitude = 21
#     # rho += 1.0 # 太陽中心からに換算
#     sin_phi = np.sin(np.radians(phi))
#     return (3.09e8 * (rho) ** -16 * (1 - 0.5 * sin_phi) +
#             1.58e8 * rho ** -6 * (1 - 0.95 * sin_phi) +
#             0.0251e8 * rho ** -2.5 * (1 - np.sqrt(sin_phi)))

# def Vrsnak2004(rho):
#     return 1e8 * (15.45 * rho ** -16 + 3.16 * rho ** -6 + 1 * rho ** -4 + 0.0033 * rho ** -2)


def Saito1977(rho):  # 2.5-5.5Rs
    C1 = [1.36e6, 5.27e6, 3.15e6]
    C2 = [1.68e8, 3.54e6, 1.60e6]
    d1 = [2.14, 3.30, 4.71]
    d2 = [6.13, 5.80, 3.01]
    
    background = C1[0]*rho**(-d1[0])+C2[0]*rho**(-d2[0])
    eq_hole = C1[1]*rho**(-d1[1])+C2[1]*rho**(-d2[1])
    pole_hole = C1[2]*rho**(-d1[2])+C2[2]*rho**(-d2[2])
    return background, eq_hole, pole_hole


def Newkirk1961(rho):
    return 4.2e4*10**(4.32/rho)

# def Wang2017(rho):
#     maximum = -4.42158e6*rho**(-1)+5.41656e7*rho**(-2)+-1.86150e8*rho**(-3)+2.13102e8*rho**(-4)
#     # minimum = 3.53766e5*rho**(-1)+1.03359e7*rho**(-2)+-5.46541e7*rho**(-3)+8.24791e7*rho**(-4)
#     return maximum

# def Kumari2017(rho): # 1.5-5.5Rs
#     model = 1.521e8*rho**(-7.279)+1.84e8*rho**(-4.852)+7.52e5*rho**(-2.024)
#     return model

# def Leblanc1996(rho): # 1.8-215Rs
#     model = 3.3e5*rho**(-2)+4.1e6*rho**(-4)+8.0e7*rho**(-6)
#     return model


# def Alvarez_and_Haddock1973(rho):  # 4.8-210Rs
#     return 2.83e6*(rho-0.9)**(-2.15)
    

# def Fainberg_and_Stone1971(rho):  # 10.0-40
#     return 5.52e7*rho**(-2.63)

# --------------------
_m_e = 9.1093837015e-31 # kg
_e = 1.602176634e-19 # C
_eps0 = 8.8541878128e-12 # F/m
kappa = (1/(2*np.pi) * np.sqrt((_e)**2 / (_eps0 * _m_e)) * 1e-3)  # exact factor to MHz


def f_plasma(n_e_cm3: np.ndarray | float) -> np.ndarray | float:
    return kappa * np.sqrt(n_e_cm3)

def invert_r_from_f(model_function: callable, f_mhz: float, branch: str = "F",
                    factor: float = 1.0, r_lo: float = 1.2, r_hi: float = 30.0,
                    max_iter: int = 120) -> float:
    target = float(f_mhz) * (0.5 if branch.upper() == "H" else 1.0)
    lo, hi = float(r_lo), float(r_hi)

    for _ in range(12):
        n_lo = factor * float(model_function(lo))
        n_hi = factor * float(model_function(hi))
        f_lo = kappa * math.sqrt(n_lo)
        f_hi = kappa * math.sqrt(n_hi)
        if f_lo >= target >= f_hi:
            break
        lo = max(1.05, 0.9 * lo)
        hi *= 1.2

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        n_mid = factor * float(model_function(mid))
        f_mid = kappa * math.sqrt(n_mid)
        if f_mid > target:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) < 1e-6:
            break
    return 0.5 * (lo + hi)

def plot_density_model(rho: np.ndarray, branch: str = "F", factor: float = 1.0):
    fig, ax = plt.subplots(figsize=(10, 8), tight_layout=True)
    
    Baumbach_Allen_model = Baumbach_Allen(rho)
    Saito1977_background_model, Saito1977_eq_hole_model, Saito1977_pole_hole_model = Saito1977(rho)
    Newkirk1961_model = Newkirk1961(rho)

    # Plot each model individually to keep (x, y) dimensions aligned
    model_series = [
        ("Baumbach-Allen", f_plasma(Baumbach_Allen_model)),
        ("Saito 1977 background", f_plasma(Saito1977_background_model)),
        ("Saito 1977 equatorial hole", f_plasma(Saito1977_eq_hole_model)),
        ("Saito 1977 polar hole", f_plasma(Saito1977_pole_hole_model)),
        ("Newkirk 1961", f_plasma(Newkirk1961_model)),
    ]
    for label, series in model_series:
        ax.plot(rho, series, label=label)
    
    # r=1.5, 6.0の間を薄いグレーで塗りつぶす
    ylim_min, ylim_max = (0.1, 500)
    xlim_min, xlim_max = (1.0, 7.0)
    ax.axvspan(1.5, 6.0, color="gray", alpha=0.5, zorder=0)
    ax.text(1.5, ylim_min, ' Middle corona\n (1.5-6.0 $R_\\odot$)', fontsize=16, color='red', ha='left', va='bottom')

    def plasma_frequency_at(target_rho: float, density_profile: np.ndarray) -> float:
        """Interpolate density at target_rho and convert to plasma frequency."""
        density_value = float(np.interp(target_rho, rho, density_profile))
        return float(f_plasma(density_value))

    for target_rho, profile, dy, color, align in [
        (1.5, Newkirk1961_model, 5, "purple", ("left", "bottom")),
        (6.0, Newkirk1961_model, 0.1, "purple", ("left", "bottom")),
        (1.5, Saito1977_background_model, -10, "orange", ("right", "bottom")),
        (6.0, Saito1977_background_model, 0.1, "orange", ("left", "bottom")),
        (6.0, Baumbach_Allen_model, -0.01, "blue", ("left", "top")),
    ]:
        freq_val = plasma_frequency_at(target_rho, profile)
        ax.text(target_rho, freq_val + dy, f"{freq_val:.2f} MHz  \n(at {target_rho:.1f} $R_\\odot$)  ",
                fontsize=16, color=color, ha=align[0], va=align[1])

    # ----- Cosmetics --------------------------------------------------------------
    ax.set_yscale('log')
    ax.set_xlabel(r'Radial distance  $r\;[R_\odot]$', fontsize=20)
    ax.set_ylabel(r'Plasma frequency $f_p\;[\mathrm{MHz}]$', fontsize=20)

    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(ylim_min, ylim_max)
    ax.grid(True, which='both', ls=':')
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=16)
    
    output_path = '/mnt/d/wsl/home/kinno-7010/Research/DensityModel/output/density_model_BA_Sa1977_Ne1961.png'
    plt.savefig(output_path)
    print(f"Saved {output_path}")

    plt.show()
    
    print(f'Baumbach-Allen (1.05 Rs): {f_plasma(Baumbach_Allen(1.05)):.2f} MHz')
    print(f'Baumbach-Allen (3.0 Rs): {f_plasma(Baumbach_Allen(3.0)):.2f} MHz')
    print(f'Newkirk1961 (1.0 Rs): {f_plasma(Newkirk1961(1.0)):.2f} MHz')
    print(f'Newkirk1961 (3.0 Rs): {f_plasma(Newkirk1961(3.0)):.2f} MHz')
    print(f'Saito1977 background (2.5 Rs): {f_plasma(Saito1977(2.5)[0]):.2f} MHz')
    print(f'Saito1977 background (5.5 Rs): {f_plasma(Saito1977(5.5)[0]):.2f} MHz')


if __name__ == "__main__":
    rho_values = np.arange(1.0, 7, 0.01)
    plot_density_model(rho_values)