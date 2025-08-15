import numpy as np
from scipy.optimize import root_scalar

try:
    from .spectrum_plot import time, frequency_mhz, data
except ImportError:
    from spectrum_plot import time, frequency_mhz, data

def Saito1970(rho, phi: float=0):
    """
    Saito1970 solar corona density model
    
    Parameters:
    rho: heliocentric distance in solar radii
    phi: latitude in degrees
    
    Returns: electron density in cm^-3
    """
    # initial_latitude = 21
    # rho += 1.0 # 太陽中心からに換算
    sin_phi = np.sin(np.radians(phi))
    return (3.09e8 * (rho) ** -16 * (1 - 0.5 * sin_phi) +
            1.58e8 * rho ** -6 * (1 - 0.95 * sin_phi) +
            0.0251e8 * rho ** -2.5 * (1 - np.sqrt(sin_phi)))


def Saito1977(rho):
    """
    Saito1977 solar corona density model (2.5-5.5Rs)
    
    Parameters:
    rho: heliocentric distance in solar radii
    
    Returns: electron density in cm^-3
    """
    C1 = [1.36e6, 5.27e4, 3.15e5]
    C2 = [1.68e8, 3.54e6, 1.60e6]
    d1 = [2.14, 3.30, 4.71]
    d2 = [6.13, 5.80, 3.01]
    
    background = C1[0]*rho**(-d1[0])+C2[0]*rho**(-d2[0])
    eq_hole = C1[1]*rho**(-d1[1])+C2[1]*rho**(-d2[1])
    pole_hole = C1[2]*rho**(-d1[2])+C2[2]*rho**(-d2[2])
    return background #, eq_hole, pole_hole


def find_rho_for_value(func, target, rho_min=1.0, rho_max=4.0):
    """
    func(rho) = 何らかの連続関数
    target      = その関数がとるべき値
    rho_min     = 探索開始点（下限）
    rho_max     = 探索終了点（上限）
    戻り値      = root_scalar で得られた rho の近似解
    """
    g = lambda rho: func(rho) - target
    sol = root_scalar(g, bracket=[rho_min, rho_max], method='brentq')
    if sol.converged:
        return sol.root
    else:
        raise RuntimeError("解の探索に失敗しました。範囲や関数の形を確認してください。")