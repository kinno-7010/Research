import numpy as np

# Physical constants
_e = 1.60217662e-19    # C
_m_e = 9.10938356e-31   # kg
_eps0 = 8.854187817e-12  # F/m

def density_from_frequency(freq_mhz):
    """
    freq_mhz [MHz] → n_e [cm^-3]
    """
    f_hz = freq_mhz * 1e6
    omega_p = 2 * np.pi * f_hz
    ne_m3 = _eps0 * _m_e / _e**2 * omega_p**2
    return ne_m3 / 1e6  # cm^-3

def frequency_from_density(dens_cm3):
    """
    dens_cm3 [cm^-3] → freq_mhz [MHz]
    """
    n_m3 = dens_cm3 * 1e6
    f_hz = np.sqrt(n_m3 * _e**2 / (_eps0 * _m_e))
    return f_hz / (2 * np.pi) / 1e6

def density_from_frequency_harmonic(freq_mhz):
    """
    freq_mhz [MHz] → n_e [cm^-3]
    """
    f_hz_harmonic = freq_mhz * 1e6 / 2
    omega_p = 2 * np.pi * f_hz_harmonic
    ne_m3 = _eps0 * _m_e / _e**2 * omega_p**2
    return ne_m3 / 1e6  # cm^-3

def frequency_from_density_harmonic(dens_cm3):
    """
    dens_cm3 [cm^-3] → freq_mhz [MHz]
    """
    n_m3 = dens_cm3 * 1e6
    f_hz_harmonic = np.sqrt(n_m3 * _e**2 / (_eps0 * _m_e)) / 2
    return f_hz_harmonic / (2 * np.pi) / 1e6