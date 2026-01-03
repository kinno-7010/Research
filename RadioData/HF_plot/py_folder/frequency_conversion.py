import numpy as np
    
_e, _m_e, _eps0 = 1.60217663e-19, 9.1093837015e-31, 8.8541878128e-12 # C, kg, F/m

def density_from_frequency(freq_mhz, mode="fundamental"):
    """
    freq_mhz [MHz] → n_e [cm^-3]
    mode: "fundamental" or "harmonic"
    """
    if mode == "fundamental":
        f_hz = freq_mhz * 1e6
    elif mode == "harmonic":
        f_hz = freq_mhz * 1e6 / 2
    else:
        raise ValueError(f"Invalid mode: {mode}")

    omega_p = 2 * np.pi * f_hz
    ne_m3 = _eps0 * _m_e / _e**2 * omega_p**2
    return ne_m3 / 1e6  # cm^-3

def frequency_from_density(dens_cm3):
    """
    dens_cm3 [cm^-3] → freq_mhz [MHz]
    """
    f_hz = np.sqrt(dens_cm3 * 1e6 * _e**2 / (_eps0 * _m_e))
    return f_hz / (2 * np.pi) / 1e6
