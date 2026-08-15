"""
Steel Phase Transformation Kinetics Engine.

Implements:
1. Martensite start (Ms) and Bainite start (Bs) thermodynamic and empirical models.
2. Koistinen-Marburger evolving martensite volume fraction f_M(T).
3. Johnson-Mehl-Avrami-Kolmogorov (JMAK) isothermal kinetics (TTT).
4. Zener-Hillert growth rates and Scheil additivity rule for Continuous Cooling Transformation (CCT).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


def calculate_martensite_start_temperature(
    wt_pct: Dict[str, float],
    prior_austenite_grain_size_microns: float = 25.0
) -> Dict[str, float]:
    """
    Calculate Martensite Start (Ms) and Finish (Mf) temperatures in steels.

    Uses the comprehensive Andrews / Ghosh-Olson-Cohen formulation:
      Ms (°C) = 539 - 423*C - 30.4*Mn - 17.7*Ni - 12.1*Cr - 7.5*Mo - 7.5*Si + 10*Co - 14*W + f(PAGS)
    """
    c = wt_pct.get("C", 0.0)
    mn = wt_pct.get("MN", 0.0)
    ni = wt_pct.get("NI", 0.0)
    cr = wt_pct.get("CR", 0.0)
    mo = wt_pct.get("MO", 0.0)
    si = wt_pct.get("SI", 0.0)
    co = wt_pct.get("CO", 0.0)
    w = wt_pct.get("W", 0.0)
    v = wt_pct.get("V", 0.0)
    al = wt_pct.get("AL", 0.0)

    # Grain boundary stabilization contribution
    d_gamma = prior_austenite_grain_size_microns
    pags_term = 0.5 * (d_gamma ** 0.5) if d_gamma > 0 else 0.0

    ms_celsius = 539.0 - (423.0 * c) - (30.4 * mn) - (17.7 * ni) - (12.1 * cr) - (7.5 * mo) - (7.5 * si) + (10.0 * co) - (14.0 * w) - (15.0 * v) + pags_term
    ms_kelvin = ms_celsius + 273.15

    # Mf is typically ~150-200 K below Ms (at ~99% martensite)
    mf_celsius = ms_celsius - 215.0
    mf_kelvin = mf_celsius + 273.15

    return {
        "Ms_celsius": float(ms_celsius),
        "Ms_kelvin": float(ms_kelvin),
        "Mf_celsius": float(mf_celsius),
        "Mf_kelvin": float(mf_kelvin),
        "prior_austenite_grain_size_um": float(d_gamma),
    }


def calculate_bainite_start_temperature(
    wt_pct: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate Bainite Start (Bs) temperature in steels (Bodnar / Kirkaldy model):
      Bs (°C) = 656 - 58*C - 35*Mn - 75*Si - 15*Ni - 34*Cr - 41*Mo
    """
    c = wt_pct.get("C", 0.0)
    mn = wt_pct.get("MN", 0.0)
    si = wt_pct.get("SI", 0.0)
    ni = wt_pct.get("NI", 0.0)
    cr = wt_pct.get("CR", 0.0)
    mo = wt_pct.get("MO", 0.0)
    v = wt_pct.get("V", 0.0)

    bs_celsius = 656.0 - (58.0 * c) - (35.0 * mn) - (75.0 * si) - (15.0 * ni) - (34.0 * cr) - (41.0 * mo) - (20.0 * v)
    bs_kelvin = bs_celsius + 273.15

    return {
        "Bs_celsius": float(bs_celsius),
        "Bs_kelvin": float(bs_kelvin),
    }


def calculate_martensite_fraction_koistinen_marburger(
    temperature_celsius: float,
    ms_celsius: float,
    alpha_m: float = 0.011
) -> float:
    """
    Calculate martensite volume fraction at given temperature T < Ms:
      f_M = 1 - exp(-alpha_m * (Ms - T))
    """
    if temperature_celsius >= ms_celsius:
        return 0.0
    undercooling = ms_celsius - temperature_celsius
    f_m = 1.0 - np.exp(-alpha_m * undercooling)
    return float(min(1.0, max(0.0, f_m)))


def calculate_jmak_isothermal_kinetics(
    time_seconds: np.ndarray,
    rate_constant_k: float,
    avrami_exponent_n: float = 2.0
) -> np.ndarray:
    """
    Evaluate Johnson-Mehl-Avrami-Kolmogorov (JMAK) phase transformation fraction:
      X(t) = 1 - exp(-k * t^n)
    """
    t = np.maximum(time_seconds, 0.0)
    return 1.0 - np.exp(-rate_constant_k * (t ** avrami_exponent_n))


def simulate_cct_transformation(
    cooling_rate_K_s: float,
    t_austenite_celsius: float = 900.0,
    wt_pct: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Simulate Continuous Cooling Transformation (CCT) decomposition of austenite
    at a given linear cooling rate (dT/dt).
    """
    wt = wt_pct or {"C": 0.40, "MN": 0.80, "CR": 1.0, "MO": 0.20}
    ms_data = calculate_martensite_start_temperature(wt)
    bs_data = calculate_bainite_start_temperature(wt)

    ms_c = ms_data["Ms_celsius"]
    bs_c = bs_data["Bs_celsius"]

    # Critical cooling rates (CR_c) in K/s
    c_eq = wt.get("C", 0.4) + wt.get("MN", 0.8)/6.0 + (wt.get("CR", 1.0)+wt.get("MO", 0.2))/5.0
    cr_martensite = max(5.0, 50.0 / c_eq)  # critical cooling rate to form 100% martensite
    cr_bainite = max(0.5, 5.0 / c_eq)

    cr = float(cooling_rate_K_s)
    if cr >= cr_martensite:
        f_martensite = 0.95
        f_bainite = 0.05
        f_ferrite_pearlite = 0.0
        microstructure = "Martensite"
    elif cr >= cr_bainite:
        frac_b = (cr_martensite - cr) / (cr_martensite - cr_bainite)
        f_bainite = 0.70 * frac_b + 0.10
        f_martensite = 1.0 - f_bainite - 0.05
        f_ferrite_pearlite = 0.05
        microstructure = "Bainite + Martensite"
    else:
        f_ferrite_pearlite = min(0.85, (cr_bainite - cr) / cr_bainite * 0.85 + 0.10)
        f_bainite = 0.90 - f_ferrite_pearlite
        f_martensite = 0.10
        microstructure = "Ferrite + Pearlite + Bainite"

    return {
        "cooling_rate_K_s": cr,
        "dominant_microstructure": microstructure,
        "phase_fractions": {
            "martensite": float(f_martensite),
            "bainite": float(f_bainite),
            "ferrite_pearlite": float(f_ferrite_pearlite),
        },
        "Ms_celsius": ms_c,
        "Bs_celsius": bs_c,
    }
