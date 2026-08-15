"""
Steel Tempering Kinetics, Carbide Coarsening, and Hardness Prediction Models.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def calculate_hollomon_jaffe_parameter(
    temperature_celsius: float,
    holding_time_hours: float,
    c_constant: float = 20.0
) -> float:
    """
    Calculate Hollomon-Jaffe tempering parameter:
      P_HJ = T_K * (log10(t_hours) + C)
    """
    t_k = temperature_celsius + 273.15
    t_hrs = max(holding_time_hours, 1e-4)
    p_hj = t_k * (np.log10(t_hrs) + c_constant)
    return float(p_hj)


def calculate_as_quenched_hardness(wt_pct_carbon: float) -> Dict[str, float]:
    """
    Calculate maximum as-quenched martensite hardness (HRC & HV) from carbon content:
      HRC_max = 20 + 60 * sqrt(%C)
    """
    c = max(0.01, min(1.2, wt_pct_carbon))
    hrc = min(67.0, 20.0 + 60.0 * np.sqrt(c))
    # Approximate conversion from HRC to Vickers HV
    hv = 100.0 + 10.0 * hrc + 0.15 * (hrc ** 2) if hrc > 20 else 200.0
    return {
        "as_quenched_HRC": float(hrc),
        "as_quenched_HV": float(hv),
        "carbon_wt_pct": float(c),
    }


def calculate_tempered_hardness(
    wt_pct: Dict[str, float],
    tempering_temperature_celsius: float,
    holding_time_hours: float = 2.0
) -> Dict[str, float]:
    """
    Calculate tempered steel hardness (HRC & HV) using Hollomon-Jaffe time-temperature parameterization
    with secondary hardening contributions from Cr, Mo, V, W carbide precipitation.
    """
    c = wt_pct.get("C", 0.40)
    cr = wt_pct.get("CR", 0.0)
    mo = wt_pct.get("MO", 0.0)
    v = wt_pct.get("V", 0.0)
    w = wt_pct.get("W", 0.0)

    p_hj = calculate_hollomon_jaffe_parameter(tempering_temperature_celsius, holding_time_hours)
    as_q = calculate_as_quenched_hardness(c)
    hrc_0 = as_q["as_quenched_HRC"]

    # Base tempering softening curve
    # Normalized parameter scaling
    dp = max(0.0, (p_hj - 10000.0) / 1000.0)
    softening = 3.5 * dp

    # Secondary hardening peak (Cr, Mo, V, W alloy carbides at 500-600 °C)
    t_c = tempering_temperature_celsius
    sec_carbides = 0.5 * cr + 1.5 * mo + 3.0 * v + 0.8 * w
    if 450.0 <= t_c <= 620.0 and sec_carbides > 0.2:
        sec_peak = sec_carbides * np.exp(-((t_c - 540.0) / 60.0) ** 2)
    else:
        sec_peak = 0.0

    hrc_tempered = max(20.0, hrc_0 - softening + sec_peak)
    hv_tempered = 100.0 + 10.0 * hrc_tempered + 0.15 * (hrc_tempered ** 2)
    yield_strength_mpa = 3.2 * hv_tempered  # Approximate relation sigma_y ~ 3.2 * HV

    return {
        "tempered_HRC": float(hrc_tempered),
        "tempered_HV": float(hv_tempered),
        "estimated_yield_strength_MPa": float(yield_strength_mpa),
        "Hollomon_Jaffe_P": float(p_hj),
        "secondary_hardening_increment_HRC": float(sec_peak),
    }
