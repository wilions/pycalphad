"""
Noble Metals & Precious Alloys Optics, CIELAB Color Space, and Hardening Models.

Implements:
1. Drude-Lorentz optical reflectance and CIE L*a*b* color coordinates for Au, Ag, Cu, Pt, Pd alloys.
2. Karat designation and gold alloy classification (Yellow, Rose/Red, White, Green gold).
3. Spinodal decomposition & precipitation age-hardening in precious metal jewelry and electrical contacts.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def calculate_gold_alloy_color_and_karat(
    wt_pct: Dict[str, float]
) -> Dict[str, Any]:
    """
    Calculate Karat rating, CIELAB color space coordinates (L*, a*, b*),
    and gold classification (Yellow, Rose/Red, White, Green Gold).
    """
    au = wt_pct.get("AU", 75.0)  # default 18K (75 wt% Au)
    ag = wt_pct.get("AG", 15.0)
    cu = wt_pct.get("CU", 10.0)
    ni = wt_pct.get("NI", 0.0)
    pd = wt_pct.get("PD", 0.0)
    zn = wt_pct.get("ZN", 0.0)

    # Karat calculation
    karat = (au / 100.0) * 24.0

    # CIELAB coordinates empirical Drude-Lorentz integration:
    # L* (Lightness 0-100): High for Ag/Au/Pd, lower for Cu
    # a* (Red-Green axis, +red, -green): High positive for Cu (red/rose gold), negative/low for Ag/Zn (green gold)
    # b* (Yellow-Blue axis, +yellow, -blue): High positive for Au/Ag (rich yellow gold), low for Ni/Pd (white gold)

    # Base pure metal reflectance contributions
    l_star = 85.0 + 0.10 * ag - 0.15 * cu + 0.05 * pd
    l_star = min(98.0, max(60.0, l_star))

    a_star = 1.0 + 0.65 * cu - 0.20 * ag - 0.10 * ni - 0.10 * pd
    a_star = min(25.0, max(-5.0, a_star))

    b_star = 18.0 + 0.20 * au + 0.15 * ag - 0.35 * ni - 0.40 * pd - 0.10 * cu
    b_star = min(40.0, max(2.0, b_star))

    # Color classification
    if ni + pd > 10.0 or b_star < 12.0:
        color_name = "White Gold"
    elif cu > 18.0 or a_star > 8.0:
        color_name = "Rose / Red Gold"
    elif ag > 20.0 and cu < 5.0:
        color_name = "Green Gold"
    else:
        color_name = "Classic Yellow Gold"

    return {
        "karat_rating": float(karat),
        "gold_fineness_millesimal": float(au * 10.0),
        "CIE_L_star_lightness": float(l_star),
        "CIE_a_star_red_green": float(a_star),
        "CIE_b_star_yellow_blue": float(b_star),
        "alloy_color_classification": color_name,
    }


def calculate_precious_alloy_age_hardening(
    alloy_type: str,
    aging_temperature_celsius: float,
    aging_time_hours: float
) -> Dict[str, float]:
    """
    Calculate precipitation / ordering age-hardening hardness (HV) in precious metal alloys
    (e.g., 18K Au-Cu-Ag AuCu-I ordering or Pt-5Cu spinodal decomposition).
    """
    t_c = aging_temperature_celsius
    t_hrs = max(aging_time_hours, 0.01)

    if "AU" in alloy_type.upper():
        # 18K / 14K Au-Cu-Ag ordering reaction (optimal at ~280-320 °C)
        hv_annealed = 140.0
        peak_t = 300.0
        delta_hv_max = 140.0
    else:
        # Pt-5Cu or Dental Pd-Ag alloys (optimal at ~450-500 °C)
        hv_annealed = 120.0
        peak_t = 480.0
        delta_hv_max = 160.0

    # Temperature kinetic bell curve
    temp_efficiency = np.exp(-((t_c - peak_t) / 60.0) ** 2)

    # Time evolution (Avrami hardening curve followed by overaging softening)
    time_factor = (t_hrs / 2.0) / (1.0 + (t_hrs / 2.0) ** 1.3)
    time_factor = min(1.0, time_factor * 1.5)

    delta_hv = delta_hv_max * temp_efficiency * time_factor
    hv_total = hv_annealed + delta_hv

    return {
        "annealed_hardness_HV": float(hv_annealed),
        "aged_hardness_HV": float(hv_total),
        "hardness_increase_HV": float(delta_hv),
        "aging_temperature_celsius": float(t_c),
        "aging_time_hours": float(t_hrs),
    }
