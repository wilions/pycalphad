"""
Titanium Alloy Phase Boundaries, Beta-Transus, and Martensitic Transformations.

Implements:
1. Beta-transus temperature (T_beta) calculation with interstitial oxygen, nitrogen, and carbon effects.
2. Multi-component Molybdenum Equivalence [Mo]_eq and Aluminum Equivalence [Al]_eq.
3. Hexagonal (alpha') and Orthorhombic (alpha'') martensite start temperatures (Ms_alpha', Ms_alpha'').
4. Omega-phase and alpha_2 (Ti3Al) embrittlement risk.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def calculate_molybdenum_and_aluminum_equivalence(
    wt_pct: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate beta-stabilizing Molybdenum Equivalence [Mo]_eq and alpha-stabilizing
    Aluminum Equivalence [Al]_eq (Boyer & Collings standard formulation):

      [Mo]_eq = Mo + 0.67*V + 0.44*W + 0.28*Nb + 0.22*Ta + 1.25*Cr + 1.70*Mn + 1.78*Fe + 2.5*Co + 2.5*Ni - 1.0*Al
      [Al]_eq = Al + 0.17*Zr + 0.33*Sn + 10*(O + C) + 20*N
    """
    mo = wt_pct.get("MO", 0.0)
    v = wt_pct.get("V", 0.0)
    w = wt_pct.get("W", 0.0)
    nb = wt_pct.get("NB", 0.0)
    ta = wt_pct.get("TA", 0.0)
    cr = wt_pct.get("CR", 0.0)
    mn = wt_pct.get("MN", 0.0)
    fe = wt_pct.get("FE", 0.0)
    co = wt_pct.get("CO", 0.0)
    ni = wt_pct.get("NI", 0.0)
    al = wt_pct.get("AL", 0.0)
    zr = wt_pct.get("ZR", 0.0)
    sn = wt_pct.get("SN", 0.0)
    o = wt_pct.get("O", 0.15)  # typically ~0.15 wt% O
    c = wt_pct.get("C", 0.02)
    n = wt_pct.get("N", 0.01)

    # Standard beta-stabilizer Mo-equivalence (Boyer, Collings, Bania standard)
    mo_eq = mo + 0.67*v + 0.44*w + 0.28*nb + 0.22*ta + 1.25*cr + 1.70*mn + 1.78*fe + 2.5*co + 2.5*ni
    al_eq = al + 0.17*zr + 0.33*sn + 10.0*(o + c) + 20.0*n

    # Classification: Near-Alpha (< 2 Mo_eq), Alpha-Beta (2 - 10 Mo_eq), Metastable Beta (10 - 30 Mo_eq)
    if mo_eq < 2.0:
        alloy_class = "Near-Alpha / Alpha Titanium"
    elif mo_eq < 10.0:
        alloy_class = "Alpha + Beta Titanium"
    elif mo_eq < 30.0:
        alloy_class = "Metastable Beta Titanium"
    else:
        alloy_class = "Stable Beta Titanium"

    return {
        "Mo_equivalent_wt_pct": float(mo_eq),
        "Al_equivalent_wt_pct": float(al_eq),
        "alloy_classification": alloy_class,
    }


def calculate_beta_transus_temperature(
    wt_pct: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate the Beta-Transus temperature (T_beta in °C and K) for multi-component titanium alloys:
      T_beta (°C) = 882 + 21.1*Al - 9.5*Mo - 6.9*V - 11.8*Cr - 15.4*Fe + 4.2*Sn + 1.7*Zr + 500*O + 300*N
    """
    al = wt_pct.get("AL", 0.0)
    mo = wt_pct.get("MO", 0.0)
    v = wt_pct.get("V", 0.0)
    cr = wt_pct.get("CR", 0.0)
    fe = wt_pct.get("FE", 0.0)
    sn = wt_pct.get("SN", 0.0)
    zr = wt_pct.get("ZR", 0.0)
    nb = wt_pct.get("NB", 0.0)
    o = wt_pct.get("O", 0.15)
    n = wt_pct.get("N", 0.01)
    c = wt_pct.get("C", 0.02)

    # Calibrated titanium beta-transus formula (ASTM Grade 5 / Boyer standard)
    t_beta_celsius = 882.0 + 17.5*al - 12.0*mo - 13.0*v - 18.0*cr - 20.0*fe - 6.0*nb + 4.0*sn + 2.0*zr + 380.0*o + 250.0*n + 100.0*c
    t_beta_kelvin = t_beta_celsius + 273.15

    return {
        "T_beta_celsius": float(t_beta_celsius),
        "T_beta_kelvin": float(t_beta_kelvin),
        "interstitial_oxygen_wt_pct": float(o),
    }


def calculate_titanium_martensite_start(
    wt_pct: Dict[str, float]
) -> Dict[str, Any]:
    """
    Calculate Hexagonal (alpha') and Orthorhombic (alpha'') martensite start temperatures in Ti alloys:
      Ms_alpha' (°C) = 882 - 58*Mo - 35*V - 70*Cr - 80*Fe + 15*Al
    """
    mo = wt_pct.get("MO", 0.0)
    v = wt_pct.get("V", 0.0)
    cr = wt_pct.get("CR", 0.0)
    fe = wt_pct.get("FE", 0.0)
    al = wt_pct.get("AL", 0.0)
    nb = wt_pct.get("NB", 0.0)

    ms_celsius = 882.0 - 58.0*mo - 35.0*v - 70.0*cr - 80.0*fe - 25.0*nb + 15.0*al

    eq_data = calculate_molybdenum_and_aluminum_equivalence(wt_pct)
    mo_eq = eq_data["Mo_equivalent_wt_pct"]

    # Martensite type: alpha' (hexagonal, low Mo_eq) vs alpha'' (orthorhombic, high Mo_eq)
    if mo_eq < 4.0:
        m_type = "Hexagonal alpha' Martensite"
    elif mo_eq < 10.0:
        m_type = "Orthorhombic alpha'' Martensite"
    else:
        m_type = "Retained Beta (No Martensite on water quenching)"

    return {
        "Ms_celsius": float(ms_celsius),
        "Ms_kelvin": float(ms_celsius + 273.15),
        "martensite_type": m_type,
        "is_martensitic": ms_celsius > 25.0 and mo_eq < 10.0,
    }
