"""
Slag-Metal Thermodynamics, Sulfide Capacity, and Refractory Degradation Models.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


OPTICAL_BASICITY_TABLE = {
    "CAO": 1.00,
    "MGO": 0.78,
    "MNO": 0.95,
    "FEO": 1.00,
    "NA2O": 1.15,
    "K2O": 1.40,
    "BAO": 1.15,
    "AL2O3": 0.60,
    "SIO2": 0.48,
    "TIO2": 0.62,
    "P2O5": 0.40,
    "FE2O3": 0.75,
    "CAF2": 1.20,
}

OXYGEN_STOICHIOMETRY = {
    "CAO": 1, "MGO": 1, "MNO": 1, "FEO": 1, "NA2O": 1, "K2O": 1, "BAO": 1,
    "AL2O3": 3, "SIO2": 2, "TIO2": 2, "P2O5": 5, "FE2O3": 3, "CAF2": 1
}


def calculate_optical_basicity(oxide_fractions: Dict[str, float]) -> float:
    """Calculate theoretical optical basicity Lambda of a metallurgical slag."""
    total_eq = 0.0
    weighted_sum = 0.0
    for ox, x in oxide_fractions.items():
        ox_clean = ox.upper().replace("-", "").replace("_", "")
        lam = OPTICAL_BASICITY_TABLE.get(ox_clean, 0.70)
        n_o = OXYGEN_STOICHIOMETRY.get(ox_clean, 1)
        equiv = x * n_o
        total_eq += equiv
        weighted_sum += equiv * lam
    return float(weighted_sum / total_eq) if total_eq > 0 else 0.50


def calculate_slag_sulfide_capacity(
    oxide_fractions: Dict[str, float],
    temperature_K: float = 1873.15
) -> Dict[str, float]:
    """
    Calculate optical basicity and sulfide capacity C_S (Young / Sommerville model):
      log10(C_S) = -13.913 + 42.84 * Lambda - 23.82 * Lambda^2 - 11710 / T - 0.02223 * (%SiO2)
    where C_S = (%S_slag) * sqrt(P_O2 / P_S2).
    """
    lam = calculate_optical_basicity(oxide_fractions)
    t = float(temperature_K)
    pct_sio2 = oxide_fractions.get("SIO2", 0.0) * 100.0 if max(oxide_fractions.values()) <= 1.0 else oxide_fractions.get("SIO2", 0.0)

    log_cs = -13.913 + 42.84 * lam - 23.82 * (lam ** 2) - (11710.0 / t) - 0.02223 * pct_sio2
    cs = 10.0 ** log_cs

    return {
        "optical_basicity": float(lam),
        "log10_sulfide_capacity": float(log_cs),
        "sulfide_capacity_Cs": float(cs),
        "temperature_K": t,
    }


def calculate_phosphorus_partition_ratio(
    oxide_fractions: Dict[str, float],
    temperature_K: float = 1873.15
) -> float:
    """
    Calculate phosphorus distribution ratio L_P = (%P_slag) / [%P_steel] using Healy's equation:
      log10(L_P) = 22350 / T - 16.0 + 0.08 * (%CaO) + 2.5 * log10(%T.Fe)
    """
    t = float(temperature_K)
    # Normalize to wt%
    total = sum(oxide_fractions.values())
    pct_cao = (oxide_fractions.get("CAO", 0.0) / total) * 100.0 if total > 0 else 40.0
    pct_feo = (oxide_fractions.get("FEO", 0.0) / total) * 100.0 if total > 0 else 15.0
    pct_fe2o3 = (oxide_fractions.get("FE2O3", 0.0) / total) * 100.0 if total > 0 else 0.0
    pct_t_fe = 0.777 * pct_feo + 0.70 * pct_fe2o3
    pct_t_fe = max(pct_t_fe, 1.0)

    log_lp = (22350.0 / t) - 16.0 + 0.08 * pct_cao + 2.5 * np.log10(pct_t_fe)
    return float(10.0 ** log_lp)


def estimate_refractory_corrosion_rate(
    oxide_fractions: Dict[str, float],
    temperature_K: float = 1873.15,
    refractory_type: str = "MGO_C"
) -> Dict[str, Any]:
    """
    Estimate refractory wear and dissolution rate (mm/heat) based on slag basicity and temperature.
    """
    lam = calculate_optical_basicity(oxide_fractions)
    t = float(temperature_K)

    # Slag saturation index for MgO
    pct_mgo = oxide_fractions.get("MGO", 0.0)
    pct_mgo = pct_mgo * 100.0 if pct_mgo <= 1.0 else pct_mgo

    # MgO saturation at 1600 °C is typically ~8-12 wt% depending on basicity
    mgo_sat = 15.0 - 8.0 * (lam - 0.5)
    mgo_sat = max(4.0, min(14.0, mgo_sat))

    delta_mgo = mgo_sat - pct_mgo
    dissolution_driving_force = max(0.0, delta_mgo)

    # Temperature factor
    temp_factor = np.exp((t - 1873.15) / 100.0)
    wear_rate_mm = 0.5 * dissolution_driving_force * temp_factor * (1.0 if lam < 0.65 else 0.4)

    return {
        "refractory_type": refractory_type,
        "optical_basicity": float(lam),
        "MgO_saturation_wt_pct": float(mgo_sat),
        "MgO_deficit_wt_pct": float(dissolution_driving_force),
        "estimated_wear_rate_mm_per_heat": float(wear_rate_mm),
        "wear_severity": "High" if wear_rate_mm > 3.0 else ("Moderate" if wear_rate_mm > 1.0 else "Low"),
    }
