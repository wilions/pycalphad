"""
High-Temperature Superalloy Metamodels & Physics-Informed Indicators.

Implements:
1. Ordered L12 gamma-prime Antiphase Boundary (APB) energy metamodel on {111} planes.
2. Order-strengthening yield stress increment Delta tau_APB.
3. Topologically Close-Packed (TCP) phase precipitation risk index (Xi_TCP) for sigma, mu, and Laves phases.
4. Larson-Miller high-temperature creep rupture estimation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def calculate_gamma_prime_apb_energy(
    bulk_mole_fractions: Dict[str, float],
    temperature_K: float = 1000.0,
    lattice_parameter_nm: float = 0.358
) -> Dict[str, float]:
    """
    Calculate the Antiphase Boundary (APB) energy on {111} slip planes of ordered L12 (Ni3Al-type) phase:
      gamma_APB = (2 / (sqrt(3) * a^2)) * [ Delta E_ord + sum_ij omega_ij (y_i y_j - x_i x_j) ]
    """
    # Key elemental contributions to APB energy in Ni-base superalloys (mJ/m^2 per at%)
    # Ti, Ta, Nb strongly increase APB energy; Cr, Mo have moderate effects; Fe decreases
    apb_coefficients = {
        "AL": 160.0,   # Base Ni3Al ~ 160-180 mJ/m^2
        "TI": 280.0,   # Strong APB enhancer
        "TA": 320.0,   # Very strong APB enhancer
        "NB": 300.0,   # Very strong APB enhancer
        "W": 220.0,
        "MO": 200.0,
        "CR": 170.0,
        "CO": 150.0,
        "FE": 120.0,
        "RE": 260.0,
    }

    # Normalize compositions in gamma-prime phase (Al-site substituters)
    al_site_elements = ["AL", "TI", "TA", "NB", "V"]
    al_sum = sum(bulk_mole_fractions.get(el, 0.0) for el in al_site_elements)
    if al_sum <= 0:
        al_sum = 0.15

    weighted_apb = 0.0
    for el in al_site_elements:
        frac = bulk_mole_fractions.get(el, 0.0) / al_sum
        weighted_apb += frac * apb_coefficients.get(el, 180.0)

    # Add matrix solid-solution modifying terms
    re_wt = bulk_mole_fractions.get("RE", 0.0)
    w_wt = bulk_mole_fractions.get("W", 0.0)
    gamma_apb = weighted_apb + 50.0 * re_wt + 30.0 * w_wt

    # Temperature softening of APB energy
    t_ref = 298.15
    dT = max(0.0, temperature_K - t_ref)
    gamma_apb_t = max(80.0, gamma_apb - 0.035 * dT)

    return {
        "APB_energy_mJ_m2": float(gamma_apb_t),
        "temperature_K": float(temperature_K),
        "lattice_parameter_nm": float(lattice_parameter_nm),
    }


def calculate_order_strengthening_increment(
    apb_energy_mJ_m2: float,
    precipitate_radius_nm: float,
    volume_fraction: float,
    shear_modulus_GPa: float = 75.0,
    burgers_vector_nm: float = 0.254
) -> Dict[str, float]:
    """
    Calculate order-strengthening critical resolved shear stress (CRSS) increment
    for weakly-coupled or strongly-coupled dislocation pairs cutting through L12 precipitates:
      Delta tau_weak = 0.5 * (gamma_APB / b) * [ sqrt( (4 * gamma_APB * r * f) / (pi * T_line) ) - f ]
    """
    gamma = apb_energy_mJ_m2 * 1e-3  # J/m^2
    r = precipitate_radius_nm * 1e-9 # m
    f = max(0.01, min(0.85, volume_fraction))
    g = shear_modulus_GPa * 1e9      # Pa
    b = burgers_vector_nm * 1e-9     # m
    t_line = 0.5 * g * (b ** 2)      # Dislocation line tension (J/m)

    # Weak pair cutting regime
    prefactor = gamma / (2.0 * b)
    rad_term = (4.0 * gamma * r * f) / (np.pi * t_line)
    if rad_term > 0:
        delta_tau = prefactor * (np.sqrt(rad_term) - f)
    else:
        delta_tau = 0.0

    delta_tau_mpa = max(0.0, delta_tau / 1e6)
    delta_sigma_mpa = 3.0 * delta_tau_mpa  # Taylor factor ~ 3.0

    return {
        "CRSS_increment_MPa": float(delta_tau_mpa),
        "yield_strength_increment_MPa": float(delta_sigma_mpa),
        "cutting_regime": "Weak Pair Shearing" if r < 20e-9 else "Strong Pair / Orowan Transition",
    }


def calculate_tcp_phase_risk_index(
    matrix_composition: Dict[str, float]
) -> Dict[str, Any]:
    """
    Calculate the Topologically Close-Packed (TCP) phase embrittlement risk index (Xi_TCP).
    Evaluates Md (d-orbital energy level) and electron vacancy (Nv) equivalents.
    """
    # Average d-orbital energy levels Md (Morinaga New PHACOMP)
    MD_LEVELS = {
        "NI": 0.717, "CO": 0.777, "FE": 0.994, "CR": 1.142,
        "MO": 1.550, "W": 1.655, "TA": 2.224, "NB": 2.117,
        "TI": 2.271, "AL": 1.900, "RE": 1.262, "RU": 1.054,
    }

    total_moles = sum(matrix_composition.values())
    if total_moles <= 0:
        return {"Md_avg": 0.0, "TCP_risk": "None"}

    md_avg = 0.0
    for el, x in matrix_composition.items():
        x_norm = x / total_moles
        md_avg += x_norm * MD_LEVELS.get(el.upper(), 1.0)

    # Critical Md threshold for TCP precipitation in Ni superalloys is ~0.985 - 0.995
    is_safe = md_avg < 0.985
    risk_level = "Low (Safe)" if md_avg < 0.980 else ("Moderate" if md_avg < 0.992 else "High (TCP Embrittlement Danger)")

    return {
        "Md_average": float(md_avg),
        "critical_Md_threshold": 0.985,
        "is_microstructurally_stable": is_safe,
        "risk_category": risk_level,
    }


def estimate_larson_miller_creep_life(
    stress_MPa: float,
    temperature_celsius: float,
    c_constant: float = 20.0
) -> Dict[str, float]:
    """
    Estimate creep rupture life using Larson-Miller parameter (LMP):
      LMP = (T_K / 1000) * (log10(t_rupture_hrs) + C)
    """
    t_k = temperature_celsius + 273.15
    # Approximate master curve for typical high-strength Ni superalloys: LMP ~ 32 - 4.5 * log10(stress)
    sigma = max(10.0, stress_MPa)
    lmp = 32.0 - 4.5 * np.log10(sigma)
    log_t = (lmp * 1000.0 / t_k) - c_constant
    t_rupture_hrs = float(10.0 ** np.clip(log_t, -2.0, 7.0))

    return {
        "applied_stress_MPa": float(sigma),
        "temperature_celsius": float(temperature_celsius),
        "Larson_Miller_Parameter": float(lmp),
        "estimated_creep_life_hours": float(t_rupture_hrs),
    }
