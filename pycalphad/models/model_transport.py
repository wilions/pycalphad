"""
Transport and Interface Property Models: Liquid Surface Tension and Dynamic Viscosity.

Implements:
1. Butler Equation Non-Linear Solver for liquid surface tension and surface composition.
2. Eyring-Kaptay Absolute Rate Theory for dynamic viscosity of multicomponent liquid alloys.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import root, minimize
from symengine import S, exp, log, sympify
from tinydb import where

from pycalphad import variables as v
from pycalphad.variables import Species

# Pure component reference surface tension data: (sigma_0 at T_melt in N/m, T_melt in K, dsigma/dT in N/m/K, molar_surface_area A_i in m^2/mol)
PURE_SURFACE_TENSION_DEFAULTS = {
    "AL": {"sigma_m": 0.865, "T_m": 933.47, "dsigma_dT": -0.15e-3, "molar_mass": 26.98e-3, "density": 2375.0},
    "NI": {"sigma_m": 1.770, "T_m": 1728.0, "dsigma_dT": -0.38e-3, "molar_mass": 58.69e-3, "density": 7810.0},
    "FE": {"sigma_m": 1.720, "T_m": 1811.0, "dsigma_dT": -0.43e-3, "molar_mass": 55.85e-3, "density": 7020.0},
    "CR": {"sigma_m": 1.700, "T_m": 2180.0, "dsigma_dT": -0.32e-3, "molar_mass": 52.00e-3, "density": 6300.0},
    "CO": {"sigma_m": 1.880, "T_m": 1768.0, "dsigma_dT": -0.48e-3, "molar_mass": 58.93e-3, "density": 7750.0},
    "TI": {"sigma_m": 1.650, "T_m": 1941.0, "dsigma_dT": -0.26e-3, "molar_mass": 47.87e-3, "density": 4110.0},
    "CU": {"sigma_m": 1.300, "T_m": 1357.7, "dsigma_dT": -0.23e-3, "molar_mass": 63.55e-3, "density": 8020.0},
    "SN": {"sigma_m": 0.544, "T_m": 505.08, "dsigma_dT": -0.07e-3, "molar_mass": 118.7e-3, "density": 6990.0},
    "AG": {"sigma_m": 0.925, "T_m": 1234.9, "dsigma_dT": -0.19e-3, "molar_mass": 107.87e-3, "density": 9320.0},
    "ZN": {"sigma_m": 0.782, "T_m": 692.68, "dsigma_dT": -0.17e-3, "molar_mass": 65.38e-3, "density": 6570.0},
    "MG": {"sigma_m": 0.559, "T_m": 923.00, "dsigma_dT": -0.35e-3, "molar_mass": 24.31e-3, "density": 1590.0},
}

R_GAS = 8.314462618  # J / (mol * K)
N_AVOGADRO = 6.02214076e23
PLANCKS_H = 6.62607015e-34  # J * s


def get_pure_liquid_surface_tension(element: str, temperature: float) -> Tuple[float, float]:
    """
    Get surface tension sigma_0 (N/m) and partial molar surface area A_0 (m^2/mol) for pure liquid element.
    """
    el = element.upper()
    data = PURE_SURFACE_TENSION_DEFAULTS.get(
        el,
        {"sigma_m": 1.20, "T_m": 1500.0, "dsigma_dT": -0.25e-3, "molar_mass": 50.0e-3, "density": 6000.0}
    )
    sigma_0 = data["sigma_m"] + data["dsigma_dT"] * (temperature - data["T_m"])
    # Partial molar surface area: A_0 = 1.091 * N_A^(1/3) * (M / rho)^(2/3)
    V_molar = data["molar_mass"] / data["density"]
    A_0 = 1.091 * (N_AVOGADRO ** (1.0 / 3.0)) * (V_molar ** (2.0 / 3.0))
    return max(float(sigma_0), 0.05), float(A_0)


def solve_butler_surface_tension(
    bulk_mole_fractions: Dict[str, float],
    temperature: float,
    excess_gibbs_interactions: Optional[Dict[Tuple[str, str], float]] = None,
    surface_coordination_ratio: float = 0.75
) -> Dict[str, Any]:
    r"""
    Solve the Butler equation for multicomponent liquid alloy surface tension.

    Butler Equation:
      sigma = sigma_i^\circ + (R*T / A_i) * ln( (x_i^s * gamma_i^s) / (x_i^b * gamma_i^b) )

    Parameters
    ----------
    bulk_mole_fractions : dict
        Dict mapping element names to bulk mole fractions (x_i^b).
    temperature : float
        Temperature in Kelvin.
    excess_gibbs_interactions : dict, optional
        Binary interaction parameters Omega_ij (J/mol) for regular solution model.
    surface_coordination_ratio : float
        Ratio of surface to bulk coordination number (typically 0.75 for close-packed liquid).

    Returns
    -------
    dict
        {
            'surface_tension': float (N/m),
            'surface_fractions': dict of {element: x_i^s},
            'pure_surface_tensions': dict of {element: sigma_i^\circ},
            'temperature': float
        }
    """
    elements = sorted([k.upper() for k, v in bulk_mole_fractions.items() if v > 1e-6])
    n_elem = len(elements)
    if n_elem == 0:
        return {"surface_tension": 0.0, "surface_fractions": {}, "temperature": temperature}

    # Normalize bulk fractions
    raw_x_b = np.array([bulk_mole_fractions.get(el, 0.0) for el in elements], dtype=float)
    x_b = raw_x_b / np.sum(raw_x_b)

    if n_elem == 1:
        sigma_pure, _ = get_pure_liquid_surface_tension(elements[0], temperature)
        return {
            "surface_tension": float(sigma_pure),
            "surface_fractions": {elements[0]: 1.0},
            "pure_surface_tensions": {elements[0]: float(sigma_pure)},
            "temperature": temperature
        }

    # Retrieve pure component properties
    sigmas_0 = []
    areas_0 = []
    for el in elements:
        s0, a0 = get_pure_liquid_surface_tension(el, temperature)
        sigmas_0.append(s0)
        areas_0.append(a0)
    sigmas_0 = np.array(sigmas_0)
    areas_0 = np.array(areas_0)

    omega_dict = excess_gibbs_interactions or {}

    def get_ln_gamma(x_vec, is_surface=False):
        """Compute ln(gamma_i) using multi-component regular solution model."""
        ln_gamma = np.zeros(n_elem)
        factor = surface_coordination_ratio if is_surface else 1.0
        for i in range(n_elem):
            g_ex_i = 0.0
            for j in range(n_elem):
                if i != j:
                    el_pair = tuple(sorted([elements[i], elements[j]]))
                    omega = omega_dict.get(el_pair, 0.0) * factor
                    g_ex_i += omega * x_vec[j] ** 2
            ln_gamma[i] = g_ex_i / (R_GAS * temperature)
        return ln_gamma

    ln_gamma_b = get_ln_gamma(x_b, is_surface=False)

    # Initial guess: x_s is equal to x_b, sigma is average
    # System of equations: for i in 1..n-1: sigma_i(x_s) - sigma_0(x_s) = 0, and sum(x_s) - 1 = 0
    def residual(vars_vec):
        # vars_vec has length n_elem: [x_s[0], x_s[1], ..., x_s[n-2], sigma]
        x_s_raw = np.zeros(n_elem)
        x_s_raw[:-1] = vars_vec[:-1]
        x_s_raw[-1] = 1.0 - np.sum(x_s_raw[:-1])

        # Penalize non-physical bounds
        x_s = np.clip(x_s_raw, 1e-9, 1.0 - 1e-9)
        x_s = x_s / np.sum(x_s)
        sigma_val = vars_vec[-1]

        ln_gamma_s = get_ln_gamma(x_s, is_surface=True)

        res = np.zeros(n_elem)
        # Butler equation for each component:
        # sigma_val - (sigma_0[i] + (R*T / A_0[i]) * ln((x_s[i] * gamma_s[i]) / (x_b[i] * gamma_b[i]))) = 0
        for i in range(n_elem):
            target_sigma_i = sigmas_0[i] + (R_GAS * temperature / areas_0[i]) * (
                np.log(x_s[i] / x_b[i]) + ln_gamma_s[i] - ln_gamma_b[i]
            )
            res[i] = sigma_val - target_sigma_i
        return res

    init_guess = np.zeros(n_elem)
    init_guess[:-1] = x_b[:-1]
    init_guess[-1] = np.sum(x_b * sigmas_0)

    sol = root(residual, init_guess, method="hybr")
    if sol.success:
        res_x_s = np.zeros(n_elem)
        res_x_s[:-1] = sol.x[:-1]
        res_x_s[-1] = 1.0 - np.sum(res_x_s[:-1])
        res_x_s = np.clip(res_x_s, 0.0, 1.0)
        res_x_s = res_x_s / np.sum(res_x_s)
        final_sigma = float(sol.x[-1])
    else:
        # Fallback to ideal surface tension model
        res_x_s = x_b.copy()
        final_sigma = float(np.sum(x_b * sigmas_0))

    surface_fractions = {elements[i]: float(res_x_s[i]) for i in range(n_elem)}
    pure_dict = {elements[i]: float(sigmas_0[i]) for i in range(n_elem)}

    return {
        "surface_tension": max(final_sigma, 0.01),
        "surface_fractions": surface_fractions,
        "pure_surface_tensions": pure_dict,
        "temperature": temperature
    }


def calculate_liquid_dynamic_viscosity(
    bulk_mole_fractions: Dict[str, float],
    temperature: float,
    molar_volume: Optional[float] = None,
    activation_energy_excess: float = 0.0
) -> Dict[str, Any]:
    """
    Calculate the dynamic viscosity of liquid alloy using the Eyring-Kaptay model.

    eta = (h * N_A / V_m) * exp( Delta G^* / (R * T) )

    Parameters
    ----------
    bulk_mole_fractions : dict
        Mole fractions of elements in liquid phase.
    temperature : float
        Temperature in Kelvin.
    molar_volume : float, optional
        Molar volume of the liquid phase in m^3/mol. If None, estimated from elements.
    activation_energy_excess : float
        Excess activation Gibbs energy for viscous flow Delta G_ex^* (J/mol).

    Returns
    -------
    dict
        {
            'dynamic_viscosity_Pa_s': float,
            'dynamic_viscosity_mPa_s': float,
            'temperature': float,
            'molar_volume_m3_mol': float
        }
    """
    elements = [k.upper() for k, v in bulk_mole_fractions.items() if v > 1e-6]
    fractions = np.array([bulk_mole_fractions[el] for el in elements], dtype=float)
    fractions = fractions / np.sum(fractions)

    # Reference activation energy for viscous flow of pure metals ~ 3.5 * R * T_m
    delta_g_ref = 0.0
    v_m_est = 0.0
    for el, x_i in zip(elements, fractions):
        data = PURE_SURFACE_TENSION_DEFAULTS.get(
            el,
            {"T_m": 1500.0, "molar_mass": 50.0e-3, "density": 6000.0}
        )
        t_m = data["T_m"]
        delta_g_i = 3.55 * R_GAS * t_m
        delta_g_ref += x_i * delta_g_i
        v_m_est += x_i * (data["molar_mass"] / data["density"])

    v_m = molar_volume if (molar_volume is not None and molar_volume > 1e-7) else v_m_est
    delta_g_total = delta_g_ref + activation_energy_excess

    # Eyring absolute rate equation:
    pre_factor = (PLANCKS_H * N_AVOGADRO) / v_m
    exponent = delta_g_total / (R_GAS * temperature)
    viscosity = pre_factor * np.exp(exponent)

    return {
        "dynamic_viscosity_Pa_s": float(viscosity),
        "dynamic_viscosity_mPa_s": float(viscosity * 1000.0),
        "temperature": temperature,
        "molar_volume_m3_mol": float(v_m),
    }
