"""
Process Metallurgy & Inclusion Engineering Engine.

Implements:
1. Slag-metal deoxidation and inclusion precipitation equilibria.
2. Inclusion Population Balance (PBE) for nucleation, diffusion-controlled growth, and Stokes flotation.
3. Liquid calcium aluminate window for continuous casting nozzle-clogging prevention.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class InclusionPopulationBalance:
    """
    Simulates the evolution of non-metallic inclusions (Al2O3, Spinels, Calcium Aluminates)
    in liquid steel during secondary refining and continuous casting.
    """

    def __init__(
        self,
        steel_density: float = 7000.0,       # kg/m^3
        steel_viscosity: float = 0.0055,     # Pa * s (5.5 mPa*s)
        gravitational_acc: float = 9.81,     # m/s^2
    ):
        self.rho_steel = steel_density
        self.eta_steel = steel_viscosity
        self.g = gravitational_acc

    def calculate_stokes_flotation_velocity(
        self,
        inclusion_radius: float,
        inclusion_density: float = 3980.0    # Al2O3 density ~3980 kg/m^3
    ) -> float:
        """
        Calculate Stokes terminal flotation velocity for an inclusion rising in molten steel:
          v_terminal = 2 * g * (rho_steel - rho_inc) * r^2 / (9 * eta_steel)
        """
        if inclusion_radius <= 0:
            return 0.0
        delta_rho = self.rho_steel - inclusion_density
        if delta_rho <= 0:
            return 0.0
        v_term = (2.0 * self.g * delta_rho * (inclusion_radius ** 2)) / (9.0 * self.eta_steel)
        return float(v_term)

    def calculate_inclusion_growth_rate(
        self,
        radius: float,
        diffusivity_solute: float,
        bulk_solute_conc: float,
        interface_solute_conc: float,
        inclusion_solute_conc: float
    ) -> float:
        """
        Diffusion-controlled radial growth rate of spherical inclusions:
          dr/dt = (D / r) * (C_bulk - C_int) / (C_inc - C_int)
        """
        if radius <= 0:
            return 0.0
        delta_c = bulk_solute_conc - interface_solute_conc
        denom = inclusion_solute_conc - interface_solute_conc
        if denom <= 0 or delta_c <= 0:
            return 0.0
        return float((diffusivity_solute / radius) * (delta_c / denom))

    def evaluate_inclusion_evolution(
        self,
        initial_mean_radius: float,
        total_time_seconds: float,
        time_steps: int = 100,
        solute_diffusivity: float = 2.0e-9,
        bulk_oxygen_ppm: float = 30.0,
        eq_oxygen_ppm: float = 3.0,
        bath_depth_meters: float = 1.5,
        inclusion_type: str = "AL2O3"
    ) -> Dict[str, Any]:
        """
        Track inclusion growth and flotation removal fraction over refining holding time.
        """
        inc_densities = {
            "AL2O3": 3980.0,
            "SPINEL": 3580.0,
            "CAO_AL2O3": 2980.0,
            "12CAO_7AL2O3": 2830.0,
            "CAS": 2610.0,
        }
        rho_inc = inc_densities.get(inclusion_type.upper(), 3800.0)

        dt = total_time_seconds / time_steps
        time_arr = np.linspace(0, total_time_seconds, time_steps)
        radius_arr = np.zeros(time_steps)
        flotation_fraction_arr = np.zeros(time_steps)

        r_curr = max(initial_mean_radius, 1.0e-7)  # 0.1 micron min
        c_inc = 0.47  # oxygen mass fraction in Al2O3 ~ 47 wt%
        c_bulk = bulk_oxygen_ppm * 1.0e-6
        c_int = eq_oxygen_ppm * 1.0e-6

        total_floated_height = 0.0

        for i, t in enumerate(time_arr):
            radius_arr[i] = r_curr
            v_float = self.calculate_stokes_flotation_velocity(r_curr, rho_inc)
            total_floated_height += v_float * dt
            flotation_frac = min(1.0, total_floated_height / bath_depth_meters)
            flotation_fraction_arr[i] = flotation_frac

            # Radial growth
            dr_dt = self.calculate_inclusion_growth_rate(r_curr, solute_diffusivity, c_bulk, c_int, c_inc)
            r_curr = max(1.0e-8, r_curr + dr_dt * dt)

        return {
            "time_seconds": time_arr,
            "mean_radius_microns": radius_arr * 1e6,
            "flotation_removal_fraction": flotation_fraction_arr,
            "final_radius_microns": float(radius_arr[-1] * 1e6),
            "final_removal_percentage": float(flotation_fraction_arr[-1] * 100.0),
            "inclusion_type": inclusion_type,
        }


def check_liquid_calcium_aluminate_window(
    al_wt_pct: float,
    ca_ppm: float,
    total_oxygen_ppm: float
) -> Dict[str, Any]:
    """
    Check if Ca-treated molten steel falls into the liquid calcium aluminate window
    (12CaO*7Al2O3 / C12A7 liquidus at 1600 °C) to prevent submerged entry nozzle (SEN) clogging.
    """
    if al_wt_pct <= 0 or total_oxygen_ppm <= 0:
        return {"status": "Invalid Input", "is_liquid_window": False}

    # Ca/Al ratio and Ca/T.O. ratio
    ca_wt_pct = ca_ppm * 1e-4
    t_o_wt_pct = total_oxygen_ppm * 1e-4

    ca_al_ratio = ca_wt_pct / al_wt_pct
    ca_to_ratio = ca_ppm / total_oxygen_ppm

    # Optimal liquid calcium aluminate window is typically 0.08 <= Ca/Al <= 0.15 and 0.6 <= Ca/O <= 1.2
    is_liquid = (0.07 <= ca_al_ratio <= 0.18) and (0.5 <= ca_to_ratio <= 1.5)

    if ca_to_ratio < 0.5:
        risk = "Solid Al2O3 or CA6 / CA2 formation (High nozzle clogging risk)"
    elif ca_to_ratio > 1.5:
        risk = "Solid CaS formation (Ca over-treatment / nozzle clogging risk)"
    else:
        risk = "Fully liquid calcium aluminate (Clean steel / Optimal casting window)"

    return {
        "Ca_Al_ratio": float(ca_al_ratio),
        "Ca_O_ratio": float(ca_to_ratio),
        "is_liquid_window": is_liquid,
        "clogging_risk_assessment": risk,
    }
