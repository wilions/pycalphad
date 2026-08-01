"""
Precipitation hardening and mechanical strengthening contribution models.
"""

from typing import Dict, List, Optional, Union
import numpy as np

class PrecipitationStrengtheningModel:
    """
    Calculates yield strength contributions from precipitation hardening,
    solid-solution strengthening, and grain boundary Hall-Petch effect.
    """
    def __init__(
        self,
        shear_modulus: float = 80.0e9,      # G (Pa)
        burgers_vector: float = 0.25e-9,     # b (m)
        poisson_ratio: float = 0.3,         # nu
        taylor_factor: float = 3.0,          # M
        hall_petch_coefficient: float = 0.2e6 # k_HP (Pa * m^0.5)
    ):
        self.G = shear_modulus
        self.b = burgers_vector
        self.nu = poisson_ratio
        self.M = taylor_factor
        self.k_HP = hall_petch_coefficient

    def orowan_strengthening(
        self,
        mean_radius: float,
        volume_fraction: float
    ) -> float:
        """
        Calculates Orowan dislocation looping yield strength contribution (Pa).
        
        Formula:
          Delta_sigma_Orowan = (M * 0.4 * G * b) / (pi * sqrt(1 - nu)) * ln(2 * r / b) / lambda_p
        where lambda_p = r * sqrt(2 * pi / (3 * f_v)) is interparticle spacing.
        """
        if mean_radius <= 0 or volume_fraction <= 0:
            return 0.0
        
        r = mean_radius
        f_v = volume_fraction
        
        # Interparticle edge-to-edge distance
        lambda_p = r * np.sqrt((2.0 * np.pi) / (3.0 * f_v))
        if lambda_p <= self.b:
            return 0.0

        prefactor = (self.M * 0.4 * self.G * self.b) / (np.pi * np.sqrt(1.0 - self.nu))
        log_term = np.log((2.0 * r) / self.b)
        
        delta_sigma = prefactor * (log_term / lambda_p)
        return float(max(0.0, delta_sigma))

    def solid_solution_strengthening(
        self,
        solute_concentrations: Dict[str, float],
        strengthening_coefficients: Dict[str, float],
        exponent: float = 1.5
    ) -> float:
        """
        Calculates solid-solution strengthening contribution (Pa):
          Delta_sigma_ss = [ sum_i (k_i^(1/n) * c_i) ]^n
        """
        total = 0.0
        n = exponent
        for el, c in solute_concentrations.items():
            k = strengthening_coefficients.get(el, 0.0)
            if k > 0 and c > 0:
                total += (k ** (1.0 / n)) * c
                
        return float(total ** n)

    def hall_petch_strengthening(self, grain_size: float) -> float:
        """
        Calculates Hall-Petch grain boundary strengthening contribution (Pa):
          Delta_sigma_HP = k_HP * d^(-0.5)
        """
        if grain_size <= 0:
            return 0.0
        return float(self.k_HP * (grain_size ** -0.5))

    def calculate_total_yield_strength(
        self,
        base_yield_strength: float,
        mean_radius: float,
        volume_fraction: float,
        solute_concentrations: Optional[Dict[str, float]] = None,
        strengthening_coefficients: Optional[Dict[str, float]] = None,
        grain_size: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculates total yield strength and breakdown of individual mechanisms (MPa).
        """
        sigma_0 = base_yield_strength
        sigma_orowan = self.orowan_strengthening(mean_radius, volume_fraction)
        
        if solute_concentrations and strengthening_coefficients:
            sigma_ss = self.solid_solution_strengthening(solute_concentrations, strengthening_coefficients)
        else:
            sigma_ss = 0.0
            
        if grain_size:
            sigma_hp = self.hall_petch_strengthening(grain_size)
        else:
            sigma_hp = 0.0

        # Pyramidal / Root-sum-square superposition of hardening mechanisms
        sigma_precip = sigma_orowan
        sigma_total = sigma_0 + np.sqrt(sigma_precip**2 + sigma_ss**2) + sigma_hp
        
        return {
            "total_yield_strength_MPa": float(sigma_total / 1e6),
            "base_strength_MPa": float(sigma_0 / 1e6),
            "orowan_strength_MPa": float(sigma_orowan / 1e6),
            "solid_solution_strength_MPa": float(sigma_ss / 1e6),
            "hall_petch_strength_MPa": float(sigma_hp / 1e6),
        }
