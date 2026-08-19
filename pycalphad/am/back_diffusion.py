"""
Solidification with Solute Back-Diffusion (Brody-Flemings & Clyne-Kurz models).
Computes cooling-rate dependent solute microsegregation, modified partition coefficients, and solidification profiles.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Sequence
import numpy as np


@dataclass
class BackDiffusionResult:
    """
    Result of a solidification simulation accounting for finite solid back-diffusion.
    """
    temperatures_K: np.ndarray
    fraction_solid: np.ndarray
    liquidus_K: float
    solidus_K: float
    alpha: float
    alpha_prime: float
    effective_partition_coeff: float
    equilibrium_partition_coeff: float
    solute_profile: np.ndarray
    kou_cracking_index: float
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def freezing_range_K(self) -> float:
        return float(self.liquidus_K - self.solidus_K)


def calculate_clyne_kurz_alpha(
    D_s: float,
    cooling_rate: float,
    dendrite_arm_spacing_m: float = 10e-6,
    freezing_range_K: float = 100.0,
) -> Tuple[float, float]:
    """
    Computes Brody-Flemings alpha and Clyne-Kurz modified alpha_prime.

    Parameters
    ----------
    D_s : float
        Solid diffusion coefficient (m^2/s).
    cooling_rate : float
        Solidification cooling rate (K/s), e.g. 1e3 to 1e6 for AM.
    dendrite_arm_spacing_m : float
        Secondary dendrite arm spacing (SDAS) in meters (e.g. 5-20 um for LPBF).
    freezing_range_K : float
        Equilibrium freezing range (T_liq - T_sol) in Kelvin.

    Returns
    -------
    Tuple[float, float]
        (alpha, alpha_prime)
    """
    if cooling_rate <= 0:
        raise ValueError("Cooling rate must be strictly positive.")
    if dendrite_arm_spacing_m <= 0:
        raise ValueError("Dendrite arm spacing must be positive.")

    # Solidification local time t_f = Delta T_f / (dT/dt)
    t_f = max(freezing_range_K / cooling_rate, 1e-8)
    L = 0.5 * dendrite_arm_spacing_m

    # Dimensionless diffusion Fourier number alpha
    alpha = (D_s * t_f) / (L ** 2)

    # Clyne-Kurz continuous correction function for 0 <= alpha <= inf
    if alpha < 1e-6:
        alpha_prime = 0.0  # Pure Scheil limit
    elif alpha > 10.0:
        alpha_prime = 0.5  # Complete equilibrium lever-rule limit
    else:
        alpha_prime = alpha * (1.0 - np.exp(-1.0 / alpha)) - 0.5 * np.exp(-0.5 / alpha)
        alpha_prime = float(np.clip(alpha_prime, 0.0, 0.5))

    return float(alpha), float(alpha_prime)


def effective_partition_coefficient(k_eq: float, alpha_prime: float) -> float:
    """
    Computes the effective partition coefficient k' modified by back diffusion.
    For alpha_prime = 0 (Scheil), k' = k_eq.
    For alpha_prime = 0.5 (Lever rule), k' -> 1.0 (equilibrium homogenization).
    """
    if alpha_prime <= 0:
        return float(k_eq)
    denom = 1.0 + 2.0 * alpha_prime * (k_eq - 1.0)
    if abs(denom) < 1e-6:
        return float(k_eq)
    # Clyne-Kurz formulation
    k_eff = k_eq * (1.0 + 2.0 * alpha_prime) / denom
    return float(np.clip(k_eff, 1e-4, 1.0 if k_eq < 1.0 else 2.0))


def simulate_back_diffusion_solidification(
    liquidus_K: float,
    equilibrium_solidus_K: float,
    k_eq: float = 0.3,
    D_s: float = 1e-13,
    cooling_rate: float = 1e4,
    dendrite_arm_spacing_m: float = 10e-6,
    num_points: int = 100,
) -> BackDiffusionResult:
    """
    Simulates solidification temperature vs. solid fraction accounting for back diffusion.
    """
    delta_T_eq = max(liquidus_K - equilibrium_solidus_K, 1.0)
    alpha, alpha_prime = calculate_clyne_kurz_alpha(
        D_s=D_s,
        cooling_rate=cooling_rate,
        dendrite_arm_spacing_m=dendrite_arm_spacing_m,
        freezing_range_K=delta_T_eq,
    )

    k_eff = effective_partition_coefficient(k_eq, alpha_prime)

    # Discretize fraction solid from 0 to 0.999
    f_s = np.linspace(0.0, 0.995, num_points)

    # Temperature profile using modified Scheil:
    # T(f_s) = T_liq - delta_T_eq * (1 - (1 - 2*alpha_prime*k_eff)*f_s)**((1-k_eff)/(1 - 2*alpha_prime*k_eff))
    exponent_denom = 1.0 - 2.0 * alpha_prime * k_eff
    if abs(exponent_denom) < 1e-4:
        exponent = (1.0 - k_eff)
        t_profile = liquidus_K - delta_T_eq * (1.0 - f_s) ** (-exponent)
    else:
        exponent = (1.0 - k_eff) / exponent_denom
        t_profile = liquidus_K - delta_T_eq * (1.0 - (1.0 - 2.0 * alpha_prime * k_eff) * f_s) ** (-exponent)

    # Ensure temperature stays bounded between liquidus and terminal eutectic
    t_profile = np.clip(t_profile, liquidus_K - 3.0 * delta_T_eq, liquidus_K)

    # Calculate Kou hot-cracking index max |dT / d(sqrt(fs))| in vulnerable regime 0.90 <= fs <= 0.99
    vulnerable_mask = (f_s >= 0.85) & (f_s <= 0.99)
    if np.sum(vulnerable_mask) >= 3:
        sqrt_fs = np.sqrt(f_s[vulnerable_mask])
        t_sub = t_profile[vulnerable_mask]
        grad = np.abs(np.gradient(t_sub, sqrt_fs))
        kou_index = float(np.max(grad))
    else:
        kou_index = 0.0

    # Solute concentration profile C_s(f_s)
    solute_profile = k_eff * (1.0 - (1.0 - 2.0 * alpha_prime * k_eff) * f_s) ** (-exponent)

    return BackDiffusionResult(
        temperatures_K=t_profile,
        fraction_solid=f_s,
        liquidus_K=liquidus_K,
        solidus_K=float(t_profile[-1]),
        alpha=alpha,
        alpha_prime=alpha_prime,
        effective_partition_coeff=k_eff,
        equilibrium_partition_coeff=k_eq,
        solute_profile=solute_profile,
        kou_cracking_index=kou_index,
        details={
            "cooling_rate_K_s": cooling_rate,
            "dendrite_arm_spacing_um": dendrite_arm_spacing_m * 1e6,
            "D_s_m2_s": D_s,
        }
    )
