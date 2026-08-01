"""
Microsegregation and solid-state back-diffusion models for solidification.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from pycalphad.am.schemas import SimulationResult

def simulate_microsegregation(
    solute_concentration_initial: float,
    partition_coefficient: float,
    diffusivity_solid: float,
    solidification_time: float,
    dendrite_arm_spacing: float,
    num_steps: int = 100,
    model: str = "clyne_kurz"
) -> SimulationResult:
    """
    Simulates solute microsegregation during solidification considering solid-state back-diffusion.

    Models:
      - 'scheil': Zero back-diffusion (alpha = 0)
      - 'lever': Equilibrium back-diffusion (alpha -> infinity)
      - 'clyne_kurz': Clyne-Kurz back-diffusion model with parameter alpha = 4 * D_s * t_f / L^2
      - 'brody_flemings': Brody-Flemings back-diffusion model

    Parameters:
      solute_concentration_initial: C_0 (wt% or mole fraction)
      partition_coefficient: k = C_s / C_l
      diffusivity_solid: D_s (m^2/s)
      solidification_time: t_f (seconds)
      dendrite_arm_spacing: L (meters, e.g. secondary dendrite arm spacing lambda_2)
      num_steps: Number of fraction-solid increments

    Returns:
      SimulationResult containing solid fraction f_s, solid solute C_s(f_s), and liquid solute C_l(f_s).
    """
    f_s = np.linspace(0, 1 - 1e-4, num_steps)
    k = partition_coefficient
    C_0 = solute_concentration_initial
    
    # Fourier back-diffusion parameter
    if dendrite_arm_spacing > 0:
        alpha = (4.0 * diffusivity_solid * solidification_time) / (dendrite_arm_spacing**2)
    else:
        alpha = 0.0

    if model == "scheil" or alpha == 0:
        alpha_eff = 0.0
    elif model == "lever":
        alpha_eff = float('inf')
    elif model == "clyne_kurz":
        # Clyne-Kurz modification: alpha_eff = alpha * (1 - exp(-1 / alpha))
        if alpha > 0:
            alpha_eff = alpha * (1.0 - np.exp(-1.0 / alpha))
        else:
            alpha_eff = 0.0
    elif model == "brody_flemings":
        alpha_eff = alpha
    else:
        raise ValueError(f"Unknown microsegregation model: {model}")

    if np.isinf(alpha_eff):
        # Lever rule limit
        C_l = C_0 / (1.0 + (k - 1.0) * f_s)
        C_s = k * C_l
    else:
        # Modified Scheil equation with back-diffusion parameter alpha_eff
        exponent = (k - 1.0) / (1.0 + 2.0 * alpha_eff * k)
        C_l = C_0 * (1.0 - (1.0 - 2.0 * alpha_eff * k) * f_s)**exponent
        C_s = k * C_l

    grid = {"f_s": f_s}
    field_data = {
        "C_liquid": C_l,
        "C_solid": C_s,
    }
    scalar_outputs = {
        "alpha": float(alpha),
        "alpha_eff": float(alpha_eff) if not np.isinf(alpha_eff) else 1e6,
        "C_l_final": float(C_l[-1]),
        "C_s_final": float(C_s[-1]),
    }
    
    return SimulationResult(
        simulation_type="microsegregation_back_diffusion",
        grid=grid,
        field_data=field_data,
        scalar_outputs=scalar_outputs,
        metadata={
            "model": model,
            "partition_coefficient": k,
            "C0": C_0,
            "diffusivity_solid": diffusivity_solid,
            "solidification_time": solidification_time,
            "dendrite_arm_spacing": dendrite_arm_spacing,
        }
    )
