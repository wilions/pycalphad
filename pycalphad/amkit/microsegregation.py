"""
Microsegregation, rapid solute trapping, and solid-state back-diffusion models for solidification.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pycalphad.am.schemas import SimulationResult


def calculate_trapped_partition_coefficient(
    k_equilibrium: float,
    interface_velocity: float,
    diffusive_velocity: float = 5.0
) -> float:
    """
    Calculate the velocity-dependent partition coefficient using the Aziz Continuous Growth Model (CGM).

    k(v) = (k_0 + v / v_D) / (1 + v / v_D)

    Parameters
    ----------
    k_equilibrium : float
        Equilibrium partition coefficient k_0 (C_s / C_l at equilibrium).
    interface_velocity : float
        Solid-liquid interface growth velocity v (m/s).
    diffusive_velocity : float, optional
        Characteristic solute diffusive speed across the interface v_D (typically 1-10 m/s for metallic systems).

    Returns
    -------
    float
        Non-equilibrium partition coefficient k(v) in range [k_0, 1.0].
    """
    v = max(float(interface_velocity), 0.0)
    v_d = max(float(diffusive_velocity), 1e-6)
    k_0 = float(k_equilibrium)

    k_v = (k_0 + (v / v_d)) / (1.0 + (v / v_d))
    return float(np.clip(k_v, min(k_0, 1.0), 1.0))


def simulate_microsegregation(
    solute_concentration_initial: float,
    partition_coefficient: float,
    diffusivity_solid: float,
    solidification_time: float,
    dendrite_arm_spacing: float,
    interface_velocity: float = 0.0,
    diffusive_velocity: float = 5.0,
    num_steps: int = 100,
    model: str = "ohnaka"
) -> SimulationResult:
    """
    Simulates solute microsegregation during solidification considering rapid solute trapping
    and solid-state back-diffusion.

    Models:
      - 'scheil': Zero back-diffusion (alpha = 0)
      - 'lever': Equilibrium back-diffusion (alpha -> infinity)
      - 'clyne_kurz': Clyne-Kurz back-diffusion model
      - 'ohnaka': Ohnaka parabolic back-diffusion model
      - 'brody_flemings': Brody-Flemings back-diffusion model

    Parameters
    ----------
      solute_concentration_initial: C_0 (wt% or mole fraction)
      partition_coefficient: k_0 = C_s / C_l at equilibrium
      diffusivity_solid: D_s (m^2/s)
      solidification_time: t_f (seconds)
      dendrite_arm_spacing: L (meters, e.g. secondary dendrite arm spacing lambda_2)
      interface_velocity: v (m/s) for Aziz solute trapping
      diffusive_velocity: v_D (m/s)
      num_steps: Number of fraction-solid increments
      model: Back-diffusion formulation name

    Returns
    -------
      SimulationResult containing solid fraction f_s, solid solute C_s(f_s), and liquid solute C_l(f_s).
    """
    f_s = np.linspace(0, 1 - 1e-4, num_steps)
    C_0 = float(solute_concentration_initial)
    k_0 = float(partition_coefficient)

    # Compute velocity-dependent trapped partition coefficient
    if interface_velocity > 0:
        k = calculate_trapped_partition_coefficient(k_0, interface_velocity, diffusive_velocity)
    else:
        k = k_0

    # Fourier back-diffusion parameter: alpha = 4 * D_s * t_f / L^2
    if dendrite_arm_spacing > 0 and diffusivity_solid > 0:
        alpha = (4.0 * diffusivity_solid * solidification_time) / (dendrite_arm_spacing ** 2)
    else:
        alpha = 0.0

    if model == "scheil" or alpha == 0:
        alpha_eff = 0.0
    elif model == "lever":
        alpha_eff = float("inf")
    elif model == "clyne_kurz":
        # Clyne-Kurz: alpha_eff = alpha * (1 - exp(-1 / alpha))
        alpha_eff = alpha * (1.0 - np.exp(-1.0 / alpha)) if alpha > 0 else 0.0
    elif model == "ohnaka":
        # Ohnaka: alpha_eff = alpha / (1.0 + 2.0 * alpha)
        alpha_eff = alpha / (1.0 + 2.0 * alpha) if alpha > 0 else 0.0
    elif model == "brody_flemings":
        alpha_eff = alpha
    else:
        raise ValueError(f"Unknown microsegregation model: {model}")

    if np.isinf(alpha_eff):
        # Lever rule limit
        C_l = C_0 / (1.0 + (k - 1.0) * f_s)
        C_s = k * C_l
    else:
        # Modified Scheil equation with effective back-diffusion parameter alpha_eff
        denom = 1.0 - 2.0 * alpha_eff * k
        if abs(denom) < 1e-6:
            denom = 1e-6
        exponent = (k - 1.0) / (1.0 + 2.0 * alpha_eff * k)
        C_l = C_0 * np.maximum(1.0 - (1.0 - 2.0 * alpha_eff * k) * f_s, 1e-6) ** exponent
        C_s = k * C_l

    grid = {"f_s": f_s}
    field_data = {
        "C_liquid": C_l,
        "C_solid": C_s,
    }
    scalar_outputs = {
        "k_trapped": float(k),
        "k_equilibrium": float(k_0),
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
            "partition_coefficient_effective": k,
            "partition_coefficient_equilibrium": k_0,
            "interface_velocity": interface_velocity,
            "C0": C_0,
            "diffusivity_solid": diffusivity_solid,
            "solidification_time": solidification_time,
            "dendrite_arm_spacing": dendrite_arm_spacing,
        },
    )


def simulate_am_rapid_solidification(
    liquidus_temp: float,
    solidus_temp_equilibrium: float,
    solute_concentration: float,
    partition_coefficient: float,
    cooling_rate: float = 1e5,
    thermal_gradient: float = 1e6,
    diffusive_velocity: float = 5.0,
    diffusivity_solid: float = 1e-12,
    num_steps: int = 100
) -> Dict[str, Any]:
    """
    Simulates rapid solidification under Additive Manufacturing (AM) processing conditions.

    Computes:
      1. Growth velocity: v = cooling_rate / thermal_gradient
      2. Solute trapping: k(v) via Aziz CGM model
      3. Solidification time: t_f = (T_liq - T_sol) / cooling_rate
      4. Secondary dendrite arm spacing (SDAS): lambda_2 = A * cooling_rate^(-1/3)
      5. Temperature profile T(f_s) and fraction solid f_s
    """
    t_liq = float(liquidus_temp)
    t_sol = float(solidus_temp_equilibrium)
    dT_eq = max(t_liq - t_sol, 10.0)

    # 1. Growth velocity
    v_growth = cooling_rate / max(thermal_gradient, 1e2)

    # 2. Solute trapping
    k_v = calculate_trapped_partition_coefficient(partition_coefficient, v_growth, diffusive_velocity)

    # 3. Solidification time
    t_f = dT_eq / max(cooling_rate, 1.0)

    # 4. SDAS scaling for metals: lambda_2 ~ 50 * (cooling_rate)^(-1/3) um
    sdas_m = 50.0e-6 * (max(cooling_rate, 1.0) ** (-1.0 / 3.0))

    # 5. Microsegregation run
    res_seg = simulate_microsegregation(
        solute_concentration_initial=solute_concentration,
        partition_coefficient=partition_coefficient,
        diffusivity_solid=diffusivity_solid,
        solidification_time=t_f,
        dendrite_arm_spacing=sdas_m,
        interface_velocity=v_growth,
        diffusive_velocity=diffusive_velocity,
        num_steps=num_steps,
        model="ohnaka"
    )

    f_s = res_seg.grid["f_s"]
    # Non-equilibrium freezing range shrinks with solute trapping as k -> 1
    # When k_v = 1, solidification is partitionless (zero freezing range at T_0)
    trapping_factor = (1.0 - k_v) / max(1.0 - partition_coefficient, 1e-4)
    dT_noneq = dT_eq * np.clip(trapping_factor, 0.05, 1.0)

    # Temperature profile following Scheil-like curve scaled by non-equilibrium freezing range
    t_profiles = t_liq - dT_noneq * (f_s ** (max(k_v, 0.05)))

    from pycalphad.am.cracking import evaluate_cracking_susceptibility
    cracking_assessment = evaluate_cracking_susceptibility(t_profiles, f_s)

    return {
        "interface_velocity_m_s": float(v_growth),
        "k_trapped": float(k_v),
        "k_equilibrium": float(partition_coefficient),
        "sdas_meters": float(sdas_m),
        "solidification_time_s": float(t_f),
        "fraction_solid": f_s,
        "temperatures": t_profiles,
        "C_liquid": res_seg.field_data["C_liquid"],
        "C_solid": res_seg.field_data["C_solid"],
        "cracking_assessment": cracking_assessment.summary(),
    }
