"""
Multi-Phase Multi-Component Co-Precipitation Kinetics & Strengthening Engine.

This module models concurrent nucleation, growth, and coarsening of multiple
competing precipitate phases (e.g., gamma-prime and gamma-double-prime in Ni-superalloys,
or theta-prime and S-phase in Al-alloys) with mean-field solute conservation
and elastic strain energy corrections.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class PrecipitatePhaseSpec:
    """Specification for a single precipitate phase in co-precipitation."""
    name: str
    interfacial_energy: float        # J/m^2 (gamma)
    molar_volume: float              # m^3/mol
    solute_stoichiometry: Dict[str, float]  # Solute mole fractions in precipitate, e.g. {'ZN': 0.5, 'MG': 0.5}
    elastic_misfit_energy: float = 0.0      # J/mol (delta G_elastic)
    diffusing_species: str = "solute"
    initial_radius: float = 1.0e-9   # Initial nucleus radius (m)
    initial_density: float = 0.0     # Initial number density (#/m^3)


@dataclass
class CoPrecipitationResult:
    """Output from multi-phase co-precipitation kinetics simulation."""
    times: np.ndarray                            # Time array in seconds
    temperatures: np.ndarray                     # Temperature history in K
    matrix_solute_fractions: Dict[str, np.ndarray]  # Solute fraction in matrix vs time
    phase_volume_fractions: Dict[str, np.ndarray]   # Phase volume fraction vs time
    phase_mean_radii: Dict[str, np.ndarray]         # Mean particle radius (m) vs time
    phase_number_densities: Dict[str, np.ndarray]   # Number density (#/m^3) vs time
    phase_strengthening_MPa: Dict[str, np.ndarray]  # Yield strength contribution (MPa) vs time
    total_strengthening_MPa: np.ndarray             # Total superposed precipitation strengthening (MPa)


class MultiPhasePrecipitationKinetics:
    """
    Coupled multi-phase KWN co-precipitation kinetics simulator.
    """

    def __init__(
        self,
        matrix_name: str = "FCC_A1",
        matrix_molar_volume: float = 1.0e-5, # m^3/mol
        initial_solute_fractions: Optional[Dict[str, float]] = None,
        burgers_vector: float = 0.286e-9,    # Burgers vector in meters (b)
        shear_modulus: float = 27.0e9,       # Matrix shear modulus in Pa (G)
    ):
        self.matrix_name = matrix_name
        self.matrix_molar_volume = matrix_molar_volume
        self.initial_solute = initial_solute_fractions or {}
        self.burgers_vector = burgers_vector
        self.shear_modulus = shear_modulus
        self.precipitate_phases: Dict[str, PrecipitatePhaseSpec] = {}

    def add_precipitate_phase(self, spec: PrecipitatePhaseSpec) -> None:
        """Register a precipitate phase."""
        self.precipitate_phases[spec.name] = spec

    def simulate(
        self,
        t_total: float = 36000.0,       # Total aging time in seconds (e.g. 10 hrs)
        num_steps: int = 1000,
        temperature: Union[float, Callable[[float], float]] = 450.0, # K
        diffusivities: Optional[Dict[str, Union[float, Callable[[float], float]]]] = None,
    ) -> CoPrecipitationResult:
        """
        Run time-stepping numerical integration for multi-phase precipitation kinetics.
        """
        if not self.precipitate_phases:
            raise ValueError("No precipitate phases registered for simulation.")

        times = np.linspace(0.0, t_total, num_steps)
        dt = times[1] - times[0] if num_steps > 1 else 1.0

        if callable(temperature):
            temps = np.array([temperature(t) for t in times])
        else:
            temps = np.full_like(times, float(temperature))

        k_B = 1.380649e-23  # Boltzmann constant J/K
        N_A = 6.02214076e23 # Avogadro constant

        # Initialize tracking dictionaries
        vol_fractions = {name: np.zeros_like(times) for name in self.precipitate_phases}
        mean_radii = {name: np.zeros_like(times) for name in self.precipitate_phases}
        num_densities = {name: np.zeros_like(times) for name in self.precipitate_phases}
        strengthening = {name: np.zeros_like(times) for name in self.precipitate_phases}
        matrix_solute = {sol: np.full_like(times, val) for sol, val in self.initial_solute.items()}

        # Initial conditions
        for name, spec in self.precipitate_phases.items():
            mean_radii[name][0] = spec.initial_radius
            num_densities[name][0] = spec.initial_density
            vol_fractions[name][0] = (4.0 / 3.0) * np.pi * (spec.initial_radius ** 3) * spec.initial_density

        # Time integration loop
        curr_radii = {name: spec.initial_radius for name, spec in self.precipitate_phases.items()}
        curr_densities = {name: spec.initial_density for name, spec in self.precipitate_phases.items()}
        curr_matrix_solute = dict(self.initial_solute)

        for step in range(1, num_steps):
            T = temps[step]
            
            # Default solute diffusivity if not provided: D = D0 * exp(-Q / RT)
            # Standard substitutional in FCC: D0 ~ 1e-4 m^2/s, Q ~ 130 kJ/mol
            R_gas = 8.314462618
            default_D = 1.0e-4 * np.exp(-130000.0 / (R_gas * T))

            for name, spec in self.precipitate_phases.items():
                # Primary solute for this precipitate
                sol_name = list(spec.solute_stoichiometry.keys())[0] if spec.solute_stoichiometry else list(curr_matrix_solute.keys())[0]
                c_matrix = curr_matrix_solute.get(sol_name, 0.01)
                c_phase = spec.solute_stoichiometry.get(sol_name, 0.5)

                D_val = default_D
                if diffusivities and sol_name in diffusivities:
                    d_item = diffusivities[sol_name]
                    D_val = d_item(T) if callable(d_item) else float(d_item)

                # Driving force estimate: Delta G_chem approx -R*T*ln(c_matrix / c_eq)
                # Equilibrium solubility proxy:
                c_eq = 0.002 * np.exp(-35000.0 / (R_gas * T))
                supersaturation = max((c_matrix - c_eq) / max(c_phase - c_eq, 1e-4), 0.0)

                delta_G_v = (R_gas * T / spec.molar_volume) * np.log(max(c_matrix / max(c_eq, 1e-6), 1.0001))
                delta_G_eff = max(delta_G_v - (spec.elastic_misfit_energy / spec.molar_volume), 1e3)

                # Critical radius: r* = 2 * gamma / Delta G_eff
                r_star = max(2.0 * spec.interfacial_energy / delta_G_eff, 0.5e-9)

                # Nucleation barrier: Delta G* = (16 pi / 3) * gamma^3 / Delta G_eff^2
                delta_G_star = (16.0 * np.pi / 3.0) * (spec.interfacial_energy ** 3) / (delta_G_eff ** 2)
                
                # Classical nucleation rate J (# / (m^3 * s))
                N0 = 1.0e28  # nucleation site density per m^3
                beta_star = 4.0 * np.pi * (r_star ** 2) * D_val * (c_matrix / spec.molar_volume * N_A) / (self.burgers_vector ** 4 * N_A)
                Z_zeldovich = 0.05
                J_nuc = N0 * Z_zeldovich * beta_star * np.exp(-min(delta_G_star / (k_B * T), 50.0))
                J_nuc = min(J_nuc, 1.0e22) if supersaturation > 1e-4 else 0.0

                # Particle Growth / Dissolution rate dr/dt (Zener-Wert-Frank approximation)
                r_curr = max(curr_radii[name], r_star)
                dr_dt = (D_val / r_curr) * ((c_matrix - c_eq) / (c_phase - c_eq)) if (c_matrix > c_eq) else -1e-12

                # Update particle population
                new_density = curr_densities[name] + J_nuc * dt
                new_radius = max(r_curr + dr_dt * dt, spec.initial_radius)

                curr_densities[name] = new_density
                curr_radii[name] = new_radius
                
                # Volume fraction: phi = (4/3) * pi * r^3 * N_v
                phi = (4.0 / 3.0) * np.pi * (new_radius ** 3) * new_density
                phi = min(phi, 0.35)  # Physical upper limit on precipitate fraction

                vol_fractions[name][step] = phi
                mean_radii[name][step] = new_radius
                num_densities[name][step] = new_density

                # Orowan Dislocation Bypassing & Shearing yield strength (MPa)
                # Orowan stress: Delta sigma_Orowan = (M * 0.81 * G * b) / (2 * pi * sqrt(1 - nu)) * ln(2r/b) / (L - 2r)
                # Mean interparticle planar spacing L = sqrt(2*pi / (3*phi)) * r
                if phi > 1e-6 and new_radius > 1e-9:
                    M_taylor = 3.06
                    nu_poisson = 0.33
                    L_spacing = np.sqrt(2.0 * np.pi / (3.0 * phi)) * new_radius
                    eff_spacing = max(L_spacing - 2.0 * new_radius, 1.0e-9)
                    
                    # Orowan looping stress
                    sigma_orowan = (
                        (M_taylor * 0.81 * self.shear_modulus * self.burgers_vector)
                        / (2.0 * np.pi * np.sqrt(1.0 - nu_poisson) * eff_spacing)
                    ) * np.log(max(2.0 * new_radius / self.burgers_vector, 1.1)) * 1e-6  # Pa to MPa

                    # Shearing / Friedel cutting stress for small coherent particles
                    gamma_apb = 0.15 # J/m^2 anti-phase boundary energy
                    sigma_shearing = (M_taylor * gamma_apb / (2.0 * self.burgers_vector)) * np.sqrt(
                        (3.0 * np.pi * phi * new_radius * gamma_apb) / (self.shear_modulus * self.burgers_vector ** 2)
                    ) * 1e-6

                    # Active mechanism is min(shearing, orowan)
                    sigma_phase = min(sigma_orowan, sigma_shearing)
                else:
                    sigma_phase = 0.0

                strengthening[name][step] = sigma_phase

            # Mean-field solute conservation update:
            # c_matrix = (c_total - sum(phi_p * c_p)) / (1 - sum(phi_p))
            total_precip_vol = sum(vol_fractions[p][step] for p in self.precipitate_phases)
            for sol_name, initial_val in self.initial_solute.items():
                precip_solute = sum(
                    vol_fractions[p][step] * self.precipitate_phases[p].solute_stoichiometry.get(sol_name, 0.0)
                    for p in self.precipitate_phases
                )
                c_rem = max((initial_val - precip_solute) / max(1.0 - total_precip_vol, 1e-3), 0.0)
                curr_matrix_solute[sol_name] = c_rem
                matrix_solute[sol_name][step] = c_rem

        # Superpose total precipitation yield strength via generalized root-sum-square
        q_exponent = 1.5
        total_strength_MPa = np.zeros_like(times)
        for name in self.precipitate_phases:
            total_strength_MPa += (strengthening[name] ** q_exponent)
        total_strength_MPa = total_strength_MPa ** (1.0 / q_exponent)

        return CoPrecipitationResult(
            times=times,
            temperatures=temps,
            matrix_solute_fractions=matrix_solute,
            phase_volume_fractions=vol_fractions,
            phase_mean_radii=mean_radii,
            phase_number_densities=num_densities,
            phase_strengthening_MPa=strengthening,
            total_strengthening_MPa=total_strength_MPa,
        )
