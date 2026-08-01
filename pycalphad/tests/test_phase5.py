"""
Unit tests for Phase 5 physical simulation extensions:
3D thermal solver, microsegregation back-diffusion, and precipitation strengthening.
"""

import pytest
import numpy as np
from pycalphad.am import Thermal3DSimulation, ThermalHistory, MicrostructureState, SimulationResult
from pycalphad.amkit.microsegregation import simulate_microsegregation
from pycalphad.precipitation.strengthening import PrecipitationStrengtheningModel

def test_thermal_history_dataclass():
    time = np.linspace(0, 10, 101)
    temp = 300 + 1000 * np.exp(-(time - 2)**2 / 2)
    
    th = ThermalHistory(time=time, temperature=temp)
    assert th.peak_temperature == pytest.approx(1300.0, abs=1.0)
    assert th.cooling_rate(1000.0) > 0
    assert th.time_above(1000.0) > 0

def test_3d_thermal_fvm_solver():
    domain = ((-0.001, 0.001), (-0.001, 0.001), (-0.001, 0.000))
    grid_shape = (11, 11, 6)
    
    sim = Thermal3DSimulation(
        domain_bounds=domain,
        grid_shape=grid_shape,
        thermal_conductivity=30.0,
        heat_capacity=500.0,
        density=7800.0,
        T_ambient=300.0
    )
    
    # Point heat source at center
    def heat_source(X, Y, Z, t):
        Q = np.zeros_like(X)
        Q[5, 5, 2] = 1e10
        return Q

    res = sim.solve(duration=0.001, dt=1e-5, heat_source_func=heat_source)
    assert res.simulation_type == "3d_thermal_fvm"
    assert res.scalar_outputs["peak_temperature"] > 300.0
    assert "temperature_field" in res.field_data

def test_microsegregation_back_diffusion_limits():
    # Scheil limit (alpha = 0)
    res_scheil = simulate_microsegregation(
        solute_concentration_initial=1.0,
        partition_coefficient=0.5,
        diffusivity_solid=0.0,
        solidification_time=1.0,
        dendrite_arm_spacing=10e-6,
        model="scheil"
    )
    
    # Lever limit (alpha -> infinity)
    res_lever = simulate_microsegregation(
        solute_concentration_initial=1.0,
        partition_coefficient=0.5,
        diffusivity_solid=1e-8,
        solidification_time=100.0,
        dendrite_arm_spacing=1e-6,
        model="lever"
    )
    
    # Scheil liquid concentration increases faster than lever rule
    assert res_scheil.field_data["C_liquid"][-1] > res_lever.field_data["C_liquid"][-1]
    assert res_scheil.scalar_outputs["alpha"] == 0.0

def test_orowan_precipitation_strengthening():
    model = PrecipitationStrengtheningModel()
    
    # Test zero precipitates
    sigma_zero = model.orowan_strengthening(mean_radius=0.0, volume_fraction=0.0)
    assert sigma_zero == 0.0
    
    # Test valid precipitate metrics (e.g. 5 nm precipitates, 1% volume fraction)
    sigma_orowan = model.orowan_strengthening(mean_radius=5e-9, volume_fraction=0.01)
    assert sigma_orowan > 0.0
    
    # Breakdown test
    breakdown = model.calculate_total_yield_strength(
        base_yield_strength=200e6,
        mean_radius=5e-9,
        volume_fraction=0.01,
        solute_concentrations={"Mg": 0.02},
        strengthening_coefficients={"Mg": 30e6},
        grain_size=20e-6
    )
    assert breakdown["total_yield_strength_MPa"] > 200.0
    assert breakdown["orowan_strength_MPa"] > 0.0
