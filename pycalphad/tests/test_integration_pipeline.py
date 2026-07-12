import pytest
import numpy as np
from importlib.resources import files
from pycalphad import Database
import pycalphad.tests.databases

# Import new/upgraded custom module wrappers
from pycalphad.amkit import (
    find_liquidus_temperature,
    simulate_solidification,
    MobilityModel,
    rosenthal_T,
    eagar_tsai_T,
    solidification_conditions,
    ThermalHistory
)
from pycalphad.am.cracking import susceptibility_from_composition
from pycalphad.am.thermal import solve_thermal_profile, extract_thermal_history_probe
from pycalphad.diffusion.couple import DiffusionCoupleSimulation
from pycalphad.precipitation import PrecipitationKineticsSimulation

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_integrated_simulation_pipeline():
    # 1. Database and Material Configuration (Al-Zn-Mg system using COST507.tdb)
    db_path = str(files(pycalphad.tests.databases).joinpath("COST507.tdb"))
    dbf = Database(db_path)
    
    comps = ['AL', 'ZN', 'MG', 'VA']
    phases = ['FCC_A1', 'HCP_A3']
    nominal_composition = {'ZN': 0.08, 'MG': 0.04}
    
    # 2. Solidification Phase: Predict solidification temperatures and cracking susceptibility
    print("\n--- [Step 2] Solidification Simulation & Cracking Susceptibility ---")
    solidification_phases = ['LIQUID', 'FCC_A1']
    indices, solid_res = susceptibility_from_composition(
        dbf, ['AL', 'ZN', 'VA'], solidification_phases, {'ZN': 0.08}, step=5.0
    )
    
    T_liq = solid_res.T_liquidus
    T_sol = solid_res.T_solidus_scheil
    assert 800.0 < T_liq < 950.0
    assert 400.0 < T_sol < 850.0
    assert indices['kou'] >= 0.0
    assert indices['csc'] >= 0.0
    assert indices['rdg'] >= 0.0
    print(f"Liquidus: {T_liq:.2f} K, Solidus (Scheil): {T_sol:.2f} K")
    print(f"Kou Index: {indices['kou']:.2f}, Clyne-Davis: {indices['csc']:.2f}, RDG: {indices['rdg']:.2f}")

    # 3. Thermal Analysis Phase: Compute melt pool parameters and local thermal history
    print("\n--- [Step 3] Thermal Analysis & Melt Pool ---")
    density = 2700.0       # Al alloy density (kg/m^3)
    specific_heat = 900.0  # (J/kg/K)
    conductivity = 160.0   # (W/m/K)
    P = 200.0              # Laser power (W)
    v_laser = 0.5          # Laser scan speed (m/s)
    
    # Find solidification conditions behind heat source using Rosenthal
    T_func = lambda x, y, z: rosenthal_T(
        x, y, z, P, v_laser, density, specific_heat, conductivity, T_ambient=298.15, absorptivity=0.8
    )
    # Probe rear melt pool boundary where temperature is at liquidus
    x_boundary = -3e-4  # trailing region
    G, R, cooling_rate = solidification_conditions(T_func, x_boundary, 0.0, 0.0, v_laser)
    assert G > 0.0
    assert R > 0.0
    assert cooling_rate > 0.0
    print(f"Solidification front gradient G: {G:.2e} K/m, rate R: {R:.2f} m/s, cooling: {cooling_rate:.2e} K/s")

    # Run the 2D finite-difference transient solver to extract local thermal history
    # dt=None automatically calculates the maximum stable time step
    X, Y, T, t_axis, T_history = solve_thermal_profile(
        Lx=2e-3, Ly=2e-3, dx=1e-4, dy=1e-4, t_max=0.004,
        laser_power=P, absorptivity=0.8, beam_radius=1e-4, scan_speed=v_laser,
        density=density, specific_heat=specific_heat, conductivity=conductivity,
        dt=None, return_history=True, strict_stability=True
    )
    t_probe, T_probe = extract_thermal_history_probe(X, Y, T_history, t_axis, 1e-3, 1e-3)
    history = ThermalHistory(t_probe, T_probe)
    t_above = history.time_above(800.0)
    print(f"Time spent above 800K: {t_above:.2e} seconds")

    # 4. Solute Redistribution Phase: Predict multicomponent diffusion couple evolution
    print("\n--- [Step 4] Multicomponent 1D Diffusion Couple ---")
    Lx_diff = 1e-8
    nx_diff = 10
    # Left and right sides of the couple
    x_left = {'ZN': 0.06, 'MG': 0.03}
    x_right = {'ZN': 0.10, 'MG': 0.05}
    
    with pytest.warns(UserWarning):
        diff_sim = DiffusionCoupleSimulation(
            dbf, comps, 'FCC_A1', T=600.0, Lx=Lx_diff, nx=nx_diff,
            x_left=x_left, x_right=x_right,
            user_constants={'AL': 1e-18, 'ZN': 2e-18, 'MG': 1.5e-18},
            solver='numpy'
        )
    
    # Step diffusion couple
    dt_diff = 1e-4
    diff_sim.step(dt_diff)
    _, z_profile = diff_sim.get_profile('ZN')
    _, m_profile = diff_sim.get_profile('MG')
    # Mass conservation validation
    assert len(z_profile) == nx_diff
    assert np.isclose(np.sum(z_profile) * diff_sim.dx, 0.8 * Lx_diff, rtol=1e-12)
    print("Diffusion couple step complete and mass conserved.")

    # 5. Precipitation Kinetics & Yield Strength Phase: Non-isothermal precipitation
    print("\n--- [Step 5] Precipitation & Yield Strength Coupling ---")
    # TBD-lookups: VM molar volume resolution check
    from pycalphad.precipitation.kinetics import resolve_molar_volume
    vm_resolved = resolve_molar_volume(dbf, 'FCC_A1')
    assert vm_resolved > 0.0
    
    # Create non-isothermal aging profile (cooling from 450K to 400K)
    precip_times = np.array([0.0, 1e4])
    precip_temps = np.array([450.0, 400.0])
    
    # Set up simulation with database-lookup diffusivity fallback
    with pytest.warns(UserWarning, match="Using user constant diffusivity/mobility overrides"):
        precip_sim = PrecipitationKineticsSimulation(
            dbf, comps, phases, T=(precip_times, precip_temps),
            initial_composition=nominal_composition, interfacial_energy=0.08,
            diffusivity=None, molar_volume=None,
            user_constants={'AL': 1e-19, 'ZN': 2e-19, 'MG': 1.5e-19}
        )
    
    # Enable dynamic Yield Strength coupling
    precip_sim.add_strength_model()
    
    # Step the kinetics model
    precip_sim.step(10.0)
    
    # Verify yield strength and phase fractions evolved
    results = precip_sim.get_results()
    assert 'yield_strength' in results
    assert results['yield_strength'][-1] >= 0.0
    assert results['volume_fraction'][-1, 0] >= 0.0
    print(f"Precipitation time steps: {len(results['time'])}")
    print(f"Final Volume Fraction: {results['volume_fraction'][-1, 0] * 100:.4f}%")
    print(f"Final Predicted Yield Strength: {results['yield_strength'][-1]:.2f} Pa")
    print("Integration pipeline completed successfully.")
