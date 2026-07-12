import numpy as np
import pytest
from pycalphad.am.cracking import calculate_clyne_davis_index, calculate_kou_index
from pycalphad.am.heat_sources import GaussianHeatSource, DoubleEllipsoidalHeatSource, ConicalHeatSource
from pycalphad.am.thermal import solve_thermal_profile

def test_cracking_indices():
    # Linear temperature vs solid fraction profile
    # T(f_S) = 1500 - 500 * f_S
    # T_0.40 = 1300
    # T_0.90 = 1050
    # T_0.99 = 1005
    # CSC = (1050 - 1005) / (1300 - 1050) = 45 / 250 = 0.18
    #
    # Kou CSI: y = sqrt(f_S). f_S = y^2. T = 1500 - 500 * y^2.
    # dT/dy = -1000 * y.
    # For 0.90 <= f_S <= 0.99, 0.948 <= y <= 0.995.
    # CSI = max |dT/dy| = 1000 * 0.995 = 995 (approximately, since it's the edge point).
    f_s = np.linspace(0.0, 1.0, 101)
    temperatures = 1500.0 - 500.0 * f_s
    
    csc = calculate_clyne_davis_index(temperatures, f_s)
    assert np.isclose(csc, 0.18)
    
    csi = calculate_kou_index(temperatures, f_s)
    # Check that CSI is calculated without crash and is around 995
    assert csi > 900.0 and csi < 1100.0


def test_heat_source_models():
    # Gaussian
    g_source = GaussianHeatSource(power=100.0, absorptivity=0.8, beam_radius=5e-5)
    q_g_max = g_source(0.0, 0.0)
    q_g_off = g_source(1e-4, 1e-4)
    assert q_g_max > q_g_off
    assert q_g_max > 0.0
    
    # Goldak Double Ellipsoidal
    goldak = DoubleEllipsoidalHeatSource(power=100.0, absorptivity=0.8, af=1e-4, ar=2e-4, b=1e-4, c=5e-5)
    # Volumetric at center
    q_goldak_center = goldak(0.0, 0.0, 0.0)
    q_goldak_front = goldak(5e-5, 0.0, 0.0) # front ellipsoid x > 0
    q_goldak_rear = goldak(-5e-5, 0.0, 0.0) # rear ellipsoid x < 0
    assert q_goldak_center > q_goldak_front
    assert q_goldak_center > q_goldak_rear
    
    # Conical
    conical = ConicalHeatSource(power=100.0, absorptivity=0.8, re=1e-4, ri=5e-5, H=2e-4)
    q_conical_top = conical(0.0, 0.0, 0.0)
    q_conical_bottom = conical(0.0, 0.0, 2e-4)
    q_conical_outside = conical(0.0, 0.0, 3e-4) # beyond depth H
    assert q_conical_top > q_conical_bottom
    assert q_conical_outside == 0.0


def test_thermal_solver():
    # Run a tiny thermal simulation to verify execution
    Lx = 1e-3
    Ly = 1e-3
    dx = 1e-4
    dy = 1e-4
    t_max = 0.005
    
    # Material properties for steel
    density = 7800.0
    specific_heat = 500.0
    conductivity = 15.0
    
    X, Y, T, t_axis = solve_thermal_profile(
        Lx, Ly, dx, dy, t_max,
        laser_power=200.0, absorptivity=0.7, beam_radius=1e-4, scan_speed=0.1,
        density=density, specific_heat=specific_heat, conductivity=conductivity,
        x0=2e-4, y0=5e-4, T_ambient=298.15, h_loss=10.0, dt=1e-4
    )
    
    assert X.shape == Y.shape
    assert T.shape == X.shape
    assert np.max(T) >= 298.15
    assert len(t_axis) > 1

def test_thermal_solver_advanced():
    # Run a small thermal simulation with latent heat, return_history, and probe extraction
    Lx = 1e-3
    Ly = 1e-3
    dx = 1e-4
    dy = 1e-4
    t_max = 0.005
    
    density = 7800.0
    specific_heat = 500.0
    conductivity = 15.0
    
    latent_heat = 2.7e5
    solidus = 1600.0
    liquidus = 1650.0
    
    from pycalphad.am.thermal import extract_thermal_history_probe
    
    X, Y, T, t_axis, T_history = solve_thermal_profile(
        Lx, Ly, dx, dy, t_max,
        laser_power=200.0, absorptivity=0.7, beam_radius=1e-4, scan_speed=0.1,
        density=density, specific_heat=specific_heat, conductivity=conductivity,
        x0=2e-4, y0=5e-4, T_ambient=298.15, h_loss=10.0, dt=1e-4,
        latent_heat=latent_heat, solidus_temp=solidus, liquidus_temp=liquidus,
        return_history=True
    )
    
    assert len(T_history) == len(t_axis)
    assert T_history[0].shape == X.shape
    
    t_prof, T_prof = extract_thermal_history_probe(X, Y, T_history, t_axis, 5e-4, 5e-4)
    assert len(t_prof) == len(t_axis)
    assert len(T_prof) == len(t_axis)


def test_susceptibility_from_composition():
    # Load database (Al-Zn system)
    from importlib.resources import files
    import pycalphad.tests.databases
    from pycalphad import Database
    from pycalphad.am.cracking import susceptibility_from_composition
    
    db_path = str(files(pycalphad.tests.databases).joinpath("alzn_mey.tdb"))
    dbf = Database(db_path)
    comps = ['AL', 'ZN', 'VA']
    phases = ['LIQUID', 'FCC_A1']
    composition = {'ZN': 0.1}
    
    indices, res = susceptibility_from_composition(dbf, comps, phases, composition, step=5.0)
    
    assert 'kou' in indices
    assert 'csc' in indices
    assert 'freezing_range' in indices
    assert 'tfr' in indices
    assert 'rdg' in indices
    
    assert indices['freezing_range'] > 0.0
    assert indices['tfr'] >= 0.0
    assert indices['rdg'] >= 0.0

def test_thermal_solver_instability():
    # Model parameters for steel
    density = 7800.0
    specific_heat = 500.0
    conductivity = 15.0
    
    # Passing dt = 0.5s is way above the stability limit (which is ~1e-4 s for dx=1e-4)
    # This should raise ValueError by default
    with pytest.raises(ValueError, match="exceeds explicit solver stability limit"):
        solve_thermal_profile(
            1e-3, 1e-3, 1e-4, 1e-4, 0.005,
            laser_power=200.0, absorptivity=0.7, beam_radius=1e-4, scan_speed=0.1,
            density=density, specific_heat=specific_heat, conductivity=conductivity,
            dt=0.5, strict_stability=True
        )
        
    # In non-strict mode, it should issue a warning (which pytest turns into error unless wrapped or ignored)
    with pytest.warns(UserWarning, match="exceeds explicit solver stability limit|Numerical instability detected"):
        # Suppress potential explosion errors in the loop as well
        solve_thermal_profile(
            1e-3, 1e-3, 1e-4, 1e-4, 0.005,
            laser_power=200.0, absorptivity=0.7, beam_radius=1e-4, scan_speed=0.1,
            density=density, specific_heat=specific_heat, conductivity=conductivity,
            dt=0.5, strict_stability=False
        )


