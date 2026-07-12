import os
import numpy as np
import pytest
from importlib.resources import files
from pycalphad import Database
from pycalphad.diffusion import BinaryDiffusionSimulation
import pycalphad.tests.databases

def test_binary_diffusion_couple():
    # Load test database (Al-Zn system)
    db_path = str(files(pycalphad.tests.databases).joinpath("alzn_mey.tdb"))
    dbf = Database(db_path)
    
    comps = ['AL', 'ZN', 'VA']
    phase = 'FCC_A1'
    T = 800.0 # 800 K is above the FCC miscibility gap critical temperature (~623 K)
    
    Lx = 1e-6    # 1 micron
    nx = 20
    x_left = 0.1
    x_right = 0.3
    
    # Mobilities (m^2*mol/J/s)
    sim = BinaryDiffusionSimulation(
        dbf, comps, phase, T, Lx, nx, x_left, x_right,
        mobility_A=1.0e-15, mobility_B=2.0e-15
    )
    
    # Check initial profile
    x_coords, xb_init = sim.get_profile()
    assert len(x_coords) == nx
    assert np.isclose(xb_init[0], x_left)
    assert np.isclose(xb_init[-1], x_right)
    
    # Calculate initial mass/solute integral
    initial_solute = np.sum(xb_init) * sim.dx
    
    # Run 5 steps with dt = 1e-4 s
    dt = 1e-4
    for _ in range(5):
        sim.step(dt)
        
    x_coords, xb_final = sim.get_profile()
    
    # Check that concentrations evolved but remain bounded
    assert np.all(xb_final >= x_left - 1e-9)
    assert np.all(xb_final <= x_right + 1e-9)
    
    # Concentration profile should start to smooth out (diffusion)
    # Left side should increase, right side should decrease
    assert xb_final[0] > x_left
    assert xb_final[-1] < x_right
    
    # Check mass conservation (integral of solute Zn should remain constant)
    final_solute = np.sum(xb_final) * sim.dx
    assert np.isclose(initial_solute, final_solute, rtol=1e-12)
    
    # Verify interdiffusion coefficient is positive
    d_coeff = sim.get_interdiffusion_coefficient(0.2)
    assert d_coeff > 0.0

def test_multicomponent_diffusion_couple():
    # Load database (COST507 multicomponent database)
    db_path = str(files(pycalphad.tests.databases).joinpath("COST507.tdb"))
    dbf = Database(db_path)
    
    # We define a ternary couple AL-ZN-MG with user constant diffusivities
    comps = ['AL', 'ZN', 'MG', 'VA']
    phase = 'FCC_A1'
    T = 600.0
    Lx = 1e-8
    nx = 20
    
    x_left = {'ZN': 0.1, 'MG': 0.05}
    x_right = {'ZN': 0.2, 'MG': 0.10}
    
    from pycalphad.diffusion.couple import DiffusionCoupleSimulation
    with pytest.warns(UserWarning):
        sim = DiffusionCoupleSimulation(
            dbf, comps, phase, T, Lx, nx, x_left, x_right,
            user_constants={'AL': 1e-15, 'ZN': 2e-15, 'MG': 1.5e-15},
            solver='numpy'
        )
    
    # Verify initial profiles
    x_coords, z_profile = sim.get_profile('ZN')
    assert len(x_coords) == nx
    assert np.isclose(z_profile[0], 0.1)
    assert np.isclose(z_profile[-1], 0.2)
    
    _, m_profile = sim.get_profile('MG')
    assert np.isclose(m_profile[0], 0.05)
    assert np.isclose(m_profile[-1], 0.10)
    
    # Get total mass before stepping
    init_zn_mass = np.sum(z_profile) * sim.dx
    init_mg_mass = np.sum(m_profile) * sim.dx
    
    # Advance time
    dt = 1e-4
    for _ in range(5):
        sim.step(dt)
        
    _, z_profile_final = sim.get_profile('ZN')
    _, m_profile_final = sim.get_profile('MG')
    
    # Profiles should start to smooth out
    assert z_profile_final[0] > 0.1
    assert z_profile_final[-1] < 0.2
    assert m_profile_final[0] > 0.05
    assert m_profile_final[-1] < 0.10
    
    # Verify mass conservation
    final_zn_mass = np.sum(z_profile_final) * sim.dx
    final_mg_mass = np.sum(m_profile_final) * sim.dx
    assert np.isclose(init_zn_mass, final_zn_mass, rtol=1e-12)
    assert np.isclose(init_mg_mass, final_mg_mass, rtol=1e-12)

