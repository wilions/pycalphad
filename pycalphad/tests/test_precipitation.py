import pytest
import numpy as np
from importlib.resources import files
from pycalphad import Database
from pycalphad.precipitation import PrecipitationKineticsSimulation
import pycalphad.tests.databases

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_precipitation_kinetics():
    # Load database (Al-Zn system)
    db_path = str(files(pycalphad.tests.databases).joinpath("alzn_mey.tdb"))
    dbf = Database(db_path)
    
    # Reference element AL, solute ZN
    elements = ['AL', 'ZN', 'VA']
    phases = ['FCC_A1', 'HCP_A3']
    T = 400.0 # Kelvin
    
    initial_comp = 0.15
    interfacial_energy = 0.08
    diffusivity = 1.0e-19 # m^2/s
    
    sim = PrecipitationKineticsSimulation(
        dbf, elements, phases, T, initial_comp,
        interfacial_energy, diffusivity
    )
    
    # Run the simulation for 1e4 seconds
    sim.step(1.0e4)
    
    # Get results
    results = sim.get_results()
    
    assert len(results['time']) > 1
    assert results['time'][0] == 0.0
    assert np.isclose(results['time'][-1], 1.0e4, rtol=1e-5)
    
    # Precipitate density should increase due to nucleation
    assert results['precipitate_density'][-1, 0] > 1e15
    
    # Volume fraction should grow
    assert results['volume_fraction'][-1, 0] > 0.0
    
    # Mean radius should increase
    assert results['mean_radius'][-1, 0] > 1.0e-9 # larger than 1 nm

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_precipitation_upgrades():
    # Load database (Al-Zn system)
    db_path = str(files(pycalphad.tests.databases).joinpath("alzn_mey.tdb"))
    dbf = Database(db_path)
    
    # Reference element AL, solute ZN
    elements = ['AL', 'ZN', 'VA']
    phases = ['FCC_A1', 'HCP_A3']
    
    # Non-isothermal: cooling from 400 K to 380 K over 1e4 seconds
    times = np.array([0.0, 1.0e4])
    temps = np.array([400.0, 380.0])
    T_profile = (times, temps)
    
    initial_comp = 0.15
    interfacial_energy = 0.08
    
    # Test 1: Instantiation with diffusivity=None (uses user constants warning bypass)
    with pytest.warns(UserWarning, match="Using user constant diffusivity/mobility overrides"):
        sim = PrecipitationKineticsSimulation(
            dbf, elements, phases, T_profile, initial_comp,
            interfacial_energy, diffusivity=None, molar_volume=None,
            user_constants={'AL': 1e-19, 'ZN': 2e-19}
        )

    # Test 2: Add dynamic strength model coupling
    sim.add_strength_model()

    # Step the simulation
    sim.step(100.0)

    # Retrieve and check results
    results = sim.get_results()
    assert len(results['time']) > 1
    assert 'yield_strength' in results
    assert len(results['yield_strength']) == len(results['time'])
    assert results['yield_strength'][-1] >= 0.0

