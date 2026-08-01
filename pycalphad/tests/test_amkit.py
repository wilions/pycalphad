import os
import numpy as np
import pytest
from importlib.resources import files
from pycalphad import Database, variables as v
import pycalphad.tests.databases

from pycalphad.amkit.solidification import find_liquidus_temperature, simulate_solidification
from pycalphad.amkit.mobility import MobilityModel, MobilityDataError
from pycalphad.amkit.thermal import ThermalHistory, rosenthal_T, eagar_tsai_T, solidification_conditions

def test_solidification_service():
    db_path = str(files(pycalphad.tests.databases).joinpath("alzn_mey.tdb"))
    dbf = Database(db_path)
    comps = ['AL', 'ZN', 'VA']
    phases = ['LIQUID', 'FCC_A1']
    composition = {'ZN': 0.15}

    # Find liquidus temperature
    T_liq = find_liquidus_temperature(dbf, comps, phases, composition)
    assert 800.0 < T_liq < 950.0

    # Simulate solidification
    res = simulate_solidification(dbf, comps, phases, composition, mode='scheil', step=2.0)
    assert res.T_liquidus > 800.0
    assert res.T_solidus_scheil < 750.0
    assert isinstance(res.partial, bool)

def test_rhea_refractory_liquidus_temperature():
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../examples/databases/TiZrHfNb_RHEA.tdb'))
    dbf = Database(db_path)
    comps = ['TI', 'ZR', 'HF', 'NB', 'VA']
    phases = list(dbf.phases.keys())
    composition = {'ZR': 0.25, 'HF': 0.25, 'NB': 0.25}

    T_liq = find_liquidus_temperature(dbf, comps, phases, composition)
    assert T_liq > 2000.0
    assert T_liq != 2000.0


def test_mobility_service():
    # Load diffusion TDB containing HCP_A3 mobility parameters for Al-Mg
    db_path = str(files(pycalphad.tests.databases).joinpath("diffusion.tdb"))
    dbf = Database(db_path)
    comps = ['MG', 'AL', 'VA']
    phase = 'HCP_A3'

    # Verify instantiation and default database calculation
    mm = MobilityModel(dbf, comps, phase)
    T = 700.0
    x = {'AL': 0.05}
    
    d_trace = mm.tracer_diffusivity(T, x)
    assert 'MG' in d_trace and 'AL' in d_trace
    assert d_trace['MG'] > 0.0

    d_inter = mm.interdiffusivity_matrix(T, x)
    assert d_inter.shape == (1, 1)
    assert d_inter[0, 0] > 0.0

    # Test missing mobility parameter error trigger
    # alzn_mey.tdb does not have MQ/MF parameters, should raise MobilityDataError
    db_path_alzn = str(files(pycalphad.tests.databases).joinpath("alzn_mey.tdb"))
    dbf_alzn = Database(db_path_alzn)
    with pytest.raises(MobilityDataError):
        MobilityModel(dbf_alzn, ['AL', 'ZN', 'VA'], 'FCC_A1')

    # Test fallback with user constants (must not raise error, throws UserWarning)
    with pytest.warns(UserWarning):
        mm_fallback = MobilityModel(dbf_alzn, ['AL', 'ZN', 'VA'], 'FCC_A1', user_constants={'AL': 1e-15, 'ZN': 2e-15})
    d_trace_fb = mm_fallback.tracer_diffusivity(T, {'ZN': 0.1})
    assert d_trace_fb['AL'] == 1e-15
    assert d_trace_fb['ZN'] == 2e-15

def test_thermal_service():
    # Model parameters for steel
    density = 7800.0
    specific_heat = 500.0
    conductivity = 15.0
    P = 200.0
    v_val = 1.0

    # Test Rosenthal solution
    T_r = rosenthal_T(x=-1e-3, y=0.0, z=0.0, P=P, v=v_val, density=density, specific_heat=specific_heat, conductivity=conductivity)
    assert T_r > 298.15

    # Test Eagar-Tsai solution
    T_et = eagar_tsai_T(x=-1e-3, y=0.0, z=0.0, P=P, v=v_val, beam_radius=1e-4, density=density, specific_heat=specific_heat, conductivity=conductivity)
    assert T_et > 298.15

    # Test solidification conditions calculation
    T_func = lambda x, y, z: rosenthal_T(x, y, z, P, v_val, density, specific_heat, conductivity)
    G, R, cooling = solidification_conditions(T_func, -1e-3, 0.0, 0.0, v_val)
    assert G > 0.0
    assert abs(R) <= abs(v_val)
    assert cooling > 0.0

    # Test ThermalHistory methods
    time_history = np.linspace(0.0, 1.0, 101)
    temp_history = 300.0 + 1000.0 * np.exp(-10.0 * (time_history - 0.1)**2)
    history = ThermalHistory(time_history, temp_history)
    
    cool_rate = history.cooling_rate(T_ref=600.0)
    assert cool_rate > 0.0
    
    t_above = history.time_above(T_thresh=800.0)
    assert 0.0 < t_above < 1.0
