import numpy as np
from pycalphad import equilibrium, variables as v

def find_liquidus_temperature(dbf, comps, phases, composition, liquid_phase_name='LIQUID'):
    """
    Find the liquidus temperature where the first solid phase becomes stable.
    """
    # Standardize composition keys
    std_comp = {}
    for k, val in composition.items():
        if isinstance(k, str):
            std_comp[v.X(k)] = val
        else:
            std_comp[k] = val

    # Binary search range for temperature
    T_high = 2000.0
    T_low = 300.0

    # Binary search to find the highest temperature where non-liquid is stable
    for _ in range(15):
        T_mid = 0.5 * (T_high + T_low)
        eq = equilibrium(dbf, comps, phases, {v.T: T_mid, v.P: 101325.0, **std_comp})
        stable_phases = [p for p in eq.Phase.values.flatten() if p != '']
        solid_stable = any(p != liquid_phase_name for p in stable_phases)

        if solid_stable:
            T_low = T_mid
        else:
            T_high = T_mid

    return T_high

def simulate_solidification(dbf, comps, phases, composition, mode='scheil', step=1.0, start_temperature=None, liquid_phase_name='LIQUID'):
    """
    Simulate solidification using Scheil-Gulliver or lever-rule equilibrium cooling.
    
    Parameters
    ----------
    dbf : Database
        Thermodynamic database.
    comps : list
        Active components.
    phases : list
        Active phases.
    composition : dict
        Dict mapping element names (str) or variables (v.X) to mole fractions.
    mode : {'scheil', 'equilibrium'}
        Solidification mode.
    step : float
        Temperature step size (Kelvin).
    start_temperature : float, optional
        Start temperature (Kelvin). If None, calculated automatically.
    liquid_phase_name : str
        Name of the liquid phase.
        
    Returns
    -------
    SolidificationResult
    """
    # Standardize composition keys
    std_comp = {}
    for k, val in composition.items():
        if isinstance(k, str):
            std_comp[v.X(k)] = val
        else:
            std_comp[k] = val

    # Determine starting temperature
    if start_temperature is None:
        T_liq = find_liquidus_temperature(dbf, comps, phases, std_comp, liquid_phase_name)
        start_temperature = T_liq + 10.0

    if mode == 'scheil':
        from scheil import simulate_scheil_solidification
        res = simulate_scheil_solidification(
            dbf, comps, phases, std_comp, start_temperature, 
            step_temperature=step, liquid_phase_name=liquid_phase_name
        )
    elif mode == 'equilibrium':
        from scheil import simulate_equilibrium_solidification
        res = simulate_equilibrium_solidification(
            dbf, comps, phases, std_comp, start_temperature, 
            step_temperature=step, liquid_phase_name=liquid_phase_name
        )
    else:
        raise ValueError(f"Unknown solidification mode: {mode}")

    # Add attributes required by specification
    idx = np.where(np.array(res.fraction_solid) > 0.0)[0]
    res.T_liquidus = res.temperatures[idx[0]] if len(idx) > 0 else res.temperatures[0]
    res.T_solidus_scheil = res.temperatures[-1]
    res.partial = not res.converged

    return res
