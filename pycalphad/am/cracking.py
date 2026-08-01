import numpy as np
from pycalphad.amkit.solidification import simulate_solidification

def calculate_clyne_davis_index(temperatures, fraction_solid):
    """
    Calculate the Clyne-Davis Crack Susceptibility Coefficient (CSC).
    
    CSC = (T_0.90 - T_0.99) / (T_0.40 - T_0.90)
    
    Parameters
    ----------
    temperatures : array-like
        solidification temperature profile in Kelvin or Celsius.
    fraction_solid : array-like
        fraction of solid formed, must be in the range [0, 1].
        
    Returns
    -------
    float
        The Clyne-Davis CSC value.
    """
    temperatures = np.array(temperatures)
    fraction_solid = np.array(fraction_solid)
    
    # Sort by fraction_solid to ensure interpolation works correctly
    sort_idx = np.argsort(fraction_solid)
    f_s = fraction_solid[sort_idx]
    t = temperatures[sort_idx]
    
    # Interpolate temperatures at specific solid fractions
    t_40 = np.interp(0.40, f_s, t)
    t_90 = np.interp(0.90, f_s, t)
    t_99 = np.interp(0.99, f_s, t)
    
    numerator = t_90 - t_99
    denominator = t_40 - t_90
    
    if denominator == 0:
        return np.inf
        
    return abs(numerator / denominator)

def calculate_kou_index(temperatures, fraction_solid):
    """
    Calculate the Kou Solidification Cracking Index (CSI).
    
    CSI = max | dT / d(f_S^0.5) | for 0.90 <= f_S <= 0.99
    
    Parameters
    ----------
    temperatures : array-like
        solidification temperature profile in Kelvin or Celsius.
    fraction_solid : array-like
        fraction of solid formed, must be in the range [0, 1].
        
    Returns
    -------
    float
        The Kou CSI value.
    """
    temperatures = np.array(temperatures)
    fraction_solid = np.array(fraction_solid)
    
    # Sort to ensure monotonicity
    sort_idx = np.argsort(fraction_solid)
    f_s = fraction_solid[sort_idx]
    t = temperatures[sort_idx]
    
    y = np.sqrt(f_s)
    
    # Calculate finite differences
    dy = np.diff(y)
    dt = np.diff(t)
    
    # Avoid division by zero
    valid = (dy > 1e-9)
    if not np.any(valid):
        return 0.0
        
    deriv = np.zeros_like(dy)
    deriv[valid] = dt[valid] / dy[valid]
    
    # Associate each derivative with the midpoint of the f_S interval
    f_s_mid = 0.5 * (f_s[:-1] + f_s[1:])
    
    # Filter for the vulnerable range: 0.90 <= f_S <= 0.99
    mask = (f_s_mid >= 0.90) & (f_s_mid <= 0.99)
    
    if not np.any(mask):
        # Fallback to the closest point near 0.90-0.99 if points are sparse
        closest_idx = np.argmin(np.abs(f_s_mid - 0.95))
        return abs(deriv[closest_idx])
        
    return float(np.max(np.abs(deriv[mask])))

def calculate_rdg_index(temperatures, fraction_solid):
    """
    Calculate the simplified Rappaz-Drezet-Gremaud (RDG) hot-tearing index.
    
    RDG = integral from T(f_s=0.99) to T(f_s=0.90) of [ f_s^2 / (1 - f_s)^3 ] dT
    """
    t = np.array(temperatures)
    f_s = np.array(fraction_solid)
    
    # Filter for vulnerable zone (0.90 <= f_s <= 0.99)
    mask = (f_s >= 0.90) & (f_s <= 0.99)
    if np.sum(mask) < 2:
        return 0.0
        
    t_vul = t[mask]
    f_s_vul = f_s[mask]
    
    # Calculate integrand, clip (1 - f_s) to avoid division by zero
    integrand = f_s_vul**2 / np.maximum(1.0 - f_s_vul, 1e-4)**3
    
    # Sort by temperature to integrate properly
    sort_idx = np.argsort(t_vul)
    t_sort = t_vul[sort_idx]
    int_sort = integrand[sort_idx]
    
    from scipy.integrate import trapezoid
    return float(trapezoid(int_sort, t_sort))

def susceptibility_from_composition(dbf, comps, phases, composition, indices=('kou', 'csc', 'freezing_range', 'tfr', 'rdg'), step=1.0, T_high=2000.0, T_low=300.0):
    """
    Perform a Scheil solidification simulation and compute requested cracking susceptibility indices.
    
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
    indices : tuple of str
        The indices to calculate ('kou', 'csc', 'freezing_range', 'tfr', 'rdg').
    step : float
        Temperature step size.
    T_high : float
        Upper temperature search bound for liquidus determination.
    T_low : float
        Lower temperature search bound for liquidus determination.
        
    Returns
    -------
    indices_dict : dict
        Calculated cracking indices.
    res : SolidificationResult
        Full solidification results.
    """
    res = simulate_solidification(dbf, comps, phases, composition, mode='scheil', step=step, T_high=T_high, T_low=T_low)
    
    results = {}
    temps = res.temperatures
    f_s = res.fraction_solid
    
    if 'csc' in indices:
        results['csc'] = calculate_clyne_davis_index(temps, f_s)
    if 'kou' in indices:
        results['kou'] = calculate_kou_index(temps, f_s)
    if 'freezing_range' in indices:
        results['freezing_range'] = res.T_liquidus - res.T_solidus_scheil
    if 'tfr' in indices:
        t_90 = np.interp(0.90, f_s, temps)
        t_98 = np.interp(0.98, f_s, temps)
        results['tfr'] = t_90 - t_98
    if 'rdg' in indices:
        results['rdg'] = calculate_rdg_index(temps, f_s)
        
    return results, res

