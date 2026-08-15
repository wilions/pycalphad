"""
Solidification Cracking and Hot-Tearing Susceptibility Criteria for Additive Manufacturing.

Implements quantitative hot tearing criteria:
1. Kou Solidification Cracking Index (CSI): max |dT / d(f_S^0.5)| in vulnerable mushy regime (0.90 <= f_S <= 0.99).
2. Clyne-Davies Crack Susceptibility Coefficient (CSC): (T_0.90 - T_0.99) / (T_0.40 - T_0.90).
3. Rappaz-Drezet-Gremaud (RDG) hot-tearing criterion based on mushy zone permeability and pressure drop.
4. Total Freezing Range (TFR) and Terminal Freezing Range (T_0.90 - T_0.98).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pycalphad.amkit.solidification import simulate_solidification


@dataclass
class SolidificationCrackingAssessment:
    """
    Comprehensive Solidification Cracking Assessment for an alloy.
    """
    kou_index: float
    clyne_davis_index: float
    rdg_index: float
    freezing_range: float
    terminal_freezing_range: float
    risk_category: str
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "kou_index": self.kou_index,
            "clyne_davis_index": self.clyne_davis_index,
            "rdg_index": self.rdg_index,
            "freezing_range_K": self.freezing_range,
            "terminal_freezing_range_K": self.terminal_freezing_range,
            "risk_category": self.risk_category,
        }


def calculate_clyne_davis_index(temperatures: Any, fraction_solid: Any) -> float:
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
    temperatures = np.array(temperatures, dtype=float)
    fraction_solid = np.array(fraction_solid, dtype=float)
    
    if len(fraction_solid) < 2:
        return 0.0

    # Sort by fraction_solid to ensure monotonic interpolation
    sort_idx = np.argsort(fraction_solid)
    f_s = fraction_solid[sort_idx]
    t = temperatures[sort_idx]
    
    # Interpolate temperatures at specific solid fractions
    t_40 = float(np.interp(0.40, f_s, t))
    t_90 = float(np.interp(0.90, f_s, t))
    t_99 = float(np.interp(0.99, f_s, t))
    
    numerator = t_90 - t_99
    denominator = t_40 - t_90
    
    if abs(denominator) < 1e-6:
        return float(np.inf) if abs(numerator) > 1e-6 else 0.0
        
    return abs(float(numerator / denominator))


def calculate_kou_index(
    temperatures: Any,
    fraction_solid: Any,
    vulnerable_range: Tuple[float, float] = (0.90, 0.99)
) -> float:
    """
    Calculate the Kou Solidification Cracking Index (CSI).
    
    CSI = max | dT / d(f_S^0.5) | for f_S in vulnerable_range (default 0.90 <= f_S <= 0.99)
    
    Parameters
    ----------
    temperatures : array-like
        solidification temperature profile in Kelvin or Celsius.
    fraction_solid : array-like
        fraction of solid formed, must be in the range [0, 1].
    vulnerable_range : tuple of (float, float)
        bounds for the vulnerable solid fraction regime.
        
    Returns
    -------
    float
        The Kou CSI value (|K / sqrt(fraction)|).
    """
    temperatures = np.array(temperatures, dtype=float)
    fraction_solid = np.array(fraction_solid, dtype=float)
    
    if len(fraction_solid) < 2:
        return 0.0

    # Sort to ensure monotonicity
    sort_idx = np.argsort(fraction_solid)
    f_s = fraction_solid[sort_idx]
    t = temperatures[sort_idx]
    
    y = np.sqrt(np.clip(f_s, 0.0, 1.0))
    
    # Calculate finite differences
    dy = np.diff(y)
    dt = np.diff(t)
    
    valid = (dy > 1e-9)
    if not np.any(valid):
        return 0.0
        
    deriv = np.zeros_like(dy)
    deriv[valid] = dt[valid] / dy[valid]
    
    # Associate each derivative with the midpoint of the f_S interval
    f_s_mid = 0.5 * (f_s[:-1] + f_s[1:])
    
    low_bnd, high_bnd = vulnerable_range
    mask = (f_s_mid >= low_bnd) & (f_s_mid <= high_bnd)
    
    if not np.any(mask):
        # Fallback to the closest point near average of vulnerable range
        target = 0.5 * (low_bnd + high_bnd)
        closest_idx = int(np.argmin(np.abs(f_s_mid - target)))
        return abs(float(deriv[closest_idx]))
        
    return float(np.max(np.abs(deriv[mask])))


def calculate_rdg_index(temperatures: Any, fraction_solid: Any) -> float:
    """
    Calculate the simplified Rappaz-Drezet-Gremaud (RDG) hot-tearing index.
    
    RDG = integral from T(f_s=0.99) to T(f_s=0.90) of [ f_s^2 / (1 - f_s)^3 ] dT
    """
    t = np.array(temperatures, dtype=float)
    f_s = np.array(fraction_solid, dtype=float)
    
    if len(f_s) < 2:
        return 0.0

    # Filter for vulnerable zone (0.90 <= f_s <= 0.99)
    mask = (f_s >= 0.90) & (f_s <= 0.99)
    if np.sum(mask) < 2:
        # Fallback with wider interpolation if points are sparse
        t_interp = np.linspace(float(np.interp(0.99, f_s, t)), float(np.interp(0.90, f_s, t)), 25)
        f_s_interp = np.interp(t_interp, t[::-1], f_s[::-1]) if t[0] > t[-1] else np.interp(t_interp, t, f_s)
        integrand = f_s_interp**2 / np.maximum(1.0 - f_s_interp, 1e-4)**3
        from scipy.integrate import trapezoid
        return abs(float(trapezoid(integrand, t_interp)))
        
    t_vul = t[mask]
    f_s_vul = f_s[mask]
    
    integrand = f_s_vul**2 / np.maximum(1.0 - f_s_vul, 1e-4)**3
    
    sort_idx = np.argsort(t_vul)
    t_sort = t_vul[sort_idx]
    int_sort = integrand[sort_idx]
    
    from scipy.integrate import trapezoid
    return float(trapezoid(int_sort, t_sort))


def calculate_freezing_range(temperatures: Any, fraction_solid: Any) -> float:
    """Calculate the total solidification freezing range (T_liquidus - T_solidus)."""
    t = np.array(temperatures, dtype=float)
    f_s = np.array(fraction_solid, dtype=float)
    if len(t) < 2:
        return 0.0
    t_liq = float(np.interp(0.01, f_s, t))
    t_sol = float(np.interp(0.99, f_s, t))
    return abs(t_liq - t_sol)


def calculate_terminal_freezing_range(
    temperatures: Any,
    fraction_solid: Any,
    f_s_start: float = 0.90,
    f_s_end: float = 0.98
) -> float:
    """Calculate the terminal freezing range (T(f_s_start) - T(f_s_end))."""
    t = np.array(temperatures, dtype=float)
    f_s = np.array(fraction_solid, dtype=float)
    if len(t) < 2:
        return 0.0
    t_start = float(np.interp(f_s_start, f_s, t))
    t_end = float(np.interp(f_s_end, f_s, t))
    return abs(t_start - t_end)


def classify_cracking_risk(kou_val: float, csc_val: float) -> str:
    """Classify solidification cracking risk based on Kou and Clyne-Davies indices."""
    if kou_val < 150.0 and csc_val < 0.5:
        return "Low"
    elif kou_val < 350.0 and csc_val < 1.2:
        return "Moderate"
    elif kou_val < 600.0 or csc_val < 2.5:
        return "High"
    else:
        return "Severe"


def evaluate_cracking_susceptibility(
    temperatures: Any,
    fraction_solid: Any
) -> SolidificationCrackingAssessment:
    """
    Perform a complete multi-index solidification cracking assessment.
    """
    kou = calculate_kou_index(temperatures, fraction_solid)
    csc = calculate_clyne_davis_index(temperatures, fraction_solid)
    rdg = calculate_rdg_index(temperatures, fraction_solid)
    fr = calculate_freezing_range(temperatures, fraction_solid)
    tfr = calculate_terminal_freezing_range(temperatures, fraction_solid)
    risk = classify_cracking_risk(kou, csc)

    return SolidificationCrackingAssessment(
        kou_index=kou,
        clyne_davis_index=csc,
        rdg_index=rdg,
        freezing_range=fr,
        terminal_freezing_range=tfr,
        risk_category=risk,
    )


def susceptibility_from_composition(
    dbf: Any,
    comps: List[str],
    phases: List[str],
    composition: Dict[Any, float],
    indices: Tuple[str, ...] = ('kou', 'csc', 'freezing_range', 'tfr', 'rdg'),
    step: float = 1.0,
    T_high: float = 2000.0,
    T_low: float = 300.0
) -> Tuple[Dict[str, Any], Any]:
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
        Calculated cracking indices and risk rating.
    res : SolidificationResult
        Full solidification results.
    """
    res = simulate_solidification(
        dbf, comps, phases, composition, mode='scheil', step=step, T_high=T_high, T_low=T_low
    )
    
    results: Dict[str, Any] = {}
    temps = res.temperatures
    f_s = res.fraction_solid
    
    if 'csc' in indices:
        results['csc'] = calculate_clyne_davis_index(temps, f_s)
    if 'kou' in indices:
        results['kou'] = calculate_kou_index(temps, f_s)
    if 'freezing_range' in indices:
        results['freezing_range'] = calculate_freezing_range(temps, f_s)
    if 'tfr' in indices:
        results['tfr'] = calculate_terminal_freezing_range(temps, f_s)
    if 'rdg' in indices:
        results['rdg'] = calculate_rdg_index(temps, f_s)
        
    kou_val = results.get('kou', calculate_kou_index(temps, f_s))
    csc_val = results.get('csc', calculate_clyne_davis_index(temps, f_s))
    results['risk_category'] = classify_cracking_risk(kou_val, csc_val)

    return results, res
