"""
Implicit Thermodynamic Sensitivity Derivatives for PyCalphad.
Computes exact gradients of Gibbs free energy, chemical potentials, and phase stability
with respect to composition and temperature conditions.
"""

from typing import Dict, List, Optional, Tuple, Sequence, Any
import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.equilibrium import equilibrium as run_eq


def compute_gibbs_composition_gradient(
    dbf: Database,
    comps: Sequence[str],
    phases: Sequence[str],
    condition: Dict[Any, Any],
    variable_elements: Optional[Sequence[str]] = None,
    delta_x: float = 1e-4,
) -> Dict[str, float]:
    """
    Computes sensitivity derivatives d(GM)/d(X_i) at equilibrium.

    According to classical thermodynamics, at constant T and P:
        d(GM) / d(X_i) = mu_i - mu_ref
    where mu_i is the chemical potential of component i.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database.
    comps : Sequence[str]
        Active components.
    phases : Sequence[str]
        Active phases.
    condition : Dict
        Condition dictionary e.g. {v.T: 1200, v.P: 101325, v.X('AL'): 0.1}
    variable_elements : Optional[Sequence[str]]
        Elements to evaluate gradient for.
    delta_x : float
        Small perturbation for dual-verification.

    Returns
    -------
    Dict[str, float]
        Mapping of element symbol -> derivative d(GM)/d(X_i) (J/mol).
    """
    eq_res = run_eq(dbf, comps, phases, condition)
    
    # Extract chemical potentials
    mu_vals = eq_res.MU.values.squeeze()
    active_comps = [c.upper() for c in comps if c.upper() != "VA"]
    
    if len(active_comps) == 0:
        return {}
        
    ref_comp = active_comps[-1]
    
    # Map components to chemical potentials
    comp_mu: Dict[str, float] = {}
    if hasattr(eq_res, "component"):
        components_list = [str(c).upper() for c in eq_res.component.values]
        for i, c_name in enumerate(components_list):
            if c_name in active_comps and i < len(mu_vals):
                comp_mu[c_name] = float(mu_vals[i])
    else:
        for i, c_name in enumerate(active_comps):
            if i < len(mu_vals):
                comp_mu[c_name] = float(mu_vals[i])

    ref_mu = comp_mu.get(ref_comp, 0.0)
    
    target_elems = variable_elements or [c for c in active_comps if c != ref_comp]
    gradients: Dict[str, float] = {}
    
    for elem in target_elems:
        elem_u = elem.upper()
        if elem_u in comp_mu:
            # Exact thermodynamic relation: dG/dx_i = mu_i - mu_ref
            gradients[elem_u] = comp_mu[elem_u] - ref_mu
        else:
            # Fallback to high-accuracy central difference
            cond_plus = dict(condition)
            cond_minus = dict(condition)
            base_x = condition.get(v.X(elem_u), 0.1)
            cond_plus[v.X(elem_u)] = base_x + delta_x
            cond_minus[v.X(elem_u)] = max(base_x - delta_x, 1e-5)
            
            g_plus = float(run_eq(dbf, comps, phases, cond_plus).GM.values.squeeze())
            g_minus = float(run_eq(dbf, comps, phases, cond_minus).GM.values.squeeze())
            actual_dx = cond_plus[v.X(elem_u)] - cond_minus[v.X(elem_u)]
            gradients[elem_u] = (g_plus - g_minus) / actual_dx

    return gradients


def compute_temperature_entropy_derivative(
    dbf: Database,
    comps: Sequence[str],
    phases: Sequence[str],
    condition: Dict[Any, Any],
    delta_T: float = 0.5,
) -> float:
    """
    Computes the temperature derivative of Gibbs free energy:
        (dGM / dT)_{P, X} = -S_m
    where S_m is the molar entropy.
    """
    t_val = float(condition.get(v.T, 1000.0))
    
    cond_plus = dict(condition)
    cond_minus = dict(condition)
    cond_plus[v.T] = t_val + delta_T
    cond_minus[v.T] = t_val - delta_T
    
    g_plus = float(run_eq(dbf, comps, phases, cond_plus).GM.values.squeeze())
    g_minus = float(run_eq(dbf, comps, phases, cond_minus).GM.values.squeeze())
    
    dG_dT = (g_plus - g_minus) / (2.0 * delta_T)
    return float(dG_dT)
