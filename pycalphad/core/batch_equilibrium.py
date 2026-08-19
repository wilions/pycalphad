"""
Batch equilibrium calculation routines for PyCalphad.
Provides high-throughput multi-condition and multi-composition equilibrium calculations.
"""

import concurrent.futures
from collections.abc import Iterable, Mapping
from typing import List, Dict, Any, Optional, Sequence
import numpy as np
import xarray as xr

from pycalphad.core.equilibrium import equilibrium
from pycalphad.io.database import Database
import pycalphad.variables as v


def batch_equilibrium(
    dbf: Database,
    comps: Sequence[str],
    phases: Sequence[str],
    conditions_list: Sequence[Mapping[Any, Any]],
    output: Optional[Sequence[str]] = None,
    model: Optional[Any] = None,
    max_workers: Optional[int] = None,
    parameters: Optional[Dict[Any, Any]] = None,
    verbose: bool = False,
    **kwargs
) -> List[xr.Dataset]:
    """
    Calculate equilibrium across a list of distinct conditions concurrently or sequentially.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database.
    comps : Sequence[str]
        Active components.
    phases : Sequence[str]
        Active phases.
    conditions_list : Sequence[Mapping[Any, Any]]
        List of condition dictionaries, e.g. [{v.T: 1000, v.P: 101325, v.X('AL'): 0.1}, ...]
    output : Optional[Sequence[str]]
        Additional properties to compute.
    model : Optional[Any]
        Phase models.
    max_workers : Optional[int]
        Number of parallel workers. Defaults to sequential/threaded pool.
    parameters : Optional[Dict]
        Parameter overrides.
    verbose : bool
        Verbosity flag.

    Returns
    -------
    List[xr.Dataset]
        List of equilibrium calculation Datasets corresponding to each condition in conditions_list.
    """
    if not conditions_list:
        return []

    def _solve_single(conds: Mapping[Any, Any]) -> xr.Dataset:
        return equilibrium(
            dbf=dbf,
            comps=comps,
            phases=phases,
            conditions=conds,
            output=output,
            model=model,
            parameters=parameters,
            verbose=verbose,
            **kwargs
        )

    # For small batch sizes or single worker, run sequentially to avoid thread context overhead
    if max_workers == 1 or len(conditions_list) == 1:
        return [_solve_single(c) for c in conditions_list]

    results: List[Optional[xr.Dataset]] = [None] * len(conditions_list)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_solve_single, conds): idx
            for idx, conds in enumerate(conditions_list)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

    return results  # type: ignore[return-value]


def evaluate_composition_grid(
    dbf: Database,
    comps: Sequence[str],
    phases: Sequence[str],
    temperatures: Sequence[float],
    compositions: Sequence[Mapping[str, float]],
    pressure: float = 101325.0,
    output: Optional[Sequence[str]] = None,
    max_workers: Optional[int] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Evaluates equilibrium across a grid of compositions and temperatures, returning
    compact structured summaries of stable phases, phase fractions, and Gibbs energies.
    """
    # Identify non-vacant elements
    pure_elements = sorted([c.upper() for c in comps if c.upper() != "VA"])
    if not pure_elements:
        return []
    dependent_element = pure_elements[-1]

    conditions_list = []
    for comp in compositions:
        for temp in temperatures:
            cond: Dict[Any, Any] = {v.T: float(temp), v.P: float(pressure)}
            for elem, frac in comp.items():
                elem_u = elem.upper()
                if elem_u != "VA" and elem_u != dependent_element and elem_u in pure_elements:
                    cond[v.X(elem_u)] = float(frac)
            conditions_list.append((comp, temp, cond))

    raw_conds = [c[2] for c in conditions_list]
    eq_datasets = batch_equilibrium(
        dbf=dbf,
        comps=comps,
        phases=phases,
        conditions_list=raw_conds,
        output=output,
        max_workers=max_workers,
        **kwargs
    )

    summaries = []
    for (comp, temp, _), ds in zip(conditions_list, eq_datasets):
        phases_arr = ds.Phase.values.squeeze()
        fractions_arr = ds.NP.values.squeeze()
        gm_val = float(ds.GM.values.squeeze()) if "GM" in ds else 0.0

        if not isinstance(phases_arr, np.ndarray) or phases_arr.ndim == 0:
            phases_arr = np.array([phases_arr])
            fractions_arr = np.array([fractions_arr])

        phase_dict: Dict[str, float] = {}
        for p, f in zip(phases_arr.flat, fractions_arr.flat):
            p_str = str(p).strip()
            if p_str and p_str != "" and not np.isnan(f) and f > 1e-6:
                phase_dict[p_str] = float(f)

        summaries.append({
            "composition": comp,
            "temperature_K": temp,
            "temperature_C": temp - 273.15,
            "stable_phases": list(phase_dict.keys()),
            "phase_fractions": phase_dict,
            "gibbs_energy": gm_val,
        })

    return summaries
