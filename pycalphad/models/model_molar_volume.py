"""
Molar Volume and Thermal Expansion Model Framework.

Implements standard CALPHAD molar volume, thermal expansivity, and density
evaluation for solution phases and stoichiometric compounds based on the
Lu-Selleby-Sundman and standard CALPHAD excess volume formalisms.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from symengine import Add, Expr, Piecewise, S, Symbol, exp, sympify
from tinydb import where

from pycalphad.model import Model, _extend_ordered_if_subset_of_disorder
from pycalphad.variables import Species
import pycalphad.variables as v


class MolarVolumeModel(Model):
    """
    Dedicated Molar Volume Model extending Model with explicit
    molar volume (V_m), thermal expansion coefficient (CTE), and density (rho).
    """

    def __init__(self, dbe, comps, phase_name, parameters=None):
        super().__init__(dbe, comps, phase_name, parameters=parameters)
        self.build_molar_volume(dbe)

    def build_molar_volume(self, dbe):
        """Construct symbolic expressions for V0, VA, VM, CTE, and density."""
        phase = dbe.phases[self.phase_name]
        if phase.model_hints.get('ordered_phase', False):
            phase = _extend_ordered_if_subset_of_disorder(dbe, self.components, phase)
        param_search = dbe.search

        # 1. Search for V0 parameters (STP reference molar volume)
        v0_query = (
            (where('phase_name') == phase.name) &
            (where('parameter_type') == 'V0') &
            (where('constituent_array').test(self._array_validity))
        )
        # 2. Search for VM parameters (direct molar volume)
        vm_query = (
            (where('phase_name') == phase.name) &
            (where('parameter_type') == 'VM') &
            (where('constituent_array').test(self._array_validity))
        )
        # 3. Search for VA parameters (integrated thermal expansivity)
        va_query = (
            (where('phase_name') == phase.name) &
            (where('parameter_type') == 'VA') &
            (where('constituent_array').test(self._array_validity))
        )
        # 4. Search for VB parameters (thermal expansion coefficient)
        vb_query = (
            (where('phase_name') == phase.name) &
            (where('parameter_type') == 'VB') &
            (where('constituent_array').test(self._array_validity))
        )
        # 5. Search for VK parameters (isothermal compressibility)
        vk_query = (
            (where('phase_name') == phase.name) &
            (where('parameter_type') == 'VK') &
            (where('constituent_array').test(self._array_validity))
        )

        v0_expr = self.symbol_replace(
            self.redlich_kister_sum(phase, param_search, v0_query) / self._site_ratio_normalization,
            self._symbols
        )
        vm_direct_expr = self.symbol_replace(
            self.redlich_kister_sum(phase, param_search, vm_query) / self._site_ratio_normalization,
            self._symbols
        )
        va_expr = self.symbol_replace(
            self.redlich_kister_sum(phase, param_search, va_query),
            self._symbols
        )
        vb_expr = self.symbol_replace(
            self.redlich_kister_sum(phase, param_search, vb_query),
            self._symbols
        )
        vk_expr = self.symbol_replace(
            self.redlich_kister_sum(phase, param_search, vk_query),
            self._symbols
        )

        self.V0 = v0_expr
        self.VA = va_expr
        self.VB = vb_expr
        self.VK = vk_expr

        # Compute standard molar volume
        if vm_direct_expr != S.Zero:
            self.VM = vm_direct_expr
        elif v0_expr != S.Zero:
            if va_expr != S.Zero:
                self.VM = v0_expr * exp(va_expr)
            elif vb_expr != S.Zero:
                self.VM = v0_expr * (1.0 + vb_expr * (v.T - 298.15))
            else:
                self.VM = v0_expr
        else:
            self.VM = S.Zero

        # CTE = (1 / V_m) * d(V_m)/dT
        if self.VM != S.Zero:
            self.CTE = self.VM.diff(v.T) / self.VM
        else:
            self.CTE = S.Zero


def compute_molar_volume_for_phase(
    dbe,
    comps: List[str],
    phase_name: str,
    temperature: float,
    pressure: float = 101325.0,
    site_fractions: Optional[Dict[Any, float]] = None
) -> float:
    """
    Evaluate the molar volume (in m^3/mol) of a phase at given T, P, and site fractions.
    """
    model = MolarVolumeModel(dbe, comps, phase_name)
    if model.VM == S.Zero:
        return 0.0

    subs_dict = {v.T: float(temperature), v.P: float(pressure)}
    
    # Populate site fractions if not explicitly passed
    if site_fractions is not None:
        for k, val in site_fractions.items():
            subs_dict[k] = float(val)
    else:
        for idx, subl in enumerate(model.constituents):
            subl_len = max(len(subl), 1)
            for sp in subl:
                sf_var = v.SiteFraction(model.phase_name, idx, sp)
                subs_dict[sf_var] = 1.0 / subl_len

    vm_val = model.VM.xreplace(subs_dict)
    return float(vm_val)
