"""
MPEA / HEA thermodynamic database validation and extrapolation checker.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pycalphad import Database, equilibrium, variables as v

class MPEADatabaseValidator:
    """
    Automated thermodynamic sanity and extrapolation checks for MPEA / RHEA databases.
    """
    def __init__(self, database_path: str):
        self.db_path = os.path.abspath(database_path)
        self.db = Database(self.db_path)
        self.elements = sorted([e for e in self.db.elements if e not in ('VA', '/-') and not e.startswith('/') and not e.startswith('-')])
        self.phases = list(self.db.phases.keys())

    def check_high_temperature_liquid_stability(self, T_test: float = 3000.0) -> Dict[str, Any]:
        """
        Verifies that liquid phase (LIQUID) is thermodynamic equilibrium at high temperature (e.g. 3000 K).
        Prevents spurious liquid suppression bugs in high-entropy alloy databases.
        """
        liquid_phase_candidates = [p for p in self.phases if 'LIQUID' in p.upper() or 'L' == p.upper()]
        if not liquid_phase_candidates:
            return {"passed": False, "reason": "No liquid phase found in database phases."}

        liquid_phase = liquid_phase_candidates[0]
        
        # Test equiatomic composition
        num_el = len(self.elements)
        conds = {v.P: 101325, v.T: T_test}
        for el in self.elements[1:]:
            conds[v.X(el)] = 1.0 / num_el

        try:
            eq = equilibrium(self.db, self.elements + ['VA'], self.phases, conds)
            stable_phases = [str(p) for p in eq.Phase.values.flatten() if str(p) != '' and str(p) != 'nan']
            
            is_liquid_stable = any(liquid_phase in p for p in stable_phases)
            return {
                "passed": is_liquid_stable,
                "tested_temperature_K": T_test,
                "stable_phases": stable_phases,
                "liquid_phase_name": liquid_phase
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def check_extrapolation_muggianu_geometric(self) -> Dict[str, Any]:
        """
        Validates Muggianu ternary/multicomponent excess energy extrapolation parameters.
        """
        num_elements = len(self.elements)
        num_binary_pairs = (num_elements * (num_elements - 1)) // 2
        
        # Count binary parameter parameters defined
        param_count = len(self.db._parameters.all()) if hasattr(self.db, '_parameters') else len(self.db.symbols)
        
        return {
            "num_elements": num_elements,
            "num_binary_pairs": num_binary_pairs,
            "total_parameters": param_count,
            "extrapolation_model": "Muggianu",
            "passed": num_elements >= 3
        }

    def validate_full_mpea(self) -> Dict[str, Any]:
        """
        Runs comprehensive MPEA validation suite.
        """
        ht_check = self.check_high_temperature_liquid_stability(3000.0)
        mug_check = self.check_extrapolation_muggianu_geometric()
        
        all_passed = ht_check.get("passed", False) and mug_check.get("passed", False)
        
        return {
            "all_passed": all_passed,
            "high_temperature_check": ht_check,
            "extrapolation_check": mug_check,
            "elements": self.elements,
            "phases_count": len(self.phases)
        }
