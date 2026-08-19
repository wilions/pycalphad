"""
Automated Sublattice & Compound Energy Formalism (CEF) Model Architect for tdb-forge.
Generates canonical phase models, sublattice site ratios, and unassessed database templates.
"""

from typing import Dict, List, Any, Optional, Tuple, Sequence
import json
import os
from dataclasses import dataclass, field
from pycalphad import Database, variables as v
from pycalphad.io.tdb import write_tdb
from symengine import Symbol


@dataclass
class PhaseModelDefinition:
    name: str
    sublattice_site_ratios: List[float]
    sublattice_occupancies: List[List[str]]
    model_hints: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name.upper(),
            "sublattice_site_ratios": self.sublattice_site_ratios,
            "sublattice_occupancies": [[c.upper() for c in occ] for occ in self.sublattice_occupancies],
            "model_hints": self.model_hints,
        }


# Standard Canonical Prototype Sublattice Configurations
PROTOTYPE_PHASES: Dict[str, Dict[str, Any]] = {
    "LIQUID": {
        "site_ratios": [1.0],
        "has_interstitial_va": False,
        "model_hints": {},
    },
    "FCC_A1": {
        "site_ratios": [1.0, 1.0],
        "has_interstitial_va": True,
        "model_hints": {"ordered_phase": "FCC_A1"},
    },
    "BCC_A2": {
        "site_ratios": [1.0, 3.0],
        "has_interstitial_va": True,
        "model_hints": {"ordered_phase": "BCC_A2"},
    },
    "HCP_A3": {
        "site_ratios": [1.0, 0.5],
        "has_interstitial_va": True,
        "model_hints": {"ordered_phase": "HCP_A3"},
    },
    "GAMMA_PRIME": {
        "site_ratios": [3.0, 1.0, 1.0],
        "has_interstitial_va": True,
        "model_hints": {"ordered_phase": "GAMMA_PRIME"},
    },
    "BCC_B2": {
        "site_ratios": [0.5, 0.5, 3.0],
        "has_interstitial_va": True,
        "model_hints": {"ordered_phase": "BCC_B2"},
    },
    "LAVES_C14": {
        "site_ratios": [2.0, 1.0],
        "has_interstitial_va": False,
        "model_hints": {},
    },
    "SIGMA": {
        "site_ratios": [10.0, 4.0, 16.0],
        "has_interstitial_va": False,
        "model_hints": {},
    },
}


class ModelArchitect:
    """
    Automated thermodynamic model architect for multicomponent systems.
    """

    def __init__(self, unaries_tdb_path: Optional[str] = None):
        if unaries_tdb_path and os.path.exists(unaries_tdb_path):
            self.unaries_path = unaries_tdb_path
        else:
            self.unaries_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../unaries/SGTE_pure_elements.tdb")
            )

    def generate_system_phase_models(
        self,
        elements: Sequence[str],
        include_phases: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generates an ESPEI-compliant phase_models.json structure for the given elements.
        """
        canon_elements = sorted([e.upper() for e in elements if e.upper() != "VA"])
        target_phases = include_phases or ["LIQUID", "FCC_A1", "BCC_A2", "HCP_A3"]

        phase_dict: Dict[str, Any] = {}

        for p_name in target_phases:
            p_upper = p_name.upper()
            proto = PROTOTYPE_PHASES.get(p_upper, PROTOTYPE_PHASES["LIQUID"])

            sublattice_occupancies = []
            if proto["has_interstitial_va"]:
                # First sublattice: substitutional elements
                sublattice_occupancies.append(list(canon_elements))
                # Second sublattice: interstitial elements or vacancy
                sublattice_occupancies.append(["VA"])
                if len(proto["site_ratios"]) == 3:
                    # E.g. ordered sublattices + interstitial vacancy
                    sublattice_occupancies = [list(canon_elements), list(canon_elements), ["VA"]]
            else:
                for _ in range(len(proto["site_ratios"])):
                    sublattice_occupancies.append(list(canon_elements))

            phase_dict[p_upper] = {
                "sublattice_model": {
                    "sublattice_site_ratios": proto["site_ratios"],
                    "sublattice_occupancies": sublattice_occupancies,
                },
                "aliases": [p_upper, p_upper.lower()],
            }

        return {
            "components": sorted(list(canon_elements) + ["VA"]),
            "phases": phase_dict,
        }

    def generate_unassessed_base_tdb(
        self,
        elements: Sequence[str],
        phases: Sequence[str],
        output_tdb_path: str,
    ) -> Database:
        """
        Creates a clean base .tdb file with pure element unaries anchored and
        phases initialized without excess parameters.
        """
        canon_elements = sorted([e.upper() for e in elements if e.upper() != "VA"])
        db = Database()

        # Load unary functions if available
        unary_db = None
        if os.path.exists(self.unaries_path):
            try:
                unary_db = Database(self.unaries_path)
            except Exception:
                pass

        # 1. Add elements and reference states
        for el in canon_elements:
            db.elements.add(el)
            db.species.add(v.Species(el))
            if unary_db and el in unary_db.refstates:
                db.refstates[el] = unary_db.refstates[el]
            else:
                db.refstates[el] = {"phase": "FCC_A1", "mass": 50.0, "H298": 0.0, "S298": 30.0}

            ghser_key = f"GHSER{el}"
            if unary_db and ghser_key in unary_db.symbols:
                db.symbols[ghser_key] = unary_db.symbols[ghser_key]
            else:
                # Add default placeholder reference function
                db.symbols[ghser_key] = -5000.0 + 100.0 * v.T - 25.0 * v.T * Symbol(f"log(T)")

        db.elements.add("VA")
        db.species.add(v.Species("VA"))
        db.refstates["VA"] = {"mass": 0.0}

        # 2. Add phase definitions
        from pycalphad.io.database import Phase
        phase_models = self.generate_system_phase_models(canon_elements, phases)["phases"]
        for p_name, p_data in phase_models.items():
            model = p_data["sublattice_model"]
            phase_obj = Phase()
            phase_obj.name = p_name
            phase_obj.sublattices = tuple(model["sublattice_site_ratios"])
            phase_obj.constituents = tuple(frozenset(v.Species(x) for x in sub) for sub in model["sublattice_occupancies"])
            phase_obj.model_hints = p_data.get("model_hints", {})
            db.phases[p_name] = phase_obj

        os.makedirs(os.path.dirname(os.path.abspath(output_tdb_path)), exist_ok=True)
        with open(output_tdb_path, "w", encoding="utf-8") as f:
            write_tdb(db, f)

        return db
