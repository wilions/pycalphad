"""
CALPHAD-IR: Standardized Intermediate Representation and Serialization Specification
for CALPHAD Thermodynamic, Thermophysical, and Kinetic Databases.

This module provides two-way conversion between pycalphad Database objects
and the validated CALPHAD-IR standard (JSON / Dictionary).
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from io import StringIO
from typing import Any, Dict, List, Optional, Union

import pycalphad.variables as v
from pycalphad.core.utils import generate_symmetric_group, recursive_tuplify
from pycalphad.io.database import Database, Phase
from pycalphad.io.tdb import TCPrinter, _sympify_string
from pycalphad.variables import Species
from symengine import And, Piecewise, sympify

CALPHAD_IR_SCHEMA_VERSION = "1.0.0"

CALPHAD_IR_SCHEMA_V1: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "CALPHAD-IR",
    "version": CALPHAD_IR_SCHEMA_VERSION,
    "description": "Standard Intermediate Representation for CALPHAD thermodynamic and thermophysical databases",
    "type": "object",
    "required": ["calphad_ir_version", "elements", "species", "phases"],
    "properties": {
        "calphad_ir_version": {"type": "string"},
        "metadata": {
            "type": "object",
            "properties": {
                "database_name": {"type": "string"},
                "version": {"type": "string"},
                "created_at": {"type": "string"},
                "generator": {"type": "string"},
                "comments": {"type": "array", "items": {"type": "string"}},
            },
        },
        "elements": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "reference_phase": {"type": "string"},
                    "mass": {"type": "number"},
                    "H298": {"type": "number"},
                    "S298": {"type": "number"},
                },
            },
        },
        "species": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "composition"],
                "properties": {
                    "name": {"type": "string"},
                    "composition": {"type": "object"},
                    "charge": {"type": "number"},
                },
            },
        },
        "symbols": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                    "piecewise": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "expression": {"type": "string"},
                                "T_low": {"type": ["number", "null"]},
                                "T_high": {"type": ["number", "null"]},
                            },
                        },
                    },
                },
            },
        },
        "phases": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "required": ["sublattices", "constituents"],
                "properties": {
                    "sublattices": {"type": "array", "items": {"type": "number"}},
                    "constituents": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "string"}},
                    },
                    "model_hints": {"type": "object"},
                    "structure_entry": {"type": ["string", "null"]},
                },
            },
        },
        "parameters": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "phase_name",
                    "parameter_type",
                    "constituent_array",
                    "parameter_order",
                ],
                "properties": {
                    "phase_name": {"type": "string"},
                    "parameter_type": {"type": "string"},
                    "constituent_array": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "string"}},
                    },
                    "parameter_order": {"type": "integer"},
                    "diffusing_species": {"type": ["string", "null"]},
                    "reference": {"type": ["string", "null"]},
                    "expression": {"type": "string"},
                    "piecewise": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "expression": {"type": "string"},
                                "T_low": {"type": ["number", "null"]},
                                "T_high": {"type": ["number", "null"]},
                            },
                        },
                    },
                },
            },
        },
    },
}


def _serialize_expr_piecewise(expr: Any) -> Dict[str, Any]:
    """Serialize a SymEngine Expr or Piecewise object to a dict."""
    if isinstance(expr, Piecewise):
        pieces = []
        # symengine.Piecewise.args is a flat tuple of (expr1, cond1, expr2, cond2, ...)
        for i in range(0, len(expr.args), 2):
            piece_expr = expr.args[i]
            cond = expr.args[i + 1]
            # Extract temperature bounds if condition is And(v.T >= T_low, v.T < T_high)
            T_low = None
            T_high = None
            if hasattr(cond, "args"):
                for rel in cond.args:
                    rel_str = str(rel)
                    if "T" in rel_str:
                        # Extract numerical bounds
                        if hasattr(rel, "args") and len(rel.args) == 2:
                            arg0, arg1 = rel.args
                            if str(arg0) == "T" and hasattr(arg1, "is_number") and arg1.is_number:
                                T_high = float(arg1)
                            elif str(arg1) == "T" and hasattr(arg0, "is_number") and arg0.is_number:
                                T_low = float(arg0)
            piece_str = str(piece_expr)
            pieces.append({
                "expression": piece_str,
                "T_low": T_low,
                "T_high": T_high,
            })
        full_str = str(expr)
        return {"expression": full_str, "piecewise": pieces}
    else:
        expr_str = str(expr)
        return {"expression": expr_str, "piecewise": [{"expression": expr_str, "T_low": 298.15, "T_high": 6000.0}]}


def _deserialize_expr_piecewise(data: Union[Dict[str, Any], str]) -> Any:
    """Deserialize a CALPHAD-IR expression dict or string to a SymEngine expression."""
    if isinstance(data, str):
        return _sympify_string(data)

    if isinstance(data, dict):
        if "piecewise" in data and len(data["piecewise"]) > 1:
            pieces = []
            for p in data["piecewise"]:
                sub_expr = _sympify_string(p["expression"])
                t_low = p.get("T_low", 298.15)
                t_high = p.get("T_high", 6000.0)
                if t_low is not None and t_high is not None:
                    cond = And(v.T >= t_low, v.T < t_high)
                elif t_low is not None:
                    cond = (v.T >= t_low)
                elif t_high is not None:
                    cond = (v.T < t_high)
                else:
                    cond = True
                pieces.append((sub_expr, cond))
            return Piecewise(*pieces)
        elif "expression" in data:
            return _sympify_string(data["expression"])
        elif "piecewise" in data and len(data["piecewise"]) == 1:
            return _sympify_string(data["piecewise"][0]["expression"])

    return _sympify_string(str(data))


def database_to_calphad_ir(db: Database, database_name: str = "PyCalphad Database") -> Dict[str, Any]:
    """
    Convert a pycalphad Database instance into a standardized CALPHAD-IR dictionary.

    Parameters
    ----------
    db : Database
        Source pycalphad Database.
    database_name : str, optional
        Human-readable name of the database.

    Returns
    -------
    dict
        CALPHAD-IR compliant dictionary representation.
    """
    ir: Dict[str, Any] = {
        "calphad_ir_version": CALPHAD_IR_SCHEMA_VERSION,
        "metadata": {
            "database_name": database_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generator": "pycalphad-calphad_ir",
            "comments": [],
        },
        "elements": [],
        "species": [],
        "symbols": {},
        "phases": {},
        "parameters": [],
    }

    # 1. Elements and Refstates
    for el in sorted(db.elements):
        el_entry: Dict[str, Any] = {"name": el.upper()}
        if el in db.refstates:
            ref = db.refstates[el]
            el_entry["reference_phase"] = ref.get("phase", "")
            el_entry["mass"] = float(ref.get("mass", 0.0))
            el_entry["H298"] = float(ref.get("H298", 0.0))
            el_entry["S298"] = float(ref.get("S298", 0.0))
        ir["elements"].append(el_entry)

    # 2. Species
    for sp in sorted(db.species, key=lambda s: s.name):
        sp_entry = {
            "name": sp.name.upper(),
            "composition": {k.upper(): v for k, v in sp.constituents.items() if k != "CHARGE"},
            "charge": float(sp.charge),
        }
        ir["species"].append(sp_entry)

    # 3. Symbols / Functions
    for sym_name, sym_expr in sorted(db.symbols.items()):
        ir["symbols"][sym_name.upper()] = _serialize_expr_piecewise(sym_expr)

    # 4. Phases
    for p_name, p_obj in sorted(db.phases.items()):
        constituents_list = [
            sorted([sp.name.upper() for sp in subl]) for subl in p_obj.constituents
        ]
        ir["phases"][p_name.upper()] = {
            "sublattices": [float(x) for x in p_obj.sublattices],
            "constituents": constituents_list,
            "model_hints": p_obj.model_hints,
            "structure_entry": db._structure_dict.get(p_name, None),
        }

    # 5. Parameters
    for param in db._parameters.all():
        const_arr = [
            [sp.name.upper() for sp in subl] for subl in param["constituent_array"]
        ]
        diff_sp = None
        if param.get("diffusing_species") and param["diffusing_species"] != Species(None):
            diff_sp = param["diffusing_species"].name.upper()

        param_entry: Dict[str, Any] = {
            "phase_name": param["phase_name"].upper(),
            "parameter_type": param["parameter_type"].upper(),
            "constituent_array": const_arr,
            "parameter_order": int(param["parameter_order"]),
            "diffusing_species": diff_sp,
            "reference": param.get("reference"),
        }
        param_entry.update(_serialize_expr_piecewise(param["parameter"]))
        ir["parameters"].append(param_entry)

    return ir


def calphad_ir_to_database(ir: Dict[str, Any]) -> Database:
    """
    Construct a pycalphad Database instance from a CALPHAD-IR dictionary.

    Parameters
    ----------
    ir : dict
        CALPHAD-IR dictionary.

    Returns
    -------
    Database
        Constructed pycalphad Database.
    """
    db = Database()

    # 1. Elements & Refstates
    for el_data in ir.get("elements", []):
        el_name = el_data["name"].upper()
        db.elements.add(el_name)
        if "reference_phase" in el_data:
            db.refstates[el_name] = {
                "phase": el_data.get("reference_phase", ""),
                "mass": float(el_data.get("mass", 0.0)),
                "H298": float(el_data.get("H298", 0.0)),
                "S298": float(el_data.get("S298", 0.0)),
            }

    # 2. Species
    for sp_data in ir.get("species", []):
        sp_name = sp_data["name"].upper()
        comp = {k.upper(): v for k, v in sp_data.get("composition", {}).items()}
        charge = sp_data.get("charge", 0.0)
        if charge != 0.0:
            comp["CHARGE"] = charge
        db.species.add(Species(sp_name, comp, charge=charge))

    # Ensure elements are also in species if not already
    for el in db.elements:
        if Species(el) not in db.species:
            db.species.add(Species(el, {el: 1.0}))

    # Ensure vacancy is in species
    if Species("VA") not in db.species:
        db.species.add(Species("VA"))

    # 3. Symbols
    for sym_name, sym_val in ir.get("symbols", {}).items():
        db.symbols[sym_name.upper()] = _deserialize_expr_piecewise(sym_val)

    # 4. Phases
    species_dict = {s.name.upper(): s for s in db.species}
    for p_name, p_data in ir.get("phases", {}).items():
        p_name_up = p_name.upper()
        sublattices = [float(x) for x in p_data["sublattices"]]
        model_hints = p_data.get("model_hints", {})
        db.add_phase(p_name_up, model_hints, sublattices)

        constituents = []
        for subl in p_data["constituents"]:
            subl_species = set()
            for sp_str in subl:
                sp_up = sp_str.upper()
                if sp_up not in species_dict:
                    species_dict[sp_up] = Species(sp_up)
                    db.species.add(species_dict[sp_up])
                subl_species.add(species_dict[sp_up])
            constituents.append(frozenset(subl_species))
        db.phases[p_name_up].constituents = tuple(constituents)

        struct = p_data.get("structure_entry")
        if struct:
            db.add_structure_entry(p_name_up, struct)

    # 5. Parameters
    for param_data in ir.get("parameters", []):
        phase_name = param_data["phase_name"].upper()
        param_type = param_data["parameter_type"].upper()
        param_order = int(param_data["parameter_order"])
        ref = param_data.get("reference")
        diff_sp_str = param_data.get("diffusing_species")
        diff_sp = diff_sp_str.upper() if diff_sp_str else None

        const_array = []
        for subl in param_data["constituent_array"]:
            subl_sp = []
            for sp_str in subl:
                sp_up = sp_str.upper()
                if sp_up not in species_dict:
                    species_dict[sp_up] = Species(sp_up)
                    db.species.add(species_dict[sp_up])
                subl_sp.append(sp_up)
            const_array.append(tuple(subl_sp))

        param_expr = _deserialize_expr_piecewise(param_data)
        db.add_parameter(
            param_type=param_type,
            phase_name=phase_name,
            constituent_array=const_array,
            param_order=param_order,
            param=param_expr,
            ref=ref,
            diffusing_species=diff_sp,
            force_insert=True,
        )

    return db


def read_calphad_ir(db: Database, fd: Any) -> None:
    """Database reader callback for CALPHAD-IR JSON files."""
    if hasattr(fd, "read"):
        content = fd.read()
    else:
        content = fd
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    data = json.loads(content)
    parsed_db = calphad_ir_to_database(data)
    # Populate target db
    db.elements = parsed_db.elements
    db.species = parsed_db.species
    db.phases = parsed_db.phases
    db.symbols = parsed_db.symbols
    db.refstates = parsed_db.refstates
    db.references = parsed_db.references
    db._structure_dict = parsed_db._structure_dict
    db._parameters = parsed_db._parameters


def write_calphad_ir(db: Database, fd: Any, indent: int = 2, **kwargs) -> None:
    """Database writer callback for CALPHAD-IR JSON files."""
    ir_dict = database_to_calphad_ir(db, **kwargs)
    json_str = json.dumps(ir_dict, indent=indent)
    if hasattr(fd, "write"):
        fd.write(json_str)
    else:
        raise ValueError("Target file descriptor must have a write() method.")


# Register JSON and CALPHAD_IR format handlers on Database
Database.register_format("json", read=read_calphad_ir, write=write_calphad_ir)
Database.register_format("calphad_ir", read=read_calphad_ir, write=write_calphad_ir)

# Attach convenient methods directly to Database class
def _to_calphad_ir(self: Database, database_name: str = "PyCalphad Database") -> Dict[str, Any]:
    return database_to_calphad_ir(self, database_name=database_name)

Database.to_calphad_ir = _to_calphad_ir
Database.from_calphad_ir = staticmethod(calphad_ir_to_database)
