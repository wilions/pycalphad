import os
import json
from pycalphad import Database
from pycalphad.io.tdb import to_interval
from symengine import Piecewise, And, S
import pycalphad.variables as v
from pycalphad.variables import Species

def extract_pieces(expr):
    """
    Extract pieces (T_limits, expression) from a SymEngine expression (Piecewise or raw).
    """
    pieces = []
    if isinstance(expr, Piecewise):
        # Filter out default zeros (0, True)
        filtered_args = [(x, cond) for x, cond in zip(*[iter(expr.args)]*2) 
                         if not ((cond == S.true) and (x == S.Zero))]
        
        for val, cond in filtered_args:
            try:
                interval = to_interval(cond)
                low = float(interval.args[0])
                high = float(interval.args[1])
            except Exception:
                # Fallback if parsing interval fails
                low = 298.15
                high = 6000.0
            
            # Stringify expression, replace log back to LN for TDB-style if desired, or keep standard log
            # In our compile.py, we support standard mathematical symbols and log/ln.
            expr_str = str(val).replace('log(T)', 'LN(T)').replace('log(v.T)', 'LN(T)')
            pieces.append({
                "T_limits": [low, high],
                "expression": expr_str
            })
    else:
        # Raw expression
        pieces.append({
            "T_limits": [298.15, 6000.0],
            "expression": str(expr).replace('log(T)', 'LN(T)').replace('log(v.T)', 'LN(T)')
        })
        
    # Sort pieces by lower limit
    return sorted(pieces, key=lambda p: p['T_limits'][0])

def convert_tdb(tdb_path, output_json_path, doi="10.1016/S0364-5916(01)00049-7", system="Ni-Al"):
    print(f"Loading {tdb_path}")
    db = Database(tdb_path)
    
    # Do not include unaries in the extracted JSON to keep it clean and let compile.py resolve them from SGTE_pure_elements.tdb
    # Let's filter out standard GHSER* functions
    elements = [el for el in db.elements if el not in ['/-', 'VA']]
    
    phases_json = []
    for name, phase_obj in db.phases.items():
        phase_name = name.upper()
        
        # Prepare constituents
        constituents = [[spec.name for spec in sorted(subl)] for subl in phase_obj.constituents]
        
        # Prepare model hints
        model_hints = {}
        for k, val in phase_obj.model_hints.items():
            if k in ['ordered_phase', 'disordered_phase']:
                model_hints[k] = val.upper()
            elif k in ['never_disorder', 'symmetry_FCC_4SL', 'symmetry_BCC_4SL', 'ionic_liquid_2SL', 'liquid', 'gas']:
                model_hints[k] = bool(val)
            elif k in ['ihj_magnetic_afm_factor', 'ihj_magnetic_structure_factor']:
                model_hints[k] = float(val)
                
        phase_dict = {
            "name": phase_name,
            "sublattices": [float(x) for x in phase_obj.sublattices],
            "constituents": constituents
        }
        if model_hints:
            phase_dict["model_hints"] = model_hints
            
        phases_json.append(phase_dict)
        
    functions_json = []
    for name, expr in db.symbols.items():
        func_name = name.upper()
        # Skip GHSER functions of our core elements
        if func_name.startswith('GHSER') and func_name[5:] in ['AL', 'MG', 'SI', 'FE', 'ZN', 'NI', 'ZR', 'SC']:
            continue
        # Skip standard reference states symbols if any
        if func_name in ['UNTIER', 'TROIS']:
            # Keep them if they are custom functions defined in Dupin
            pass
            
        functions_json.append({
            "name": func_name,
            "pieces": extract_pieces(expr)
        })
        
    parameters_json = []
    for param in db._parameters.all():
        # Skip parameters from symmetry generation (compile.py will re-generate them)
        if param.get('_generated_by_symmetry_option', False):
            continue
            
        param_type = param['parameter_type'].upper()
        phase_name = param['phase_name'].upper()
        constituent_array = [[spec.name for spec in sorted(subl)] for subl in param['constituent_array']]
        param_order = int(param['parameter_order'])
        ref = param.get('reference', None)
        
        param_dict = {
            "type": param_type,
            "phase": phase_name,
            "constituent_array": constituent_array,
            "parameter_order": param_order,
            "pieces": extract_pieces(param['parameter'])
        }
        
        diff_sp = param.get('diffusing_species', None)
        if diff_sp and diff_sp != Species(None):
            param_dict["diffusing_species"] = diff_sp.name.upper()
            
        if ref:
            param_dict["reference_tag"] = str(ref)
            
        parameters_json.append(param_dict)
        
    json_data = {
        "doi": doi,
        "system": system,
        "elements": [el.upper() for el in elements],
        "phases": phases_json,
        "functions": functions_json,
        "parameters": parameters_json
    }
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Successfully wrote JSON to {output_json_path}")

if __name__ == '__main__':
    # Convert Dupin Ni-Al
    convert_tdb(
        '/Users/pei/My Drive/PyCalphad/examples/databases/NI_AL_DUPIN_2001.TDB',
        '/Users/pei/My Drive/PyCalphad/tdb-forge/extracted/alni_dupin_2001.json',
        doi="10.1016/S0364-5916(01)00049-7",
        system="Ni-Al"
    )
