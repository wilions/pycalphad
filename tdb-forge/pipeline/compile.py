import os
import json
import jsonschema
import yaml
from symengine import sympify, Piecewise, And
import pycalphad.variables as v
from pycalphad import Database
from pycalphad.io.tdb import write_tdb, _sympify_string
from pycalphad.variables import Species

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), '../schemas/assessment.schema.json')
UNARIES_PATH = os.path.join(os.path.dirname(__file__), '../unaries/SGTE_pure_elements.tdb')

def load_schema():
    with open(SCHEMA_PATH, 'r') as f:
        return json.load(f)

def validate_assessment(data, schema):
    jsonschema.validate(instance=data, schema=schema)

def check_piecewise_continuity(pieces, identifier):
    """
    Check if piecewise temperature ranges are continuous and do not overlap.
    """
    if not pieces:
        raise ValueError(f"No pieces defined for {identifier}")
    
    # Sort pieces by lower temperature limit
    sorted_pieces = sorted(pieces, key=lambda p: p['T_limits'][0])
    
    for i in range(len(sorted_pieces) - 1):
        current_high = sorted_pieces[i]['T_limits'][1]
        next_low = sorted_pieces[i + 1]['T_limits'][0]
        if current_high != next_low:
            raise ValueError(
                f"Piecewise continuity error in {identifier}: gap or overlap detected between "
                f"{current_high} and {next_low}"
            )
            
    return sorted_pieces

def build_piecewise_expr(pieces, identifier):
    sorted_pieces = check_piecewise_continuity(pieces, identifier)
    expr_cond_pairs = []
    for piece in sorted_pieces:
        low, high = piece['T_limits']
        expr = _sympify_string(piece['expression'])
        expr_cond_pairs.append((expr, And(low <= v.T, v.T < high)))
    # Add default zero piece
    expr_cond_pairs.append((0, True))
    return Piecewise(*expr_cond_pairs)

def compile_json_to_db(json_data, unary_db=None):
    dbf = Database()
    
    # 1. Register elements and copy unaries
    elements = [el.upper() for el in json_data['elements']]
    for el in elements:
        dbf.elements.add(el)
        # Copy species, refstate, and GHSER function from unary DB if available
        if unary_db:
            # Copy species
            for sp in unary_db.species:
                if sp.name.upper() == el:
                    dbf.species.add(sp)
                    break
            # Copy refstate
            if el in unary_db.refstates:
                dbf.refstates[el] = unary_db.refstates[el]
            # Copy GHSER function
            ghser_name = f"GHSER{el}"
            if ghser_name in unary_db.symbols:
                dbf.symbols[ghser_name] = unary_db.symbols[ghser_name]
                
    # Copy default species if not already added
    if unary_db:
        for el in ['/-', 'VA']:
            dbf.elements.add(el)
            for sp in unary_db.species:
                if sp.name.upper() == el:
                    dbf.species.add(sp)
                    break

    # 2. Register functions
    for func in json_data.get('functions', []):
        func_name = func['name'].upper()
        expr = build_piecewise_expr(func['pieces'], f"function {func_name}")
        dbf.symbols[func_name] = expr

    # 3. Register phases
    for phase in json_data.get('phases', []):
        phase_name = phase['name'].upper()
        dbf.add_structure_entry(phase_name, phase_name)
        
        # Prepare model hints
        model_hints = {}
        if 'model_hints' in phase:
            for k, val in phase['model_hints'].items():
                # Convert keys back to the standard casing if necessary, or keep as is
                model_hints[k] = val
                
        dbf.add_phase(phase_name, model_hints, phase['sublattices'])
        # Map constituents to frozensets of Species
        constituents = phase['constituents']
        dbf.add_phase_constituents(phase_name, constituents)

    # 4. Register parameters
    for param in json_data.get('parameters', []):
        param_type = param['type'].upper()
        phase_name = param['phase'].upper()
        constituent_array = param['constituent_array']
        param_order = param['parameter_order']
        ref = param.get('reference_tag', None)
        diff_sp = param.get('diffusing_species', None)
        if diff_sp:
            diff_sp = diff_sp.upper()
            
        param_id = f"parameter {param_type}({phase_name}, {constituent_array}; {param_order})"
        expr = build_piecewise_expr(param['pieces'], param_id)
        
        dbf.add_parameter(
            param_type, 
            phase_name, 
            [[c.upper() for c in sorted(subl)] for subl in constituent_array], 
            param_order, 
            expr, 
            ref=ref, 
            diffusing_species=diff_sp, 
            force_insert=True
        )
        
    from pycalphad.io.tdb import add_phase_symmetry_ordering_parameters
    add_phase_symmetry_ordering_parameters(dbf)
    
    return dbf

def compile_file(input_json_path, output_tdb_path):
    print(f"Loading and validating {input_json_path}")
    with open(input_json_path, 'r') as f:
        data = json.load(f)
        
    schema = load_schema()
    validate_assessment(data, schema)
    
    unary_db = None
    if os.path.exists(UNARIES_PATH):
        print(f"Loading unary references from {UNARIES_PATH}")
        unary_db = Database(UNARIES_PATH)
        
    dbf = compile_json_to_db(data, unary_db)
    
    print(f"Writing database to {output_tdb_path}")
    os.makedirs(os.path.dirname(output_tdb_path), exist_ok=True)
    with open(output_tdb_path, 'w') as f:
        write_tdb(dbf, f)
    print("Compilation successful!")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python compile.py <input_json> <output_tdb>")
        sys.exit(1)
    compile_file(sys.argv[1], sys.argv[2])
