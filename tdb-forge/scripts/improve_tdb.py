import os
from pycalphad import Database
from pycalphad.variables import Species
from pycalphad.io.tdb import write_tdb
from symengine import sympify
from tinydb import Query

def remove_parameter_if_exists(db, p_type, phase_name, const_arr, p_order):
    q = Query()
    
    # We want to match:
    # 1. phase_name
    # 2. parameter_type == p_type
    # 3. parameter_order == p_order
    # 4. constituent_array match
    def match_constituents(val):
        if len(val) != len(const_arr):
            return False
        for subl_val, subl_target in zip(val, const_arr):
            val_names = sorted([sp.name.upper() for sp in subl_val])
            target_names = sorted([c.upper() for c in subl_target])
            if val_names != target_names:
                return False
        return True

    matching = db._parameters.search(
        (q.phase_name == phase_name) & 
        (q.parameter_type == p_type) & 
        (q.parameter_order == p_order) &
        (q.constituent_array.test(match_constituents))
    )
    
    if matching:
        print(f"Removing {len(matching)} duplicate or incorrect parameter(s) for {p_type}({phase_name}, {const_arr}; order={p_order})")
        doc_ids = [doc.doc_id for doc in matching]
        db._parameters.remove(doc_ids=doc_ids)

def main():
    tdb_path = "/Users/pei/My Drive/Antigravity/workspace/FeCoCrNiMnAlTiVCu_HEA.tdb"
    output_tdb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../examples/databases/FeCoCrNiMnAlTiVCu_HEA.tdb"))
    output_mcp_db_path = output_tdb_path  # Same destination after consolidation
    
    print(f"Loading database {tdb_path}")
    db = Database(tdb_path)
    
    # Ensure V and CU are in constituents of BCC_B2 phase
    print("Checking constituents of BCC_B2 phase...")
    b2 = db.phases['BCC_B2']
    current_constituents = [[sp.name.upper() for sp in subl] for subl in b2.constituents]
    updated = False
    for el in ['V', 'CU']:
        if el in db.elements:
            if el not in current_constituents[0]:
                current_constituents[0].append(el)
                updated = True
            if el not in current_constituents[1]:
                current_constituents[1].append(el)
                updated = True
    if updated:
        b2.constituents = tuple(frozenset(Species(s) for s in subl) for subl in current_constituents)
        print("Updated BCC_B2 constituents to include V and CU.")
    
    # Define parameters to add
    # 1. Al-Co-Ti system (Zhou et al., 2026)
    al_co_ti_params = [
        # Disordered parameters (A1, A2, A3) - corrected with correct signs
        ('L', 'FCC_A1', [['AL', 'CO', 'TI'], ['VA']], 1, '-104955.8 + 97.8153 * T'),
        ('L', 'FCC_A1', [['AL', 'CO', 'TI'], ['VA']], 2, '120000'),
        ('L', 'BCC_A2', [['AL', 'CO', 'TI'], ['VA']], 1, '-5456.7 + 96.7235 * T'),
        ('L', 'BCC_A2', [['AL', 'CO', 'TI'], ['VA']], 2, '14543.3 + 96.7235 * T'),
        ('L', 'HCP_A3', [['AL', 'CO', 'TI'], ['VA']], 1, '-5456.7 + 96.7235 * T'),
        ('L', 'HCP_A3', [['AL', 'CO', 'TI'], ['VA']], 2, '114543.3 + 96.7235 * T'),
        # BCC_B2 ordered parameters
        ('L', 'BCC_B2', [['CO'], ['AL', 'TI'], ['VA']], 0, '-26838.8 + 8.7502 * T'),
        ('L', 'BCC_B2', [['CO'], ['AL', 'TI'], ['VA']], 1, '-1043.7 + 6.7489 * T'),
        ('L', 'BCC_B2', [['AL'], ['CO', 'TI'], ['VA']], 0, '14978.2'),
        # LIQUID parameters (Zhou et al., 2026)
        ('L', 'LIQUID', [['AL', 'CO', 'TI']], 0, '228182.7 - 91.8456 * T'),
        ('L', 'LIQUID', [['AL', 'CO', 'TI']], 1, '-6005.7 - 43.3286 * T'),
        ('L', 'LIQUID', [['AL', 'CO', 'TI']], 2, '270376.9 - 92.8732 * T'),
    ]
    
    # 2. Co-Al-V system (Wang et al., 2025)
    co_al_v_params = [
        # LIQUID
        ('L', 'LIQUID', [['AL', 'CO', 'V']], 0, '95721.2'),
        # BCC_A2
        ('L', 'BCC_A2', [['AL', 'CO', 'V'], ['VA']], 0, '-46963.8 + 28.9513 * T'),
        # FCC_A1
        ('L', 'FCC_A1', [['AL', 'CO', 'V'], ['VA']], 0, '1653.3 - 63.4191 * T'),
        
        # BCC_B2 ordered parameters (Wang et al., 2025)
        # End-members (G parameters)
        ('G', 'BCC_B2', [['CO'], ['V'], ['VA']], 0, '-17900.5 + 3.1954 * T'),
        ('G', 'BCC_B2', [['V'], ['CO'], ['VA']], 0, '-17900.5 + 3.1954 * T'),
        # Binary interaction parameters within B2 sublattices (L parameters)
        ('L', 'BCC_B2', [['AL'], ['CO', 'V'], ['VA']], 0, '55023.1 - 18.1231 * T'),
        ('L', 'BCC_B2', [['CO', 'V'], ['AL'], ['VA']], 0, '55023.1 - 18.1231 * T'),
        
        ('L', 'BCC_B2', [['AL'], ['CO', 'V'], ['VA']], 1, '-72695.7 + 82.0598 * T'),
        ('L', 'BCC_B2', [['CO', 'V'], ['AL'], ['VA']], 1, '-72695.7 + 82.0598 * T'),
        
        ('L', 'BCC_B2', [['AL'], ['CO', 'V'], ['VA']], 2, '-67505.1 + 34.6024 * T'),
        ('L', 'BCC_B2', [['CO', 'V'], ['AL'], ['VA']], 2, '-67505.1 + 34.6024 * T'),
        
        ('L', 'BCC_B2', [['AL', 'V'], ['CO'], ['VA']], 0, '66046.5 - 48.8273 * T'),
        ('L', 'BCC_B2', [['CO'], ['AL', 'V'], ['VA']], 0, '66046.5 - 48.8273 * T'),
        
        ('L', 'BCC_B2', [['AL', 'V'], ['CO'], ['VA']], 1, '68533.8 - 50.6225 * T'),
        ('L', 'BCC_B2', [['CO'], ['AL', 'V'], ['VA']], 1, '68533.8 - 50.6225 * T'),
        
        ('L', 'BCC_B2', [['AL', 'CO', 'V'], ['CO'], ['VA']], 0, '-47522.4 + 49.5217 * T'),
        ('L', 'BCC_B2', [['CO'], ['AL', 'CO', 'V'], ['VA']], 0, '-47522.4 + 49.5217 * T'),
    ]
    
    all_new_params = al_co_ti_params + co_al_v_params
    
    print(f"Adding/updating {len(all_new_params)} ternary parameters in database...")
    added_count = 0
    for p_type, phase_name, const_arr, p_order, expr_str in all_new_params:
        if phase_name in db.phases:
            # 1. Clean up duplicate or incorrect parameters first
            remove_parameter_if_exists(db, p_type, phase_name, const_arr, p_order)
            
            # 2. Add the correct new parameter
            expr = sympify(expr_str)
            db.add_parameter(
                p_type,
                phase_name,
                [[c.upper() for c in sorted(subl)] for subl in const_arr],
                p_order,
                expr,
                force_insert=True
            )
            added_count += 1
            print(f"  Added {p_type}({phase_name}, {const_arr}; order={p_order})")
        else:
            print(f"  Warning: phase {phase_name} not found in database. Skipping.")
            
    print(f"Successfully added/updated {added_count} parameters.")
    
    # Save the updated database
    print(f"Writing updated TDB to {output_tdb_path}")
    os.makedirs(os.path.dirname(output_tdb_path), exist_ok=True)
    with open(output_tdb_path, 'w') as f:
        write_tdb(db, f)
        
    print(f"Writing updated TDB to {output_mcp_db_path}")
    os.makedirs(os.path.dirname(output_mcp_db_path), exist_ok=True)
    with open(output_mcp_db_path, 'w') as f:
        write_tdb(db, f)
        
    print("Database improvement complete.")

if __name__ == '__main__':
    main()
