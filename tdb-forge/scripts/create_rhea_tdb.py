import os
from pycalphad import Database, equilibrium
import pycalphad.variables as v
from pycalphad.io.tdb import write_tdb

def main():
    cost507_path = "/Users/pei/My Drive/Antigravity/Alloy Agents/PyCalphad/pycalphad/tests/databases/COST507.tdb"
    output_tdb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tdbs/TiZrHfNb_RHEA.tdb"))
    
    print(f"Loading parent database from {cost507_path}")
    db_src = Database(cost507_path)
    
    db_dst = Database()
    
    # 1. Select elements
    target_elements = ['TI', 'ZR', 'HF', 'NB', 'VA', '/-']
    for el in target_elements:
        db_dst.elements.add(el)
        
    # 2. Copy species
    for sp in db_src.species:
        constituents = sp.constituents
        if all(el.upper() in target_elements for el in constituents):
            db_dst.species.add(sp)
            
    # 3. Copy refstates
    for el, refstate in db_src.refstates.items():
        if el.upper() in target_elements:
            db_dst.refstates[el] = refstate
            
    # 4. Copy phases
    target_phases = ['LIQUID', 'BCC_A2', 'HCP_A3']
    for phase_name in target_phases:
        if phase_name in db_src.phases:
            p_src = db_src.phases[phase_name]
            db_dst.add_structure_entry(phase_name, phase_name)
            db_dst.add_phase(phase_name, p_src.model_hints, p_src.sublattices)
            
            # Map constituents to list of strings
            filtered_constituents = []
            for subl in p_src.constituents:
                filtered_subl = [sp.name for sp in subl if all(el.upper() in target_elements for el in sp.constituents)]
                filtered_constituents.append(filtered_subl)
            db_dst.add_phase_constituents(phase_name, filtered_constituents)
            
    # 5. Copy parameters
    for param in db_src._parameters.all():
        if param['phase_name'] in target_phases:
            consts = param['constituent_array']
            match = True
            for subl in consts:
                if not all(sp.name.upper() in target_elements for sp in subl):
                    match = False
                    break
            if match:
                db_dst.add_parameter(
                    param['parameter_type'],
                    param['phase_name'],
                    [[sp.name.upper() for sp in sorted(subl)] for subl in consts],
                    param['parameter_order'],
                    param['parameter'],
                    diffusing_species=param['diffusing_species'],
                    reference=param['reference'],
                    force_insert=True
                )
                
    # 6. Copy symbols/functions used by parameters or pure elements
    for el in ['TI', 'ZR', 'HF', 'NB']:
        func_name = f"GHSER{el}"
        if func_name in db_src.symbols:
            db_dst.symbols[func_name] = db_src.symbols[func_name]
            
    for sym_name, sym_val in db_src.symbols.items():
        if sym_name.startswith('G') and any(el in sym_name for el in ['TI', 'ZR', 'HF', 'NB']):
            if sym_name not in db_dst.symbols:
                db_dst.symbols[sym_name] = sym_val

    print(f"Writing extracted RHEA TDB to {output_tdb_path}")
    os.makedirs(os.path.dirname(output_tdb_path), exist_ok=True)
    with open(output_tdb_path, 'w') as f:
        write_tdb(db_dst, f)
    print("Database extraction complete.")
    
    # 7. Validate database with pycalphad equilibrium
    print("\nValidating database with pycalphad equilibrium...")
    try:
        db_validate = Database(output_tdb_path)
        print("Successfully loaded the new TDB database!")
        
        conds = {
            v.T: 1473.15,
            v.P: 101325,
            v.X('ZR'): 0.25,
            v.X('HF'): 0.25,
            v.X('NB'): 0.25
        }
        res = equilibrium(db_validate, ['TI', 'ZR', 'HF', 'NB', 'VA'], ['LIQUID', 'BCC_A2', 'HCP_A3'], conds)
        print("\nEquilibrium results at 1200 C (1473.15 K):")
        print(res)
        print("\nStable phases:")
        import numpy as np
        phases = res.Phase.values.squeeze()
        fractions = res.NP.values.squeeze()
        
        # Handle single point or array output
        if isinstance(phases, np.ndarray) and phases.ndim > 0:
            for p, f in zip(phases.flat, fractions.flat):
                if f > 1e-4:
                    print(f"  {p}: fraction = {f:.4f}")
        else:
            if fractions > 1e-4:
                print(f"  {phases}: fraction = {fractions:.4f}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
