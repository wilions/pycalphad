import os
import traceback
from pycalphad import Database, equilibrium
import pycalphad.variables as v
from pycalphad.io.tdb import write_tdb

def main():
    parent_tdb_path = "/Users/pei/My Drive/Antigravity/Alloy Agents/PyCalphad/examples/databases/mc_fe_v2.059.pycalphad.tdb"
    output_tdb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../examples/databases/FeCoCrNiMnAlTiVCu_HEA.tdb"))
    
    print(f"Loading parent multicomponent database from {parent_tdb_path}")
    db_src = Database(parent_tdb_path)
    
    db_dst = Database()
    
    # 1. Select elements
    target_elements = ['FE', 'CO', 'CR', 'NI', 'MN', 'AL', 'TI', 'V', 'CU', 'VA', '/-']
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
    # We will copy all phases, but filter their constituents
    copied_phases = []
    for phase_name, p_src in db_src.phases.items():
        # Map constituents to list of strings
        filtered_constituents = []
        possible = True
        for subl in p_src.constituents:
            filtered_subl = [sp.name for sp in subl if all(el.upper() in target_elements for el in sp.constituents)]
            if not filtered_subl:
                # If a sublattice has no valid species, this phase cannot be formed
                possible = False
                break
            filtered_constituents.append(filtered_subl)
            
        if possible:
            db_dst.add_structure_entry(phase_name, phase_name)
            db_dst.add_phase(phase_name, p_src.model_hints, p_src.sublattices)
            db_dst.add_phase_constituents(phase_name, filtered_constituents)
            copied_phases.append(phase_name)
            
    print(f"Copied {len(copied_phases)} compatible phases out of {len(db_src.phases)}")
    
    # 5. Copy parameters
    copied_params_count = 0
    for param in db_src._parameters.all():
        if param['phase_name'] in copied_phases:
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
                copied_params_count += 1
                
    print(f"Copied {copied_params_count} parameters")
    
    # 6. Copy symbols/functions used by parameters or pure elements
    copied_symbols = 0
    for sym_name, sym_val in db_src.symbols.items():
        # Check if the symbol is relevant to our elements
        # Or starts with 'G' (like Gibbs energy functions)
        if any(el in sym_name for el in target_elements) or sym_name.startswith('G'):
            db_dst.symbols[sym_name] = sym_val
            copied_symbols += 1
            
    print(f"Copied {copied_symbols} thermodynamic symbols")

    print(f"Writing compiled HEA TDB to {output_tdb_path}")
    os.makedirs(os.path.dirname(output_tdb_path), exist_ok=True)
    with open(output_tdb_path, 'w') as f:
        write_tdb(db_dst, f)
        
    print("Database compilation complete.")
        
    print("Database compilation complete.")
    
    # 7. Validate database with pycalphad equilibrium (Cantor alloy Fe-Co-Cr-Mn-Ni at 1000 °C)
    print("\nValidating database with pycalphad equilibrium...")
    try:
        db_validate = Database(output_tdb_path)
        print("Successfully loaded the new multi-component TDB database!")
        
        # Cantor alloy (FeCoCrMnNi) at 1000 °C (1273.15 K)
        conds = {
            v.T: 1273.15,
            v.P: 101325,
            v.X('CO'): 0.2,
            v.X('CR'): 0.2,
            v.X('MN'): 0.2,
            v.X('NI'): 0.2
        }
        
        # We check phases: LIQUID, FCC_A1, BCC_A2, HCP_A3
        active_phases = [p for p in ['LIQUID', 'FCC_A1', 'BCC_A2', 'HCP_A3'] if p in db_validate.phases]
        print(f"Calculating equilibrium for Cantor alloy using phases: {active_phases}")
        
        res = equilibrium(db_validate, ['FE', 'CO', 'CR', 'MN', 'NI', 'VA'], active_phases, conds)
        print("\nEquilibrium results at 1000 C (1273.15 K):")
        
        import numpy as np
        phases = res.Phase.values.squeeze()
        fractions = res.NP.values.squeeze()
        
        if isinstance(phases, np.ndarray) and phases.ndim > 0:
            for p, f in zip(phases.flat, fractions.flat):
                if f > 1e-4:
                    print(f"  {p}: fraction = {f:.4f}")
        else:
            if fractions > 1e-4:
                print(f"  {phases}: fraction = {fractions:.4f}")
                
    except Exception as e:
        traceback.print_exc()

if __name__ == '__main__':
    main()
