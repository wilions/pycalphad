import os
from pycalphad import Database
from pycalphad.io.tdb import write_tdb
from pycalphad.variables import Species
from tinydb import where

def merge_databases(backbone_path, compiled_path, supersede_config, output_path):
    """
    Merge the compiled assessment database into the backbone database applying the supersede config.
    """
    print(f"Merging {compiled_path} onto backbone {backbone_path}")
    backbone = Database(backbone_path)
    compiled = Database(compiled_path)
    
    # 1. Merge elements and reference states
    for el in compiled.elements:
        backbone.elements.add(el)
        if el in compiled.refstates:
            backbone.refstates[el] = compiled.refstates[el]
            
    # 2. Merge species
    backbone_sp_names = {sp.name.upper() for sp in backbone.species}
    for sp in compiled.species:
        if sp.name.upper() not in backbone_sp_names:
            backbone.species.add(sp)
            backbone_sp_names.add(sp.name.upper())
            
    # 3. Merge phases
    for phase_name, compiled_phase in compiled.phases.items():
        phase_name_upper = phase_name.upper()
        if phase_name_upper not in backbone.phases:
            print(f"Adding new phase {phase_name_upper} to backbone.")
            backbone.add_structure_entry(phase_name_upper, phase_name_upper)
            backbone.add_phase(phase_name_upper, compiled_phase.model_hints, compiled_phase.sublattices)
            
            # Map constituents to lists of strings for add_phase_constituents
            const_list = []
            for subl in compiled_phase.constituents:
                subl_list = [sp.name for sp in subl]
                const_list.append(subl_list)
            backbone.add_phase_constituents(phase_name_upper, const_list)
        else:
            # Update existing phase constituents if necessary
            # For each sublattice, add new constituents
            backbone_phase = backbone.phases[phase_name_upper]
            new_constituents = []
            for i, compiled_subl in enumerate(compiled_phase.constituents):
                backbone_subl = set(backbone_phase.constituents[i])
                backbone_subl.update(compiled_subl)
                new_constituents.append(list(backbone_subl))
            
            backbone_phase.constituents = tuple(
                frozenset(xs) for xs in new_constituents
            )
            # Merge model hints
            backbone_phase.model_hints.update(compiled_phase.model_hints)
            
    # 4. Merge symbols (functions)
    for sym_name, sym_val in compiled.symbols.items():
        sym_name_upper = sym_name.upper()
        if sym_name_upper in backbone.symbols:
            print(f"Overwriting symbol {sym_name_upper} with compiled version.")
        backbone.symbols[sym_name_upper] = sym_val
        
    # 5. Apply selective supersede policy on parameters
    # The supersede_config defines target constituent arrays to supersede (e.g. [['AL', 'ZN']])
    supersede_constituents = {tuple(sorted(x)) for x in supersede_config.get('supersede_constituents', [])}
    
    # We find parameters in backbone matching the phases and constituent arrays
    # to remove before inserting compiled ones
    backbone_params = list(backbone._parameters.all())
    for p in backbone_params:
        # Check if the flattened constituent array matches a supersede target
        flat_const = tuple(sorted(set(
            spec.name.upper() 
            for subl in p['constituent_array'] 
            for spec in subl 
            if spec.name.upper() not in ['VA', '/-']
        )))
        
        if flat_const in supersede_constituents:
            # Delete this parameter from backbone
            backbone._parameters.remove(doc_ids=[p.doc_id])
            print(f"Superseded backbone parameter: {p['parameter_type']}({p['phase_name']}, {flat_const})")
            
    # 6. Insert all compiled parameters
    for p in compiled._parameters.all():
        backbone._parameters.insert({k: v for k, v in p.items()})
        
    # 7. Write output merged TDB file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        write_tdb(backbone, f)
        
    print(f"Successfully merged and saved database to {output_path}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 5:
        print("Usage: python merge.py <backbone_tdb> <compiled_tdb> <supersede_yaml> <output_tdb>")
        sys.exit(1)
        
    import yaml
    with open(sys.argv[3], 'r') as f:
        config = yaml.safe_load(f)
        
    merge_databases(sys.argv[1], sys.argv[2], config, sys.argv[4])
