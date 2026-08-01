import os
import yaml
from pipeline.compile import compile_file
from pipeline.verify import verify_t1_mechanical, verify_t2_thermo_kinetic, verify_t3_integration
from pipeline.merge import merge_databases

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(PROJECT_ROOT, 'registry/selection_registry.yaml')
BACKBONE_PATH = '/Users/pei/My Drive/Antigravity/Alloy Agents/PyCalphad/pycalphad/tests/databases/COST507.tdb'

def run():
    print(f"Reading selection registry from {REGISTRY_PATH}")
    with open(REGISTRY_PATH, 'r') as f:
        registry = yaml.safe_load(f)
        
    entries = registry.get('entries', [])
    print(f"Found {len(entries)} entries in registry.")
    
    merged_thermo_path = os.path.join(PROJECT_ROOT, 'tdbs/AM_Al_thermo.tdb')
    merged_mobility_path = os.path.join(PROJECT_ROOT, 'tdbs/AM_Al_mobility.tdb')
    
    # Initialize merged databases with backbone or a copy of it
    import shutil
    shutil.copyfile(BACKBONE_PATH, merged_thermo_path)
    
    # For mobility, we start with an empty or fresh database structure (we can initialize with the compiled unaries)
    unaries_tdb = os.path.join(PROJECT_ROOT, 'unaries/SGTE_pure_elements.tdb')
    shutil.copyfile(unaries_tdb, merged_mobility_path)
    
    for entry in entries:
        system = entry['system']
        json_path = os.path.join(PROJECT_ROOT, entry['path'])
        compiled_tdb = os.path.join(PROJECT_ROOT, f"tdbs/{system.lower()}_compiled.tdb")
        report_dir = os.path.join(PROJECT_ROOT, f"reports/{system.lower()}")
        
        print("\n" + "="*50)
        print(f"Processing system: {system} ({entry['type']})")
        print("="*50)
        
        # 1. Compile
        compile_file(json_path, compiled_tdb)
        
        # 2. T1 Mechanical Verify
        if not verify_t1_mechanical(compiled_tdb):
            print(f"[ERROR] T1 verification failed for {system}. Halting.")
            return False
            
        # 3. T2 Thermo-Kinetic Verify
        cfg = entry.get('verification', {"T_plot_limits": [300, 1000]})
        elements = entry.get('supersede_constituents', [[]])[0]
        if not verify_t2_thermo_kinetic(compiled_tdb, system, elements, cfg, report_dir):
            print(f"[ERROR] T2 verification failed for {system}. Halting.")
            return False

            
        # 4. Merge and T3 Integration check
        if entry['type'] == 'thermodynamic':
            # Check T3 integration before final merge
            if not verify_t3_integration(merged_thermo_path, compiled_tdb, entry):
                print(f"[ERROR] T3 integration verification failed for {system}. Halting.")
                return False
            # Perform merge
            merge_databases(merged_thermo_path, compiled_tdb, entry, merged_thermo_path)
        else:
            # Mobility merge
            if not verify_t3_integration(merged_mobility_path, compiled_tdb, entry):
                print(f"[ERROR] T3 integration verification failed for {system}. Halting.")
                return False
            merge_databases(merged_mobility_path, compiled_tdb, entry, merged_mobility_path)
            
    print("\n" + "="*50)
    print("Pipeline Execution Completed Successfully!")
    print(f"Merged Thermodynamic Database saved to: {merged_thermo_path}")
    print(f"Merged Companion Mobility Database saved to: {merged_mobility_path}")
    print("="*50)
    return True

if __name__ == '__main__':
    run()
