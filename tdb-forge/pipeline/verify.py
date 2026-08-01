import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycalphad import Database, Model, calculate, binplot
from pycalphad.variables import Species, T, P, X
from tinydb import where

def verify_t1_mechanical(db_path):
    """
    T1 Mechanical Verification:
    - Verifies the TDB file can be successfully parsed by pycalphad.
    - Verifies every phase can be instantiated as a Model object with no undefined symbols.
    """
    print(f"[T1] Running mechanical checks on {db_path}")
    try:
        db = Database(db_path)
    except Exception as e:
        print(f"[T1] [FAIL] TDB parsing failed: {e}")
        return False
        
    for phase_name in db.phases.keys():
        try:
            # Try instantiating Model. This catches missing symbol definitions
            # like missing reference states or parameter symbols.
            elements = sorted([el for el in db.elements if el != '/-'])
            mod = Model(db, elements, phase_name)
            print(f"[T1] Phase {phase_name} instantiated successfully.")
        except Exception as e:
            print(f"[T1] [FAIL] Phase {phase_name} model instantiation failed: {e}")
            return False
            
    print("[T1] [PASS] Mechanical checks passed.")
    return True

def verify_t2_thermo_kinetic(db_path, system_name, elements, checks_config, report_dir):
    """
    T2 Thermodynamic & Kinetic Verification:
    - Compares computed thermodynamic values (mixing enthalpy, invariants) or diffusivities
      against reference values in checks_config.
    - Generates and saves binary phase diagrams or validation plots in report_dir.
    """
    print(f"[T2] Running thermo-kinetic checks for system {system_name} on {db_path}")
    db = Database(db_path)
    os.makedirs(report_dir, exist_ok=True)
    
    report_lines = [f"# Verification Report for {system_name}", ""]
    all_passed = True
    
    # 1. Plot Phase Diagram and capture strategy for invariant checks
    strategy = None
    if len(elements) == 2:
        try:
            print(f"[T2] Plotting phase diagram for {elements}")
            fig = plt.figure(figsize=(8, 6))
            ax = fig.gca()
            
            # Setup conditions
            # If database lacks pure elements from backbone, merge backbone for mapping
            eval_db = db
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            backbone_path = os.path.join(project_root, 'pycalphad/tests/databases/COST507.tdb')
            if not os.path.exists(backbone_path):
                backbone_path = os.path.abspath(os.path.join(project_root, '../pycalphad/tests/databases/COST507.tdb'))
            if os.path.exists(backbone_path):
                try:
                    from pipeline.merge import merge_databases
                    temp_merged = os.path.join(report_dir, "_temp_eval.tdb")
                    entry_stub = {"system": system_name, "type": "thermodynamic", "supersede_constituents": [elements]}
                    merge_databases(backbone_path, db_path, entry_stub, temp_merged)
                    eval_db = Database(temp_merged)
                except Exception as e_merge:
                    print(f"[T2] Note: Using unmerged db for plotting/invariants: {e_merge}")

            phases = list(eval_db.phases.keys())
            T_min = checks_config.get('T_plot_limits', [300, 1000])[0]
            T_max = checks_config.get('T_plot_limits', [300, 1000])[1]
            ax, strategy = binplot(
                eval_db, elements + ['VA'], phases,
                {X(elements[1]): (0, 1, 0.02), T: (T_min, T_max, 10), P: 101325},
                plot_kwargs={'ax': ax},
                return_strategy=True
            )
            
            plot_path = os.path.join(report_dir, f"{system_name.lower()}_phase_diagram.png")
            ax.get_figure().savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"[T2] Saved phase diagram to {plot_path}")
            report_lines.append(f"## Phase Diagram Plot\n![Phase Diagram]({os.path.basename(plot_path)})\n")
        except Exception as e:
            print(f"[T2] Warning: Phase diagram plotting failed: {e}")
            report_lines.append(f"## Phase Diagram Plot\nFailed to plot: {e}\n")
            
    # 2. Invariant reaction check
    invariants = checks_config.get('invariants', [])
    if invariants:
        report_lines.append("## Invariant Reaction Checks")
        report_lines.append("| Reaction Type | Literature T (K) | Computed T (K) | Diff | Status |")
        report_lines.append("| --- | --- | --- | --- | --- |")
        
        computed_invariants = []
        if strategy is not None:
            try:
                inv_data_list = strategy.get_invariant_data(X(elements[1]), T)
                for inv_item in inv_data_list:
                    if hasattr(inv_item, 'ylim') and len(inv_item.ylim) > 0:
                        computed_invariants.append(float(inv_item.ylim[0]))
                    elif hasattr(inv_item, 'y') and len(inv_item.y) > 0:
                        computed_invariants.append(float(np.mean(inv_item.y)))
            except Exception as e_inv:
                print(f"[T2] Warning: Extraction of invariant data failed: {e_inv}")

        for inv in invariants:
            lit_T = float(inv['T'])
            reaction_type = inv['type']
            tol_K = float(inv.get('tol_K', 5.0))
            
            print(f"[T2] Checking invariant transition near {lit_T} K (tol={tol_K} K)")
            if computed_invariants:
                diffs = [comp_T - lit_T for comp_T in computed_invariants]
                min_idx = int(np.argmin([abs(d) for d in diffs]))
                closest_T = computed_invariants[min_idx]
                closest_diff = diffs[min_idx]
                
                if abs(closest_diff) <= tol_K:
                    status = "PASS"
                    print(f"[T2] [PASS] {reaction_type} at literature {lit_T} K: computed {closest_T:.2f} K (diff {closest_diff:+.2f} K)")
                else:
                    status = "FAIL"
                    all_passed = False
                    print(f"[T2] [FAIL] {reaction_type} at literature {lit_T} K: closest computed {closest_T:.2f} K outside tolerance {tol_K} K (diff {closest_diff:+.2f} K)")
                report_lines.append(f"| {reaction_type} | {lit_T} | {closest_T:.2f} | {closest_diff:+.2f} | {status} |")
            else:
                status = "FAIL"
                all_passed = False
                print(f"[T2] [FAIL] {reaction_type} at literature {lit_T} K: no computed invariants found")
                report_lines.append(f"| {reaction_type} | {lit_T} | N/A | N/A | FAIL |")

            
    # 3. Kinetic / Diffusivity checks
    mobility_checks = checks_config.get('mobility_checks', [])
    if mobility_checks:
        report_lines.append("## Kinetic Diffusivity Verification")
        report_lines.append("| Element | Phase | T (K) | Calculated D* (m2/s) | Reference D* (m2/s) | Ratio | Status |")
        report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        
        for check in mobility_checks:
            el = check['element'].upper()
            phase = check['phase'].upper()
            temp = check['T']
            ref_D0 = check['D0']
            ref_Q = check['Q']
            
            # Evaluate parameter MQ from database
            # MQ is defined in db.symbols
            mq_symbol_name = None
            for name in db.symbols.keys():
                if name.startswith('MQ') and phase in name and el in name:
                    mq_symbol_name = name
                    break
                    
            if not mq_symbol_name:
                # Search in parameters
                params = db.search((where('parameter_type') == 'MQ') & (where('phase_name') == phase))
                # Match diffusing species
                for p in params:
                    if p.get('diffusing_species') == Species(el):
                        # We can evaluate the parameter value
                        mq_val = p['parameter']
                        break
                else:
                    print(f"[T2] [FAIL] Mobility parameter MQ for {el} in {phase} not found.")
                    all_passed = False
                    report_lines.append(f"| {el} | {phase} | {temp} | N/A | {ref_D0:.2e} | N/A | FAIL |")
                    continue
            else:
                mq_val = db.symbols[mq_symbol_name]
                
            # Evaluate SymEngine expression at temperature
            R_val = 8.31451
            t_subs = {T: temp}
            phi_val = float(mq_val.subs(t_subs).evalf())
            
            computed_D = np.exp(phi_val / (R_val * temp))
            ref_D = ref_D0 * np.exp(-ref_Q / (R_val * temp))
            
            ratio = computed_D / ref_D
            status = "PASS" if np.isclose(ratio, 1.0, rtol=1e-2) else "FAIL"
            if status == "FAIL":
                all_passed = False
                
            print(f"[T2] {el} in {phase} at {temp}K: calculated {computed_D:.4e}, reference {ref_D:.4e}, ratio {ratio:.4f}")
            report_lines.append(f"| {el} | {phase} | {temp} | {computed_D:.4e} | {ref_D:.4e} | {ratio:.4f} | {status} |")

    # Save report
    report_path = os.path.join(report_dir, f"{system_name.lower()}_report.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"[T2] Saved report to {report_path}")
    
    return all_passed

def verify_t3_integration(backbone_db_path, compiled_db_path, supersede_config):
    """
    T3 Integration checks:
    - Merges compiled database into backbone.
    - Asserts no phase name or parameter collisions outside the declared supersede constituent list.
    """
    print("[T3] Running integration checks")
    backbone = Database(backbone_db_path)
    compiled = Database(compiled_db_path)
    
    # Check for phase-name collisions
    for phase_name in compiled.phases.keys():
        if phase_name in backbone.phases:
            # If phase name matches, check if models are compatible
            backbone_subl = backbone.phases[phase_name].sublattices
            compiled_subl = compiled.phases[phase_name].sublattices
            if backbone_subl != compiled_subl:
                print(f"[T3] [WARNING] Sublattice model mismatch for phase {phase_name}: "
                      f"backbone {backbone_subl} vs compiled {compiled_subl}")
                
    # Check parameter collisions
    supersede_constituents = {tuple(sorted(x)) for x in supersede_config.get('supersede_constituents', [])}
    
    for param in compiled._parameters.all():
        param_type = param['parameter_type']
        phase_name = param['phase_name']
        const_array = tuple(tuple(spec.name for spec in subl) for subl in param['constituent_array'])
        
        # Check if parameter is in backbone
        matches = backbone.search(
            (where('parameter_type') == param_type) & 
            (where('phase_name') == phase_name) & 
            (where('parameter_order') == param['parameter_order'])
        )
        
        for match in matches:
            match_const = tuple(tuple(spec.name for spec in subl) for subl in match['constituent_array'])
            if match_const == const_array:
                # Collision detected. Check if this constituent array is in the supersede list
                # Flatten constituent array to find elements of interest
                flat_const = tuple(sorted(set(item for subl in const_array for item in subl if item not in ['VA', '/-'])))
                if flat_const not in supersede_constituents:
                    print(f"[T3] [FAIL] Collision detected for parameter {param_type}({phase_name}, {const_array}) "
                          f"which is NOT declared in supersede constituent list.")
                    return False
                    
    print("[T3] [PASS] Integration checks passed.")
    return True
