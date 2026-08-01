import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycalphad import Database, equilibrium
import pycalphad.variables as v

def run_scheil_simulation(db, components, active_phases, nominal_comp, T_start=1650, T_stop=1000, dT=2.0):
    """
    Run a Scheil-Gulliver solidification simulation.
    Returns:
        temperatures: list of temperatures where solidification occurred
        liquid_fraction: list of liquid fractions
        solid_fractions: dict mapping solid phase names to lists of cumulative fractions
    """
    temperatures = []
    liquid_fractions = []
    solid_fractions = {} # phase_name -> list of cumulative fractions
    
    # Initialize liquid composition to nominal
    current_comp = nominal_comp.copy()
    L_moles = 1.0 # Moles of remaining liquid
    
    T = T_start
    
    # Track cumulative solid moles
    cumulative_solids = {} # phase_name -> total moles formed
    
    while T >= T_stop and L_moles > 1e-4:
        # Build conditions for local equilibrium
        conds = {
            v.T: T,
            v.P: 101325
        }
        for el, val in current_comp.items():
            if el != 'NI': # Ni is the dependent component
                conds[v.X(el)] = val
                
        try:
            res = equilibrium(db, components, active_phases, conds)
        except Exception as e:
            print(f"Equilibrium failed at T={T} K: {e}. Skipping step.")
            T -= dT
            continue
            
        phases = res.Phase.values.squeeze()
        fractions = res.NP.values.squeeze()
        x_vals = res.X.values.squeeze()
        
        # Ensure flat arrays
        if not isinstance(phases, np.ndarray) or phases.ndim == 0:
            phases = np.array([phases])
            fractions = np.array([fractions])
            x_vals = np.expand_dims(x_vals, axis=0) if x_vals.ndim == 1 else x_vals
            
        # Find LIQUID phase
        liquid_idx = -1
        for idx, p in enumerate(phases):
            if str(p) == 'LIQUID':
                liquid_idx = idx
                break
                
        f_L = float(fractions[liquid_idx]) if liquid_idx != -1 and fractions[liquid_idx] > 1e-4 else 0.0
        
        # If no liquid is stable, we have finished solidification
        if f_L == 0.0:
            # All remaining liquid transforms to solid at this temperature
            # We distribute it to solid phases formed in this step
            solid_phases_formed = [str(p) for idx, p in enumerate(phases) if str(p) != 'LIQUID' and fractions[idx] > 1e-4]
            if solid_phases_formed:
                for p_str in solid_phases_formed:
                    # Distribute remaining liquid proportionally
                    p_idx = np.where(phases == p_str)[0][0]
                    f_P = float(fractions[p_idx])
                    # Total solid fraction sum without LIQUID
                    tot_s = sum(float(fractions[idx]) for idx, p in enumerate(phases) if str(p) != 'LIQUID')
                    weight = f_P / tot_s if tot_s > 0 else 1.0 / len(solid_phases_formed)
                    moles_formed = weight * L_moles
                    cumulative_solids[p_str] = cumulative_solids.get(p_str, 0.0) + moles_formed
            current_len = len(temperatures)
            L_moles = 0.0
            temperatures.append(T)
            liquid_fractions.append(0.0)
            for p_name in set(list(solid_fractions.keys()) + list(cumulative_solids.keys())):
                if p_name not in solid_fractions:
                    solid_fractions[p_name] = [0.0] * current_len
                solid_fractions[p_name].append(cumulative_solids.get(p_name, 0.0))
            break
            
        # Calculate amount of solid phases formed in this step
        # Local solid fractions in equilibrium (sum to 1-f_L)
        for idx, p in enumerate(phases):
            p_str = str(p)
            if p_str != 'LIQUID' and p_str != '' and fractions[idx] > 1e-4:
                f_P = float(fractions[idx])
                # Moles of this solid phase formed in this step
                moles_formed = f_P * L_moles
                cumulative_solids[p_str] = cumulative_solids.get(p_str, 0.0) + moles_formed
                
        # Update remaining liquid moles and liquid composition
        L_old = L_moles
        L_moles = L_old * f_L
        
        # New liquid composition is the composition of LIQUID phase in equilibrium
        # Extract liquid composition for next step
        comp_list = res.coords['component'].values.tolist()
        new_comp = {}
        for c_idx, comp in enumerate(comp_list):
            if comp in nominal_comp:
                new_comp[comp] = float(x_vals[liquid_idx, c_idx])
        current_comp = new_comp
        
        # Save step results
        current_len = len(temperatures)
        temperatures.append(T)
        liquid_fractions.append(f_L * L_old) # Fraction of original moles that is liquid
        for p_name in set(list(solid_fractions.keys()) + list(cumulative_solids.keys())):
            if p_name not in solid_fractions:
                solid_fractions[p_name] = [0.0] * current_len
            solid_fractions[p_name].append(cumulative_solids.get(p_name, 0.0))
            
        T -= dT
        
    return np.array(temperatures), np.array(liquid_fractions), solid_fractions

def main():
    tdb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../examples/databases/FeCoCrNiMnAlTiVCu_HEA.tdb"))
    output_img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../reports/alfeocrni_ehea/alfeocrni_scheil_vs_equilibrium.png"))

    
    print(f"Loading database {tdb_path}")
    db = Database(tdb_path)
    
    # Define composition: AlFeCoCrNi2.1
    n_total = 6.1
    nominal_comp = {
        'AL': 1.0 / n_total,
        'FE': 1.0 / n_total,
        'CO': 1.0 / n_total,
        'CR': 1.0 / n_total,
        'NI': 2.1 / n_total
    }
    
    components = ['AL', 'FE', 'CO', 'CR', 'NI', 'VA']
    active_phases = list(db.phases.keys())
    
    # 1. Run Scheil Simulation
    print("Running Scheil-Gulliver simulation (representing rapid cooling 10^6 K/s)...")
    scheil_T, scheil_fL, scheil_solids = run_scheil_simulation(
        db, components, active_phases, nominal_comp, T_start=1650, T_stop=1100, dT=2.0
    )
    
    # 2. Run Equilibrium Solidification (Lever Rule, representing infinitely slow cooling)
    print("Running Equilibrium (Lever Rule) solidification...")
    eq_T = np.arange(1100, 1650 + 5, 5)
    conds = {
        v.T: eq_T,
        v.P: 101325,
        v.X('AL'): nominal_comp['AL'],
        v.X('FE'): nominal_comp['FE'],
        v.X('CO'): nominal_comp['CO'],
        v.X('CR'): nominal_comp['CR']
    }
    eq_res = equilibrium(db, components, active_phases, conds)
    
    # Parse equilibrium results
    eq_fractions = {} # phase_name -> list of fractions over eq_T
    for t_idx, T in enumerate(eq_T):
        t_res = eq_res.isel(T=t_idx)
        phases = t_res.Phase.values.squeeze()
        fractions = t_res.NP.values.squeeze()
        if not isinstance(phases, np.ndarray) or phases.ndim == 0:
            phases = np.array([phases])
            fractions = np.array([fractions])
            
        temp_fracs = {}
        for p, f in zip(phases, fractions):
            if p != '' and f > 1e-4 and not np.isnan(f):
                p_str = str(p)
                temp_fracs[p_str] = temp_fracs.get(p_str, 0.0) + float(f)
                
        for p_str, total_f in temp_fracs.items():
            eq_fractions.setdefault(p_str, np.zeros(len(eq_T)))[t_idx] = total_f

    # 3. Plotting comparison
    print("Plotting comparison chart...")
    plt.figure(figsize=(11, 7.5))
    
    # Standard colors
    colors = {
        'LIQUID': '#d62728',     # Red
        'FCC_A1': '#1f77b4',     # Blue
        'BCC_A2': '#ff7f0e',     # Orange
        'BCC_B2': '#2ca02c',     # Green
        'SIGMA': '#9467bd'       # Purple
    }
    default_colors = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Plot Scheil results (solid lines)
    plt.plot(scheil_T, scheil_fL, color=colors['LIQUID'], label='LIQUID (Scheil)', linewidth=2.5)
    color_idx = 0
    for phase_name, fracs in scheil_solids.items():
        color = colors.get(phase_name, default_colors[color_idx % len(default_colors)])
        if phase_name not in colors:
            color_idx += 1
        plt.plot(scheil_T, fracs, color=color, label=f'{phase_name} (Scheil)', linewidth=2.5)
        
    # Plot Equilibrium results (dashed lines)
    # Filter only relevant phases that appear in the temperature range
    if 'LIQUID' in eq_fractions:
        plt.plot(eq_T, eq_fractions['LIQUID'], color=colors['LIQUID'], linestyle='--', linewidth=1.8, label='LIQUID (Equilibrium)')
    for phase_name, fracs in eq_fractions.items():
        if phase_name != 'LIQUID' and np.any(fracs > 1e-3):
            color = colors.get(phase_name, default_colors[color_idx % len(default_colors)])
            if phase_name not in colors:
                color_idx += 1
            plt.plot(eq_T, fracs, color=color, linestyle='--', linewidth=1.8, label=f'{phase_name} (Equil.)')
            
    plt.xlabel('Temperature (K)', fontsize=12, fontweight='bold')
    plt.ylabel('Phase Fraction (NP)', fontsize=12, fontweight='bold')
    plt.title('Solidification Pathway for AlFeCoCrNi2.1: Scheil ($10^6$ K/s) vs. Equilibrium\nDatabase: FeCoCrNiMnAlTiVCu_HEA.tdb', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Clean up legend (remove duplicate labels if any, and sort them)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title='Solidification Mode', title_fontsize=11, fontsize=9, loc='center right', framealpha=0.9)
    
    plt.xlim(1100, 1650)
    plt.ylim(0, 1.05)
    
    # Save chart
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    plt.savefig(output_img_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved Scheil comparison chart to {output_img_path}")

if __name__ == '__main__':
    main()
