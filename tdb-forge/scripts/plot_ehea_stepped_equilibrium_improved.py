import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycalphad import Database, equilibrium
import pycalphad.variables as v

def main():
    tdb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../examples/databases/FeCoCrNiMnAlTiVCu_HEA.tdb"))
    output_img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../reports/alfeocrni_ehea/alfeocrni_phase_temp_chart_improved.png"))

    
    print(f"Loading database {tdb_path}")
    db = Database(tdb_path)
    
    # Define composition: AlFeCoCrNi2.1
    # Al=1, Fe=1, Co=1, Cr=1, Ni=2.1 -> Total = 6.1 moles
    n_total = 6.1
    x_al = 1.0 / n_total
    x_fe = 1.0 / n_total
    x_co = 1.0 / n_total
    x_cr = 1.0 / n_total
    x_ni = 2.1 / n_total
    
    # High-resolution temperature grid: 600 K to 1800 K with a step of 5 K
    temperatures = np.arange(600, 1800 + 5, 5)
    
    # Select components
    components = ['AL', 'FE', 'CO', 'CR', 'NI', 'VA']
    
    # Select all active phases from database
    active_phases = list(db.phases.keys())
    
    conds = {
        v.T: temperatures,
        v.P: 101325,
        v.X('AL'): x_al,
        v.X('FE'): x_fe,
        v.X('CO'): x_co,
        v.X('CR'): x_cr
    }
    
    print("Calculating high-resolution stepped equilibrium...")
    try:
        res = equilibrium(db, components, active_phases, conds)
        
        # Plotting
        plt.figure(figsize=(11, 7.5))
        
        # Map phase name to its fractions over temperature
        phase_fractions = {}
        
        for t_idx, T in enumerate(temperatures):
            t_res = res.isel(T=t_idx)
            phases = t_res.Phase.values.squeeze()
            fractions = t_res.NP.values.squeeze()
            
            # Convert to arrays if they are single values
            if not isinstance(phases, np.ndarray) or phases.ndim == 0:
                phases = np.array([phases])
                fractions = np.array([fractions])
                
            # Group and sum fractions for identical phase names to handle miscibility gaps correctly
            temp_fractions = {}
            for p, f in zip(phases, fractions):
                if p != '' and f > 1e-4 and not np.isnan(f):
                    p_str = str(p)
                    temp_fractions[p_str] = temp_fractions.get(p_str, 0.0) + float(f)
                    
            for p_str, total_f in temp_fractions.items():
                phase_fractions.setdefault(p_str, np.zeros(len(temperatures)))[t_idx] = total_f
                
        # Draw lines for each phase with distinct colors and styles
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
        
        for idx, (phase_name, fracs) in enumerate(sorted(phase_fractions.items())):
            color = colors[idx % len(colors)]
            style = line_styles[idx % len(line_styles)]
            plt.plot(temperatures, fracs, label=phase_name, color=color, linestyle=style, linewidth=2.5)
            
        plt.xlabel('Temperature (K)', fontsize=12, fontweight='bold')
        plt.ylabel('Phase Fraction (NP)', fontsize=12, fontweight='bold')
        plt.title('Improved Phase Fraction vs. Temperature for AlFeCoCrNi2.1 EHEA\n(Accounting for Miscibility Gaps & Eutectic Transitions)', fontsize=13, fontweight='bold')
        
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(title='Stable Phases', title_fontsize=11, fontsize=10, loc='center right', framealpha=0.9)
        plt.xlim(600, 1800)
        plt.ylim(0, 1.05)
        
        # Add details as text on the plot
        plt.text(620, 0.95, "AlFeCoCrNi2.1 nominal composition\nFeCoCrNiMnAlTiVCu_HEA.tdb", fontsize=9, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        plt.savefig(output_img_path, dpi=250, bbox_inches='tight')
        plt.close()
        print(f"Successfully saved improved phase-temperature chart to {output_img_path}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
