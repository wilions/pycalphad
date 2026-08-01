import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycalphad import Database, equilibrium
import pycalphad.variables as v

def main():
    tdb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../examples/databases/FeCoCrNiMnAlTiVCu_HEA.tdb"))
    output_img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../reports/alfeocrni_ehea/alfeocrni_phase_temp_chart.png"))

    
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
    
    # Temperature range: 600 K to 1800 K with a step of 10 K
    temperatures = np.arange(600, 1800 + 10, 10)
    
    # Select components: AL, FE, CO, CR, NI, VA
    components = ['AL', 'FE', 'CO', 'CR', 'NI', 'VA']
    
    # Select all active phases from database
    active_phases = list(db.phases.keys())
    print(f"Active phases in database: {active_phases}")
    
    conds = {
        v.T: temperatures,
        v.P: 101325,
        v.X('AL'): x_al,
        v.X('FE'): x_fe,
        v.X('CO'): x_co,
        v.X('CR'): x_cr
    }
    
    print("Calculating stepped equilibrium...")
    try:
        res = equilibrium(db, components, active_phases, conds)
        
        # Plotting
        plt.figure(figsize=(10, 7))
        
        # In pycalphad, res.NP has shape (N, P, T, vertex)
        # res.Phase has shape (N, P, T, vertex)
        # Let's map phase name to its fractions over temperature
        phase_fractions = {}
        
        for t_idx, T in enumerate(temperatures):
            # Select results at this temperature
            t_res = res.isel(T=t_idx)
            phases = t_res.Phase.values.squeeze()
            fractions = t_res.NP.values.squeeze()
            
            # Ensure they are flat/iterable
            if not isinstance(phases, np.ndarray) or phases.ndim == 0:
                phases = np.array([phases])
                fractions = np.array([fractions])
                
            for p, f in zip(phases, fractions):
                if p != '' and f > 1e-4:
                    p_str = str(p)
                    phase_fractions.setdefault(p_str, np.zeros(len(temperatures)))[t_idx] = float(f)
                    
        # Draw lines for each phase
        for phase_name, fracs in phase_fractions.items():
            plt.plot(temperatures, fracs, label=phase_name, linewidth=2)
            
        plt.xlabel('Temperature (K)', fontsize=12)
        plt.ylabel('Phase Fraction (NP)', fontsize=12)
        plt.title('Phase Fraction vs. Temperature for AlFeCoCrNi2.1 EHEA\nDatabase: FeCoCrNiMnAlTiVCu_HEA.tdb', fontsize=13, fontweight='bold')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(title='Stable Phases', fontsize=10, loc='best')
        plt.xlim(600, 1800)
        plt.ylim(0, 1.05)
        
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        plt.savefig(output_img_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Successfully saved phase-temperature chart to {output_img_path}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
