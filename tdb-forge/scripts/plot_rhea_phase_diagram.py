import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycalphad import Database, equilibrium
import pycalphad.variables as v

def main():
    tdb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tdbs/TiZrHfNb_RHEA.tdb"))
    output_img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../reports/tizrhfnb_rhea/tizrhfnb_phase_diagram.png"))

    
    print(f"Loading database {tdb_path}")
    db = Database(tdb_path)
    
    # Define composition and temperature ranges
    # Vary X(NB) from 0.01 to 0.99
    # Keep X(ZR) = X(HF) = X(TI) = (1 - X(NB))/3
    t_min, t_max, t_step = 600, 2400, 30
    x_min, x_max, x_step = 0.01, 0.99, 0.02
    
    temperatures = np.arange(t_min, t_max + t_step, t_step)
    x_nb_vals = np.arange(x_min, x_max + x_step, x_step)
    
    print(f"Grid size: {len(temperatures)} T points x {len(x_nb_vals)} X_NB points")
    
    # Store results
    phase_grid = []
    
    for x_nb in x_nb_vals:
        row = []
        # ZR, HF, TI are equimolar
        x_other = (1.0 - x_nb) / 3.0
        
        # We need to run equilibrium for a range of temperatures
        conds = {
            v.T: temperatures,
            v.P: 101325,
            v.X('ZR'): x_other,
            v.X('HF'): x_other,
            v.X('NB'): x_nb
        }
        
        try:
            res = equilibrium(db, ['TI', 'ZR', 'HF', 'NB', 'VA'], ['LIQUID', 'BCC_A2', 'HCP_A3'], conds)
            
            # Extract stable phases for each temperature
            for t_idx in range(len(temperatures)):
                # Get stable phases (fraction > 1e-4)
                point_res = res.isel(T=t_idx)
                stable_phases = []
                # res.Phase is of shape (N, P, T, X_ZR, X_HF, X_NB, vertex)
                # After squeezing/slicing, it is 1D over vertex
                phases = point_res.Phase.values.squeeze()
                fractions = point_res.NP.values.squeeze()
                
                if isinstance(phases, np.ndarray) and phases.ndim > 0:
                    for p, f in zip(phases, fractions):
                        if f > 1e-4 and p != '':
                            stable_phases.append(p)
                else:
                    if fractions > 1e-4 and phases != '':
                        stable_phases.append(str(phases))
                
                # Sort to ensure uniqueness of combinations (e.g. A+B same as B+A)
                stable_phases_str = " + ".join(sorted(list(set(stable_phases))))
                row.append(stable_phases_str if stable_phases_str else "None")
        except Exception as e:
            # Scribing None or error state for failed grid points
            row.extend(["Error"] * len(temperatures))
            
        phase_grid.append(row)
        
    # Convert to 2D numpy array: shape is (X_NB, T)
    phase_grid = np.array(phase_grid)
    
    # Find all unique phase combinations
    unique_combinations = sorted(list(set(phase_grid.flatten())))
    print(f"Unique phase combinations found: {unique_combinations}")
    
    # Map combinations to integers for plotting
    comb_to_id = {comb: idx for idx, comb in enumerate(unique_combinations)}
    numeric_grid = np.zeros(phase_grid.shape)
    for i in range(phase_grid.shape[0]):
        for j in range(phase_grid.shape[1]):
            numeric_grid[i, j] = comb_to_id[phase_grid[i, j]]
            
    # Plotting
    plt.figure(figsize=(10, 7))
    
    # Custom colormap
    cmap = matplotlib.colormaps['tab10'].resampled(len(unique_combinations))
    
    # Mesh plot
    X, Y = np.meshgrid(x_nb_vals, temperatures)
    # numeric_grid has shape (len(x_nb_vals), len(temperatures))
    # transpose numeric_grid to match Y (T) on vertical axis, X (X_NB) on horizontal axis
    im = plt.pcolormesh(X, Y, numeric_grid.T, cmap=cmap, shading='auto', alpha=0.85)
    
    # Set labels and title
    plt.xlabel('Mole Fraction of Nb, $x$(Nb)', fontsize=12)
    plt.ylabel('Temperature (K)', fontsize=12)
    plt.title('Pseudo-Binary Phase Diagram of Ti-Zr-Hf-Nb (with $x$(Ti)=$x$(Zr)=$x$(Hf))\nDatabase: TiZrHfNb_RHEA.tdb', fontsize=13, fontweight='bold')
    
    # Colorbar legend
    cbar = plt.colorbar(im, ticks=range(len(unique_combinations)))
    cbar.ax.set_yticklabels(unique_combinations)
    cbar.set_label('Stable Phases', rotation=275, labelpad=15, fontsize=11)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    plt.savefig(output_img_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully saved phase diagram to {output_img_path}")

if __name__ == '__main__':
    main()
