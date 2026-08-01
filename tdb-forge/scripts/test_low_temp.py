import os
from pycalphad import Database, equilibrium
import pycalphad.variables as v
import numpy as np

def main():
    tdb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tdbs/TiZrHfNb_RHEA.tdb"))
    db = Database(tdb_path)
    
    # 600 C (873.15 K)
    temp = 873.15
    conds = {
        v.T: temp,
        v.P: 101325,
        v.X('ZR'): 0.25,
        v.X('HF'): 0.25,
        v.X('NB'): 0.25
    }
    
    print(f"Calculating equilibrium for equimolar TiZrHfNb at {temp-273.15:.1f} C ({temp:.2f} K)...")
    res = equilibrium(db, ['TI', 'ZR', 'HF', 'NB', 'VA'], ['LIQUID', 'BCC_A2', 'HCP_A3'], conds)
    
    phases = res.Phase.values.squeeze()
    fractions = res.NP.values.squeeze()
    
    print("\nStable phases:")
    if isinstance(phases, np.ndarray) and phases.ndim > 0:
        for p, f in zip(phases.flat, fractions.flat):
            if f > 1e-4:
                print(f"  {p}: fraction = {f:.4f}")
    else:
        if fractions > 1e-4:
            print(f"  {phases}: fraction = {fractions:.4f}")

if __name__ == '__main__':
    main()
