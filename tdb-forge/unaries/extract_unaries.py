import os
from pycalphad import Database
from pycalphad.io.tdb import write_tdb
from pyparsing import ParseException
from symengine import sympify, Piecewise, And
import pycalphad.variables as v
from pycalphad.variables import Species

def main():
    cost507_path = '/Users/pei/My Drive/PyCalphad/pycalphad/tests/databases/COST507.tdb'
    print(f"Loading database from {cost507_path}")
    db_src = Database(cost507_path)
    
    db_dst = Database()
    
    # 1. Copy default elements and species (/- and VA)
    default_els = ['/-', 'VA']
    for el in default_els:
        db_dst.elements.add(el)
        # Find and copy species
        for sp in db_src.species:
            if sp.name.upper() == el.upper():
                db_dst.species.add(sp)
                break
                
    # 2. Copy target elements, species, refstates, and GHSER functions
    target_els = ['AL', 'MG', 'SI', 'FE', 'ZN', 'NI', 'ZR']
    for el in target_els:
        print(f"Copying element {el}")
        db_dst.elements.add(el)
        
        # Copy species
        for sp in db_src.species:
            if sp.name.upper() == el.upper():
                db_dst.species.add(sp)
                break
                
        # Copy reference states
        if el in db_src.refstates:
            db_dst.refstates[el] = db_src.refstates[el]
            
        # Copy GHSER function
        func_name = f'GHSER{el}'
        if func_name in db_src.symbols:
            db_dst.symbols[func_name] = db_src.symbols[func_name]
        else:
            print(f"Warning: function {func_name} not found in COST507")
            
    # 3. Define Scandium (SC)
    print("Defining element SC")
    db_dst.elements.add('SC')
    db_dst.species.add(Species('SC', {'SC': 1.0}))
    
    # Sc reference state
    db_dst.refstates['SC'] = {
        'phase': 'HCP_A3',
        'mass': 44.9559,
        'H298': 0.0,
        'S298': 0.0
    }
    
    # GHSERSC piecewise definition
    # 298.15 <= T < 1608.0
    expr1 = sympify("-8689.547 + 153.48097 * T - 28.1882 * T * log(T) - 0.0039276 * T**2 + 1.22E-7 * T**3 + 47000 / T")
    # 1608.0 <= T < 3000.0
    expr2 = sympify("-14751.27 + 224.2386 * T - 38.3 * T * log(T) + 1.70889E+29 * T**(-9)")
    
    db_dst.symbols['GHSERSC'] = Piecewise(
        (expr1, And(298.15 <= v.T, v.T < 1608.0)),
        (expr2, And(1608.0 <= v.T, v.T < 3000.0)),
        (0, True)
    )
    
    os.makedirs('/Users/pei/My Drive/PyCalphad/tdb-forge/unaries', exist_ok=True)
    out_path = '/Users/pei/My Drive/PyCalphad/tdb-forge/unaries/SGTE_pure_elements.tdb'
    print(f"Writing unary TDB to {out_path}")
    with open(out_path, 'w') as f:
        write_tdb(db_dst, f)
    print("Successfully completed!")

if __name__ == '__main__':
    main()
