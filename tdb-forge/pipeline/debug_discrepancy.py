from pycalphad import Database, calculate

def main():
    db_compiled = Database('/Users/pei/My Drive/PyCalphad/tdb-forge/tdbs/alzn_mey_compiled.tdb')
    db_ref = Database('/Users/pei/My Drive/PyCalphad/pycalphad/tests/databases/alzn_mey.tdb')
    
    print("Reference symbols:")
    for name in ['GHSERAL', 'GHSERZN', 'GALLIQ', 'GALHCP', 'GZNLIQ', 'GZNFCC']:
        if name in db_ref.symbols:
            print(f"ref: {name} = {db_ref.symbols[name]}")
        if name in db_compiled.symbols:
            print(f"comp: {name} = {db_compiled.symbols[name]}")
            
    print("\nReference FCC_A1 ZN parameter:")
    ref_param = db_ref.search(where("phase_name") == "FCC_A1")
    for p in ref_param:
        if p["constituent_array"] == (('ZN',),):
            print(f"ref param: {p}")
            
    comp_param = db_compiled.search(where("phase_name") == "FCC_A1")
    for p in comp_param:
        if p["constituent_array"] == (('ZN',),):
            print(f"comp param: {p}")
            
    # Calculate for pure ZN in FCC_A1 at 300K
    res_c = calculate(db_compiled, ['ZN'], 'FCC_A1', T=300)
    print(f"Compiled GM for ZN at 300K: {res_c.GM.values[0][0][0]}")
    
    res_r = calculate(db_ref, ['ZN'], 'FCC_A1', T=300)
    print(f"Reference GM for ZN at 300K: {res_r.GM.values[0][0][0]}")
    
    # Calculate for pure AL in FCC_A1 at 300K
    res_c_al = calculate(db_compiled, ['AL'], 'FCC_A1', T=300)
    print(f"Compiled GM for AL at 300K: {res_c_al.GM.values[0][0][0]}")
    
    res_r_al = calculate(db_ref, ['AL'], 'FCC_A1', T=300)
    print(f"Reference GM for AL at 300K: {res_r_al.GM.values[0][0][0]}")

if __name__ == '__main__':
    from tinydb import where
    main()
