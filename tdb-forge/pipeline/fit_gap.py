import os
import yaml
import numpy as np
from espei import run_espei

def main():
    print("Running ESPEI fit_gap parameter generation...")
    # Define ESPEI configuration
    config = {
        "system": {
            "phase_models": "phase_models.json",
            "datasets": "datasets"
        },
        "output": {
            "output_db": "tdbs/alzn_fitted.tdb"
        },
        "generate_parameters": {
            "excess_model": "linear",
            "ref_state": "SGTE91",
            "ridge_alpha": None,
            "aicc_penalty_factor": None
        }
    }
    
    # Run espei using the config dictionary directly
    run_espei(config)
    
    print("ESPEI parameter generation complete! Checking fitted database...")
    if os.path.exists("tdbs/alzn_fitted.tdb"):
        from pycalphad import Database
        db = Database("tdbs/alzn_fitted.tdb")
        liquid_params = [p for p in db._parameters.all() if p['phase_name'] == 'LIQUID']
        for p in liquid_params:
            print(f"Fitted LIQUID parameter: {p['parameter_type']}({p['constituent_array']}, {p['parameter_order']}) = {p['parameter']}")
            
        # Validate that the fitted parameter is within 0.1% of the reference Mey (1993) value (10465.3 J/mol)
        fitted_val = float(db.symbols['VV0000'].args[0][0])
        ref_val = 10465.3
        ratio = fitted_val / ref_val
        print(f"Fitted value: {fitted_val:.2f}, Reference value: {ref_val:.2f}, Ratio: {ratio:.5f}")
        assert np.isclose(fitted_val, ref_val, rtol=1e-3), f"Fitted value {fitted_val} is not close to reference {ref_val}"
        print("[PASS] ESPEI gap-fitting demonstrator successfully verified!")
    else:
        print("[ERROR] fitted TDB file was not created!")

if __name__ == '__main__':
    main()
