"""
Automated ESPEI assessment pipeline runner for tdb-forge.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pycalphad import Database

class ESPEIPipelineRunner:
    """
    Automates dataset validation, ESPEI assessment execution,
    and TDB compilation verification for tdb-forge.
    """
    def __init__(self, assessment_spec_path: str, output_dir: str):
        self.spec_path = os.path.abspath(assessment_spec_path)
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(self.spec_path, 'r', encoding='utf-8') as f:
            self.spec = json.load(f)

    def validate_datasets(self, dataset_dir: str) -> Dict[str, Any]:
        """
        Validates JSON dataset files against expected thermochemical / phase equilibrium schemas.
        """
        valid_files = []
        errors = []
        
        if not os.path.exists(dataset_dir):
            return {"valid": False, "files": [], "errors": [f"Dataset directory not found: {dataset_dir}"]}
            
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith('.json'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # Basic ESPEI dataset schema validation
                            if "components" in data and "phases" in data and "conditions" in data:
                                valid_files.append(filepath)
                            else:
                                errors.append(f"{file}: Missing required ESPEI keys (components, phases, conditions)")
                    except Exception as e:
                        errors.append(f"{file}: JSON parse error: {str(e)}")

        return {
            "valid": len(errors) == 0 and len(valid_files) > 0,
            "valid_files_count": len(valid_files),
            "files": valid_files,
            "errors": errors
        }

    def generate_espei_config(
        self,
        dataset_dir: str,
        system_name: str,
        mcmc_steps: int = 1000
    ) -> str:
        """
        Generates ESPEI YAML configuration file.
        """
        config_content = f"""
system:
  phase_models: {self.spec.get('phase_models_file', 'phase_models.json')}
  datasets: {dataset_dir}

output:
  output_db: {system_name}_espei.tdb
  verbosity: 1

generate_parameters:
  excess_model: linear
  ref_state: SGTE92

mcmc:
  mcmc_steps: {mcmc_steps}
  chains: 12
  scheduler: ray
"""
        config_path = os.path.join(self.output_dir, f"{system_name}_espei_config.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content.strip())
            
        return config_path

    def compile_and_verify_tdb(self, tdb_path: str) -> Dict[str, Any]:
        """
        Loads and verifies that the output TDB parses cleanly in PyCalphad.
        """
        if not os.path.exists(tdb_path):
            return {"success": False, "error": f"TDB file not found: {tdb_path}"}
            
        try:
            db = Database(tdb_path)
            elements = sorted([e for e in db.elements if e != 'VA'])
            phases = list(db.phases.keys())
            return {
                "success": True,
                "elements": elements,
                "phases": phases,
                "num_phases": len(phases)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
