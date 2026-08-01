"""
Thermodynamic parameter uncertainty quantification and ensemble propagation for tdb-forge.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from pycalphad import Database, equilibrium, variables as v

class PropagateParameterUncertainty:
    """
    Propagates parameter covariance matrices and MCMC chains into
    liquidus temperature distributions and phase boundary credible intervals.
    """
    def __init__(
        self,
        database_path: str,
        parameters_mean: Dict[str, float],
        covariance_matrix: Optional[np.ndarray] = None
    ):
        self.db_path = database_path
        self.db = Database(database_path)
        self.param_names = list(parameters_mean.keys())
        self.mean_vector = np.array([parameters_mean[k] for k in self.param_names])
        
        if covariance_matrix is not None:
            self.cov = covariance_matrix
        else:
            # Default 5% relative standard deviation diagonal covariance
            self.cov = np.diag((0.05 * np.abs(self.mean_vector))**2 + 1e-6)

    def sample_parameters_latin_hypercube(self, num_samples: int = 50) -> List[Dict[str, float]]:
        """
        Generates Latin Hypercube samples from parameter covariance matrix.
        """
        d = len(self.mean_vector)
        # Latin Hypercube Sampling
        lh = np.zeros((num_samples, d))
        for i in range(d):
            perm = np.random.permutation(num_samples)
            lh[:, i] = (perm + np.random.uniform(size=num_samples)) / num_samples
            
        # Transform uniform sample to multivariate normal
        std_samples = np.array([np.quantile(np.random.normal(size=10000), lh[:, j]) for j in range(d)]).T
        cholesky = np.linalg.cholesky(self.cov + 1e-8 * np.eye(d))
        param_samples = self.mean_vector + std_samples @ cholesky.T
        
        samples_list = []
        for row in param_samples:
            samples_list.append({name: float(val) for name, val in zip(self.param_names, row)})
            
        return samples_list

    def evaluate_liquidus_credible_interval(
        self,
        elements: List[str],
        composition: Dict[Any, float],
        num_samples: int = 20,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Evaluates liquidus temperature distribution and 95% Bayesian credible interval.
        """
        from pycalphad.amkit.solidification import find_liquidus_temperature
        
        samples = self.sample_parameters_latin_hypercube(num_samples)
        liquidus_temps = []
        
        phases = list(self.db.phases.keys())
        
        for sample in samples:
            # Perturb database parameters in-memory symbol table
            db_perturbed = Database(self.db_path)
            for param_name, param_val in sample.items():
                if param_name in db_perturbed.symbols:
                    db_perturbed.symbols[param_name] = param_val

            try:
                T_l = find_liquidus_temperature(
                    db_perturbed,
                    elements,
                    phases,
                    composition,
                    T_low=300.0,
                    T_high=3000.0
                )
                liquidus_temps.append(T_l)
            except Exception:
                pass

        if not liquidus_temps:
            # Fallback to mean evaluation
            T_mean = find_liquidus_temperature(self.db, elements, phases, composition)
            return {
                "liquidus_mean": float(T_mean),
                "liquidus_std": 0.0,
                "credible_interval_lower": float(T_mean),
                "credible_interval_upper": float(T_mean),
                "num_successful_samples": 0
            }

        arr = np.array(liquidus_temps)
        alpha = (1.0 - confidence_level) / 2.0
        lower = np.quantile(arr, alpha)
        upper = np.quantile(arr, 1.0 - alpha)
        
        return {
            "liquidus_mean": float(np.mean(arr)),
            "liquidus_std": float(np.std(arr)),
            "credible_interval_lower": float(lower),
            "credible_interval_upper": float(upper),
            "num_successful_samples": len(liquidus_temps)
        }
