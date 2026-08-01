import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

from tinydb import where
from kawin.thermo import BinaryThermodynamics, MulticomponentThermodynamics
from kawin.precipitation import PrecipitateParameters, MatrixParameters, PrecipitateModel
from pycalphad.amkit.mobility import MobilityModel

def resolve_molar_volume(dbf, phase, default_val=1e-5):
    """
    Search and resolve molar volume parameter (V0 or VM) from database for a phase.
    """
    params = dbf.search(
        (where('phase_name') == phase.upper()) &
        ((where('parameter_type') == 'V0') | (where('parameter_type') == 'VM'))
    )
    if len(params) == 0:
        return default_val

    try:
        val = params[0]['parameter']
        from sympy import sympify
        expr = sympify(val)
        
        # Substitute database symbols
        subs_dict = {}
        for sym_name, sym_val in dbf.symbols.items():
            subs_dict[sym_name] = sym_val
        resolved = expr.subs(subs_dict)
        
        # Substitute free symbols (e.g. T, site fractions) with sensible defaults
        if resolved.free_symbols:
            sub_free = {}
            for s in resolved.free_symbols:
                if s.name == 'T':
                    sub_free[s] = 298.15
                else:
                    sub_free[s] = 0.5
            resolved = resolved.subs(sub_free)
            
        return float(resolved)
    except Exception:
        return default_val

class PrecipitationKineticsSimulation:
    def __init__(self, dbf, elements, phases, T, initial_composition, interfacial_energy, diffusivity=None, molar_volume=None, user_constants=None):
        """
        Coupled precipitation kinetics simulation wrapper using Kawin (KWN model).
        
        Parameters
        ----------
        dbf : pycalphad Database or str
            Path to TDB database or Database object.
        elements : list of str
            Components to consider (e.g. ['AL', 'ZN']). First component must be reference solvent.
        phases : list of str
            Phases involved. First phase must be matrix, subsequent phases are precipitates.
            e.g. ['FCC_A1', 'HCP_A3']
        T : float or callable or tuple(times, temperatures)
            Temperature in Kelvin. If callable, T(t) function of time (s).
            If tuple, (times, temperatures) profile with times in seconds.
        initial_composition : float or dict or list
            Solute mole fraction(s). If binary, float representing solute element mole fraction.
            If multicomponent, list or dict of solute mole fractions.
        interfacial_energy : float
            Precipitate interfacial surface energy in J/m^2.
        diffusivity : callable or float, optional
            Diffusion coefficient of matrix in m^2/s. If None, calculated via MobilityModel.
        molar_volume : float, optional
            Molar volume of phases in m^3/mol. If None, resolved from database.
        user_constants : dict, optional
            User constant overrides for MobilityModel.
        """
        if isinstance(dbf, str):
            from pycalphad import Database
            dbf = Database(dbf)

        # Determine solute elements (excluding reference solvent and vacancies)
        solute_elements = [e for e in elements if e not in ['VA', elements[0]]]
        
        self.dbf = dbf
        self.elements = elements
        self.phases = phases
        self.solute_elements = solute_elements

        # Handle temperature input (support non-isothermal profile)
        from kawin.precipitation.PrecipitationParameters import TemperatureParameters
        if isinstance(T, tuple):
            times, temps = T
            times_hrs = [ti / 3600.0 for ti in times]
            self.temperature_params = TemperatureParameters(times_hrs, temps)
            self.T = temps[0]
        elif callable(T):
            self.temperature_params = TemperatureParameters(T)
            self.T = T(0.0)
        else:
            self.T = T
            self.temperature_params = T
        
        # 1. Setup Thermodynamics (using curvature method for stability)
        if len(solute_elements) == 1:
            self.therm = BinaryThermodynamics(dbf, elements, phases, interfacialCompMethod='curvature')
        else:
            self.therm = MulticomponentThermodynamics(dbf, elements, phases)
            
        # 2. Set Diffusivity (database-lookup fallback if None)
        if diffusivity is None:
            self.mobility_model = MobilityModel(dbf, elements, phases[0], user_constants=user_constants)
            if len(self.solute_elements) == 1:
                solute = self.solute_elements[0]
                def matrix_diff_callable(temp):
                    d_trace = self.mobility_model.tracer_diffusivity(temp, initial_composition)
                    return d_trace[solute]
                self.therm.setDiffusivity(matrix_diff_callable, phases[0])
            else:
                diff_dict = {}
                for el in [e for e in elements if e != 'VA']:
                    diff_dict[el] = lambda temp, element=el: self.mobility_model.tracer_diffusivity(temp, initial_composition)[element]
                self.therm.setDiffusivity(diff_dict, phases[0])
        elif callable(diffusivity):
            self.therm.setDiffusivity(diffusivity, phases[0])
        else:
            self.therm.setDiffusivity(lambda *args: diffusivity, phases[0])
            
        # 3. Setup Matrix Parameters (resolve molar volume VM from TDB if None)
        self.matrix = MatrixParameters(solute_elements)
        vm_matrix = molar_volume if molar_volume is not None else resolve_molar_volume(dbf, phases[0], default_val=1e-5)
        self.matrix.volume.setVolume(vm_matrix, 'VM', 4)
        
        # Set composition
        if isinstance(initial_composition, dict):
            self.matrix.initComposition = [initial_composition[e] for e in solute_elements]
        else:
            self.matrix.initComposition = initial_composition
            
        # 4. Setup Precipitate Parameters
        self.precipitates = []
        for p in phases[1:]:
            prec = PrecipitateParameters(p)
            prec.gamma = interfacial_energy
            vm_prec = molar_volume if molar_volume is not None else resolve_molar_volume(dbf, p, default_val=vm_matrix)
            prec.volume.setVolume(vm_prec, 'VM', 4)
            self.precipitates.append(prec)
            
        # 5. Create PrecipitateModel
        self.model = PrecipitateModel(self.matrix, self.precipitates, self.therm, self.temperature_params)

    def add_strength_model(self, dislocation_params=None, contributions=None, ss_model=None, sigma0=0.0):
        """
        Couple a Yield Strength Model dynamically to the precipitation kinetics simulation.
        """
        from kawin.precipitation.coupling import StrengthModel, DislocationParameters, OrowanContribution
        if dislocation_params is None:
            dislocation_params = DislocationParameters(b=2.86e-10, G=26e9, nu=0.33)  # default aluminum values
        if contributions is None:
            contributions = [OrowanContribution()]

        self.strength_model = StrengthModel(
            self.precipitates, contributions, dislocation_params,
            ssModel=ss_model, sigma0=sigma0
        )
        self.model.addCouplingModel(self.strength_model)
        
    def step(self, sim_time, verbose=False):
        """
        Solve/run the precipitation kinetics simulation for a given duration.
        """
        self.model.solve(sim_time, verbose=verbose)
        
    def get_results(self):
        """
        Retrieve simulation results.
        
        Returns
        -------
        dict
            Dictionary containing time, volume fraction, mean radius, precipitate density,
            and optionally yield strength.
        """
        res = {
            'time': np.array(self.model.data.time),
            'precipitate_density': np.array(self.model.data.precipitateDensity),
            'volume_fraction': np.array(self.model.data.volFrac),
            'mean_radius': np.array(self.model.data.Ravg)
        }
        if hasattr(self, 'strength_model'):
            res['yield_strength'] = np.array(self.strength_model.totalStrength(self.model))
        return res
        
    def plot_results(self, save_path=None):
        """
        Create a multi-panel plot showing the kinetics of precipitation.
        """
        res = self.get_results()
        time = res['time']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, prec in enumerate(self.precipitates):
            # Precipitate density
            axes[0].plot(time, res['precipitate_density'][:, idx], label=prec.name, linewidth=2)
            # Volume fraction
            axes[1].plot(time, res['volume_fraction'][:, idx] * 100, label=prec.name, linewidth=2)
            # Mean radius
            axes[2].plot(time, res['mean_radius'][:, idx] * 1e9, label=prec.name, linewidth=2)
            
        axes[0].set_ylabel('Precipitate Density (#/m$^3$)', fontsize=12)
        axes[0].set_xlabel('Time (s)', fontsize=12)
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[0].set_yscale('log')
        
        axes[1].set_ylabel('Volume Fraction (%)', fontsize=12)
        axes[1].set_xlabel('Time (s)', fontsize=12)
        axes[1].grid(True, linestyle='--', alpha=0.6)
        
        axes[2].set_ylabel('Mean Radius (nm)', fontsize=12)
        axes[2].set_xlabel('Time (s)', fontsize=12)
        axes[2].grid(True, linestyle='--', alpha=0.6)
        
        plt.suptitle(f'Precipitation Kinetics at {self.T} K', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Saved precipitation kinetics plot to {save_path}")
        else:
            plt.show()
