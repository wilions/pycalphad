import warnings
import numpy as np
from tinydb import where
from scipy.interpolate import CubicSpline
from pycalphad import variables as v, equilibrium
from kawin.thermo import MulticomponentThermodynamics, BinaryThermodynamics
from kawin.thermo.Mobility import (
    interdiffusivity as kawin_interdiffusivity,
    interdiffusivity_from_diff as kawin_interdiffusivity_from_diff
)
from kawin.thermo.FreeEnergyHessian import dMudX

class MobilityDataError(ValueError):
    """Exception raised when required mobility or diffusivity data is missing from TDB."""
    def __init__(self, elements, phase):
        self.elements = elements
        self.phase = phase
        super().__init__(f"Missing mobility/diffusivity parameters for elements {elements} in phase {phase}")

class MobilityModel:
    """
    Facade over Kawin's mobility models, supporting multicomponent systems,
    user-constant overrides, and spline-interpolated thermodynamic factors.
    """
    def __init__(self, dbf, comps, phase, user_constants=None):
        self.dbf = dbf
        self.comps = comps
        self.phase = phase
        self.user_constants = user_constants or {}

        # Standard elements list (excluding vacancies)
        self.elements = [c for c in comps if c != 'VA']
        self.ref_element = self.elements[0]
        self.independent_elements = self.elements[1:]
        self.num_elements = len(self.elements)

        # Trigger warning if user overrides are active
        if self.user_constants:
            warnings.warn(
                f"Using user constant diffusivity/mobility overrides: {self.user_constants}", 
                UserWarning
            )

        # Verify database parameters for elements not in user overrides
        missing_elements = []
        for el in self.elements:
            if el in self.user_constants:
                continue
            # Search for MQ/MF (mobility) or DQ/DF (diffusivity) parameters
            query_mob = dbf.search(
                (where('phase_name') == phase) &
                ((where('parameter_type') == 'MQ') | (where('parameter_type') == 'MF')) &
                (where('diffusing_species') == v.Species(el))
            )
            query_diff = dbf.search(
                (where('phase_name') == phase) &
                ((where('parameter_type') == 'DQ') | (where('parameter_type') == 'DF')) &
                (where('diffusing_species') == v.Species(el))
            )
            if len(query_mob) == 0 and len(query_diff) == 0:
                missing_elements.append(el)

        if missing_elements:
            raise MobilityDataError(missing_elements, phase)

        # Initialize Kawin's thermodynamics class
        if self.num_elements == 2:
            self.therm = BinaryThermodynamics(dbf, self.elements, [phase])
        else:
            self.therm = MulticomponentThermodynamics(dbf, self.elements, [phase])

        # Pre-compute thermodynamic factor spline for binary systems
        self.curv_spline = None
        if self.num_elements == 2:
            self._setup_binary_spline()

    def _setup_binary_spline(self):
        """Build 1D spline interpolation of free energy curvature (dmu_B/dx_B) for binary system."""
        solute = self.independent_elements[0]
        x_grid = np.linspace(0.001, 0.999, 200)
        curv_grid = []
        
        for xv in x_grid:
            # Evaluate chemical potential derivative at xv
            # We use Kawin's local equilibrium solver to extract the thermodynamic factor
            res, comp_sets = self.therm.getLocalEq(xv, 1000.0, 0, [self.phase])
            cs = comp_sets[0]
            # Reference element is self.ref_element (elements[0])
            dmudx = dMudX(res.chemical_potentials, cs, self.ref_element)
            curv_grid.append(float(dmudx[0, 0]))
            
        self.x_grid = x_grid
        self.curv_spline = CubicSpline(x_grid, curv_grid)

    def _build_callables(self, T):
        """Construct element mobility and diffusivity callables merging TDB and user_constants."""
        mob_callables = {}
        diff_callables = {}
        R = 8.314

        for el in self.elements:
            if el in self.user_constants:
                val = self.user_constants[el]
                # Treat user constants as tracer diffusivity D
                # D = M * R * T => M = D / (R * T)
                diff_callables[el] = lambda dof, v_const=val: v_const
                mob_callables[el] = lambda dof, v_const=val, temp=T: v_const / (R * temp)
            else:
                # Use compiled callables from TDB
                if self.therm.mobCallables[self.phase] and el in self.therm.mobCallables[self.phase]:
                    mob_callables[el] = self.therm.mobCallables[self.phase][el]
                if self.therm.diffCallables[self.phase] and el in self.therm.diffCallables[self.phase]:
                    diff_callables[el] = self.therm.diffCallables[self.phase][el]

        return mob_callables, diff_callables

    def tracer_diffusivity(self, T, x):
        """
        Evaluate tracer diffusivity at temperature T and composition x.
        
        Parameters
        ----------
        T : float
            Temperature in Kelvin.
        x : dict or list or float
            Mole fractions of independent components.
            
        Returns
        -------
        dict[element, float]
        """
        # Convert list/float x to list format
        if isinstance(x, (float, np.float64)):
            x_list = [x]
        elif isinstance(x, dict):
            x_list = [x[el] for el in self.independent_elements]
        else:
            x_list = list(x)

        # Get local equilibrium state
        result, comp_sets = self.therm.getLocalEq(x_list, T, 0, [self.phase])
        cs = comp_sets[0]

        mob_calls, diff_calls = self._build_callables(T)

        # Retrieve tracer diffusivities
        R = 8.314
        d_trace = {}
        for idx, el in enumerate(self.elements):
            if el in self.user_constants:
                d_trace[el] = self.user_constants[el]
            elif self.therm.mobCallables[self.phase] and el in mob_calls:
                # D = M * R * T
                mob_val = mob_calls[el](cs.dof)
                d_trace[el] = mob_val * R * T
            elif self.therm.diffCallables[self.phase] and el in diff_calls:
                d_trace[el] = diff_calls[el](cs.dof)
            else:
                d_trace[el] = 1e-15  # safe fallback

        return d_trace

    def interdiffusivity_matrix(self, T, x):
        """
        Evaluate interdiffusivity matrix at temperature T and composition x.
        
        Parameters
        ----------
        T : float
            Temperature in Kelvin.
        x : dict or list or float
            Mole fractions of independent components.
            
        Returns
        -------
        (n-1)×(n-1) ndarray
        """
        if isinstance(x, (float, np.float64)):
            x_list = [x]
        elif isinstance(x, dict):
            x_list = [x[el] for el in self.independent_elements]
        else:
            x_list = list(x)

        # Get local equilibrium state
        result, comp_sets = self.therm.getLocalEq(x_list, T, 0, [self.phase])
        cs = comp_sets[0]

        mob_calls, diff_calls = self._build_callables(T)

        if self.num_elements == 2:
            # For binary systems, utilize spline-interpolated curvature
            xb = x_list[0]
            if self.curv_spline is not None:
                curv = float(self.curv_spline(xb))
            else:
                curv = float(dMudX(result.chemical_potentials, cs, self.ref_element)[0, 0])

            R = 8.314
            # Retrieve mobilities of element A (ref) and B (independent)
            ma = mob_calls[self.ref_element](cs.dof) if self.ref_element in mob_calls else diff_calls[self.ref_element](cs.dof) / (R * T)
            mb = mob_calls[self.independent_elements[0]](cs.dof) if self.independent_elements[0] in mob_calls else diff_calls[self.independent_elements[0]](cs.dof) / (R * T)

            # Darken-Manning formula: D_tilde = ((1 - xb) * mb + xb * ma) * xb * dmu_B/dx_B
            d_tilde = ((1.0 - xb) * mb + xb * ma) * xb * curv
            return np.array([[d_tilde]])

        else:
            # For multicomponent systems, use Kawin's analytical Hessian formulation
            if self.therm.mobCallables[self.phase]:
                from kawin.thermo.Mobility import inverseMobility
                Dnkj, _, _ = inverseMobility(
                    result.chemical_potentials, cs, self.ref_element,
                    mobility_callables=mob_calls,
                    parameters=self.therm._parameters
                )
            else:
                from kawin.thermo.Mobility import inverseMobility_from_diffusivity
                diff_corr = {el: 1.0 for el in self.elements}
                Dnkj, _, _ = inverseMobility_from_diffusivity(
                    result.chemical_potentials, cs, self.ref_element,
                    diffusivity_callables=diff_calls,
                    diffusivity_correction=diff_corr,
                    parameters=self.therm._parameters
                )

            # Reorder Dnkj columns/rows to match independent_elements list input order
            sort_indices = np.argsort(self.elements[1:])
            unsort_indices = np.argsort(sort_indices)
            Dnkj = Dnkj[unsort_indices, :]
            Dnkj = Dnkj[:, unsort_indices]
            return Dnkj
