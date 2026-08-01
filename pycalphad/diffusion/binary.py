import warnings
import logging
import numpy as np
import pycalphad
from pycalphad import equilibrium
import pycalphad.variables as v
from fipy import Grid1D, CellVariable, TransientTerm, DiffusionTerm

logger = logging.getLogger(__name__)

class BinaryDiffusionSimulation:
    """
    1D Binary Diffusion Simulator coupling pycalphad and FiPy (Deprecated: use DiffusionCoupleSimulation instead).
    """
    def __init__(self, dbf, comps, phase, T, Lx, nx, x_left, x_right, 
                 mobility_A=1.0e-13, mobility_B=1.0e-13):
        warnings.warn(
            "BinaryDiffusionSimulation is deprecated and will be removed in a future release. "
            "Please use pycalphad.diffusion.couple.DiffusionCoupleSimulation instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.dbf = dbf
        # Filter active components (remove VA if present to identify elements A and B)
        self.elements = [c for c in comps if c != 'VA']
        if len(self.elements) != 2:
            raise ValueError("BinaryDiffusionSimulation requires exactly two active elements (excluding VA).")
            
        self.comps = comps
        self.phase = phase
        self.T = T
        self.Lx = Lx
        self.nx = nx
        self.dx = Lx / nx
        self.mobility_A = mobility_A
        self.mobility_B = mobility_B
        
        # Element A is index 0, Element B is index 1 (independent)
        self.A = self.elements[0]
        self.B = self.elements[1]
        
        # 1. Pre-calculate chemical potentials using a vectorized equilibrium call
        # to construct fast interpolating functions for thermodynamic derivatives.
        logger.info("Pre-calculating thermodynamic functions...")

        x_grid = np.linspace(0.005, 0.995, 200)
        conditions = {
            v.T: self.T,
            v.P: 101325.0,
            v.X(self.B): x_grid
        }
        
        eq = equilibrium(self.dbf, self.comps, [self.phase], conditions)
        
        # Extract chemical potentials
        # Squeeze out size-1 dimensions
        mu_A_vals = np.squeeze(eq.MU.sel(component=self.A).values)
        mu_B_vals = np.squeeze(eq.MU.sel(component=self.B).values)
        
        # Store thermodynamic lookup tables
        self._x_lookup = x_grid
        self._mu_A_lookup = mu_A_vals
        self._mu_B_lookup = mu_B_vals
        
        # 2. Set up FiPy FVM 1D grid and variable
        self.mesh = Grid1D(dx=self.dx, nx=self.nx)
        
        # Define initial composition profile (step function / diffusion couple)
        initial_val = np.zeros(self.nx)
        initial_val[:self.nx // 2] = x_left
        initial_val[self.nx // 2:] = x_right
        
        self.x_B_var = CellVariable(mesh=self.mesh, value=initial_val, name=f"Mole fraction of {self.B}")
        
    def _get_thermo_derivative(self, xb):
        """
        Compute d(mu_B - mu_A) / d x_B using linear interpolation.
        """
        dx = 1e-4
        xb_plus = np.clip(xb + dx, 0.006, 0.994)
        xb_minus = np.clip(xb - dx, 0.006, 0.994)
        
        mu_A_plus = np.interp(xb_plus, self._x_lookup, self._mu_A_lookup)
        mu_B_plus = np.interp(xb_plus, self._x_lookup, self._mu_B_lookup)
        
        mu_A_minus = np.interp(xb_minus, self._x_lookup, self._mu_A_lookup)
        mu_B_minus = np.interp(xb_minus, self._x_lookup, self._mu_B_lookup)
        
        diff_plus = mu_B_plus - mu_A_plus
        diff_minus = mu_B_minus - mu_A_minus
        
        return (diff_plus - diff_minus) / (2.0 * dx)
        
    def _get_mobility(self, xb, mob_param):
        if callable(mob_param):
            return mob_param(xb)
        return mob_param
        
    def get_interdiffusion_coefficient(self, xb):
        """
        Compute the interdiffusion coefficient D_tilde(xb).
        D = ( (1 - x_B)*M_B + x_B*M_A ) * x_B * d(mu_B - mu_A)/dx_B
        """
        # Ensure xb is clipped to valid range to avoid lookup errors
        xb = np.clip(xb, 0.006, 0.994)
        
        ma = self._get_mobility(xb, self.mobility_A)
        mb = self._get_mobility(xb, self.mobility_B)
        
        d_mu_dx = self._get_thermo_derivative(xb)
        
        # Darken-Manning interdiffusion coefficient expression
        d_tilde = ((1.0 - xb) * mb + xb * ma) * xb * d_mu_dx
        return np.clip(d_tilde, 1e-25, 1e-10)
        
    def step(self, dt):
        """
        Advance the diffusion simulation by one time step dt.
        """
        # Evaluate D_eff at current cell center concentrations
        xb_vals = np.array(self.x_B_var.value)
        d_eff = self.get_interdiffusion_coefficient(xb_vals)
        
        # Wrap d_eff as a CellVariable for FiPy compatibility
        d_eff_var = CellVariable(mesh=self.mesh, value=d_eff)
        
        # Solve transient diffusion PDE in FiPy using harmonic face values
        eq = TransientTerm() == DiffusionTerm(coeff=d_eff_var.harmonicFaceValue)
        eq.solve(var=self.x_B_var, dt=dt)
        
    def get_profile(self):
        """
        Return the grid coordinates and current mole fractions of B.
        """
        # Get face centers or cell centers
        x_coords = np.array(self.mesh.cellCenters[0])
        xb_vals = np.array(self.x_B_var.value)
        return x_coords, xb_vals
