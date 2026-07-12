import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pycalphad.amkit.mobility import MobilityModel
from pycalphad import variables as v

class DiffusionCoupleSimulation:
    """
    1D Multicomponent Diffusion Couple Simulator supporting arbitrary number of elements,
    coupling to CALPHAD thermodynamic and mobility databases, with a robust fully implicit FD solver.
    """
    def __init__(self, dbf, comps, phase, T, Lx, nx, x_left, x_right, user_constants=None, solver='numpy'):
        self.dbf = dbf
        self.comps = comps
        self.phase = phase
        self.T = T
        self.Lx = Lx
        self.nx = nx
        self.dx = Lx / nx
        self.solver_name = solver

        # Standard elements list (excluding vacancies)
        self.elements = [c for c in comps if c != 'VA']
        self.ref_element = self.elements[0]
        self.independent_elements = self.elements[1:]
        self.num_independent = len(self.independent_elements)

        # Standardize compositions
        self.x_left = self._standardize_composition(x_left)
        self.x_right = self._standardize_composition(x_right)

        # Grid setup (cell centers)
        self.x_coords = np.linspace(self.dx / 2.0, Lx - self.dx / 2.0, nx)

        # Initialize profiles for each independent element
        self.x_profiles = {}
        for el in self.independent_elements:
            profile = np.zeros(nx)
            profile[:nx // 2] = self.x_left[el]
            profile[nx // 2:] = self.x_right[el]
            self.x_profiles[el] = profile

        # Initialize mobility model
        self.mobility_model = MobilityModel(dbf, comps, phase, user_constants=user_constants)

        # If solver is 'fipy' and system is binary, initialize FiPy structures
        if self.solver_name == 'fipy' and self.num_independent == 1:
            from fipy import Grid1D, CellVariable
            self.mesh = Grid1D(dx=self.dx, nx=self.nx)
            el = self.independent_elements[0]
            initial_val = np.zeros(self.nx)
            initial_val[:self.nx // 2] = self.x_left[el]
            initial_val[self.nx // 2:] = self.x_right[el]
            self.x_B_var = CellVariable(mesh=self.mesh, value=initial_val, name=f"Mole fraction of {el}")

    def _standardize_composition(self, comp):
        """Standardize input composition mapping element names (str) to float values."""
        if isinstance(comp, (float, np.float64)):
            # Binary system case where composition is passed as a float
            return {self.independent_elements[0]: comp}
        elif isinstance(comp, dict):
            std = {}
            for k, val in comp.items():
                if isinstance(k, v.X):
                    std[k.species.name] = val
                else:
                    std[str(k).upper()] = val
            return std
        else:
            # List or array input matching independent_elements order
            return {el: val for el, val in zip(self.independent_elements, comp)}

    def get_interdiffusion_coefficient(self, xb):
        """
        Evaluate the interdiffusivity coefficient(s) at composition xb.
        For binary systems, returns a float. For multicomponent, returns the full matrix.
        """
        # For binary backward compatibility where xb is a scalar
        if isinstance(xb, (float, np.float64)) and self.num_independent == 1:
            mat = self.mobility_model.interdiffusivity_matrix(self.T, xb)
            return float(mat[0, 0])

        # Evaluate at full composition dictionary/array
        return self.mobility_model.interdiffusivity_matrix(self.T, xb)

    def step(self, dt):
        """Advance the diffusion couple profile by one time step dt."""
        if self.solver_name == 'fipy' and self.num_independent == 1:
            # FiPy solver for binary system (compatibility path)
            from fipy import CellVariable, TransientTerm, DiffusionTerm
            xb_vals = np.array(self.x_B_var.value)
            d_eff = np.array([self.get_interdiffusion_coefficient(val) for val in xb_vals])
            d_eff_var = CellVariable(mesh=self.mesh, value=d_eff)
            
            eq = TransientTerm() == DiffusionTerm(coeff=d_eff_var.harmonicFaceValue)
            eq.solve(var=self.x_B_var, dt=dt)
            # Update internal profiles
            self.x_profiles[self.independent_elements[0]] = np.array(self.x_B_var.value)
        else:
            # Fully implicit 1D block-sparse NumPy/SciPy solver for general multicomponent diffusion
            self._step_implicit_numpy(dt)

    def _step_implicit_numpy(self, dt):
        N = self.nx
        Nc = self.num_independent
        dx = self.dx
        C = dt / (dx**2)

        # 1. Compute interdiffusivity matrices at each cell center
        D_grid = []
        for i in range(N):
            # Construct composition dictionary for cell i
            x_cell = {el: self.x_profiles[el][i] for el in self.independent_elements}
            D = self.mobility_model.interdiffusivity_matrix(self.T, x_cell)
            D_grid.append(D)
        
        # 2. Compute face values of interdiffusivities (arithmetic mean interpolation)
        D_faces = []
        for i in range(N - 1):
            D_faces.append(0.5 * (D_grid[i] + D_grid[i + 1]))

        # 3. Assemble global sparse system A * x^{m+1} = x^m
        # Variable ordering: V = [x_00, x_01, ..., x_0,N-1, x_10, ..., x_Nc-1,N-1]
        A_rows = []
        A_cols = []
        A_data = []

        # Right-hand side vector
        RHS = np.zeros(N * Nc)

        # Loop over cells
        for i in range(N):
            # Loop over components
            for k in range(Nc):
                row_idx = k * N + i
                RHS[row_idx] = self.x_profiles[self.independent_elements[k]][i]

                # Self-contribution of time derivative
                A_rows.append(row_idx)
                A_cols.append(row_idx)
                A_data.append(1.0)

                # Loop over components for coupling
                for j in range(Nc):
                    col_self = j * N + i
                    col_prev = j * N + (i - 1)
                    col_next = j * N + (i + 1)

                    # Compute boundary face values
                    D_prev = D_faces[i - 1][k, j] if i > 0 else 0.0
                    D_next = D_faces[i][k, j] if i < N - 1 else 0.0

                    # Contribution to self node (i)
                    A_rows.append(row_idx)
                    A_cols.append(col_self)
                    A_data.append(C * (D_prev + D_next))

                    # Contribution to previous node (i-1)
                    if i > 0:
                        A_rows.append(row_idx)
                        A_cols.append(col_prev)
                        A_data.append(-C * D_prev)

                    # Contribution to next node (i+1)
                    if i < N - 1:
                        A_rows.append(row_idx)
                        A_cols.append(col_next)
                        A_data.append(-C * D_next)

        # Solve sparse system
        A = sp.coo_matrix((A_data, (A_rows, A_cols)), shape=(N * Nc, N * Nc)).tocsr()
        V_new = spla.spsolve(A, RHS)

        # Update profiles
        for k in range(Nc):
            self.x_profiles[self.independent_elements[k]] = V_new[k * N : (k + 1) * N]

    def get_profile(self, element=None):
        """
        Return the coordinates and composition profile for a given element.
        If element is None, returns profile for the first independent element (compatibility path).
        """
        if element is None:
            element = self.independent_elements[0]
        else:
            element = str(element).upper()

        if element == self.ref_element:
            # Sum of independent elements subtraction
            tot = np.zeros(self.nx)
            for el in self.independent_elements:
                tot += self.x_profiles[el]
            return self.x_coords, 1.0 - tot

        return self.x_coords, self.x_profiles[element]
