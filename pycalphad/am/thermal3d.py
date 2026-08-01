"""
3D Finite-Volume Method (FVM) thermal simulation for additive manufacturing.
"""

from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from pycalphad.am.heat_sources import GaussianHeatSource, DoubleEllipsoidalHeatSource, ConicalHeatSource
from pycalphad.am.schemas import ThermalHistory, SimulationResult

class Thermal3DSimulation:
    """
    Transient 3D heat transfer solver using Finite Volume Method (FVM).
    
    Solves:
        rho(T) * c_p^*(T) * dT/dt = div(k(T) * grad(T)) + Q(x, y, z, t)
    where c_p^*(T) includes latent heat of phase transformation over mushy zone.
    """
    def __init__(
        self,
        domain_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        grid_shape: Tuple[int, int, int],
        thermal_conductivity: Union[float, Callable[[float], float]] = 30.0,
        heat_capacity: Union[float, Callable[[float], float]] = 500.0,
        density: Union[float, Callable[[float], float]] = 7800.0,
        latent_heat: float = 2.7e5,  # J/kg
        T_solidus: float = 1600.0,   # K
        T_liquidus: float = 1700.0,  # K
        T_ambient: float = 300.0     # K
    ):
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = domain_bounds
        nx, ny, nz = grid_shape
        
        self.nx, self.ny, self.nz = nx, ny, nz
        self.x = np.linspace(x_min, x_max, nx)
        self.y = np.linspace(y_min, y_max, ny)
        self.z = np.linspace(z_min, z_max, nz)
        
        self.dx = (x_max - x_min) / max(1, nx - 1)
        self.dy = (y_max - y_min) / max(1, ny - 1)
        self.dz = (z_max - z_min) / max(1, nz - 1)
        
        self.k_func = thermal_conductivity if callable(thermal_conductivity) else lambda T: float(thermal_conductivity)
        self.cp_func = heat_capacity if callable(heat_capacity) else lambda T: float(heat_capacity)
        self.rho_func = density if callable(density) else lambda T: float(density)
        
        self.latent_heat = latent_heat
        self.T_solidus = T_solidus
        self.T_liquidus = T_liquidus
        self.T_ambient = T_ambient
        
        self.T = np.full((nx, ny, nz), T_ambient, dtype=np.float64)

    def _apparent_heat_capacity(self, T: float) -> float:
        cp = self.cp_func(T)
        if self.T_solidus < T < self.T_liquidus and (self.T_liquidus > self.T_solidus):
            cp += self.latent_heat / (self.T_liquidus - self.T_solidus)
        return cp

    def step_explicit(
        self,
        dt: float,
        heat_source_func: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]] = None,
        current_time: float = 0.0
    ) -> float:
        """
        Executes one explicit forward Euler time step.
        Returns actual dt used.
        """
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # Volumetric heat generation (W/m^3)
        if heat_source_func is not None:
            Q = heat_source_func(X, Y, Z, current_time)
        else:
            Q = np.zeros_like(self.T)

        T_new = np.copy(self.T)
        
        # Internal 3D finite volume conduction stencil
        # d2T/dx2 + d2T/dy2 + d2T/dz2
        d2T_dx2 = (np.roll(self.T, -1, axis=0) - 2 * self.T + np.roll(self.T, 1, axis=0)) / (self.dx**2)
        d2T_dy2 = (np.roll(self.T, -1, axis=1) - 2 * self.T + np.roll(self.T, 1, axis=1)) / (self.dy**2)
        d2T_dz2 = (np.roll(self.T, -1, axis=2) - 2 * self.T + np.roll(self.T, 1, axis=2)) / (self.dz**2)
        
        # Vectorized temperature dependent properties
        cp_app = np.vectorize(self._apparent_heat_capacity)(self.T)
        rho_val = np.vectorize(self.rho_func)(self.T)
        k_val = np.vectorize(self.k_func)(self.T)
        
        alpha = k_val / (rho_val * cp_app)
        
        # Internal field update
        dT_dt = alpha * (d2T_dx2 + d2T_dy2 + d2T_dz2) + Q / (rho_val * cp_app)
        
        # Apply interior updates (excluding 1-cell Neumann boundaries)
        T_new[1:-1, 1:-1, 1:-1] = self.T[1:-1, 1:-1, 1:-1] + dt * dT_dt[1:-1, 1:-1, 1:-1]
        
        # Adiabatic / Neumann boundary conditions (dT/dn = 0)
        T_new[0, :, :] = T_new[1, :, :]
        T_new[-1, :, :] = T_new[-2, :, :]
        T_new[:, 0, :] = T_new[:, 1, :]
        T_new[:, -1, :] = T_new[:, -2, :]
        T_new[:, :, 0] = T_new[:, :, 1]
        T_new[:, :, -1] = T_new[:, :, -2]
        
        self.T = T_new
        return dt

    def solve(
        self,
        duration: float,
        dt: float,
        heat_source_func: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]] = None,
        probe_coords: Optional[Tuple[float, float, float]] = None
    ) -> SimulationResult:
        """
        Executes full transient 3D thermal simulation over duration (s).
        """
        steps = int(np.ceil(duration / dt))
        time_history = []
        probe_history = []
        
        current_time = 0.0
        
        # Determine probe index
        if probe_coords is not None:
            px, py, pz = probe_coords
            ix = int(np.clip(np.argmin(np.abs(self.x - px)), 0, self.nx - 1))
            iy = int(np.clip(np.argmin(np.abs(self.y - py)), 0, self.ny - 1))
            iz = int(np.clip(np.argmin(np.abs(self.z - pz)), 0, self.nz - 1))
        else:
            ix, iy, iz = self.nx // 2, self.ny // 2, self.nz // 2

        for step in range(steps):
            time_history.append(current_time)
            probe_history.append(float(self.T[ix, iy, iz]))
            
            self.step_explicit(dt, heat_source_func, current_time)
            current_time += dt

        grid = {"x": self.x, "y": self.y, "z": self.z}
        field_data = {"temperature_field": self.T}
        scalar_outputs = {
            "peak_temperature": float(np.max(self.T)),
            "probe_peak_temperature": float(np.max(probe_history)),
            "final_time": current_time
        }
        
        return SimulationResult(
            simulation_type="3d_thermal_fvm",
            grid=grid,
            time=np.array(time_history),
            field_data=field_data,
            scalar_outputs=scalar_outputs,
            metadata={"dt": dt, "probe_coords": (self.x[ix], self.y[iy], self.z[iz])}
        )
