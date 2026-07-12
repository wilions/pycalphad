import numpy as np
from scipy.integrate import quad

class ThermalHistory:
    """
    Dataclass representing time-temperature history at a specific probe location.
    """
    def __init__(self, t, T):
        self.t = np.array(t)
        self.T = np.array(T)

    def cooling_rate(self, T_ref):
        """
        Compute the cooling rate dT/dt (K/s) at reference temperature T_ref during cooling.
        """
        peak_idx = np.argmax(self.T)
        t_cooling = self.t[peak_idx:]
        T_cooling = self.T[peak_idx:]

        if len(T_cooling) < 2:
            return 0.0

        # Find index closest to T_ref
        idx = np.argmin(np.abs(T_cooling - T_ref))
        global_idx = peak_idx + idx

        # Numerical derivative
        if global_idx == 0:
            dt = self.t[1] - self.t[0]
            dT = self.T[1] - self.T[0]
        elif global_idx == len(self.t) - 1:
            dt = self.t[-1] - self.t[-2]
            dT = self.T[-1] - self.T[-2]
        else:
            dt = self.t[global_idx + 1] - self.t[global_idx - 1]
            dT = self.T[global_idx + 1] - self.T[global_idx - 1]

        # Cooling rate is positive (magnitude of cooling rate)
        return -dT / dt if dt != 0.0 else 0.0

    def time_above(self, T_thresh):
        """
        Calculate the total time (seconds) spent above temperature T_thresh.
        """
        above_mask = self.T > T_thresh
        indices = np.where(above_mask)[0]
        if len(indices) == 0:
            return 0.0
        
        t_entry = self.t[indices[0]]
        t_exit = self.t[indices[-1]]
        return float(t_exit - t_entry)

def rosenthal_T(x, y, z, P, v, density, specific_heat, conductivity, T_ambient=298.15, absorptivity=1.0):
    """
    Rosenthal moving point source solution in a semi-infinite solid.
    """
    alpha = conductivity / (density * specific_heat)
    R = np.sqrt(x**2 + y**2 + z**2)
    # Safeguard against singularity at origin
    R = np.maximum(R, 1e-9)
    dT = (P * absorptivity) / (2.0 * np.pi * conductivity * R) * np.exp(-v * (R + x) / (2.0 * alpha))
    return T_ambient + dT

def _eagar_tsai_integrand(u, x, y, z, P, v, beam_radius, alpha, conductivity, T_ambient, absorptivity):
    sigma = beam_radius / 2.0
    if u == 0:
        if z != 0:
            return 0.0
        return (1.0 / sigma**2) * np.exp(-(x**2 + y**2) / sigma**2)
    
    denom = 4.0 * alpha * u**2 + sigma**2
    exp_term = -((x + v * u**2)**2 + y**2) / denom - (z**2) / (4.0 * alpha * u**2)
    return (1.0 / denom) * np.exp(exp_term)

def eagar_tsai_T(x, y, z, P, v, beam_radius, density, specific_heat, conductivity, T_ambient=298.15, absorptivity=1.0):
    """
    Eagar-Tsai distributed heat source solution in a semi-infinite solid.
    """
    alpha = conductivity / (density * specific_heat)
    prefactor = (P * absorptivity) / (conductivity * np.pi * np.sqrt(np.pi))

    # Vectorize evaluation over coordinates
    x_arr = np.atleast_1d(x)
    y_arr = np.atleast_1d(y)
    z_arr = np.atleast_1d(z)

    max_len = max(len(x_arr), len(y_arr), len(z_arr))
    x_arr = np.broadcast_to(x_arr, (max_len,))
    y_arr = np.broadcast_to(y_arr, (max_len,))
    z_arr = np.broadcast_to(z_arr, (max_len,))

    results = []
    for xi, yi, zi in zip(x_arr, y_arr, z_arr):
        val, _ = quad(
            _eagar_tsai_integrand, 0.0, np.inf, 
            args=(xi, yi, zi, P, v, beam_radius, alpha, conductivity, T_ambient, absorptivity)
        )
        results.append(T_ambient + prefactor * val)

    if isinstance(x, (list, np.ndarray)) or isinstance(y, (list, np.ndarray)) or isinstance(z, (list, np.ndarray)):
        return np.array(results)
    return results[0]

def solidification_conditions(T_func, x, y, z, v, h=1e-5):
    """
    Compute solidification conditions (G, R, cooling rate) at the pool boundary.
    
    Parameters
    ----------
    T_func : callable
        Function T(x, y, z) returning temperature.
    x, y, z : float
        Coordinates of interest (rear melt pool boundary).
    v : float
        Laser scan speed.
    h : float
        Step size for finite differences.
        
    Returns
    -------
    G : float
        Thermal gradient magnitude (K/m).
    R : float
        Solidification rate (m/s).
    cooling_rate : float
        Cooling rate (K/s).
    """
    # Numerical gradient
    dT_dx = (T_func(x + h, y, z) - T_func(x - h, y, z)) / (2.0 * h)
    dT_dy = (T_func(x, y + h, z) - T_func(x, y - h, z)) / (2.0 * h)
    dT_dz = (T_func(x, y, z + h) - T_func(x, y, z - h)) / (2.0 * h)

    G = np.sqrt(dT_dx**2 + dT_dy**2 + dT_dz**2)
    # In steady state co-moving frame, dT/dt = -v * dT/dx
    # The cooling rate (as a positive quantity) is -dT/dt = v * dT/dx
    cooling_rate = v * dT_dx
    
    if G > 1e-9:
        R = cooling_rate / G
    else:
        R = 0.0

    return float(G), float(R), float(cooling_rate)
