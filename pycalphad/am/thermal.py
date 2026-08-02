import numpy as np
from .heat_sources import GaussianHeatSource

def solve_thermal_profile(Lx, Ly, dx, dy, t_max, 
                          laser_power, absorptivity, beam_radius, scan_speed,
                          density, specific_heat, conductivity,
                          x0=0.0, y0=0.0, T_ambient=298.15, h_loss=10.0, dt=None,
                          latent_heat=None, solidus_temp=None, liquidus_temp=None,
                          return_history=False, strict_stability=True):
    """
    Solve the 2D transient heat equation for a moving laser pass on the surface.
    
    Parameters
    ----------
    Lx, Ly : float
        Domain length in x and y directions (meters).
    dx, dy : float
        Grid spacing (meters).
    t_max : float
        Total simulation time (seconds).
    laser_power : float
        Laser power in Watts.
    absorptivity : float
        Laser absorptivity (0 to 1).
    beam_radius : float
        Laser beam radius (meters).
    scan_speed : float
        Laser scanning speed along x-axis (meters/second).
    density : float or callable
        Density (kg/m^3) or callable f(T).
    specific_heat : float or callable
        Specific heat capacity (J/kg/K) or callable f(T).
    conductivity : float or callable
        Thermal conductivity (W/m/K) or callable f(T).
    x0, y0 : float
        Initial position of the laser (meters).
    T_ambient : float
        Ambient/substrate temperature (Kelvin).
    h_loss : float
        Heat loss coefficient from surface (W/m^2/K).
    dt : float, optional
        Time step (seconds). If None, calculated automatically for stability.
    latent_heat : float, optional
        Latent heat of fusion (J/kg) to include solidification phase change.
    solidus_temp, liquidus_temp : float, optional
        Solidus and liquidus temperatures (Kelvin) for phase change.
    return_history : bool, optional
        If True, returns the temperature history at each time step.
        
    Returns
    -------
    X, Y : 2D ndarray
        Grid coordinates.
    T : 2D ndarray
        Final temperature distribution.
    t_axis : 1D ndarray
        Simulation time steps.
    T_history : list of 2D ndarray, optional
        Returned only if return_history is True.
    """
    # Grid coordinates
    nx = int(Lx / dx) + 1
    ny = int(Ly / dy) + 1
    x_axis = np.linspace(0, Lx, nx)
    y_axis = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x_axis, y_axis)
    
    # Initialize temperature field
    T = np.full((ny, nx), T_ambient, dtype=float)
    T_history = [T.copy()] if return_history else None
    
    # Helper to evaluate temperature-dependent properties
    def get_prop(prop, T_val):
        if callable(prop):
            return prop(T_val)
        return prop

    # Determine time step dt based on stability criterion if not specified
    rho0 = get_prop(density, T_ambient)
    cp0 = get_prop(specific_heat, T_ambient)
    k0 = get_prop(conductivity, T_ambient)
    alpha0 = k0 / (rho0 * cp0)
    max_dt = 1.0 / (2.0 * alpha0 * (1.0 / dx**2 + 1.0 / dy**2))

    if dt is None:
        dt = 0.25 * min(dx**2, dy**2) / alpha0
    else:
        if dt > max_dt:
            msg = f"Time step dt ({dt} s) exceeds explicit solver stability limit ({max_dt} s)."
            if strict_stability:
                raise ValueError(msg)
            else:
                import warnings
                warnings.warn(msg, UserWarning)
        
    nt = int(t_max / dt) + 1
    t_axis = np.linspace(0, t_max, nt)
    
    # Instantiate Gaussian heat source
    source = GaussianHeatSource(laser_power, absorptivity, beam_radius)
    
    # Time loop (Explicit Euler integration)
    instability_count = 0
    for step in range(1, nt):
        t = t_axis[step]
        
        # Current laser center position (moving along x-axis)
        x_laser = x0 + scan_speed * t
        y_laser = y0
        
        # Calculate laser heat input at each grid point
        Q = source(X - x_laser, Y - y_laser)
        
        # Evaluate properties at current temperature (vectorized!)
        rho = density(T) if callable(density) else np.full_like(T, density)
        cp = specific_heat(T) if callable(specific_heat) else np.full_like(T, specific_heat)
        k = conductivity(T) if callable(conductivity) else np.full_like(T, conductivity)
        
        # Model latent heat of fusion via effective heat capacity method
        if latent_heat is not None and solidus_temp is not None and liquidus_temp is not None:
            mushy_mask = (T >= solidus_temp) & (T <= liquidus_temp)
            cp[mushy_mask] += latent_heat / (liquidus_temp - solidus_temp)
            
        # Thermal diffusivity
        alpha = k / (rho * cp)
        
        # Temperature laplacian using central differences (with boundary padding)
        d2T_dx2 = np.zeros_like(T)
        d2T_dy2 = np.zeros_like(T)
        
        # Inner nodes
        d2T_dx2[:, 1:-1] = (T[:, 2:] - 2.0 * T[:, 1:-1] + T[:, :-2]) / dx**2
        d2T_dy2[1:-1, :] = (T[2:, :] - 2.0 * T[1:-1, :] + T[:-2, :]) / dy**2
        
        # Boundary conditions (Adiabatic / Neuman BC)
        # Left/Right
        d2T_dx2[:, 0] = (2.0 * T[:, 1] - 2.0 * T[:, 0]) / dx**2
        d2T_dx2[:, -1] = (2.0 * T[:, -2] - 2.0 * T[:, -1]) / dx**2
        # Top/Bottom
        d2T_dy2[0, :] = (2.0 * T[1, :] - 2.0 * T[0, :]) / dy**2
        d2T_dy2[-1, :] = (2.0 * T[-2, :] - 2.0 * T[-1, :]) / dy**2
        
        # Heat loss term (convection/radiation modeled as linear loss coefficient)
        loss = h_loss * (T - T_ambient) / (rho * cp * dy)
        
        # Power source rate (divided by dy as the thickness to maintain unit consistency)
        source_rate = Q / (rho * cp * dy)
        
        # Update temperature
        T_new = T + dt * (alpha * (d2T_dx2 + d2T_dy2) + source_rate - loss)
        
        # Safeguard against stability crash
        if np.any(T_new > 50000.0) or np.any(T_new < T_ambient - 1.0):
            instability_count += 1
            msg = f"Numerical instability detected: temperature went out of bounds. Computed stable dt limit is {max_dt} s."
            if strict_stability or instability_count > 10:
                raise ValueError(msg)
            else:
                import warnings
                warnings.warn(msg, UserWarning)
        else:
            instability_count = 0

        T_new = np.clip(T_new, T_ambient, 50000.0)
        T = T_new
        
        if return_history:
            T_history.append(T.copy())

            
    if return_history:
        return X, Y, T, t_axis, T_history
    return X, Y, T, t_axis

def extract_thermal_history_probe(X, Y, T_history, t_axis, probe_x, probe_y):
    """
    Extract the time-temperature history T(t) at a specific coordinate (probe_x, probe_y).
    
    Parameters
    ----------
    X, Y : 2D ndarray
        Grid coordinates.
    T_history : list of 2D ndarray
        Temperature distributions at each time step.
    t_axis : 1D ndarray
        Time axis.
    probe_x, probe_y : float
        Coordinates of the probe.
        
    Returns
    -------
    t_axis : 1D ndarray
        Time axis (seconds).
    T_profile : 1D ndarray
        Temperature profile (Kelvin) over time.
    """
    dist = (X - probe_x)**2 + (Y - probe_y)**2
    idx_y, idx_x = np.unravel_index(np.argmin(dist), dist.shape)
    
    T_profile = np.array([T_step[idx_y, idx_x] for T_step in T_history])
    return t_axis, T_profile


def rosenthal_T(x, y, z, power_eff, scan_speed, density, specific_heat, conductivity, T_ambient=298.15):
    """
    3D Rosenthal analytical moving point heat source solution for temperature field T(x,y,z).
    """
    alpha = conductivity / (density * specific_heat)
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.maximum(r, 1e-9)
    term = np.exp(-scan_speed * (x + r) / (2.0 * alpha))
    delta_T = (power_eff / (2.0 * np.pi * conductivity * r)) * term
    return T_ambient + delta_T


def predict_lpbf_melt_pool(power, speed, spot_diameter, density, specific_heat, conductivity,
                           melting_temp, boiling_temp, absorptivity=0.35, T_ambient=298.15,
                           layer_thickness=500e-6, hatch_spacing=1200e-6):
    """
    Predicts melt pool width, depth, length, aspect ratio, normalized enthalpy, and VED
    using PyCALPHAD's analytical 3D thermal moving heat source solver.
    """
    p_eff = power * absorptivity
    delta_t = max(10.0, melting_temp - T_ambient)
    vol_cap = density * specific_heat
    alpha = conductivity / vol_cap

    v_mps = max(1e-4, speed)
    w_m = np.sqrt(8.0 * p_eff / (np.pi * np.e * vol_cap * delta_t * v_mps))
    w_m = max(spot_diameter * 0.8, w_m)
    d_m_cond = w_m / 2.0

    denom = np.pi * vol_cap * delta_t * np.sqrt(max(1e-8, alpha * v_mps * (spot_diameter**3)))
    norm_enthalpy = p_eff / denom if denom > 0 else 0.0
    keyhole_thresh = np.pi * boiling_temp / melting_temp

    if norm_enthalpy > keyhole_thresh:
        overheat = norm_enthalpy / keyhole_thresh
        d_m = d_m_cond * (overheat ** 0.85)
    else:
        d_m = d_m_cond

    r_scale = p_eff / (2.0 * np.pi * conductivity * delta_t)
    pe = (v_mps * (spot_diameter / 2.0)) / (2.0 * alpha)
    l_m = max(w_m, 2.0 * r_scale * (1.0 + 0.5 * pe))

    ved = power / (speed * 1e3 * hatch_spacing * 1e3 * layer_thickness * 1e3) if speed > 0 else 0.0

    return {
        "width_m": float(w_m),
        "depth_m": float(d_m),
        "length_m": float(l_m),
        "width_um": float(w_m * 1e6),
        "depth_um": float(d_m * 1e6),
        "length_um": float(l_m * 1e6),
        "aspect_ratio": float(d_m / w_m) if w_m > 0 else 0.0,
        "normalized_enthalpy": float(norm_enthalpy),
        "keyhole_threshold": float(keyhole_thresh),
        "ved_j_mm3": float(ved),
    }
