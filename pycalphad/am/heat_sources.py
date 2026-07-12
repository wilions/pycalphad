import numpy as np

class GaussianHeatSource:
    """
    Gaussian Surface Heat Source model.
    """
    def __init__(self, power, absorptivity, beam_radius):
        self.power = power
        self.absorptivity = absorptivity
        self.r0 = beam_radius

    def __call__(self, x, y):
        """
        Calculate surface heat flux q(x, y) in W/m^2.
        """
        coeff = (2.0 * self.power * self.absorptivity) / (np.pi * self.r0**2)
        exponent = -2.0 * (x**2 + y**2) / (self.r0**2)
        return coeff * np.exp(exponent)


class DoubleEllipsoidalHeatSource:
    """
    Goldak Volumetric Double Ellipsoidal Heat Source model.
    """
    def __init__(self, power, absorptivity, af, ar, b, c):
        self.power = power
        self.absorptivity = absorptivity
        self.af = af
        self.ar = ar
        self.b = b
        self.c = c
        
        # Energy distribution factors front/rear to satisfy power conservation
        self.ff = 2.0 * self.af / (self.af + self.ar)
        self.fr = 2.0 * self.ar / (self.af + self.ar)

    def __call__(self, x, y, z):
        """
        Calculate volumetric heat source density q(x, y, z) in W/m^3.
        z points downwards (z >= 0).
        """
        # Ensure coordinates are numpy arrays
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        
        q = np.zeros_like(x, dtype=float)
        
        # Front ellipsoid (x >= 0)
        coeff_f = (6.0 * np.sqrt(3.0) * self.ff * self.power * self.absorptivity) / (self.af * self.b * self.c * np.pi * np.sqrt(np.pi))
        exponent_f = -3.0 * (x**2 / self.af**2 + y**2 / self.b**2 + z**2 / self.c**2)
        mask_f = (x >= 0)
        q[mask_f] = coeff_f * np.exp(exponent_f[mask_f])
        
        # Rear ellipsoid (x < 0)
        coeff_r = (6.0 * np.sqrt(3.0) * self.fr * self.power * self.absorptivity) / (self.ar * self.b * self.c * np.pi * np.sqrt(np.pi))
        exponent_r = -3.0 * (x**2 / self.ar**2 + y**2 / self.b**2 + z**2 / self.c**2)
        mask_r = (x < 0)
        q[mask_r] = coeff_r * np.exp(exponent_r[mask_r])
        
        return q


class ConicalHeatSource:
    """
    Volumetric Conical Heat Source model.
    """
    def __init__(self, power, absorptivity, re, ri, H):
        self.power = power
        self.absorptivity = absorptivity
        self.re = re # top radius
        self.ri = ri # bottom radius
        self.H = H   # depth of cone

    def __call__(self, x, y, z):
        """
        Calculate volumetric heat source density q(x, y, z) in W/m^3.
        z points downwards (0 <= z <= H).
        """
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        
        q = np.zeros_like(x, dtype=float)
        
        # Conical domain mask
        mask = (z >= 0) & (z <= self.H)
        
        # Dynamic beam radius as linear function of depth z
        rc_z = self.re - (self.re - self.ri) * (z / self.H)
        
        # Normalization factor for power density conservation
        volume_integral = np.pi * self.H * (self.re**2 + self.re * self.ri + self.ri**2) / 3.0
        coeff = (9.0 * self.power * self.absorptivity) / (np.pi * self.H * (self.re**2 + self.re * self.ri + self.ri**2))
        
        # Compute exponential distribution
        exponent = -3.0 * (x**2 + y**2) / (rc_z**2)
        
        # Linear decay with depth to represent penetration absorption decay
        linear_decay = 1.0 - z / self.H
        
        q[mask] = coeff * linear_decay[mask] * np.exp(exponent[mask])
        
        return q
