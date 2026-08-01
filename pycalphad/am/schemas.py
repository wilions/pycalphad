"""
Shared data schemas and result containers for AM physical models.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import numpy as np

@dataclass
class ThermalHistory:
    """
    Encapsulates temperature-time traces, cooling rates, and solidification parameters.
    """
    time: np.ndarray
    temperature: np.ndarray
    spatial_coords: Optional[Dict[str, float]] = None
    
    @property
    def peak_temperature(self) -> float:
        """Maximum temperature experienced during thermal cycle (K)."""
        return float(np.max(self.temperature))

    def cooling_rate(self, T_ref: float) -> float:
        """
        Calculates dT/dt (K/s) near a reference temperature T_ref.
        Returns positive value for cooling rate.
        """
        if len(self.time) < 2:
            return 0.0
        
        idx = np.argmin(np.abs(self.temperature - T_ref))
        if idx == 0:
            dT = self.temperature[1] - self.temperature[0]
            dt = self.time[1] - self.time[0]
        elif idx == len(self.time) - 1:
            dT = self.temperature[-1] - self.temperature[-2]
            dt = self.time[-1] - self.time[-2]
        else:
            dT = self.temperature[idx + 1] - self.temperature[idx - 1]
            dt = self.time[idx + 1] - self.time[idx - 1]
            
        rate = dT / dt if dt > 0 else 0.0
        return float(-rate) if rate < 0 else float(rate)

    def time_above(self, T_threshold: float) -> float:
        """Total duration (seconds) temperature remains above T_threshold."""
        mask = self.temperature >= T_threshold
        if not np.any(mask):
            return 0.0
        indices = np.where(mask)[0]
        t_start = self.time[indices[0]]
        t_end = self.time[indices[-1]]
        return float(t_end - t_start)


@dataclass
class MicrostructureState:
    """
    Represents phase fractions, precipitate metrics, and microsegregation state.
    """
    solid_fraction: float = 0.0
    liquid_fraction: float = 1.0
    phase_fractions: Dict[str, float] = field(default_factory=dict)
    mean_radius: float = 0.0  # meters
    volume_fraction: float = 0.0  # precipitate fraction
    number_density: float = 0.0  # m^-3
    composition_profiles: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """
    Unified result object for AM physical simulations.
    """
    simulation_type: str
    grid: Optional[Dict[str, np.ndarray]] = None
    time: Optional[np.ndarray] = None
    field_data: Dict[str, np.ndarray] = field(default_factory=dict)
    scalar_outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result object into a JSON-serializable dictionary."""
        out = {
            "simulation_type": self.simulation_type,
            "scalar_outputs": self.scalar_outputs,
            "metadata": self.metadata,
        }
        if self.time is not None:
            out["time"] = self.time.tolist()
        if self.grid is not None:
            out["grid"] = {k: v.tolist() for k, v in self.grid.items()}
        out["fields"] = {}
        for k, v in self.field_data.items():
            if isinstance(v, np.ndarray):
                out["fields"][k] = v.tolist()
            else:
                out["fields"][k] = v
        return out
