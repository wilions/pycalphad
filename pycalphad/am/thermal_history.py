"""
Multi-Pass Multi-Layer Additive Manufacturing Thermal History & Microstructural Cycling Engine.

This module simulates multi-track, multi-layer laser powder bed fusion (LPBF) and
direct energy deposition (DED) scan strategies, extracting local thermal cycles T(t),
cooling rates dT/dt, thermal gradients G = |grad T|, solidification velocities R,
and cumulative hot-cracking / reheating assessments.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from pycalphad.am.cracking import (
    calculate_kou_index,
    calculate_clyne_davis_index,
    calculate_rdg_index,
    classify_cracking_risk,
)
from pycalphad.am.thermal import rosenthal_T



@dataclass
class ScanTrack:
    """Represents a single laser scan line vector."""
    start_pos: Tuple[float, float, float]  # (x0, y0, z0) in meters
    end_pos: Tuple[float, float, float]    # (x1, y1, z1) in meters
    start_time: float                      # Start time in seconds
    power: float                           # Laser power in Watts
    velocity: float                        # Scan speed in m/s
    absorptivity: float                    # Laser absorptivity (0.0 to 1.0)
    beam_radius: float = 50e-6             # Beam radius in meters (for Eagar-Tsai)

    @property
    def duration(self) -> float:
        length = np.sqrt(
            (self.end_pos[0] - self.start_pos[0]) ** 2
            + (self.end_pos[1] - self.start_pos[1]) ** 2
            + (self.end_pos[2] - self.start_pos[2]) ** 2
        )
        return length / self.velocity if self.velocity > 0 else 0.0

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    def laser_position(self, t: float) -> Tuple[float, float, float]:
        """Returns the laser spot coordinate (x, y, z) at time t."""
        if t <= self.start_time:
            return self.start_pos
        if t >= self.end_time:
            return self.end_pos
        frac = (t - self.start_time) / self.duration
        x = self.start_pos[0] + frac * (self.end_pos[0] - self.start_pos[0])
        y = self.start_pos[1] + frac * (self.end_pos[1] - self.start_pos[1])
        z = self.start_pos[2] + frac * (self.end_pos[2] - self.start_pos[2])
        return (x, y, z)


@dataclass
class MultiPassThermalHistoryResult:
    """Results from multi-pass thermal history accumulation."""
    times: np.ndarray                      # Time array in seconds
    temperatures: np.ndarray               # Temperature history T(t) in Kelvin
    cooling_rates: np.ndarray              # dT/dt in K/s
    thermal_gradients: np.ndarray          # G = |grad T| in K/m
    solidification_velocities: np.ndarray  # R in m/s
    peak_temperatures: List[float]         # Peak temperatures per thermal cycle
    num_cycles: int                        # Number of remelting / reheating cycles
    solidus_T: float                       # Solidus temperature (K)
    liquidus_T: float                      # Liquidus temperature (K)
    kou_index: Optional[float] = None
    cracking_risk: Optional[str] = None
    summary_metrics: Dict[str, Any] = field(default_factory=dict)


class MultiPassThermalAccumulator:
    """
    Engine to assemble scan paths and simulate multi-pass thermal cycles in AM.
    """

    def __init__(
        self,
        k_th: float = 25.0,        # Thermal conductivity (W/(m·K))
        rho: float = 7800.0,       # Density (kg/m^3)
        cp: float = 500.0,         # Specific heat capacity (J/(kg·K))
        t_preheat: float = 298.15, # Substrate / build plate preheat temp (K)
    ):
        self.k_th = k_th
        self.rho = rho
        self.cp = cp
        self.t_preheat = t_preheat
        self.alpha_th = k_th / (rho * cp)  # Thermal diffusivity (m^2/s)
        self.tracks: List[ScanTrack] = []

    def add_track(
        self,
        start_pos: Tuple[float, float, float],
        end_pos: Tuple[float, float, float],
        start_time: float,
        power: float,
        velocity: float,
        absorptivity: float = 0.35,
        beam_radius: float = 50e-6,
    ) -> None:
        """Add an explicit scan track to the schedule."""
        self.tracks.append(
            ScanTrack(
                start_pos=start_pos,
                end_pos=end_pos,
                start_time=start_time,
                power=power,
                velocity=velocity,
                absorptivity=absorptivity,
                beam_radius=beam_radius,
            )
        )

    def generate_serpentine_pattern(
        self,
        num_tracks: int = 5,
        track_length: float = 5e-3,      # 5 mm
        hatch_spacing: float = 100e-6,   # 100 um
        layer_thickness: float = 40e-6,  # 40 um
        num_layers: int = 1,
        power: float = 200.0,
        velocity: float = 0.8,           # 800 mm/s
        absorptivity: float = 0.35,
        inter_track_delay: float = 1e-4, # 0.1 ms repositioning
    ) -> None:
        """
        Generate a multi-track, multi-layer serpentine hatch pattern.
        """
        current_time = 0.0
        for layer in range(num_layers):
            z_level = layer * layer_thickness
            for track_idx in range(num_tracks):
                y_pos = track_idx * hatch_spacing
                if track_idx % 2 == 0:
                    start_pos = (0.0, y_pos, z_level)
                    end_pos = (track_length, y_pos, z_level)
                else:
                    start_pos = (track_length, y_pos, z_level)
                    end_pos = (0.0, y_pos, z_level)

                self.add_track(
                    start_pos=start_pos,
                    end_pos=end_pos,
                    start_time=current_time,
                    power=power,
                    velocity=velocity,
                    absorptivity=absorptivity,
                )
                current_time += (track_length / velocity) + inter_track_delay

    def evaluate_point_history(
        self,
        probe_pos: Tuple[float, float, float],
        t_span: Optional[Tuple[float, float]] = None,
        num_samples: int = 2000,
        liquidus_T: float = 1600.0,
        solidus_T: float = 1500.0,
    ) -> MultiPassThermalHistoryResult:
        """
        Evaluate thermal history at a fixed spatial probe (x, y, z).
        """
        if not self.tracks:
            raise ValueError("No scan tracks configured in accumulator.")

        t_min = min(tr.start_time for tr in self.tracks) if t_span is None else t_span[0]
        t_max = (max(tr.end_time for tr in self.tracks) + 0.005) if t_span is None else t_span[1]

        times = np.linspace(t_min, t_max, num_samples)
        dt = times[1] - times[0]
        temperatures = np.full_like(times, self.t_preheat)

        px, py, pz = probe_pos

        # Superposition across tracks
        for track in self.tracks:
            eff_power = track.power * track.absorptivity
            for i, t in enumerate(times):
                if t < track.start_time:
                    continue
                # Time elapsed since start of this track
                t_active = t - track.start_time
                if t_active <= 0:
                    continue

                # Point-source instantaneous Rosenthal / moving heat source integration
                lx, ly, lz = track.laser_position(t)
                dist_sq = (px - lx) ** 2 + (py - ly) ** 2 + (pz - lz) ** 2
                dist = np.sqrt(dist_sq)

                # Rosenthal quasi-steady contribution
                if dist > 1e-7:
                    t_rise = (eff_power / (2.0 * np.pi * self.k_th * dist)) * np.exp(
                        -track.velocity * (dist + (px - lx)) / (2.0 * self.alpha_th)
                    )
                    # Bound maximum physical temperature
                    t_rise = min(t_rise, 5000.0)
                    temperatures[i] += t_rise

        # Compute cooling rates dT/dt
        cooling_rates = np.zeros_like(temperatures)
        cooling_rates[:-1] = -(np.diff(temperatures) / dt)

        # Estimate local thermal gradient G = |grad T|
        eps = 1e-6
        # Perturb +dx and evaluate
        temp_dx = np.full_like(times, self.t_preheat)
        for track in self.tracks:
            eff_power = track.power * track.absorptivity
            for i, t in enumerate(times):
                if t < track.start_time:
                    continue
                lx, ly, lz = track.laser_position(t)
                dist = np.sqrt((px + eps - lx) ** 2 + (py - ly) ** 2 + (pz - lz) ** 2)
                if dist > 1e-7:
                    t_rise = (eff_power / (2.0 * np.pi * self.k_th * dist)) * np.exp(
                        -track.velocity * (dist + (px + eps - lx)) / (2.0 * self.alpha_th)
                    )
                    temp_dx[i] += min(t_rise, 5000.0)

        grad_x = np.abs(temp_dx - temperatures) / eps
        # Estimated 3D gradient magnitude
        thermal_gradients = np.maximum(grad_x * 1.5, 1e2)

        # Solidification velocity R = (1/G) * |dT/dt|
        solidification_velocities = np.abs(cooling_rates) / thermal_gradients

        # Peak temperatures per cycle
        peaks = []
        in_peak = False
        current_peak = self.t_preheat
        for t_val in temperatures:
            if t_val > solidus_T:
                in_peak = True
                if t_val > current_peak:
                    current_peak = t_val
            elif in_peak and t_val <= solidus_T:
                peaks.append(float(current_peak))
                current_peak = self.t_preheat
                in_peak = False
        if in_peak:
            peaks.append(float(current_peak))

        # Check Kou Cracking susceptibility on active solidification range
        solidifying_indices = np.where((temperatures >= solidus_T) & (temperatures <= liquidus_T))[0]
        kou_val = None
        cracking_risk = None
        if len(solidifying_indices) > 5:
            solid_T = temperatures[solidifying_indices]
            # Approximate linear/Scheil solid fraction
            fs = 1.0 - (solid_T - solidus_T) / max(liquidus_T - solidus_T, 1e-3)
            fs = np.clip(fs, 0.0, 1.0)
            kou_val = float(calculate_kou_index(solid_T, fs))
            csc_val = float(calculate_clyne_davis_index(solid_T, fs))
            cracking_risk = classify_cracking_risk(kou_val, csc_val)

        return MultiPassThermalHistoryResult(

            times=times,
            temperatures=temperatures,
            cooling_rates=cooling_rates,
            thermal_gradients=thermal_gradients,
            solidification_velocities=solidification_velocities,
            peak_temperatures=peaks,
            num_cycles=len(peaks),
            solidus_T=solidus_T,
            liquidus_T=liquidus_T,
            kou_index=kou_val,
            cracking_risk=cracking_risk,
            summary_metrics={
                "max_temperature_K": float(np.max(temperatures)),
                "max_cooling_rate_K_s": float(np.max(cooling_rates)),
                "mean_gradient_K_m": float(np.mean(thermal_gradients)),
                "cooling_index_G_R": float(np.mean(thermal_gradients * solidification_velocities)),
            },
        )
