"""Weather system — deterministic per-season weather from overlapping sinusoidal cycles.

Generates seasonal weather conditions (temperature, precipitation, frost, storm
intensity) by evaluating a set of overlapping sinusoidal cycles whose parameters
are derived deterministically from the world seed.  The cycles operate at
timescales from a few years (ENSO-like oscillations) to tens of thousands of
years (Milankovitch-like drift), producing weather that has both short-term
variability and deep historical trends.

The base climate (temperature from latitude + lapse rate, precipitation from
orographic effects) is computed once from the geology heightmap.  Each call to
``generate()`` evaluates all cycles at the requested (year, season) and applies
the resulting anomalies spatially, respecting topographic amplification (valleys
amplify temperature swings, mountains amplify precipitation swings).

All randomness flows through ``create_rng`` with tier_id="weather", ensuring
full reproducibility from the world seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi, sin

import numpy as np

from bike_sim.rng import create_rng


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class WeatherCycle:
    """A single sinusoidal cycle contributing to weather variability."""

    period: float  # years
    amplitude: float  # strength (unitless multiplier or absolute offset)
    phase: float  # 0 to 2*pi, derived from seed
    target: str  # "temperature", "precipitation", or "both"
    correlation: float  # for "both": positive = warm&wet, negative = warm&dry


@dataclass
class SeasonalWeather:
    """Output for one season at one point in time."""

    temperature: np.ndarray  # (H, W) mean temp in C for this season
    precipitation: np.ndarray  # (H, W) total precip in mm for this season
    frost_severity: np.ndarray  # (H, W) frost damage potential [0, 1]
    storm_intensity: float  # world-level scalar, drives erosion
    season: int  # 0=winter, 1=spring, 2=summer, 3=fall


# ---------------------------------------------------------------------------
# Seasonal modulation constants
# ---------------------------------------------------------------------------

# (temp_scale, precip_scale) per season index
_SEASON_SCALES = {
    0: (0.6, 0.8),   # winter
    1: (0.85, 1.1),  # spring
    2: (1.2, 0.7),   # summer
    3: (0.9, 1.0),   # fall
}

_STORM_SCALE_FACTOR = 2.0


# ---------------------------------------------------------------------------
# WeatherSystem
# ---------------------------------------------------------------------------


class WeatherSystem:
    """Deterministic weather generator driven by overlapping sinusoidal cycles.

    Parameters
    ----------
    world_seed : int
        Master seed for the world — all cycle parameters derive from this.
    geology_heightmap : np.ndarray
        (H, W) elevation in metres from the geology tier.
    grid_size : int
        Number of cells along each axis.
    cell_size : float
        Metres per cell edge.
    """

    def __init__(
        self,
        world_seed: int,
        geology_heightmap: np.ndarray,
        grid_size: int = 1000,
        cell_size: float = 50.0,
    ) -> None:
        self._world_seed = world_seed
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.cycles = self._generate_cycles(world_seed)
        self._base_temp, self._base_precip = self._compute_base_climate(
            geology_heightmap
        )
        self._valley_depth = self._compute_valley_depth(geology_heightmap)
        self._elevation_factor = np.clip(
            geology_heightmap / (geology_heightmap.max() + 1e-10), 0, 1
        )
        # Orographic gradient (used for rain-shadow dampening)
        self._dx_heightmap = np.gradient(geology_heightmap, axis=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, year: float, season: int) -> SeasonalWeather:
        """Produce deterministic weather for a given year and season.

        Parameters
        ----------
        year : float
            Simulation year (may be fractional for sub-year resolution).
        season : int
            0=winter, 1=spring, 2=summer, 3=fall.

        Returns
        -------
        SeasonalWeather
        """
        time = year + season * 0.25
        temp_anomaly, precip_multiplier = self._evaluate_cycles(time)

        temp_scale, precip_scale = _SEASON_SCALES[season]

        # --- Temperature ---
        # Seasonal modulation of base, then anomaly with valley amplification.
        temperature = self._base_temp * temp_scale
        valley_amplifier = 1.0 + self._valley_depth * 0.3
        temperature = temperature + temp_anomaly * valley_amplifier

        # --- Precipitation ---
        # Seasonal modulation of base, then multiplicative anomaly with
        # mountain amplification and rain-shadow dampening.
        base_p = self._base_precip * precip_scale

        # Mountains amplify the multiplier deviation from 1.0
        mountain_boost = 1.0 + self._elevation_factor * 0.2
        spatial_multiplier = 1.0 + (precip_multiplier - 1.0) * mountain_boost

        # Rain shadow: dampen on lee side (where terrain drops away, dx > 0
        # means descending eastward — a simplistic model).  We reduce the
        # multiplier toward 1.0 on lee slopes.
        lee_mask = np.clip(self._dx_heightmap / 100.0, 0, 1)  # 0-1 lee factor
        spatial_multiplier = spatial_multiplier * (1.0 - lee_mask * 0.3)

        precipitation = base_p * spatial_multiplier
        precipitation = np.clip(precipitation, 0, None)

        # --- Frost severity ---
        frost_severity = self._compute_frost(temperature, year, season)

        # --- Storm intensity ---
        storm_intensity = max(0.0, precip_multiplier - 1.0) * _STORM_SCALE_FACTOR

        return SeasonalWeather(
            temperature=temperature.astype(np.float64),
            precipitation=precipitation.astype(np.float64),
            frost_severity=frost_severity.astype(np.float64),
            storm_intensity=float(storm_intensity),
            season=season,
        )

    def get_cycles(self) -> list[WeatherCycle]:
        """Return the list of cycles for serialization / inspection."""
        return list(self.cycles)

    # ------------------------------------------------------------------
    # Cycle generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_cycles(world_seed: int) -> list[WeatherCycle]:
        """Deterministically generate 8-9 weather cycles from the world seed."""
        rng = create_rng(world_seed, "weather", "cycles", 0)
        cycles: list[WeatherCycle] = []

        def _add(
            period_lo: float,
            period_hi: float,
            amp_lo: float,
            amp_hi: float,
            target: str,
            correlation: float = 0.0,
        ) -> None:
            cycles.append(
                WeatherCycle(
                    period=rng.uniform(period_lo, period_hi),
                    amplitude=rng.uniform(amp_lo, amp_hi),
                    phase=rng.uniform(0, 2 * pi),
                    target=target,
                    correlation=correlation,
                )
            )

        # Short cycles
        _add(3, 7, 0.2, 0.3, "precipitation")
        _add(4, 8, 0.5, 1.5, "temperature")

        # Medium cycles
        _add(25, 50, 0.15, 0.25, "precipitation")
        _add(30, 70, 1.0, 2.0, "temperature")

        # Compound cycles (1-2)
        n_compound = rng.integers(1, 3)  # 1 or 2
        for _ in range(int(n_compound)):
            corr = rng.uniform(-1.0, 1.0)
            _add(15, 40, 0.1, 0.15, "both", correlation=corr)

        # Long cycles
        _add(200, 500, 0.1, 0.2, "precipitation")
        _add(300, 800, 2.0, 4.0, "temperature")

        # Very long drift
        corr = rng.uniform(-1.0, 1.0)
        _add(50_000, 150_000, 3.0, 6.0, "both", correlation=corr)

        return cycles

    # ------------------------------------------------------------------
    # Cycle evaluation
    # ------------------------------------------------------------------

    def _evaluate_cycles(self, time: float) -> tuple[float, float]:
        """Evaluate all cycles at a given time, returning anomalies.

        Returns
        -------
        temp_anomaly : float
            Additive temperature anomaly in degrees C.
        precip_multiplier : float
            Multiplicative precipitation factor (always > 0).
        """
        temp_anomaly = 0.0
        precip_log_anomaly = 0.0  # work in log space for multiplicative

        for cycle in self.cycles:
            value = cycle.amplitude * sin(
                2 * pi * time / cycle.period + cycle.phase
            )
            if cycle.target == "temperature":
                temp_anomaly += value
            elif cycle.target == "precipitation":
                precip_log_anomaly += value
            elif cycle.target == "both":
                temp_anomaly += value
                precip_log_anomaly += value * cycle.correlation

        precip_multiplier = exp(precip_log_anomaly)  # always positive
        return temp_anomaly, precip_multiplier

    # ------------------------------------------------------------------
    # Base climate (absorbed from ClimateHydrologyTier._compute_climate)
    # ------------------------------------------------------------------

    def _compute_base_climate(
        self, heightmap: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute baseline temperature and precipitation from terrain.

        Temperature: latitude gradient (15 C south to 5 C north) minus
        elevation lapse rate (6.5 C/km).

        Precipitation: base 800 mm/yr + orographic effects + elevation factor.

        No random noise — the weather cycles replace that.
        """
        h, w = heightmap.shape

        # Temperature: latitude gradient minus lapse rate
        y_coords = np.linspace(0, 1, h).reshape(-1, 1)
        base_temp = 15.0 - 10.0 * y_coords  # south-to-north gradient
        lapse_rate = 6.5 / 1000.0  # C per metre
        temperature = base_temp - lapse_rate * heightmap

        # Precipitation: orographic + elevation
        base_precip = 800.0
        dx = np.gradient(heightmap, axis=1)
        orographic = np.clip(dx * 0.5, -200, 400)
        elev_factor = np.clip(heightmap / 1500.0, 0, 1) * 400
        precipitation = base_precip + orographic + elev_factor
        precipitation = np.clip(precipitation, 50, 4000)

        return temperature.astype(np.float64), precipitation.astype(np.float64)

    # ------------------------------------------------------------------
    # Valley depth (topographic amplification factor)
    # ------------------------------------------------------------------

    def _compute_valley_depth(self, heightmap: np.ndarray) -> np.ndarray:
        """Compute how much lower each cell is than its local neighbourhood.

        Returns a non-negative array where 0 = at or above local mean,
        larger values = deeper valleys.  Normalised to [0, 1].
        """
        # Local mean via box blur (kernel_size ~ 15 cells)
        local_mean = _box_blur(heightmap, passes=5)
        depth = np.clip(local_mean - heightmap, 0, None)
        max_depth = depth.max()
        if max_depth > 0:
            depth = depth / max_depth
        return depth.astype(np.float64)

    # ------------------------------------------------------------------
    # Frost model
    # ------------------------------------------------------------------

    def _compute_frost(
        self, temperature: np.ndarray, year: float, season: int
    ) -> np.ndarray:
        """Compute frost severity field.

        Frost risk is a sigmoid of how far temperature is below a threshold.
        A stochastic spatial perturbation (deterministic from seed) is added,
        scaled by the magnitude of the temperature anomaly.
        """
        frost_threshold = 2.0  # C — frost becomes likely below this

        # Sigmoid: 1 / (1 + exp(-(threshold - temp)))
        frost_risk = _sigmoid(frost_threshold - temperature)

        # Stochastic frost event
        tick_encoding = int(year * 4 + season)
        rng = create_rng(self._world_seed, "weather", "frost", tick_encoding)

        # Sigma scales with absolute temperature anomaly (colder = more variable)
        _, precip_mult = self._evaluate_cycles(year + season * 0.25)
        temp_anom, _ = self._evaluate_cycles(year + season * 0.25)
        sigma = 0.05 + 0.05 * min(abs(temp_anom), 5.0)  # cap contribution

        frost_event = rng.normal(0, sigma, temperature.shape)
        frost_severity = np.clip(frost_risk + frost_event, 0, 1)

        return frost_severity


# ---------------------------------------------------------------------------
# Module-level utility functions
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Element-wise sigmoid, numerically stable."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _box_blur(arr: np.ndarray, passes: int = 1) -> np.ndarray:
    """Simple 3x3 box blur applied multiple times (approximates Gaussian)."""
    result = arr.copy()
    for _ in range(passes):
        padded = np.pad(result, 1, mode="edge")
        result = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 9.0
    return result
