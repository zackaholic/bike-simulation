"""Weather system — deterministic per-season weather from fractal noise (fBm).

Generates seasonal weather conditions (temperature, precipitation, frost, storm
intensity) by evaluating 1D fractional Brownian motion (fBm) at the requested
time.  Multiple octaves of value noise produce aperiodic climate variability
with a red-noise power spectrum: low frequencies (century-scale trends) dominate,
overlaid with progressively smaller high-frequency variation.  Every "climate
epoch" is unique — unlike sinusoidal cycles, the noise never repeats.

The base climate (temperature from latitude + lapse rate, precipitation from
orographic effects) is computed once from the geology heightmap.  Each call to
``generate()`` evaluates the fractal noise at the requested (year, season) and
applies the resulting anomalies spatially, respecting topographic amplification
(valleys amplify temperature swings, mountains amplify precipitation swings).

All randomness flows through ``create_rng`` with tier_id="weather", ensuring
full reproducibility from the world seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np

from bike_sim.rng import create_rng


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


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
# 1D fractal noise helpers
# ---------------------------------------------------------------------------


def _hash_float(seed: int, i: int) -> float:
    """Deterministic float in [-1, 1] from seed and integer position.

    Uses murmurhash-style bit mixing, masked to 32 bits so Python's
    arbitrary-precision ints behave like a hardware hash.
    """
    h = (seed ^ (i * 0x9E3779B9)) & 0xFFFFFFFF
    h = (((h >> 16) ^ h) * 0x45D9F3B) & 0xFFFFFFFF
    h = (((h >> 16) ^ h) * 0x45D9F3B) & 0xFFFFFFFF
    h = ((h >> 16) ^ h) & 0xFFFFFFFF
    return (h & 0xFFFFFF) / 0x800000 - 1.0  # map to [-1, 1]


def _smoothstep(t: float) -> float:
    """Smooth interpolation curve: 3t² - 2t³."""
    return t * t * (3.0 - 2.0 * t)


def _noise1d(seed: int, t: float) -> float:
    """Deterministic 1D value noise at position *t*, seeded.

    Hashes floor(t) and ceil(t) to get two deterministic values,
    then smoothstep-interpolates between them.
    """
    i = int(np.floor(t))
    frac = t - i
    v0 = _hash_float(seed, i)
    v1 = _hash_float(seed, i + 1)
    return v0 + (v1 - v0) * _smoothstep(frac)


# ---------------------------------------------------------------------------
# WeatherSystem
# ---------------------------------------------------------------------------


class WeatherSystem:
    """Deterministic weather generator driven by fractal noise (fBm).

    Parameters
    ----------
    world_seed : int
        Master seed for the world — fractal parameters derive from this.
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
        moisture_bias: np.ndarray | None = None,
        continentality: np.ndarray | None = None,
    ) -> None:
        self._world_seed = world_seed
        self.grid_size = grid_size
        self.cell_size = cell_size

        # Fractal noise parameters
        self._num_octaves = 7
        self._base_freq = 1.0 / 800.0   # lowest octave wavelength ~800 years
        self._lacunarity = 2.0           # each octave doubles in frequency
        self._persistence = 0.55         # each octave has 55% of previous amplitude
        self._base_amp_temp = 3.0        # lowest octave: ±3°C
        self._base_amp_precip = 0.15     # lowest octave: ±15% precip (log space)

        # Independent noise seeds for temperature and precipitation
        rng = create_rng(world_seed, "weather", "fractal", 0)
        self._seed_temp = int(rng.integers(0, 2**31))
        self._seed_precip = int(rng.integers(0, 2**31))

        # Spatial climate bias fields (None = no spatial variation, backward compat)
        self._moisture_bias = moisture_bias     # [0.5, 2.0] multiplicative
        self._continentality = continentality   # [0.0, 1.0] temp extreme scaling

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
        temp_anomaly, precip_multiplier = self._evaluate_climate(time)

        temp_scale, precip_scale = _SEASON_SCALES[season]

        # --- Temperature ---
        # Seasonal modulation of base, then anomaly with valley amplification.
        temperature = self._base_temp * temp_scale
        valley_amplifier = 1.0 + self._valley_depth * 0.3

        # Continentality scales temperature anomaly extremes:
        # high continentality = full anomaly effect (continental climate)
        # low continentality = dampened anomaly (maritime climate)
        if self._continentality is not None:
            anomaly_scale = 0.7 + 0.6 * self._continentality
            temperature = temperature + temp_anomaly * valley_amplifier * anomaly_scale
        else:
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

        # Apply moisture bias: structurally wetter/drier regions
        if self._moisture_bias is not None:
            precipitation = precipitation * self._moisture_bias

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

    def get_climate_params(self) -> dict:
        """Return fractal climate parameters for serialization / inspection."""
        return {
            "num_octaves": self._num_octaves,
            "base_freq": self._base_freq,
            "lacunarity": self._lacunarity,
            "persistence": self._persistence,
            "base_amp_temp": self._base_amp_temp,
            "base_amp_precip": self._base_amp_precip,
            "seed_temp": self._seed_temp,
            "seed_precip": self._seed_precip,
        }

    # ------------------------------------------------------------------
    # Fractal climate evaluation
    # ------------------------------------------------------------------

    def _evaluate_climate(self, time: float) -> tuple[float, float]:
        """Evaluate fractal noise at *time*, returning climate anomalies.

        Returns
        -------
        temp_anomaly : float
            Additive temperature anomaly in degrees C.
        precip_multiplier : float
            Multiplicative precipitation factor (always > 0).
        """
        temp_anomaly = 0.0
        precip_anomaly = 0.0

        for octave in range(self._num_octaves):
            freq = self._base_freq * (self._lacunarity ** octave)
            amp_t = self._base_amp_temp * (self._persistence ** octave)
            amp_p = self._base_amp_precip * (self._persistence ** octave)

            temp_anomaly += _noise1d(self._seed_temp, time * freq) * amp_t
            precip_anomaly += _noise1d(self._seed_precip, time * freq) * amp_p

        precip_multiplier = exp(precip_anomaly)  # always positive

        # Safety caps — wide enough that they rarely bind
        temp_anomaly = max(-8.0, min(temp_anomaly, 8.0))
        precip_multiplier = max(0.25, min(precip_multiplier, 3.0))
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

        # Temperature: latitude gradient minus lapse rate.
        # Base range 20-10°C (warm temperate latitude) ensures terrain at
        # ~1000m mean elevation still has warm growing seasons.
        y_coords = np.linspace(0, 1, h).reshape(-1, 1)
        base_temp = 20.0 - 10.0 * y_coords  # south-to-north gradient
        lapse_rate = 5.0 / 1000.0  # C per metre (slightly reduced for variety)
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
        temp_anom, precip_mult = self._evaluate_climate(year + season * 0.25)
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
