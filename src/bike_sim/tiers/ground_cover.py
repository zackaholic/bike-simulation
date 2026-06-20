"""Ground cover layer — direct function of current climate conditions.

The ground cover layer represents low-level vegetation (grass, herbs, moss,
lichen) and bare ground states (soil, sand, rock). Unlike the canopy ecology
tier which simulates population dynamics, ground cover is a simple lookup
from current conditions: temperature, precipitation, season, and optionally
canopy density.

This layer guarantees the world is never visually empty. It changes with
seasons and climate cycles, providing an immediate visual signal of current
conditions without needing dispersal or competition mechanics.

Output:
  - ground_cover_type: categorical [0-7] indicating dominant ground cover
  - ground_cover_vigor: float [0, 1] indicating density/health

Types:
  0 = bare rock / ice (very cold or very dry)
  1 = lichen / sparse (cold + dry)
  2 = dry grass / scrub (warm + dry)
  3 = patchy grass (moderate)
  4 = meadow / herbs (cool + wet)
  5 = lush grass (warm + wet)
  6 = ferns / moss (shaded or very wet)
  7 = alpine meadow (cold + moderate moisture)
"""

from __future__ import annotations

import numpy as np

from bike_sim.weather import SeasonalWeather


def compute_ground_cover(
    weather: SeasonalWeather,
    canopy_density: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ground cover type and vigor from current conditions.

    Parameters
    ----------
    weather : SeasonalWeather
        Current season's weather (temperature, precipitation, frost, season).
    canopy_density : np.ndarray, optional
        Total canopy density across all species. If provided, shaded areas
        shift toward fern/moss types.

    Returns
    -------
    cover_type : np.ndarray (int32)
        Categorical ground cover type [0-7].
    vigor : np.ndarray (float64)
        Ground cover density/health [0, 1].
    """
    temp = weather.temperature
    precip = weather.precipitation
    season = weather.season

    shape = temp.shape

    # Normalize conditions to [0, 1] for classification
    # Temperature: -5°C → 0, 25°C → 1
    temp_norm = np.clip((temp + 5.0) / 30.0, 0, 1)
    # Precipitation: 0mm → 0, 1500mm → 1
    precip_norm = np.clip(precip / 1500.0, 0, 1)

    # Start with "patchy grass" everywhere
    cover_type = np.full(shape, 3, dtype=np.int32)
    vigor = np.full(shape, 0.5, dtype=np.float64)

    # --- Type classification ---

    # Very cold: bare rock/ice
    very_cold = temp < -2.0
    cover_type[very_cold] = 0

    # Cold + dry: lichen
    cold_dry = (temp >= -2.0) & (temp < 5.0) & (precip_norm < 0.3)
    cover_type[cold_dry] = 1

    # Cold + moderate moisture: alpine meadow
    alpine = (temp >= -2.0) & (temp < 8.0) & (precip_norm >= 0.3)
    cover_type[alpine] = 7

    # Warm + dry: dry grass
    warm_dry = (temp >= 8.0) & (precip_norm < 0.25)
    cover_type[warm_dry] = 2

    # Cool + wet: meadow/herbs
    cool_wet = (temp >= 5.0) & (temp < 15.0) & (precip_norm >= 0.4)
    cover_type[cool_wet] = 4

    # Warm + wet: lush grass
    warm_wet = (temp >= 15.0) & (precip_norm >= 0.4)
    cover_type[warm_wet] = 5

    # Very wet or shaded: ferns/moss
    if canopy_density is not None:
        shaded = canopy_density > 5.0  # significant canopy
        cover_type[shaded & (precip_norm >= 0.3)] = 6
    very_wet = precip_norm > 0.7
    cover_type[very_wet & (temp >= 5.0)] = 6

    # --- Vigor ---
    # Base vigor from temperature and moisture (geometric mean of suitability)
    temp_suit = np.clip(1.0 - np.abs(temp_norm - 0.5) * 2.0, 0.1, 1.0)
    precip_suit = np.clip(precip_norm * 2.0, 0.1, 1.0)
    vigor = np.sqrt(temp_suit * precip_suit)

    # Seasonal modulation
    season_scale = {0: 0.3, 1: 0.8, 2: 1.0, 3: 0.6}
    vigor *= season_scale.get(season, 0.7)

    # Frost suppresses vigor
    vigor *= (1.0 - weather.frost_severity * 0.5)

    # Very cold or very dry → minimal vigor
    vigor[very_cold] = 0.05
    vigor[cover_type == 1] = np.clip(vigor[cover_type == 1], 0.05, 0.2)
    vigor[cover_type == 2] = np.clip(vigor[cover_type == 2], 0.1, 0.5)

    vigor = np.clip(vigor, 0.0, 1.0)

    return cover_type.astype(np.int32), vigor.astype(np.float64)
