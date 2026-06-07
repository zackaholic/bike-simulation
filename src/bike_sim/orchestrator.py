"""Orchestrator — seasonal tick loop bridging ride logs and the three-tier simulation.

The Orchestrator coordinates geology, climate-hydrology, and ecology tiers
using seasonal ticks as the core simulation loop. Each tick represents one
season (0.25 years). Weather is generated per-tick from deterministic cycles,
and lightweight erosion is applied continuously as a side effect of weather.

Tick ordering per season:
1. Generate weather (from WeatherSystem)
2. Tick ecology (receives weather)
3. Seasonal erosion (driven by storm_intensity)
4. Thermal diffusion
5. Recompute flow accumulation (every FLOW_RECOMPUTE_INTERVAL ticks)
6. Advance clock (season cycles 0-3, year increments when season wraps)
"""

from __future__ import annotations

import numpy as np

from bike_sim.tiers.climate_hydrology import ClimateHydrologyTier
from bike_sim.tiers.ecology import EcologyTier
from bike_sim.tiers.erosion import (
    ErosionParams,
    SeasonalErosionParams,
    erode_seasonal,
    thermal_diffusion,
)
from bike_sim.tiers.geology import GeologyTier
from bike_sim.weather import WeatherSystem
from bike_sim.world import World


class Orchestrator:
    """Schedule ticks across all three simulation tiers using seasonal loop."""

    SEASONS_PER_MINUTE: float = 1.0  # 1 season per minute of riding
    MAX_SEASONS_PER_RIDE: int = 120  # cap at 30 simulated years
    FLOW_RECOMPUTE_INTERVAL: int = 40  # recompute flow every 40 ticks (~10 years)

    # Legacy constants kept for backward compatibility
    RIDE_YEARS_PER_MINUTE: float = 0.25  # 1 season = 0.25 years per minute
    RIDE_MAX_YEARS: float = 30.0  # 120 seasons * 0.25

    def __init__(
        self,
        world: World,
        erosion_params: ErosionParams | None = None,
        seasonal_erosion_params: SeasonalErosionParams | None = None,
        seasonal_history_ticks: int = 0,
    ) -> None:
        self._world = world
        self._erosion_params = erosion_params
        self._seasonal_erosion_params = seasonal_erosion_params or SeasonalErosionParams()
        self._seasonal_history_ticks = seasonal_history_ticks

    def create_world(self) -> None:
        """Two-phase world creation: deep history (coarse) + seasonal recent history.

        Phase A: Geology + climate-hydrology (batched erosion, as before)
        Phase B: Run seasonal_history_ticks of seasonal ecology+erosion
        """
        next_version = self._world.current_version + 1
        self._world.rasters.set_version(next_version)

        # Phase A: Deep history
        GeologyTier(self._world).tick()
        ClimateHydrologyTier(self._world, erosion_params=self._erosion_params).tick()

        # Phase B: Seasonal recent history (configurable, default 0 for backward compat)
        if self._seasonal_history_ticks > 0:
            self._advance_seasonal(self._seasonal_history_ticks)

        self._world.commit_version(trigger="create_world")

    def advance(self, years: float) -> dict:
        """Advance by years (converted to seasonal ticks internally).

        Backward-compatible: still accepts years, returns dict with
        years_advanced and ecology_ticks keys.
        """
        num_seasons = int(years / 0.25)
        return self.advance_seasons(num_seasons)

    def advance_seasons(self, num_seasons: int) -> dict:
        """Advance by a specific number of seasonal ticks."""
        next_version = self._world.current_version + 1
        self._world.rasters.set_version(next_version)

        self._advance_seasonal(num_seasons)

        self._world.commit_version(trigger=f"advance {num_seasons} seasons")
        return {
            "years_advanced": num_seasons * 0.25,
            "ecology_ticks": num_seasons,
            "seasons_advanced": num_seasons,
        }

    def advance_ride(self, ride_duration_minutes: float) -> dict:
        """Convert ride duration to seasonal ticks and advance.

        Uses SEASONS_PER_MINUTE (1.0) to convert, capped at
        MAX_SEASONS_PER_RIDE (120) so a long ride doesn't skip too much
        world history.
        """
        seasons = min(
            int(ride_duration_minutes * self.SEASONS_PER_MINUTE),
            self.MAX_SEASONS_PER_RIDE,
        )
        return self.advance_seasons(max(1, seasons))

    def _advance_seasonal(self, num_seasons: int) -> None:
        """Internal: run the seasonal loop without version management."""
        eco = EcologyTier(self._world)
        heightmap = self._world.rasters.read_layer("geology", "heightmap")
        weather_sys = WeatherSystem(self._world.seed, heightmap)

        # Load mutable terrain state
        store = self._world.rasters
        ch_layers = store.list_layers("climate_hydrology")

        if "eroded_heightmap" in ch_layers:
            eroded_hm = store.read_layer("climate_hydrology", "eroded_heightmap").copy()
        else:
            eroded_hm = heightmap.copy()

        if "sediment_depth" in ch_layers:
            sediment = store.read_layer("climate_hydrology", "sediment_depth").copy()
        else:
            sediment = np.zeros_like(heightmap)

        if "flow_accumulation" in ch_layers:
            flow_acc = store.read_layer("climate_hydrology", "flow_accumulation").copy()
        else:
            flow_acc = np.ones_like(heightmap)

        bedrock_type = self._world.rasters.read_layer("geology", "bedrock_type")

        eco_clock = self._world.tier_clocks["ecology"]
        ticks_since_flow_recompute = 0

        for i in range(num_seasons):
            year = eco_clock.simulated_year
            season = eco_clock.tick_number % 4

            # 1. Generate weather
            weather = weather_sys.generate(year, season)

            # 2. Tick ecology
            eco.tick(weather)

            # 3. Seasonal erosion
            erode_seasonal(
                eroded_hm,
                sediment,
                flow_acc,
                weather.precipitation,
                weather.storm_intensity,
                bedrock_type,
                self._seasonal_erosion_params,
            )

            # 4. Thermal diffusion
            thermal_diffusion(eroded_hm, sediment)

            # 5. Recompute flow if needed
            ticks_since_flow_recompute += 1
            if ticks_since_flow_recompute >= self.FLOW_RECOMPUTE_INTERVAL:
                combined = eroded_hm + sediment
                flow_acc = self._compute_flow_accumulation(combined)
                ticks_since_flow_recompute = 0

        # Write final terrain state
        tick_num = eco_clock.tick_number  # already advanced by eco.tick()
        store.write_layer(
            "climate_hydrology", "eroded_heightmap", eroded_hm.astype(np.float64), tick_num
        )
        store.write_layer(
            "climate_hydrology", "sediment_depth", sediment.astype(np.float64), tick_num
        )
        store.write_layer(
            "climate_hydrology", "flow_accumulation", flow_acc.astype(np.float64), tick_num
        )

    def _compute_flow_accumulation(self, heightmap: np.ndarray) -> np.ndarray:
        """D8 flow accumulation — reuses ClimateHydrologyTier's implementation."""
        ch = ClimateHydrologyTier(self._world, erosion_params=self._erosion_params)
        return ch._compute_flow_accumulation(heightmap)

    def introduce_fire(self, x: float, y: float) -> None:
        """Manually trigger a fire event at the given world coordinates."""
        current_year = self._world.tier_clocks["ecology"].simulated_year
        self._world.events.add_event(
            "fire",
            x,
            y,
            current_year,
            radius=500.0,
            data={"source": "manual"},
        )

    def status(self) -> dict:
        """Return current world status summary.

        Includes seed, tier clock positions, species count, and individual
        count (searched across the full world extent).
        """
        species = self._world.events.list_species()
        individuals = self._world.events.find_individuals_near(25000, 25000, 50000)
        return {
            "seed": self._world.seed,
            "simulated_year": self._world.tier_clocks["ecology"].simulated_year,
            "geology_tick": self._world.tier_clocks["geology"].tick_number,
            "climate_hydrology_tick": self._world.tier_clocks["climate_hydrology"].tick_number,
            "ecology_tick": self._world.tier_clocks["ecology"].tick_number,
            "species_count": len(species),
            "individual_count": len(individuals),
            "current_version": self._world.current_version,
            "total_versions": len(self._world.list_versions()),
        }
