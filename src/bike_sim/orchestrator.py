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

from bike_sim.tiers.climate_hydrology import ClimateHydrologyTier, _distance_transform
from bike_sim.tiers.ecology import EcologyTier
from bike_sim.tiers.ground_cover import compute_ground_cover
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
        summary_interval: int = 4,
        snapshot_interval: int = 400,
    ) -> None:
        self._world = world
        self._erosion_params = erosion_params
        self._seasonal_erosion_params = seasonal_erosion_params or SeasonalErosionParams()
        self._seasonal_history_ticks = seasonal_history_ticks
        self._summary_interval = summary_interval
        self._snapshot_interval = snapshot_interval

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

        # Only commit a version if _advance_seasonal didn't already snapshot
        # at the final tick (avoids duplicate snapshots).
        current_tick = self._world.tier_clocks["ecology"].tick_number
        if self._snapshot_interval <= 0 or current_tick % self._snapshot_interval != 0:
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

        # Load spatial climate bias fields if available
        ch_layers = self._world.rasters.list_layers("climate_hydrology")
        moisture_bias = (
            self._world.rasters.read_layer("climate_hydrology", "moisture_bias")
            if "moisture_bias" in ch_layers else None
        )
        continentality = (
            self._world.rasters.read_layer("climate_hydrology", "continentality")
            if "continentality" in ch_layers else None
        )
        weather_sys = WeatherSystem(
            self._world.seed, heightmap,
            moisture_bias=moisture_bias,
            continentality=continentality,
        )

        # Load mutable terrain state
        store = self._world.rasters

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

            # 2b. Compute ground cover from current conditions
            cover_type, cover_vigor = compute_ground_cover(weather)
            store.write_layer("ecology", "ground_cover_type", cover_type, eco_clock.tick_number)
            store.write_layer("ecology", "ground_cover_vigor", cover_vigor, eco_clock.tick_number)

            # 3. Tick summary logging
            current_tick = eco_clock.tick_number  # already advanced by eco.tick()
            if self._summary_interval > 0 and current_tick % self._summary_interval == 0:
                self._log_tick_summary(current_tick, year, season, weather)

            # 4. Intermediate snapshot
            if self._snapshot_interval > 0 and current_tick % self._snapshot_interval == 0 and current_tick > 0:
                # Write terrain state before committing snapshot
                store.write_layer(
                    "climate_hydrology", "eroded_heightmap", eroded_hm.astype(np.float64), current_tick
                )
                store.write_layer(
                    "climate_hydrology", "sediment_depth", sediment.astype(np.float64), current_tick
                )
                store.write_layer(
                    "climate_hydrology", "flow_accumulation", flow_acc.astype(np.float64), current_tick
                )
                # Update derived climate layers from live weather
                combined = eroded_hm + sediment
                self._write_derived_climate(
                    combined, sediment, weather, flow_acc, current_tick
                )
                self._world.commit_version(trigger=f"snapshot at tick {current_tick}")
                next_ver = self._world.current_version + 1
                self._world.rasters.set_version(next_ver)

            # 5. Seasonal erosion
            erode_seasonal(
                eroded_hm,
                sediment,
                flow_acc,
                weather.precipitation,
                weather.storm_intensity,
                bedrock_type,
                self._seasonal_erosion_params,
            )

            # 6. Thermal diffusion
            thermal_diffusion(eroded_hm, sediment)

            # 7. Recompute flow if needed
            ticks_since_flow_recompute += 1
            if ticks_since_flow_recompute >= self.FLOW_RECOMPUTE_INTERVAL:
                combined = eroded_hm + sediment
                flow_acc = self._compute_flow_accumulation(combined)
                ticks_since_flow_recompute = 0

        # Write final terrain state (only if we actually ran ticks)
        if num_seasons > 0:
            tick_num = eco_clock.tick_number
            store.write_layer(
                "climate_hydrology", "eroded_heightmap", eroded_hm.astype(np.float64), tick_num
            )
            store.write_layer(
                "climate_hydrology", "sediment_depth", sediment.astype(np.float64), tick_num
            )
            store.write_layer(
                "climate_hydrology", "flow_accumulation", flow_acc.astype(np.float64), tick_num
            )
            # Update derived climate layers from final weather state
            combined = eroded_hm + sediment
            self._write_derived_climate(combined, sediment, weather, flow_acc, tick_num)

    def _log_tick_summary(
        self, tick: int, year: float, season: int, weather: object
    ) -> None:
        """Collect per-species and weather summaries and write to the EventStore."""
        store = self._world.rasters
        events = self._world.events
        species_list = events.list_species()
        ecology_layers = store.list_layers("ecology")

        # Per-species summaries
        summaries: list[dict] = []
        for sp in species_list:
            sid = sp["species_id"]
            density_layer = f"species_{sid}_density"
            if density_layer not in ecology_layers:
                continue
            density = store.read_layer("ecology", density_layer)
            total_density = float(density.sum())
            occupied_cells = int((density > 0).sum())

            summaries.append({
                "species_id": sid,
                "total_density": total_density,
                "occupied_cells": occupied_cells,
            })

        if summaries:
            events.write_tick_summary(tick, year, season, summaries)

        # Weather summary
        mean_temp = float(weather.temperature.mean())
        mean_precip = float(weather.precipitation.mean())

        events.write_tick_weather(tick, year, season, mean_temp, mean_precip, 0.0)

    def _write_derived_climate(
        self,
        combined_surface: np.ndarray,
        sediment: np.ndarray,
        weather: object,
        flow_acc: np.ndarray,
        tick_number: int,
    ) -> None:
        """Recompute and write climate-derived layers from live weather.

        Updates temperature, precipitation, soil_moisture, frost_days, GDD,
        solar_insolation, and distance_to_water so that snapshots reflect
        the current climate state rather than creation-time values.
        """
        store = self._world.rasters

        # Write the live weather fields directly
        store.write_layer(
            "climate_hydrology", "temperature",
            weather.temperature.astype(np.float64), tick_number
        )
        store.write_layer(
            "climate_hydrology", "precipitation",
            weather.precipitation.astype(np.float64), tick_number
        )

        # Soil moisture — same formula as ClimateHydrologyTier._compute_derived_cache
        base_moisture = np.clip(weather.precipitation / 2000.0, 0, 1)
        elev_norm = combined_surface / (combined_surface.max() + 1e-10)
        base_moisture += (1 - elev_norm) * 0.2
        sediment_bonus = np.minimum(sediment / 5.0, 0.15)
        base_moisture += sediment_bonus
        base_moisture = np.clip(base_moisture, 0, 1)

        summer = np.clip(base_moisture * 0.7, 0, 1)
        winter = np.clip(base_moisture * 1.0, 0, 1)
        winter = np.maximum(winter, summer)
        store.write_layer(
            "climate_hydrology", "soil_moisture_summer",
            summer.astype(np.float64), tick_number
        )
        store.write_layer(
            "climate_hydrology", "soil_moisture_winter",
            winter.astype(np.float64), tick_number
        )

        # Frost days
        frost_days = np.clip(365.0 * (1.0 - weather.temperature / 20.0), 0, 365)
        store.write_layer(
            "climate_hydrology", "frost_days",
            frost_days.astype(np.float64), tick_number
        )

        # Growing degree days
        gdd = np.clip((weather.temperature - 5.0) * 365.0 * 0.5, 0, None)
        store.write_layer(
            "climate_hydrology", "growing_degree_days",
            gdd.astype(np.float64), tick_number
        )

        # Solar insolation (terrain-dependent, not weather-dependent, but
        # terrain changes via erosion so recompute)
        cell_size = 50.0
        dy, dx = np.gradient(combined_surface, cell_size)
        slope_angle = np.arctan(np.sqrt(dx**2 + dy**2))
        aspect = np.arctan2(-dx, dy)
        insolation = np.cos(slope_angle) * (0.5 + 0.5 * np.cos(aspect - np.pi))
        insolation = np.clip(insolation, 0, 1)
        store.write_layer(
            "climate_hydrology", "solar_insolation",
            insolation.astype(np.float64), tick_number
        )

        # Distance to water
        water_threshold = np.percentile(flow_acc, 99.5)
        is_water = flow_acc >= water_threshold
        distance_to_water = _distance_transform(is_water, cell_size)
        store.write_layer(
            "climate_hydrology", "distance_to_water",
            distance_to_water.astype(np.float64), tick_number
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
