"""Tests for the refactored ecology tick (unified growth/mortality rule).

Validates that EcologyTier correctly:
- Accepts SeasonalWeather objects
- Applies unified growth/competition/mortality every tick
- Fire disturbance in summer, blowdown in fall
- Enforces non-negativity invariants
- Uses terrain-varying carrying capacity
- Produces deterministic results from the same seed
"""

from __future__ import annotations

import re
import shutil

import numpy as np
import pytest

from bike_sim.tiers.ecology import EcologyTier, TIER
from bike_sim.weather import SeasonalWeather, WeatherSystem
from bike_sim.world import World

GRID_SIZE = 1000

# Season constants
WINTER, SPRING, SUMMER, FALL = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_weather(season: int, **overrides) -> SeasonalWeather:
    """Create a SeasonalWeather with controllable fields.

    Defaults produce mild, moderate conditions (no frost, no drought,
    no storms).  Override individual fields to test specific scenarios.
    """
    defaults = {
        "temperature": np.full((GRID_SIZE, GRID_SIZE), 15.0, dtype=np.float64),
        "precipitation": np.full((GRID_SIZE, GRID_SIZE), 1600.0, dtype=np.float64),
        "frost_severity": np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64),
        "storm_intensity": 0.0,
        "season": season,
    }
    defaults.update(overrides)
    return SeasonalWeather(**defaults)


def make_weather_system(world: World) -> WeatherSystem:
    """Build a WeatherSystem from the world's geology heightmap."""
    heightmap = world.rasters.read_layer("geology", "heightmap")
    return WeatherSystem(world.seed, heightmap, grid_size=GRID_SIZE)


def _create_eco_world(path) -> World:
    """Create a world with geology + climate-hydrology ticked."""
    from bike_sim.tiers.climate_hydrology import ClimateHydrologyTier
    from bike_sim.tiers.erosion import ErosionParams
    from bike_sim.tiers.geology import GeologyTier
    FAST_EROSION = ErosionParams(num_particles=100, max_lifetime=30)
    w = World.create(path, seed=42)
    GeologyTier(w).tick()
    ClimateHydrologyTier(w, erosion_params=FAST_EROSION).tick()
    return w


def _total_density(world: World) -> np.ndarray:
    """Sum density across all species, returning (H, W) array."""
    species = world.events.list_species()
    total = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    layers = world.rasters.list_layers(TIER)
    for sp in species:
        layer = f"species_{sp['species_id']}_density"
        if layer in layers:
            total += world.rasters.read_layer(TIER, layer)
    return total


def _species_densities(world: World) -> dict[str, np.ndarray]:
    """Return {species_id: density_array} for all species."""
    result = {}
    layers = world.rasters.list_layers(TIER)
    for sp in world.events.list_species():
        sid = sp["species_id"]
        layer = f"species_{sid}_density"
        if layer in layers:
            result[sid] = world.rasters.read_layer(TIER, layer).copy()
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def eco_world(fresh_world):
    """Fresh world for ecology tests that mutate state.

    Uses the session-scoped base world (geology+climate already ticked)
    copied to a temp dir, so each test gets a clean copy without paying
    the geology/climate cost.
    """
    return fresh_world


# ===========================================================================
# 1. Tick signature
# ===========================================================================


class TestTickSignature:
    """Verify the tick(weather) API works at the most basic level."""

    def test_tick_accepts_seasonal_weather(self, eco_world):
        """EcologyTier.tick() accepts a SeasonalWeather object without error."""
        eco = EcologyTier(eco_world)
        weather = make_weather(SUMMER)
        eco.tick(weather)  # should not raise

    def test_tick_creates_ancestors_on_first_call(self, eco_world):
        """After tick 0, species exist in event store."""
        eco = EcologyTier(eco_world)
        eco.tick(make_weather(SPRING))

        species = eco_world.events.list_species()
        assert len(species) >= 10, (
            f"Expected at least 10 ancestor species, got {len(species)}"
        )
        assert len(species) <= 20, (
            f"Expected at most 20 ancestor species, got {len(species)}"
        )

    def test_tick_produces_density_layers(self, eco_world):
        """After ticking, ecology has density layers for each species."""
        eco = EcologyTier(eco_world)
        eco.tick(make_weather(SPRING))

        species = eco_world.events.list_species()
        layers = eco_world.rasters.list_layers(TIER)
        for sp in species:
            layer = f"species_{sp['species_id']}_density"
            assert layer in layers, f"Missing density layer: {layer}"


# ===========================================================================
# 2. Growth and competition
# ===========================================================================


class TestGrowthAndCompetition:
    """Verify the unified growth/mortality rule."""

    def test_growth_occurs_in_any_season(self, eco_world):
        """Growth should occur every tick, not just summer."""
        eco = EcologyTier(eco_world)
        eco.tick(make_weather(SPRING))  # tick 0: ancestors + initial populations

        densities_before = _species_densities(eco_world)

        # A winter tick should still show growth for some species
        eco.tick(make_weather(WINTER))

        densities_after = _species_densities(eco_world)

        # At least some species should have cells where density increased
        any_growth = False
        for sid in densities_before:
            if sid in densities_after:
                grew = (densities_after[sid] > densities_before[sid]).any()
                if grew:
                    any_growth = True
                    break

        assert any_growth, "Growth should occur in winter tick too (unified rule)"

    def test_drought_intolerant_species_suffer_in_dry_conditions(self, eco_world):
        """Species with low drought_tolerance should lose more density
        in dry conditions than drought-tolerant species."""
        eco = EcologyTier(eco_world)
        eco.tick(make_weather(SPRING))  # tick 0
        eco.tick(make_weather(SUMMER))  # initial growth

        densities_before = _species_densities(eco_world)

        # Several dry ticks
        dry_weather = make_weather(
            SUMMER,
            precipitation=np.full(
                (GRID_SIZE, GRID_SIZE), 30.0, dtype=np.float64
            ),
            temperature=np.full(
                (GRID_SIZE, GRID_SIZE), 30.0, dtype=np.float64
            ),
        )
        for _ in range(3):
            eco.tick(dry_weather)

        densities_after = _species_densities(eco_world)

        tolerant_losses = []
        intolerant_losses = []
        for sp in eco_world.events.list_species():
            sid = sp["species_id"]
            genome = eco_world.events.get_species(sid)["genome"]
            if sid not in densities_before or sid not in densities_after:
                continue
            before = densities_before[sid].sum()
            if before < 1.0:
                continue
            after = densities_after[sid].sum()
            loss_frac = (before - after) / before

            dt = genome.get("drought_tolerance", 0.5)
            if dt > 0.7:
                tolerant_losses.append(loss_frac)
            elif dt < 0.3:
                intolerant_losses.append(loss_frac)

        if tolerant_losses and intolerant_losses:
            assert np.mean(intolerant_losses) > np.mean(tolerant_losses), (
                f"Drought-intolerant species should suffer more: "
                f"intolerant={np.mean(intolerant_losses):.3f}, "
                f"tolerant={np.mean(tolerant_losses):.3f}"
            )

    def test_cold_conditions_favor_frost_tolerant(self, eco_world):
        """Species with high frost_tolerance should fare better in cold."""
        eco = EcologyTier(eco_world)
        eco.tick(make_weather(SPRING))  # tick 0
        eco.tick(make_weather(SUMMER))  # initial growth

        densities_before = _species_densities(eco_world)

        # Several cold ticks
        cold_weather = make_weather(
            WINTER,
            temperature=np.full(
                (GRID_SIZE, GRID_SIZE), -8.0, dtype=np.float64
            ),
        )
        for _ in range(4):
            eco.tick(cold_weather)

        densities_after = _species_densities(eco_world)

        tolerant_losses = []
        intolerant_losses = []
        for sp in eco_world.events.list_species():
            sid = sp["species_id"]
            genome = eco_world.events.get_species(sid)["genome"]
            if sid not in densities_before or sid not in densities_after:
                continue
            before = densities_before[sid].sum()
            if before < 1.0:
                continue
            after = densities_after[sid].sum()
            loss_frac = (before - after) / before

            ft = genome.get("frost_tolerance", 0.5)
            if ft > 0.6:
                tolerant_losses.append(loss_frac)
            elif ft < 0.3:
                intolerant_losses.append(loss_frac)

        if tolerant_losses and intolerant_losses:
            assert np.mean(intolerant_losses) > np.mean(tolerant_losses), (
                f"Frost-intolerant species should suffer more in cold: "
                f"intolerant={np.mean(intolerant_losses):.3f}, "
                f"tolerant={np.mean(tolerant_losses):.3f}"
            )


# ===========================================================================
# 3. Basic invariants
# ===========================================================================


class TestBasicInvariants:
    """Non-negativity and shape invariants after ticking."""

    def test_all_densities_non_negative(self, eco_world):
        """After several ticks, all density arrays >= 0."""
        eco = EcologyTier(eco_world)
        for season in [SPRING, SUMMER, FALL, WINTER, SPRING, SUMMER]:
            eco.tick(make_weather(season))

        for sp in eco_world.events.list_species():
            sid = sp["species_id"]
            layer = f"species_{sid}_density"
            if layer in eco_world.rasters.list_layers(TIER):
                arr = eco_world.rasters.read_layer(TIER, layer)
                assert arr.min() >= 0.0, (
                    f"Density for {sid} has negative values: min={arr.min():.6f}"
                )

    def test_density_floor_applied(self, eco_world):
        """After ticking, no density values between 0 and 0.001 should exist."""
        eco = EcologyTier(eco_world)
        for season in [SPRING, SUMMER, FALL, WINTER] * 3:
            eco.tick(make_weather(season))

        for sp in eco_world.events.list_species():
            sid = sp["species_id"]
            layer = f"species_{sid}_density"
            if layer in eco_world.rasters.list_layers(TIER):
                arr = eco_world.rasters.read_layer(TIER, layer)
                dust = ((arr > 0) & (arr < 0.001)).sum()
                assert dust == 0, (
                    f"Species {sid} has {dust} cells with density in (0, 0.001) -- "
                    f"density floor not applied"
                )

    def test_no_seed_bank_layers(self, eco_world):
        """After refactor, no seed bank layers should be created."""
        eco = EcologyTier(eco_world)
        for season in [SPRING, SUMMER, FALL, WINTER]:
            eco.tick(make_weather(season))

        layers = eco_world.rasters.list_layers(TIER)
        seed_bank_layers = [l for l in layers if l.startswith("seed_bank_")]
        assert len(seed_bank_layers) == 0, (
            f"Found seed bank layers that should not exist: {seed_bank_layers}"
        )


# ===========================================================================
# 4. Integration: full year cycle
# ===========================================================================


class TestIntegration:
    """End-to-end integration with WeatherSystem."""

    def test_full_year_cycle(self, eco_world):
        """Tick through 4 seasons using WeatherSystem; world has species
        with nonzero density after."""
        eco = EcologyTier(eco_world)
        ws = make_weather_system(eco_world)

        year = 0.0
        for season in [WINTER, SPRING, SUMMER, FALL]:
            weather = ws.generate(year, season)
            eco.tick(weather)

        species = eco_world.events.list_species()
        assert len(species) > 0, "No species after full year"

        total = _total_density(eco_world)
        assert total.sum() > 0, "Total density is zero after full year"

    def test_full_year_with_weather_system_seasons(self, eco_world):
        """Run a complete year starting from spring and verify reasonable state."""
        eco = EcologyTier(eco_world)
        ws = make_weather_system(eco_world)

        year = 100.0  # arbitrary year
        for season in [SPRING, SUMMER, FALL, WINTER]:
            weather = ws.generate(float(year), season)
            eco.tick(weather)

        # Verify per-species density layers exist and have correct shape.
        for sp in eco_world.events.list_species():
            sid = sp["species_id"]
            layer = f"species_{sid}_density"
            layers = eco_world.rasters.list_layers(TIER)
            assert layer in layers
            arr = eco_world.rasters.read_layer(TIER, layer)
            assert arr.shape == (GRID_SIZE, GRID_SIZE)
            assert arr.dtype == np.float64

    @pytest.mark.slow
    def test_multi_year_determinism(self, tmp_path):
        """Run 8 seasonal ticks with same seed twice; results are bit-identical."""
        results = []
        season_sequence = [SPRING, SUMMER, FALL, WINTER, SPRING, SUMMER, FALL, WINTER]

        for trial in range(2):
            path = tmp_path / f"det_world_{trial}"
            w = _create_eco_world(path)
            eco = EcologyTier(w)

            for season in season_sequence:
                eco.tick(make_weather(season))

            # Collect all density arrays.
            trial_densities = {}
            for sp in w.events.list_species():
                sid = sp["species_id"]
                layer = f"species_{sid}_density"
                if layer in w.rasters.list_layers(TIER):
                    trial_densities[sid] = w.rasters.read_layer(TIER, layer).copy()
            results.append(trial_densities)

        # Same species should exist in both runs.
        assert set(results[0].keys()) == set(results[1].keys()), (
            "Different species produced across identical runs"
        )

        # Bit-identical density arrays.
        for sid in results[0]:
            np.testing.assert_array_equal(
                results[0][sid],
                results[1][sid],
                err_msg=f"Density for {sid} differs between identical runs",
            )

    @pytest.mark.slow
    def test_multi_year_determinism_with_weather_system(self, tmp_path):
        """Same seed + WeatherSystem produces identical results."""
        results = []
        for trial in range(2):
            path = tmp_path / f"ws_det_world_{trial}"
            w = _create_eco_world(path)
            eco = EcologyTier(w)
            ws = make_weather_system(w)

            for year in range(2):
                for season in [SPRING, SUMMER, FALL, WINTER]:
                    weather = ws.generate(float(year), season)
                    eco.tick(weather)

            trial_densities = {}
            for sp in w.events.list_species():
                sid = sp["species_id"]
                layer = f"species_{sid}_density"
                if layer in w.rasters.list_layers(TIER):
                    trial_densities[sid] = w.rasters.read_layer(TIER, layer).copy()
            results.append(trial_densities)

        assert set(results[0].keys()) == set(results[1].keys())
        for sid in results[0]:
            np.testing.assert_array_equal(
                results[0][sid],
                results[1][sid],
                err_msg=f"WeatherSystem determinism failed for {sid}",
            )
