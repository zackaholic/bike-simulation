"""Tests for Step 3 (Seasonal Ecology Model) of the seasonal redesign.

Validates that the rewritten EcologyTier correctly:
- Accepts SeasonalWeather objects and branches by season
- Applies winter frost mortality, spring leaf-out/frost damage,
  summer growth/drought, and fall seed production/dispersal
- Tracks cumulative drought stress with multi-year memory
- Enforces carrying capacity and non-negativity invariants
- Uses max_height for canopy shading competition
- Produces deterministic results from the same seed
"""

from __future__ import annotations

import re
import shutil

import numpy as np
import pytest

from bike_sim.tiers.climate_hydrology import ClimateHydrologyTier
from bike_sim.tiers.ecology import EcologyTier, TIER
from bike_sim.tiers.erosion import ErosionParams
from bike_sim.tiers.geology import GeologyTier
from bike_sim.weather import SeasonalWeather, WeatherSystem
from bike_sim.world import World

# Fast erosion for test speed.
FAST_EROSION = ErosionParams(num_particles=100, max_lifetime=30)

GRID_SIZE = 1000
CARRYING_CAPACITY = 15.0

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


def _total_seed_bank(world: World) -> np.ndarray:
    """Sum seed banks across all species, returning (H, W) array."""
    species = world.events.list_species()
    total = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    layers = world.rasters.list_layers(TIER)
    for sp in species:
        layer = f"seed_bank_{sp['species_id']}"
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


def _species_seed_banks(world: World) -> dict[str, np.ndarray]:
    """Return {species_id: seed_bank_array} for all species."""
    result = {}
    layers = world.rasters.list_layers(TIER)
    for sp in world.events.list_species():
        sid = sp["species_id"]
        layer = f"seed_bank_{sid}"
        if layer in layers:
            result[sid] = world.rasters.read_layer(TIER, layer).copy()
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_world(tmp_path_factory):
    """World with geology + climate ticked (terrain ready for ecology).

    Module-scoped because geology + climate are expensive and read-only
    for ecology tests.
    """
    path = tmp_path_factory.mktemp("seasonal_eco")
    return _create_eco_world(path / "world")


@pytest.fixture
def eco_world(tmp_path):
    """Fresh world for ecology tests that mutate state."""
    return _create_eco_world(tmp_path / "eco_world")


# ===========================================================================
# 7. Tick signature
# ===========================================================================


class TestTickSignature:
    """Verify the new tick(weather) API works at the most basic level."""

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
        assert len(species) >= 5, (
            f"Expected at least 5 ancestor species, got {len(species)}"
        )
        assert len(species) <= 8, (
            f"Expected at most 8 ancestor species, got {len(species)}"
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
# 1. Seasonal correctness
# ===========================================================================


class TestSeasonalCorrectness:
    """Verify season-specific ecology behaviour (spec section 11)."""

    def test_winter_frost_kills_frost_intolerant(self, eco_world):
        """Species with low frost_tolerance should lose density in a harsh winter."""
        eco = EcologyTier(eco_world)
        # Tick 0: create ancestors + initial populations.
        eco.tick(make_weather(SPRING))
        # Let populations grow before testing winter kill.
        eco.tick(make_weather(SUMMER))

        density_before = _total_density(eco_world)
        total_before = density_before.sum()

        # Harsh winter: high frost severity everywhere.
        harsh_winter = make_weather(
            WINTER,
            frost_severity=np.full(
                (GRID_SIZE, GRID_SIZE), 0.9, dtype=np.float64
            ),
            temperature=np.full(
                (GRID_SIZE, GRID_SIZE), -15.0, dtype=np.float64
            ),
        )
        eco.tick(harsh_winter)

        density_after = _total_density(eco_world)
        total_after = density_after.sum()

        # Total density should decrease after a harsh winter.
        assert total_after < total_before, (
            f"Harsh winter should reduce total density: {total_before:.1f} -> {total_after:.1f}"
        )

    def test_spring_frost_damages_aggressive_species(self, eco_world):
        """Species with high phenological_aggressiveness and low frost_tolerance
        should lose more density than conservative species in a frosty spring."""
        eco = EcologyTier(eco_world)
        # Tick 0: ancestors.
        eco.tick(make_weather(SPRING))

        # Record per-species density.
        densities_before = _species_densities(eco_world)

        # Spring with moderate frost.
        frosty_spring = make_weather(
            SPRING,
            frost_severity=np.full(
                (GRID_SIZE, GRID_SIZE), 0.6, dtype=np.float64
            ),
        )
        eco.tick(frosty_spring)

        densities_after = _species_densities(eco_world)

        # Find aggressive vs conservative species.
        aggressive_losses = []
        conservative_losses = []
        for sp in eco_world.events.list_species():
            sid = sp["species_id"]
            genome = eco_world.events.get_species(sid)["genome"]
            if sid not in densities_before or sid not in densities_after:
                continue
            before_sum = densities_before[sid].sum()
            if before_sum < 1.0:
                continue  # skip species with negligible density
            after_sum = densities_after[sid].sum()
            loss_frac = (before_sum - after_sum) / before_sum

            aggr = genome.get("phenological_aggressiveness", 0.5)
            frost_tol = genome.get("frost_tolerance", 0.5)
            if aggr > 0.6 and frost_tol < 0.4:
                aggressive_losses.append(loss_frac)
            elif aggr < 0.3:
                conservative_losses.append(loss_frac)

        # Aggressive species should on average lose more.
        if aggressive_losses and conservative_losses:
            mean_aggr = np.mean(aggressive_losses)
            mean_cons = np.mean(conservative_losses)
            assert mean_aggr > mean_cons, (
                f"Aggressive species should lose more density in frosty spring: "
                f"aggressive={mean_aggr:.3f}, conservative={mean_cons:.3f}"
            )

    def test_summer_has_growth_activity(self, eco_world):
        """Summer growth method should increase density in at least some cells."""
        eco = EcologyTier(eco_world)
        eco.tick(make_weather(SPRING))  # tick 0: ancestors + initial populations

        # Get density snapshot before summer
        species = eco_world.events.list_species()
        densities_before = {}
        for sp in species:
            sid = sp["species_id"]
            layer = f"species_{sid}_density"
            if layer in eco_world.rasters.list_layers("ecology"):
                densities_before[sid] = eco_world.rasters.read_layer("ecology", layer).copy()

        # Favorable summer (use default precipitation for viable suitability)
        summer_weather = make_weather(
            SUMMER,
            temperature=np.full((GRID_SIZE, GRID_SIZE), 18.0, dtype=np.float64),
        )
        eco.tick(summer_weather)

        # At least some species should have cells where density increased
        any_growth = False
        for sp in species:
            sid = sp["species_id"]
            layer = f"species_{sid}_density"
            if layer in eco_world.rasters.list_layers("ecology") and sid in densities_before:
                after = eco_world.rasters.read_layer("ecology", layer)
                grew = (after > densities_before[sid]).any()
                if grew:
                    any_growth = True
                    break

        assert any_growth, "Summer should produce growth in at least some cells for some species"

    def test_no_seed_production_outside_fall(self, eco_world):
        """Seed banks should not increase after a summer tick (seed production is fall-only)."""
        eco = EcologyTier(eco_world)
        eco.tick(make_weather(SPRING))  # tick 0: ancestors

        seed_bank_after_init = _total_seed_bank(eco_world).sum()

        # Tick through summer — seed banks should NOT increase.
        eco.tick(make_weather(SUMMER))
        seed_bank_after_summer = _total_seed_bank(eco_world).sum()

        assert seed_bank_after_summer <= seed_bank_after_init + 1e-6, (
            f"Seed bank should not grow outside fall: "
            f"{seed_bank_after_init:.1f} -> {seed_bank_after_summer:.1f}"
        )

    def test_fall_produces_seeds(self, eco_world):
        """Seed banks should increase after a fall tick."""
        eco = EcologyTier(eco_world)
        # Run a full year to build up populations, then measure fall.
        eco.tick(make_weather(SPRING))   # tick 0: ancestors
        eco.tick(make_weather(SUMMER))   # growth
        eco.tick(make_weather(SUMMER))   # more growth

        seed_bank_before_fall = _total_seed_bank(eco_world).sum()

        eco.tick(make_weather(FALL))
        seed_bank_after_fall = _total_seed_bank(eco_world).sum()

        assert seed_bank_after_fall > seed_bank_before_fall, (
            f"Fall should produce seeds: "
            f"{seed_bank_before_fall:.1f} -> {seed_bank_after_fall:.1f}"
        )


# ===========================================================================
# 2. Drought stress
# ===========================================================================


class TestDroughtStress:
    """Verify cumulative drought stress mechanics."""

    def test_drought_stress_accumulates(self, eco_world):
        """Multiple dry summers should increase drought_stress."""
        eco = EcologyTier(eco_world)
        eco.tick(make_weather(SPRING))  # tick 0: ancestors

        dry_summer = make_weather(
            SUMMER,
            precipitation=np.full(
                (GRID_SIZE, GRID_SIZE), 50.0, dtype=np.float64
            ),
        )

        # Tick several dry summers.
        for _ in range(3):
            eco.tick(dry_summer)

        # Read drought stress layer.
        layers = eco_world.rasters.list_layers(TIER)
        assert "drought_stress" in layers, (
            "Expected drought_stress layer in ecology rasters"
        )
        stress = eco_world.rasters.read_layer(TIER, "drought_stress")
        assert stress.max() > 0, "Drought stress should be positive after dry summers"

    def test_drought_stress_increases_with_consecutive_droughts(self, eco_world):
        """Drought stress after 3 dry summers should exceed stress after 1."""
        eco = EcologyTier(eco_world)
        eco.tick(make_weather(SPRING))  # tick 0

        dry_summer = make_weather(
            SUMMER,
            precipitation=np.full(
                (GRID_SIZE, GRID_SIZE), 50.0, dtype=np.float64
            ),
        )

        eco.tick(dry_summer)
        layers = eco_world.rasters.list_layers(TIER)
        if "drought_stress" in layers:
            stress_1 = eco_world.rasters.read_layer(TIER, "drought_stress").copy()
        else:
            pytest.skip("drought_stress layer not written after first summer")

        eco.tick(dry_summer)
        eco.tick(dry_summer)
        stress_3 = eco_world.rasters.read_layer(TIER, "drought_stress")

        assert stress_3.mean() > stress_1.mean(), (
            f"3 dry summers should produce more stress than 1: "
            f"{stress_1.mean():.3f} -> {stress_3.mean():.3f}"
        )

    def test_drought_stress_recovers(self, eco_world):
        """Drought stress should decrease in non-summer seasons (decay by 0.9)."""
        eco = EcologyTier(eco_world)
        eco.tick(make_weather(SPRING))  # tick 0

        # Build up drought stress.
        dry_summer = make_weather(
            SUMMER,
            precipitation=np.full(
                (GRID_SIZE, GRID_SIZE), 50.0, dtype=np.float64
            ),
        )
        eco.tick(dry_summer)
        eco.tick(dry_summer)

        layers = eco_world.rasters.list_layers(TIER)
        if "drought_stress" not in layers:
            pytest.skip("drought_stress layer not written")

        stress_peak = eco_world.rasters.read_layer(TIER, "drought_stress").copy()
        peak_mean = stress_peak.mean()

        # Non-summer ticks should reduce stress.
        eco.tick(make_weather(FALL))
        eco.tick(make_weather(WINTER))
        eco.tick(make_weather(SPRING))

        stress_after = eco_world.rasters.read_layer(TIER, "drought_stress")
        after_mean = stress_after.mean()

        assert after_mean < peak_mean, (
            f"Drought stress should recover in non-summer seasons: "
            f"{peak_mean:.3f} -> {after_mean:.3f}"
        )


# ===========================================================================
# 3. Carrying capacity invariant
# ===========================================================================


class TestCarryingCapacity:
    """Total density per cell must never exceed carrying capacity."""

    def test_total_density_never_exceeds_capacity(self, eco_world):
        """Run 20 seasonal ticks; at every step total density <= 15.0."""
        eco = EcologyTier(eco_world)
        seasons = [WINTER, SPRING, SUMMER, FALL] * 5  # 20 ticks = 5 years

        for i, season in enumerate(seasons):
            eco.tick(make_weather(season))
            total = _total_density(eco_world)
            max_total = total.max()
            assert max_total <= CARRYING_CAPACITY + 1e-6, (
                f"Tick {i} (season {season}): max total density {max_total:.3f} "
                f"exceeds carrying capacity {CARRYING_CAPACITY}"
            )


# ===========================================================================
# 4. Mast seeding
# ===========================================================================


class TestMastSeeding:
    """Species with mast_interval > 1 should have variable seed production."""

    def test_mast_species_variable_seed_production(self, eco_world):
        """Over 8+ fall seasons, mast species should show variation in
        seed bank additions (not adding equally every year)."""
        eco = EcologyTier(eco_world)
        eco.tick(make_weather(SPRING))  # tick 0: ancestors

        # Identify a mast species (mast_interval > 1).
        mast_sid = None
        for sp in eco_world.events.list_species():
            genome = eco_world.events.get_species(sp["species_id"])["genome"]
            if genome.get("mast_interval", 1) > 1:
                mast_sid = sp["species_id"]
                break

        if mast_sid is None:
            pytest.skip("No mast species found in ancestors")

        # Run 8 years (cycle through seasons) and record seed bank after each fall.
        fall_seed_additions = []
        for year in range(8):
            eco.tick(make_weather(SPRING))
            eco.tick(make_weather(SUMMER))

            sb_layer = f"seed_bank_{mast_sid}"
            layers = eco_world.rasters.list_layers(TIER)
            sb_before = 0.0
            if sb_layer in layers:
                sb_before = eco_world.rasters.read_layer(TIER, sb_layer).sum()

            eco.tick(make_weather(FALL))

            if sb_layer in eco_world.rasters.list_layers(TIER):
                sb_after = eco_world.rasters.read_layer(TIER, sb_layer).sum()
            else:
                sb_after = 0.0

            fall_seed_additions.append(sb_after - sb_before)
            eco.tick(make_weather(WINTER))

        # Mast species should show variation: not all additions are equal.
        if len(fall_seed_additions) >= 4:
            additions = np.array(fall_seed_additions)
            cv = additions.std() / (abs(additions.mean()) + 1e-10)
            assert cv > 0.1, (
                f"Mast species seed production should vary across years "
                f"(coefficient of variation = {cv:.3f})"
            )


# ===========================================================================
# 5. Canopy shading
# ===========================================================================


class TestCanopyShading:
    """max_height should be load-bearing for light competition."""

    def test_tall_species_cast_shade(self, eco_world):
        """When a tall species (max_height > 10) has high density,
        shade-intolerant species should have reduced effective suitability
        (manifesting as lower growth or density)."""
        eco = EcologyTier(eco_world)

        # Run several ticks to establish populations.
        for season in [SPRING, SUMMER, FALL, WINTER] * 3:
            eco.tick(make_weather(season))

        # Find a tall species and a shade-intolerant species.
        tall_sid = None
        shade_intolerant_sid = None
        for sp in eco_world.events.list_species():
            genome = eco_world.events.get_species(sp["species_id"])["genome"]
            if genome.get("max_height", 0) > 10:
                tall_sid = sp["species_id"]
            if genome.get("shade_tolerance", 1.0) < 0.2 and genome.get("max_height", 100) < 2:
                shade_intolerant_sid = sp["species_id"]

        if tall_sid is None or shade_intolerant_sid is None:
            pytest.skip("Need both a tall and a shade-intolerant species")

        # In cells where tall species has high density, shade-intolerant species
        # should have lower density than in cells where tall species is absent.
        tall_density = eco_world.rasters.read_layer(
            TIER, f"species_{tall_sid}_density"
        )
        intolerant_density = eco_world.rasters.read_layer(
            TIER, f"species_{shade_intolerant_sid}_density"
        )

        tall_present = tall_density > np.percentile(tall_density[tall_density > 0], 75)
        tall_absent = tall_density < 0.01

        if tall_present.sum() < 100 or tall_absent.sum() < 100:
            pytest.skip("Not enough cells to compare shading effect")

        mean_under_canopy = intolerant_density[tall_present].mean()
        mean_in_open = intolerant_density[tall_absent].mean()

        # Shade-intolerant species should do worse under tall canopy.
        assert mean_under_canopy < mean_in_open, (
            f"Shade-intolerant species should have lower density under tall canopy: "
            f"under={mean_under_canopy:.3f}, open={mean_in_open:.3f}"
        )


# ===========================================================================
# 6. Basic invariants
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

    def test_seed_bank_non_negative(self, eco_world):
        """After several ticks, all seed bank arrays >= 0."""
        eco = EcologyTier(eco_world)
        for season in [SPRING, SUMMER, FALL, WINTER, SPRING, SUMMER]:
            eco.tick(make_weather(season))

        for sp in eco_world.events.list_species():
            sid = sp["species_id"]
            layer = f"seed_bank_{sid}"
            if layer in eco_world.rasters.list_layers(TIER):
                arr = eco_world.rasters.read_layer(TIER, layer)
                assert arr.min() >= 0.0, (
                    f"Seed bank for {sid} has negative values: min={arr.min():.6f}"
                )

    def test_drought_stress_non_negative(self, eco_world):
        """drought_stress raster is >= 0 after drought + recovery."""
        eco = EcologyTier(eco_world)
        eco.tick(make_weather(SPRING))  # tick 0

        # Drought then recovery.
        eco.tick(make_weather(
            SUMMER,
            precipitation=np.full((GRID_SIZE, GRID_SIZE), 50.0, dtype=np.float64),
        ))
        eco.tick(make_weather(FALL))
        eco.tick(make_weather(WINTER))

        layers = eco_world.rasters.list_layers(TIER)
        if "drought_stress" in layers:
            stress = eco_world.rasters.read_layer(TIER, "drought_stress")
            assert stress.min() >= 0.0, (
                f"Drought stress has negative values: min={stress.min():.6f}"
            )


# ===========================================================================
# 8. Integration: full year cycle
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
        """Run a complete year starting from spring (the natural order for
        a newly-created world) and verify the ecology is in a reasonable state."""
        eco = EcologyTier(eco_world)
        ws = make_weather_system(eco_world)

        year = 100.0  # arbitrary year
        for season in [SPRING, SUMMER, FALL, WINTER]:
            weather = ws.generate(year, season)
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


# ===========================================================================
# Additional seasonal behaviour tests
# ===========================================================================


class TestWinterDetails:
    """Additional winter behaviour checks."""

    def test_evergreen_species_retain_some_density(self, eco_world):
        """Evergreen species (high evergreenness) should not lose all density
        in winter, even with moderate frost."""
        eco = EcologyTier(eco_world)
        eco.tick(make_weather(SPRING))  # tick 0
        eco.tick(make_weather(SUMMER))  # growth

        # Find an evergreen species.
        evergreen_sid = None
        for sp in eco_world.events.list_species():
            genome = eco_world.events.get_species(sp["species_id"])["genome"]
            if genome.get("evergreenness", 0) > 0.6:
                evergreen_sid = sp["species_id"]
                break

        if evergreen_sid is None:
            pytest.skip("No evergreen species found")

        layer = f"species_{evergreen_sid}_density"
        density_before = eco_world.rasters.read_layer(TIER, layer).sum()

        if density_before == 0:
            pytest.skip("Evergreen species has no density (not viable in test weather)")

        # Moderate winter (some frost, not extreme).
        moderate_winter = make_weather(
            WINTER,
            frost_severity=np.full(
                (GRID_SIZE, GRID_SIZE), 0.4, dtype=np.float64
            ),
            temperature=np.full(
                (GRID_SIZE, GRID_SIZE), -5.0, dtype=np.float64
            ),
        )
        eco.tick(moderate_winter)

        density_after = eco_world.rasters.read_layer(TIER, layer).sum()

        # Should retain a significant fraction (not killed off entirely).
        assert density_after > density_before * 0.3, (
            f"Evergreen species lost too much density in moderate winter: "
            f"{density_before:.1f} -> {density_after:.1f}"
        )


class TestDroughtMortality:
    """Drought effects on species with different drought tolerance."""

    def test_drought_intolerant_species_suffer_more(self, eco_world):
        """Species with low drought_tolerance should lose more density
        during prolonged drought than drought-tolerant species."""
        eco = EcologyTier(eco_world)
        eco.tick(make_weather(SPRING))  # tick 0
        eco.tick(make_weather(SUMMER))  # initial growth

        densities_before = _species_densities(eco_world)

        # Three dry summers.
        dry_summer = make_weather(
            SUMMER,
            precipitation=np.full(
                (GRID_SIZE, GRID_SIZE), 30.0, dtype=np.float64
            ),
            temperature=np.full(
                (GRID_SIZE, GRID_SIZE), 30.0, dtype=np.float64
            ),
        )
        for _ in range(3):
            eco.tick(dry_summer)

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
