"""Tests for the ClimateHydrologyTier — climate fields, river networks, and derived state."""

import numpy as np
import pytest

from bike_sim.tiers.climate_hydrology import ClimateHydrologyTier
from bike_sim.tiers.erosion import ErosionParams
from bike_sim.tiers.geology import GeologyTier
from bike_sim.world import TierId, World

# Reduced erosion params for fast tests.
FAST_PARAMS = ErosionParams(num_particles=100, max_lifetime=30)

# All layers the climate-hydrology tier must produce.
EXPECTED_LAYERS = [
    "temperature",
    "precipitation",
    "flow_accumulation",
    "eroded_heightmap",
    "sediment_depth",
    "soil_moisture_summer",
    "soil_moisture_winter",
    "frost_days",
    "growing_degree_days",
    "solar_insolation",
    "distance_to_water",
]


@pytest.fixture
def geo_world(tmp_path):
    """Create a world with geology already ticked, ready for climate-hydrology."""
    world = World.create(tmp_path / "world", seed=42)
    GeologyTier(world).tick()
    return world


# ---------- Tier interface tests ----------


def test_first_tick_produces_all_layers(geo_world):
    """After geology tick + climate-hydrology tick, all expected layers exist."""
    ClimateHydrologyTier(geo_world, erosion_params=FAST_PARAMS).tick()

    layers = geo_world.rasters.list_layers(TierId.CLIMATE_HYDROLOGY)
    for name in EXPECTED_LAYERS:
        assert name in layers, f"Missing layer: {name}"


def test_all_layers_shape_and_dtype(geo_world):
    """All climate-hydrology layers are (1000, 1000) float64."""
    ClimateHydrologyTier(geo_world, erosion_params=FAST_PARAMS).tick()

    for name in EXPECTED_LAYERS:
        arr = geo_world.rasters.read_layer(TierId.CLIMATE_HYDROLOGY, name)
        assert arr.shape == (1000, 1000), f"{name} has wrong shape: {arr.shape}"
        assert arr.dtype == np.float64, f"{name} has wrong dtype: {arr.dtype}"


def test_deterministic_from_seed(tmp_path):
    """Two worlds with the same seed produce bit-identical climate-hydrology layers."""
    w1 = World.create(tmp_path / "world_a", seed=12345)
    w2 = World.create(tmp_path / "world_b", seed=12345)

    GeologyTier(w1).tick()
    GeologyTier(w2).tick()
    ClimateHydrologyTier(w1, erosion_params=FAST_PARAMS).tick()
    ClimateHydrologyTier(w2, erosion_params=FAST_PARAMS).tick()

    for name in EXPECTED_LAYERS:
        a = w1.rasters.read_layer(TierId.CLIMATE_HYDROLOGY, name)
        b = w2.rasters.read_layer(TierId.CLIMATE_HYDROLOGY, name)
        np.testing.assert_array_equal(a, b, err_msg=f"{name} differs between identical seeds")


def test_different_seeds_differ(tmp_path):
    """Two worlds with different seeds produce different climate-hydrology output."""
    w1 = World.create(tmp_path / "world_a", seed=1)
    w2 = World.create(tmp_path / "world_b", seed=2)

    GeologyTier(w1).tick()
    GeologyTier(w2).tick()
    ClimateHydrologyTier(w1, erosion_params=FAST_PARAMS).tick()
    ClimateHydrologyTier(w2, erosion_params=FAST_PARAMS).tick()

    # At least one layer must differ (check temperature as representative).
    t1 = w1.rasters.read_layer(TierId.CLIMATE_HYDROLOGY, "temperature")
    t2 = w2.rasters.read_layer(TierId.CLIMATE_HYDROLOGY, "temperature")
    assert not np.array_equal(t1, t2)


def test_tier_clock_advances(geo_world):
    """After one tick, clock tick_number=1 and simulated_year > 0."""
    ClimateHydrologyTier(geo_world, erosion_params=FAST_PARAMS).tick()

    clock = geo_world.tier_clocks[TierId.CLIMATE_HYDROLOGY]
    assert clock.tick_number == 1
    assert clock.simulated_year > 0.0


def test_requires_geology_first(tmp_path):
    """Ticking climate-hydrology without geology first should raise an error."""
    world = World.create(tmp_path / "world", seed=42)

    with pytest.raises(RuntimeError):
        ClimateHydrologyTier(world).tick()


# ---------- Climate envelope tests ----------


def test_temperature_decreases_with_elevation(geo_world):
    """Higher cells should generally have lower temperature (lapse rate)."""
    ClimateHydrologyTier(geo_world, erosion_params=FAST_PARAMS).tick()

    heightmap = geo_world.rasters.read_layer(TierId.GEOLOGY, "heightmap")
    temperature = geo_world.rasters.read_layer(TierId.CLIMATE_HYDROLOGY, "temperature")

    median_elev = np.median(heightmap)
    high_mask = heightmap > median_elev
    low_mask = heightmap <= median_elev

    mean_temp_high = temperature[high_mask].mean()
    mean_temp_low = temperature[low_mask].mean()

    assert mean_temp_low > mean_temp_high, (
        f"Low-elevation mean temp ({mean_temp_low:.1f}) should exceed "
        f"high-elevation mean temp ({mean_temp_high:.1f})"
    )


def test_precipitation_reasonable_range(geo_world):
    """All precipitation values should be positive and plausible (0 < precip < 5000 mm/yr)."""
    ClimateHydrologyTier(geo_world, erosion_params=FAST_PARAMS).tick()

    precip = geo_world.rasters.read_layer(TierId.CLIMATE_HYDROLOGY, "precipitation")
    assert np.all(np.isfinite(precip))
    assert precip.min() > 0.0, f"Minimum precipitation is {precip.min()}"
    assert precip.max() < 5000.0, f"Maximum precipitation is {precip.max()}"


# ---------- Hydrology tests ----------


def test_flow_accumulation_positive(geo_world):
    """All flow accumulation values should be >= 1 (each cell counts at least itself)."""
    ClimateHydrologyTier(geo_world, erosion_params=FAST_PARAMS).tick()

    flow = geo_world.rasters.read_layer(TierId.CLIMATE_HYDROLOGY, "flow_accumulation")
    assert np.all(np.isfinite(flow))
    assert flow.min() >= 1.0, f"Minimum flow accumulation is {flow.min()}"


def test_rivers_in_valleys(geo_world):
    """Cells with high flow accumulation (top 1%) should generally have below-median elevation."""
    ClimateHydrologyTier(geo_world, erosion_params=FAST_PARAMS).tick()

    flow = geo_world.rasters.read_layer(TierId.CLIMATE_HYDROLOGY, "flow_accumulation")
    eroded = geo_world.rasters.read_layer(TierId.CLIMATE_HYDROLOGY, "eroded_heightmap")

    threshold = np.percentile(flow, 99)
    river_mask = flow >= threshold

    median_elev = np.median(eroded)
    river_elevations = eroded[river_mask]

    # Majority of high-flow cells should be below median elevation.
    frac_below = np.mean(river_elevations < median_elev)
    assert frac_below > 0.5, (
        f"Only {frac_below:.1%} of river cells are below median elevation — "
        f"rivers should flow in valleys"
    )


def test_erosion_lowers_terrain(geo_world):
    """Eroded heightmap should have lower or equal mean elevation vs. the original."""
    ClimateHydrologyTier(geo_world, erosion_params=FAST_PARAMS).tick()

    original = geo_world.rasters.read_layer(TierId.GEOLOGY, "heightmap")
    eroded = geo_world.rasters.read_layer(TierId.CLIMATE_HYDROLOGY, "eroded_heightmap")

    assert eroded.mean() <= original.mean(), (
        f"Eroded mean ({eroded.mean():.1f}) should not exceed "
        f"original mean ({original.mean():.1f})"
    )


def test_erosion_preserves_extent(geo_world):
    """Eroded heightmap is same shape as geology heightmap, all values finite."""
    ClimateHydrologyTier(geo_world, erosion_params=FAST_PARAMS).tick()

    original = geo_world.rasters.read_layer(TierId.GEOLOGY, "heightmap")
    eroded = geo_world.rasters.read_layer(TierId.CLIMATE_HYDROLOGY, "eroded_heightmap")

    assert eroded.shape == original.shape
    assert np.all(np.isfinite(eroded))


# ---------- Derived-state cache tests ----------


def test_soil_moisture_range(geo_world):
    """Both summer and winter soil moisture should be in [0, 1]."""
    ClimateHydrologyTier(geo_world, erosion_params=FAST_PARAMS).tick()

    for season in ("soil_moisture_summer", "soil_moisture_winter"):
        arr = geo_world.rasters.read_layer(TierId.CLIMATE_HYDROLOGY, season)
        assert np.all(np.isfinite(arr)), f"{season} contains non-finite values"
        assert arr.min() >= 0.0, f"{season} min is {arr.min()}"
        assert arr.max() <= 1.0, f"{season} max is {arr.max()}"


def test_winter_moisture_gte_summer(geo_world):
    """Mean winter soil moisture should be >= mean summer moisture."""
    ClimateHydrologyTier(geo_world, erosion_params=FAST_PARAMS).tick()

    summer = geo_world.rasters.read_layer(TierId.CLIMATE_HYDROLOGY, "soil_moisture_summer")
    winter = geo_world.rasters.read_layer(TierId.CLIMATE_HYDROLOGY, "soil_moisture_winter")

    assert winter.mean() >= summer.mean(), (
        f"Winter mean moisture ({winter.mean():.3f}) should be >= "
        f"summer mean moisture ({summer.mean():.3f})"
    )


def test_growing_degree_days_positive(geo_world):
    """All growing degree day values should be >= 0."""
    ClimateHydrologyTier(geo_world, erosion_params=FAST_PARAMS).tick()

    gdd = geo_world.rasters.read_layer(TierId.CLIMATE_HYDROLOGY, "growing_degree_days")
    assert np.all(np.isfinite(gdd))
    assert gdd.min() >= 0.0, f"Minimum GDD is {gdd.min()}"


def test_distance_to_water_nonnegative(geo_world):
    """All distance-to-water values >= 0, and some cells should be 0 (water/river cells)."""
    ClimateHydrologyTier(geo_world, erosion_params=FAST_PARAMS).tick()

    dtw = geo_world.rasters.read_layer(TierId.CLIMATE_HYDROLOGY, "distance_to_water")
    assert np.all(np.isfinite(dtw))
    assert dtw.min() >= 0.0, f"Minimum distance_to_water is {dtw.min()}"
    assert np.any(dtw == 0.0), "No cells have distance_to_water == 0 (expected some water cells)"


def test_solar_insolation_range(geo_world):
    """Solar insolation values should be in [0, 1]."""
    ClimateHydrologyTier(geo_world, erosion_params=FAST_PARAMS).tick()

    sol = geo_world.rasters.read_layer(TierId.CLIMATE_HYDROLOGY, "solar_insolation")
    assert np.all(np.isfinite(sol))
    assert sol.min() >= 0.0, f"solar_insolation min is {sol.min()}"
    assert sol.max() <= 1.0, f"solar_insolation max is {sol.max()}"


# ---------- Sediment / particle-erosion tests ----------


def test_sediment_nonnegative(geo_world):
    """All sediment_depth values should be >= 0."""
    ClimateHydrologyTier(geo_world, erosion_params=FAST_PARAMS).tick()
    sed = geo_world.rasters.read_layer(TierId.CLIMATE_HYDROLOGY, "sediment_depth")
    assert np.all(np.isfinite(sed))
    assert sed.min() >= 0.0, f"Minimum sediment_depth is {sed.min()}"


def test_sediment_deposited_in_low_areas(geo_world):
    """Mean sediment should be higher in low-elevation areas (deposition zones)."""
    ClimateHydrologyTier(geo_world, erosion_params=FAST_PARAMS).tick()
    eroded = geo_world.rasters.read_layer(TierId.CLIMATE_HYDROLOGY, "eroded_heightmap")
    sed = geo_world.rasters.read_layer(TierId.CLIMATE_HYDROLOGY, "sediment_depth")

    median_elev = np.median(eroded)
    low_mean = sed[eroded <= median_elev].mean()
    high_mean = sed[eroded > median_elev].mean()

    assert low_mean > high_mean, (
        f"Low-elevation mean sediment ({low_mean:.4f}) should exceed "
        f"high-elevation mean ({high_mean:.4f})"
    )


def test_hard_rock_erodes_less(geo_world):
    """Harder bedrock types should show less erosion than softer types."""
    ClimateHydrologyTier(geo_world, erosion_params=FAST_PARAMS).tick()

    original = geo_world.rasters.read_layer(TierId.GEOLOGY, "heightmap")
    eroded = geo_world.rasters.read_layer(TierId.CLIMATE_HYDROLOGY, "eroded_heightmap")
    bedrock = geo_world.rasters.read_layer(TierId.GEOLOGY, "bedrock_type")

    erosion = original - eroded

    # Type 3 = granite (erodibility 0.3), type 2 = shale (erodibility 1.2)
    # Compare mean erosion depth. If either type isn't present, skip.
    hard_mask = bedrock == 3
    soft_mask = bedrock == 2
    if hard_mask.sum() < 100 or soft_mask.sum() < 100:
        pytest.skip("Not enough cells of rock types 2 and 3 to compare")

    hard_erosion = erosion[hard_mask].mean()
    soft_erosion = erosion[soft_mask].mean()

    assert soft_erosion > hard_erosion, (
        f"Soft rock (type 2) erosion ({soft_erosion:.4f}) should exceed "
        f"hard rock (type 3) erosion ({hard_erosion:.4f})"
    )


def test_flow_accumulation_reflects_erosion(geo_world):
    """Post-erosion flow accumulation should differ from pre-erosion flow."""
    from bike_sim.tiers.climate_hydrology import ClimateHydrologyTier as CHT

    original_hm = geo_world.rasters.read_layer(TierId.GEOLOGY, "heightmap")

    tier = CHT(geo_world, erosion_params=FAST_PARAMS)
    tier.tick()

    stored_flow = geo_world.rasters.read_layer(TierId.CLIMATE_HYDROLOGY, "flow_accumulation")

    # Compute what flow would be on the original unmodified heightmap
    pre_erosion_flow = tier._compute_flow_accumulation(original_hm)

    # They should differ because erosion changed the terrain
    assert not np.array_equal(stored_flow, pre_erosion_flow), (
        "Post-erosion flow accumulation should differ from pre-erosion"
    )
