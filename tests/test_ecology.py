"""Tests for the EcologyTier — species creation, population dynamics, and niche differentiation."""

import re

import numpy as np
import pytest

from bike_sim.tiers.climate_hydrology import ClimateHydrologyTier
from bike_sim.tiers.ecology import EcologyTier
from bike_sim.tiers.geology import GeologyTier
from bike_sim.world import TierId, World

REQUIRED_GENOME_KEYS = {
    "drought_tolerance",
    "frost_tolerance",
    "shade_tolerance",
    "growth_rate",
    "seed_mass",
    "max_height",
    "lifespan",
}


@pytest.fixture
def eco_world(tmp_path):
    """Create a world with geology and climate-hydrology already ticked."""
    world = World.create(tmp_path / "world", seed=42)
    GeologyTier(world).tick()
    ClimateHydrologyTier(world).tick()
    return world


# ---------- Tier interface tests ----------


def test_first_tick_creates_ancestor_species(eco_world):
    """After a full-stack tick, EventStore should contain 5-8 ancestor species."""
    EcologyTier(eco_world).tick()

    species_list = eco_world.events.list_species()
    assert 5 <= len(species_list) <= 8, f"Expected 5-8 ancestor species, got {len(species_list)}"

    for sp in species_list:
        info = eco_world.events.get_species(sp["species_id"])
        genome = info["genome"]
        assert isinstance(genome, dict)
        missing = REQUIRED_GENOME_KEYS - genome.keys()
        assert not missing, f"Species {sp['species_id']} genome missing keys: {missing}"


def test_first_tick_produces_density_layers(eco_world):
    """After ticking, ecology tier should have a density layer per species."""
    EcologyTier(eco_world).tick()

    species_list = eco_world.events.list_species()
    layers = eco_world.rasters.list_layers(TierId.ECOLOGY)

    for sp in species_list:
        layer_name = f"species_{sp['species_id']}_density"
        assert layer_name in layers, f"Missing density layer: {layer_name}"


def test_density_layers_shape_and_dtype(eco_world):
    """All species density layers should be (1000, 1000) float64."""
    EcologyTier(eco_world).tick()

    layers = eco_world.rasters.list_layers(TierId.ECOLOGY)
    density_layers = [lyr for lyr in layers if re.match(r"species_.*_density$", lyr)]
    assert len(density_layers) > 0, "No density layers found"

    for name in density_layers:
        arr = eco_world.rasters.read_layer(TierId.ECOLOGY, name)
        assert arr.shape == (1000, 1000), f"{name} has wrong shape: {arr.shape}"
        assert arr.dtype == np.float64, f"{name} has wrong dtype: {arr.dtype}"


def test_density_nonnegative(eco_world):
    """All density values should be >= 0."""
    EcologyTier(eco_world).tick()

    layers = eco_world.rasters.list_layers(TierId.ECOLOGY)
    density_layers = [lyr for lyr in layers if re.match(r"species_.*_density$", lyr)]

    for name in density_layers:
        arr = eco_world.rasters.read_layer(TierId.ECOLOGY, name)
        assert np.all(arr >= 0), f"{name} contains negative densities (min={arr.min()})"


def test_deterministic_from_seed(tmp_path):
    """Two worlds with the same seed produce bit-identical ecology density layers."""
    worlds = []
    for suffix in ("a", "b"):
        w = World.create(tmp_path / f"world_{suffix}", seed=12345)
        GeologyTier(w).tick()
        ClimateHydrologyTier(w).tick()
        EcologyTier(w).tick()
        worlds.append(w)

    w1, w2 = worlds
    layers1 = sorted(
        lyr
        for lyr in w1.rasters.list_layers(TierId.ECOLOGY)
        if re.match(r"species_.*_density$", lyr)
    )
    layers2 = sorted(
        lyr
        for lyr in w2.rasters.list_layers(TierId.ECOLOGY)
        if re.match(r"species_.*_density$", lyr)
    )
    assert layers1 == layers2, "Different density layer sets between identical seeds"

    for name in layers1:
        a = w1.rasters.read_layer(TierId.ECOLOGY, name)
        b = w2.rasters.read_layer(TierId.ECOLOGY, name)
        np.testing.assert_array_equal(a, b, err_msg=f"{name} differs between identical seeds")


def test_different_seeds_differ(tmp_path):
    """Different seeds should produce different ecology output."""
    results = {}
    for seed in (1, 2):
        w = World.create(tmp_path / f"world_{seed}", seed=seed)
        GeologyTier(w).tick()
        ClimateHydrologyTier(w).tick()
        EcologyTier(w).tick()
        layers = sorted(
            lyr
            for lyr in w.rasters.list_layers(TierId.ECOLOGY)
            if re.match(r"species_.*_density$", lyr)
        )
        # Read the first density layer as representative.
        results[seed] = w.rasters.read_layer(TierId.ECOLOGY, layers[0])

    assert not np.array_equal(results[1], results[2])


def test_tier_clock_advances(eco_world):
    """After one tick, ecology clock tick_number=1 and simulated_year > 0."""
    EcologyTier(eco_world).tick()

    clock = eco_world.tier_clocks[TierId.ECOLOGY]
    assert clock.tick_number == 1
    assert clock.simulated_year > 0.0


def test_requires_climate_hydrology_first(tmp_path):
    """Ticking ecology without climate-hydrology should raise RuntimeError."""
    world = World.create(tmp_path / "world", seed=42)

    with pytest.raises(RuntimeError):
        EcologyTier(world).tick()


# ---------- Ecological behavior tests ----------


def test_species_have_different_distributions(eco_world):
    """Different species should occupy different niches (correlation < 0.9)."""
    EcologyTier(eco_world).tick()

    layers = sorted(
        lyr
        for lyr in eco_world.rasters.list_layers(TierId.ECOLOGY)
        if re.match(r"species_.*_density$", lyr)
    )
    assert len(layers) >= 2, "Need at least 2 species to compare distributions"

    a = eco_world.rasters.read_layer(TierId.ECOLOGY, layers[0]).ravel()
    b = eco_world.rasters.read_layer(TierId.ECOLOGY, layers[1]).ravel()

    # Avoid degenerate case where both are all zeros.
    if np.std(a) > 0 and np.std(b) > 0:
        corr = np.corrcoef(a, b)[0, 1]
        assert corr < 0.9, (
            f"Species distributions too similar (correlation={corr:.3f}), "
            f"expected niche differentiation"
        )


def test_total_density_bounded(eco_world):
    """Sum of all species densities at each cell should be bounded by competition."""
    EcologyTier(eco_world).tick()

    layers = [
        lyr
        for lyr in eco_world.rasters.list_layers(TierId.ECOLOGY)
        if re.match(r"species_.*_density$", lyr)
    ]

    total = np.zeros((1000, 1000), dtype=np.float64)
    for name in layers:
        total += eco_world.rasters.read_layer(TierId.ECOLOGY, name)

    assert np.all(total < 100.0), (
        f"Total density exceeds bound at some cells (max={total.max():.1f})"
    )


def test_species_prefer_suitable_habitat(eco_world):
    """Species with high drought_tolerance should have higher mean density in dry cells."""
    EcologyTier(eco_world).tick()

    species_list = eco_world.events.list_species()
    # Sort species by drought_tolerance.
    species_by_dt = sorted(
        species_list,
        key=lambda s: eco_world.events.get_species(s["species_id"])["genome"]["drought_tolerance"],
    )

    low_dt_sp = species_by_dt[0]
    high_dt_sp = species_by_dt[-1]

    # Use summer soil moisture as proxy for drought stress (low moisture = dry).
    moisture = eco_world.rasters.read_layer(TierId.CLIMATE_HYDROLOGY, "soil_moisture_summer")
    dry_mask = moisture < np.percentile(moisture, 25)

    low_dt_density = eco_world.rasters.read_layer(
        TierId.ECOLOGY, f"species_{low_dt_sp['species_id']}_density"
    )
    high_dt_density = eco_world.rasters.read_layer(
        TierId.ECOLOGY, f"species_{high_dt_sp['species_id']}_density"
    )

    mean_low_dt_in_dry = low_dt_density[dry_mask].mean()
    mean_high_dt_in_dry = high_dt_density[dry_mask].mean()

    assert mean_high_dt_in_dry > mean_low_dt_in_dry, (
        f"Drought-tolerant species mean density in dry cells ({mean_high_dt_in_dry:.4f}) "
        f"should exceed drought-intolerant species ({mean_low_dt_in_dry:.4f})"
    )


def test_multiple_ticks_change_density(eco_world):
    """After 3 ticks, density fields should differ from tick 1 (populations are dynamic)."""
    eco = EcologyTier(eco_world)
    eco.tick()

    layers = sorted(
        lyr
        for lyr in eco_world.rasters.list_layers(TierId.ECOLOGY)
        if re.match(r"species_.*_density$", lyr)
    )
    densities_after_1 = {
        name: eco_world.rasters.read_layer(TierId.ECOLOGY, name).copy() for name in layers
    }

    eco.tick()
    eco.tick()

    changed = False
    for name in layers:
        current = eco_world.rasters.read_layer(TierId.ECOLOGY, name)
        if not np.array_equal(current, densities_after_1[name]):
            changed = True
            break

    assert changed, "Density fields did not change between tick 1 and tick 3"


def test_multiple_ticks_density_stays_nonneg(eco_world):
    """After 3 ticks, all density values should still be >= 0."""
    eco = EcologyTier(eco_world)
    eco.tick()
    eco.tick()
    eco.tick()

    layers = [
        lyr
        for lyr in eco_world.rasters.list_layers(TierId.ECOLOGY)
        if re.match(r"species_.*_density$", lyr)
    ]

    for name in layers:
        arr = eco_world.rasters.read_layer(TierId.ECOLOGY, name)
        assert np.all(arr >= 0), (
            f"{name} contains negative densities after 3 ticks (min={arr.min()})"
        )


def test_seed_bank_exists_after_tick(eco_world):
    """After ticking, a 'seed_bank_total' layer should exist with nonnegative values."""
    EcologyTier(eco_world).tick()

    layers = eco_world.rasters.list_layers(TierId.ECOLOGY)
    assert "seed_bank_total" in layers, "Missing seed_bank_total layer"

    arr = eco_world.rasters.read_layer(TierId.ECOLOGY, "seed_bank_total")
    assert arr.shape == (1000, 1000)
    assert np.all(arr >= 0), f"seed_bank_total contains negative values (min={arr.min()})"


def test_not_all_species_everywhere(eco_world):
    """At least one species should have zero density in >10% of cells."""
    EcologyTier(eco_world).tick()

    layers = [
        lyr
        for lyr in eco_world.rasters.list_layers(TierId.ECOLOGY)
        if re.match(r"species_.*_density$", lyr)
    ]

    found_sparse = False
    for name in layers:
        arr = eco_world.rasters.read_layer(TierId.ECOLOGY, name)
        zero_frac = np.mean(arr == 0.0)
        if zero_frac > 0.10:
            found_sparse = True
            break

    assert found_sparse, (
        "Every species has density > 0 in >90% of cells — "
        "expected at least one species to be absent from >10% of cells"
    )
