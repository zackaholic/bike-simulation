"""Tests for the WorldQuery layer-B query interface."""

import numpy as np

from bike_sim.query.world_query import WorldQuery
from bike_sim.world import World

# World constants: 50km at 50m resolution = 1000x1000 cells.
WORLD_SIZE_M = 50_000
CELL_SIZE_M = 50
GRID_SIZE = 1000


def _make_world(tmp_path):
    """Create a fresh World for testing and return (world, query)."""
    world = World.create(tmp_path / "test_world", seed=42)
    query = WorldQuery(world)
    return world, query


def test_sample_layer_point(tmp_path):
    """Point query maps world coordinates to the correct cell value."""
    world, query = _make_world(tmp_path)

    # Heightmap where value at cell (row, col) = row index.
    data = np.arange(GRID_SIZE, dtype=np.float64).reshape(-1, 1) * np.ones(
        (1, GRID_SIZE), dtype=np.float64
    )
    assert data.shape == (GRID_SIZE, GRID_SIZE)
    world.rasters.write_layer("geology", "heightmap", data, tick_number=0)

    # Cell (0, 0) corresponds to world coords near (0, 0).
    val = query.sample_layer("geology", "heightmap", 0.0, 0.0)
    assert val == 0.0

    # Cell (999, 999) corresponds to world coords (49950, 49950).
    val = query.sample_layer("geology", "heightmap", 49_950.0, 49_950.0)
    assert val == 999.0

    # Cell (500, 300) corresponds to world coords (15000, 25000).
    # x=15000 -> col 300, y=25000 -> row 500. Value = row = 500.
    val = query.sample_layer("geology", "heightmap", 15_000.0, 25_000.0)
    assert val == 500.0


def test_sample_layer_point_mid_cell(tmp_path):
    """Nearest-neighbor sampling returns the cell value for mid-cell coords."""
    world, query = _make_world(tmp_path)

    # Heightmap where value = row * 1000 + col, giving unique values per cell.
    rows = np.arange(GRID_SIZE, dtype=np.float64).reshape(-1, 1)
    cols = np.arange(GRID_SIZE, dtype=np.float64).reshape(1, -1)
    data = rows * 1000 + cols
    world.rasters.write_layer("geology", "heightmap", data, tick_number=0)

    # World coords (75, 125): x=75 -> col floor(75/50) = 1,
    # y=125 -> row floor(125/50) = 2. Expected value = 2*1000 + 1 = 2001.
    val = query.sample_layer("geology", "heightmap", 75.0, 125.0)
    assert val == 2001.0

    # World coords (49, 99): x=49 -> col 0, y=99 -> row 1. Value = 1000.
    val = query.sample_layer("geology", "heightmap", 49.0, 99.0)
    assert val == 1000.0


def test_sample_layer_region(tmp_path):
    """Region query returns the correct sub-array for a bounding box."""
    world, query = _make_world(tmp_path)

    # Heightmap where value = row * 1000 + col.
    rows = np.arange(GRID_SIZE, dtype=np.float64).reshape(-1, 1)
    cols = np.arange(GRID_SIZE, dtype=np.float64).reshape(1, -1)
    data = rows * 1000 + cols
    world.rasters.write_layer("geology", "heightmap", data, tick_number=0)

    # Query region: x in [100, 250], y in [200, 350].
    # x_min=100 -> col_min = floor(100/50) = 2
    # x_max=250 -> col_max = floor(250/50) = 5
    # y_min=200 -> row_min = floor(200/50) = 4
    # y_max=350 -> row_max = floor(350/50) = 7
    # Expected slice: data[4:8, 2:6] (inclusive on both ends -> +1 for slice).
    result = query.sample_layer_region("geology", "heightmap", 100.0, 200.0, 250.0, 350.0)

    expected = data[4:8, 2:6]
    assert result.shape == expected.shape
    np.testing.assert_array_equal(result, expected)


def test_species_at(tmp_path):
    """species_at returns only species with positive density at the target."""
    world, query = _make_world(tmp_path)

    # Register species in the EventStore.
    world.events.add_species("tree01", genome={"height": 30}, appeared_year=0.0)
    world.events.add_species("grass01", genome={"height": 0.5}, appeared_year=0.0)

    # Write density layers in ecology tier.
    # tree01: density = 5.0 everywhere.
    tree_density = np.full((GRID_SIZE, GRID_SIZE), 5.0)
    world.rasters.write_layer("ecology", "species_tree01_density", tree_density, tick_number=0)

    # grass01: density = 0.0 everywhere.
    grass_density = np.zeros((GRID_SIZE, GRID_SIZE))
    world.rasters.write_layer("ecology", "species_grass01_density", grass_density, tick_number=0)

    # Query at a point where tree01 has density but grass01 does not.
    results = query.species_at(5000.0, 5000.0, radius=100.0)

    species_ids = [r["species_id"] for r in results]
    assert "tree01" in species_ids
    assert "grass01" not in species_ids

    # Verify the returned density value for tree01.
    tree_entry = next(r for r in results if r["species_id"] == "tree01")
    assert tree_entry["density"] > 0


def test_individuals_near(tmp_path):
    """individuals_near delegates to EventStore and returns correct results."""
    world, query = _make_world(tmp_path)

    world.events.add_species("oak", genome={"height": 25}, appeared_year=0.0)
    world.events.add_individual("oak_001", "oak", x=1000.0, y=1000.0, appeared_year=100.0)
    world.events.add_individual("oak_002", "oak", x=1050.0, y=1050.0, appeared_year=200.0)
    world.events.add_individual("oak_003", "oak", x=5000.0, y=5000.0, appeared_year=300.0)

    # Query near (1000, 1000) with radius 200 — should find oak_001 and oak_002.
    results = query.individuals_near(1000.0, 1000.0, radius=200.0)
    ids = {r["individual_id"] for r in results}
    assert "oak_001" in ids
    assert "oak_002" in ids
    assert "oak_003" not in ids


def test_events_near_spatial(tmp_path):
    """events_near filters events by spatial proximity."""
    world, query = _make_world(tmp_path)

    world.events.add_event("fire", x=1000.0, y=1000.0, year=100.0, radius=50.0)
    world.events.add_event("flood", x=1100.0, y=1100.0, year=150.0, radius=30.0)
    world.events.add_event("fire", x=40000.0, y=40000.0, year=200.0, radius=100.0)

    # Query near (1000, 1000) with radius 500 — should find fire and flood, not distant fire.
    results = query.events_near(1000.0, 1000.0, radius=500.0)
    assert len(results) == 2
    event_types = {r["event_type"] for r in results}
    assert "fire" in event_types
    assert "flood" in event_types


def test_events_near_with_time_filter(tmp_path):
    """events_near respects year_start and year_end filters."""
    world, query = _make_world(tmp_path)

    world.events.add_event("fire", x=1000.0, y=1000.0, year=100.0, radius=50.0)
    world.events.add_event("flood", x=1050.0, y=1050.0, year=200.0, radius=30.0)
    world.events.add_event("drought", x=1020.0, y=1020.0, year=300.0, radius=20.0)

    # All three are spatially close. Filter to year 150..250 — only flood.
    results = query.events_near(1000.0, 1000.0, radius=500.0, year_start=150.0, year_end=250.0)
    assert len(results) == 1
    assert results[0]["event_type"] == "flood"

    # Filter to year 0..150 — only fire.
    results = query.events_near(1000.0, 1000.0, radius=500.0, year_start=0.0, year_end=150.0)
    assert len(results) == 1
    assert results[0]["event_type"] == "fire"


def test_available_layers(tmp_path):
    """available_layers returns correct layer names per tier."""
    world, query = _make_world(tmp_path)

    data = np.zeros((GRID_SIZE, GRID_SIZE))
    world.rasters.write_layer("geology", "heightmap", data, tick_number=0)
    world.rasters.write_layer("geology", "bedrock_type", data, tick_number=0)
    world.rasters.write_layer("ecology", "species_tree01_density", data, tick_number=0)

    geo_layers = query.available_layers("geology")
    assert set(geo_layers) == {"heightmap", "bedrock_type"}

    eco_layers = query.available_layers("ecology")
    assert set(eco_layers) == {"species_tree01_density"}

    # Tier with no layers should return an empty list.
    climate_layers = query.available_layers("climate_hydrology")
    assert climate_layers == []
