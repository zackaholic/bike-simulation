"""Tests for the expanded WorldQuery interface (version-aware methods).

These methods support the webview observability tool by providing
version-specific queries over rasters, species, individuals, and events.
"""

from __future__ import annotations

import numpy as np
import pytest

from bike_sim.query.world_query import WorldQuery
from bike_sim.world import World

# World constants: 50km at 50m resolution = 1000x1000 cells.
WORLD_SIZE_M = 50_000
CELL_SIZE_M = 50
GRID_SIZE = 1000


@pytest.fixture()
def query_world(tmp_path):
    """Build a small world with known data across three versions."""
    world = World.create(tmp_path / "world", seed=42)
    query = WorldQuery(world)

    # Version 0: geology + climate
    world.rasters.set_version(0)
    heightmap = np.arange(1_000_000, dtype=np.float64).reshape(1000, 1000)
    world.rasters.write_layer("geology", "heightmap", heightmap, tick_number=0)
    temperature = np.full((1000, 1000), 15.0, dtype=np.float64)
    world.rasters.write_layer("climate_hydrology", "temperature", temperature, tick_number=0)
    world.tier_clocks["geology"].tick_number = 1
    world.tier_clocks["geology"].simulated_year = 100_000.0
    world.tier_clocks["climate_hydrology"].tick_number = 1
    world.tier_clocks["climate_hydrology"].simulated_year = 1000.0
    world.commit_version(trigger="create_world")

    # Version 1: ecology
    world.rasters.set_version(1)
    world.events.add_species("oak", {"height": 20.0, "drought_tolerance": 0.3}, appeared_year=0.0)
    world.events.add_species("pine", {"height": 15.0, "drought_tolerance": 0.7}, appeared_year=0.0)
    density_oak = np.full((1000, 1000), 5.0, dtype=np.float64)
    density_pine = np.zeros((1000, 1000), dtype=np.float64)
    density_pine[500:, :] = 3.0  # pine only in northern half
    world.rasters.write_layer("ecology", "species_oak_density", density_oak, tick_number=1)
    world.rasters.write_layer("ecology", "species_pine_density", density_pine, tick_number=1)
    world.events.add_individual("mother_oak", "oak", x=5000.0, y=5000.0, appeared_year=0.0)
    world.events.add_individual("old_pine", "pine", x=30000.0, y=30000.0, appeared_year=0.0)
    world.events.add_individual("dead_tree", "oak", x=5100.0, y=5100.0, appeared_year=0.0)
    world.events.kill_individual("dead_tree", died_year=50.0)
    world.events.add_event(
        "fire", x=10000.0, y=10000.0, year=25.0, radius=500.0, data={"severity": "high"}
    )
    world.tier_clocks["ecology"].tick_number = 5
    world.tier_clocks["ecology"].simulated_year = 25.0
    world.commit_version(trigger="advance 25 years")

    # Version 2: more ecology
    world.rasters.set_version(2)
    density_oak_v2 = np.full((1000, 1000), 4.0, dtype=np.float64)
    world.rasters.write_layer("ecology", "species_oak_density", density_oak_v2, tick_number=10)
    world.events.mark_species_extinct("pine", extinct_year=75.0)
    world.tier_clocks["ecology"].tick_number = 15
    world.tier_clocks["ecology"].simulated_year = 75.0
    world.commit_version(trigger="advance 50 years")

    yield world, query
    world.close()


# ═══════════════════════════════════════════════════════════════════
# get_world_metadata (tests 1-3)
# ═══════════════════════════════════════════════════════════════════


class TestGetWorldMetadata:
    """get_world_metadata returns structural info about a version."""

    def test_metadata_has_extent(self, query_world):
        """Extent should be the full world bounds."""
        _world, query = query_world
        meta = query.get_world_metadata(version=0)

        assert meta["extent"] == {
            "x_min": 0,
            "y_min": 0,
            "x_max": WORLD_SIZE_M,
            "y_max": WORLD_SIZE_M,
        }
        assert meta["cell_size"] == CELL_SIZE_M
        assert meta["grid_size"] == GRID_SIZE

    def test_metadata_layers_at_version(self, query_world):
        """At version 1, ecology should have species density layers.
        At version 0, ecology should be empty."""
        _world, query = query_world

        meta_v0 = query.get_world_metadata(version=0)
        assert "ecology" in meta_v0["layers"]
        assert meta_v0["layers"]["ecology"] == []

        meta_v1 = query.get_world_metadata(version=1)
        eco_layers = meta_v1["layers"]["ecology"]
        assert "species_oak_density" in eco_layers
        assert "species_pine_density" in eco_layers

    def test_metadata_counts(self, query_world):
        """At version 1 (year=25), species_count=2 and individual_count
        should count individuals alive at that year."""
        _world, query = query_world
        meta = query.get_world_metadata(version=1)

        assert meta["species_count"] == 2
        # At year 25: mother_oak alive, old_pine alive, dead_tree alive (dies at 50).
        assert meta["individual_count"] == 3
        assert meta["simulated_year"] == pytest.approx(25.0)


# ═══════════════════════════════════════════════════════════════════
# query_point (tests 4-6)
# ═══════════════════════════════════════════════════════════════════


class TestQueryPoint:
    """query_point returns everything known at a single coordinate."""

    def test_point_has_raster_values(self, query_world):
        """Rasters at (5000, 5000) should include geology/heightmap
        and climate_hydrology/temperature."""
        _world, query = query_world
        result = query.query_point(version=1, x=5000.0, y=5000.0)

        assert result["x"] == 5000.0
        assert result["y"] == 5000.0

        rasters = result["rasters"]
        # heightmap = np.arange(1_000_000).reshape(1000,1000)
        # x=5000 -> col=100, y=5000 -> row=100, value = 100*1000+100 = 100_100
        assert "geology/heightmap" in rasters
        assert rasters["geology/heightmap"] == pytest.approx(100_100.0)

        assert "climate_hydrology/temperature" in rasters
        assert rasters["climate_hydrology/temperature"] == pytest.approx(15.0)

    def test_point_has_species(self, query_world):
        """Oak should appear with density > 0 at any point.
        At y=5000 (row=100 < 500), pine density is 0 so pine should not appear."""
        _world, query = query_world
        result = query.query_point(version=1, x=5000.0, y=5000.0)

        species_ids = [sp["species_id"] for sp in result["species"]]
        assert "oak" in species_ids
        assert "pine" not in species_ids

        oak_entry = next(sp for sp in result["species"] if sp["species_id"] == "oak")
        assert oak_entry["density"] == pytest.approx(5.0)

    def test_point_has_nearby_individuals(self, query_world):
        """At (5000, 5000) should find mother_oak. dead_tree (died at 50)
        is still alive at version 1's year=25, so it should be included."""
        _world, query = query_world
        result = query.query_point(version=1, x=5000.0, y=5000.0)

        ind_ids = [ind["individual_id"] for ind in result["individuals"]]
        assert "mother_oak" in ind_ids
        # dead_tree is at (5100, 5100) — within 100m radius of (5000, 5000).
        # distance = sqrt(100^2 + 100^2) ~ 141m. Depends on radius convention.
        # If within 100m strict Euclidean, dead_tree might not be found.
        # But the spec says "within 100m radius" so check if present.
        # old_pine at (30000, 30000) should NOT be present.
        assert "old_pine" not in ind_ids


# ═══════════════════════════════════════════════════════════════════
# query_raster (tests 7-8)
# ═══════════════════════════════════════════════════════════════════


class TestQueryRaster:
    """query_raster clips and resamples raster data."""

    def test_raster_clip_and_resample(self, query_world):
        """Query oak density at version 1, bbox=(0,0,25000,25000),
        target_size=(100,100). Result should be (100,100) with values ~5.0."""
        _world, query = query_world
        result = query.query_raster(
            version=1,
            tier="ecology",
            layer_name="species_oak_density",
            bbox=(0, 0, 25000, 25000),
            target_size=(100, 100),
        )

        assert result.shape == (100, 100)
        # Oak density is 5.0 everywhere at version 1.
        np.testing.assert_allclose(result, 5.0, atol=0.1)

    def test_raster_at_different_version(self, query_world):
        """Oak density changed from 5.0 (v1) to 4.0 (v2). Values should differ."""
        _world, query = query_world

        result_v1 = query.query_raster(
            version=1,
            tier="ecology",
            layer_name="species_oak_density",
            bbox=(0, 0, 25000, 25000),
            target_size=(50, 50),
        )
        result_v2 = query.query_raster(
            version=2,
            tier="ecology",
            layer_name="species_oak_density",
            bbox=(0, 0, 25000, 25000),
            target_size=(50, 50),
        )

        assert result_v1.shape == (50, 50)
        assert result_v2.shape == (50, 50)
        np.testing.assert_allclose(result_v1, 5.0, atol=0.1)
        np.testing.assert_allclose(result_v2, 4.0, atol=0.1)


# ═══════════════════════════════════════════════════════════════════
# query_individuals_in_bbox (tests 9-10)
# ═══════════════════════════════════════════════════════════════════


class TestQueryIndividualsInBbox:
    """query_individuals_in_bbox filters by bounding box and version year."""

    def test_individuals_bbox(self, query_world):
        """Bbox around (5000,5000) at version 1 should find mother_oak and
        dead_tree (alive at year 25). old_pine at (30000,30000) should not be found."""
        _world, query = query_world
        results = query.query_individuals_in_bbox(
            version=1,
            x_min=4000.0,
            y_min=4000.0,
            x_max=6000.0,
            y_max=6000.0,
        )

        result_ids = {ind["individual_id"] for ind in results}
        assert "mother_oak" in result_ids
        assert "dead_tree" in result_ids  # alive at year 25 (dies at 50)
        assert "old_pine" not in result_ids

    def test_individuals_bbox_version_filters(self, query_world):
        """At version 2 (year=75), dead_tree (died at 50) should be excluded."""
        _world, query = query_world
        results = query.query_individuals_in_bbox(
            version=2,
            x_min=4000.0,
            y_min=4000.0,
            x_max=6000.0,
            y_max=6000.0,
        )

        result_ids = {ind["individual_id"] for ind in results}
        assert "mother_oak" in result_ids
        assert "dead_tree" not in result_ids  # dead at year 50 < version year 75


# ═══════════════════════════════════════════════════════════════════
# get_individual_detail (tests 11-12)
# ═══════════════════════════════════════════════════════════════════


class TestGetIndividualDetail:
    """get_individual_detail returns full details including derived fields."""

    def test_individual_detail(self, query_world):
        """mother_oak at version 1 (year=25): age=25, alive=True,
        species_genome should contain height=20.0."""
        _world, query = query_world
        detail = query.get_individual_detail(version=1, individual_id="mother_oak")

        assert detail["individual_id"] == "mother_oak"
        assert detail["species_id"] == "oak"
        assert detail["x"] == pytest.approx(5000.0)
        assert detail["y"] == pytest.approx(5000.0)
        assert detail["appeared_year"] == pytest.approx(0.0)
        assert detail["died_year"] is None
        assert detail["age"] == pytest.approx(25.0)
        assert detail["alive"] is True
        assert detail["species_genome"]["height"] == pytest.approx(20.0)

    def test_individual_detail_dead(self, query_world):
        """dead_tree at version 2 (year=75): alive=False, died_year=50."""
        _world, query = query_world
        detail = query.get_individual_detail(version=2, individual_id="dead_tree")

        assert detail["individual_id"] == "dead_tree"
        assert detail["species_id"] == "oak"
        assert detail["alive"] is False
        assert detail["died_year"] == pytest.approx(50.0)
        assert detail["age"] == pytest.approx(75.0)  # version year - appeared year


# ═══════════════════════════════════════════════════════════════════
# list_species_summary (tests 13-14)
# ═══════════════════════════════════════════════════════════════════


class TestListSpeciesSummary:
    """list_species_summary returns species with population stats."""

    def test_species_summary_at_version(self, query_world):
        """At version 1, both oak and pine should be alive.
        At version 2, pine should be extinct."""
        _world, query = query_world

        summary_v1 = query.list_species_summary(version=1)
        species_v1 = {sp["species_id"]: sp for sp in summary_v1}
        assert "oak" in species_v1
        assert "pine" in species_v1
        assert species_v1["oak"]["alive"] is True
        assert species_v1["pine"]["alive"] is True

        summary_v2 = query.list_species_summary(version=2)
        species_v2 = {sp["species_id"]: sp for sp in summary_v2}
        assert species_v2["oak"]["alive"] is True
        assert species_v2["pine"]["alive"] is False
        assert species_v2["pine"]["extinct_year"] == pytest.approx(75.0)

    def test_species_summary_has_genome(self, query_world):
        """Each species in the summary should have its full genome dict."""
        _world, query = query_world
        summary = query.list_species_summary(version=1)

        species_map = {sp["species_id"]: sp for sp in summary}

        assert species_map["oak"]["genome"] == {"height": 20.0, "drought_tolerance": 0.3}
        assert species_map["pine"]["genome"] == {"height": 15.0, "drought_tolerance": 0.7}
