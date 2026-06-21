"""Tests for the timeline WorldQuery methods.

Verifies that the timeline query methods handle sample data correctly
and return empty results gracefully when no data exists. (These remain a
Layer-B query surface for global stats; the webview no longer exposes them
as routes.)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bike_sim.query.world_query import WorldQuery
from bike_sim.world import World

GRID_SIZE = 1000


# ── Helpers ────────────────────────────────────────────────────────


def _populate_timeline_world(world: World) -> None:
    """Insert tick_summary, tick_weather, and disturbance events into *world*."""
    # Register ancestor species and a child species
    world.events.add_species("anc_00_oak", genome={"height": 25.0}, appeared_year=0.0)
    world.events.add_species("anc_01_pine", genome={"height": 20.0}, appeared_year=0.0)
    world.events.add_species(
        "oak_child_01", genome={"height": 22.0},
        parent_id="anc_00_oak", appeared_year=5.0,
    )

    # Write tick summaries for 2 years (8 ticks, 4 per year)
    # Year 0: ticks 0-3, Year 1: ticks 4-7
    for tick in range(8):
        year = tick * 0.25
        season = tick % 4
        species_summaries = [
            {
                "species_id": "anc_00_oak",
                "total_density": 10.0 + tick,
                "occupied_cells": 100,
            },
            {
                "species_id": "anc_01_pine",
                "total_density": 5.0 + tick * 0.5,
                "occupied_cells": 50,
            },
        ]
        world.events.write_tick_summary(tick, year, season, species_summaries)

        world.events.write_tick_weather(
            tick, year, season,
            mean_temp=15.0 + season,
            mean_precip=100.0 - season * 10,
            mean_drought=0.1 + season * 0.05,
        )

    # Add disturbance events
    world.events.add_event(
        "fire", x=1000.0, y=2000.0, year=0.5,
        data={"cells_burned": 42},
    )
    world.events.add_event(
        "fire", x=3000.0, y=4000.0, year=1.0,
        data={"cells_burned": 100},
    )
    world.events.add_event(
        "blowdown", x=5000.0, y=6000.0, year=0.75,
        data={"cells_affected": 15},
    )
    # A non-disturbance event (should be ignored)
    world.events.add_event(
        "flood", x=7000.0, y=8000.0, year=1.25,
        data={"severity": "low"},
    )


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture()
def timeline_world_dir(tmp_path):
    """Create a world with timeline data, return the directory path."""
    world_path = tmp_path / "timeline_world"
    world = World.create(world_path, seed=42)

    heightmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    world.rasters.set_version(0)
    world.rasters.write_layer("geology", "heightmap", heightmap, tick_number=0)

    _populate_timeline_world(world)

    world.commit_version(trigger="test setup")
    world.save(world_path / "world.json")
    world.close()

    return world_path


@pytest.fixture()
def timeline_query(timeline_world_dir):
    """Open the timeline world and return a WorldQuery."""
    world = World.open(timeline_world_dir)
    query = WorldQuery(world)
    yield query
    world.close()


@pytest.fixture()
def empty_world_dir(tmp_path):
    """Create a world with no tick summary data, return the directory path."""
    world_path = tmp_path / "empty_world"
    world = World.create(world_path, seed=99)

    heightmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    world.rasters.set_version(0)
    world.rasters.write_layer("geology", "heightmap", heightmap, tick_number=0)

    world.commit_version(trigger="test setup")
    world.save(world_path / "world.json")
    world.close()

    return world_path


@pytest.fixture()
def empty_query(empty_world_dir):
    """Open the empty world and return a WorldQuery."""
    world = World.open(empty_world_dir)
    query = WorldQuery(world)
    yield query
    world.close()


# ── WorldQuery method tests ──────────────────────────────────────


class TestSpeciesTimeline:
    def test_basic_structure(self, timeline_query):
        result = timeline_query.get_species_timeline()

        assert "years" in result
        assert "species" in result
        assert len(result["years"]) == 2  # years 0 and 1
        assert 0.0 in result["years"]
        assert 1.0 in result["years"]

    def test_species_present(self, timeline_query):
        result = timeline_query.get_species_timeline()

        assert "anc_00_oak" in result["species"]
        assert "anc_01_pine" in result["species"]
        # Each species should have one value per year
        assert len(result["species"]["anc_00_oak"]) == 2
        assert len(result["species"]["anc_01_pine"]) == 2

    def test_uses_last_tick_of_year(self, timeline_query):
        result = timeline_query.get_species_timeline()

        # Year 0: last tick is 3, oak density = 10 + 3 = 13
        # Year 1: last tick is 7, oak density = 10 + 7 = 17
        oak = result["species"]["anc_00_oak"]
        assert oak[0] == pytest.approx(13.0)
        assert oak[1] == pytest.approx(17.0)

    def test_ancestor_filter(self, timeline_query):
        result = timeline_query.get_species_timeline(ancestor="anc_00_oak")

        # Should include anc_00_oak but not anc_01_pine
        assert "anc_00_oak" in result["species"]
        assert "anc_01_pine" not in result["species"]

    def test_empty_world(self, empty_query):
        result = empty_query.get_species_timeline()
        assert result == {"years": [], "species": {}}


class TestWeatherTimeline:
    def test_basic_structure(self, timeline_query):
        result = timeline_query.get_weather_timeline()

        assert "years" in result
        assert "temperature" in result
        assert "precipitation" in result
        assert "drought" in result
        assert len(result["years"]) == 2

    def test_yearly_mean(self, timeline_query):
        result = timeline_query.get_weather_timeline()

        # Year 0: seasons 0,1,2,3 -> temps 15,16,17,18 -> mean 16.5
        assert result["temperature"][0] == pytest.approx(16.5)
        # Year 0: precip 100,90,80,70 -> mean 85
        assert result["precipitation"][0] == pytest.approx(85.0)

    def test_empty_world(self, empty_query):
        result = empty_query.get_weather_timeline()
        assert result == {"years": [], "temperature": [], "precipitation": [], "drought": []}


class TestDiversityTimeline:
    def test_basic_structure(self, timeline_query):
        result = timeline_query.get_diversity_timeline()

        assert "years" in result
        assert "species_count" in result
        assert "shannon" in result
        assert "total_density" in result
        assert len(result["years"]) == 2

    def test_species_count(self, timeline_query):
        result = timeline_query.get_diversity_timeline()

        # Both species have density > 0.01 in both years
        assert result["species_count"][0] == 2
        assert result["species_count"][1] == 2

    def test_shannon_index(self, timeline_query):
        result = timeline_query.get_diversity_timeline()

        # Year 0, tick 3: oak=13, pine=5+1.5=6.5 -> total=19.5
        # p_oak = 13/19.5, p_pine = 6.5/19.5
        p_oak = 13.0 / 19.5
        p_pine = 6.5 / 19.5
        expected_h = -(p_oak * math.log(p_oak) + p_pine * math.log(p_pine))
        assert result["shannon"][0] == pytest.approx(expected_h, rel=1e-6)

    def test_total_density(self, timeline_query):
        result = timeline_query.get_diversity_timeline()

        # Year 0, tick 3: oak=13, pine=6.5 -> total=19.5
        assert result["total_density"][0] == pytest.approx(19.5)

    def test_empty_world(self, empty_query):
        result = empty_query.get_diversity_timeline()
        assert result == {"years": [], "species_count": [], "shannon": [], "total_density": []}


class TestDisturbanceTimeline:
    def test_fires(self, timeline_query):
        result = timeline_query.get_disturbance_timeline()

        assert len(result["fires"]) == 2
        fire_years = sorted(f["year"] for f in result["fires"])
        assert fire_years == [0.5, 1.0]

        # Check cells_burned is present
        fire_42 = [f for f in result["fires"] if f["cells_burned"] == 42]
        assert len(fire_42) == 1
        assert fire_42[0]["x"] == 1000.0

    def test_blowdowns(self, timeline_query):
        result = timeline_query.get_disturbance_timeline()

        assert len(result["blowdowns"]) == 1
        assert result["blowdowns"][0]["cells_affected"] == 15
        assert result["blowdowns"][0]["year"] == 0.75

    def test_ignores_other_events(self, timeline_query):
        result = timeline_query.get_disturbance_timeline()

        # The flood event should not appear
        total = len(result["fires"]) + len(result["blowdowns"])
        assert total == 3  # 2 fires + 1 blowdown, not 4

    def test_empty_world(self, empty_query):
        result = empty_query.get_disturbance_timeline()
        assert result == {"fires": [], "blowdowns": []}
