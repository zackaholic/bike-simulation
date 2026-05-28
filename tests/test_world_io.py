"""Tests for World directory creation, persistence, and round-trip loading.

These tests exercise the integrated World API (World.create / World.open)
which manages a world directory containing:
  - world.json  (manifest)
  - rasters/    (Zarr store via RasterStore)
  - events.db   (SQLite database via EventStore)
"""

from __future__ import annotations

import numpy as np

from bike_sim.state.event_store import EventStore
from bike_sim.state.raster_store import RasterStore
from bike_sim.world import World


class TestWorldCreate:
    """World.create builds a valid world directory."""

    def test_creates_directory_structure(self, tmp_path):
        """World.create makes the directory with world.json, rasters/, events.db."""
        world_dir = tmp_path / "test_world"
        World.create(world_dir, seed=42)

        assert world_dir.is_dir()
        assert (world_dir / "world.json").is_file()
        assert (world_dir / "rasters").exists()
        assert (world_dir / "events.db").is_file()

    def test_returns_world_with_correct_seed(self, tmp_path):
        """World.create returns a World instance with the given seed."""
        world_dir = tmp_path / "test_world"
        world = World.create(world_dir, seed=12345)

        assert isinstance(world, World)
        assert world.seed == 12345

    def test_has_raster_store_attribute(self, tmp_path):
        """The returned World has a working rasters attribute."""
        world_dir = tmp_path / "test_world"
        world = World.create(world_dir, seed=42)

        assert isinstance(world.rasters, RasterStore)

    def test_has_event_store_attribute(self, tmp_path):
        """The returned World has a working events attribute."""
        world_dir = tmp_path / "test_world"
        world = World.create(world_dir, seed=42)

        assert isinstance(world.events, EventStore)

    def test_rasters_are_usable(self, tmp_path):
        """Can write and read raster data through the World's RasterStore."""
        world_dir = tmp_path / "test_world"
        world = World.create(world_dir, seed=42)

        data = np.ones((64, 64), dtype=np.float64) * 3.14
        world.rasters.write_layer("geology", "heightmap", data, tick_number=0)
        result = world.rasters.read_layer("geology", "heightmap")
        np.testing.assert_array_equal(result, data)

    def test_events_are_usable(self, tmp_path):
        """Can add and query events through the World's EventStore."""
        world_dir = tmp_path / "test_world"
        world = World.create(world_dir, seed=42)

        world.events.add_event(
            "uplift",
            x=100.0,
            y=200.0,
            year=50000.0,
            data={"region": "north", "magnitude": 500.0},
        )
        events = world.events.get_events_in_region(0.0, 0.0, 500.0, 500.0)
        assert len(events) == 1
        assert events[0]["event_type"] == "uplift"


class TestWorldRoundTrip:
    """Write data into a world, close it, reopen, verify everything persists."""

    def test_manifest_round_trip(self, tmp_path):
        """World.open restores seed, tier_clocks, and simulated_year."""
        world_dir = tmp_path / "test_world"
        world = World.create(world_dir, seed=98765)
        world.simulated_year = 500_000.0
        world.tier_clocks["geology"].tick_number = 10
        world.tier_clocks["geology"].simulated_year = 500_000.0
        world.tier_clocks["ecology"].tick_number = 200
        world.tier_clocks["ecology"].simulated_year = 499_800.0
        # Save manifest so changes persist (create writes initial state;
        # mutations need an explicit save before close).
        world.save(world_dir / "world.json")
        world.close()

        reopened = World.open(world_dir)
        assert reopened.seed == 98765
        assert reopened.simulated_year == 500_000.0
        assert reopened.tier_clocks["geology"].tick_number == 10
        assert reopened.tier_clocks["geology"].simulated_year == 500_000.0
        assert reopened.tier_clocks["ecology"].tick_number == 200
        assert reopened.tier_clocks["ecology"].simulated_year == 499_800.0
        # Unchanged tier should still be at defaults
        assert reopened.tier_clocks["climate_hydrology"].tick_number == 0
        reopened.close()

    def test_raster_data_persists(self, tmp_path):
        """Raster data written before close is readable after World.open."""
        world_dir = tmp_path / "test_world"
        rng = np.random.default_rng(42)
        heightmap = rng.random((128, 128))
        bedrock = rng.integers(0, 10, size=(128, 128), dtype=np.int32)

        world = World.create(world_dir, seed=42)
        world.rasters.write_layer("geology", "heightmap", heightmap, tick_number=1)
        world.rasters.write_layer("geology", "bedrock_type", bedrock, tick_number=1)
        world.close()

        reopened = World.open(world_dir)
        np.testing.assert_array_equal(
            reopened.rasters.read_layer("geology", "heightmap"), heightmap
        )
        np.testing.assert_array_equal(
            reopened.rasters.read_layer("geology", "bedrock_type"), bedrock
        )
        assert reopened.rasters.read_layer("geology", "bedrock_type").dtype == np.int32
        reopened.close()

    def test_event_data_persists(self, tmp_path):
        """Events written before close are queryable after World.open."""
        world_dir = tmp_path / "test_world"
        world = World.create(world_dir, seed=42)

        world.events.add_event(
            "uplift",
            x=100.0,
            y=100.0,
            year=10000.0,
            data={"magnitude": 100.0},
        )
        world.events.add_event(
            "fire",
            x=500.0,
            y=500.0,
            year=50000.0,
            data={"area_ha": 250.0, "severity": "high"},
        )
        world.close()

        reopened = World.open(world_dir)
        # Spatial query that captures both events
        all_events = reopened.events.get_events_in_region(0.0, 0.0, 1000.0, 1000.0)
        assert len(all_events) == 2

        types = {e["event_type"] for e in all_events}
        assert types == {"uplift", "fire"}
        reopened.close()

    def test_full_round_trip(self, tmp_path):
        """Full integration: manifest + rasters + events all survive close/reopen."""
        world_dir = tmp_path / "test_world"
        world = World.create(world_dir, seed=77777)
        world.simulated_year = 10_000.0
        world.tier_clocks["geology"].tick_number = 3
        world.tier_clocks["geology"].simulated_year = 10_000.0

        heightmap = np.arange(256, dtype=np.float64).reshape(16, 16)
        world.rasters.write_layer("geology", "heightmap", heightmap, tick_number=3)

        world.events.add_event(
            "erosion",
            x=250.0,
            y=250.0,
            year=8000.0,
            data={"rate": 0.01},
        )

        world.save(world_dir / "world.json")
        world.close()

        reopened = World.open(world_dir)

        # Manifest
        assert reopened.seed == 77777
        assert reopened.simulated_year == 10_000.0
        assert reopened.tier_clocks["geology"].tick_number == 3

        # Raster
        np.testing.assert_array_equal(
            reopened.rasters.read_layer("geology", "heightmap"), heightmap
        )

        # Events
        events = reopened.events.get_events_in_region(0.0, 0.0, 500.0, 500.0)
        assert len(events) == 1
        assert events[0]["event_type"] == "erosion"
        reopened.close()


class TestWorldClose:
    """Closing a world is safe and the directory remains valid."""

    def test_close_does_not_error(self, tmp_path):
        """world.close() completes without raising."""
        world_dir = tmp_path / "test_world"
        world = World.create(world_dir, seed=42)
        world.close()  # Should not raise

    def test_directory_valid_after_close(self, tmp_path):
        """After close the directory can be reopened successfully."""
        world_dir = tmp_path / "test_world"
        world = World.create(world_dir, seed=42)

        data = np.zeros((32, 32), dtype=np.float64)
        world.rasters.write_layer("geology", "test", data, tick_number=0)
        world.events.add_event("init", x=0.0, y=0.0, year=0.0)
        world.close()

        # Directory still has the right structure
        assert (world_dir / "world.json").is_file()
        assert (world_dir / "rasters").exists()
        assert (world_dir / "events.db").is_file()

        # And it can be reopened
        reopened = World.open(world_dir)
        np.testing.assert_array_equal(reopened.rasters.read_layer("geology", "test"), data)
        events = reopened.events.get_events_in_region(-1.0, -1.0, 1.0, 1.0)
        assert len(events) == 1
        reopened.close()

    def test_double_close_does_not_error(self, tmp_path):
        """Calling close() twice should be safe (idempotent)."""
        world_dir = tmp_path / "test_world"
        world = World.create(world_dir, seed=42)
        world.close()
        world.close()  # Second close should not raise
