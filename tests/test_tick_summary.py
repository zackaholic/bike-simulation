"""Tests for tick summary logging infrastructure.

Covers:
  - EventStore tick_summary and tick_weather write/read methods.
  - Orchestrator produces summaries at the configured interval.
  - snapshot_interval creates multiple raster versions mid-loop.
"""

import numpy as np
import pytest

from bike_sim.orchestrator import Orchestrator
from bike_sim.state.event_store import EventStore
from bike_sim.tiers.erosion import ErosionParams
from bike_sim.world import World

FAST_PARAMS = ErosionParams(num_particles=1_000, max_lifetime=30)


# ── EventStore unit tests ───────────────────────────────────────────────


class TestEventStoreSummary:
    """Direct write/read tests for tick_summary and tick_weather tables."""

    @pytest.fixture
    def store(self, tmp_path):
        return EventStore.create(tmp_path / "events.db")

    def test_write_and_read_tick_summary(self, store):
        summaries = [
            {"species_id": "sp_a", "total_density": 100.5, "occupied_cells": 42, "mean_biomass_age": 3.2},
            {"species_id": "sp_b", "total_density": 50.0, "occupied_cells": 20, "mean_biomass_age": 1.0},
        ]
        store.write_tick_summary(tick=4, year=1.0, season=0, species_summaries=summaries)

        rows = store.get_tick_summaries()
        assert len(rows) == 2
        assert rows[0]["species_id"] == "sp_a"
        assert rows[0]["total_density"] == pytest.approx(100.5)
        assert rows[0]["occupied_cells"] == 42
        assert rows[1]["species_id"] == "sp_b"

    def test_write_and_read_tick_weather(self, store):
        store.write_tick_weather(tick=4, year=1.0, season=0, mean_temp=12.5, mean_precip=800.0, mean_drought=0.1)
        rows = store.get_tick_weather()
        assert len(rows) == 1
        assert rows[0]["tick"] == 4
        assert rows[0]["mean_temp"] == pytest.approx(12.5)
        assert rows[0]["mean_precip"] == pytest.approx(800.0)
        assert rows[0]["mean_drought"] == pytest.approx(0.1)

    def test_filter_by_species(self, store):
        store.write_tick_summary(tick=4, year=1.0, season=0, species_summaries=[
            {"species_id": "sp_a", "total_density": 10, "occupied_cells": 5, "mean_biomass_age": 1},
            {"species_id": "sp_b", "total_density": 20, "occupied_cells": 8, "mean_biomass_age": 2},
        ])
        store.write_tick_summary(tick=8, year=2.0, season=0, species_summaries=[
            {"species_id": "sp_a", "total_density": 15, "occupied_cells": 6, "mean_biomass_age": 1.5},
        ])
        rows = store.get_tick_summaries(species_id="sp_a")
        assert len(rows) == 2
        assert all(r["species_id"] == "sp_a" for r in rows)

    def test_filter_by_tick_range(self, store):
        for tick in [4, 8, 12, 16]:
            store.write_tick_summary(tick=tick, year=tick * 0.25, season=0, species_summaries=[
                {"species_id": "sp_a", "total_density": tick, "occupied_cells": 1, "mean_biomass_age": 0},
            ])
        rows = store.get_tick_summaries(tick_start=8, tick_end=12)
        assert len(rows) == 2
        assert rows[0]["tick"] == 8
        assert rows[1]["tick"] == 12

    def test_filter_weather_by_tick_range(self, store):
        for tick in [4, 8, 12]:
            store.write_tick_weather(tick=tick, year=tick * 0.25, season=0, mean_temp=10, mean_precip=500, mean_drought=0)
        rows = store.get_tick_weather(tick_start=8)
        assert len(rows) == 2

    def test_upsert_on_duplicate(self, store):
        """write_tick_summary uses INSERT OR REPLACE, so re-writing the same tick+species overwrites."""
        store.write_tick_summary(tick=4, year=1.0, season=0, species_summaries=[
            {"species_id": "sp_a", "total_density": 10, "occupied_cells": 5, "mean_biomass_age": 1},
        ])
        store.write_tick_summary(tick=4, year=1.0, season=0, species_summaries=[
            {"species_id": "sp_a", "total_density": 99, "occupied_cells": 50, "mean_biomass_age": 9},
        ])
        rows = store.get_tick_summaries()
        assert len(rows) == 1
        assert rows[0]["total_density"] == pytest.approx(99)

    def test_migrate_adds_tables(self, tmp_path):
        """Opening an old database (without tick tables) should add them via _migrate."""
        import sqlite3
        db_path = tmp_path / "old.db"
        conn = sqlite3.connect(str(db_path))
        # Create only the original tables (simulate old schema)
        conn.executescript("""
            CREATE TABLE species (
                species_id TEXT PRIMARY KEY, genome_json TEXT NOT NULL,
                parent_id TEXT, appeared_year REAL DEFAULT 0.0, extinct_year REAL
            );
            CREATE TABLE individuals (
                individual_id TEXT PRIMARY KEY, species_id TEXT NOT NULL,
                x REAL NOT NULL, y REAL NOT NULL, appeared_year REAL DEFAULT 0.0,
                died_year REAL, state TEXT DEFAULT 'alive'
            );
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT NOT NULL,
                x REAL NOT NULL, y REAL NOT NULL, year REAL NOT NULL,
                radius REAL DEFAULT 0.0, data_json TEXT
            );
        """)
        conn.close()

        store = EventStore.open(db_path)
        # Should be able to write tick summaries after migration
        store.write_tick_weather(tick=1, year=0.25, season=1, mean_temp=10, mean_precip=500, mean_drought=0)
        rows = store.get_tick_weather()
        assert len(rows) == 1
        store.close()


# ── Orchestrator integration tests ──────────────────────────────────────


@pytest.fixture
def ready_world(tmp_path):
    """Create a world with geology+climate ready, using fast erosion params."""
    world = World.create(tmp_path / "world", seed=42)
    orch = Orchestrator(world, erosion_params=FAST_PARAMS)
    orch.create_world()
    return world, orch


def test_advance_produces_tick_summaries(ready_world):
    """Advancing 8 seasons with summary_interval=4 should produce summaries at ticks 4 and 8."""
    world, orch = ready_world
    # Default summary_interval=4
    orch.advance_seasons(8)

    summaries = world.events.get_tick_summaries()
    weather = world.events.get_tick_weather()

    # Should have summaries at tick 4 and tick 8
    summary_ticks = sorted(set(r["tick"] for r in summaries))
    weather_ticks = sorted(r["tick"] for r in weather)

    assert 4 in summary_ticks
    assert 8 in summary_ticks
    assert 4 in weather_ticks
    assert 8 in weather_ticks


def test_summary_interval_respected(tmp_path):
    """Custom summary_interval=2 should log every 2 ticks."""
    world = World.create(tmp_path / "world", seed=42)
    orch = Orchestrator(world, erosion_params=FAST_PARAMS, summary_interval=2)
    orch.create_world()
    orch.advance_seasons(8)

    summaries = world.events.get_tick_summaries()
    summary_ticks = sorted(set(r["tick"] for r in summaries))
    assert 2 in summary_ticks
    assert 4 in summary_ticks
    assert 6 in summary_ticks
    assert 8 in summary_ticks


def test_summary_interval_zero_disables(tmp_path):
    """summary_interval=0 should produce no summaries."""
    world = World.create(tmp_path / "world", seed=42)
    orch = Orchestrator(world, erosion_params=FAST_PARAMS, summary_interval=0)
    orch.create_world()
    orch.advance_seasons(8)

    assert len(world.events.get_tick_summaries()) == 0
    assert len(world.events.get_tick_weather()) == 0


def test_summaries_have_species_data(ready_world):
    """Each summary row should have non-negative density and cell counts."""
    world, orch = ready_world
    orch.advance_seasons(4)

    summaries = world.events.get_tick_summaries()
    assert len(summaries) > 0
    for s in summaries:
        assert s["total_density"] >= 0
        assert s["occupied_cells"] >= 0
        assert s["mean_biomass_age"] >= 0
        assert isinstance(s["species_id"], str)


def test_snapshot_interval_creates_versions(tmp_path):
    """snapshot_interval=8 over 20 ticks should create intermediate versions."""
    world = World.create(tmp_path / "world", seed=42)
    orch = Orchestrator(
        world,
        erosion_params=FAST_PARAMS,
        snapshot_interval=8,
    )
    orch.create_world()

    versions_before = len(world.list_versions())
    orch.advance_seasons(20)
    versions_after = len(world.list_versions())

    # advance_seasons commits a final version, plus intermediate snapshots
    # at ticks 8 and 16 (tick 0 excluded by the >0 guard).
    # So we expect at least 2 intermediate + 1 final = 3 new versions.
    new_versions = versions_after - versions_before
    assert new_versions >= 3, f"Expected >= 3 new versions, got {new_versions}"


def test_snapshot_interval_disabled(ready_world):
    """snapshot_interval=0 should not create intermediate versions."""
    world, orch = ready_world
    # Override snapshot interval to 0
    orch._snapshot_interval = 0

    versions_before = len(world.list_versions())
    orch.advance_seasons(20)
    versions_after = len(world.list_versions())

    # Only the final commit from advance_seasons
    assert versions_after - versions_before == 1
