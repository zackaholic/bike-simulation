"""Tests for the versioning system across RasterStore, EventStore, and World.

RasterStore gains version-specific Zarr groups (copy-on-write per version).
EventStore gains lifecycle columns (died_year, extinct_year) with temporal queries.
World gains a version log tracking tier clock snapshots and layer ownership.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bike_sim.state.event_store import EventStore
from bike_sim.state.raster_store import RasterStore
from bike_sim.world import TierClock, World

# ── RasterStore fixtures ────────────────────────────────────────────


@pytest.fixture()
def raster_store(tmp_path):
    """Create a fresh RasterStore in a temporary directory."""
    return RasterStore.create(tmp_path / "rasters.zarr")


# ── EventStore fixtures ─────────────────────────────────────────────


@pytest.fixture()
def event_store(tmp_path):
    """Create a fresh EventStore for each test."""
    db_path = tmp_path / "events.db"
    store = EventStore.create(db_path)
    yield store
    store.close()


# ═══════════════════════════════════════════════════════════════════
# RasterStore versioning (tests 1-7)
# ═══════════════════════════════════════════════════════════════════


class TestRasterStoreVersioning:
    """Version-specific Zarr groups with copy-on-write semantics."""

    def test_write_to_version(self, raster_store):
        """set_version(1), write a layer, read it at version 1 -- data matches."""
        data = np.random.default_rng(0).random((64, 64))
        raster_store.set_version(1)
        raster_store.write_layer("geology", "heightmap", data, tick_number=1)

        result = raster_store.read_layer("geology", "heightmap", version=1)
        np.testing.assert_array_equal(result, data)

    def test_version_isolation(self, raster_store):
        """Layers written in version 1 are visible at version 2 via fallback,
        but layers written in version 2 are not visible at version 1."""
        data_a = np.ones((64, 64), dtype=np.float64)
        data_b = np.ones((64, 64), dtype=np.float64) * 2.0

        raster_store.set_version(1)
        raster_store.write_layer("geology", "layer_a", data_a, tick_number=1)

        raster_store.set_version(2)
        raster_store.write_layer("geology", "layer_b", data_b, tick_number=2)

        # layer_a should be readable at version 2 (falls back to version 1)
        result_a = raster_store.read_layer("geology", "layer_a", version=2)
        np.testing.assert_array_equal(result_a, data_a)

        # layer_b should NOT be readable at version 1
        with pytest.raises(KeyError):
            raster_store.read_layer("geology", "layer_b", version=1)

    def test_overwrite_in_new_version(self, raster_store):
        """Writing the same layer in a new version creates a separate copy;
        each version returns its own data."""
        data_x = np.full((64, 64), 10.0, dtype=np.float64)
        data_y = np.full((64, 64), 99.0, dtype=np.float64)

        raster_store.set_version(1)
        raster_store.write_layer("geology", "heightmap", data_x, tick_number=1)

        raster_store.set_version(2)
        raster_store.write_layer("geology", "heightmap", data_y, tick_number=2)

        result_v1 = raster_store.read_layer("geology", "heightmap", version=1)
        result_v2 = raster_store.read_layer("geology", "heightmap", version=2)

        np.testing.assert_array_equal(result_v1, data_x)
        np.testing.assert_array_equal(result_v2, data_y)

    def test_list_versions(self, raster_store):
        """list_versions returns sorted version IDs that have been written to."""
        for ver in [1, 2, 3]:
            raster_store.set_version(ver)
            raster_store.write_layer(
                "geology", f"layer_{ver}", np.zeros((16, 16)), tick_number=ver
            )

        assert raster_store.list_versions() == [1, 2, 3]

    def test_get_layer_version(self, raster_store):
        """get_layer_version returns which version owns the latest copy of a layer."""
        raster_store.set_version(1)
        raster_store.write_layer("geology", "heightmap", np.zeros((32, 32)), tick_number=1)

        raster_store.set_version(2)
        # Don't write heightmap at version 2 -- it should still be owned by v1.

        assert raster_store.get_layer_version("geology", "heightmap") == 1

    def test_read_latest_default(self, raster_store):
        """read_layer with no version parameter returns the latest available data."""
        data = np.random.default_rng(7).random((32, 32))

        raster_store.set_version(1)
        raster_store.write_layer("geology", "heightmap", data, tick_number=1)

        raster_store.set_version(2)
        # No writes at version 2.

        result = raster_store.read_layer("geology", "heightmap")
        np.testing.assert_array_equal(result, data)

    def test_list_layers_at_version(self, raster_store):
        """list_layers with a version includes layers from earlier versions
        (via ownership) plus layers written at the requested version."""
        raster_store.set_version(1)
        raster_store.write_layer("geology", "layer_a", np.zeros((16, 16)), tick_number=1)
        raster_store.write_layer("geology", "layer_b", np.zeros((16, 16)), tick_number=1)

        raster_store.set_version(2)
        raster_store.write_layer("geology", "layer_c", np.zeros((16, 16)), tick_number=2)

        layers_v2 = sorted(raster_store.list_layers("geology", version=2))
        assert layers_v2 == ["layer_a", "layer_b", "layer_c"]

        layers_v1 = sorted(raster_store.list_layers("geology", version=1))
        assert layers_v1 == ["layer_a", "layer_b"]


# ═══════════════════════════════════════════════════════════════════
# EventStore lifecycle (tests 8-13)
# ═══════════════════════════════════════════════════════════════════


class TestEventStoreLifecycle:
    """Lifecycle tracking: died_year for individuals, extinct_year for species."""

    def test_kill_individual(self, event_store):
        """kill_individual sets died_year on the individual record."""
        event_store.add_species("oak", {"height": 20.0})
        event_store.add_individual("tree_1", "oak", x=10.0, y=20.0, appeared_year=0.0)

        event_store.kill_individual("tree_1", died_year=100.0)

        ind = event_store.get_individual("tree_1")
        assert ind["died_year"] == 100.0

    def test_find_individuals_alive_at(self, event_store):
        """find_individuals_near with alive_at_year filters out individuals
        that have not yet appeared or have already died."""
        event_store.add_species("birch", {"height": 15.0})

        # Individual that never dies
        event_store.add_individual("alive", "birch", x=50.0, y=50.0, appeared_year=0.0)
        # Individual that died at year 50 (dead at year 100)
        event_store.add_individual("died_early", "birch", x=50.5, y=50.0, appeared_year=0.0)
        event_store.kill_individual("died_early", died_year=50.0)
        # Individual that died at year 150 (still alive at year 100)
        event_store.add_individual("died_late", "birch", x=49.5, y=50.0, appeared_year=0.0)
        event_store.kill_individual("died_late", died_year=150.0)

        results = event_store.find_individuals_near(50.0, 50.0, radius=5.0, alive_at_year=100.0)
        result_ids = {ind["individual_id"] for ind in results}

        assert result_ids == {"alive", "died_late"}

    def test_find_individuals_no_filter(self, event_store):
        """find_individuals_near without alive_at_year returns all individuals
        including dead ones (backward compatibility)."""
        event_store.add_species("birch", {"height": 15.0})

        event_store.add_individual("alive", "birch", x=50.0, y=50.0, appeared_year=0.0)
        event_store.add_individual("dead", "birch", x=50.5, y=50.0, appeared_year=0.0)
        event_store.kill_individual("dead", died_year=50.0)

        results = event_store.find_individuals_near(50.0, 50.0, radius=5.0)
        result_ids = {ind["individual_id"] for ind in results}

        assert result_ids == {"alive", "dead"}

    def test_mark_species_extinct(self, event_store):
        """mark_species_extinct sets extinct_year on the species record."""
        event_store.add_species("fern", {"frond_count": 12.0}, appeared_year=0.0)

        event_store.mark_species_extinct("fern", extinct_year=500.0)

        sp = event_store.get_species("fern")
        assert sp["extinct_year"] == 500.0

    def test_list_species_alive_at(self, event_store):
        """list_species with alive_at_year filters out species that have not
        yet appeared or have gone extinct."""
        # Extant species (no extinction)
        event_store.add_species("sp_extant", {"trait": 1.0}, appeared_year=0.0)
        # Extinct at year 50 (dead at year 100)
        event_store.add_species("sp_extinct_early", {"trait": 2.0}, appeared_year=0.0)
        event_store.mark_species_extinct("sp_extinct_early", extinct_year=50.0)
        # Extinct at year 150 (still alive at year 100)
        event_store.add_species("sp_extinct_late", {"trait": 3.0}, appeared_year=0.0)
        event_store.mark_species_extinct("sp_extinct_late", extinct_year=150.0)

        results = event_store.list_species(alive_at_year=100.0)
        result_ids = {sp["species_id"] for sp in results}

        assert result_ids == {"sp_extant", "sp_extinct_late"}

    def test_list_species_no_filter(self, event_store):
        """list_species without filter returns all species including extinct ones
        (backward compatibility)."""
        event_store.add_species("sp_alive", {"trait": 1.0})
        event_store.add_species("sp_dead", {"trait": 2.0})
        event_store.mark_species_extinct("sp_dead", extinct_year=100.0)

        results = event_store.list_species()
        result_ids = {sp["species_id"] for sp in results}

        assert result_ids == {"sp_alive", "sp_dead"}


# ═══════════════════════════════════════════════════════════════════
# World version log (tests 14-20)
# ═══════════════════════════════════════════════════════════════════


class TestWorldVersionLog:
    """World tracks a version log with tier clock snapshots and layer ownership."""

    def test_commit_version(self, tmp_path):
        """commit_version creates a version entry with incremented ID,
        tier_clocks snapshot, and a timestamp."""
        world = World.create(tmp_path / "world", seed=42)

        world.commit_version(trigger="create_world")

        assert world.current_version == 0
        entry = world.get_version(0)
        assert entry["trigger"] == "create_world"
        assert "timestamp" in entry
        assert "tier_clocks" in entry

        world.close()

    def test_version_has_tier_clocks(self, tmp_path):
        """The version entry's tier_clocks should match the world's
        current tier_clocks at commit time."""
        world = World.create(tmp_path / "world", seed=42)
        world.tier_clocks["geology"].tick_number = 5
        world.tier_clocks["geology"].simulated_year = 500_000.0

        world.commit_version(trigger="geology advance")

        entry = world.get_version(world.current_version)
        geo_clock = entry["tier_clocks"]["geology"]
        assert geo_clock["tick_number"] == 5
        assert geo_clock["simulated_year"] == 500_000.0

        world.close()

    def test_version_has_layer_index(self, tmp_path):
        """After writing raster layers and committing, the version entry's
        layer_index maps each tier/layer to the correct owning version."""
        world = World.create(tmp_path / "world", seed=42)

        world.rasters.set_version(0)
        world.rasters.write_layer("geology", "heightmap", np.zeros((32, 32)), tick_number=0)
        world.rasters.write_layer("geology", "bedrock", np.zeros((32, 32)), tick_number=0)

        world.commit_version(trigger="initial generation")

        entry = world.get_version(world.current_version)
        layer_index = entry["layer_index"]

        assert layer_index["geology/heightmap"] == 0
        assert layer_index["geology/bedrock"] == 0

        world.close()

    def test_version_for_tier(self, tmp_path):
        """version_for_tier returns the TierClock snapshot for a given tier
        at a given version."""
        world = World.create(tmp_path / "world", seed=42)
        world.tier_clocks["geology"].tick_number = 10
        world.tier_clocks["geology"].simulated_year = 1_000_000.0

        world.commit_version(trigger="geology done")

        clock = world.version_for_tier(world.current_version, "geology")
        assert isinstance(clock, TierClock)
        assert clock.tick_number == 10
        assert clock.simulated_year == 1_000_000.0

        world.close()

    def test_multiple_versions(self, tmp_path):
        """Committing 3 versions with different tier states produces a complete
        version log with correct tier_clocks at each entry."""
        world = World.create(tmp_path / "world", seed=42)

        for tick in range(3):
            world.tier_clocks["geology"].tick_number = tick
            world.tier_clocks["geology"].simulated_year = float(tick * 100_000)
            world.commit_version(trigger=f"advance step {tick}")

        versions = world.list_versions()
        assert len(versions) == 3
        assert [v["version_id"] for v in versions] == [0, 1, 2]

        # Each version should have captured its own tier state.
        assert versions[0]["tier_clocks"]["geology"]["tick_number"] == 0
        assert versions[1]["tier_clocks"]["geology"]["tick_number"] == 1
        assert versions[2]["tier_clocks"]["geology"]["tick_number"] == 2

        world.close()

    def test_version_persists_save_load(self, tmp_path):
        """Committed versions survive a save/close/reopen cycle."""
        world_path = tmp_path / "world"
        world = World.create(world_path, seed=42)

        world.tier_clocks["ecology"].tick_number = 7
        world.commit_version(trigger="ecology advance")
        world.tier_clocks["ecology"].tick_number = 14
        world.commit_version(trigger="ecology advance 2")

        world.save(world_path / "world.json")
        world.close()

        reopened = World.open(world_path)
        versions = reopened.list_versions()

        assert len(versions) == 2
        assert versions[0]["trigger"] == "ecology advance"
        assert versions[0]["tier_clocks"]["ecology"]["tick_number"] == 7
        assert versions[1]["trigger"] == "ecology advance 2"
        assert versions[1]["tier_clocks"]["ecology"]["tick_number"] == 14

        reopened.close()

    def test_atomic_manifest_write(self, tmp_path):
        """After commit_version, world.json contains the version log as valid JSON."""
        world_path = tmp_path / "world"
        world = World.create(world_path, seed=42)

        world.commit_version(trigger="initial")
        world.save(world_path / "world.json")

        # Read the manifest file directly and parse it.
        raw = (world_path / "world.json").read_text()
        manifest = json.loads(raw)  # Should not raise.

        assert "versions" in manifest
        assert len(manifest["versions"]) == 1
        assert manifest["versions"][0]["trigger"] == "initial"

        world.close()
