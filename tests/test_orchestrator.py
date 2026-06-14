"""Tests for the Orchestrator — seasonal tick loop across all three simulation tiers."""

import numpy as np
import pytest

from bike_sim.orchestrator import Orchestrator
from bike_sim.tiers.erosion import ErosionParams
from bike_sim.world import World

from bike_sim.tiers.erosion import ErosionParams

FAST_EROSION = ErosionParams(num_particles=100, max_lifetime=30)


@pytest.fixture
def orch(tmp_path):
    """Create a fresh world with an Orchestrator, nothing ticked yet."""
    world = World.create(tmp_path / "world", seed=42)
    return Orchestrator(world, erosion_params=FAST_EROSION)


@pytest.fixture
def ready_world(fresh_world):
    """World with geology+climate ready, using the session-scoped base."""
    orch = Orchestrator(fresh_world, erosion_params=FAST_EROSION)
    return fresh_world, orch


# ── Core orchestration ────────────────────────────────────────────────


def test_create_world_ticks_geology_and_climate(orch):
    """After create_world(), geology=1 tick, climate_hydrology=1 tick, ecology=0."""
    orch.create_world()
    world = orch._world

    assert world.tier_clocks["geology"].tick_number == 1
    assert world.tier_clocks["climate_hydrology"].tick_number == 1
    assert world.tier_clocks["ecology"].tick_number == 0


def test_create_world_produces_heightmap(orch):
    """After create_world(), the geology heightmap layer should exist."""
    orch.create_world()
    world = orch._world
    assert "heightmap" in world.rasters.list_layers("geology")


def test_advance_5_years(ready_world):
    """advance(5) should produce 20 seasonal ticks (5 / 0.25)."""
    world, orch = ready_world

    result = orch.advance(5)

    assert result["ecology_ticks"] == 20
    assert result["seasons_advanced"] == 20
    assert world.tier_clocks["ecology"].tick_number == 20


@pytest.mark.slow
def test_advance_small_increments(tmp_path):
    """Five advance(1) calls should produce the same state as one advance(5)."""
    # World A: incremental advances.
    world_a = World.create(tmp_path / "world_a", seed=42)
    orch_a = Orchestrator(world_a, erosion_params=FAST_EROSION)
    orch_a.create_world()
    for _ in range(5):
        orch_a.advance(1)

    # World B: single large advance.
    world_b = World.create(tmp_path / "world_b", seed=42)
    orch_b = Orchestrator(world_b, erosion_params=FAST_EROSION)
    orch_b.create_world()
    orch_b.advance(5)

    # Both should have the same ecology tick count: 5 / 0.25 = 20.
    assert world_a.tier_clocks["ecology"].tick_number == 20
    assert world_b.tier_clocks["ecology"].tick_number == 20
    # Same simulated year.
    eco_year_a = world_a.tier_clocks["ecology"].simulated_year
    eco_year_b = world_b.tier_clocks["ecology"].simulated_year
    assert eco_year_a == eco_year_b


def test_advance_returns_summary(ready_world):
    """advance() return dict should contain the required keys."""
    _world, orch = ready_world
    result = orch.advance(5)

    assert "years_advanced" in result
    assert "ecology_ticks" in result
    assert "seasons_advanced" in result
    assert result["years_advanced"] == pytest.approx(5.0)


def test_advance_ride(ready_world):
    """advance_ride(20) should advance 20 seasons (5 years)."""
    _world, orch = ready_world
    result = orch.advance_ride(20)

    # 20 minutes * 1 season/minute = 20 seasons = 5 years
    assert result["seasons_advanced"] == 20
    assert result["years_advanced"] == pytest.approx(5.0)


@pytest.mark.slow
def test_advance_ride_capped(ready_world):
    """advance_ride(200) should cap at MAX_SEASONS_PER_RIDE (120 seasons = 30 years)."""
    _world, orch = ready_world
    result = orch.advance_ride(200)

    assert result["seasons_advanced"] == 120
    assert result["years_advanced"] == pytest.approx(30.0)


# ── Reproducibility ──────────────────────────────────────────────────


@pytest.mark.slow
def test_advance_deterministic(tmp_path):
    """Two worlds with the same seed should produce identical ecology layers."""
    worlds = []
    for name in ("world_a", "world_b"):
        world = World.create(tmp_path / name, seed=42)
        orch = Orchestrator(world, erosion_params=FAST_EROSION)
        orch.create_world()
        orch.advance(5)
        worlds.append(world)

    layers_a = worlds[0].rasters.list_layers("ecology")
    layers_b = worlds[1].rasters.list_layers("ecology")
    assert layers_a == layers_b

    for lyr in layers_a:
        arr_a = worlds[0].rasters.read_layer("ecology", lyr)
        arr_b = worlds[1].rasters.read_layer("ecology", lyr)
        np.testing.assert_array_equal(arr_a, arr_b)


@pytest.mark.slow
def test_advance_deterministic_incremental(tmp_path):
    """advance(2.5) twice vs advance(5) once should yield the same final state."""
    world_a = World.create(tmp_path / "world_a", seed=42)
    orch_a = Orchestrator(world_a, erosion_params=FAST_EROSION)
    orch_a.create_world()
    orch_a.advance(2.5)
    orch_a.advance(2.5)

    world_b = World.create(tmp_path / "world_b", seed=42)
    orch_b = Orchestrator(world_b, erosion_params=FAST_EROSION)
    orch_b.create_world()
    orch_b.advance(5)

    layers_a = world_a.rasters.list_layers("ecology")
    layers_b = world_b.rasters.list_layers("ecology")
    assert layers_a == layers_b

    for lyr in layers_a:
        arr_a = world_a.rasters.read_layer("ecology", lyr)
        arr_b = world_b.rasters.read_layer("ecology", lyr)
        np.testing.assert_array_equal(arr_a, arr_b)


# ── Manual commands ──────────────────────────────────────────────────


def test_introduce_fire(ready_world):
    """introduce_fire() should record a fire event at the given coordinates."""
    world, orch = ready_world
    orch.advance(5)
    orch.introduce_fire(25000.0, 25000.0)

    events = world.events.get_events_in_region(24000, 24000, 26000, 26000)
    fire_events = [evt for evt in events if evt["event_type"] == "fire"]
    assert len(fire_events) >= 1
    assert fire_events[0]["x"] == pytest.approx(25000.0)
    assert fire_events[0]["y"] == pytest.approx(25000.0)


def test_status_returns_info(ready_world):
    """status() should return a dict with seed, tier clocks, and counts."""
    world, orch = ready_world
    orch.advance(5)

    info = orch.status()

    assert "seed" in info
    assert info["seed"] == world.seed
    assert "species_count" in info
    assert info["species_count"] >= 0
    assert "individual_count" in info
    assert info["individual_count"] >= 0


# ── State persistence ────────────────────────────────────────────────


@pytest.mark.slow
def test_world_persists_after_advance(tmp_path):
    """Save, close, and reopen — simulated years should be preserved."""
    path = tmp_path / "world"
    world = World.create(path, seed=42)
    orch = Orchestrator(world, erosion_params=FAST_EROSION)
    orch.create_world()
    orch.advance(5)

    expected_eco_year = world.tier_clocks["ecology"].simulated_year
    expected_eco_ticks = world.tier_clocks["ecology"].tick_number

    world.save(path / "world.json")
    world.close()

    world2 = World.open(path)
    assert world2.tier_clocks["ecology"].simulated_year == expected_eco_year
    assert world2.tier_clocks["ecology"].tick_number == expected_eco_ticks
    world2.close()


@pytest.mark.slow
def test_advance_after_reopen(tmp_path):
    """Advance, save, close, reopen, advance again — should not crash."""
    path = tmp_path / "world"
    world = World.create(path, seed=42)
    orch = Orchestrator(world, erosion_params=FAST_EROSION)
    orch.create_world()
    orch.advance(5)
    eco_ticks_before = world.tier_clocks["ecology"].tick_number

    world.save(path / "world.json")
    world.close()

    world2 = World.open(path)
    orch2 = Orchestrator(world2, erosion_params=FAST_EROSION)
    result = orch2.advance(5)

    assert result["ecology_ticks"] == 20
    assert world2.tier_clocks["ecology"].tick_number == eco_ticks_before + 20
    world2.close()


# ── Seasonal erosion integration ─────────────────────────────────────


def test_advance_updates_terrain(ready_world):
    """After advancing, eroded_heightmap and sediment_depth should be updated."""
    world, orch = ready_world
    result = orch.advance(5)

    # Should have written terrain layers
    ch_layers = world.rasters.list_layers("climate_hydrology")
    assert "eroded_heightmap" in ch_layers
    assert "sediment_depth" in ch_layers
    assert "flow_accumulation" in ch_layers


def test_advance_seasons_direct(ready_world):
    """advance_seasons() should work with explicit season count."""
    world, orch = ready_world
    result = orch.advance_seasons(8)

    assert result["seasons_advanced"] == 8
    assert result["ecology_ticks"] == 8
    assert result["years_advanced"] == pytest.approx(2.0)
