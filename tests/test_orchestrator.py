"""Tests for the Orchestrator — scheduling ticks across all three simulation tiers."""

import numpy as np
import pytest

from bike_sim.orchestrator import Orchestrator
from bike_sim.world import World


@pytest.fixture
def orch(tmp_path):
    """Create a fresh world with an Orchestrator, nothing ticked yet."""
    world = World.create(tmp_path / "world", seed=42)
    return Orchestrator(world)


@pytest.fixture
def ready_world(tmp_path):
    """Create a world and run create_world() so geology+climate are ready."""
    world = World.create(tmp_path / "world", seed=42)
    orch = Orchestrator(world)
    orch.create_world()
    return world, orch


# ── Core orchestration ────────────────────────────────────────────────


def test_create_world_ticks_geology_and_climate(orch):
    """After create_world(), geology=1 tick, climate_hydrology=1 tick, ecology=0."""
    orch.create_world()
    world = orch._world if hasattr(orch, "_world") else orch.world

    assert world.tier_clocks["geology"].tick_number == 1
    assert world.tier_clocks["climate_hydrology"].tick_number == 1
    assert world.tier_clocks["ecology"].tick_number == 0


def test_create_world_produces_heightmap(orch):
    """After create_world(), the geology heightmap layer should exist."""
    orch.create_world()
    world = orch._world if hasattr(orch, "_world") else orch.world

    assert "heightmap" in world.rasters.list_layers("geology")


def test_advance_50_years(ready_world):
    """advance(50) should tick ecology 10 times (50/5), no extra climate tick."""
    world, orch = ready_world

    result = orch.advance(50)

    assert result["ecology_ticks"] == 10
    assert result["climate_hydrology_ticks"] == 0
    assert result["geology_ticks"] == 0
    # Ecology clock should reflect 10 ticks.
    assert world.tier_clocks["ecology"].tick_number == 10


def test_advance_triggers_climate_tick(ready_world):
    """Advancing past the 1000-year threshold should trigger a climate-hydrology tick."""
    world, orch = ready_world

    # Fast-forward ecology clock close to the climate threshold (1000 years)
    # so we don't need to run 200 slow ecology ticks in the test.
    world.tier_clocks["ecology"].simulated_year = 995.0
    world.tier_clocks["ecology"].tick_number = 199

    # A small advance should push us past 1000 and trigger a climate tick.
    result = orch.advance(10)

    assert result["climate_hydrology_ticks"] >= 1
    # Climate clock was at 1 after create_world; should now be >= 2.
    assert world.tier_clocks["climate_hydrology"].tick_number >= 2


def test_advance_small_increments(tmp_path):
    """Five advance(25) calls should produce the same state as one advance(125)."""
    # World A: incremental advances.
    world_a = World.create(tmp_path / "world_a", seed=42)
    orch_a = Orchestrator(world_a)
    orch_a.create_world()
    for _ in range(5):
        orch_a.advance(25)

    # World B: single large advance.
    world_b = World.create(tmp_path / "world_b", seed=42)
    orch_b = Orchestrator(world_b)
    orch_b.create_world()
    orch_b.advance(125)

    # Both should have the same ecology tick count: 125 / 5 = 25.
    assert world_a.tier_clocks["ecology"].tick_number == 25
    assert world_b.tier_clocks["ecology"].tick_number == 25
    # Same simulated year.
    eco_year_a = world_a.tier_clocks["ecology"].simulated_year
    eco_year_b = world_b.tier_clocks["ecology"].simulated_year
    assert eco_year_a == eco_year_b


def test_advance_returns_summary(ready_world):
    """advance() return dict should contain the four required keys."""
    _world, orch = ready_world
    result = orch.advance(50)

    assert "years_advanced" in result
    assert "ecology_ticks" in result
    assert "climate_hydrology_ticks" in result
    assert "geology_ticks" in result
    assert result["years_advanced"] == pytest.approx(50.0)


def test_advance_ride(ready_world):
    """advance_ride(30) should advance ~30 years → 6 ecology ticks."""
    _world, orch = ready_world
    result = orch.advance_ride(30)

    assert result["ecology_ticks"] == 6
    assert result["years_advanced"] == pytest.approx(30.0)


def test_advance_ride_capped(ready_world):
    """advance_ride(120) should cap at 50 years, not 120."""
    _world, orch = ready_world
    result = orch.advance_ride(120)

    assert result["years_advanced"] == pytest.approx(50.0)
    # 50 years / 5 years per tick = 10 ecology ticks.
    assert result["ecology_ticks"] == 10


# ── Reproducibility ──────────────────────────────────────────────────


def test_advance_deterministic(tmp_path):
    """Two worlds with the same seed should produce identical ecology layers."""
    worlds = []
    for name in ("world_a", "world_b"):
        world = World.create(tmp_path / name, seed=42)
        orch = Orchestrator(world)
        orch.create_world()
        orch.advance(50)
        worlds.append(world)

    layers_a = worlds[0].rasters.list_layers("ecology")
    layers_b = worlds[1].rasters.list_layers("ecology")
    assert layers_a == layers_b

    for lyr in layers_a:
        arr_a = worlds[0].rasters.read_layer("ecology", lyr)
        arr_b = worlds[1].rasters.read_layer("ecology", lyr)
        np.testing.assert_array_equal(arr_a, arr_b)


def test_advance_deterministic_incremental(tmp_path):
    """advance(25) twice vs advance(50) once should yield the same final state."""
    world_a = World.create(tmp_path / "world_a", seed=42)
    orch_a = Orchestrator(world_a)
    orch_a.create_world()
    orch_a.advance(25)
    orch_a.advance(25)

    world_b = World.create(tmp_path / "world_b", seed=42)
    orch_b = Orchestrator(world_b)
    orch_b.create_world()
    orch_b.advance(50)

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
    orch.advance(25)
    orch.introduce_fire(25000.0, 25000.0)

    events = world.events.get_events_in_region(24000, 24000, 26000, 26000)
    fire_events = [evt for evt in events if evt["event_type"] == "fire"]
    assert len(fire_events) >= 1
    assert fire_events[0]["x"] == pytest.approx(25000.0)
    assert fire_events[0]["y"] == pytest.approx(25000.0)


def test_status_returns_info(ready_world):
    """status() should return a dict with seed, tier clocks, and counts."""
    world, orch = ready_world
    orch.advance(25)

    info = orch.status()

    assert "seed" in info
    assert info["seed"] == world.seed
    assert "species_count" in info
    assert info["species_count"] >= 0
    assert "individual_count" in info
    assert info["individual_count"] >= 0


# ── State persistence ────────────────────────────────────────────────


def test_world_persists_after_advance(tmp_path):
    """Save, close, and reopen — simulated years should be preserved."""
    path = tmp_path / "world"
    world = World.create(path, seed=42)
    orch = Orchestrator(world)
    orch.create_world()
    orch.advance(50)

    expected_eco_year = world.tier_clocks["ecology"].simulated_year
    expected_eco_ticks = world.tier_clocks["ecology"].tick_number

    world.save(path / "world.json")
    world.close()

    world2 = World.open(path)
    assert world2.tier_clocks["ecology"].simulated_year == expected_eco_year
    assert world2.tier_clocks["ecology"].tick_number == expected_eco_ticks
    world2.close()


def test_advance_after_reopen(tmp_path):
    """Advance, save, close, reopen, advance again — should not crash."""
    path = tmp_path / "world"
    world = World.create(path, seed=42)
    orch = Orchestrator(world)
    orch.create_world()
    orch.advance(50)
    eco_ticks_before = world.tier_clocks["ecology"].tick_number

    world.save(path / "world.json")
    world.close()

    world2 = World.open(path)
    orch2 = Orchestrator(world2)
    result = orch2.advance(50)

    assert result["ecology_ticks"] == 10
    assert world2.tier_clocks["ecology"].tick_number == eco_ticks_before + 10
    world2.close()
