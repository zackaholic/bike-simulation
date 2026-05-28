"""Tests for the GeologyTier — base terrain generation from seed."""

import numpy as np

from bike_sim.tiers.geology import GeologyTier
from bike_sim.world import TierId, World


def _make_world(tmp_path, seed=42):
    """Helper: create a fresh World in a temporary directory."""
    return World.create(tmp_path / "world", seed=seed)


# ---------- Layer existence ----------


def test_first_tick_produces_layers(tmp_path):
    """After one geology tick, heightmap/bedrock_type/soil_parent exist."""
    world = _make_world(tmp_path)
    geo = GeologyTier(world)
    geo.tick()

    layers = world.rasters.list_layers(TierId.GEOLOGY)
    assert "heightmap" in layers
    assert "bedrock_type" in layers
    assert "soil_parent" in layers


# ---------- Shape and dtype ----------


def test_heightmap_shape_and_dtype(tmp_path):
    """Heightmap is (1000, 1000) float64."""
    world = _make_world(tmp_path)
    GeologyTier(world).tick()

    hm = world.rasters.read_layer(TierId.GEOLOGY, "heightmap")
    assert hm.shape == (1000, 1000)
    assert hm.dtype == np.float64


def test_bedrock_shape_and_dtype(tmp_path):
    """Bedrock type is (1000, 1000) int32."""
    world = _make_world(tmp_path)
    GeologyTier(world).tick()

    br = world.rasters.read_layer(TierId.GEOLOGY, "bedrock_type")
    assert br.shape == (1000, 1000)
    assert br.dtype == np.int32


def test_soil_parent_shape_and_dtype(tmp_path):
    """Soil parent material is (1000, 1000) int32."""
    world = _make_world(tmp_path)
    GeologyTier(world).tick()

    sp = world.rasters.read_layer(TierId.GEOLOGY, "soil_parent")
    assert sp.shape == (1000, 1000)
    assert sp.dtype == np.int32


# ---------- Value ranges ----------


def test_heightmap_range(tmp_path):
    """Heightmap values are finite and in a reasonable range (0–5000 m)."""
    world = _make_world(tmp_path)
    GeologyTier(world).tick()

    hm = world.rasters.read_layer(TierId.GEOLOGY, "heightmap")
    assert np.all(np.isfinite(hm))
    assert hm.min() >= 0.0
    assert hm.max() <= 5000.0


def test_bedrock_types_valid(tmp_path):
    """All bedrock values are in the range [0, 7] (at most 8 rock types)."""
    world = _make_world(tmp_path)
    GeologyTier(world).tick()

    br = world.rasters.read_layer(TierId.GEOLOGY, "bedrock_type")
    assert br.min() >= 0
    assert br.max() <= 7


# ---------- Reproducibility ----------


def test_deterministic_from_seed(tmp_path):
    """Two worlds with the same seed produce bit-identical heightmaps."""
    w1 = World.create(tmp_path / "world_a", seed=12345)
    w2 = World.create(tmp_path / "world_b", seed=12345)

    GeologyTier(w1).tick()
    GeologyTier(w2).tick()

    hm1 = w1.rasters.read_layer(TierId.GEOLOGY, "heightmap")
    hm2 = w2.rasters.read_layer(TierId.GEOLOGY, "heightmap")
    np.testing.assert_array_equal(hm1, hm2)

    br1 = w1.rasters.read_layer(TierId.GEOLOGY, "bedrock_type")
    br2 = w2.rasters.read_layer(TierId.GEOLOGY, "bedrock_type")
    np.testing.assert_array_equal(br1, br2)

    sp1 = w1.rasters.read_layer(TierId.GEOLOGY, "soil_parent")
    sp2 = w2.rasters.read_layer(TierId.GEOLOGY, "soil_parent")
    np.testing.assert_array_equal(sp1, sp2)


def test_different_seeds_differ(tmp_path):
    """Different seeds produce different heightmaps."""
    w1 = World.create(tmp_path / "world_a", seed=1)
    w2 = World.create(tmp_path / "world_b", seed=2)

    GeologyTier(w1).tick()
    GeologyTier(w2).tick()

    hm1 = w1.rasters.read_layer(TierId.GEOLOGY, "heightmap")
    hm2 = w2.rasters.read_layer(TierId.GEOLOGY, "heightmap")
    assert not np.array_equal(hm1, hm2)


# ---------- Tier clock ----------


def test_tier_clock_advances(tmp_path):
    """After one tick the geology clock reads tick 1 and simulated_year > 0."""
    world = _make_world(tmp_path)
    GeologyTier(world).tick()

    clock = world.tier_clocks[TierId.GEOLOGY]
    assert clock.tick_number == 1
    assert clock.simulated_year > 0.0


# ---------- Second tick behavior (v1 no-op) ----------


def test_second_tick_is_noop(tmp_path):
    """Layers are unchanged after a second tick, but tick_number advances."""
    world = _make_world(tmp_path)
    geo = GeologyTier(world)

    geo.tick()
    hm_after_1 = world.rasters.read_layer(TierId.GEOLOGY, "heightmap").copy()
    br_after_1 = world.rasters.read_layer(TierId.GEOLOGY, "bedrock_type").copy()
    sp_after_1 = world.rasters.read_layer(TierId.GEOLOGY, "soil_parent").copy()

    geo.tick()
    hm_after_2 = world.rasters.read_layer(TierId.GEOLOGY, "heightmap")
    br_after_2 = world.rasters.read_layer(TierId.GEOLOGY, "bedrock_type")
    sp_after_2 = world.rasters.read_layer(TierId.GEOLOGY, "soil_parent")

    np.testing.assert_array_equal(hm_after_1, hm_after_2)
    np.testing.assert_array_equal(br_after_1, br_after_2)
    np.testing.assert_array_equal(sp_after_1, sp_after_2)

    assert world.tier_clocks[TierId.GEOLOGY].tick_number == 2


# ---------- Cross-layer invariant ----------


def test_soil_derived_from_bedrock(tmp_path):
    """Cells with identical bedrock type should map to the same soil parent."""
    world = _make_world(tmp_path)
    GeologyTier(world).tick()

    br = world.rasters.read_layer(TierId.GEOLOGY, "bedrock_type")
    sp = world.rasters.read_layer(TierId.GEOLOGY, "soil_parent")

    # Build mapping: for each bedrock type, collect the set of soil types seen.
    bedrock_to_soil = {}
    for rock_type in np.unique(br):
        soil_values = np.unique(sp[br == rock_type])
        bedrock_to_soil[int(rock_type)] = soil_values

    # Each bedrock type should deterministically produce exactly one soil type.
    for rock_type, soils in bedrock_to_soil.items():
        assert len(soils) == 1, f"Bedrock type {rock_type} maps to multiple soil types: {soils}"
