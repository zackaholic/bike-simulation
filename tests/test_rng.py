"""Tests for the seeded RNG infrastructure."""

import numpy as np
from numpy.testing import assert_array_equal

from bike_sim.rng import _stable_hash, create_rng


def test_determinism():
    """Same inputs always produce the same output sequence."""
    rng1 = create_rng(world_seed=42, tier_id="ecology", pass_id="dispersal", tick_number=0)
    rng2 = create_rng(world_seed=42, tier_id="ecology", pass_id="dispersal", tick_number=0)
    vals1 = rng1.random(1000)
    vals2 = rng2.random(1000)
    assert_array_equal(vals1, vals2)


def test_different_seeds_diverge():
    """Different world seeds produce different sequences."""
    rng1 = create_rng(world_seed=1, tier_id="ecology", pass_id="dispersal", tick_number=0)
    rng2 = create_rng(world_seed=2, tier_id="ecology", pass_id="dispersal", tick_number=0)
    vals1 = rng1.random(100)
    vals2 = rng2.random(100)
    assert not np.array_equal(vals1, vals2)


def test_different_tiers_diverge():
    """Different tier IDs produce independent streams."""
    rng1 = create_rng(world_seed=42, tier_id="geology", pass_id="erosion", tick_number=0)
    rng2 = create_rng(world_seed=42, tier_id="ecology", pass_id="erosion", tick_number=0)
    vals1 = rng1.random(100)
    vals2 = rng2.random(100)
    assert not np.array_equal(vals1, vals2)


def test_different_passes_diverge():
    """Different pass IDs within the same tier produce independent streams."""
    rng1 = create_rng(world_seed=42, tier_id="ecology", pass_id="fire", tick_number=0)
    rng2 = create_rng(world_seed=42, tier_id="ecology", pass_id="dispersal", tick_number=0)
    vals1 = rng1.random(100)
    vals2 = rng2.random(100)
    assert not np.array_equal(vals1, vals2)


def test_different_ticks_diverge():
    """Different tick numbers produce independent streams."""
    rng1 = create_rng(world_seed=42, tier_id="ecology", pass_id="dispersal", tick_number=0)
    rng2 = create_rng(world_seed=42, tier_id="ecology", pass_id="dispersal", tick_number=1)
    vals1 = rng1.random(100)
    vals2 = rng2.random(100)
    assert not np.array_equal(vals1, vals2)


def test_stable_hash_consistency():
    """_stable_hash returns the same value across calls (no PYTHONHASHSEED dependence)."""
    h1 = _stable_hash("ecology")
    h2 = _stable_hash("ecology")
    assert h1 == h2
    # And different strings differ
    assert _stable_hash("geology") != _stable_hash("ecology")
