"""Tests for the World data model."""

from bike_sim.world import TierId, World


def test_world_defaults(tmp_path):
    """A fresh World has clocks at zero for all three tiers."""
    w = World(seed=42)
    assert w.seed == 42
    assert w.simulated_year == 0.0
    assert set(w.tier_clocks.keys()) == {t.value for t in TierId}
    for clock in w.tier_clocks.values():
        assert clock.tick_number == 0
        assert clock.simulated_year == 0.0


def test_world_json_round_trip(tmp_path):
    """Save and load produce identical World state."""
    w = World(seed=12345)
    w.simulated_year = 1_000_000.0
    w.tier_clocks["geology"].tick_number = 10
    w.tier_clocks["geology"].simulated_year = 1_000_000.0
    w.tier_clocks["ecology"].tick_number = 500
    w.tier_clocks["ecology"].simulated_year = 999_950.0

    path = tmp_path / "world.json"
    w.save(path)
    loaded = World.load(path)

    assert loaded.seed == w.seed
    assert loaded.simulated_year == w.simulated_year
    assert loaded.version == w.version
    for tier in TierId:
        orig = w.tier_clocks[tier.value]
        back = loaded.tier_clocks[tier.value]
        assert back.tick_number == orig.tick_number
        assert back.simulated_year == orig.simulated_year


def test_world_to_dict_from_dict():
    """Round-trip through dict representation."""
    w = World(seed=99, simulated_year=500.0)
    w.tier_clocks["ecology"].tick_number = 7
    d = w.to_dict()
    w2 = World.from_dict(d)
    assert w2.seed == 99
    assert w2.tier_clocks["ecology"].tick_number == 7
