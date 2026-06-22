"""Tests for the time-bounded mechanism-modifier facility.

Two layers are covered:

1. The EcologyTier hook (``set_mechanism_modifiers`` + ``_modifier``): plain
   per-species multipliers on growth / mortality / dispersal / carrying_capacity,
   identity by default so unmodified runs are unchanged.
2. The harness resolver (``resolve_modifiers`` in scripts/test_ecology.py): turns
   declarative modifier specs (windows + targeting) into the resolved
   {species: {mechanism: multiplier}} dict the hook consumes.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest

from bike_sim.tiers.ecology import TIER, EcologyTier
from bike_sim.weather import SeasonalWeather
from bike_sim.world import World

GRID_SIZE = 1000
WINTER, SPRING, SUMMER, FALL = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_weather(season: int, **overrides) -> SeasonalWeather:
    defaults = {
        "temperature": np.full((GRID_SIZE, GRID_SIZE), 15.0, dtype=np.float64),
        "precipitation": np.full((GRID_SIZE, GRID_SIZE), 1600.0, dtype=np.float64),
        "frost_severity": np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64),
        "storm_intensity": 0.0,
        "season": season,
    }
    defaults.update(overrides)
    return SeasonalWeather(**defaults)


def _configure(eco: EcologyTier) -> EcologyTier:
    """Disable the deferred mechanics so only grow/compete/disperse runs."""
    eco.enable_extinction = False
    eco.enable_speciation = False
    eco.enable_individuals = False
    eco.refugium_floor = 0.0
    return eco


def _species_densities(world: World) -> dict[str, np.ndarray]:
    result = {}
    layers = world.rasters.list_layers(TIER)
    for sp in world.events.list_species():
        sid = sp["species_id"]
        layer = f"species_{sid}_density"
        if layer in layers:
            result[sid] = world.rasters.read_layer(TIER, layer).copy()
    return result


def _totals(world: World) -> dict[str, float]:
    return {sid: float(arr.sum()) for sid, arr in _species_densities(world).items()}


def _warm_up(eco: EcologyTier, n_ticks: int) -> None:
    for t in range(n_ticks):
        eco.tick(make_weather(t % 4))


@pytest.fixture
def eco_world(fresh_world):
    """Alias for the conftest fresh_world (geology + climate already ticked)."""
    return fresh_world


@pytest.fixture
def twin_worlds(base_world_path, tmp_path):
    """Two byte-identical pre-ecology worlds (a control + a treatment).

    Both copy the session base world (geology + climate ticked, same seed) so
    they evolve identically until a modifier is introduced on one of them.
    """
    a_dir = tmp_path / "twin_a"
    b_dir = tmp_path / "twin_b"
    shutil.copytree(base_world_path, a_dir)
    shutil.copytree(base_world_path, b_dir)
    return World.open(a_dir), World.open(b_dir)


# ---------------------------------------------------------------------------
# 1. EcologyTier hook
# ---------------------------------------------------------------------------


class TestHookDefaults:
    def test_default_modifiers_empty(self, eco_world):
        eco = EcologyTier(eco_world)
        assert eco.mechanism_modifiers == {}

    def test_modifier_defaults_to_one(self, eco_world):
        eco = EcologyTier(eco_world)
        assert eco._modifier("anything", "growth") == 1.0
        assert eco._modifier("anything", "mortality") == 1.0

    def test_set_and_clear(self, eco_world):
        eco = EcologyTier(eco_world)
        eco.set_mechanism_modifiers({"sp": {"growth": 0.5}})
        assert eco._modifier("sp", "growth") == 0.5
        assert eco._modifier("sp", "mortality") == 1.0  # unlisted → identity
        eco.set_mechanism_modifiers({})
        assert eco._modifier("sp", "growth") == 1.0


class TestHookBehaviour:
    """Treatment vs control: a modifier must move density the predicted way."""

    def _largest_species(self, world: World) -> str:
        totals = _totals(world)
        return max(totals, key=totals.get)

    def test_empty_modifiers_is_identity(self, twin_worlds):
        """Setting {} every tick reproduces an unmodified run bit-for-bit."""
        wa, wb = twin_worlds
        eco_a, eco_b = _configure(EcologyTier(wa)), _configure(EcologyTier(wb))
        _warm_up(eco_a, 6)
        _warm_up(eco_b, 6)
        for t in range(6, 14):
            eco_a.tick(make_weather(t % 4))
            eco_b.set_mechanism_modifiers({})
            eco_b.tick(make_weather(t % 4))
        for sid, arr in _species_densities(wa).items():
            np.testing.assert_array_equal(arr, _species_densities(wb)[sid])

    def test_zero_growth_suppresses_species(self, twin_worlds):
        """growth=0 makes a species decline — it cannot replenish its losses.

        We assert the self-decline invariant rather than comparing to the
        untouched control: a *growing* control can overshoot carrying capacity
        and shed more to competition, so the vs-control inequality is regime-
        dependent. With growth zeroed and mortality > 0 the species can only
        lose, so its total must fall below where it started — the robust
        signature of growth suppression.
        """
        wa, wb = twin_worlds
        eco_a, eco_b = _configure(EcologyTier(wa)), _configure(EcologyTier(wb))
        _warm_up(eco_a, 8)
        _warm_up(eco_b, 8)
        target = self._largest_species(wb)
        before = _totals(wb)[target]
        for t in range(8, 20):
            eco_a.tick(make_weather(t % 4))
            eco_b.set_mechanism_modifiers({target: {"growth": 0.0}})
            eco_b.tick(make_weather(t % 4))
        b_after = _totals(wb)[target]
        assert b_after < before, "growth=0 species should decline from its start"

    def test_high_mortality_suppresses_species(self, twin_worlds):
        wa, wb = twin_worlds
        eco_a, eco_b = _configure(EcologyTier(wa)), _configure(EcologyTier(wb))
        _warm_up(eco_a, 8)
        _warm_up(eco_b, 8)
        target = self._largest_species(wb)
        for t in range(8, 20):
            eco_a.tick(make_weather(t % 4))
            eco_b.set_mechanism_modifiers({target: {"mortality": 4.0}})
            eco_b.tick(make_weather(t % 4))
        assert _totals(wb)[target] < _totals(wa)[target]

    def test_low_carrying_capacity_suppresses_species(self, twin_worlds):
        wa, wb = twin_worlds
        eco_a, eco_b = _configure(EcologyTier(wa)), _configure(EcologyTier(wb))
        _warm_up(eco_a, 8)
        _warm_up(eco_b, 8)
        target = self._largest_species(wb)
        for t in range(8, 20):
            eco_a.tick(make_weather(t % 4))
            eco_b.set_mechanism_modifiers({target: {"carrying_capacity": 0.1}})
            eco_b.tick(make_weather(t % 4))
        assert _totals(wb)[target] < _totals(wa)[target]


class TestAlleeThreshold:
    """The Allee establishment threshold (positive density dependence).

    With ``allee_theta > 0``, colonizing growth in near-empty cells is gated
    by ``density/(density+theta)``, so a sub-threshold sparse population can't
    self-bootstrap — it needs a dense source to recolonize from. Off by default
    (``allee_theta == 0``), so it never changes existing runs.
    """

    def test_off_by_default(self, eco_world):
        assert EcologyTier(eco_world).allee_theta == 0.0

    def test_allee_suppresses_sparse_establishment(self, twin_worlds):
        """A sparse, sub-threshold field grows in the control but not with Allee.

        We isolate colonizing growth: zero every species, then seed only the
        target at a uniform sub-threshold density. With no competitors, the
        control grows logistically everywhere; with Allee on, that growth is
        gated down to a trickle, so the treatment ends well below the control.
        """
        wa, wb = twin_worlds
        eco_a, eco_b = _configure(EcologyTier(wa)), _configure(EcologyTier(wb))
        _warm_up(eco_a, 1)  # one tick just to create ancestors + density layers
        _warm_up(eco_b, 1)
        target = max(_totals(wb), key=_totals(wb).get)
        species_ids = [sp["species_id"] for sp in wb.events.list_species()]

        zero = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
        sparse = np.full((GRID_SIZE, GRID_SIZE), 0.4, dtype=np.float64)
        for w in (wa, wb):
            tick = w.tier_clocks["ecology"].tick_number
            for sid in species_ids:
                arr = sparse.copy() if sid == target else zero.copy()
                w.rasters.write_layer("ecology", f"species_{sid}_density", arr, tick)

        eco_b.allee_theta = 3.0  # Allee ON for the treatment only
        for t in range(1, 7):
            eco_a.tick(make_weather(t % 4))
            eco_b.tick(make_weather(t % 4))

        a_after = _totals(wa)[target]
        b_after = _totals(wb)[target]
        assert b_after < a_after, (
            "Allee should suppress sub-threshold establishment relative to control"
        )


# ---------------------------------------------------------------------------
# 2. Harness resolver
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def harness():
    """Import scripts/test_ecology.py as a module for resolver-level tests."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "test_ecology.py"
    spec = importlib.util.spec_from_file_location("test_ecology_harness", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def ticked_world(eco_world):
    """A world with ancestors created (one tick) so resolver targeting has data."""
    eco = _configure(EcologyTier(eco_world))
    eco.tick(make_weather(SPRING))
    return eco_world


class TestResolver:
    def test_inactive_window_resolves_empty(self, harness, ticked_world):
        mods = [{"mechanism": "growth", "multiplier": 0.0,
                 "start_year": 50, "end_year": 100}]
        assert harness.resolve_modifiers(ticked_world, mods, rel_year=10) == {}

    def test_active_window_world_wide(self, harness, ticked_world):
        mods = [{"mechanism": "mortality", "multiplier": 2.0,
                 "start_year": 0, "end_year": 100}]
        resolved = harness.resolve_modifiers(ticked_world, mods, rel_year=50)
        alive = [s["species_id"] for s in ticked_world.events.list_species()]
        assert set(resolved) == set(alive)
        assert all(r["mortality"] == 2.0 for r in resolved.values())

    def test_species_targeting(self, harness, ticked_world):
        sid = ticked_world.events.list_species()[0]["species_id"]
        mods = [{"mechanism": "growth", "multiplier": 0.0,
                 "start_year": 0, "end_year": 100, "species": sid}]
        resolved = harness.resolve_modifiers(ticked_world, mods, rel_year=10)
        assert list(resolved) == [sid]
        assert resolved[sid]["growth"] == 0.0

    def test_trait_targeting(self, harness, ticked_world):
        mods = [{"mechanism": "growth", "multiplier": 0.5,
                 "start_year": 0, "end_year": 100,
                 "target": {"trait": "max_height", "op": ">", "value": 10}}]
        resolved = harness.resolve_modifiers(ticked_world, mods, rel_year=10)
        for sid in resolved:
            genome = ticked_world.events.get_species(sid)["genome"]
            assert genome["max_height"] > 10

    def test_overlapping_modifiers_multiply(self, harness, ticked_world):
        sid = ticked_world.events.list_species()[0]["species_id"]
        mods = [
            {"mechanism": "growth", "multiplier": 0.5,
             "start_year": 0, "end_year": 100, "species": sid},
            {"mechanism": "growth", "multiplier": 0.5,
             "start_year": 0, "end_year": 100, "species": sid},
        ]
        resolved = harness.resolve_modifiers(ticked_world, mods, rel_year=10)
        assert resolved[sid]["growth"] == pytest.approx(0.25)


class TestPartitionShocks:
    def test_no_at_year_is_immediate(self, harness):
        shocks = [{"type": "fire"}, {"type": "flood"}]
        immediate, timed = harness.partition_shocks(shocks)
        assert immediate == shocks
        assert timed == []

    def test_at_year_zero_is_immediate(self, harness):
        shocks = [{"type": "fire", "at_year": 0}]
        immediate, timed = harness.partition_shocks(shocks)
        assert len(immediate) == 1 and timed == []

    def test_timed_partitioned_and_sorted(self, harness):
        shocks = [
            {"type": "a", "at_year": 200},
            {"type": "b"},
            {"type": "c", "at_year": 50},
        ]
        immediate, timed = harness.partition_shocks(shocks)
        assert [s["type"] for s in immediate] == ["b"]
        assert [s["at_year"] for s in timed] == [50, 200]


class TestScenarioValidation:
    def test_bad_mechanism_rejected(self, harness, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "name: bad\nmodifiers:\n"
            "  - {mechanism: teleport, multiplier: 0.0, start_year: 0, end_year: 1}\n"
        )
        with pytest.raises(ValueError, match="unknown mechanism"):
            harness.load_scenario(bad)

    def test_inverted_window_rejected(self, harness, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "name: bad\nmodifiers:\n"
            "  - {mechanism: growth, multiplier: 0.0, start_year: 100, end_year: 50}\n"
        )
        with pytest.raises(ValueError, match="end_year"):
            harness.load_scenario(bad)

    def test_valid_blight_scenario_loads(self, harness):
        path = Path(__file__).resolve().parents[1] / "scenarios" / "blight.yaml"
        scenario = harness.load_scenario(path)
        assert len(scenario.modifiers) == 1
        assert scenario.modifiers[0]["mechanism"] == "growth"
