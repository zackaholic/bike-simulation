"""Tests for Phase 7 advanced ecology: individuals, speciation, disturbance."""

import re

import numpy as np
import pytest

from bike_sim.tiers.climate_hydrology import ClimateHydrologyTier
from bike_sim.tiers.ecology import EcologyTier
from bike_sim.tiers.geology import GeologyTier
from bike_sim.world import TierId, World


@pytest.fixture
def eco_world(tmp_path):
    """Create a world with geology and climate-hydrology already ticked."""
    world = World.create(tmp_path / "world", seed=42)
    GeologyTier(world).tick()
    ClimateHydrologyTier(world).tick()
    return world


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

WORLD_SIZE = 50_000  # 50 km in meters
GRID_SIZE = 1000
CELL_SIZE = WORLD_SIZE / GRID_SIZE  # 50 m


def _run_ticks(eco_world, n):
    """Run n ecology ticks and return the EcologyTier instance."""
    eco = EcologyTier(eco_world)
    for _ in range(n):
        eco.tick()
    return eco


def _all_individuals(eco_world, radius=50_000):
    """Return all distinguished individuals in the world."""
    center = WORLD_SIZE / 2.0
    return eco_world.events.find_individuals_near(center, center, radius)


def _all_events(eco_world):
    """Return all events in the world bounds."""
    return eco_world.events.get_events_in_region(0, 0, WORLD_SIZE, WORLD_SIZE)


def _initial_species_ids(eco_world):
    """Return the set of ancestor species IDs (those with parent_id=None)."""
    return {sp["species_id"] for sp in eco_world.events.list_species() if sp["parent_id"] is None}


# ===========================================================================
# Distinguished individuals tests
# ===========================================================================


class TestDistinguishedIndividuals:
    def test_individuals_created_after_ticks(self, eco_world):
        """After 5 ecology ticks, EventStore should contain distinguished individuals."""
        _run_ticks(eco_world, 5)
        individuals = _all_individuals(eco_world)
        assert len(individuals) > 0, "No distinguished individuals found after 5 ticks"

    def test_individual_has_valid_species(self, eco_world):
        """Each individual's species_id should match a registered species."""
        _run_ticks(eco_world, 5)
        species_ids = {sp["species_id"] for sp in eco_world.events.list_species()}
        individuals = _all_individuals(eco_world)
        assert len(individuals) > 0, "No individuals to test"
        for ind in individuals:
            assert ind["species_id"] in species_ids, (
                f"Individual {ind['individual_id']} has unknown species {ind['species_id']}"
            )

    def test_individual_positions_in_world_bounds(self, eco_world):
        """All individual x, y should be in [0, 50000]."""
        _run_ticks(eco_world, 5)
        individuals = _all_individuals(eco_world)
        assert len(individuals) > 0, "No individuals to test"
        for ind in individuals:
            assert 0 <= ind["x"] <= WORLD_SIZE, f"Individual x={ind['x']} out of bounds"
            assert 0 <= ind["y"] <= WORLD_SIZE, f"Individual y={ind['y']} out of bounds"

    def test_individuals_at_high_density_locations(self, eco_world):
        """Each individual should be located where its species has nonzero density."""
        _run_ticks(eco_world, 5)
        individuals = _all_individuals(eco_world)
        assert len(individuals) > 0, "No individuals to test"

        for ind in individuals:
            sid = ind["species_id"]
            layer_name = f"species_{sid}_density"
            layers = eco_world.rasters.list_layers(TierId.ECOLOGY)
            if layer_name not in layers:
                continue  # species may have been removed; skip
            density = eco_world.rasters.read_layer(TierId.ECOLOGY, layer_name)
            # Convert position (meters) to grid cell.
            col = min(int(ind["x"] / CELL_SIZE), GRID_SIZE - 1)
            row = min(int(ind["y"] / CELL_SIZE), GRID_SIZE - 1)
            assert density[row, col] > 0.0, (
                f"Individual {ind['individual_id']} at ({row},{col}) "
                f"has zero density for species {sid}"
            )

    def test_more_ticks_more_individuals(self, eco_world):
        """Running more ticks should produce at least as many individuals."""
        _run_ticks(eco_world, 3)
        count_early = len(_all_individuals(eco_world))

        _run_ticks(eco_world, 5)  # 5 additional ticks (8 total)
        count_late = len(_all_individuals(eco_world))

        assert count_late >= count_early, (
            f"Expected more individuals after more ticks: "
            f"{count_early} after 3 ticks vs {count_late} after 8 ticks"
        )


# ===========================================================================
# Speciation tests
# ===========================================================================


class TestSpeciation:
    SPECIATION_TICKS = 25  # enough ticks to trigger speciation

    def test_speciation_occurs_with_many_ticks(self, eco_world):
        """After many ticks, there should be more species than the initial ancestors."""
        _run_ticks(eco_world, self.SPECIATION_TICKS)
        species = eco_world.events.list_species()
        ancestors = [sp for sp in species if sp["parent_id"] is None]
        assert len(species) > len(ancestors), (
            f"Expected speciation to produce new species beyond the "
            f"{len(ancestors)} ancestors, but only found {len(species)} total"
        )

    def test_new_species_has_parent(self, eco_world):
        """Any species beyond the original ancestors should have a non-None parent_id."""
        _run_ticks(eco_world, self.SPECIATION_TICKS)
        species = eco_world.events.list_species()
        ancestors = {sp["species_id"] for sp in species if sp["parent_id"] is None}
        derived = [sp for sp in species if sp["species_id"] not in ancestors]
        assert len(derived) > 0, "No derived species found"
        for sp in derived:
            assert sp["parent_id"] is not None, (
                f"Derived species {sp['species_id']} has parent_id=None"
            )

    def test_new_species_has_density_layer(self, eco_world):
        """New species created by speciation should get their own density layers."""
        _run_ticks(eco_world, self.SPECIATION_TICKS)
        species = eco_world.events.list_species()
        ancestors = {sp["species_id"] for sp in species if sp["parent_id"] is None}
        derived = [sp for sp in species if sp["species_id"] not in ancestors]
        assert len(derived) > 0, "No derived species found"

        layers = eco_world.rasters.list_layers(TierId.ECOLOGY)
        for sp in derived:
            layer_name = f"species_{sp['species_id']}_density"
            assert layer_name in layers, f"Missing density layer for new species: {layer_name}"

    def test_new_species_genome_differs_from_parent(self, eco_world):
        """A child species' genome should differ from its parent's genome."""
        _run_ticks(eco_world, self.SPECIATION_TICKS)
        species = eco_world.events.list_species()
        ancestors = {sp["species_id"] for sp in species if sp["parent_id"] is None}
        derived = [sp for sp in species if sp["species_id"] not in ancestors]
        assert len(derived) > 0, "No derived species found"

        for sp in derived:
            child_info = eco_world.events.get_species(sp["species_id"])
            parent_info = eco_world.events.get_species(sp["parent_id"])
            child_genome = child_info["genome"]
            parent_genome = parent_info["genome"]

            # At least one trait should differ by more than 0.01.
            diffs = [
                abs(child_genome[key] - parent_genome[key])
                for key in parent_genome
                if key in child_genome
            ]
            max_diff = max(diffs) if diffs else 0.0
            assert max_diff > 0.01, (
                f"Child species {sp['species_id']} genome is too similar to "
                f"parent {sp['parent_id']} (max trait diff = {max_diff:.4f})"
            )

    def test_speciation_deterministic(self, tmp_path):
        """Two worlds with same seed should produce identical speciation results."""
        results = []
        for suffix in ("a", "b"):
            w = World.create(tmp_path / f"world_{suffix}", seed=42)
            GeologyTier(w).tick()
            ClimateHydrologyTier(w).tick()
            eco = EcologyTier(w)
            for _ in range(20):
                eco.tick()
            species = w.events.list_species()
            results.append(species)

        ids_a = sorted(sp["species_id"] for sp in results[0])
        ids_b = sorted(sp["species_id"] for sp in results[1])
        assert len(ids_a) == len(ids_b), f"Species count differs: {len(ids_a)} vs {len(ids_b)}"
        assert ids_a == ids_b, "Species IDs differ between identical-seed worlds"


# ===========================================================================
# Disturbance tests
# ===========================================================================


class TestDisturbance:
    DISTURBANCE_TICKS = 12  # enough ticks for stochastic fire/blowdown

    def test_fire_events_recorded(self, eco_world):
        """After enough ticks, there should be at least one fire event."""
        _run_ticks(eco_world, self.DISTURBANCE_TICKS)
        events = _all_events(eco_world)
        fire_events = [ev for ev in events if ev["event_type"] == "fire"]
        assert len(fire_events) >= 1, (
            f"Expected at least one fire event after {self.DISTURBANCE_TICKS} ticks, "
            f"found {len(fire_events)}"
        )

    def test_fire_reduces_density_locally(self, eco_world):
        """Density at a fire location should be lower than the global mean."""
        _run_ticks(eco_world, self.DISTURBANCE_TICKS)
        events = _all_events(eco_world)
        fire_events = [ev for ev in events if ev["event_type"] == "fire"]
        if not fire_events:
            pytest.skip("No fire events occurred in this run")

        fire = fire_events[-1]  # most recent fire
        col = min(int(fire["x"] / CELL_SIZE), GRID_SIZE - 1)
        row = min(int(fire["y"] / CELL_SIZE), GRID_SIZE - 1)

        # Sum all species densities at the fire location.
        layers = eco_world.rasters.list_layers(TierId.ECOLOGY)
        density_layers = [lyr for lyr in layers if re.match(r"species_.*_density$", lyr)]
        local_total = 0.0
        global_total = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
        for name in density_layers:
            arr = eco_world.rasters.read_layer(TierId.ECOLOGY, name)
            local_total += arr[row, col]
            global_total += arr

        global_mean = global_total.mean()
        assert local_total < global_mean, (
            f"Density at fire location ({local_total:.4f}) should be below "
            f"global mean ({global_mean:.4f})"
        )

    def test_disturbance_events_have_location(self, eco_world):
        """All disturbance events should have valid x, y and year > 0."""
        _run_ticks(eco_world, self.DISTURBANCE_TICKS)
        events = _all_events(eco_world)
        disturbance_events = [ev for ev in events if ev["event_type"] in ("fire", "blowdown")]
        assert len(disturbance_events) > 0, "No disturbance events found"
        for ev in disturbance_events:
            assert 0 <= ev["x"] <= WORLD_SIZE, f"Event x={ev['x']} out of bounds"
            assert 0 <= ev["y"] <= WORLD_SIZE, f"Event y={ev['y']} out of bounds"
            assert ev["year"] > 0, f"Event year={ev['year']} should be > 0"

    def test_seed_bank_responds_to_disturbance(self, eco_world):
        """After disturbance, seed bank total should be nonzero in burned areas."""
        _run_ticks(eco_world, self.DISTURBANCE_TICKS)
        events = _all_events(eco_world)
        fire_events = [ev for ev in events if ev["event_type"] == "fire"]
        if not fire_events:
            pytest.skip("No fire events occurred in this run")

        seed_bank = eco_world.rasters.read_layer(TierId.ECOLOGY, "seed_bank_total")

        # Check that at least one fire location has nonzero seed bank.
        found_nonzero = False
        for fire in fire_events:
            col = min(int(fire["x"] / CELL_SIZE), GRID_SIZE - 1)
            row = min(int(fire["y"] / CELL_SIZE), GRID_SIZE - 1)
            if seed_bank[row, col] > 0:
                found_nonzero = True
                break

        assert found_nonzero, (
            "Seed bank is zero at all fire locations; "
            "expected germination activity post-disturbance"
        )

    def test_blowdown_events_exist(self, eco_world):
        """After enough ticks, at least one blowdown or disturbance event should exist."""
        _run_ticks(eco_world, 15)
        events = _all_events(eco_world)
        disturbance_events = [ev for ev in events if ev["event_type"] in ("fire", "blowdown")]
        assert len(disturbance_events) >= 1, (
            "Expected at least one disturbance event (fire or blowdown) after 15 ticks, found none"
        )


# ===========================================================================
# Integration test
# ===========================================================================


def test_full_stack_many_ticks_no_crash(tmp_path):
    """Run full stack for 20 ecology ticks: no crashes, valid invariants."""
    world = World.create(tmp_path / "world", seed=42)
    GeologyTier(world).tick()
    ClimateHydrologyTier(world).tick()
    eco = EcologyTier(world)
    for _ in range(20):
        eco.tick()

    # All densities non-negative.
    layers = world.rasters.list_layers(TierId.ECOLOGY)
    density_layers = [lyr for lyr in layers if re.match(r"species_.*_density$", lyr)]
    for name in density_layers:
        arr = world.rasters.read_layer(TierId.ECOLOGY, name)
        assert np.all(arr >= 0), f"{name} has negative densities after 20 ticks"

    # At least the 6 ancestor species.
    species = world.events.list_species()
    assert len(species) >= 6, f"Expected >= 6 species, got {len(species)}"

    # At least some distinguished individuals exist.
    center = WORLD_SIZE / 2.0
    individuals = world.events.find_individuals_near(center, center, WORLD_SIZE)
    assert len(individuals) > 0, "No distinguished individuals after 20 ticks"
