"""Tests for genome: 6 core traits after ecology tick refactor.

The minimum viable genome has 6 traits, each mapping to exactly one mechanism:
  - drought_tolerance, frost_tolerance, growth_rate (bounded [0,1])
  - max_height, lifespan (positive, unbounded)
  - dispersal_range (integer, [1,6])

These tests verify trait completeness, value ranges, ancestor template values,
speciation drift integrity, and determinism.
"""

import numpy as np
import pytest

from bike_sim.tiers.climate_hydrology import ClimateHydrologyTier
from bike_sim.tiers.ecology import _CORE_TRAITS, EcologyTier
from bike_sim.tiers.erosion import ErosionParams
from bike_sim.tiers.geology import GeologyTier
from bike_sim.weather import WeatherSystem
from bike_sim.world import World

FAST_EROSION = ErosionParams(num_particles=100, max_lifetime=30)

# The full set of expected genome traits.
ALL_TRAITS = set(_CORE_TRAITS)

# Functional traits that must be bounded to [0, 1].
BOUNDED_FUNCTIONAL = {
    "drought_tolerance",
    "frost_tolerance",
    "growth_rate",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def eco_world(tmp_path_factory):
    """Create a world with geology + climate + ecology ticked once.

    Module-scoped for speed -- tests must not mutate the world.
    """
    tmp = tmp_path_factory.mktemp("genome")
    w = World.create(tmp / "world", seed=42)
    GeologyTier(w).tick()
    ClimateHydrologyTier(w, erosion_params=FAST_EROSION).tick()
    heightmap = w.rasters.read_layer("geology", "heightmap")
    ws = WeatherSystem(w.seed, heightmap)
    eco = EcologyTier(w)
    eco.tick(ws.generate(0.0, 1))  # spring tick 0
    return w


@pytest.fixture(scope="module")
def species_genomes(eco_world):
    """Map of species_id -> genome dict from a ticked world."""
    species = eco_world.events.list_species()
    return {
        sp["species_id"]: eco_world.events.get_species(sp["species_id"])["genome"]
        for sp in species
    }


@pytest.fixture(scope="module")
def ancestor_genomes(eco_world):
    """Map of species_id -> genome dict for ancestor species only."""
    species = eco_world.events.list_species()
    result = {}
    for sp in species:
        if sp["parent_id"] is None:
            sid = sp["species_id"]
            result[sid] = eco_world.events.get_species(sid)["genome"]
    return result


@pytest.fixture(scope="module")
def multi_tick_world(tmp_path_factory):
    """A world advanced 200+ ecology ticks to trigger speciation."""
    tmp = tmp_path_factory.mktemp("genome_multi")
    w = World.create(tmp / "world", seed=42)
    GeologyTier(w).tick()
    ClimateHydrologyTier(w, erosion_params=FAST_EROSION).tick()
    heightmap = w.rasters.read_layer("geology", "heightmap")
    ws = WeatherSystem(w.seed, heightmap)
    eco = EcologyTier(w)
    # Run 201 ticks to hit speciation check at tick 200
    for i in range(201):
        year = i * 0.25
        season = i % 4
        eco.tick(ws.generate(year, season))
    return w


# ---------------------------------------------------------------------------
# 1. Genome completeness
# ---------------------------------------------------------------------------


class TestGenomeCompleteness:
    def test_every_genome_has_6_traits(self, species_genomes):
        """Every species genome must have exactly 6 traits."""
        for sid, genome in species_genomes.items():
            assert len(genome) == 6, (
                f"Species {sid} has {len(genome)} traits, expected 6. "
                f"Keys: {sorted(genome.keys())}"
            )

    def test_all_expected_trait_names_present(self, species_genomes):
        """Every genome must contain all expected trait names."""
        for sid, genome in species_genomes.items():
            missing = ALL_TRAITS - genome.keys()
            extra = genome.keys() - ALL_TRAITS
            assert not missing, f"Species {sid} missing traits: {missing}"
            assert not extra, f"Species {sid} has unexpected traits: {extra}"


# ---------------------------------------------------------------------------
# 2. Trait ranges
# ---------------------------------------------------------------------------


class TestTraitRanges:
    def test_bounded_functional_in_unit_interval(self, species_genomes):
        """Bounded functional traits must be in [0, 1]."""
        for sid, genome in species_genomes.items():
            for trait in BOUNDED_FUNCTIONAL:
                val = genome[trait]
                assert 0.0 <= val <= 1.0, (
                    f"Species {sid}: {trait}={val} outside [0, 1]"
                )

    def test_dispersal_range_is_int_in_range(self, species_genomes):
        """dispersal_range must be an integer in [1, 6]."""
        for sid, genome in species_genomes.items():
            val = genome["dispersal_range"]
            assert isinstance(val, int), (
                f"Species {sid}: dispersal_range={val} is {type(val).__name__}, expected int"
            )
            assert 1 <= val <= 6, (
                f"Species {sid}: dispersal_range={val} outside [1, 6]"
            )

    def test_max_height_positive(self, species_genomes):
        """max_height must be positive."""
        for sid, genome in species_genomes.items():
            assert genome["max_height"] > 0, (
                f"Species {sid}: max_height={genome['max_height']} is not positive"
            )

    def test_lifespan_positive(self, species_genomes):
        """lifespan must be positive."""
        for sid, genome in species_genomes.items():
            assert genome["lifespan"] > 0, (
                f"Species {sid}: lifespan={genome['lifespan']} is not positive"
            )


# ---------------------------------------------------------------------------
# 3. Ancestor trait values
# ---------------------------------------------------------------------------


class TestAncestorTraitValues:
    def test_ancestors_created(self, ancestor_genomes):
        """All ancestor species should be created."""
        from bike_sim.tiers.ecology import _ANCESTOR_TEMPLATES
        expected = len(_ANCESTOR_TEMPLATES)
        assert len(ancestor_genomes) == expected, (
            f"Expected {expected} ancestors, got {len(ancestor_genomes)}"
        )

    def test_ancestor_traits_close_to_templates(self, ancestor_genomes):
        """Traits should be within perturbation range of templates.

        The perturbation range is [-0.05, 0.05] for each trait, so actual
        values should be within 0.05 of the template value.

        Exception: ``drought_tolerance`` is intentionally NOT template-bound.
        It is remapped at creation onto the world's achievable climate manifold
        (warmth → dryness) so no archetype strands in an unreachable corner, so
        it can legitimately differ a lot from the template's starting intent.
        """
        from bike_sim.tiers.ecology import _ANCESTOR_TEMPLATES

        for template_name, template_genome in _ANCESTOR_TEMPLATES:
            matches = [
                (sid, g) for sid, g in ancestor_genomes.items()
                if template_name in sid
            ]
            assert len(matches) == 1, (
                f"Expected 1 ancestor matching '{template_name}', found {len(matches)}"
            )
            sid, genome = matches[0]
            for trait, expected in template_genome.items():
                if trait == "drought_tolerance":
                    continue  # world-derived; see docstring
                actual = genome[trait]
                if trait == "dispersal_range":
                    # Integer trait: allow +-1 from rounding
                    assert abs(actual - expected) <= 1, (
                        f"Ancestor {sid}: {trait}={actual}, expected within 1 of {expected}"
                    )
                else:
                    assert abs(actual - expected) <= 0.05 + 1e-9, (
                        f"Ancestor {sid}: {trait}={actual}, expected within 0.05 of {expected}"
                    )

    def test_no_ancestor_stranded_off_manifold(self, eco_world):
        """Every ancestor must reach strong suitability somewhere in the cycle.

        After moisture-niche redistribution, no archetype should be stuck in an
        unreachable climate corner (the old warm-wet stranding). Because the
        climate drifts aperiodically, a wet specialist is only viable at a wet
        phase, so we sample the realized climate across a wide window (many
        years × seasons) and require each species' peak suitability over that
        whole envelope to clear a healthy bar.
        """
        from bike_sim.tiers.ecology import (
            PRECIP_REF_MAX,
            PRECIP_REF_MIN,
            TEMP_REF_MAX,
            TEMP_REF_MIN,
            _gaussian_match,
        )

        heightmap = eco_world.rasters.read_layer("geology", "heightmap")
        ws = WeatherSystem(eco_world.seed, heightmap)

        # Build a cloud of realized (temp_norm, drought_stress) points across a
        # wide climate window (captures wet peaks and dry troughs).
        tn_cloud, ds_cloud = [], []
        for year in np.linspace(0.0, 600.0, 10):
            for season in range(4):
                w = ws.generate(float(year), season)
                t = w.temperature[::20, ::20].ravel()
                p = w.precipitation[::20, ::20].ravel()
                t_span = TEMP_REF_MAX - TEMP_REF_MIN
                p_span = PRECIP_REF_MAX - PRECIP_REF_MIN
                tn_cloud.append(np.clip((t - TEMP_REF_MIN) / t_span, 0, 1))
                ds_cloud.append(1.0 - np.clip((p - PRECIP_REF_MIN) / p_span, 0, 1))
        tn = np.concatenate(tn_cloud)
        ds = np.concatenate(ds_cloud)

        weak = []
        for sid, genome in {
            sp["species_id"]: eco_world.events.get_species(sp["species_id"])["genome"]
            for sp in eco_world.events.list_species()
        }.items():
            warmth_pref = 1.0 - genome["frost_tolerance"]
            suit = (
                _gaussian_match(tn, warmth_pref, sigma=0.25)
                * _gaussian_match(ds, genome["drought_tolerance"], sigma=0.25)
            )
            peak = float(suit.max())
            if peak < 0.4:
                weak.append((sid, round(peak, 3)))
        assert not weak, f"Ancestors stranded off the climate manifold: {weak}"

    def test_valley_hardwood_is_tall(self, ancestor_genomes):
        """valley_hardwood should be a tall tree (max_height > 25)."""
        tree = [g for sid, g in ancestor_genomes.items() if "valley_hardwood" in sid]
        assert len(tree) == 1
        assert tree[0]["max_height"] > 25, (
            f"valley_hardwood max_height={tree[0]['max_height']}, expected > 25"
        )

    def test_alpine_cushion_is_frost_tolerant(self, ancestor_genomes):
        """alpine_cushion should have frost_tolerance > 0.7."""
        cushion = [g for sid, g in ancestor_genomes.items() if "alpine_cushion" in sid]
        assert len(cushion) == 1
        assert cushion[0]["frost_tolerance"] > 0.7, (
            f"alpine_cushion frost_tolerance={cushion[0]['frost_tolerance']}, expected > 0.7"
        )


# ---------------------------------------------------------------------------
# 4. Speciation drift
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSpeciationDrift:
    def test_speciated_species_have_6_traits(self, multi_tick_world):
        """Any species produced by speciation must have all 6 traits."""
        species = multi_tick_world.events.list_species()
        children = [sp for sp in species if sp["parent_id"] is not None]
        if len(children) == 0:
            pytest.skip("No speciation occurred -- stochastic, cannot test drift")
        for sp in children:
            sid = sp["species_id"]
            genome = multi_tick_world.events.get_species(sid)["genome"]
            missing = ALL_TRAITS - genome.keys()
            assert not missing, (
                f"Speciated species {sid} missing traits: {missing}"
            )
            assert len(genome) == 6, (
                f"Speciated species {sid} has {len(genome)} traits, expected 6"
            )

    def test_dispersal_range_stays_int_after_speciation(self, multi_tick_world):
        """dispersal_range must remain an integer in [1, 6] after speciation."""
        species = multi_tick_world.events.list_species()
        children = [sp for sp in species if sp["parent_id"] is not None]
        if not children:
            pytest.skip("No speciation occurred -- stochastic")
        for sp in children:
            sid = sp["species_id"]
            genome = multi_tick_world.events.get_species(sid)["genome"]
            val = genome["dispersal_range"]
            assert isinstance(val, int), (
                f"Speciated {sid}: dispersal_range={val} is {type(val).__name__}, expected int"
            )
            assert 1 <= val <= 6, (
                f"Speciated {sid}: dispersal_range={val} outside [1, 6]"
            )

    def test_bounded_traits_stay_in_bounds_after_speciation(self, multi_tick_world):
        """All bounded traits must stay within their valid ranges after drift."""
        species = multi_tick_world.events.list_species()
        children = [sp for sp in species if sp["parent_id"] is not None]
        if not children:
            pytest.skip("No speciation occurred -- stochastic")
        for sp in children:
            sid = sp["species_id"]
            genome = multi_tick_world.events.get_species(sid)["genome"]
            for trait in BOUNDED_FUNCTIONAL:
                val = genome[trait]
                assert 0.0 <= val <= 1.0, (
                    f"Speciated {sid}: {trait}={val} outside [0, 1]"
                )
            assert genome["max_height"] > 0
            assert genome["lifespan"] > 0


# ---------------------------------------------------------------------------
# 5. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_genomes(self, tmp_path):
        """Two worlds with the same seed must produce identical genomes."""
        genomes = []
        for i in range(2):
            w = World.create(tmp_path / f"det_{i}", seed=42)
            GeologyTier(w).tick()
            ClimateHydrologyTier(w, erosion_params=FAST_EROSION).tick()
            heightmap = w.rasters.read_layer("geology", "heightmap")
            ws = WeatherSystem(w.seed, heightmap)
            EcologyTier(w).tick(ws.generate(0.0, 1))
            species = w.events.list_species()
            g = {
                sp["species_id"]: w.events.get_species(sp["species_id"])["genome"]
                for sp in species
            }
            genomes.append(g)

        assert genomes[0].keys() == genomes[1].keys(), "Species IDs differ between runs"
        for sid in genomes[0]:
            for trait in genomes[0][sid]:
                assert genomes[0][sid][trait] == genomes[1][sid][trait], (
                    f"Determinism failure: species {sid}, trait {trait}: "
                    f"{genomes[0][sid][trait]} != {genomes[1][sid][trait]}"
                )

    def test_different_seed_different_genomes(self, tmp_path):
        """Two worlds with different seeds must produce at least one differing trait."""
        worlds_genomes = []
        for seed in (42, 99):
            w = World.create(tmp_path / f"seed_{seed}", seed=seed)
            GeologyTier(w).tick()
            ClimateHydrologyTier(w, erosion_params=FAST_EROSION).tick()
            heightmap = w.rasters.read_layer("geology", "heightmap")
            ws = WeatherSystem(w.seed, heightmap)
            EcologyTier(w).tick(ws.generate(0.0, 1))
            species = w.events.list_species()
            g = {
                sp["species_id"]: w.events.get_species(sp["species_id"])["genome"]
                for sp in species
            }
            worlds_genomes.append(g)

        common_ids = worlds_genomes[0].keys() & worlds_genomes[1].keys()
        assert len(common_ids) > 0, "No common species IDs between different seeds"

        any_differ = False
        for sid in common_ids:
            for trait in worlds_genomes[0][sid]:
                if trait in worlds_genomes[1].get(sid, {}):
                    if worlds_genomes[0][sid][trait] != worlds_genomes[1][sid][trait]:
                        any_differ = True
                        break
            if any_differ:
                break

        assert any_differ, "All genomes identical across different seeds -- RNG not seeded properly"
