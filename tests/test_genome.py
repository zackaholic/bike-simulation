"""Tests for genome expansion: 17 traits (10 functional + 7 morphological).

Step 2 of the seasonal redesign expands species genomes from 7 traits to 17.
These tests verify trait completeness, value ranges, morphological coupling,
ancestor template values, speciation drift integrity, and determinism.
"""

import numpy as np
import pytest

from bike_sim.tiers.climate_hydrology import ClimateHydrologyTier
from bike_sim.tiers.ecology import EcologyTier
from bike_sim.tiers.erosion import ErosionParams
from bike_sim.tiers.geology import GeologyTier
from bike_sim.weather import SeasonalWeather, WeatherSystem
from bike_sim.world import World

FAST_PARAMS = ErosionParams(num_particles=1_000, max_lifetime=30)

# The full set of expected genome traits after expansion.
FUNCTIONAL_TRAITS = {
    "drought_tolerance",
    "frost_tolerance",
    "shade_tolerance",
    "growth_rate",
    "seed_mass",
    "max_height",
    "lifespan",
    "phenological_aggressiveness",
    "evergreenness",
    "mast_interval",
}

MORPHOLOGICAL_TRAITS = {
    "growth_form",
    "leaf_size",
    "leaf_shape",
    "flower_color",
    "flower_size",
    "bark_texture",
    "stem_woodiness",
}

ALL_TRAITS = FUNCTIONAL_TRAITS | MORPHOLOGICAL_TRAITS

# Functional traits that must be bounded to [0, 1].
BOUNDED_FUNCTIONAL = {
    "drought_tolerance",
    "frost_tolerance",
    "shade_tolerance",
    "growth_rate",
    "seed_mass",
    "phenological_aggressiveness",
    "evergreenness",
}

# Morphological traits that are continuous floats in [0, 1].
MORPHOLOGICAL_FLOATS = {
    "leaf_size",
    "leaf_shape",
    "flower_color",
    "flower_size",
    "bark_texture",
    "stem_woodiness",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def eco_world(tmp_path_factory):
    """Create a world with geology + climate + ecology ticked once.

    Module-scoped for speed — tests must not mutate the world.
    """
    tmp = tmp_path_factory.mktemp("genome")
    w = World.create(tmp / "world", seed=42)
    GeologyTier(w).tick()
    ClimateHydrologyTier(w, erosion_params=FAST_PARAMS).tick()
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
    ClimateHydrologyTier(w, erosion_params=FAST_PARAMS).tick()
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
    def test_every_genome_has_17_traits(self, species_genomes):
        """Every species genome must have exactly 17 traits."""
        for sid, genome in species_genomes.items():
            assert len(genome) == 17, (
                f"Species {sid} has {len(genome)} traits, expected 17. "
                f"Keys: {sorted(genome.keys())}"
            )

    def test_all_expected_trait_names_present(self, species_genomes):
        """Every genome must contain all expected functional and morphological trait names."""
        for sid, genome in species_genomes.items():
            missing = ALL_TRAITS - genome.keys()
            extra = genome.keys() - ALL_TRAITS
            assert not missing, f"Species {sid} missing traits: {missing}"
            assert not extra, f"Species {sid} has unexpected traits: {extra}"

    def test_functional_traits_present(self, species_genomes):
        """All 10 functional traits are present in every genome."""
        for sid, genome in species_genomes.items():
            missing = FUNCTIONAL_TRAITS - genome.keys()
            assert not missing, f"Species {sid} missing functional traits: {missing}"

    def test_morphological_traits_present(self, species_genomes):
        """All 7 morphological traits are present in every genome."""
        for sid, genome in species_genomes.items():
            missing = MORPHOLOGICAL_TRAITS - genome.keys()
            assert not missing, f"Species {sid} missing morphological traits: {missing}"


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

    def test_mast_interval_is_int_in_range(self, species_genomes):
        """mast_interval must be an integer in [1, 7]."""
        for sid, genome in species_genomes.items():
            val = genome["mast_interval"]
            assert isinstance(val, int), (
                f"Species {sid}: mast_interval={val} is {type(val).__name__}, expected int"
            )
            assert 1 <= val <= 7, (
                f"Species {sid}: mast_interval={val} outside [1, 7]"
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

    def test_growth_form_is_int_in_range(self, species_genomes):
        """growth_form must be an integer in [0, 4]."""
        for sid, genome in species_genomes.items():
            val = genome["growth_form"]
            assert isinstance(val, int), (
                f"Species {sid}: growth_form={val} is {type(val).__name__}, expected int"
            )
            assert 0 <= val <= 4, (
                f"Species {sid}: growth_form={val} outside [0, 4]"
            )

    def test_morphological_floats_in_unit_interval(self, species_genomes):
        """Continuous morphological traits must be in [0, 1]."""
        for sid, genome in species_genomes.items():
            for trait in MORPHOLOGICAL_FLOATS:
                val = genome[trait]
                assert 0.0 <= val <= 1.0, (
                    f"Species {sid}: {trait}={val} outside [0, 1]"
                )


# ---------------------------------------------------------------------------
# 3. Morphological coupling (soft constraints across ancestors)
# ---------------------------------------------------------------------------


class TestMorphologicalCoupling:
    """Soft tests for morphological-functional correlations across ancestors.

    These test design intent: morphological traits should be loosely coupled
    to functional traits so that species *look* like what they *are*.
    """

    def test_evergreenness_leaf_shape_negative_correlation(self, ancestor_genomes):
        """High evergreenness should correlate with lower leaf_shape (needle-like).

        We check that across the 6 ancestors, the Pearson correlation between
        evergreenness and leaf_shape is negative.
        """
        evergreenness = [g["evergreenness"] for g in ancestor_genomes.values()]
        leaf_shape = [g["leaf_shape"] for g in ancestor_genomes.values()]
        corr = np.corrcoef(evergreenness, leaf_shape)[0, 1]
        assert corr < 0, (
            f"Expected negative correlation between evergreenness and leaf_shape, "
            f"got r={corr:.3f}"
        )

    def test_lifespan_bark_texture_positive_correlation(self, ancestor_genomes):
        """High lifespan should correlate with higher bark_texture (thick bark)."""
        lifespan = [g["lifespan"] for g in ancestor_genomes.values()]
        bark = [g["bark_texture"] for g in ancestor_genomes.values()]
        corr = np.corrcoef(lifespan, bark)[0, 1]
        assert corr > 0, (
            f"Expected positive correlation between lifespan and bark_texture, "
            f"got r={corr:.3f}"
        )

    def test_tall_species_are_trees_or_shrubs(self, ancestor_genomes):
        """Species with high max_height should have growth_form 0 (tree) or 1 (shrub)."""
        for sid, genome in ancestor_genomes.items():
            if genome["max_height"] >= 3.0:
                assert genome["growth_form"] in (0, 1), (
                    f"Species {sid} has max_height={genome['max_height']} "
                    f"but growth_form={genome['growth_form']} (expected 0=tree or 1=shrub)"
                )

    def test_short_species_are_not_trees(self, ancestor_genomes):
        """Species with max_height < 0.5 should have growth_form 2, 3, or 4."""
        for sid, genome in ancestor_genomes.items():
            if genome["max_height"] < 0.5:
                assert genome["growth_form"] in (2, 3, 4), (
                    f"Species {sid} has max_height={genome['max_height']} "
                    f"but growth_form={genome['growth_form']} (expected 2/3/4 for short plants)"
                )


# ---------------------------------------------------------------------------
# 4. Ancestor trait values
# ---------------------------------------------------------------------------


class TestAncestorTraitValues:
    def test_six_ancestors_created(self, ancestor_genomes):
        """Exactly 6 ancestor species should be created."""
        assert len(ancestor_genomes) == 6, (
            f"Expected 6 ancestors, got {len(ancestor_genomes)}"
        )

    def test_new_functional_traits_present(self, ancestor_genomes):
        """The three new functional traits must be present in every ancestor."""
        new_traits = {"phenological_aggressiveness", "evergreenness", "mast_interval"}
        for sid, genome in ancestor_genomes.items():
            missing = new_traits - genome.keys()
            assert not missing, (
                f"Ancestor {sid} missing new functional traits: {missing}"
            )

    def test_pioneer_forb_phenological_aggressiveness(self, ancestor_genomes):
        """pioneer_forb should have phenological_aggressiveness > 0.6."""
        forb = [g for sid, g in ancestor_genomes.items() if "pioneer_forb" in sid]
        assert len(forb) == 1, f"Expected 1 pioneer_forb, found {len(forb)}"
        val = forb[0]["phenological_aggressiveness"]
        assert val > 0.6, (
            f"pioneer_forb phenological_aggressiveness={val}, expected > 0.6"
        )

    def test_alpine_cushion_evergreenness(self, ancestor_genomes):
        """alpine_cushion should have evergreenness > 0.6."""
        cushion = [g for sid, g in ancestor_genomes.items() if "alpine_cushion" in sid]
        assert len(cushion) == 1, f"Expected 1 alpine_cushion, found {len(cushion)}"
        val = cushion[0]["evergreenness"]
        assert val > 0.6, (
            f"alpine_cushion evergreenness={val}, expected > 0.6"
        )

    def test_valley_tree_mast_interval(self, ancestor_genomes):
        """valley_tree should have mast_interval >= 3."""
        tree = [g for sid, g in ancestor_genomes.items() if "valley_tree" in sid]
        assert len(tree) == 1, f"Expected 1 valley_tree, found {len(tree)}"
        val = tree[0]["mast_interval"]
        assert val >= 3, (
            f"valley_tree mast_interval={val}, expected >= 3"
        )

    def test_ancestor_traits_close_to_templates(self, ancestor_genomes):
        """Original 7 traits should be within perturbation range of templates.

        The perturbation range is [-0.05, 0.05] for each trait, so actual
        values should be within 0.05 of the template value.
        """
        # Build a lookup from ancestor name fragment to template.
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
                actual = genome[trait]
                assert abs(actual - expected) <= 0.05 + 1e-9, (
                    f"Ancestor {sid}: {trait}={actual}, expected within 0.05 of {expected}"
                )


# ---------------------------------------------------------------------------
# 5. Speciation drift
# ---------------------------------------------------------------------------


class TestSpeciationDrift:
    def test_speciated_species_have_17_traits(self, multi_tick_world):
        """Any species produced by speciation must have all 17 traits."""
        species = multi_tick_world.events.list_species()
        children = [sp for sp in species if sp["parent_id"] is not None]
        if len(children) == 0:
            pytest.skip("No speciation occurred — stochastic, cannot test drift")
        for sp in children:
            sid = sp["species_id"]
            genome = multi_tick_world.events.get_species(sid)["genome"]
            missing = ALL_TRAITS - genome.keys()
            assert not missing, (
                f"Speciated species {sid} missing traits: {missing}"
            )
            assert len(genome) == 17, (
                f"Speciated species {sid} has {len(genome)} traits, expected 17"
            )

    def test_growth_form_stays_int_after_speciation(self, multi_tick_world):
        """growth_form must remain an integer in [0, 4] after speciation."""
        species = multi_tick_world.events.list_species()
        children = [sp for sp in species if sp["parent_id"] is not None]
        if not children:
            pytest.skip("No speciation occurred — stochastic")
        for sp in children:
            sid = sp["species_id"]
            genome = multi_tick_world.events.get_species(sid)["genome"]
            val = genome["growth_form"]
            assert isinstance(val, int), (
                f"Speciated {sid}: growth_form={val} is {type(val).__name__}, expected int"
            )
            assert 0 <= val <= 4, (
                f"Speciated {sid}: growth_form={val} outside [0, 4]"
            )

    def test_mast_interval_stays_int_after_speciation(self, multi_tick_world):
        """mast_interval must remain an integer in [1, 7] after speciation."""
        species = multi_tick_world.events.list_species()
        children = [sp for sp in species if sp["parent_id"] is not None]
        if not children:
            pytest.skip("No speciation occurred — stochastic")
        for sp in children:
            sid = sp["species_id"]
            genome = multi_tick_world.events.get_species(sid)["genome"]
            val = genome["mast_interval"]
            assert isinstance(val, int), (
                f"Speciated {sid}: mast_interval={val} is {type(val).__name__}, expected int"
            )
            assert 1 <= val <= 7, (
                f"Speciated {sid}: mast_interval={val} outside [1, 7]"
            )

    def test_bounded_traits_stay_in_bounds_after_speciation(self, multi_tick_world):
        """All bounded traits must stay within their valid ranges after drift."""
        species = multi_tick_world.events.list_species()
        children = [sp for sp in species if sp["parent_id"] is not None]
        if not children:
            pytest.skip("No speciation occurred — stochastic")
        for sp in children:
            sid = sp["species_id"]
            genome = multi_tick_world.events.get_species(sid)["genome"]
            for trait in BOUNDED_FUNCTIONAL:
                val = genome[trait]
                assert 0.0 <= val <= 1.0, (
                    f"Speciated {sid}: {trait}={val} outside [0, 1]"
                )
            for trait in MORPHOLOGICAL_FLOATS:
                val = genome[trait]
                assert 0.0 <= val <= 1.0, (
                    f"Speciated {sid}: {trait}={val} outside [0, 1]"
                )
            assert genome["max_height"] > 0
            assert genome["lifespan"] > 0


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_genomes(self, tmp_path):
        """Two worlds with the same seed must produce identical genomes."""
        genomes = []
        for i in range(2):
            w = World.create(tmp_path / f"det_{i}", seed=42)
            GeologyTier(w).tick()
            ClimateHydrologyTier(w, erosion_params=FAST_PARAMS).tick()
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
            ClimateHydrologyTier(w, erosion_params=FAST_PARAMS).tick()
            heightmap = w.rasters.read_layer("geology", "heightmap")
            ws = WeatherSystem(w.seed, heightmap)
            EcologyTier(w).tick(ws.generate(0.0, 1))
            species = w.events.list_species()
            g = {
                sp["species_id"]: w.events.get_species(sp["species_id"])["genome"]
                for sp in species
            }
            worlds_genomes.append(g)

        # Species IDs should be the same (template-derived), but trait values should differ.
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

        assert any_differ, "All genomes identical across different seeds — RNG not seeded properly"
