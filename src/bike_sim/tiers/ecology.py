"""Ecology simulation tier — species establishment, population dynamics, and niche differentiation.

This is the third and topmost tier in the three-tier simulation stack, reading
climate-hydrology derived state and producing per-species density fields and
a seed bank.  Phase 6 implements basic ecology: ancestor species creation,
suitability-driven establishment, logistic growth with competition, simple
dispersal, and a persistent seed bank.  No speciation, distinguished
individuals, or disturbance regimes yet.

All randomness flows through ``create_rng`` with tier_id="ecology" and
distinct pass_ids, ensuring full reproducibility from the world seed.

The tier operates at 5 years per tick.  Each tick reads the climate-hydrology
cache, computes per-species suitability surfaces, runs population dynamics,
and writes density layers plus a combined seed-bank layer to the "ecology"
namespace in the RasterStore.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bike_sim.rng import create_rng
from bike_sim.world import World

TIER = "ecology"

# Climate layers the ecology tier reads from climate_hydrology.
_CLIMATE_LAYERS = (
    "soil_moisture_summer",
    "soil_moisture_winter",
    "frost_days",
    "growing_degree_days",
    "solar_insolation",
    "distance_to_water",
)

# Archetype trait templates: (name, genome dict before perturbation).
_ANCESTOR_TEMPLATES: list[tuple[str, dict]] = [
    (
        "lowland_herb",
        {
            "drought_tolerance": 0.15,
            "frost_tolerance": 0.15,
            "shade_tolerance": 0.3,
            "growth_rate": 0.85,
            "seed_mass": 0.1,
            "max_height": 0.5,
            "lifespan": 10.0,
        },
    ),
    (
        "upland_grass",
        {
            "drought_tolerance": 0.45,
            "frost_tolerance": 0.75,
            "shade_tolerance": 0.2,
            "growth_rate": 0.6,
            "seed_mass": 0.15,
            "max_height": 0.8,
            "lifespan": 15.0,
        },
    ),
    (
        "valley_tree",
        {
            "drought_tolerance": 0.15,
            "frost_tolerance": 0.25,
            "shade_tolerance": 0.8,
            "growth_rate": 0.2,
            "seed_mass": 0.75,
            "max_height": 25.0,
            "lifespan": 300.0,
        },
    ),
    (
        "ridge_shrub",
        {
            "drought_tolerance": 0.8,
            "frost_tolerance": 0.5,
            "shade_tolerance": 0.15,
            "growth_rate": 0.35,
            "seed_mass": 0.4,
            "max_height": 3.0,
            "lifespan": 60.0,
        },
    ),
    (
        "pioneer_forb",
        {
            "drought_tolerance": 0.35,
            "frost_tolerance": 0.35,
            "shade_tolerance": 0.1,
            "growth_rate": 0.95,
            "seed_mass": 0.05,
            "max_height": 0.3,
            "lifespan": 5.0,
        },
    ),
    (
        "alpine_cushion",
        {
            "drought_tolerance": 0.85,
            "frost_tolerance": 0.95,
            "shade_tolerance": 0.05,
            "growth_rate": 0.1,
            "seed_mass": 0.2,
            "max_height": 0.1,
            "lifespan": 80.0,
        },
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian_match(
    field: NDArray[np.float64],
    optimum: float,
    sigma: float = 0.3,
) -> NDArray[np.float64]:
    """Gaussian match function: peak at *optimum*, falls off with distance."""
    return np.exp(-0.5 * ((field - optimum) / sigma) ** 2)


def _disperse(
    density: NDArray[np.float64],
    radius: int = 1,
) -> NDArray[np.float64]:
    """Spread *density* to neighbours using a distance-weighted kernel."""
    kernel_size = 2 * radius + 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    center = radius
    for i in range(kernel_size):
        for j in range(kernel_size):
            dist = np.sqrt(float((i - center) ** 2 + (j - center) ** 2))
            if dist <= radius:
                kernel[i, j] = 1.0 / (1.0 + dist)
    kernel /= kernel.sum()

    n = density.shape[0]
    padded = np.pad(density, radius, mode="constant", constant_values=0)
    result = np.zeros_like(density)
    for i in range(kernel_size):
        for j in range(kernel_size):
            result += kernel[i, j] * padded[i : i + n, j : j + n]
    return result


# ---------------------------------------------------------------------------
# EcologyTier
# ---------------------------------------------------------------------------


class EcologyTier:
    """Phase-6 ecology: ancestor species, suitability, competition, seed bank."""

    GRID_SIZE: int = 1000
    YEARS_PER_TICK: int = 5
    NUM_ANCESTORS: int = 6  # within the test-required 5-8 range

    def __init__(self, world: World) -> None:
        self._world = world

    # ── public API ─────────────────────────────────────────────────

    def tick(self) -> None:
        """Advance the ecology tier by one tick."""
        clock = self._world.tier_clocks[TIER]
        tick_num = clock.tick_number

        # Guard: climate-hydrology must have been run.
        if "temperature" not in self._world.rasters.list_layers("climate_hydrology"):
            raise RuntimeError("Climate-hydrology must be ticked before ecology")

        # On first tick, seed ancestor species and initial populations.
        if tick_num == 0:
            self._create_ancestors(tick_num)
            self._seed_initial_populations(tick_num)

        # Run one round of population dynamics.
        self._update_populations(tick_num)

        clock.tick_number += 1
        clock.simulated_year += self.YEARS_PER_TICK

    # ── ancestor creation ──────────────────────────────────────────

    def _create_ancestors(self, tick_number: int) -> None:
        rng = create_rng(self._world.seed, "ecology", "ancestors", tick_number)

        for idx, (name, template) in enumerate(_ANCESTOR_TEMPLATES):
            genome = {}
            for key, val in template.items():
                perturb = rng.uniform(-0.05, 0.05)
                genome[key] = float(np.clip(val + perturb, 0.0, None))
                # Keep tolerances in [0, 1]
                if key in (
                    "drought_tolerance",
                    "frost_tolerance",
                    "shade_tolerance",
                    "growth_rate",
                    "seed_mass",
                ):
                    genome[key] = float(np.clip(genome[key], 0.0, 1.0))

            self._world.events.add_species(
                species_id=f"anc_{idx:02d}_{name}",
                genome=genome,
                parent_id=None,
                appeared_year=0.0,
            )

    # ── initial populations ────────────────────────────────────────

    def _seed_initial_populations(self, tick_number: int) -> None:
        rng = create_rng(self._world.seed, "ecology", "init_pop", tick_number)
        store = self._world.rasters
        climate_cache = self._load_climate_cache()

        for sp in self._world.events.list_species():
            sid = sp["species_id"]
            genome = self._world.events.get_species(sid)["genome"]
            suit = self._compute_suitability(genome, climate_cache)
            # Place low density in areas with good suitability.
            initial = np.where(
                suit > 0.3,
                suit * rng.uniform(0.5, 2.0, suit.shape),
                0.0,
            )
            store.write_layer(
                TIER,
                f"species_{sid}_density",
                initial.astype(np.float64),
                tick_number,
            )

    # ── suitability ────────────────────────────────────────────────

    def _compute_suitability(
        self,
        genome: dict,
        climate_cache: dict[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Product of Gaussian match factors — Liebig's law of the minimum."""
        suit = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float64)

        # Drought: high drought_tolerance → thrives where moisture is low.
        drought_stress = 1.0 - climate_cache["soil_moisture_summer"]
        suit *= _gaussian_match(drought_stress, genome["drought_tolerance"], sigma=0.3)

        # Frost severity.
        frost_severity = climate_cache["frost_days"] / 365.0
        suit *= _gaussian_match(frost_severity, genome["frost_tolerance"], sigma=0.3)

        # GDD / warmth: frost-tolerant species don't need warmth.
        gdd = climate_cache["growing_degree_days"]
        gdd_norm = gdd / (gdd.max() + 1e-10)
        warmth_preference = 1.0 - genome["frost_tolerance"]
        suit *= _gaussian_match(gdd_norm, warmth_preference, sigma=0.4)

        # Light / shade tolerance.
        light_need = 1.0 - genome["shade_tolerance"]
        insolation = climate_cache["solar_insolation"]
        insol_norm = insolation / (insolation.max() + 1e-10)
        suit *= _gaussian_match(insol_norm, 1.0 - light_need * 0.5, sigma=0.4)

        return suit

    # ── population dynamics ────────────────────────────────────────

    def _update_populations(self, tick_number: int) -> None:
        rng = create_rng(self._world.seed, "ecology", "dynamics", tick_number)
        store = self._world.rasters
        climate_cache = self._load_climate_cache()

        species_list = self._world.events.list_species()
        n = self.GRID_SIZE
        carrying_capacity = 15.0

        # Load or initialise densities and seed banks.
        ecology_layers = store.list_layers(TIER)
        densities: dict[str, NDArray[np.float64]] = {}
        seed_banks: dict[str, NDArray[np.float64]] = {}

        for sp in species_list:
            sid = sp["species_id"]
            layer = f"species_{sid}_density"
            if layer in ecology_layers:
                densities[sid] = store.read_layer(TIER, layer).copy()
            else:
                densities[sid] = np.zeros((n, n), dtype=np.float64)

            sb_layer = f"seed_bank_{sid}"
            if sb_layer in ecology_layers:
                seed_banks[sid] = store.read_layer(TIER, sb_layer).copy()
            else:
                seed_banks[sid] = np.zeros((n, n), dtype=np.float64)

        # Total density and available capacity.
        total_density = np.zeros((n, n), dtype=np.float64)
        for d in densities.values():
            total_density += d
        available = np.clip(carrying_capacity - total_density, 0.0, None)

        for sp in species_list:
            sid = sp["species_id"]
            genome = self._world.events.get_species(sid)["genome"]
            density = densities[sid]
            sb = seed_banks[sid]

            suit = self._compute_suitability(genome, climate_cache)

            # Dispersal — light seeds spread further.
            dispersal_radius = 1 + int(2 * (1.0 - genome["seed_mass"]))
            seed_production = density * genome["growth_rate"] * 0.5
            seed_input = _disperse(seed_production, dispersal_radius)

            # Rare long-distance dispersal (designed stochasticity).
            if rng.random() < 0.1:
                ldd_count = int(rng.integers(1, 5))
                for _ in range(ldd_count):
                    r = int(rng.integers(0, n))
                    c = int(rng.integers(0, n))
                    seed_input[r, c] += rng.uniform(0.1, 1.0)

            # Seed bank: decay + input.
            half_life = 5.0 + genome["seed_mass"] * 195.0
            decay = 0.5 ** (self.YEARS_PER_TICK / half_life)
            sb = sb * decay + seed_input * 0.3

            # Establishment from seeds.
            establishment = (seed_input + sb * 0.1) * suit * (available / carrying_capacity)

            # Growth of existing populations.
            growth = density * genome["growth_rate"] * suit * 0.1 * (available / carrying_capacity)

            # Mortality.
            base_mortality = 0.02 * (1.0 / max(genome["lifespan"], 1.0) * 100.0)
            stress_mortality = 0.1 * (1.0 - suit)
            mortality = density * (base_mortality + stress_mortality)

            # Update density.
            density = density + establishment + growth - mortality
            density = np.clip(density, 0.0, None)

            # Tiny stochastic noise where populations exist.
            noise = rng.uniform(0.0, 0.001, density.shape)
            density += np.where(density > 0.0, noise, 0.0)

            # Zero out negligible densities in unsuitable habitat so that
            # species are genuinely absent from parts of the map.
            density = np.where((density < 0.01) & (suit < 0.2), 0.0, density)
            density = np.clip(density, 0.0, None)

            densities[sid] = density
            seed_banks[sid] = np.clip(sb, 0.0, None)

            # Recompute available capacity for next species.
            total_density = np.zeros((n, n), dtype=np.float64)
            for d in densities.values():
                total_density += d
            available = np.clip(carrying_capacity - total_density, 0.0, None)

        # Write results.
        seed_bank_total = np.zeros((n, n), dtype=np.float64)
        for sp in species_list:
            sid = sp["species_id"]
            store.write_layer(
                TIER,
                f"species_{sid}_density",
                densities[sid].astype(np.float64),
                tick_number,
            )
            store.write_layer(
                TIER,
                f"seed_bank_{sid}",
                seed_banks[sid].astype(np.float64),
                tick_number,
            )
            seed_bank_total += seed_banks[sid]

        store.write_layer(
            TIER,
            "seed_bank_total",
            seed_bank_total.astype(np.float64),
            tick_number,
        )

    # ── helpers ────────────────────────────────────────────────────

    def _load_climate_cache(self) -> dict[str, NDArray[np.float64]]:
        store = self._world.rasters
        cache: dict[str, NDArray[np.float64]] = {}
        for name in _CLIMATE_LAYERS:
            cache[name] = store.read_layer("climate_hydrology", name)
        return cache
