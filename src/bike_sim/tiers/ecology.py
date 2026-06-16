"""Ecology simulation tier — seasonal species dynamics, disturbance, and niche differentiation.

This is the third and topmost tier in the three-tier simulation stack.  It
receives a ``SeasonalWeather`` object each tick and produces per-species density
fields, seed banks, and disturbance events.

The tier operates at **0.25 years per tick** (one season).  Each tick runs
season-specific operations:

- **Winter (0):** frost-driven mortality.
- **Spring (1):** leaf-out risk, seed bank establishment.
- **Summer (2):** growth/competition, drought mortality, fire disturbance.
- **Fall (3):** seed production/dispersal, senescence, blowdown disturbance.

Every tick also updates cumulative drought stress, biomass age tracking, and
enforces carrying capacity.  Individual promotion runs annually (every 4 ticks)
and speciation checks run every 50 years (200 ticks).

All randomness flows through ``create_rng`` with tier_id="ecology" and
distinct pass_ids, ensuring full reproducibility from the world seed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bike_sim.rng import create_rng
from bike_sim.weather import SeasonalWeather
from bike_sim.world import World

TIER = "ecology"

# Trait categories for speciation drift.
_MORPHOLOGICAL_TRAITS = {
    "growth_form", "leaf_size", "leaf_shape", "flower_color",
    "flower_size", "bark_texture", "stem_woodiness",
}
_BOUNDED_FUNCTIONAL = {
    "drought_tolerance", "frost_tolerance", "shade_tolerance",
    "growth_rate", "seed_mass", "phenological_aggressiveness", "evergreenness",
}

# Archetype trait templates: (name, genome dict before perturbation).
# Organized by structural role for broad world coverage:
#   Canopy trees (3): slow, tall, heavy seeds, expensive to render
#   Understory trees (2): faster, shorter, moderate seeds
#   Tall shrubs (4): moderate height, wide tolerance ranges
#   Low shrubs/forbs (5): fast, short, light seeds, cheap to render
_ANCESTOR_TEMPLATES: list[tuple[str, dict]] = [
    # ── Canopy trees ──────────────────────────────────────────────
    (
        "valley_hardwood",  # warm, wet lowlands — deciduous canopy tree
        {
            "drought_tolerance": 0.25,
            "frost_tolerance": 0.2,
            "shade_tolerance": 0.7,
            "growth_rate": 0.2,
            "seed_mass": 0.8,
            "max_height": 30.0,
            "lifespan": 400.0,
            "phenological_aggressiveness": 0.3,
            "evergreenness": 0.1,
            "mast_interval": 5,
        },
    ),
    (
        "upland_conifer",  # cool, moderate moisture — evergreen canopy tree
        {
            "drought_tolerance": 0.5,
            "frost_tolerance": 0.7,
            "shade_tolerance": 0.6,
            "growth_rate": 0.15,
            "seed_mass": 0.6,
            "max_height": 25.0,
            "lifespan": 500.0,
            "phenological_aggressiveness": 0.1,
            "evergreenness": 0.9,
            "mast_interval": 4,
        },
    ),
    (
        "wet_broadleaf",  # wet, mild — tall broadleaf in moist areas
        {
            "drought_tolerance": 0.15,
            "frost_tolerance": 0.3,
            "shade_tolerance": 0.8,
            "growth_rate": 0.25,
            "seed_mass": 0.7,
            "max_height": 28.0,
            "lifespan": 350.0,
            "phenological_aggressiveness": 0.4,
            "evergreenness": 0.3,
            "mast_interval": 3,
        },
    ),
    # ── Understory / edge trees ───────────────────────────────────
    (
        "gap_filler",  # fast-growing edge tree, colonizes clearings
        {
            "drought_tolerance": 0.35,
            "frost_tolerance": 0.35,
            "shade_tolerance": 0.4,
            "growth_rate": 0.5,
            "seed_mass": 0.4,
            "max_height": 12.0,
            "lifespan": 80.0,
            "phenological_aggressiveness": 0.6,
            "evergreenness": 0.2,
            "mast_interval": 2,
        },
    ),
    (
        "riparian_tree",  # moisture-loving streamside tree
        {
            "drought_tolerance": 0.1,
            "frost_tolerance": 0.4,
            "shade_tolerance": 0.3,
            "growth_rate": 0.45,
            "seed_mass": 0.3,
            "max_height": 15.0,
            "lifespan": 100.0,
            "phenological_aggressiveness": 0.5,
            "evergreenness": 0.1,
            "mast_interval": 1,
        },
    ),
    # ── Tall shrubs ───────────────────────────────────────────────
    (
        "ridge_scrub",  # drought-hardy exposed ridgeline shrub
        {
            "drought_tolerance": 0.8,
            "frost_tolerance": 0.5,
            "shade_tolerance": 0.1,
            "growth_rate": 0.3,
            "seed_mass": 0.35,
            "max_height": 3.0,
            "lifespan": 60.0,
            "phenological_aggressiveness": 0.2,
            "evergreenness": 0.7,
            "mast_interval": 2,
        },
    ),
    (
        "valley_thicket",  # wet valley shrub, forms dense stands
        {
            "drought_tolerance": 0.2,
            "frost_tolerance": 0.3,
            "shade_tolerance": 0.5,
            "growth_rate": 0.4,
            "seed_mass": 0.3,
            "max_height": 4.0,
            "lifespan": 40.0,
            "phenological_aggressiveness": 0.5,
            "evergreenness": 0.3,
            "mast_interval": 1,
        },
    ),
    (
        "heath_shrub",  # cold-tolerant moorland/heathland shrub
        {
            "drought_tolerance": 0.55,
            "frost_tolerance": 0.7,
            "shade_tolerance": 0.15,
            "growth_rate": 0.25,
            "seed_mass": 0.2,
            "max_height": 1.5,
            "lifespan": 50.0,
            "phenological_aggressiveness": 0.15,
            "evergreenness": 0.8,
            "mast_interval": 2,
        },
    ),
    (
        "dry_scrub",  # arid-adapted shrub
        {
            "drought_tolerance": 0.9,
            "frost_tolerance": 0.4,
            "shade_tolerance": 0.05,
            "growth_rate": 0.2,
            "seed_mass": 0.25,
            "max_height": 2.0,
            "lifespan": 70.0,
            "phenological_aggressiveness": 0.1,
            "evergreenness": 0.6,
            "mast_interval": 3,
        },
    ),
    # ── Low shrubs / forbs / ground-hugging ───────────────────────
    (
        "meadow_herb",  # mesic meadow flowering herb
        {
            "drought_tolerance": 0.3,
            "frost_tolerance": 0.3,
            "shade_tolerance": 0.3,
            "growth_rate": 0.8,
            "seed_mass": 0.1,
            "max_height": 0.5,
            "lifespan": 10.0,
            "phenological_aggressiveness": 0.7,
            "evergreenness": 0.1,
            "mast_interval": 1,
        },
    ),
    (
        "upland_grass",  # cold-tolerant grassland species
        {
            "drought_tolerance": 0.45,
            "frost_tolerance": 0.75,
            "shade_tolerance": 0.2,
            "growth_rate": 0.6,
            "seed_mass": 0.15,
            "max_height": 0.8,
            "lifespan": 15.0,
            "phenological_aggressiveness": 0.4,
            "evergreenness": 0.3,
            "mast_interval": 1,
        },
    ),
    (
        "pioneer_forb",  # fast-growing disturbance colonizer
        {
            "drought_tolerance": 0.35,
            "frost_tolerance": 0.35,
            "shade_tolerance": 0.1,
            "growth_rate": 0.95,
            "seed_mass": 0.05,
            "max_height": 0.3,
            "lifespan": 5.0,
            "phenological_aggressiveness": 0.8,
            "evergreenness": 0.0,
            "mast_interval": 1,
        },
    ),
    (
        "alpine_cushion",  # high-altitude stress-tolerator
        {
            "drought_tolerance": 0.65,
            "frost_tolerance": 0.85,
            "shade_tolerance": 0.05,
            "growth_rate": 0.1,
            "seed_mass": 0.2,
            "max_height": 0.1,
            "lifespan": 80.0,
            "phenological_aggressiveness": 0.1,
            "evergreenness": 0.8,
            "mast_interval": 3,
        },
    ),
    (
        "dry_grass",  # drought-adapted grassland species
        {
            "drought_tolerance": 0.75,
            "frost_tolerance": 0.4,
            "shade_tolerance": 0.1,
            "growth_rate": 0.5,
            "seed_mass": 0.1,
            "max_height": 0.6,
            "lifespan": 8.0,
            "phenological_aggressiveness": 0.3,
            "evergreenness": 0.2,
            "mast_interval": 1,
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


def _bfs_label(mask: NDArray[np.bool_]) -> NDArray[np.int32]:
    """Label connected components in a boolean mask using BFS. Returns labels 1..N."""
    rows, cols = mask.shape
    labels = np.zeros((rows, cols), dtype=np.int32)
    current_label = 0

    for r in range(rows):
        for c in range(cols):
            if mask[r, c] and labels[r, c] == 0:
                current_label += 1
                queue = [(r, c)]
                labels[r, c] = current_label
                head = 0
                while head < len(queue):
                    cr, cc = queue[head]
                    head += 1
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            and mask[nr, nc]
                            and labels[nr, nc] == 0
                        ):
                            labels[nr, nc] = current_label
                            queue.append((nr, nc))

    return labels


def _fragments_connect_through_hospitable(
    fragment: NDArray[np.bool_],
    main: NDArray[np.bool_],
    hospitable: NDArray[np.bool_],
) -> bool:
    """BFS from *fragment* through *hospitable* cells; return True if it reaches *main*.

    All arrays must be the same shape (typically the downsampled 100×100 grid).
    A True result means there is no real barrier — the species could grow
    through the gap — so speciation should be rejected.
    """
    rows, cols = fragment.shape
    # Seed BFS from all fragment cells
    visited = fragment.copy()
    queue = list(zip(*np.where(fragment)))
    head = 0
    while head < len(queue):
        r, c = queue[head]
        head += 1
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                if main[nr, nc]:
                    return True  # reached main population — no barrier
                if hospitable[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
    return False  # couldn't reach main — barrier exists


def _connected_components_coarse(
    mask: NDArray[np.bool_], downsample: int = 10
) -> NDArray[np.int32]:
    """Connected components on a downsampled grid for performance.

    A coarse cell is occupied if any fine cell within it is occupied.
    Labels are upsampled back to the original resolution.
    """
    rows, cols = mask.shape
    cr = rows // downsample
    cc = cols // downsample

    # Downsample using reshape + any — much faster than Python loops.
    coarse = (
        mask[: cr * downsample, : cc * downsample]
        .reshape(cr, downsample, cc, downsample)
        .any(axis=(1, 3))
    )

    # BFS on the small coarse grid.
    labels_coarse = _bfs_label(coarse)

    # Upsample labels back to full resolution.
    labels = np.repeat(np.repeat(labels_coarse, downsample, axis=0), downsample, axis=1)
    # Pad if original size wasn't evenly divisible.
    if labels.shape[0] < rows or labels.shape[1] < cols:
        full = np.zeros((rows, cols), dtype=np.int32)
        full[: labels.shape[0], : labels.shape[1]] = labels
        labels = full
    # Zero out unoccupied cells.
    labels[~mask] = 0

    return labels


# ---------------------------------------------------------------------------
# EcologyTier
# ---------------------------------------------------------------------------


class EcologyTier:
    """Ecology tier: seasonal species dynamics, disturbance, and niche differentiation."""

    GRID_SIZE: int = 1000
    YEARS_PER_TICK: float = 0.25
    NUM_ANCESTORS: int = 14  # 3 canopy + 2 understory + 4 shrub + 5 forb/grass
    CELL_SIZE: float = 50.0  # meters per grid cell (50 km / 1000 cells)

    def __init__(self, world: World) -> None:
        self._world = world

    # ── public API ─────────────────────────────────────────────────

    def tick(self, weather: SeasonalWeather) -> None:
        """Advance ecology by one seasonal tick."""
        clock = self._world.tier_clocks[TIER]
        tick_num = clock.tick_number
        season = weather.season  # 0=winter, 1=spring, 2=summer, 3=fall

        # On first tick, seed ancestor species and initial populations.
        if tick_num == 0:
            self._create_ancestors(tick_num)
            self._seed_initial_populations(tick_num, weather)

        # Load all species state (exclude extinct species)
        current_year = self._world.tier_clocks[TIER].simulated_year
        species_list = self._world.events.list_species(alive_at_year=current_year)
        densities, seed_banks = self._load_species_state(species_list)

        # Season-specific operations
        if season == 0:  # Winter
            self._winter_mortality(weather, species_list, densities)
        elif season == 1:  # Spring
            self._spring_leafout_and_frost(weather, species_list, densities)
            self._spring_establishment(weather, species_list, densities, seed_banks)
        elif season == 2:  # Summer
            self._summer_growth_and_competition(weather, species_list, densities, tick_num)
            self._summer_drought_mortality(weather, species_list, densities)
            self._fire_disturbance(weather, species_list, densities, tick_num)
        elif season == 3:  # Fall
            self._seed_production_and_dispersal(weather, species_list, densities, seed_banks, tick_num)
            self._senescence_and_fuel(weather, species_list, densities)
            self._blowdown_disturbance(weather, species_list, densities, tick_num)

        # Every tick:
        self._update_cumulative_drought(weather)
        self._update_biomass_age(species_list, densities)

        # Individual lifecycle (aging, death, post-mortem transitions)
        self._update_individual_lifecycle(tick_num)

        # Enforce carrying capacity
        self._enforce_carrying_capacity(densities)

        # Biotic pressure: oscillating top-down mortality from pathogens/herbivores
        # Skip first 8 ticks (2 years) — let species establish before pressure builds
        if tick_num >= 8:
            self._apply_biotic_pressure(species_list, densities, weather)

        # Check for species extinction (minimum viable population)
        # Skip on tick 0 — species were just created and need time to establish
        if tick_num > 0:
            self._check_extinction(species_list, densities, seed_banks, tick_num)

        # Write all state
        self._write_species_state(species_list, densities, seed_banks, tick_num)

        # Promote individuals every 4 ticks (annually)
        if tick_num % 4 == 0:
            self._promote_individuals(tick_num)

        # Speciation check every 200 ticks (50 years)
        if tick_num > 0 and tick_num % 200 == 0:
            self._check_speciation(tick_num, weather)
            self._check_reabsorption(tick_num, weather)

        clock.tick_number += 1
        clock.simulated_year += self.YEARS_PER_TICK

    # ── ancestor creation ──────────────────────────────────────────

    def _create_ancestors(self, tick_number: int) -> None:
        rng = create_rng(self._world.seed, "ecology", "ancestors", tick_number)

        for idx, (name, template) in enumerate(_ANCESTOR_TEMPLATES):
            genome = {}
            for key, val in template.items():
                if key == "mast_interval":
                    # Integer trait: perturb by rounding after uniform offset.
                    genome["mast_interval"] = int(
                        np.clip(round(val + rng.uniform(-0.5, 0.5)), 1, 7)
                    )
                else:
                    perturb = rng.uniform(-0.05, 0.05)
                    genome[key] = float(np.clip(val + perturb, 0.0, None))
                    # Keep bounded functional traits in [0, 1]
                    if key in (
                        "drought_tolerance",
                        "frost_tolerance",
                        "shade_tolerance",
                        "growth_rate",
                        "seed_mass",
                        "phenological_aggressiveness",
                        "evergreenness",
                    ):
                        genome[key] = float(np.clip(genome[key], 0.0, 1.0))

            # ── Morphological trait derivation (soft coupling + random offset) ──

            # growth_form: enum 0-4 (tree=0, shrub=1, herb=2, grass=3, cushion=4)
            if genome["max_height"] > 10:
                base_form = 0  # tree
            elif genome["max_height"] > 1.5:
                base_form = 1  # shrub
            elif genome["lifespan"] > 20:
                base_form = 2  # herb
            elif genome["growth_rate"] > 0.5:
                base_form = 3  # grass
            else:
                base_form = 4  # cushion
            genome["growth_form"] = base_form

            # leaf_size: [0, 1], coupled to shade_tolerance (+) and drought_tolerance (-)
            genome["leaf_size"] = float(np.clip(
                genome["shade_tolerance"] * 0.6
                - genome["drought_tolerance"] * 0.4
                + 0.4
                + rng.normal(0, 0.1),
                0.0, 1.0,
            ))

            # leaf_shape: [0, 1] (0=needle, 1=broad), coupled to evergreenness
            genome["leaf_shape"] = float(np.clip(
                1.0 - genome["evergreenness"] * 0.7 + rng.normal(0, 0.1),
                0.0, 1.0,
            ))

            # flower_color: [0, 1] (hue wheel), fully independent
            genome["flower_color"] = float(rng.uniform(0, 1))

            # flower_size: [0, 1], inversely correlated with seed_mass and evergreenness
            genome["flower_size"] = float(np.clip(
                0.6
                - genome["seed_mass"] * 0.3
                - genome["evergreenness"] * 0.3
                + rng.normal(0, 0.1),
                0.0, 1.0,
            ))

            # bark_texture: [0, 1] (smooth to rough), correlated with lifespan
            genome["bark_texture"] = float(np.clip(
                min(genome["lifespan"] / 300.0, 1.0) + rng.normal(0, 0.1),
                0.0, 1.0,
            ))

            # stem_woodiness: [0, 1] (herbaceous to woody)
            genome["stem_woodiness"] = float(np.clip(
                min(genome["lifespan"] / 200.0, 1.0) * 0.6
                + (1.0 - genome["growth_rate"]) * 0.4
                + rng.normal(0, 0.1),
                0.0, 1.0,
            ))

            self._world.events.add_species(
                species_id=f"anc_{idx:02d}_{name}",
                genome=genome,
                parent_id=None,
                appeared_year=0.0,
            )

    # ── initial populations ────────────────────────────────────────

    def _seed_initial_populations(self, tick_number: int, weather: SeasonalWeather) -> None:
        """Broad noise-based placement: scatter species across suitable habitat.

        Each species gets a unique noise field (from world seed + species index)
        that creates natural clustering (groves, patches). Initial density is
        suitability * noise * role-based max density, zeroed below a threshold.

        This produces a world that is already "full" — every habitable cell has
        vegetation appropriate to its climate zone.
        """
        from bike_sim.tiers.geology import _bilinear_upsample

        rng = create_rng(self._world.seed, "ecology", "init_pop", tick_number)
        store = self._world.rasters

        # Use annual-mean climate for initial placement, not the current
        # season's weather.  Tick 0 is winter — warm-loving species would
        # get zero suitability and never establish if we used winter weather.
        base_temp = store.read_layer("climate_hydrology", "temperature")
        base_precip = store.read_layer("climate_hydrology", "precipitation")
        annual_weather = SeasonalWeather(
            temperature=base_temp,
            precipitation=base_precip,
            frost_severity=np.zeros_like(base_temp),
            storm_intensity=0.0,
            season=1,  # cosmetic; suitability doesn't check this field
        )

        # Role-based max density: trees are sparse, forbs are dense
        role_max_density = {
            0: 1.5,   # canopy tree: sparse (heavy geometry)
            1: 2.5,   # understory tree: moderate
            2: 3.0,   # tall shrub: moderate-dense
            3: 4.0,   # low shrub/forb: dense
        }

        for idx, sp in enumerate(self._world.events.list_species()):
            sid = sp["species_id"]
            genome = self._world.events.get_species(sid)["genome"]
            suit = self._compute_suitability_from_weather(genome, annual_weather)

            # Determine structural role from growth_form
            growth_form = genome.get("growth_form", 3)
            if growth_form == 0:  # tree
                if genome.get("max_height", 0) > 15:
                    role = 0  # canopy
                else:
                    role = 1  # understory
            elif growth_form == 1:  # shrub
                role = 2
            else:  # herb, grass, cushion
                role = 3

            max_density = role_max_density[role]

            # Generate species-specific clustering noise
            # Use different noise sizes per role for different spatial patterns:
            # Trees cluster in groves (large patches), forbs are more dispersed
            noise_sizes = {0: (5, 10), 1: (6, 12), 2: (8, 16), 3: (10, 20)}
            sz1, sz2 = noise_sizes[role]

            noise = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float64)
            n1 = rng.random((sz1, sz1))
            noise += _bilinear_upsample(n1, self.GRID_SIZE) * 1.0
            n2 = rng.random((sz2, sz2))
            noise += _bilinear_upsample(n2, self.GRID_SIZE) * 0.5

            # Normalize noise to [0, 1]
            lo, hi = noise.min(), noise.max()
            if hi > lo:
                noise = (noise - lo) / (hi - lo)
            else:
                noise[:] = 0.5

            # Initial density = suitability * noise * max_density
            # with a suitability threshold to prevent placement in truly
            # unsuitable areas
            initial = np.where(
                suit > 0.15,
                suit * noise * max_density,
                0.0,
            )

            store.write_layer(
                TIER,
                f"species_{sid}_density",
                initial.astype(np.float64),
                tick_number,
            )

    # ── suitability from weather ──────────────────────────────────

    def _compute_suitability_from_weather(
        self,
        genome: dict,
        weather: SeasonalWeather,
        canopy_shade: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Suitability from current seasonal weather conditions."""
        n = self.GRID_SIZE
        suit = np.ones((n, n), dtype=np.float64)

        # Moisture suitability — normalize against actual map range so the
        # full [0, 1] drought_stress spectrum is available.  Fixed /2000
        # denominator clamped drought_stress to [0, ~0.6], making dry-niche
        # species (drought_tolerance > 0.6) structurally impossible.
        # When spatial range is narrow (e.g. uniform test input), center at
        # the absolute position within a reference range (0-3000mm).
        p_min = float(weather.precipitation.min())
        p_max = float(weather.precipitation.max())
        p_span = p_max - p_min
        if p_span > 200.0:
            # Real spatial variation — normalize to fill [0, 1]
            precip_norm = np.clip((weather.precipitation - p_min) / p_span, 0, 1)
        else:
            # Narrow range — use absolute position in reference range
            precip_norm = np.clip(weather.precipitation / 3000.0, 0, 1)
        drought_stress = 1.0 - precip_norm
        suit *= _gaussian_match(drought_stress, genome["drought_tolerance"], sigma=0.25)

        # Temperature suitability — same approach: relative when there's
        # spatial variation, absolute when uniform.
        t_min = float(weather.temperature.min())
        t_max = float(weather.temperature.max())
        t_span = t_max - t_min
        if t_span > 5.0:
            temp_norm = np.clip((weather.temperature - t_min) / t_span, 0, 1)
        else:
            # Narrow range — use absolute position in reference range (-10 to 25°C)
            temp_norm = np.clip((weather.temperature - (-10.0)) / 35.0, 0, 1)
        warmth_preference = 1.0 - genome["frost_tolerance"]
        suit *= _gaussian_match(temp_norm, warmth_preference, sigma=0.25)

        # Light competition via canopy shading (max_height becomes load-bearing)
        if canopy_shade is not None:
            # shade_tolerance determines ability to grow under canopy
            light_available = 1.0 - canopy_shade
            light_need = 1.0 - genome["shade_tolerance"]
            # Species that need light are penalized by shade
            suit *= _gaussian_match(light_available, 1.0 - light_need * 0.5, sigma=0.25)

        return suit

    # ── niche overlap ──────────────────────────────────────────────

    @staticmethod
    def _genome_distance(g1: dict, g2: dict) -> float:
        """Euclidean distance between two genomes in functional trait space."""
        keys = [
            "drought_tolerance", "frost_tolerance", "shade_tolerance",
            "growth_rate", "seed_mass", "phenological_aggressiveness",
            "evergreenness",
        ]
        return float(np.sqrt(sum((g1.get(k, 0) - g2.get(k, 0)) ** 2 for k in keys)))

    @staticmethod
    def _competition_alpha(distance: float, niche_width: float = 0.3) -> float:
        """Lotka-Volterra competition coefficient from genome distance.

        alpha = 1.0 when distance = 0 (identical species compete fully).
        alpha -> 0 as distance >> niche_width (different species barely compete).
        """
        return float(np.exp(-(distance / niche_width) ** 2))

    # ── canopy shade ──────────────────────────────────────────────

    def _compute_canopy_shade(
        self,
        species_list: list,
        densities: dict,
    ) -> NDArray[np.float64]:
        """Compute canopy shade from species heights and densities."""
        n = self.GRID_SIZE
        shade = np.zeros((n, n), dtype=np.float64)
        for sp in species_list:
            sid = sp["species_id"]
            genome = self._world.events.get_species(sid)["genome"]
            # Shade cast is proportional to density * relative height
            height_factor = min(genome["max_height"] / 30.0, 1.0)  # normalize to ~30m max
            shade += densities[sid] * height_factor * 0.1  # 0.1 = shade per unit density
        return np.clip(shade, 0.0, 1.0)

    # ── season-specific methods ───────────────────────────────────

    def _winter_mortality(self, weather: SeasonalWeather, species_list: list, densities: dict) -> None:
        """Winter kill: species with low frost tolerance die in cold conditions."""
        for sp in species_list:
            sid = sp["species_id"]
            genome = self._world.events.get_species(sid)["genome"]
            cold_hardiness = genome["frost_tolerance"] * 0.8 + genome["evergreenness"] * 0.2
            # Mortality where frost exceeds hardiness
            excess_frost = np.clip(weather.frost_severity - cold_hardiness, 0, None)
            kill_fraction = excess_frost * 0.15  # 15% of excess frost kills
            densities[sid] *= (1.0 - kill_fraction)
            densities[sid] = np.clip(densities[sid], 0.0, None)

    def _spring_leafout_and_frost(self, weather: SeasonalWeather, species_list: list, densities: dict) -> None:
        """Early leafers risk late frost damage."""
        for sp in species_list:
            sid = sp["species_id"]
            genome = self._world.events.get_species(sid)["genome"]
            aggressiveness = genome["phenological_aggressiveness"]
            # Only aggressive species leaf out in spring
            if aggressiveness < 0.3:
                continue
            # Frost damage = frost_severity * aggressiveness * (1 - frost_tolerance)
            damage = weather.frost_severity * aggressiveness * (1.0 - genome["frost_tolerance"])
            kill_fraction = np.clip(damage * 0.2, 0, 0.5)  # cap at 50% loss
            densities[sid] *= (1.0 - kill_fraction)
            densities[sid] = np.clip(densities[sid], 0.0, None)

    def _spring_establishment(
        self,
        weather: SeasonalWeather,
        species_list: list,
        densities: dict,
        seed_banks: dict,
    ) -> None:
        """Recruitment from seed bank, weighted by suitability. Early leafers get bonus."""
        n = self.GRID_SIZE
        carrying_capacity = 15.0
        total = sum(densities.values())
        if isinstance(total, int):
            total = np.zeros((n, n), dtype=np.float64)
        available = np.clip(carrying_capacity - total, 0, None)

        canopy = self._compute_canopy_shade(species_list, densities)

        for sp in species_list:
            sid = sp["species_id"]
            genome = self._world.events.get_species(sid)["genome"]
            sb = seed_banks.get(sid, np.zeros((n, n), dtype=np.float64))
            suit = self._compute_suitability_from_weather(genome, weather, canopy)

            # Early leafers get establishment bonus in spring
            aggressiveness_bonus = 1.0 + genome["phenological_aggressiveness"] * 0.5
            establishment = sb * 0.05 * suit * (available / carrying_capacity) * aggressiveness_bonus
            densities[sid] = densities[sid] + establishment
            densities[sid] = np.clip(densities[sid], 0.0, None)

            # Recompute available
            total = np.zeros((n, n), dtype=np.float64)
            for d in densities.values():
                total += d
            available = np.clip(carrying_capacity - total, 0, None)

    def _summer_growth_and_competition(
        self,
        weather: SeasonalWeather,
        species_list: list,
        densities: dict,
        tick_num: int,
    ) -> None:
        """Main growth season. Height-based light competition with niche overlap."""
        rng = create_rng(self._world.seed, "ecology", "summer_growth", tick_num)
        n = self.GRID_SIZE
        carrying_capacity = 15.0

        # Compute canopy shade
        canopy = self._compute_canopy_shade(species_list, densities)

        # Precompute genomes and alpha matrix for niche-aware growth
        sids = [sp["species_id"] for sp in species_list]
        genomes = {
            sid: self._world.events.get_species(sid)["genome"]
            for sid in sids
        }
        alphas: dict[tuple[str, str], float] = {}
        for i, s1 in enumerate(sids):
            for j, s2 in enumerate(sids):
                if j < i:
                    alphas[(s1, s2)] = alphas[(s2, s1)]
                elif s1 == s2:
                    alphas[(s1, s2)] = 1.0
                else:
                    dist = self._genome_distance(genomes[s1], genomes[s2])
                    alphas[(s1, s2)] = self._competition_alpha(dist)

        # Load biomass age for establishment advantage
        biomass_ages = self._load_biomass_age()

        for sp in species_list:
            sid = sp["species_id"]
            genome = genomes[sid]
            density = densities[sid]
            suit = self._compute_suitability_from_weather(genome, weather, canopy)

            # Niche-aware available capacity: weight competitors by overlap
            effective_load = np.zeros((n, n), dtype=np.float64)
            for other_sid in sids:
                alpha = alphas[(sid, other_sid)]
                if alpha > 0.01:
                    effective_load += alpha * densities[other_sid]
            available = np.clip(carrying_capacity - effective_load, 0, None)

            # Growth scaled by growth_rate, suitability, and available capacity
            # Established populations (high biomass_age) grow more efficiently
            age_bonus = 1.0
            if sid in biomass_ages:
                age_bonus = 1.0 + np.clip(biomass_ages[sid] / 100.0, 0, 0.5)

            growth = density * genome["growth_rate"] * suit * 0.15 * (available / carrying_capacity) * age_bonus

            # Base mortality (1/lifespan scaled for seasonal tick)
            base_mortality_rate = 0.25 / max(genome["lifespan"], 1.0)
            stress_mortality = 0.02 * (1.0 - suit)
            mortality = density * (base_mortality_rate + stress_mortality)

            density = density + growth - mortality
            density = np.clip(density, 0.0, None)

            # Zero out negligible
            density = np.where((density < 0.01) & (suit < 0.2), 0.0, density)
            densities[sid] = density

    def _summer_drought_mortality(self, weather: SeasonalWeather, species_list: list, densities: dict) -> None:
        """Drought stress kills species with low drought tolerance."""
        # Load cumulative drought stress
        drought_stress = self._load_drought_stress()

        for sp in species_list:
            sid = sp["species_id"]
            genome = self._world.events.get_species(sid)["genome"]
            # Mortality scales with cumulative drought and inversely with tolerance
            vulnerability = 1.0 - genome["drought_tolerance"]
            kill_fraction = np.clip(drought_stress * vulnerability * 0.05, 0, 0.3)
            densities[sid] *= (1.0 - kill_fraction)
            densities[sid] = np.clip(densities[sid], 0.0, None)

    def _fire_disturbance(
        self,
        weather: SeasonalWeather,
        species_list: list,
        densities: dict,
        tick_num: int,
    ) -> None:
        """Fire in summer, driven by dryness and storm intensity."""
        rng = create_rng(self._world.seed, "ecology", "fire", tick_num)
        current_year = self._world.tier_clocks[TIER].simulated_year

        # Skip on first tick
        if tick_num == 0:
            return

        # Derive moisture from precipitation (absolute scale for fire risk)
        moisture = np.clip(weather.precipitation / 3000.0, 0, 1)

        # Fire probability scales with dryness and storm intensity
        # Fewer fires per seasonal tick than per 5-year tick
        fire_rate = 0.3 + weather.storm_intensity * 0.2  # base + storm bonus
        n_fires = int(rng.poisson(fire_rate))

        for _ in range(n_fires):
            dryness = 1.0 - moisture
            dryness_flat = dryness.ravel()
            total = dryness_flat.sum()
            if total <= 0:
                continue
            probs = dryness_flat / total
            ignition_idx = int(rng.choice(len(probs), p=probs))
            ig_row, ig_col = divmod(ignition_idx, self.GRID_SIZE)

            burned = self._spread_fire(ig_row, ig_col, moisture, rng)

            if burned.sum() > 0:
                x = float(ig_col * self.CELL_SIZE + self.CELL_SIZE / 2)
                y = float(ig_row * self.CELL_SIZE + self.CELL_SIZE / 2)
                radius = float(np.sqrt(burned.sum()) * self.CELL_SIZE / 2)
                self._world.events.add_event(
                    "fire", x, y, current_year,
                    radius=radius,
                    data={"cells_burned": int(burned.sum()), "tick": tick_num},
                )

                for sp in species_list:
                    sid = sp["species_id"]
                    kill_fraction = float(rng.uniform(0.7, 0.95))
                    densities[sid] = np.where(burned, densities[sid] * (1 - kill_fraction), densities[sid])

                # Kill individuals in burned area
                for ind in self._world.events.find_individuals_near(
                    25000, 25000, 50000
                ):
                    if ind.get("state") != "alive":
                        continue
                    col = int(ind["x"] / self.CELL_SIZE)
                    row = int(ind["y"] / self.CELL_SIZE)
                    if 0 <= row < self.GRID_SIZE and 0 <= col < self.GRID_SIZE:
                        if burned[row, col]:
                            if rng.random() < 0.7:  # 70% kill chance in fire
                                self._world.events.kill_individual(ind["individual_id"], current_year)
                                self._world.events.update_individual_state(ind["individual_id"], "snag")

    def _seed_production_and_dispersal(
        self,
        weather: SeasonalWeather,
        species_list: list,
        densities: dict,
        seed_banks: dict,
        tick_num: int,
    ) -> None:
        """Fall: produce seeds, disperse, update seed bank."""
        rng = create_rng(self._world.seed, "ecology", "dispersal", tick_num)
        n = self.GRID_SIZE
        current_year = self._world.tier_clocks[TIER].simulated_year

        for sp in species_list:
            sid = sp["species_id"]
            genome = self._world.events.get_species(sid)["genome"]
            density = densities[sid]

            # Mast seeding: full production only in mast years
            mast_interval = int(genome.get("mast_interval", 1))
            year = int(current_year)
            # Species-specific phase offset from genome (use hash of sid)
            phase = hash(sid) % max(mast_interval, 1)
            if mast_interval > 1 and (year % mast_interval) != phase:
                seed_multiplier = 0.1  # 10% in non-mast years
            else:
                seed_multiplier = 1.0

            seed_production = density * genome["growth_rate"] * 0.3 * seed_multiplier

            # Local dispersal (kernel radius scales with seed lightness)
            # Increased from 1-3 to 2-6 cells (100-300m) for meaningful spread
            dispersal_radius = 2 + int(4 * (1.0 - genome["seed_mass"]))
            seed_input = _disperse(seed_production, dispersal_radius)

            # Long-distance dispersal scaled by seed_mass.
            # Light seeds (low mass) = wind/bird carried, frequent long jumps.
            # Heavy seeds (high mass) = gravity/mammal, rare short jumps.
            # This bridges gaps and enables colonization of distant suitable habitat.
            seed_mass = genome["seed_mass"]
            ldd_probability = 0.6 * (1.0 - seed_mass) + 0.1  # 10-70% chance per tick
            if rng.random() < ldd_probability:
                # Number of landing sites scales with lightness
                ldd_count = int(rng.integers(5, 15 + int(20 * (1.0 - seed_mass))))
                # Max jump distance scales inversely with mass
                max_jump = int(n * (0.15 + 0.4 * (1.0 - seed_mass)))  # 15-55% of grid
                for _ in range(ldd_count):
                    # Pick a source cell with high density
                    source_cells = np.argwhere(density > 1.0)
                    if len(source_cells) == 0:
                        break
                    src_idx = int(rng.integers(0, len(source_cells)))
                    sr, sc = source_cells[src_idx]
                    # Jump in random direction
                    dr = int(rng.integers(-max_jump, max_jump + 1))
                    dc = int(rng.integers(-max_jump, max_jump + 1))
                    tr = int(np.clip(sr + dr, 0, n - 1))
                    tc = int(np.clip(sc + dc, 0, n - 1))
                    # Deposit seeds — enough to establish a viable colony
                    deposit = float(density[sr, sc]) * 0.05 * rng.uniform(0.5, 2.0)
                    seed_input[tr, tc] += deposit

            # Seed bank: decay + input
            half_life = 5.0 + genome["seed_mass"] * 195.0
            decay = 0.5 ** (0.25 / half_life)  # seasonal decay
            sb = seed_banks.get(sid, np.zeros((n, n), dtype=np.float64))
            seed_banks[sid] = np.clip(sb * decay + seed_input * 0.3, 0.0, None)

    def _senescence_and_fuel(self, weather: SeasonalWeather, species_list: list, densities: dict) -> None:
        """Deciduous species drop leaves. Fuel accumulates for next summer's fire."""
        # Fuel load is implicit via density — higher density = more fuel for fire spread
        # Deciduous species (low evergreenness) lose some density in fall as leaf drop
        # This is a minor effect — mainly narrative/future use
        pass  # Fuel is computed from density at fire time; no explicit fuel layer yet

    def _blowdown_disturbance(
        self,
        weather: SeasonalWeather,
        species_list: list,
        densities: dict,
        tick_num: int,
    ) -> None:
        """Storm-driven windthrow in fall."""
        rng = create_rng(self._world.seed, "ecology", "blowdown", tick_num)
        current_year = self._world.tier_clocks[TIER].simulated_year

        if tick_num == 0:
            return

        # Blowdown probability scales with storm intensity
        blowdown_rate = 0.05 + weather.storm_intensity * 0.15
        n_blowdown = int(rng.poisson(blowdown_rate))

        # Need heightmap for exposure
        store = self._world.rasters
        try:
            eroded_hm = store.read_layer("climate_hydrology", "eroded_heightmap")
        except Exception:
            return  # No heightmap available

        for _ in range(n_blowdown):
            elev_norm = eroded_hm / (eroded_hm.max() + 1e-10)
            exposure = elev_norm.ravel()
            total = exposure.sum()
            if total <= 0:
                continue
            probs = exposure / total
            idx = int(rng.choice(len(probs), p=probs))
            bd_row, bd_col = divmod(idx, self.GRID_SIZE)

            patch_radius = int(rng.integers(3, 10))  # smaller patches for seasonal
            y_coords, x_coords = np.ogrid[: self.GRID_SIZE, : self.GRID_SIZE]
            dist = np.sqrt((y_coords - bd_row) ** 2 + (x_coords - bd_col) ** 2).astype(np.float64)
            affected = dist <= patch_radius

            if affected.sum() > 0:
                x = float(bd_col * self.CELL_SIZE + self.CELL_SIZE / 2)
                y = float(bd_row * self.CELL_SIZE + self.CELL_SIZE / 2)
                self._world.events.add_event(
                    "blowdown", x, y, current_year,
                    radius=float(patch_radius * self.CELL_SIZE),
                    data={"cells_affected": int(affected.sum()), "tick": tick_num},
                )
                for sp in species_list:
                    sid = sp["species_id"]
                    kill_fraction = float(rng.uniform(0.4, 0.8))
                    densities[sid] = np.where(affected, densities[sid] * (1 - kill_fraction), densities[sid])

                # Kill individuals in blowdown area
                for ind in self._world.events.find_individuals_near(
                    float(bd_col * self.CELL_SIZE), float(bd_row * self.CELL_SIZE),
                    float(patch_radius * self.CELL_SIZE * 1.5)
                ):
                    if ind.get("state") != "alive":
                        continue
                    col = int(ind["x"] / self.CELL_SIZE)
                    row = int(ind["y"] / self.CELL_SIZE)
                    if 0 <= row < self.GRID_SIZE and 0 <= col < self.GRID_SIZE:
                        if affected[row, col]:
                            if rng.random() < 0.5:  # 50% kill chance in blowdown
                                self._world.events.kill_individual(ind["individual_id"], current_year)
                                self._world.events.update_individual_state(ind["individual_id"], "snag")

    # ── fire spread ──────────────────────────────────────────────

    def _spread_fire(
        self,
        start_row: int,
        start_col: int,
        moisture: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.bool_]:
        """Spread fire from ignition point. Returns boolean burned mask."""
        n = self.GRID_SIZE
        burned = np.zeros((n, n), dtype=bool)
        burned[start_row, start_col] = True
        active = [(start_row, start_col)]

        max_cells = int(rng.integers(20, 200))

        while active and burned.sum() < max_cells:
            new_active: list[tuple[int, int]] = []
            for r, c in active:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n and not burned[nr, nc]:
                        spread_prob = 0.4 * (1.0 - moisture[nr, nc])
                        if rng.random() < spread_prob:
                            burned[nr, nc] = True
                            new_active.append((nr, nc))
            active = new_active

        return burned

    # ── cumulative drought stress ─────────────────────────────────

    def _load_drought_stress(self) -> NDArray[np.float64]:
        store = self._world.rasters
        try:
            if "drought_stress" in store.list_layers(TIER):
                return store.read_layer(TIER, "drought_stress").copy()
        except Exception:
            pass
        return np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float64)

    def _get_base_precipitation(self) -> NDArray[np.float64]:
        """Return the base climate precipitation (terrain-modulated, no weather anomaly).

        This is the precipitation field written by ClimateHydrologyTier at world
        creation — the long-term average that weather anomalies perturb around.
        Cached after first read.
        """
        if not hasattr(self, "_base_precip_cache"):
            store = self._world.rasters
            if "precipitation" in store.list_layers("climate_hydrology"):
                self._base_precip_cache = store.read_layer(
                    "climate_hydrology", "precipitation"
                ).copy()
            else:
                # Fallback: use a uniform baseline
                self._base_precip_cache = np.full(
                    (self.GRID_SIZE, self.GRID_SIZE), 800.0, dtype=np.float64
                )
        return self._base_precip_cache

    def _update_cumulative_drought(self, weather: SeasonalWeather) -> None:
        drought = self._load_drought_stress()
        if weather.season == 2:  # Summer
            # Compare current precipitation to the fixed base climate envelope.
            # This makes drought stress accumulate during genuinely dry epochs
            # rather than self-cancelling against the shifted mean.
            base_precip = self._get_base_precipitation()
            # Relative deficit: how far below the base each cell is (0 if above)
            deficit = np.clip(1.0 - weather.precipitation / (base_precip + 1e-10), 0, None)
            drought = drought * 0.7 + deficit * 0.3
        else:
            drought *= 0.9  # slow recovery
        drought = np.clip(drought, 0.0, 1.0)
        self._world.rasters.write_layer(
            TIER, "drought_stress", drought.astype(np.float64),
            self._world.tier_clocks[TIER].tick_number,
        )

    # ── biomass age tracking ──────────────────────────────────────

    def _load_biomass_age(self) -> dict[str, NDArray[np.float64]]:
        store = self._world.rasters
        ages: dict[str, NDArray[np.float64]] = {}
        ecology_layers = store.list_layers(TIER)
        for sp in self._world.events.list_species():
            sid = sp["species_id"]
            layer = f"biomass_age_{sid}"
            if layer in ecology_layers:
                ages[sid] = store.read_layer(TIER, layer).copy()
        return ages

    def _update_biomass_age(self, species_list: list, densities: dict) -> None:
        store = self._world.rasters
        tick_num = self._world.tier_clocks[TIER].tick_number
        ecology_layers = store.list_layers(TIER)

        for sp in species_list:
            sid = sp["species_id"]
            layer = f"biomass_age_{sid}"
            if layer in ecology_layers:
                age = store.read_layer(TIER, layer).copy()
            else:
                age = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float64)

            # Accumulate with density, decay where density is low
            age += densities[sid] * 0.25  # 0.25 years per tick
            age = np.where(densities[sid] < 0.01, age * 0.9, age)  # decay where absent

            store.write_layer(TIER, layer, age.astype(np.float64), tick_num)

    # ── individual lifecycle ────────────────────────────────────────

    def _update_individual_lifecycle(self, tick_num: int) -> None:
        """Age individuals, check for death, transition post-mortem states."""
        rng = create_rng(self._world.seed, "ecology", "lifecycle", tick_num)
        current_year = self._world.tier_clocks[TIER].simulated_year

        # Get all individuals (alive and post-mortem)
        all_individuals = self._world.events.find_individuals_near(
            25000, 25000, 50000  # full world extent
        )

        for ind in all_individuals:
            state = ind.get("state", "alive")

            if state == "alive":
                # Check age-based death
                age = current_year - ind["appeared_year"]
                species_data = self._world.events.get_species(ind["species_id"])
                lifespan = species_data["genome"].get("lifespan", 100.0)

                # Stochastic death: probability increases as age approaches lifespan
                # At lifespan, ~50% chance per tick. At 1.5*lifespan, very high.
                if age > lifespan * 0.7:
                    death_prob = 0.01 * ((age / lifespan) ** 3)
                    death_prob = min(death_prob, 0.5)  # cap at 50% per tick
                    if rng.random() < death_prob:
                        self._world.events.kill_individual(ind["individual_id"], current_year)
                        self._world.events.update_individual_state(ind["individual_id"], "snag")

            elif state == "snag":
                # Snag for ~10 years, then becomes log
                died_year = ind.get("died_year")
                if died_year is not None and (current_year - died_year) > 10:
                    self._world.events.update_individual_state(ind["individual_id"], "log")

            elif state == "log":
                # Log for ~50 years, then becomes mound
                died_year = ind.get("died_year")
                if died_year is not None and (current_year - died_year) > 60:  # 10 snag + 50 log
                    self._world.events.update_individual_state(ind["individual_id"], "mound")

            elif state == "mound":
                # Mound for ~200 years, then removed
                died_year = ind.get("died_year")
                if died_year is not None and (current_year - died_year) > 260:  # 10+50+200
                    self._world.events.update_individual_state(ind["individual_id"], "removed")

    # ── state loading / writing ───────────────────────────────────

    def _load_species_state(
        self, species_list: list,
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
        store = self._world.rasters
        n = self.GRID_SIZE
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

        return densities, seed_banks

    def _write_species_state(
        self,
        species_list: list,
        densities: dict,
        seed_banks: dict,
        tick_num: int,
    ) -> None:
        store = self._world.rasters
        n = self.GRID_SIZE
        seed_bank_total = np.zeros((n, n), dtype=np.float64)

        for sp in species_list:
            sid = sp["species_id"]
            store.write_layer(
                TIER, f"species_{sid}_density",
                densities[sid].astype(np.float64), tick_num,
            )
            sb = seed_banks.get(sid, np.zeros((n, n), dtype=np.float64))
            store.write_layer(
                TIER, f"seed_bank_{sid}", sb.astype(np.float64), tick_num,
            )
            seed_bank_total += sb

        store.write_layer(TIER, "seed_bank_total", seed_bank_total.astype(np.float64), tick_num)

    def _enforce_carrying_capacity(self, densities: dict) -> None:
        """Enforce carrying capacity with niche-overlap competition.

        Species with similar genomes compete harder (higher alpha) than
        species with different genomes.  Each species experiences an
        effective competition load that weights other species' densities
        by their niche overlap.  If a species' effective load exceeds K,
        its density is scaled down.
        """
        n = self.GRID_SIZE
        carrying_capacity = 15.0

        # Precompute genomes and pairwise alphas
        sids = list(densities.keys())
        genomes = {
            sid: self._world.events.get_species(sid)["genome"]
            for sid in sids
        }
        # Cache alpha matrix (symmetric)
        alphas: dict[tuple[str, str], float] = {}
        for i, s1 in enumerate(sids):
            for j, s2 in enumerate(sids):
                if j < i:
                    alphas[(s1, s2)] = alphas[(s2, s1)]
                elif s1 == s2:
                    alphas[(s1, s2)] = 1.0
                else:
                    dist = self._genome_distance(genomes[s1], genomes[s2])
                    alphas[(s1, s2)] = self._competition_alpha(dist)

        # For each species, compute effective competition load
        for sid in sids:
            effective_load = np.zeros((n, n), dtype=np.float64)
            for other_sid in sids:
                alpha = alphas[(sid, other_sid)]
                if alpha > 0.01:  # skip negligible interactions
                    effective_load += alpha * densities[other_sid]

            # Scale down where effective load exceeds K
            excess_mask = effective_load > carrying_capacity
            if excess_mask.any():
                scale = np.where(
                    excess_mask,
                    carrying_capacity / (effective_load + 1e-10),
                    1.0,
                )
                densities[sid] *= scale
                densities[sid] = np.clip(densities[sid], 0.0, None)

    # ── biotic pressure (Janzen-Connell proxy) ─────────────────────

    def _apply_biotic_pressure(
        self,
        species_list: list,
        densities: dict,
        weather: SeasonalWeather,
    ) -> None:
        """Oscillating top-down pressure from pathogens/herbivores/fungi.

        Pressure accumulates when a species (or its close relatives) is abundant,
        and decays when rare.  Applied as density-dependent mortality.
        This creates boom-bust cycles without explicit animal agents.

        Climate responsiveness: warm + wet conditions favour pathogens (faster
        pressure buildup), cold + dry conditions suppress them (faster decay).
        """
        # Load current pressure state
        pressures = self._world.events.get_biotic_pressures()

        # Precompute genome distance between all alive species for shared pressure
        genomes: dict[str, dict] = {}
        for sp in species_list:
            sid = sp["species_id"]
            genomes[sid] = self._world.events.get_species(sid)["genome"]

        # Base parameters — tuned for a 1000x1000 grid where a healthy species
        # has ~10,000-50,000 total density across all cells.
        growth_k = 0.0005  # how fast pressure builds per unit density above baseline
        decay_rate = 0.95  # pressure decays by 5% per tick when species is sparse
        relatedness_threshold = 0.5  # genome distance below which pressure is shared
        max_pressure = 1.0  # cap to prevent runaway
        mortality_strength = 0.08  # max mortality fraction at full pressure

        # Climate modulation — warm + wet = pathogen-friendly
        mean_temp = float(weather.temperature.mean())
        mean_precip = float(weather.precipitation.mean())
        temp_factor = float(np.clip((mean_temp - 2.0) / 16.0, 0, 1))    # 2°C→0, 18°C→1
        precip_factor = float(np.clip(mean_precip / 3000.0, 0, 1))      # 0mm→0, 3000mm→1
        pathogen_favorability = temp_factor * precip_factor               # 0 to 1

        # Modulate: growth_k 50%-150% of base, decay faster when unfavorable
        growth_k = growth_k * (0.5 + pathogen_favorability)
        decay_rate = decay_rate + (1.0 - decay_rate) * 0.5 * (1.0 - pathogen_favorability)

        # Precompute per-species suitability for climate-modulated baseline.
        # In favorable climate, a species can sustain higher density before
        # pathogens build up; in unfavorable climate, the threshold drops
        # and pressure kicks in earlier.  This makes biotic pressure an
        # amplifier of climate signal rather than a suppressor.
        baseline_min = 5000.0
        baseline_max = 20000.0
        species_baselines: dict[str, float] = {}
        for sp in species_list:
            sid = sp["species_id"]
            suit = self._compute_suitability_from_weather(genomes[sid], weather)
            mean_suit = float(suit.mean())
            species_baselines[sid] = baseline_min + (baseline_max - baseline_min) * mean_suit

        # Update pressure for each species
        for sp in species_list:
            sid = sp["species_id"]
            density = densities[sid]
            total_density = float(density.sum())
            baseline_density = species_baselines[sid]

            current_pressure = pressures.get(sid, 0.0)

            # Accumulate or decay based on total density
            if total_density > baseline_density:
                # Pressure grows proportional to excess density
                excess = (total_density - baseline_density) / baseline_density
                current_pressure += growth_k * excess
            else:
                # Pressure decays when species is rare
                current_pressure *= decay_rate

            # Add shared pressure from close relatives (same pathogen pool)
            shared = 0.0
            for other_sp in species_list:
                other_sid = other_sp["species_id"]
                if other_sid == sid:
                    continue
                dist = self._genome_distance(genomes[sid], genomes[other_sid])
                if dist < relatedness_threshold:
                    # Closer relatives share more pressure
                    share_weight = 1.0 - (dist / relatedness_threshold)
                    other_density = float(densities[other_sid].sum())
                    other_baseline = species_baselines.get(other_sid, baseline_max)
                    if other_density > other_baseline:
                        shared += share_weight * growth_k * 0.3 * (
                            (other_density - other_baseline) / other_baseline
                        )
            current_pressure += shared

            # Clamp
            current_pressure = min(current_pressure, max_pressure)
            pressures[sid] = current_pressure

            # Apply mortality from pressure
            if current_pressure > 0.01:
                mortality = mortality_strength * current_pressure
                densities[sid] *= (1.0 - mortality)

        # Persist updated pressures
        self._world.events.set_biotic_pressures(pressures)

    # ── extinction ────────────────────────────────────────────────

    def _check_extinction(
        self,
        species_list: list,
        densities: dict,
        seed_banks: dict,
        tick_num: int,
    ) -> None:
        """Remove species below minimum viable population via demographic stochasticity.

        Species with very low total density face increasing extinction probability.
        This prevents indefinite persistence of near-zero populations.
        """
        rng = create_rng(self._world.seed, "ecology", "extinction", tick_num)
        current_year = self._world.tier_clocks[TIER].simulated_year

        mvp_density = 5.0  # minimum total density across all cells
        mvp_cells = 10  # minimum occupied cells

        to_remove: list[str] = []

        for sp in species_list:
            sid = sp["species_id"]
            density = densities[sid]
            total = float(density.sum())
            occupied = int((density > 0.01).sum())

            # Deterministic extinction: zero density and zero seed bank
            if total <= 0.01 and occupied == 0:
                seed_total = float(seed_banks.get(sid, np.zeros(1)).sum())
                if seed_total <= 0.01:
                    to_remove.append(sid)
                    continue

            if total < mvp_density or occupied < mvp_cells:
                # Extinction probability increases as population shrinks
                if total <= 0:
                    ext_prob = 1.0
                else:
                    ext_prob = 1.0 - (total / mvp_density)
                    ext_prob = float(np.clip(ext_prob, 0.05, 0.9))

                if rng.random() < ext_prob:
                    to_remove.append(sid)

        for sid in to_remove:
            densities[sid][:] = 0.0
            if sid in seed_banks:
                seed_banks[sid][:] = 0.0
            self._world.events.mark_species_extinct(sid, current_year)

    # ── distinguished individuals ─────────────────────────────────

    def _promote_individuals(self, tick_number: int) -> None:
        """Scan density fields and promote prominent plants to named individuals."""
        rng = create_rng(self._world.seed, "ecology", "individuals", tick_number)
        store = self._world.rasters
        current_year = self._world.tier_clocks[TIER].simulated_year
        species_list = self._world.events.list_species(alive_at_year=current_year)

        for sp in species_list:
            sid = sp["species_id"]
            genome = self._world.events.get_species(sid)["genome"]
            growth_form = genome.get("growth_form", 3)
            # Only promote trees (canopy + understory) to DI status.
            # Shrubs/forbs are too small and numerous to be narratively meaningful.
            if growth_form != 0:  # 0 = tree
                continue
            layer_name = f"species_{sid}_density"
            if layer_name not in store.list_layers(TIER):
                continue
            density = store.read_layer(TIER, layer_name)

            # Find cells with high density (above 80th percentile for this species).
            if density.max() <= 0:
                continue
            nonzero = density[density > 0]
            if len(nonzero) == 0:
                continue
            threshold = np.percentile(nonzero, 80)
            candidates = np.argwhere(density >= threshold)

            if len(candidates) == 0:
                continue
            # Promote ~2-5 individuals per species per tick.
            n_promote = min(len(candidates), int(rng.integers(2, 6)))
            chosen = candidates[rng.choice(len(candidates), size=n_promote, replace=False)]

            for row, col in chosen:
                x = float(col * self.CELL_SIZE + self.CELL_SIZE / 2)
                y = float(row * self.CELL_SIZE + self.CELL_SIZE / 2)

                # Avoid crowding: skip if too many individuals already nearby.
                existing = self._world.events.find_individuals_near(x, y, self.CELL_SIZE * 3)
                if len(existing) > 2:
                    continue

                ind_id = f"ind_{sid}_{tick_number}_{row}_{col}"
                self._world.events.add_individual(ind_id, sid, x, y, appeared_year=current_year)

    # ── speciation ────────────────────────────────────────────────

    def _check_speciation(self, tick_number: int, weather: SeasonalWeather | None = None) -> None:
        """Check for population fragmentation and potentially create new species."""
        rng = create_rng(self._world.seed, "ecology", "speciation", tick_number)
        store = self._world.rasters
        current_year = self._world.tier_clocks[TIER].simulated_year
        species_list = self._world.events.list_species(alive_at_year=current_year)

        # Niche saturation: speciation gets harder as niches fill up.
        alive_count = len(species_list)
        saturation_factor = max(0.1, 1.0 - alive_count / 100.0)
        base_speciation_prob = 0.08 * saturation_factor  # lower base prob for ~500yr average

        min_genome_divergence = 0.15  # reject speciation if daughter too similar
        pressure_inheritance = 0.6   # daughter inherits 60% of parent's pressure

        # Role-dependent speciation multiplier:
        # Trees speciate most readily (heavy seeds can't bridge barriers)
        # Herbs/forbs rarely speciate (light seeds maintain gene flow)
        role_speciation_mult = {
            0: 2.0,  # canopy tree — most likely
            1: 1.5,  # understory tree
            2: 1.0,  # shrub — baseline
            3: 0.5,  # herb/forb/grass — light seeds prevent isolation
        }

        # Load pressure state so daughters can inherit parent pressure
        pressures = self._world.events.get_biotic_pressures()
        pressures_changed = False

        for sp in species_list:
            sid = sp["species_id"]

            # Cooldown: species must be at least 300 years old to speciate.
            # Young species need time to establish identity before fragmenting.
            species_age = current_year - sp.get("appeared_year", 0.0)
            if species_age < 300.0:
                continue

            layer_name = f"species_{sid}_density"
            if layer_name not in store.list_layers(TIER):
                continue
            density = store.read_layer(TIER, layer_name).copy()

            # Find occupied cells — species must be substantial to fragment.
            occupied = density > 0.1
            if occupied.sum() < 500:
                continue

            # Connected components on a downsampled grid for performance.
            components = _connected_components_coarse(occupied, downsample=10)
            n_components = int(components.max())

            if n_components < 2:
                continue

            # Find the main (largest) component.
            component_sizes = []
            for c in range(1, n_components + 1):
                component_sizes.append(int((components == c).sum()))
            main_component = int(np.argmax(component_sizes)) + 1

            # Compute suitability on downsampled grid for barrier check.
            # A fragment separated only by hospitable terrain (suitability > 0.2)
            # is not behind a real barrier — dispersal will reconnect it.
            parent_genome = self._world.events.get_species(sid)["genome"]
            if weather is not None:
                suit_full = self._compute_suitability_from_weather(parent_genome, weather)
                ds = 10  # must match downsample used for components
                cr, cc = suit_full.shape[0] // ds, suit_full.shape[1] // ds
                suit_coarse = (
                    suit_full[: cr * ds, : cc * ds]
                    .reshape(cr, ds, cc, ds)
                    .mean(axis=(1, 3))
                )
                hospitable_coarse = suit_coarse > 0.2
                main_mask_coarse = components[::ds, ::ds][:cr, :cc] == main_component
            else:
                hospitable_coarse = None

            for c in range(1, n_components + 1):
                if c == main_component:
                    continue
                fragment_mask = components == c
                fragment_size = int(fragment_mask.sum())
                if fragment_size < 200:
                    continue

                # Barrier check: reject if fragment connects to main through
                # hospitable terrain.  Only real environmental barriers
                # (mountains, drought zones) should allow speciation.
                if hospitable_coarse is not None:
                    frag_coarse = components[::ds, ::ds][:cr, :cc] == c
                    if _fragments_connect_through_hospitable(
                        frag_coarse, main_mask_coarse, hospitable_coarse
                    ):
                        continue  # no barrier — skip this fragment

                # Apply role-dependent probability
                growth_form = parent_genome.get("growth_form", 3)
                if growth_form == 0:  # tree
                    if parent_genome.get("max_height", 0) > 15:
                        role_mult = role_speciation_mult[0]
                    else:
                        role_mult = role_speciation_mult[1]
                elif growth_form == 1:  # shrub
                    role_mult = role_speciation_mult[2]
                else:
                    role_mult = role_speciation_mult[3]

                if rng.random() < base_speciation_prob * role_mult:
                    # ── Adaptive drift: bias toward local environment ──
                    # Compute mean environmental conditions over the fragment
                    local_env: dict[str, float] = {}
                    if weather is not None:
                        p_min = float(weather.precipitation.min())
                        p_max = float(weather.precipitation.max())
                        p_span = p_max - p_min
                        if p_span > 200.0:
                            precip_norm = np.clip((weather.precipitation - p_min) / p_span, 0, 1)
                        else:
                            precip_norm = np.clip(weather.precipitation / 3000.0, 0, 1)
                        local_drought = float((1.0 - precip_norm)[fragment_mask].mean())
                        t_min = float(weather.temperature.min())
                        t_max = float(weather.temperature.max())
                        t_span = t_max - t_min
                        if t_span > 5.0:
                            temp_norm = np.clip((weather.temperature - t_min) / t_span, 0, 1)
                        else:
                            temp_norm = np.clip((weather.temperature - (-10.0)) / 35.0, 0, 1)
                        local_warmth = float(temp_norm[fragment_mask].mean())
                        local_env["drought_tolerance"] = local_drought
                        local_env["frost_tolerance"] = 1.0 - local_warmth

                    adapt_strength = 0.3  # how strongly to bias toward local optimum

                    new_genome: dict = {}
                    for key, val in parent_genome.items():
                        if key == "growth_form":
                            # Discrete: 10% chance of shifting +-1
                            new_val = int(val)
                            if rng.random() < 0.1:
                                new_val += int(rng.choice([-1, 1]))
                            new_genome[key] = int(np.clip(new_val, 0, 4))
                        elif key == "mast_interval":
                            # Integer: +-1 with 20% probability
                            new_val = int(val)
                            if rng.random() < 0.2:
                                new_val += int(rng.choice([-1, 1]))
                            new_genome[key] = int(np.clip(new_val, 1, 7))
                        elif key in _MORPHOLOGICAL_TRAITS:
                            # Much higher variance for visual divergence —
                            # speciation should produce visibly distinct species
                            drift = float(rng.normal(0, 0.25))
                            new_genome[key] = float(np.clip(val + drift, 0.0, 1.0))
                        elif key in _BOUNDED_FUNCTIONAL:
                            # Adaptive drift: bias toward local environment
                            bias = 0.0
                            if key in local_env:
                                bias = (local_env[key] - val) * adapt_strength
                            drift = float(rng.normal(bias, 0.08))
                            new_genome[key] = float(np.clip(val + drift, 0.01, 0.99))
                        else:
                            # Unbounded traits (max_height, lifespan)
                            drift = float(rng.normal(0, 0.1))
                            new_genome[key] = max(0.1, val + val * drift)

                    # Reject speciation if daughter isn't genetically distinct enough.
                    # Geographic separation alone doesn't make a new species —
                    # the fragment must have diverged under different selection pressure.
                    divergence = self._genome_distance(new_genome, parent_genome)
                    if divergence < min_genome_divergence:
                        continue

                    new_id = f"{sid}_d{tick_number}_{c}"
                    self._world.events.add_species(
                        new_id, new_genome, parent_id=sid, appeared_year=current_year
                    )

                    # Daughter inherits a fraction of parent's pathogen pressure.
                    # Speciation doesn't grant a clean escape from the pathogen shadow.
                    parent_pressure = pressures.get(sid, 0.0)
                    if parent_pressure > 0.01:
                        pressures[new_id] = parent_pressure * pressure_inheritance
                        pressures_changed = True

                    # Transfer fragment density to new species.
                    new_density = np.where(fragment_mask, density, 0.0)
                    density = np.where(fragment_mask, 0.0, density)

                    store.write_layer(
                        TIER,
                        f"species_{new_id}_density",
                        new_density.astype(np.float64),
                        tick_number,
                    )
                    store.write_layer(
                        TIER,
                        f"seed_bank_{new_id}",
                        np.zeros_like(new_density),
                        tick_number,
                    )

            # Update parent density (fragments removed).
            store.write_layer(TIER, layer_name, density.astype(np.float64), tick_number)

        # Persist inherited pressure for any new daughter species.
        if pressures_changed:
            self._world.events.set_biotic_pressures(pressures)

    # ── gene flow / reabsorption ─────────────────────────────────

    def _check_reabsorption(self, tick_number: int, weather: SeasonalWeather | None = None) -> None:
        """Merge closely-related species that overlap without a barrier.

        This is the inverse of speciation: when geographic isolation dissolves
        and two similar species come back into contact, the smaller population
        is absorbed into the larger one with gene flow.
        """
        store = self._world.rasters
        current_year = self._world.tier_clocks[TIER].simulated_year
        species_list = self._world.events.list_species(alive_at_year=current_year)

        if len(species_list) < 2:
            return

        max_reabsorption_distance = 0.25  # must be below niche_width (0.3)
        min_overlap_fraction = 0.2  # 20% of smaller species' range must overlap
        min_species_age = 300.0  # years — prevent immediate reabsorption after speciation

        # Load genomes and densities for all living species.
        species_data: list[dict] = []
        for sp in species_list:
            sid = sp["species_id"]
            species_age = current_year - sp.get("appeared_year", 0.0)
            if species_age < min_species_age:
                continue
            layer_name = f"species_{sid}_density"
            if layer_name not in store.list_layers(TIER):
                continue
            density = store.read_layer(TIER, layer_name)
            total = float(density.sum())
            if total < 1.0:
                continue
            genome = self._world.events.get_species(sid)["genome"]
            species_data.append({
                "species_id": sid,
                "genome": genome,
                "density": density,
                "total": total,
                "occupied": density > 0.1,
            })

        if len(species_data) < 2:
            return

        # Pre-compute suitability for barrier checks (same logic as speciation).
        hospitable_coarse = None
        ds = 10
        if weather is not None:
            # We need hospitable terrain per-pair, but a general hospitable map
            # (using mean suitability of both species) is a reasonable proxy.
            # Use a fixed moderate genome for the hospitable check.
            p_min = float(weather.precipitation.min())
            p_max = float(weather.precipitation.max())
            p_span = p_max - p_min
            if p_span > 200.0:
                suit_full = np.clip((weather.precipitation - p_min) / p_span, 0, 1)
            else:
                suit_full = np.clip(weather.precipitation / 3000.0, 0, 1)
            cr, cc = suit_full.shape[0] // ds, suit_full.shape[1] // ds
            # Use actual suitability: a cell is hospitable if either species
            # could survive there.  We'll compute per-pair below.

        # Find candidate pairs (sorted by genome distance, closest first).
        absorbed: set[str] = set()  # track already-absorbed species this tick
        pressures = self._world.events.get_biotic_pressures()
        pressures_changed = False

        for i in range(len(species_data)):
            if species_data[i]["species_id"] in absorbed:
                continue
            for j in range(i + 1, len(species_data)):
                if species_data[j]["species_id"] in absorbed:
                    continue

                sp_a = species_data[i]
                sp_b = species_data[j]
                dist = self._genome_distance(sp_a["genome"], sp_b["genome"])

                if dist >= max_reabsorption_distance:
                    continue

                # Determine which is larger (absorber) and smaller (absorbed).
                if sp_a["total"] >= sp_b["total"]:
                    absorber, absorbed_sp = sp_a, sp_b
                else:
                    absorber, absorbed_sp = sp_b, sp_a

                # Check spatial overlap: fraction of smaller species' range
                # that overlaps with the larger.
                overlap = np.logical_and(absorber["occupied"], absorbed_sp["occupied"])
                absorbed_cells = int(absorbed_sp["occupied"].sum())
                if absorbed_cells == 0:
                    continue
                overlap_fraction = float(overlap.sum()) / absorbed_cells
                if overlap_fraction < min_overlap_fraction:
                    continue

                # Barrier check: can the two populations reach each other
                # through hospitable terrain?  If a barrier separates them,
                # they stay distinct (geographic isolation persists).
                if weather is not None:
                    # Compute suitability for the absorber species on coarse grid.
                    suit_full = self._compute_suitability_from_weather(
                        absorber["genome"], weather
                    )
                    cr, cc = suit_full.shape[0] // ds, suit_full.shape[1] // ds
                    suit_coarse = (
                        suit_full[: cr * ds, : cc * ds]
                        .reshape(cr, ds, cc, ds)
                        .mean(axis=(1, 3))
                    )
                    hospitable_coarse = suit_coarse > 0.2

                    # Treat absorber's range as "main" and absorbed's as "fragment".
                    absorber_coarse = absorber["occupied"][::ds, ::ds][:cr, :cc]
                    absorbed_coarse = absorbed_sp["occupied"][::ds, ::ds][:cr, :cc]

                    if not _fragments_connect_through_hospitable(
                        absorbed_coarse, absorber_coarse, hospitable_coarse
                    ):
                        continue  # barrier still separates them

                # ── Perform reabsorption ──
                absorber_sid = absorber["species_id"]
                absorbed_sid = absorbed_sp["species_id"]

                # Gene flow: shift absorber genome toward absorbed, weighted
                # by population ratio.  Small absorbed population = small shift.
                pop_ratio = absorbed_sp["total"] / (absorber["total"] + absorbed_sp["total"])
                gene_flow_weight = pop_ratio * 0.5  # dampen to prevent large jumps
                updated_genome = {}
                for key, val in absorber["genome"].items():
                    absorbed_val = absorbed_sp["genome"].get(key, val)
                    if key in ("growth_form", "mast_interval"):
                        # Discrete traits: keep absorber's value
                        updated_genome[key] = val
                    else:
                        # Continuous traits: weighted average
                        updated_genome[key] = val + (absorbed_val - val) * gene_flow_weight
                self._world.events.update_species_genome(absorber_sid, updated_genome)

                # Merge densities: add absorbed density to absorber.
                absorber_layer = f"species_{absorber_sid}_density"
                absorbed_layer = f"species_{absorbed_sid}_density"
                absorber_density = store.read_layer(TIER, absorber_layer).copy()
                absorbed_density = store.read_layer(TIER, absorbed_layer).copy()
                merged_density = absorber_density + absorbed_density
                store.write_layer(TIER, absorber_layer, merged_density.astype(np.float64), tick_number)

                # Zero out absorbed species' density and seed bank.
                store.write_layer(TIER, absorbed_layer, np.zeros_like(absorbed_density), tick_number)
                sb_layer = f"seed_bank_{absorbed_sid}"
                if sb_layer in store.list_layers(TIER):
                    store.write_layer(TIER, sb_layer, np.zeros_like(absorbed_density), tick_number)

                # Merge biotic pressure (weighted average).
                p_absorber = pressures.get(absorber_sid, 0.0)
                p_absorbed = pressures.get(absorbed_sid, 0.0)
                pressures[absorber_sid] = p_absorber + (p_absorbed - p_absorber) * pop_ratio
                if absorbed_sid in pressures:
                    del pressures[absorbed_sid]
                pressures_changed = True

                # Mark absorbed species extinct.
                self._world.events.mark_species_extinct(absorbed_sid, current_year)
                absorbed.add(absorbed_sid)

                # Update absorber's cached data for subsequent pair checks.
                absorber["density"] = merged_density
                absorber["total"] = float(merged_density.sum())
                absorber["occupied"] = merged_density > 0.1
                absorber["genome"] = updated_genome

        if pressures_changed:
            self._world.events.set_biotic_pressures(pressures)
