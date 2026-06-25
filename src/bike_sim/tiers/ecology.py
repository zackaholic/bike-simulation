"""Ecology simulation tier — species dynamics, disturbance, and niche differentiation.

This is the third and topmost tier in the three-tier simulation stack.  It
receives a ``SeasonalWeather`` object each tick and produces per-species density
fields and disturbance events.

The tier operates at **0.25 years per tick** (one season).  Every tick runs the
same rule: compute suitability, grow, die at a fixed base rate, compete via
Lotka-Volterra alpha matrix, and disperse.  Seasonal variation comes from
weather inputs, not from season-specific code paths.

Fire disturbance runs in summer (season 2) and blowdown in fall (season 3).
Individual promotion runs annually (every 4 ticks).

All randomness flows through ``create_rng`` with tier_id="ecology" and
distinct pass_ids, ensuring full reproducibility from the world seed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve

from bike_sim.rng import create_rng
from bike_sim.weather import SeasonalWeather
from bike_sim.world import World

TIER = "ecology"

# Absolute reference ranges for suitability normalization. These are LOCKED to
# our world's actual climate envelope (seed-42 archetype) measured across a full
# climate cycle (precip 344-2658mm, temp -5.4 to 21.1°C across 600yr x 4 seasons).
# Fitting the references to the achievable envelope maps the genome distribution
# onto conditions the world actually visits, so every species reaches strong
# suitability at some phase of the cycle (wet specialists at the wet peak, dry at
# the trough) rather than being permanently marginal. Fixed (not per-tick) so
# temporal climate signal is preserved.
PRECIP_REF_MIN = 350.0   # mm  (driest the world gets)
PRECIP_REF_MAX = 2650.0  # mm  (wettest the world gets)
TEMP_REF_MIN = -10.0     # °C
TEMP_REF_MAX = 30.0      # °C

# Baseline competition: how much functionally dissimilar species compete for
# shared physical space. 0.0 = only similar species compete (uniform soup,
# no biome structure); 1.0 = all species compete fully regardless of niche
# (risk of global competitive exclusion / monoculture). This is the primary
# knob for biome differentiation — tune via the equilibrium test.
COMPETITION_BASELINE = 0.4

# Core genome traits (6 traits, each maps to exactly one mechanism).
_CORE_TRAITS = [
    "drought_tolerance", "frost_tolerance", "growth_rate",
    "max_height", "lifespan", "dispersal_range",
]

# Archetype trait templates: (name, genome dict before perturbation).
# Organized by structural role for broad world coverage:
#   Canopy trees (3): slow, tall, heavy seeds
#   Understory trees (2): faster, shorter, moderate seeds
#   Tall shrubs (4): moderate height, wide tolerance ranges
#   Low shrubs/forbs (5): fast, short, light seeds
_ANCESTOR_TEMPLATES: list[tuple[str, dict]] = [
    # ── Canopy trees ──────────────────────────────────────────────
    (
        "valley_hardwood",  # warm, wet lowlands — deciduous canopy tree
        {
            "drought_tolerance": 0.25,
            "frost_tolerance": 0.2,
            "growth_rate": 0.2,
            "max_height": 30.0,
            "lifespan": 400.0,
            "dispersal_range": 2,  # from seed_mass 0.8: 2 + int(4 * 0.2) = 2
        },
    ),
    (
        "upland_conifer",  # cool, moderate moisture — evergreen canopy tree
        {
            "drought_tolerance": 0.5,
            "frost_tolerance": 0.7,
            "growth_rate": 0.15,
            "max_height": 25.0,
            "lifespan": 500.0,
            "dispersal_range": 3,  # from seed_mass 0.6: 2 + int(4 * 0.4) = 3
        },
    ),
    (
        "wet_broadleaf",  # wet, mild — tall broadleaf in moist areas
        {
            "drought_tolerance": 0.15,
            "frost_tolerance": 0.3,
            "growth_rate": 0.25,
            "max_height": 28.0,
            "lifespan": 350.0,
            "dispersal_range": 3,  # from seed_mass 0.7: 2 + int(4 * 0.3) = 3
        },
    ),
    # ── Understory / edge trees ───────────────────────────────────
    (
        "gap_filler",  # fast-growing edge tree, colonizes clearings
        {
            "drought_tolerance": 0.35,
            "frost_tolerance": 0.35,
            "growth_rate": 0.5,
            "max_height": 12.0,
            "lifespan": 80.0,
            "dispersal_range": 4,  # from seed_mass 0.4: 2 + int(4 * 0.6) = 4
        },
    ),
    (
        "riparian_tree",  # moisture-loving streamside tree
        {
            "drought_tolerance": 0.1,
            "frost_tolerance": 0.4,
            "growth_rate": 0.45,
            "max_height": 15.0,
            "lifespan": 100.0,
            "dispersal_range": 4,  # from seed_mass 0.3: 2 + int(4 * 0.7) = 4
        },
    ),
    # ── Tall shrubs ───────────────────────────────────────────────
    (
        "ridge_scrub",  # drought-hardy exposed ridgeline shrub
        {
            "drought_tolerance": 0.8,
            "frost_tolerance": 0.5,
            "growth_rate": 0.3,
            "max_height": 3.0,
            "lifespan": 60.0,
            "dispersal_range": 4,  # from seed_mass 0.35: 2 + int(4 * 0.65) = 4
        },
    ),
    (
        "valley_thicket",  # wet valley shrub, forms dense stands
        {
            "drought_tolerance": 0.2,
            "frost_tolerance": 0.3,
            "growth_rate": 0.4,
            "max_height": 4.0,
            "lifespan": 40.0,
            "dispersal_range": 4,  # from seed_mass 0.3: 2 + int(4 * 0.7) = 4
        },
    ),
    (
        "heath_shrub",  # cold-tolerant moorland/heathland shrub
        {
            "drought_tolerance": 0.55,
            "frost_tolerance": 0.7,
            "growth_rate": 0.25,
            "max_height": 1.5,
            "lifespan": 50.0,
            "dispersal_range": 5,  # from seed_mass 0.2: 2 + int(4 * 0.8) = 5
        },
    ),
    (
        "dry_scrub",  # arid-adapted shrub
        {
            "drought_tolerance": 0.9,
            "frost_tolerance": 0.4,
            "growth_rate": 0.2,
            "max_height": 2.0,
            "lifespan": 70.0,
            "dispersal_range": 5,  # from seed_mass 0.25: 2 + int(4 * 0.75) = 5
        },
    ),
    # ── Low shrubs / forbs / ground-hugging ───────────────────────
    (
        "meadow_herb",  # mesic meadow flowering herb
        {
            "drought_tolerance": 0.3,
            "frost_tolerance": 0.3,
            "growth_rate": 0.8,
            "max_height": 0.5,
            "lifespan": 10.0,
            "dispersal_range": 5,  # from seed_mass 0.1: 2 + int(4 * 0.9) = 5
        },
    ),
    (
        "upland_grass",  # cold-tolerant grassland species
        {
            "drought_tolerance": 0.45,
            "frost_tolerance": 0.75,
            "growth_rate": 0.6,
            "max_height": 0.8,
            "lifespan": 15.0,
            "dispersal_range": 5,  # from seed_mass 0.15: 2 + int(4 * 0.85) = 5
        },
    ),
    (
        "pioneer_forb",  # fast-growing disturbance colonizer
        {
            "drought_tolerance": 0.35,
            "frost_tolerance": 0.35,
            "growth_rate": 0.95,
            "max_height": 0.3,
            "lifespan": 5.0,
            "dispersal_range": 5,  # from seed_mass 0.05: 2 + int(4 * 0.95) = 5
        },
    ),
    (
        "alpine_cushion",  # high-altitude stress-tolerator
        {
            "drought_tolerance": 0.65,
            "frost_tolerance": 0.85,
            "growth_rate": 0.1,
            "max_height": 0.1,
            "lifespan": 80.0,
            "dispersal_range": 5,  # from seed_mass 0.2: 2 + int(4 * 0.8) = 5
        },
    ),
    (
        "dry_grass",  # drought-adapted grassland species
        {
            "drought_tolerance": 0.75,
            "frost_tolerance": 0.4,
            "growth_rate": 0.5,
            "max_height": 0.6,
            "lifespan": 8.0,
            "dispersal_range": 5,  # from seed_mass 0.1: 2 + int(4 * 0.9) = 5
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


_DISPERSAL_KERNELS: dict[int, NDArray[np.float64]] = {}


def _dispersal_kernel(radius: int) -> NDArray[np.float64]:
    """Distance-weighted, sum-normalised dispersal kernel for *radius*.

    Cached per radius — the kernel depends only on radius, not on density,
    so it is built once and reused across species and ticks.
    """
    cached = _DISPERSAL_KERNELS.get(radius)
    if cached is not None:
        return cached
    kernel_size = 2 * radius + 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    center = radius
    for i in range(kernel_size):
        for j in range(kernel_size):
            dist = np.sqrt(float((i - center) ** 2 + (j - center) ** 2))
            if dist <= radius:
                kernel[i, j] = 1.0 / (1.0 + dist)
    kernel /= kernel.sum()
    _DISPERSAL_KERNELS[radius] = kernel
    return kernel


def _disperse(
    density: NDArray[np.float64],
    radius: int = 1,
) -> NDArray[np.float64]:
    """Spread *density* to neighbours using a distance-weighted kernel.

    Equivalent to correlating *density* with a radial kernel and zero-padding
    the borders. Implemented via ``scipy.ndimage.convolve`` (optimised C) — the
    kernel is symmetric, so convolution and correlation coincide.
    """
    kernel = _dispersal_kernel(radius)
    return convolve(density, kernel, mode="constant", cval=0.0)


# ---------------------------------------------------------------------------
# EcologyTier
# ---------------------------------------------------------------------------


class EcologyTier:
    """Ecology tier: species dynamics, disturbance, and niche differentiation."""

    GRID_SIZE: int = 1000
    YEARS_PER_TICK: float = 0.25
    TICKS_PER_YEAR: int = 4
    NUM_ANCESTORS: int = 14  # 3 canopy + 2 understory + 4 shrub + 5 forb/grass
    CELL_SIZE: float = 50.0  # meters per grid cell (50 km / 1000 cells)

    def __init__(self, world: World) -> None:
        self._world = world
        # Distinguished-individual promotion + lifecycle. Not part of the core
        # grow/compete/disperse dynamics under test, adds SQLite overhead, and
        # keys DI ids on tick number (which collides if ticks are ever re-run).
        # Tests validating density dynamics disable it.
        self.enable_individuals = True
        # Refugium floor: minimum total density a species retains, seeded into its
        # single most-suitable cell so it can wait out unfavorable phases and
        # rebound when conditions return. 0.0 = off. This is what lets wet
        # specialists survive dry centuries (and vice versa) for full cyclic swings.
        self.refugium_floor = 0.0
        # Allee establishment threshold. When > 0, positive (colonizing) growth
        # is scaled by density/(density+allee_theta), so near-empty cells cannot
        # self-bootstrap from a stray propagule — they must be recolonized by
        # sustained dispersal from a real source. This makes disturbance scars
        # refill slowly from their edges (and lets some persist), and is what
        # makes priority/incumbency effects produce *lasting* idiosyncratic
        # patches. 0.0 = off (no positive density dependence; current behavior).
        self.allee_theta = 0.0
        # Incumbency advantage: asymmetric competition where the locally
        # dominant species (higher density) exerts more competitive pressure
        # on invaders. At strength 0.3, the incumbent gets a ±30% edge.
        # Combined with Allee (B), this produces path-dependent post-
        # disturbance communities — the first species to recolonize a scar
        # holds it. 0.0 = off (symmetric competition; current behavior).
        self.incumbency_strength = 0.0
        # Per-species mechanism multipliers, identity by default. Maps
        # species_id -> {mechanism: multiplier} where mechanism is one of
        # "growth", "mortality", "dispersal", "carrying_capacity". The tier
        # itself knows nothing about *why* a rate is scaled (blight, plague,
        # fertility boost) — callers (e.g. the test harness) push plain
        # multipliers for whatever window/targeting they want. Empty dict means
        # no effect, so normal runs and tests are unchanged.
        self.mechanism_modifiers: dict[str, dict[str, float]] = {}

    # ── public API ─────────────────────────────────────────────────

    def set_mechanism_modifiers(
        self, modifiers: dict[str, dict[str, float]]
    ) -> None:
        """Replace the active per-species mechanism multipliers.

        ``modifiers`` maps species_id -> {mechanism: multiplier}. Mechanisms not
        listed default to 1.0 (no effect). Pass ``{}`` to clear all modifiers.
        """
        self.mechanism_modifiers = modifiers

    def _modifier(self, sid: str, mechanism: str) -> float:
        """Return the active multiplier for *mechanism* on species *sid* (default 1.0)."""
        return self.mechanism_modifiers.get(sid, {}).get(mechanism, 1.0)

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
        densities = self._load_species_state(species_list)

        # Core ecology: growth, competition, mortality
        self._grow_and_compete(weather, species_list, densities)

        # Dispersal every tick
        self._disperse_all(species_list, densities, tick_num)

        # Apply density floor
        for sid in densities:
            densities[sid] = np.where(densities[sid] < 0.001, 0.0, densities[sid])

        # Fire disturbance (summer only)
        if season == 2:
            self._fire_disturbance(weather, species_list, densities, tick_num)

        # Blowdown (fall only)
        if season == 3:
            self._blowdown_disturbance(weather, species_list, densities, tick_num)

        # Refugium floor: keep a trace of each species alive in its best cell so
        # it can rebound when its favorable phase returns.
        if self.refugium_floor > 0.0:
            self._apply_refugium_floor(weather, species_list, densities)

        # Write all state
        self._write_species_state(species_list, densities, tick_num)

        if self.enable_individuals:
            # Individual lifecycle (aging, death, post-mortem transitions)
            self._update_individual_lifecycle(tick_num)

            # Promote individuals every 4 ticks (annually)
            if tick_num % 4 == 0:
                self._promote_individuals(tick_num)

        clock.tick_number += 1
        clock.simulated_year += self.YEARS_PER_TICK

    # ── ancestor creation ──────────────────────────────────────────

    def _create_ancestors(self, tick_number: int) -> None:
        rng = create_rng(self._world.seed, "ecology", "ancestors", tick_number)

        # Redistribute the moisture niche onto the world's achievable climate
        # manifold. This world's temperature and precipitation are correlated
        # (here anticorrelated ~-0.5: warm lowlands are dry, cold uplands wet),
        # so the warm-AND-wet (and cold-AND-dry) corners never occur. Archetypes
        # placed there by hand strand at the refugium floor forever. We keep each
        # archetype's temperature identity (frost_tolerance) and structure, and
        # derive drought_tolerance by *conditioning on the realized climate*:
        # for the archetype's warmth, use the moisture the world actually pairs
        # with that temperature (mean drought_stress of nearby cells), plus
        # jitter scaled to the local spread. That guarantees every niche lands
        # on the manifold rather than in an unreachable corner.
        niche_cloud = self._climate_niche_cloud()
        if niche_cloud is not None:
            tn_cloud, ds_cloud = niche_cloud
            tn_lo, tn_hi = (float(x) for x in np.percentile(tn_cloud, [2, 98]))

        for idx, (name, template) in enumerate(_ANCESTOR_TEMPLATES):
            genome = {}
            for key, val in template.items():
                if key == "dispersal_range":
                    # Integer trait: perturb by rounding after uniform offset.
                    genome["dispersal_range"] = int(
                        np.clip(round(val + rng.uniform(-0.5, 0.5)), 1, 6)
                    )
                else:
                    perturb = rng.uniform(-0.05, 0.05)
                    genome[key] = float(np.clip(val + perturb, 0.0, None))
                    # Keep bounded functional traits in [0, 1]
                    if key in (
                        "drought_tolerance",
                        "frost_tolerance",
                        "growth_rate",
                    ):
                        genome[key] = float(np.clip(genome[key], 0.0, 1.0))

            # Override drought_tolerance by conditioning on the realized climate
            # at this archetype's warmth (see comment above).
            if niche_cloud is not None:
                warmth = float(np.clip(1.0 - genome["frost_tolerance"], tn_lo, tn_hi))
                mask = np.abs(tn_cloud - warmth) < 0.12
                if int(mask.sum()) < 100:
                    mask = np.abs(tn_cloud - warmth) < 0.25
                if int(mask.sum()) < 20:
                    ds_target, ds_std = float(ds_cloud.mean()), float(ds_cloud.std())
                else:
                    ds_target = float(ds_cloud[mask].mean())
                    ds_std = float(ds_cloud[mask].std())
                jitter = rng.uniform(-1.0, 1.0) * min(ds_std, 0.12)
                genome["drought_tolerance"] = float(np.clip(ds_target + jitter, 0.05, 0.95))

            self._world.events.add_species(
                species_id=f"anc_{idx:02d}_{name}",
                genome=genome,
                parent_id=None,
                appeared_year=0.0,
            )

    def _climate_niche_cloud(self) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """Realized (temp_norm, drought_stress) point cloud from current climate.

        Normalizes the current temperature/precipitation rasters onto the same
        axes the suitability function uses, subsampled for speed. Returns
        ``(temp_norm, drought_stress)`` flat arrays, or None if climate layers
        are unavailable (then ancestor moisture niches fall back to templates).
        """
        store = self._world.rasters
        try:
            temp = store.read_layer("climate_hydrology", "temperature").ravel()
            precip = store.read_layer("climate_hydrology", "precipitation").ravel()
        except KeyError:
            return None
        step = max(1, temp.size // 50_000)
        tn = np.clip((temp[::step] - TEMP_REF_MIN) / (TEMP_REF_MAX - TEMP_REF_MIN), 0, 1)
        ds = 1.0 - np.clip(
            (precip[::step] - PRECIP_REF_MIN) / (PRECIP_REF_MAX - PRECIP_REF_MIN), 0, 1
        )
        return tn, ds

    # ── initial populations ────────────────────────────────────────

    def _seed_initial_populations(self, tick_number: int, weather: SeasonalWeather) -> None:
        """Broad noise-based placement: scatter species across suitable habitat.

        Each species gets a unique noise field (from world seed + species index)
        that creates natural clustering (groves, patches). Initial density is
        suitability * noise * height-based max density, zeroed below a threshold.

        This produces a world that is already "full" -- every habitable cell has
        vegetation appropriate to its climate zone.
        """
        from bike_sim.tiers.geology import _bilinear_upsample

        rng = create_rng(self._world.seed, "ecology", "init_pop", tick_number)
        store = self._world.rasters

        # Use annual-mean climate for initial placement, not the current
        # season's weather.  Tick 0 is winter -- warm-loving species would
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

        for idx, sp in enumerate(self._world.events.list_species()):
            sid = sp["species_id"]
            genome = self._world.events.get_species(sid)["genome"]
            suit = self._compute_suitability(genome, annual_weather)

            # Height-based max density and noise scale:
            # tall trees are sparse with large patches, short plants are dense and dispersed
            max_height = genome.get("max_height", 1.0)
            if max_height > 15:
                max_density = 1.5
                sz1, sz2 = 5, 10
            elif max_height > 5:
                max_density = 2.5
                sz1, sz2 = 6, 12
            elif max_height > 1.5:
                max_density = 3.0
                sz1, sz2 = 8, 16
            else:
                max_density = 4.0
                sz1, sz2 = 10, 20

            # Generate species-specific clustering noise
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

    # ── suitability ───────────────────────────────────────────────

    def _compute_suitability(
        self,
        genome: dict,
        weather: SeasonalWeather,
    ) -> NDArray[np.float64]:
        """Absolute-normalized suitability from weather and genome traits."""
        # Moisture axis: two-parameter affine map onto the world's precip envelope.
        precip_norm = np.clip(
            (weather.precipitation - PRECIP_REF_MIN) / (PRECIP_REF_MAX - PRECIP_REF_MIN),
            0, 1,
        )
        drought_stress = 1.0 - precip_norm
        suit = _gaussian_match(drought_stress, genome["drought_tolerance"], sigma=0.25)

        # Temperature axis
        temp_norm = np.clip(
            (weather.temperature - TEMP_REF_MIN) / (TEMP_REF_MAX - TEMP_REF_MIN),
            0, 1,
        )
        warmth_preference = 1.0 - genome["frost_tolerance"]
        suit *= _gaussian_match(temp_norm, warmth_preference, sigma=0.25)

        return suit

    # ── carrying capacity ─────────────────────────────────────────

    def _compute_carrying_capacity(
        self, weather: SeasonalWeather,
    ) -> NDArray[np.float64]:
        """Terrain-varying K: ~5 on dry ridges to ~20 in wet lowlands."""
        moisture = np.clip(weather.precipitation / PRECIP_REF_MAX, 0, 1)
        # Try geology heightmap, fall back to eroded_heightmap or uniform
        store = self._world.rasters
        try:
            elevation = store.read_layer("geology", "heightmap")
        except KeyError:
            try:
                elevation = store.read_layer("climate_hydrology", "eroded_heightmap")
            except KeyError:
                elevation = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float64)
        elev_norm = elevation / (elevation.max() + 1e-10)
        # Higher moisture, lower elevation = higher K
        K = 5.0 + 15.0 * moisture * (1.0 - elev_norm * 0.5)
        return K

    # ── core ecology rule ─────────────────────────────────────────

    def _grow_and_compete(
        self,
        weather: SeasonalWeather,
        species_list: list,
        densities: dict,
    ) -> None:
        """Per-tick growth, mortality, and Lotka-Volterra competition."""
        n = self.GRID_SIZE
        K = self._compute_carrying_capacity(weather)

        sids = [sp["species_id"] for sp in species_list]
        genomes = {
            sid: self._world.events.get_species(sid)["genome"]
            for sid in sids
        }

        # Precompute alpha matrix (symmetric)
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

        for sid in sids:
            genome = genomes[sid]
            density = densities[sid]
            suit = self._compute_suitability(genome, weather)

            # Effective competition load (with optional incumbency asymmetry).
            effective_load = np.zeros((n, n), dtype=np.float64)
            for other in sids:
                alpha = alphas[(sid, other)]
                if alpha > 0.01:
                    if self.incumbency_strength > 0.0 and other != sid:
                        # Incumbency: the locally dominant species exerts
                        # more pressure. ratio→1 when other dominates,
                        # →0 when sid dominates, 0.5 when equal.
                        ratio = densities[other] / (
                            densities[other] + density + 1e-10)
                        factor = 1.0 + self.incumbency_strength * (
                            2.0 * ratio - 1.0)
                        effective_load += alpha * densities[other] * factor
                    else:
                        effective_load += alpha * densities[other]

            # Logistic growth with per-species carrying capacity set by suitability.
            # K_eff = K * suit means the species saturates at density K*suit.
            # The (K_eff - load)/K term goes NEGATIVE when overcrowded, providing
            # the negative feedback that corrects overshoot — this is the corrective
            # force that a clipped "available" term would silently discard.
            # growth_rate is per-year; 0.25 factor for seasonal tick.
            k_eff = K * suit * self._modifier(sid, "carrying_capacity")
            logistic = (k_eff - effective_load) / K
            growth = (
                density * genome["growth_rate"] * logistic * 0.25
                * self._modifier(sid, "growth")
            )

            # Allee effect: gate *colonizing* (positive) growth by local density
            # so near-empty cells can't self-bootstrap (negative die-back growth
            # is left untouched). Off when allee_theta == 0.
            if self.allee_theta > 0.0:
                allee = density / (density + self.allee_theta)
                growth = np.where(growth > 0.0, growth * allee, growth)

            # Mortality: fixed base rate from lifespan (per seasonal tick)
            base_turnover = 1.0 / (genome["lifespan"] * float(self.TICKS_PER_YEAR))
            mortality = density * base_turnover * self._modifier(sid, "mortality")

            densities[sid] = np.clip(density + growth - mortality, 0.0, None)

    # ── refugium floor ────────────────────────────────────────────

    def _apply_refugium_floor(
        self,
        weather: SeasonalWeather,
        species_list: list,
        densities: dict,
    ) -> None:
        """Guarantee each species a trace presence in its most-suitable cell.

        If a species' total density falls below ``refugium_floor``, seed
        ``refugium_floor`` density into the single cell where it is currently
        best suited. This prevents a species from hitting unrecoverable zero
        during an unfavorable phase, so it can rebound (and act as a dispersal
        source) when its favorable conditions return.
        """
        for sp in species_list:
            sid = sp["species_id"]
            if float(densities[sid].sum()) >= self.refugium_floor:
                continue
            genome = self._world.events.get_species(sid)["genome"]
            suit = self._compute_suitability(genome, weather)
            best = np.unravel_index(int(np.argmax(suit)), suit.shape)
            densities[sid][best] = max(
                float(densities[sid][best]), self.refugium_floor
            )

    # ── dispersal ─────────────────────────────────────────────────

    def _disperse_all(
        self,
        species_list: list,
        densities: dict,
        tick_num: int,
    ) -> None:
        """Run local and long-distance dispersal for all species every tick."""
        rng = create_rng(self._world.seed, "ecology", "dispersal", tick_num)
        n = self.GRID_SIZE

        for sp in species_list:
            sid = sp["species_id"]
            genome = self._world.events.get_species(sid)["genome"]
            density = densities[sid]

            # Local dispersal
            radius = int(genome.get("dispersal_range", 3))
            spread = _disperse(density, radius)
            # Only deposit a fraction (dispersal, not teleportation)
            deposit_fraction = 0.02 * self._modifier(sid, "dispersal")  # 2% spreads/tick
            densities[sid] = density * (1.0 - deposit_fraction) + spread * deposit_fraction

            # Long-distance dispersal. The dispersal modifier scales only the
            # deposited amount, never the RNG draws, so the stochastic stream is
            # identical whether or not a modifier is active (modifier == 1.0
            # reproduces the unmodified run bit-for-bit).
            disp_mult = self._modifier(sid, "dispersal")
            ldd_prob = 0.1 + 0.4 * (radius / 6.0)  # wider dispersers do more LDD
            if rng.random() < ldd_prob:
                source_cells = np.argwhere(density > 0.5)
                if len(source_cells) > 0:
                    n_jumps = int(rng.integers(3, 10))
                    max_jump = int(n * 0.3)
                    for _ in range(min(n_jumps, len(source_cells))):
                        src_idx = int(rng.integers(0, len(source_cells)))
                        sr, sc = source_cells[src_idx]
                        tr = int(np.clip(sr + rng.integers(-max_jump, max_jump + 1), 0, n - 1))
                        tc = int(np.clip(sc + rng.integers(-max_jump, max_jump + 1), 0, n - 1))
                        densities[sid][tr, tc] += float(density[sr, sc]) * 0.02 * disp_mult

    # ── niche overlap ─────────────────────────────────────────────

    @staticmethod
    def _genome_distance(g1: dict, g2: dict) -> float:
        """Euclidean distance between two genomes in functional trait space."""
        keys = _CORE_TRAITS
        return float(np.sqrt(sum((g1.get(k, 0) - g2.get(k, 0)) ** 2 for k in keys)))

    @staticmethod
    def _competition_alpha(distance: float, niche_width: float = 0.3) -> float:
        """Lotka-Volterra competition coefficient from genome distance.

        alpha = 1.0 when distance = 0 (identical species compete fully).
        alpha -> COMPETITION_BASELINE as distance grows (different species still
        compete for shared physical space — light, water, ground — even when
        functionally dissimilar).

        The baseline is what makes competition a *spatial* limiter: without it,
        a poorly-suited species in a cell ignores the well-suited species that
        should exclude it, and every species survives everywhere (uniform soup).
        With it, the species with the highest local K_eff (= K * suitability)
        suppresses the others, so biome boundaries emerge where the suitability
        ranking flips between species.
        """
        niche_term = np.exp(-(distance / niche_width) ** 2)
        return float(COMPETITION_BASELINE + (1.0 - COMPETITION_BASELINE) * niche_term)

    # ── fire disturbance ──────────────────────────────────────────

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

    # ── blowdown disturbance ──────────────────────────────────────

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
            # Only promote trees (max_height > 10m) to DI status.
            # Shrubs/forbs are too small and numerous to be narratively meaningful.
            if genome.get("max_height", 0) <= 10:
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

    # ── state loading / writing ───────────────────────────────────

    def _load_species_state(
        self, species_list: list,
    ) -> dict[str, NDArray[np.float64]]:
        """Load density arrays for all living species."""
        store = self._world.rasters
        n = self.GRID_SIZE
        ecology_layers = store.list_layers(TIER)
        densities: dict[str, NDArray[np.float64]] = {}

        for sp in species_list:
            sid = sp["species_id"]
            layer = f"species_{sid}_density"
            if layer in ecology_layers:
                densities[sid] = store.read_layer(TIER, layer).copy()
            else:
                densities[sid] = np.zeros((n, n), dtype=np.float64)

        return densities

    def _write_species_state(
        self,
        species_list: list,
        densities: dict,
        tick_num: int,
    ) -> None:
        """Write density arrays for all living species."""
        store = self._world.rasters

        for sp in species_list:
            sid = sp["species_id"]
            store.write_layer(
                TIER, f"species_{sid}_density",
                densities[sid].astype(np.float64), tick_num,
            )
