"""Ecology simulation tier — species establishment, population dynamics, and niche differentiation.

This is the third and topmost tier in the three-tier simulation stack, reading
climate-hydrology derived state and producing per-species density fields and
a seed bank.  Phase 6 implements basic ecology: ancestor species creation,
suitability-driven establishment, logistic growth with competition, simple
dispersal, and a persistent seed bank.  Phase 7 adds distinguished individuals,
speciation via population fragmentation, and disturbance regimes (fire, blowdown).

All randomness flows through ``create_rng`` with tier_id="ecology" and
distinct pass_ids, ensuring full reproducibility from the world seed.

The tier operates at 5 years per tick.  Each tick reads the climate-hydrology
cache, computes per-species suitability surfaces, runs population dynamics,
applies disturbance events, promotes distinguished individuals, and periodically
checks for speciation.  Results are written as density layers plus a combined
seed-bank layer to the "ecology" namespace in the RasterStore.
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
    """Ecology tier: ancestor species, suitability, competition, seed bank,
    distinguished individuals, speciation, and disturbance."""

    GRID_SIZE: int = 1000
    YEARS_PER_TICK: int = 5
    NUM_ANCESTORS: int = 6  # within the test-required 5-8 range
    CELL_SIZE: float = 50.0  # meters per grid cell (50 km / 1000 cells)

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

        # Phase 7: Disturbance (fire, blowdown) — skip on first tick.
        if tick_num > 0:
            self._run_disturbance(tick_num)

        # Phase 7: Distinguished individuals.
        self._promote_individuals(tick_num)

        # Phase 7: Speciation (check every 10 ticks, starting at tick 10).
        if tick_num > 0 and tick_num % 10 == 0:
            self._check_speciation(tick_num)

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

    # ── Phase 7: distinguished individuals ──────────────────────────

    def _promote_individuals(self, tick_number: int) -> None:
        """Scan density fields and promote prominent plants to named individuals."""
        rng = create_rng(self._world.seed, "ecology", "individuals", tick_number)
        store = self._world.rasters
        species_list = self._world.events.list_species()
        current_year = self._world.tier_clocks[TIER].simulated_year

        for sp in species_list:
            sid = sp["species_id"]
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

    # ── Phase 7: disturbance ──────────────────────────────────────

    def _run_disturbance(self, tick_number: int) -> None:
        """Run stochastic disturbance events: fire and blowdown."""
        rng = create_rng(self._world.seed, "ecology", "disturbance", tick_number)
        store = self._world.rasters
        current_year = self._world.tier_clocks[TIER].simulated_year

        # Read climate data for fire probability.
        moisture = store.read_layer("climate_hydrology", "soil_moisture_summer")
        eroded_hm = store.read_layer("climate_hydrology", "eroded_heightmap")

        # === Fire ===
        n_fires = int(rng.poisson(1.5))
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
                    "fire",
                    x,
                    y,
                    current_year,
                    radius=radius,
                    data={"cells_burned": int(burned.sum()), "tick": tick_number},
                )

                # Kill density in burned cells.
                for sp in self._world.events.list_species():
                    sid = sp["species_id"]
                    layer_name = f"species_{sid}_density"
                    if layer_name not in store.list_layers(TIER):
                        continue
                    density = store.read_layer(TIER, layer_name)
                    kill_fraction = float(rng.uniform(0.7, 0.95))
                    density = np.where(burned, density * (1 - kill_fraction), density)
                    store.write_layer(TIER, layer_name, density.astype(np.float64), tick_number)

        # === Blowdown ===
        n_blowdown = int(rng.poisson(0.3))
        for _ in range(n_blowdown):
            elev_norm = eroded_hm / (eroded_hm.max() + 1e-10)
            exposure = elev_norm.ravel()
            total = exposure.sum()
            if total <= 0:
                continue
            probs = exposure / total
            idx = int(rng.choice(len(probs), p=probs))
            bd_row, bd_col = divmod(idx, self.GRID_SIZE)

            patch_radius = int(rng.integers(5, 15))
            y_coords, x_coords = np.ogrid[: self.GRID_SIZE, : self.GRID_SIZE]
            dist = np.sqrt((y_coords - bd_row) ** 2 + (x_coords - bd_col) ** 2).astype(np.float64)
            affected = dist <= patch_radius

            if affected.sum() > 0:
                x = float(bd_col * self.CELL_SIZE + self.CELL_SIZE / 2)
                y = float(bd_row * self.CELL_SIZE + self.CELL_SIZE / 2)
                self._world.events.add_event(
                    "blowdown",
                    x,
                    y,
                    current_year,
                    radius=float(patch_radius * self.CELL_SIZE),
                    data={"cells_affected": int(affected.sum()), "tick": tick_number},
                )

                for sp in self._world.events.list_species():
                    sid = sp["species_id"]
                    layer_name = f"species_{sid}_density"
                    if layer_name not in store.list_layers(TIER):
                        continue
                    density = store.read_layer(TIER, layer_name)
                    kill_fraction = float(rng.uniform(0.5, 0.9))
                    density = np.where(affected, density * (1 - kill_fraction), density)
                    store.write_layer(TIER, layer_name, density.astype(np.float64), tick_number)

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

    # ── Phase 7: speciation ───────────────────────────────────────

    def _check_speciation(self, tick_number: int) -> None:
        """Check for population fragmentation and potentially create new species."""
        rng = create_rng(self._world.seed, "ecology", "speciation", tick_number)
        store = self._world.rasters
        species_list = self._world.events.list_species()
        current_year = self._world.tier_clocks[TIER].simulated_year

        for sp in species_list:
            sid = sp["species_id"]
            layer_name = f"species_{sid}_density"
            if layer_name not in store.list_layers(TIER):
                continue
            density = store.read_layer(TIER, layer_name).copy()

            # Find occupied cells.
            occupied = density > 0.1
            if occupied.sum() < 100:
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

            for c in range(1, n_components + 1):
                if c == main_component:
                    continue
                fragment_mask = components == c
                fragment_size = int(fragment_mask.sum())
                if fragment_size < 50:
                    continue

                if rng.random() < 0.3:
                    parent_genome = self._world.events.get_species(sid)["genome"]
                    new_genome: dict = {}
                    for key, val in parent_genome.items():
                        if isinstance(val, float) and val <= 1.0:
                            drift = float(rng.normal(0, 0.08))
                            new_genome[key] = float(np.clip(val + drift, 0.01, 0.99))
                        else:
                            drift = float(rng.normal(0, 0.1))
                            new_genome[key] = max(0.1, val + val * drift)

                    new_id = f"{sid}_d{tick_number}_{c}"
                    self._world.events.add_species(
                        new_id, new_genome, parent_id=sid, appeared_year=current_year
                    )

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

    # ── helpers ────────────────────────────────────────────────────

    def _load_climate_cache(self) -> dict[str, NDArray[np.float64]]:
        store = self._world.rasters
        cache: dict[str, NDArray[np.float64]] = {}
        for name in _CLIMATE_LAYERS:
            cache[name] = store.read_layer("climate_hydrology", name)
        return cache
