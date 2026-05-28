"""Synthetic world generator for visual testing.

Generates a fake world populated with plausible-looking data so the debug
visualizer can be tested without a real simulation. Also serves as a
smoke-test and demo: if the visualizer renders this world correctly, the
data pipeline from World -> query -> extract -> render is working.

All randomness uses create_rng for reproducibility from seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bike_sim.rng import create_rng
from bike_sim.world import World

# World constants
GRID_SIZE = 1000  # 1000x1000 cells
WORLD_EXTENT = 50_000.0  # 50km in meters
CELL_SIZE = WORLD_EXTENT / GRID_SIZE  # 50m per cell

# Synthetic tier ID — keeps RNG streams separate from real simulation tiers
TIER = "synthetic"


def _make_heightmap(seed: int) -> np.ndarray:
    """Generate a multi-octave noise heightmap at 1000x1000.

    Sums several scales of random grids upsampled via np.repeat, with
    decreasing weight at finer scales. Result is in meters (roughly 0-2000m).
    """
    rng = create_rng(seed, TIER, "heightmap", tick_number=0)
    heightmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)

    # Octaves: (source_size, weight). Sizes chosen to divide evenly into 1000.
    octaves = [
        (8, 1.0),
        (10, 0.6),
        (20, 0.35),
        (40, 0.18),
        (100, 0.08),
    ]

    for src_size, weight in octaves:
        small = rng.random((src_size, src_size))
        # Upscale to 1000x1000 by repeating (all sizes divide 1000 evenly)
        repeat_factor = GRID_SIZE // src_size
        upscaled = np.repeat(np.repeat(small, repeat_factor, axis=0), repeat_factor, axis=1)
        heightmap += weight * upscaled

    # Normalize to 0-1 then scale to 0-2000m
    heightmap -= heightmap.min()
    heightmap /= heightmap.max()
    heightmap *= 2000.0
    return heightmap


def _make_bedrock_type(seed: int) -> np.ndarray:
    """Generate Voronoi-ish bedrock type regions.

    Places random seed points and assigns each cell to the nearest one,
    giving each region an integer bedrock type (0-5).
    """
    rng = create_rng(seed, TIER, "bedrock", tick_number=0)
    n_seeds = rng.integers(15, 21)  # 15-20 seed points

    # Random seed point positions in cell coordinates
    seed_xs = rng.random(n_seeds) * GRID_SIZE
    seed_ys = rng.random(n_seeds) * GRID_SIZE
    seed_types = rng.integers(0, 6, size=n_seeds)  # bedrock types 0-5

    # Assign each cell to nearest seed point
    ys, xs = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE]
    bedrock = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)

    # Vectorized: compute distance to each seed and pick the closest
    min_dist = np.full((GRID_SIZE, GRID_SIZE), np.inf)
    for i in range(n_seeds):
        dist = (xs - seed_xs[i]) ** 2 + (ys - seed_ys[i]) ** 2
        closer = dist < min_dist
        bedrock[closer] = seed_types[i]
        min_dist[closer] = dist[closer]

    return bedrock


def _make_soil_moisture(seed: int, heightmap: np.ndarray) -> np.ndarray:
    """Derive soil moisture loosely from elevation.

    Lower elevation = more moisture, plus some noise. Normalized to 0-1.
    """
    rng = create_rng(seed, TIER, "soil_moisture", tick_number=0)

    # Invert and normalize elevation contribution
    elev_norm = heightmap / 2000.0  # 0-1
    moisture = 1.0 - elev_norm  # low elevation -> high moisture

    # Add noise
    noise = rng.random((GRID_SIZE, GRID_SIZE)) * 0.3 - 0.15
    moisture += noise

    # Clamp to 0-1
    np.clip(moisture, 0.0, 1.0, out=moisture)
    return moisture


def _make_species_densities(
    seed: int,
    world: World,
    heightmap: np.ndarray,
    moisture: np.ndarray,
) -> list[str]:
    """Create fake species and their density fields.

    Each species prefers a certain elevation/moisture band. Returns a list
    of species IDs that were created.
    """
    rng = create_rng(seed, TIER, "species", tick_number=0)

    species_defs = [
        {
            "id": "sp_lowland_fern",
            "genome": {
                "growth_rate": 0.8,
                "shade_tolerance": 0.9,
                "drought_tolerance": 0.1,
            },
            "elev_center": 300.0,
            "elev_width": 400.0,
            "moist_center": 0.8,
            "moist_width": 0.3,
        },
        {
            "id": "sp_ridge_pine",
            "genome": {
                "growth_rate": 0.3,
                "shade_tolerance": 0.2,
                "drought_tolerance": 0.8,
            },
            "elev_center": 1400.0,
            "elev_width": 500.0,
            "moist_center": 0.3,
            "moist_width": 0.4,
        },
        {
            "id": "sp_valley_oak",
            "genome": {
                "growth_rate": 0.5,
                "shade_tolerance": 0.5,
                "drought_tolerance": 0.5,
            },
            "elev_center": 600.0,
            "elev_width": 600.0,
            "moist_center": 0.6,
            "moist_width": 0.4,
        },
        {
            "id": "sp_alpine_moss",
            "genome": {
                "growth_rate": 0.2,
                "shade_tolerance": 0.7,
                "drought_tolerance": 0.3,
            },
            "elev_center": 1800.0,
            "elev_width": 300.0,
            "moist_center": 0.5,
            "moist_width": 0.5,
        },
    ]

    species_ids = []
    for sp in species_defs:
        # Register species in event store
        world.events.add_species(
            species_id=sp["id"],
            genome=sp["genome"],
            parent_id=None,
            appeared_year=-10000.0,
        )

        # Build density field from Gaussian preferences
        elev_score = np.exp(-0.5 * ((heightmap - sp["elev_center"]) / sp["elev_width"]) ** 2)
        moist_score = np.exp(-0.5 * ((moisture - sp["moist_center"]) / sp["moist_width"]) ** 2)
        density = elev_score * moist_score

        # Add some noise for realism
        noise = rng.random((GRID_SIZE, GRID_SIZE)) * 0.15
        density = np.clip(density + noise - 0.075, 0.0, 1.0)

        # Store as raster
        world.rasters.write_layer("ecology", f"species_{sp['id']}_density", density, tick_number=0)
        species_ids.append(sp["id"])

    return species_ids


def _place_individuals(
    seed: int,
    world: World,
    heightmap: np.ndarray,
    species_ids: list[str],
) -> None:
    """Place 25 distinguished individuals at interesting positions.

    Picks peaks, valleys, and other notable terrain positions.
    """
    rng = create_rng(seed, TIER, "individuals", tick_number=0)

    # Find interesting positions: local extremes
    # Top 10 peaks (highest cells sampled sparsely)
    flat = heightmap.ravel()
    n_individuals = 25

    # Mix of strategies: some at high points, some at low points, some random
    high_indices = np.argsort(flat)[-200:]  # top 200 cells
    low_indices = np.argsort(flat)[:200]  # bottom 200 cells

    positions = []
    # 8 at peaks
    chosen_high = rng.choice(high_indices, size=8, replace=False)
    for idx in chosen_high:
        r, c = divmod(int(idx), GRID_SIZE)
        positions.append((r, c))

    # 8 at valleys
    chosen_low = rng.choice(low_indices, size=8, replace=False)
    for idx in chosen_low:
        r, c = divmod(int(idx), GRID_SIZE)
        positions.append((r, c))

    # 9 random
    for _ in range(n_individuals - len(positions)):
        r = int(rng.integers(0, GRID_SIZE))
        c = int(rng.integers(0, GRID_SIZE))
        positions.append((r, c))

    for i, (r, c) in enumerate(positions):
        # Convert cell to world coordinates (center of cell)
        x = (c + 0.5) * CELL_SIZE
        y = (r + 0.5) * CELL_SIZE
        sp = species_ids[i % len(species_ids)]
        world.events.add_individual(
            individual_id=f"ind_{i:03d}",
            species_id=sp,
            x=x,
            y=y,
            appeared_year=float(rng.integers(-5000, 0)),
        )


def _place_events(seed: int, world: World) -> None:
    """Add 8 fake events (fires, floods, etc.) at random positions and years."""
    rng = create_rng(seed, TIER, "events", tick_number=0)

    event_types = [
        ("fire", {"severity": "high", "cause": "lightning"}),
        ("fire", {"severity": "low", "cause": "lightning"}),
        ("flood", {"peak_flow_m3s": 450.0}),
        ("flood", {"peak_flow_m3s": 120.0}),
        ("fire", {"severity": "medium", "cause": "drought"}),
        ("landslide", {"volume_m3": 5000.0}),
        ("beetle_outbreak", {"species_affected": "sp_ridge_pine"}),
        ("windthrow", {"max_gust_ms": 35.0}),
    ]

    for event_type, data in event_types:
        x = float(rng.random() * WORLD_EXTENT)
        y = float(rng.random() * WORLD_EXTENT)
        year = float(rng.integers(-3000, 0))
        radius = float(rng.random() * 2000 + 200)  # 200-2200m
        world.events.add_event(
            event_type=event_type,
            x=x,
            y=y,
            year=year,
            radius=radius,
            data=data,
        )


def create_synthetic_world(path: Path, seed: int = 42) -> World:
    """Create a world populated with plausible synthetic data for visual testing.

    Generates heightmap, bedrock types, soil moisture, species density fields,
    distinguished individuals, and historical events. All randomness is
    deterministic from the given seed.

    Parameters
    ----------
    path : Path
        Directory to create the world in.
    seed : int
        World seed for reproducible generation.

    Returns
    -------
    World
        A fully populated World ready for visualization.
    """
    world = World.create(path, seed)

    # 1. Heightmap
    heightmap = _make_heightmap(seed)
    world.rasters.write_layer("geology", "heightmap", heightmap, tick_number=0)

    # 2. Bedrock type
    bedrock = _make_bedrock_type(seed)
    world.rasters.write_layer("geology", "bedrock_type", bedrock, tick_number=0)

    # 3. Soil moisture
    moisture = _make_soil_moisture(seed, heightmap)
    world.rasters.write_layer("climate_hydrology", "soil_moisture_summer", moisture, tick_number=0)

    # 4. Species density fields
    species_ids = _make_species_densities(seed, world, heightmap, moisture)

    # 5. Distinguished individuals
    _place_individuals(seed, world, heightmap, species_ids)

    # 6. Events
    _place_events(seed, world)

    return world
