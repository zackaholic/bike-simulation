"""Geology simulation tier (v1 stub — noise-based terrain generation).

This is the bottom tier in the three-tier simulation stack. It produces the
base terrain layers that climate-hydrology and ecology build on: a heightmap,
bedrock type map, and soil parent material.

For v1, terrain is generated via multi-octave noise rather than plate tectonics.
The architecture is designed so that a tectonic model can swap in later without
changing the tier interface. Generation and advancement use the same ``tick()``
method — creating a new world means ticking geology once at deep-time scale.
After the initial tick, subsequent ticks are no-ops (geology is static in v1).

All randomness flows through ``create_rng`` with tier_id="geology" and a
distinct pass_id for each generation pass, ensuring full reproducibility
from the world seed.
"""

from __future__ import annotations

import numpy as np

from bike_sim.rng import create_rng
from bike_sim.world import World

# Mapping from bedrock type to soil parent material type.
# Some bedrock types weather to the same soil parent (e.g., types 1 and 2
# are mineralogically similar, types 4 and 5 produce the same coarse regolith).
BEDROCK_TO_SOIL: dict[int, int] = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 3}


class GeologyTier:
    """Noise-based geology tier (v1).

    Produces three raster layers on the first tick:
      - ``heightmap`` (float64): elevation in metres, roughly 0–2000 m
      - ``bedrock_type`` (int32): rock type index 0–5
      - ``soil_parent`` (int32): soil parent material derived from bedrock
    """

    # Configuration constants
    GRID_SIZE: int = 1000  # 1000×1000 cells
    CELL_SIZE: float = 50.0  # metres per cell
    NUM_ROCK_TYPES: int = 6  # bedrock types 0–5
    YEARS_PER_TICK: int = 100_000  # 100K years per geology tick

    def __init__(self, world: World) -> None:
        self._world = world

    def tick(self) -> None:
        """Advance geology by one tick.

        On tick 0 this generates the initial terrain. Subsequent ticks are
        no-ops (geology is static in v1) but still advance the tier clock so
        that the world timeline stays consistent.
        """
        clock = self._world.tier_clocks["geology"]
        tick_num = clock.tick_number

        if tick_num == 0:
            self._generate_initial_terrain(tick_num)

        # Advance clock (even for no-op ticks after the first).
        clock.tick_number += 1
        clock.simulated_year += self.YEARS_PER_TICK

    # ------------------------------------------------------------------
    # Internal generation passes
    # ------------------------------------------------------------------

    def _generate_initial_terrain(self, tick_number: int) -> None:
        """Run all generation passes for the initial terrain."""
        heightmap = self._generate_heightmap(tick_number)
        bedrock = self._generate_bedrock(tick_number)
        soil_parent = self._derive_soil_parent(bedrock)

        store = self._world.rasters
        assert store is not None, "World must have an open RasterStore"
        store.write_layer("geology", "heightmap", heightmap, tick_number)
        store.write_layer("geology", "bedrock_type", bedrock, tick_number)
        store.write_layer("geology", "soil_parent", soil_parent, tick_number)

    def _generate_heightmap(self, tick_number: int) -> np.ndarray:
        """Multi-octave noise heightmap.

        Each octave is a small random grid bilinearly interpolated up to the
        full grid size. Large-scale octaves dominate to produce mountain ranges
        and valleys; small-scale octaves add local texture.
        """
        rng = create_rng(self._world.seed, "geology", "heightmap", tick_number)

        size = self.GRID_SIZE
        octave_sizes = [8, 16, 32, 64, 128, 256]
        octave_weights = [32.0, 16.0, 8.0, 4.0, 2.0, 1.0]

        accumulated = np.zeros((size, size), dtype=np.float64)

        for n, weight in zip(octave_sizes, octave_weights, strict=True):
            noise = rng.random((n, n))
            upscaled = _bilinear_upsample(noise, size)
            accumulated += upscaled * weight

        # Normalize to 0–2000 m elevation range.
        lo, hi = accumulated.min(), accumulated.max()
        if hi > lo:
            heightmap = (accumulated - lo) / (hi - lo) * 2000.0
        else:
            heightmap = np.full((size, size), 1000.0, dtype=np.float64)

        return heightmap

    def _generate_bedrock(self, tick_number: int) -> np.ndarray:
        """Voronoi-ish bedrock regions.

        A small set of seed points are scattered across the grid, each assigned
        a random rock type. Every cell inherits the type of its nearest seed
        point, producing irregular geological provinces.
        """
        rng = create_rng(self._world.seed, "geology", "bedrock", tick_number)

        size = self.GRID_SIZE
        num_seeds = rng.integers(15, 21)  # 15–20 seed points

        # Random seed-point positions (in cell coordinates).
        seed_xs = rng.random(num_seeds) * size
        seed_ys = rng.random(num_seeds) * size
        seed_types = rng.integers(0, self.NUM_ROCK_TYPES, size=num_seeds)

        # For every cell, find the nearest seed point.
        row_coords = np.arange(size, dtype=np.float64)
        col_coords = np.arange(size, dtype=np.float64)
        # Shape: (size, size, 1) vs (num_seeds,) — broadcast to get distances.
        rows = row_coords[:, np.newaxis, np.newaxis]  # (size, 1, 1)
        cols = col_coords[np.newaxis, :, np.newaxis]  # (1, size, 1)
        sx = seed_xs[np.newaxis, np.newaxis, :]  # (1, 1, num_seeds)
        sy = seed_ys[np.newaxis, np.newaxis, :]  # (1, 1, num_seeds)

        dist_sq = (rows - sy) ** 2 + (cols - sx) ** 2  # (size, size, num_seeds)
        nearest = np.argmin(dist_sq, axis=2)  # (size, size)

        bedrock = seed_types[nearest].astype(np.int32)
        return bedrock

    def _derive_soil_parent(self, bedrock: np.ndarray) -> np.ndarray:
        """Deterministic mapping from bedrock type to soil parent material."""
        # Build a lookup array for vectorised mapping.
        max_type = max(BEDROCK_TO_SOIL.keys())
        lookup = np.zeros(max_type + 1, dtype=np.int32)
        for k, v in BEDROCK_TO_SOIL.items():
            lookup[k] = v

        return lookup[bedrock]


def _bilinear_upsample(grid: np.ndarray, target_size: int) -> np.ndarray:
    """Upsample a small 2D grid to target_size×target_size via bilinear interpolation.

    Uses np.interp row-by-row then column-by-column (separable bilinear).
    """
    n = grid.shape[0]
    src_coords = np.arange(n, dtype=np.float64)
    dst_coords = np.linspace(0, n - 1, target_size)

    # Interpolate along columns (axis 1) for each source row.
    row_interp = np.empty((n, target_size), dtype=np.float64)
    for i in range(n):
        row_interp[i] = np.interp(dst_coords, src_coords, grid[i])

    # Interpolate along rows (axis 0) for each target column.
    result = np.empty((target_size, target_size), dtype=np.float64)
    src_row_coords = np.arange(n, dtype=np.float64)
    dst_row_coords = np.linspace(0, n - 1, target_size)
    for j in range(target_size):
        result[:, j] = np.interp(dst_row_coords, src_row_coords, row_interp[:, j])

    return result
