"""Climate-hydrology simulation tier — climate fields, river networks, and derived state.

This is the second tier in the three-tier simulation stack, sitting between
geology (below) and ecology (above). It reads the geology heightmap and
produces:

  1. **Climate envelope**: temperature and precipitation fields derived from
     elevation (lapse rate) and orographic effects.
  2. **Hydrology**: D8 flow accumulation and hydraulic erosion, producing
     river networks and an eroded heightmap.
  3. **Derived-state cache**: soil moisture (summer/winter), frost days,
     growing degree days, solar insolation, and distance to water — all
     fields that ecology needs but that are fundamentally climate-driven.

All randomness flows through ``create_rng`` with tier_id="climate_hydrology"
and distinct pass_ids per generation pass, ensuring full reproducibility from
the world seed.

The tier operates at ~1000 years per tick. Each tick reads the geology
heightmap, computes climate and hydrology, and writes 10 raster layers to
the "climate_hydrology" namespace in the RasterStore.
"""

from __future__ import annotations

import numpy as np

from bike_sim.rng import create_rng
from bike_sim.world import World

TIER = "climate_hydrology"


class ClimateHydrologyTier:
    """Climate-hydrology tier: climate envelope, hydrology, and derived cache.

    Reads the geology heightmap and produces 10 raster layers per tick:
    temperature, precipitation, flow_accumulation, eroded_heightmap,
    soil_moisture_summer, soil_moisture_winter, frost_days,
    growing_degree_days, solar_insolation, distance_to_water.
    """

    GRID_SIZE: int = 1000
    CELL_SIZE: float = 50.0  # metres per cell
    YEARS_PER_TICK: int = 1000

    def __init__(self, world: World) -> None:
        self._world = world

    def tick(self) -> None:
        """Advance climate-hydrology by one tick."""
        clock = self._world.tier_clocks[TIER]
        tick_num = clock.tick_number

        # Verify geology has been ticked.
        if "heightmap" not in self._world.rasters.list_layers("geology"):
            raise RuntimeError("Geology must be ticked before climate-hydrology")

        heightmap = self._world.rasters.read_layer("geology", "heightmap")

        # 1. Climate envelope.
        temperature, precipitation = self._compute_climate(heightmap, tick_num)

        # 2. Hydrology: flow accumulation + erosion.
        flow_acc = self._compute_flow_accumulation(heightmap)
        eroded = self._erode(heightmap, precipitation, flow_acc, tick_num)

        # 3. Derived-state cache (writes its own layers).
        self._compute_derived_cache(eroded, temperature, precipitation, flow_acc, tick_num)

        # Write climate and hydrology layers.
        store = self._world.rasters
        store.write_layer(TIER, "temperature", temperature, tick_num)
        store.write_layer(TIER, "precipitation", precipitation, tick_num)
        store.write_layer(TIER, "flow_accumulation", flow_acc, tick_num)
        store.write_layer(TIER, "eroded_heightmap", eroded, tick_num)

        # Advance clock.
        clock.tick_number += 1
        clock.simulated_year += self.YEARS_PER_TICK

    # ------------------------------------------------------------------
    # Climate envelope
    # ------------------------------------------------------------------

    def _compute_climate(
        self, heightmap: np.ndarray, tick_number: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute temperature and precipitation fields from elevation."""
        rng = create_rng(self._world.seed, TIER, "climate", tick_number)

        # Temperature: latitude gradient (y-axis) reduced by elevation lapse rate.
        # Base: 15 °C at southern edge (y=0), 5 °C at northern edge (y=max).
        y_coords = np.linspace(0, 1, self.GRID_SIZE).reshape(-1, 1)
        base_temp = 15.0 - 10.0 * y_coords  # south-to-north gradient
        lapse_rate = 6.5 / 1000.0  # °C per metre
        temperature = base_temp - lapse_rate * heightmap
        temperature += rng.normal(0, 0.5, temperature.shape)

        # Precipitation: base 800 mm/yr with orographic and elevation effects.
        base_precip = 800.0
        dx = np.gradient(heightmap, axis=1)
        orographic = np.clip(dx * 0.5, -200, 400)
        elev_factor = np.clip(heightmap / 1500.0, 0, 1) * 400
        precipitation = base_precip + orographic + elev_factor
        precipitation += rng.normal(0, 50, precipitation.shape)
        precipitation = np.clip(precipitation, 50, 4000)

        return temperature.astype(np.float64), precipitation.astype(np.float64)

    # ------------------------------------------------------------------
    # Hydrology
    # ------------------------------------------------------------------

    def _compute_flow_direction(self, heightmap: np.ndarray) -> np.ndarray:
        """D8 flow direction: each cell drains to its steepest downhill neighbour."""
        rows, cols = heightmap.shape
        flow_dir = np.zeros((rows, cols), dtype=np.int8)

        # Neighbour offsets: 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE
        dr = [0, -1, -1, -1, 0, 1, 1, 1]
        dc = [1, 1, 0, -1, -1, -1, 0, 1]

        padded = np.pad(heightmap, 1, mode="edge")
        min_slope = np.zeros((rows, cols), dtype=np.float64)

        for i in range(8):
            neighbor = padded[1 + dr[i] : rows + 1 + dr[i], 1 + dc[i] : cols + 1 + dc[i]]
            slope = heightmap - neighbor
            mask = slope > min_slope
            min_slope = np.where(mask, slope, min_slope)
            flow_dir = np.where(mask, i, flow_dir)

        return flow_dir

    def _compute_flow_accumulation(self, heightmap: np.ndarray) -> np.ndarray:
        """D8 flow accumulation — each cell counts itself plus all upstream cells."""
        rows, cols = heightmap.shape
        flow_dir = self._compute_flow_direction(heightmap)

        dr = [0, -1, -1, -1, 0, 1, 1, 1]
        dc = [1, 1, 0, -1, -1, -1, 0, 1]

        # Sort cells by elevation (highest first), accumulate downstream.
        acc = np.ones((rows, cols), dtype=np.float64)
        flat_indices = np.argsort(heightmap.ravel())[::-1]

        flow_dir_flat = flow_dir.ravel()
        acc_flat = acc.ravel()

        for idx in flat_indices:
            r, c = divmod(int(idx), cols)
            d = int(flow_dir_flat[idx])
            nr, nc = r + dr[d], c + dc[d]
            if 0 <= nr < rows and 0 <= nc < cols:
                acc_flat[nr * cols + nc] += acc_flat[idx]

        return acc

    def _erode(
        self,
        heightmap: np.ndarray,
        precipitation: np.ndarray,
        flow_acc: np.ndarray,
        tick_number: int,
    ) -> np.ndarray:
        """Simplified grid-based hydraulic erosion."""
        rng = create_rng(self._world.seed, TIER, "erosion", tick_number)  # noqa: F841
        eroded = heightmap.copy()

        # Erosion power: proportional to sqrt(flow) * slope, normalised.
        dy, dx = np.gradient(eroded)
        slope = np.sqrt(dx**2 + dy**2)
        erosion_power = np.sqrt(flow_acc) * slope
        erosion_power /= erosion_power.max() + 1e-10

        # Multiple passes of gentle erosion + soil-creep diffusion.
        for _ in range(5):
            dy, dx = np.gradient(eroded)
            slope = np.sqrt(dx**2 + dy**2)
            erosion = erosion_power * slope * 2.0
            eroded -= erosion
            eroded = _smooth(eroded, sigma=0.5)

        return np.clip(eroded, 0, None).astype(np.float64)

    # ------------------------------------------------------------------
    # Derived-state cache
    # ------------------------------------------------------------------

    def _compute_derived_cache(
        self,
        eroded_heightmap: np.ndarray,
        temperature: np.ndarray,
        precipitation: np.ndarray,
        flow_acc: np.ndarray,
        tick_number: int,
    ) -> None:
        """Compute and write all derived-state layers for the ecology tier."""
        rng = create_rng(self._world.seed, TIER, "derived", tick_number)
        store = self._world.rasters

        # Water cells: top 0.5% of flow accumulation.
        water_threshold = np.percentile(flow_acc, 99.5)
        is_water = flow_acc >= water_threshold

        # --- Soil moisture (summer and winter) ---
        base_moisture = np.clip(precipitation / 2000.0, 0, 1)
        elev_norm = eroded_heightmap / (eroded_heightmap.max() + 1e-10)
        base_moisture += (1 - elev_norm) * 0.2
        base_moisture = np.clip(base_moisture, 0, 1)

        summer_moisture = np.clip(
            base_moisture * 0.7 + rng.uniform(-0.05, 0.05, base_moisture.shape), 0, 1
        )
        winter_moisture = np.clip(
            base_moisture * 1.0 + rng.uniform(-0.05, 0.05, base_moisture.shape), 0, 1
        )
        winter_moisture = np.maximum(winter_moisture, summer_moisture)

        store.write_layer(
            TIER, "soil_moisture_summer", summer_moisture.astype(np.float64), tick_number
        )
        store.write_layer(
            TIER, "soil_moisture_winter", winter_moisture.astype(np.float64), tick_number
        )

        # --- Frost days ---
        frost_days = np.clip(365.0 * (1.0 - temperature / 20.0), 0, 365)
        store.write_layer(TIER, "frost_days", frost_days.astype(np.float64), tick_number)

        # --- Growing degree days ---
        gdd = np.clip((temperature - 5.0) * 365.0 * 0.5, 0, None)
        store.write_layer(TIER, "growing_degree_days", gdd.astype(np.float64), tick_number)

        # --- Solar insolation ---
        dy, dx = np.gradient(eroded_heightmap, self.CELL_SIZE)
        slope_angle = np.arctan(np.sqrt(dx**2 + dy**2))
        aspect = np.arctan2(-dx, dy)
        insolation = np.cos(slope_angle) * (0.5 + 0.5 * np.cos(aspect - np.pi))
        insolation = np.clip(insolation, 0, 1)
        store.write_layer(TIER, "solar_insolation", insolation.astype(np.float64), tick_number)

        # --- Distance to water ---
        distance_to_water = _distance_transform(is_water, self.CELL_SIZE)
        store.write_layer(
            TIER, "distance_to_water", distance_to_water.astype(np.float64), tick_number
        )


# ------------------------------------------------------------------
# Module-level utility functions
# ------------------------------------------------------------------


def _smooth(arr: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Simple 3×3 box blur applied ceil(sigma) times (approximates Gaussian)."""
    kernel_passes = max(1, int(np.ceil(sigma)))
    result = arr.copy()
    for _ in range(kernel_passes):
        padded = np.pad(result, 1, mode="edge")
        result = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 9.0
    return result


def _distance_transform(is_water: np.ndarray, cell_size: float) -> np.ndarray:
    """Approximate Euclidean distance transform via vectorised Chamfer sweeps.

    Uses cardinal and diagonal row/column sweeps — O(n) per sweep, vectorised
    per row or column, so much faster than a cell-by-cell Python loop.
    """
    rows, cols = is_water.shape
    INF = float(rows + cols) * cell_size
    dist = np.where(is_water, 0.0, INF)

    DIAG = cell_size * 1.414
    CARD = cell_size

    # Forward pass: top-to-bottom, left-to-right
    for r in range(1, rows):
        dist[r, :] = np.minimum(dist[r, :], dist[r - 1, :] + CARD)
        # Diagonal from top-left
        dist[r, 1:] = np.minimum(dist[r, 1:], dist[r - 1, :-1] + DIAG)
        # Diagonal from top-right
        dist[r, :-1] = np.minimum(dist[r, :-1], dist[r - 1, 1:] + DIAG)
    for c in range(1, cols):
        dist[:, c] = np.minimum(dist[:, c], dist[:, c - 1] + CARD)

    # Backward pass: bottom-to-top, right-to-left
    for r in range(rows - 2, -1, -1):
        dist[r, :] = np.minimum(dist[r, :], dist[r + 1, :] + CARD)
        # Diagonal from bottom-right
        dist[r, :-1] = np.minimum(dist[r, :-1], dist[r + 1, 1:] + DIAG)
        # Diagonal from bottom-left
        dist[r, 1:] = np.minimum(dist[r, 1:], dist[r + 1, :-1] + DIAG)
    for c in range(cols - 2, -1, -1):
        dist[:, c] = np.minimum(dist[:, c], dist[:, c + 1] + CARD)

    return dist
