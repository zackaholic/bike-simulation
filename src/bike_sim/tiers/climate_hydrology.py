"""Climate-hydrology simulation tier — climate fields, river networks, and derived state.

This is the second tier in the three-tier simulation stack, sitting between
geology (below) and ecology (above). It reads the geology heightmap and
produces:

  1. **Climate envelope**: temperature and precipitation fields derived from
     elevation (lapse rate) and orographic effects.
  2. **Hydrology**: D8 flow accumulation and particle-based hydraulic erosion
     (replacing the earlier grid-diffusion placeholder), producing river
     channels, alluvial fans, and an eroded heightmap with a separate
     sediment depth layer.
  3. **Thermal erosion** (talus creep) that softens cliffs and fills gullies.
  4. **Derived-state cache**: soil moisture (summer/winter), frost days,
     growing degree days, solar insolation, and distance to water — all
     fields that ecology needs but that are fundamentally climate-driven.

All randomness flows through ``create_rng`` with tier_id="climate_hydrology"
and distinct pass_ids per generation pass, ensuring full reproducibility from
the world seed.

The tier operates at ~1000 years per tick. Each tick reads the geology
heightmap, computes climate and hydrology, and writes 11 raster layers to
the "climate_hydrology" namespace in the RasterStore.
"""

from __future__ import annotations

import numpy as np

from bike_sim.rng import create_rng
from bike_sim.tiers.erosion import ErosionParams, erode_hydraulic, erode_thermal
from bike_sim.tiers.geology import _bilinear_upsample
from bike_sim.world import World

TIER = "climate_hydrology"


class ClimateHydrologyTier:
    """Climate-hydrology tier: climate envelope, hydrology, and derived cache.

    Reads the geology heightmap and produces 7 raster layers per tick:
    temperature, precipitation, flow_accumulation, eroded_heightmap,
    sediment_depth, soil_moisture_summer, soil_moisture_winter.
    """

    GRID_SIZE: int = 1000
    CELL_SIZE: float = 50.0  # metres per cell
    YEARS_PER_TICK: int = 1000

    def __init__(self, world: World, erosion_params: ErosionParams | None = None) -> None:
        self._world = world
        self._erosion_params = erosion_params or ErosionParams()

    def tick(self) -> None:
        """Advance climate-hydrology by one tick."""
        clock = self._world.tier_clocks[TIER]
        tick_num = clock.tick_number

        # Verify geology.
        if "heightmap" not in self._world.rasters.list_layers("geology"):
            raise RuntimeError("Geology must be ticked before climate-hydrology")

        geology_heightmap = self._world.rasters.read_layer("geology", "heightmap")
        bedrock_type = self._world.rasters.read_layer("geology", "bedrock_type")

        # On subsequent ticks, erode the already-eroded surface rather than
        # starting from the geology baseline.  This lets erosion accumulate
        # across climate ticks — the second pass deepens channels carved by
        # the first, which is the "process over outcome" principle.
        if tick_num > 0 and "eroded_heightmap" in self._world.rasters.list_layers(TIER):
            heightmap = self._world.rasters.read_layer(TIER, "eroded_heightmap")
        else:
            heightmap = geology_heightmap

        # 1. Climate envelope (unchanged — always from geology for lapse rate).
        temperature, precipitation = self._compute_climate(geology_heightmap, tick_num)

        # 2. Pre-erosion flow accumulation (for particle spawn weighting).
        pre_flow = self._compute_flow_accumulation(heightmap)

        # 3. Load or initialize sediment.
        sediment = self._load_sediment(tick_num)

        # 4. Hydraulic erosion.
        rng = create_rng(self._world.seed, TIER, "erosion", tick_num)
        eroded, sediment = erode_hydraulic(
            heightmap, bedrock_type, precipitation, pre_flow,
            sediment, rng, self._erosion_params,
        )

        # 5. Thermal erosion.
        erode_thermal(eroded, sediment, self._erosion_params)

        # 6. Post-erosion flow accumulation on combined surface.
        combined_surface = eroded + sediment
        flow_acc = self._compute_flow_accumulation(combined_surface)

        # 7. Soil-moisture cache (derived from combined surface + precipitation).
        self._compute_derived_cache(
            combined_surface, sediment, precipitation, tick_num
        )

        # 8. Write layers.
        store = self._world.rasters
        store.write_layer(TIER, "temperature", temperature, tick_num)
        store.write_layer(TIER, "precipitation", precipitation, tick_num)
        store.write_layer(TIER, "flow_accumulation", flow_acc, tick_num)
        store.write_layer(TIER, "eroded_heightmap", eroded, tick_num)
        store.write_layer(TIER, "sediment_depth", sediment, tick_num)

        # 9. Spatial climate bias fields (generated once on tick 0).
        if tick_num == 0:
            moisture_bias, continentality = self._generate_spatial_climate_fields(
                geology_heightmap, tick_num
            )
            store.write_layer(TIER, "moisture_bias", moisture_bias, tick_num)
            store.write_layer(TIER, "continentality", continentality, tick_num)

        # Advance clock.
        clock.tick_number += 1
        clock.simulated_year += self.YEARS_PER_TICK

    def _load_sediment(self, tick_number: int) -> np.ndarray:
        """Load previous sediment layer, or zeros if none exists."""
        try:
            if "sediment_depth" in self._world.rasters.list_layers(TIER):
                return self._world.rasters.read_layer(TIER, "sediment_depth")
        except Exception:
            pass
        return np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float64)

    # ------------------------------------------------------------------
    # Spatial climate bias fields
    # ------------------------------------------------------------------

    def _generate_spatial_climate_fields(
        self, heightmap: np.ndarray, tick_number: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate moisture_bias and continentality noise fields.

        These are static 2D fields that create distinct climate regions
        without hard biome boundaries. They modulate the weather system's
        output spatially:

        - moisture_bias [0.5, 2.0]: multiplied against precipitation.
          Correlates with elevation (lower = wetter) to create natural
          rain shadow effects.
        - continentality [0.0, 1.0]: scales temperature extremes.
          High = hot summers/cold winters. Low = mild/maritime.

        Both use low-frequency noise (2-3 octaves) to produce 4-6
        landscape-scale regions with gradual transitions.
        """
        rng = create_rng(self._world.seed, TIER, "spatial_climate", tick_number)
        size = self.GRID_SIZE

        # --- Moisture bias ---
        # Low-frequency noise: broad regions with one detail octave
        moisture_raw = np.zeros((size, size), dtype=np.float64)
        noise = rng.random((3, 3))
        moisture_raw += _bilinear_upsample(noise, size) * 1.0
        noise = rng.random((7, 7))
        moisture_raw += _bilinear_upsample(noise, size) * 0.3

        # Normalize to [0, 1]
        lo, hi = moisture_raw.min(), moisture_raw.max()
        if hi > lo:
            moisture_raw = (moisture_raw - lo) / (hi - lo)
        else:
            moisture_raw[:] = 0.5

        # Correlate with elevation: lower areas are wetter
        elev_norm = heightmap / (heightmap.max() + 1e-10)
        # Blend: 70% noise, 30% elevation influence (inverted)
        moisture_raw = 0.7 * moisture_raw + 0.3 * (1.0 - elev_norm)

        # Map to [0.5, 2.0] range
        moisture_bias = 0.5 + moisture_raw * 1.5

        # --- Continentality ---
        # Single low-frequency octave for broad regions, plus one detail octave
        cont_raw = np.zeros((size, size), dtype=np.float64)
        noise = rng.random((3, 3))
        cont_raw += _bilinear_upsample(noise, size) * 1.0
        noise = rng.random((6, 6))
        cont_raw += _bilinear_upsample(noise, size) * 0.3

        lo, hi = cont_raw.min(), cont_raw.max()
        if hi > lo:
            continentality = (cont_raw - lo) / (hi - lo)
        else:
            continentality = np.full((size, size), 0.5, dtype=np.float64)

        return moisture_bias.astype(np.float64), continentality.astype(np.float64)

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

    # ------------------------------------------------------------------
    # Derived-state cache
    # ------------------------------------------------------------------

    def _compute_derived_cache(
        self,
        combined_surface: np.ndarray,
        sediment: np.ndarray,
        precipitation: np.ndarray,
        tick_number: int,
    ) -> None:
        """Compute and write the soil-moisture layers for the ecology tier."""
        rng = create_rng(self._world.seed, TIER, "derived", tick_number)
        store = self._world.rasters

        # --- Soil moisture (summer and winter) ---
        base_moisture = np.clip(precipitation / 2000.0, 0, 1)
        elev_norm = combined_surface / (combined_surface.max() + 1e-10)
        base_moisture += (1 - elev_norm) * 0.2
        sediment_bonus = np.minimum(sediment / 5.0, 0.15)
        base_moisture += sediment_bonus
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


