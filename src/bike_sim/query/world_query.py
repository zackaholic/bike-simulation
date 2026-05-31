"""Layer B query interface for world simulation state.

This module sits between the raw data stores (RasterStore, EventStore) and
all consumers — both simulation tiers that need to read neighboring-tier
outputs and extractors that prepare data for rendering.

Why does Layer B exist?

The simulation produces canonical state in two forms: dense 2D raster grids
(Zarr-backed via RasterStore) and sparse relational data — species, individual
organisms, historical events (SQLite-backed via EventStore). Nobody should
read those stores directly. WorldQuery provides a single, stable API that:

  1. Translates between world coordinates (meters) and cell indices.
  2. Handles spatial queries with consistent conventions.
  3. Shields consumers from storage layout changes.
  4. Makes it easy to compose spatial + temporal + species filters.

All coordinate inputs are in world-space meters. The world is 50 000 m × 50 000 m
with 50 m cells, giving a 1000 × 1000 grid. Rows correspond to the y axis,
columns to the x axis.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bike_sim.world import World

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CELL_SIZE: float = 50.0
_MAX_INDEX: int = 999

# Pre-compiled pattern for species density layer names.
_SPECIES_DENSITY_RE: re.Pattern[str] = re.compile(r"^species_(.+)_density$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _world_to_cell(coord: float) -> int:
    """Convert a world-space coordinate (meters) to a cell index.

    Uses nearest-neighbor mapping: ``floor(coord / 50.0)``, clamped to [0, 999].
    """
    return min(max(int(coord / _CELL_SIZE), 0), _MAX_INDEX)


# ---------------------------------------------------------------------------
# WorldQuery
# ---------------------------------------------------------------------------


class WorldQuery:
    """Read-only query interface over a World's simulation state.

    Wraps RasterStore and EventStore with coordinate translation and
    convenient spatial / temporal filters.  Both simulation tiers and
    render-side extractors should use this rather than touching the
    stores directly.
    """

    def __init__(self, world: World) -> None:
        self._world = world

    # -- raster point sampling ------------------------------------------------

    def sample_layer(self, tier: str, layer_name: str, x: float, y: float) -> float:
        """Sample a raster layer at world coordinates (*x*, *y*).

        Uses nearest-neighbor lookup: the coordinate is converted to a cell
        index and the value at that cell is returned as a Python float.
        """
        col = _world_to_cell(x)
        row = _world_to_cell(y)
        data = self._world.rasters.read_layer(tier, layer_name)
        return float(data[row, col])

    # -- raster region sampling -----------------------------------------------

    def sample_layer_region(
        self,
        tier: str,
        layer_name: str,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ) -> np.ndarray:
        """Return a 2D slice of a raster layer within a bounding box.

        Coordinates are converted to cell indices and the sub-array
        ``data[row_min:row_max+1, col_min:col_max+1]`` is returned.
        """
        col_min = _world_to_cell(x_min)
        col_max = _world_to_cell(x_max)
        row_min = _world_to_cell(y_min)
        row_max = _world_to_cell(y_max)
        data = self._world.rasters.read_layer(tier, layer_name)
        return data[row_min : row_max + 1, col_min : col_max + 1]

    # -- species queries ------------------------------------------------------

    def species_at(self, x: float, y: float, radius: float = 100.0) -> list[dict]:
        """Find species with positive density at the given point.

        Scans all ecology-tier layers matching ``species_*_density``, samples
        each at (*x*, *y*), and returns a list of ``{species_id, density}``
        dicts for every species whose density is greater than zero.

        *radius* is accepted for API consistency but currently unused — the
        density check is a single-cell nearest-neighbor sample.
        """
        results: list[dict] = []
        for layer_name in self.available_layers("ecology"):
            match = _SPECIES_DENSITY_RE.match(layer_name)
            if match is None:
                continue
            species_id = match.group(1)
            density = self.sample_layer("ecology", layer_name, x, y)
            if density > 0:
                results.append({"species_id": species_id, "density": density})
        return results

    # -- individual / event queries -------------------------------------------

    def individuals_near(self, x: float, y: float, radius: float) -> list[dict]:
        """Return individuals within *radius* meters of (*x*, *y*).

        Delegates directly to ``EventStore.find_individuals_near``.
        """
        return self._world.events.find_individuals_near(x, y, radius)

    def events_near(
        self,
        x: float,
        y: float,
        radius: float,
        year_start: float | None = None,
        year_end: float | None = None,
    ) -> list[dict]:
        """Return events within a bounding box around (*x*, *y*).

        The spatial filter is a square with half-width *radius*:
        ``(x - radius, y - radius, x + radius, y + radius)``.

        If *year_start* and/or *year_end* are given, the spatial results are
        further filtered to include only events whose ``"year"`` falls within
        the specified range (inclusive on both ends).
        """
        x_min = x - radius
        y_min = y - radius
        x_max = x + radius
        y_max = y + radius

        events = self._world.events.get_events_in_region(x_min, y_min, x_max, y_max)

        if year_start is not None:
            events = [e for e in events if e.get("year", float("-inf")) >= year_start]
        if year_end is not None:
            events = [e for e in events if e.get("year", float("inf")) <= year_end]

        return events

    # -- layer introspection --------------------------------------------------

    def available_layers(self, tier: str) -> list[str]:
        """List all raster layer names for the given tier.

        Delegates directly to ``RasterStore.list_layers``.
        """
        return self._world.rasters.list_layers(tier)

    # -- version-aware helpers -----------------------------------------------

    def _version_year(self, version: int) -> float:
        """Get the ecology simulated_year at a given version."""
        clock = self._world.version_for_tier(version, "ecology")
        return clock.simulated_year

    # -- version-aware queries -----------------------------------------------

    def get_world_metadata(self, version: int) -> dict:
        """Return structural metadata about the world at a given version.

        Includes extent, cell size, available layers grouped by tier,
        species/individual counts alive at the version's ecology year,
        and the tier clock snapshot.
        """
        ver_entry = self._world.get_version(version)
        year = self._version_year(version)

        # Layers available at this version, grouped by tier
        layers: dict[str, list[str]] = {}
        for tier in ["geology", "climate_hydrology", "ecology"]:
            layers[tier] = self._world.rasters.list_layers(tier, version=version)

        # Species alive at this version's year
        species = self._world.events.list_species(alive_at_year=year)

        # Individuals alive at this version's year
        individuals = self._world.events.find_individuals_near(
            25000, 25000, 50000, alive_at_year=year
        )

        return {
            "extent": {"x_min": 0, "y_min": 0, "x_max": 50000, "y_max": 50000},
            "cell_size": 50.0,
            "grid_size": 1000,
            "layers": layers,
            "species_count": len(species),
            "individual_count": len(individuals),
            "simulated_year": year,
            "tier_clocks": ver_entry["tier_clocks"],
        }

    def query_point(self, version: int, x: float, y: float) -> dict:
        """Return everything known at a single world coordinate for a version.

        Samples all raster layers, finds species with positive density,
        nearby individuals (alive at the version year), and nearby events.
        """
        year = self._version_year(version)

        # Sample all raster layers at this version
        rasters: dict[str, float] = {}
        for tier in ["geology", "climate_hydrology", "ecology"]:
            for layer_name in self._world.rasters.list_layers(tier, version=version):
                data = self._world.rasters.read_layer(tier, layer_name, version=version)
                col = _world_to_cell(x)
                row = _world_to_cell(y)
                rasters[f"{tier}/{layer_name}"] = float(data[row, col])

        # Species at this point with density > 0
        species: list[dict] = []
        for tier_layer, val in rasters.items():
            if tier_layer.startswith("ecology/species_") and tier_layer.endswith("_density"):
                if val > 0:
                    sid = tier_layer.replace("ecology/species_", "").replace("_density", "")
                    species.append({"species_id": sid, "density": val})

        # Individuals near this point, alive at this version
        individuals = self._world.events.find_individuals_near(
            x, y, radius=100.0, alive_at_year=year
        )

        # Events near this point
        events = self._world.events.get_events_in_region(x - 100, y - 100, x + 100, y + 100)

        return {
            "x": x,
            "y": y,
            "rasters": rasters,
            "species": species,
            "individuals": individuals,
            "events": events,
        }

    def query_raster(
        self,
        version: int,
        tier: str,
        layer_name: str,
        bbox: tuple[float, float, float, float],
        target_size: tuple[int, int],
    ) -> np.ndarray:
        """Clip a raster to *bbox* and resample to *target_size*.

        Uses nearest-neighbor resampling. *bbox* is ``(x_min, y_min, x_max, y_max)``
        in world coordinates. *target_size* is ``(rows, cols)``.
        """
        data = self._world.rasters.read_layer(tier, layer_name, version=version)

        x_min, y_min, x_max, y_max = bbox
        col_min = _world_to_cell(x_min)
        row_min = _world_to_cell(y_min)
        col_max = _world_to_cell(x_max)
        row_max = _world_to_cell(y_max)

        # Clip
        clipped = data[row_min : row_max + 1, col_min : col_max + 1]

        # Resample to target_size using simple nearest-neighbor
        target_rows, target_cols = target_size
        src_rows, src_cols = clipped.shape

        row_indices = (np.arange(target_rows) * src_rows / target_rows).astype(int)
        col_indices = (np.arange(target_cols) * src_cols / target_cols).astype(int)
        row_indices = np.clip(row_indices, 0, src_rows - 1)
        col_indices = np.clip(col_indices, 0, src_cols - 1)

        return clipped[np.ix_(row_indices, col_indices)]

    def query_individuals_in_bbox(
        self,
        version: int,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ) -> list[dict]:
        """Return individuals within a bounding box, alive at the version year."""
        year = self._version_year(version)
        # Use a large radius search centered on bbox center, then filter to bbox
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        radius = max(x_max - x_min, y_max - y_min)  # generous radius

        candidates = self._world.events.find_individuals_near(cx, cy, radius, alive_at_year=year)

        # Filter to exact bbox
        return [
            ind for ind in candidates if x_min <= ind["x"] <= x_max and y_min <= ind["y"] <= y_max
        ]

    def get_individual_detail(self, version: int, individual_id: str) -> dict:
        """Return full details for a distinguished individual at a version.

        Includes derived fields: age (version year minus appeared_year),
        alive (whether still living at version year), and the species genome.
        """
        year = self._version_year(version)
        ind = self._world.events.get_individual(individual_id)
        species_info = self._world.events.get_species(ind["species_id"])

        died_year = ind.get("died_year")
        alive = died_year is None or died_year > year
        age = year - ind["appeared_year"]

        return {
            "individual_id": ind["individual_id"],
            "species_id": ind["species_id"],
            "x": ind["x"],
            "y": ind["y"],
            "appeared_year": ind["appeared_year"],
            "died_year": died_year,
            "age": age,
            "alive": alive,
            "species_genome": species_info["genome"],
        }

    def list_species_summary(self, version: int) -> list[dict]:
        """Return a summary of all species that existed at a given version.

        Each entry includes the species genome, alive/extinct status at the
        version year, and a count of living individuals.
        """
        year = self._version_year(version)
        species_list = self._world.events.list_species()  # ALL species, not filtered

        # Get all living individuals once for counting
        all_inds = self._world.events.find_individuals_near(
            25000, 25000, 50000, alive_at_year=year
        )

        result: list[dict] = []
        for sp in species_list:
            # Only include species that existed at this version
            sp_info = self._world.events.get_species(sp["species_id"])
            if sp_info["appeared_year"] > year:
                continue

            extinct_year = sp_info.get("extinct_year")
            alive = extinct_year is None or extinct_year > year

            # Count living individuals for this species at this version
            ind_count = sum(1 for ind in all_inds if ind["species_id"] == sp["species_id"])

            result.append(
                {
                    "species_id": sp["species_id"],
                    "parent_id": sp_info["parent_id"],
                    "appeared_year": sp_info["appeared_year"],
                    "extinct_year": extinct_year,
                    "alive": alive,
                    "genome": sp_info["genome"],
                    "individual_count": ind_count,
                }
            )

        return result
