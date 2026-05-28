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
