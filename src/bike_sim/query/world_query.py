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

import math
import re
from collections import defaultdict
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
        for tier in self._world.rasters.list_tiers():
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
        for tier in self._world.rasters.list_tiers():
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

    # -- timeline queries ---------------------------------------------------

    def get_species_timeline(self, ancestor: str | None = None) -> dict:
        """Return species population data aggregated to yearly resolution.

        If *ancestor* is given, only include species descended from that
        ancestor (matching species whose ID starts with the ancestor prefix,
        or whose lineage traces back to it via parent_id).

        Returns ``{"years": [...], "species": {species_id: [densities...]}}``.
        """
        summaries = self._world.events.get_tick_summaries()
        if not summaries:
            return {"years": [], "species": {}}

        # If ancestor filter requested, resolve the set of matching species IDs
        allowed_species: set[str] | None = None
        if ancestor is not None:
            all_species = self._world.events.list_species()
            allowed_species = set()
            # Build parent lookup
            parent_map: dict[str, str | None] = {
                sp["species_id"]: sp["parent_id"] for sp in all_species
            }
            for sp in all_species:
                sid = sp["species_id"]
                # Walk up lineage to check if ancestor is in the chain
                cur = sid
                while cur is not None:
                    if cur == ancestor:
                        allowed_species.add(sid)
                        break
                    cur = parent_map.get(cur)

        # Group by (year_int, species_id) — take last tick per year
        # year = tick * 0.25, so year_int = int(year)
        # We want the last tick of each integer year
        by_year_species: dict[int, dict[str, float]] = defaultdict(dict)
        tick_by_year: dict[int, int] = {}
        for row in summaries:
            year_int = int(row["year"])
            tick = row["tick"]
            sid = row["species_id"]
            if allowed_species is not None and sid not in allowed_species:
                continue
            # Keep the entry from the latest tick in each year
            if year_int not in tick_by_year or tick >= tick_by_year[year_int]:
                if tick > tick_by_year.get(year_int, -1):
                    # New latest tick for this year — reset species for this year
                    tick_by_year[year_int] = tick
                    by_year_species[year_int] = {}
                by_year_species[year_int][sid] = row["total_density"]

        if not by_year_species:
            return {"years": [], "species": {}}

        years = sorted(by_year_species.keys())

        # Collect all species that ever had density > 0
        all_sids: set[str] = set()
        for sp_map in by_year_species.values():
            for sid, density in sp_map.items():
                if density > 0:
                    all_sids.add(sid)

        species_data: dict[str, list[float]] = {}
        for sid in sorted(all_sids):
            species_data[sid] = [by_year_species[y].get(sid, 0.0) for y in years]

        return {"years": [float(y) for y in years], "species": species_data}

    def get_weather_timeline(self) -> dict:
        """Return weather data aggregated to yearly means.

        Returns ``{"years": [...], "temperature": [...], "precipitation": [...],
        "drought": [...]}``.
        """
        weather = self._world.events.get_tick_weather()
        if not weather:
            return {"years": [], "temperature": [], "precipitation": [], "drought": []}

        # Group by integer year, compute mean of each field
        by_year: dict[int, list[dict]] = defaultdict(list)
        for row in weather:
            year_int = int(row["year"])
            by_year[year_int].append(row)

        years = sorted(by_year.keys())
        temperature = []
        precipitation = []
        drought = []
        for y in years:
            rows = by_year[y]
            n = len(rows)
            temperature.append(sum(r["mean_temp"] for r in rows) / n)
            precipitation.append(sum(r["mean_precip"] for r in rows) / n)
            drought.append(sum(r["mean_drought"] for r in rows) / n)

        return {
            "years": [float(y) for y in years],
            "temperature": temperature,
            "precipitation": precipitation,
            "drought": drought,
        }

    def get_diversity_timeline(self, threshold: float = 0.01) -> dict:
        """Return diversity metrics aggregated to yearly resolution.

        Returns ``{"years": [...], "species_count": [...], "shannon": [...],
        "total_density": [...]}``.

        *threshold* is the minimum total_density for a species to be counted
        as present (for species_count and Shannon index).
        """
        summaries = self._world.events.get_tick_summaries()
        if not summaries:
            return {"years": [], "species_count": [], "shannon": [], "total_density": []}

        # Group by year, take last tick per year (same logic as species timeline)
        tick_by_year: dict[int, int] = {}
        by_year: dict[int, dict[str, float]] = defaultdict(dict)
        for row in summaries:
            year_int = int(row["year"])
            tick = row["tick"]
            sid = row["species_id"]
            if year_int not in tick_by_year or tick >= tick_by_year[year_int]:
                if tick > tick_by_year.get(year_int, -1):
                    tick_by_year[year_int] = tick
                    by_year[year_int] = {}
                by_year[year_int][sid] = row["total_density"]

        years = sorted(by_year.keys())
        species_count_list = []
        shannon_list = []
        total_density_list = []

        for y in years:
            densities = by_year[y]
            total = sum(densities.values())
            total_density_list.append(total)

            # Count species above threshold
            above = [d for d in densities.values() if d > threshold]
            species_count_list.append(len(above))

            # Shannon diversity index
            if total > 0 and len(above) > 0:
                h = 0.0
                for d in above:
                    p = d / total
                    if p > 0:
                        h -= p * math.log(p)
                shannon_list.append(h)
            else:
                shannon_list.append(0.0)

        return {
            "years": [float(y) for y in years],
            "species_count": species_count_list,
            "shannon": shannon_list,
            "total_density": total_density_list,
        }

    def get_speciation_timeline(self) -> dict:
        """Return species tree and population history for the speciation chart.

        Returns::

            {
                "species": [
                    {
                        "species_id": "anc0_...",
                        "parent_id": "anc0" | null,
                        "ancestor": 0,
                        "appeared_year": 0.0,
                        "extinct_year": null | 500.0,
                        "population": [
                            {"year": 0.0, "density": 1234.5},
                            ...
                        ]
                    },
                    ...
                ],
                "max_year": 1000.0
            }
        """
        all_species = self._world.events.list_species()
        if not all_species:
            return {"species": [], "max_year": 0.0}

        # Build species info lookup
        species_info: dict[str, dict] = {}
        for sp in all_species:
            sid = sp["species_id"]
            full = self._world.events.get_species(sid)
            species_info[sid] = full

        # Build parent lookup for ancestor resolution
        parent_map: dict[str, str | None] = {
            sid: info["parent_id"] for sid, info in species_info.items()
        }

        def get_ancestor_idx(sid: str) -> int:
            """Walk up lineage to find root ancestor index."""
            m = re.match(r"^anc_(\d+)", sid)
            return int(m.group(1)) if m else 0

        # Get population history from tick summaries
        summaries = self._world.events.get_tick_summaries()

        # Group by (year_int, species_id) — last tick per year
        tick_by_year: dict[int, int] = {}
        by_year_species: dict[int, dict[str, float]] = defaultdict(dict)
        for row in summaries:
            year_int = int(row["year"])
            tick = row["tick"]
            sid = row["species_id"]
            if year_int not in tick_by_year or tick >= tick_by_year[year_int]:
                if tick > tick_by_year.get(year_int, -1):
                    tick_by_year[year_int] = tick
                    by_year_species[year_int] = {}
                by_year_species[year_int][sid] = row["total_density"]

        years = sorted(by_year_species.keys())

        # Build per-species population arrays
        pop_by_species: dict[str, list[dict]] = defaultdict(list)
        for y in years:
            for sid, density in by_year_species[y].items():
                pop_by_species[sid].append({"year": float(y), "density": density})

        max_year = float(years[-1]) if years else 0.0

        result = []
        for sid, info in species_info.items():
            result.append({
                "species_id": sid,
                "parent_id": info["parent_id"],
                "ancestor": get_ancestor_idx(sid),
                "appeared_year": info["appeared_year"],
                "extinct_year": info["extinct_year"],
                "population": pop_by_species.get(sid, []),
            })

        # Sort by appeared_year then species_id for stable ordering
        result.sort(key=lambda s: (s["appeared_year"], s["species_id"]))

        return {"species": result, "max_year": max_year}

    def get_disturbance_timeline(self) -> dict:
        """Return fire and blowdown events for timeline display.

        Returns ``{"fires": [{year, x, y, cells_burned}, ...],
        "blowdowns": [{year, x, y, cells_affected}, ...]}``.
        """
        # Query all events across the full world extent and all time
        all_events = self._world.events.get_events_in_region(
            0, 0, 50_000, 50_000
        )

        fires = []
        blowdowns = []
        for ev in all_events:
            if ev["event_type"] == "fire":
                entry = {
                    "year": ev["year"],
                    "x": ev["x"],
                    "y": ev["y"],
                    "cells_burned": ev["data"].get("cells_burned", 0) if ev["data"] else 0,
                }
                fires.append(entry)
            elif ev["event_type"] == "blowdown":
                entry = {
                    "year": ev["year"],
                    "x": ev["x"],
                    "y": ev["y"],
                    "cells_affected": ev["data"].get("cells_affected", 0) if ev["data"] else 0,
                }
                blowdowns.append(entry)

        return {"fires": fires, "blowdowns": blowdowns}
