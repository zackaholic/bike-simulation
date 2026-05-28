"""Zarr-backed storage for named raster layers, organized by simulation tier.

The RasterStore manages 2D numpy arrays (heightmaps, density fields, etc.)
persisted to disk via Zarr. Layers are organized into tier groups — each
simulation tier (geology, climate-hydrology, ecology) gets its own namespace,
so "heightmap" in geology and "heightmap" in ecology are independent.

On disk the structure looks like:
    rasters/
      geology/
        heightmap      (Zarr array)
        bedrock_type   (Zarr array)
      ecology/
        density        (Zarr array)

Zarr handles chunking, compression, and efficient partial reads for us.
We just provide the naming and tier-grouping layer on top.

Versioning: each write_layer call takes a tick_number. For v1 this is stored
as array-level metadata (attrs) rather than keeping old versions around.
The architecture supports full version history later if needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr


class RasterStore:
    """Named raster layers organized by tier, backed by a Zarr store on disk."""

    def __init__(self, root: zarr.Group) -> None:
        self._root = root

    @classmethod
    def create(cls, path: Path) -> RasterStore:
        """Create a new RasterStore at the given path."""
        root = zarr.open_group(str(path), mode="w")
        return cls(root)

    @classmethod
    def open(cls, path: Path) -> RasterStore:
        """Open an existing RasterStore from disk."""
        root = zarr.open_group(str(path), mode="r+")
        return cls(root)

    def write_layer(
        self,
        tier: str,
        name: str,
        data: np.ndarray,
        tick_number: int,
    ) -> None:
        """Write a 2D array as a named layer within a tier group.

        If the tier group doesn't exist yet, it's created. If the layer
        already exists, it's overwritten with the new data.

        Parameters
        ----------
        tier : str
            Tier name (e.g. "geology", "climate_hydrology", "ecology").
        name : str
            Layer name (e.g. "heightmap", "soil_moisture_summer").
        data : np.ndarray
            2D array to store.
        tick_number : int
            Which simulation tick produced this data (stored as metadata).
        """
        # Get or create the tier group
        if tier not in self._root:
            tier_group = self._root.create_group(tier)
        else:
            tier_group = self._root[tier]

        # Delete existing layer if present (overwrite)
        if name in tier_group:
            del tier_group[name]

        # Write the array with metadata.
        # Zarr v3: can't pass both data and dtype, so we let it infer dtype from data.
        arr = tier_group.create_array(
            name=name,
            data=data,
            chunks=_choose_chunks(data.shape),
        )
        arr.attrs["tick_number"] = tick_number

    def read_layer(self, tier: str, name: str) -> np.ndarray:
        """Read a named layer from a tier group, returning a numpy array."""
        return np.array(self._root[tier][name])

    def list_layers(self, tier: str) -> list[str]:
        """List all layer names within a tier group.

        Returns an empty list if the tier group doesn't exist yet.
        """
        if tier not in self._root:
            return []
        return list(self._root[tier].keys())


def _choose_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Pick chunk sizes for a given array shape.

    For the 50m-resolution world (1000x1000), 256x256 chunks give ~16 chunks
    per layer — a good balance between I/O granularity and overhead. For
    smaller arrays (tests, coarse grids), just use the full shape.
    """
    chunk_target = 256
    return tuple(min(s, chunk_target) for s in shape)
