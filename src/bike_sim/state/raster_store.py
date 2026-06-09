"""Zarr-backed storage for named raster layers, organized by simulation tier.

The RasterStore manages 2D numpy arrays (heightmaps, density fields, etc.)
persisted to disk via Zarr. Layers are organized into tier groups — each
simulation tier (geology, climate-hydrology, ecology) gets its own namespace,
so "heightmap" in geology and "heightmap" in ecology are independent.

Without versioning (legacy mode), the on-disk structure is flat:
    rasters/
      geology/
        heightmap      (Zarr array)
        bedrock_type   (Zarr array)

With versioning enabled (after calling set_version), each version gets its own
group, and only layers actually written during that version appear there
(copy-on-write):
    rasters/
      v0001/
        geology/
          heightmap    (Zarr array)
      v0002/
        geology/
          bedrock_type (Zarr array)

Zarr handles chunking, compression, and efficient partial reads for us.
We provide naming, tier-grouping, and version management on top.
"""

from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np
import zarr


class RasterStore:
    """Named raster layers organized by tier, backed by a Zarr store on disk."""

    def __init__(self, root: zarr.Group) -> None:
        self._root = root
        self._current_version: int | None = None
        # "tier/layer" -> sorted list of versions that wrote it
        self._layer_history: dict[str, list[int]] = {}

    @classmethod
    def create(cls, path: Path) -> RasterStore:
        """Create a new RasterStore at the given path."""
        root = zarr.open_group(str(path), mode="w")
        return cls(root)

    @classmethod
    def open(cls, path: Path) -> RasterStore:
        """Open an existing RasterStore from disk."""
        root = zarr.open_group(str(path), mode="r+")
        store = cls(root)
        store._rebuild_layer_history()
        return store

    def _rebuild_layer_history(self) -> None:
        """Scan the Zarr root for version groups and rebuild _layer_history."""
        for key in sorted(self._root.keys()):
            if key.startswith("v") and len(key) == 5 and key[1:].isdigit():
                version = int(key[1:])
                version_group = self._root[key]
                for tier_name in version_group.keys():
                    tier_group = version_group[tier_name]
                    # Only iterate arrays (layers), not sub-groups
                    for layer_name in tier_group.keys():
                        layer_key = f"{tier_name}/{layer_name}"
                        if layer_key not in self._layer_history:
                            self._layer_history[layer_key] = []
                        history = self._layer_history[layer_key]
                        if version not in history:
                            bisect.insort(history, version)

    def set_version(self, version_id: int) -> None:
        """Set the active version for subsequent writes."""
        self._current_version = version_id

    def _version_group_name(self, version: int) -> str:
        """Return the Zarr group name for a version, e.g. 'v0001'."""
        return f"v{version:04d}"

    def write_layer(
        self,
        tier: str,
        name: str,
        data: np.ndarray,
        tick_number: int,
    ) -> None:
        """Write a 2D array as a named layer within a tier group.

        If versioning is active (set_version has been called), writes to
        root/v{version}/tier/name. Otherwise falls back to the flat
        root/tier/name layout for backward compatibility.

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
        if self._current_version is None:
            # Legacy (unversioned) mode
            self._write_layer_flat(tier, name, data, tick_number)
        else:
            self._write_layer_versioned(tier, name, data, tick_number)

    def _write_layer_flat(self, tier: str, name: str, data: np.ndarray, tick_number: int) -> None:
        """Write to the flat root/tier/name layout (legacy mode)."""
        if tier not in self._root:
            tier_group = self._root.create_group(tier)
        else:
            tier_group = self._root[tier]

        if name in tier_group:
            del tier_group[name]

        arr = tier_group.create_array(
            name=name,
            data=data,
            chunks=_choose_chunks(data.shape),
        )
        arr.attrs["tick_number"] = tick_number

    def _write_layer_versioned(
        self, tier: str, name: str, data: np.ndarray, tick_number: int
    ) -> None:
        """Write to root/v{version}/tier/name and update layer history."""
        version = self._current_version
        assert version is not None

        vgroup_name = self._version_group_name(version)
        if vgroup_name not in self._root:
            vgroup = self._root.create_group(vgroup_name)
        else:
            vgroup = self._root[vgroup_name]

        if tier not in vgroup:
            tier_group = vgroup.create_group(tier)
        else:
            tier_group = vgroup[tier]

        if name in tier_group:
            del tier_group[name]

        arr = tier_group.create_array(
            name=name,
            data=data,
            chunks=_choose_chunks(data.shape),
        )
        arr.attrs["tick_number"] = tick_number

        # Update layer history
        layer_key = f"{tier}/{name}"
        if layer_key not in self._layer_history:
            self._layer_history[layer_key] = []
        history = self._layer_history[layer_key]
        if version not in history:
            bisect.insort(history, version)

    def read_layer(self, tier: str, name: str, version: int | None = None) -> np.ndarray:
        """Read a named layer from a tier group, returning a numpy array.

        If version is specified, finds the most recent version <= the requested
        version that wrote this layer. If version is None and versioning is
        active, returns the latest available data. If versioning has never been
        used, falls back to the flat layout.
        """
        if not self._layer_history:
            # Legacy (unversioned) mode — flat layout
            return np.array(self._root[tier][name])

        layer_key = f"{tier}/{name}"
        if layer_key not in self._layer_history:
            raise KeyError(f"Layer '{layer_key}' not found in any version")

        history = self._layer_history[layer_key]

        if version is None:
            # Return the latest version
            owning_version = history[-1]
        else:
            # Find the max version in history that is <= version
            idx = bisect.bisect_right(history, version) - 1
            if idx < 0:
                raise KeyError(
                    f"Layer '{layer_key}' does not exist at or before version {version}"
                )
            owning_version = history[idx]

        vgroup_name = self._version_group_name(owning_version)
        return np.array(self._root[vgroup_name][tier][name])

    def list_layers(self, tier: str, version: int | None = None) -> list[str]:
        """List all layer names within a tier group.

        If versioning is active, returns all layers for this tier that exist
        at or before the given version. If version is None, returns all known
        layers. Falls back to flat layout if versioning was never used.
        """
        if not self._layer_history:
            # Legacy mode
            if tier not in self._root:
                return []
            return sorted(self._root[tier].keys())

        result = []
        prefix = f"{tier}/"
        for layer_key, history in self._layer_history.items():
            if not layer_key.startswith(prefix):
                continue
            layer_name = layer_key[len(prefix) :]
            if version is None:
                result.append(layer_name)
            else:
                # Check if any version <= requested version wrote this layer
                idx = bisect.bisect_right(history, version) - 1
                if idx >= 0:
                    result.append(layer_name)

        return sorted(result)

    def list_tiers(self) -> list[str]:
        """Return sorted list of tier names that have layers stored."""
        if not self._layer_history:
            # Legacy mode: root keys are tier names directly
            return sorted(
                k
                for k in self._root.keys()
                if not (k.startswith("v") and len(k) == 5 and k[1:].isdigit())
            )
        tiers: set[str] = set()
        for layer_key in self._layer_history:
            tiers.add(layer_key.split("/")[0])
        return sorted(tiers)

    def list_versions(self) -> list[int]:
        """Return sorted list of version IDs that have been written to."""
        versions: set[int] = set()
        for history in self._layer_history.values():
            versions.update(history)
        return sorted(versions)

    def get_layer_version(self, tier: str, name: str) -> int:
        """Return which version owns the latest copy of this layer."""
        layer_key = f"{tier}/{name}"
        if layer_key not in self._layer_history or not self._layer_history[layer_key]:
            raise KeyError(f"Layer '{layer_key}' not found")
        return self._layer_history[layer_key][-1]

    def get_layer_index(self) -> dict[str, int]:
        """Return the full layer ownership index: {"tier/name": latest_version}."""
        return {key: history[-1] for key, history in self._layer_history.items() if history}


def _choose_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Pick chunk sizes for a given array shape.

    For the 50m-resolution world (1000x1000), 256x256 chunks give ~16 chunks
    per layer — a good balance between I/O granularity and overhead. For
    smaller arrays (tests, coarse grids), just use the full shape.
    """
    chunk_target = 256
    return tuple(min(s, chunk_target) for s in shape)
