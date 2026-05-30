"""The World class — central data model for a simulated world.

A World is the metadata envelope that ties together all simulation state.
The heavy data (raster layers, species databases, event logs) lives in
Zarr stores and SQLite databases alongside the World manifest; this class
tracks the seed, tier clocks, and version info needed to interpret that data.

A world directory on disk looks like:
    my_world/
      world.json     # manifest (seed, tier clocks, version)
      rasters/       # Zarr store (RasterStore)
      events.db      # SQLite database (EventStore)

The world seed is the single number from which all randomness is derived.
Combined with the simulation code and the sequence of advancement triggers,
it fully determines the world's history. See rng.py for how substreams
are derived from the seed.
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

from bike_sim.state.event_store import EventStore
from bike_sim.state.raster_store import RasterStore


class TierId(StrEnum):
    """Identifiers for the three simulation tiers.

    These are used as keys in tier clocks and as part of the RNG substream
    derivation: (world_seed, tier_id, pass_id, tick_number).
    """

    GEOLOGY = "geology"
    CLIMATE_HYDROLOGY = "climate_hydrology"
    ECOLOGY = "ecology"


@dataclass
class TierClock:
    """Tracks simulation progress for a single tier.

    Each tier advances at its own timescale:
      - Geology: 10K-100K years per tick
      - Climate-hydrology: 10-100 years per tick
      - Ecology: 1 season to a few years per tick

    tick_number increments each time the tier runs. simulated_year tracks
    what year in the world's history this tier has been advanced to.
    """

    tick_number: int = 0
    simulated_year: float = 0.0


def _default_tier_clocks() -> dict[str, TierClock]:
    return {tier.value: TierClock() for tier in TierId}


@dataclass
class World:
    """Central metadata for a simulated world.

    This is the manifest that ties together the seed, tier state, and
    pointers to the heavy data stores. It serializes to a small JSON file;
    the raster data (Zarr) and individual/event data (SQLite) live alongside
    it in the world directory.

    Use World.create() to initialize a new world directory, or World.open()
    to load an existing one. Both return a World with .rasters and .events
    attributes ready to use.
    """

    seed: int
    tier_clocks: dict[str, TierClock] = field(default_factory=_default_tier_clocks)
    simulated_year: float = 0.0
    version: int = 1
    rasters: RasterStore | None = field(default=None, repr=False)
    events: EventStore | None = field(default=None, repr=False)
    _version_log: list[dict] = field(default_factory=list, repr=False)
    _current_version: int = field(default=-1, repr=False)

    @classmethod
    def create(cls, path: Path, seed: int) -> World:
        """Create a new world directory with empty stores.

        Sets up the directory structure, initializes the RasterStore and
        EventStore, writes the manifest, and returns a ready-to-use World.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        rasters = RasterStore.create(path / "rasters")
        events = EventStore.create(path / "events.db")

        world = cls(seed=seed, rasters=rasters, events=events)
        world.save(path / "world.json")
        return world

    @classmethod
    def open(cls, path: Path) -> World:
        """Open an existing world directory.

        Loads the manifest and connects to both stores.
        """
        path = Path(path)
        manifest = json.loads((path / "world.json").read_text())
        tier_clocks = {
            name: TierClock(**clock_data) for name, clock_data in manifest["tier_clocks"].items()
        }

        rasters = RasterStore.open(path / "rasters")
        events = EventStore.open(path / "events.db")

        world = cls(
            seed=manifest["seed"],
            tier_clocks=tier_clocks,
            simulated_year=manifest["simulated_year"],
            version=manifest["version"],
            rasters=rasters,
            events=events,
        )
        world._version_log = manifest.get("versions", [])
        world._current_version = manifest.get("current_version", -1)
        return world

    def close(self) -> None:
        """Close both data stores. Safe to call multiple times."""
        if self.events is not None:
            try:
                self.events.close()
            except Exception:
                pass
            self.events = None
        # RasterStore (Zarr) doesn't need explicit closing, but we clear
        # the reference for consistency.
        self.rasters = None

    @property
    def current_version(self) -> int:
        """The latest version ID, or -1 if no versions committed."""
        return self._current_version

    def commit_version(self, trigger: str) -> int:
        """Create a new version snapshot.

        Records: version_id, tier_clocks snapshot, timestamp, trigger string,
        and layer_index from the RasterStore.

        Returns the new version_id.
        """
        new_id = self._current_version + 1

        entry = {
            "version_id": new_id,
            "tier_clocks": {
                name: {
                    "tick_number": clock.tick_number,
                    "simulated_year": clock.simulated_year,
                }
                for name, clock in self.tier_clocks.items()
            },
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "trigger": trigger,
            "layer_index": self.rasters.get_layer_index() if self.rasters else {},
        }

        self._version_log.append(entry)
        self._current_version = new_id
        return new_id

    def get_version(self, version_id: int) -> dict:
        """Return the version entry for a given version_id."""
        for entry in self._version_log:
            if entry["version_id"] == version_id:
                return entry
        raise KeyError(f"Version {version_id} not found")

    def list_versions(self) -> list[dict]:
        """Return all version entries."""
        return list(self._version_log)

    def version_for_tier(self, version_id: int, tier: str) -> TierClock:
        """Return the TierClock snapshot for a given tier at a given version."""
        entry = self.get_version(version_id)
        clock_data = entry["tier_clocks"][tier]
        return TierClock(**clock_data)

    def to_dict(self) -> dict:
        """Serialize manifest fields to a plain dict (JSON-compatible).

        Does not include the store references — those are managed by the
        world directory structure, not the manifest.
        """
        return {
            "seed": self.seed,
            "tier_clocks": {name: asdict(clock) for name, clock in self.tier_clocks.items()},
            "simulated_year": self.simulated_year,
            "version": self.version,
            "versions": self._version_log,
            "current_version": self._current_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> World:
        """Deserialize from a plain dict (manifest only, no stores)."""
        tier_clocks = {
            name: TierClock(**clock_data) for name, clock_data in data["tier_clocks"].items()
        }
        world = cls(
            seed=data["seed"],
            tier_clocks=tier_clocks,
            simulated_year=data["simulated_year"],
            version=data["version"],
        )
        world._version_log = data.get("versions", [])
        world._current_version = data.get("current_version", -1)
        return world

    def save(self, path: Path) -> None:
        """Write the world manifest atomically (write to temp, then rename)."""
        content = json.dumps(self.to_dict(), indent=2) + "\n"
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            os.write(fd, content.encode())
            os.close(fd)
            os.replace(tmp_path, str(path))
        except BaseException:
            os.close(fd)
            os.unlink(tmp_path)
            raise

    @classmethod
    def load(cls, path: Path) -> World:
        """Load a world manifest from a JSON file (manifest only, no stores)."""
        data = json.loads(path.read_text())
        return cls.from_dict(data)
