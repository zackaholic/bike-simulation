"""The World class — central data model for a simulated world.

A World is the metadata envelope that ties together all simulation state.
The heavy data (raster layers, species databases, event logs) lives in
Zarr stores and SQLite databases alongside the World manifest; this class
tracks the seed, tier clocks, and version info needed to interpret that data.

The world seed is the single number from which all randomness is derived.
Combined with the simulation code and the sequence of advancement triggers,
it fully determines the world's history. See rng.py for how substreams
are derived from the seed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path


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
    """

    seed: int
    tier_clocks: dict[str, TierClock] = field(default_factory=_default_tier_clocks)
    simulated_year: float = 0.0
    version: int = 1

    def to_dict(self) -> dict:
        """Serialize to a plain dict (JSON-compatible)."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> World:
        """Deserialize from a plain dict."""
        tier_clocks = {
            name: TierClock(**clock_data) for name, clock_data in data["tier_clocks"].items()
        }
        return cls(
            seed=data["seed"],
            tier_clocks=tier_clocks,
            simulated_year=data["simulated_year"],
            version=data["version"],
        )

    def save(self, path: Path) -> None:
        """Write the world manifest to a JSON file."""
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> World:
        """Load a world manifest from a JSON file."""
        data = json.loads(path.read_text())
        return cls.from_dict(data)
