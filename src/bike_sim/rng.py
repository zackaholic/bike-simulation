"""Deterministic RNG substream infrastructure.

Every random draw in the simulation must be reproducible from the world seed.
This module provides the mechanism: given a world seed and a context tuple
like ("ecology", "fire_ignition", 42), it produces a numpy Generator that
is deterministic and statistically independent from every other substream.

Why this matters:
  - Replaying from seed reproduces bit-identical worlds.
  - Changing code in one tier/pass doesn't alter randomness in another.
  - Debugging can replay a single tick's randomness in isolation.

How it works:
  numpy's SeedSequence is designed for exactly this pattern. You give it a
  base entropy (our world seed) and a "spawn key" (a tuple of integers).
  It uses a mixing function to derive a new seed that is statistically
  independent from any other spawn key. We convert string identifiers
  (tier names, pass names) into stable integers via hashing so the spawn
  key is always a tuple of ints.

Usage:
  rng = create_rng(world_seed=12345, tier_id="ecology",
                   pass_id="dispersal", tick_number=42)
  values = rng.random(100)   # deterministic, independent substream
"""

from __future__ import annotations

import hashlib

from numpy.random import PCG64, Generator, SeedSequence


def _stable_hash(s: str) -> int:
    """Convert a string to a stable integer for use in SeedSequence spawn keys.

    We use SHA-256 truncated to 64 bits rather than Python's built-in hash(),
    because hash() is randomized across Python processes (PYTHONHASHSEED).
    That would break reproducibility.
    """
    return int.from_bytes(
        hashlib.sha256(s.encode("utf-8")).digest()[:8],
        byteorder="little",
    )


def create_rng(
    world_seed: int,
    tier_id: str,
    pass_id: str,
    tick_number: int,
) -> Generator:
    """Create a deterministic RNG for a specific simulation context.

    Parameters
    ----------
    world_seed : int
        The world's master seed.
    tier_id : str
        Which tier is drawing ("geology", "climate_hydrology", "ecology").
    pass_id : str
        Which pass within the tier ("erosion", "fire_ignition", "dispersal", etc.).
    tick_number : int
        The current tick number for this tier.

    Returns
    -------
    numpy.random.Generator
        A fresh Generator whose output is fully determined by the inputs
        and statistically independent from any other input combination.
    """
    spawn_key = (_stable_hash(tier_id), _stable_hash(pass_id), tick_number)
    seq = SeedSequence(world_seed, spawn_key=spawn_key)
    return Generator(PCG64(seq))
