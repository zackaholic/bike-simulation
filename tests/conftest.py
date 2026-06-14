"""Shared test fixtures and configuration.

Provides:
- ``--run-slow`` CLI flag to include tests marked ``@pytest.mark.slow``
- ``base_world_path`` session-scoped fixture: geology + climate-hydrology
  ticked once with fast erosion (100 particles).  Copied per-test via
  ``fresh_world`` so every test gets a clean, mutable world without
  paying the geology/climate cost again.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from bike_sim.tiers.climate_hydrology import ClimateHydrologyTier
from bike_sim.tiers.erosion import ErosionParams
from bike_sim.tiers.geology import GeologyTier
from bike_sim.world import World

# Standardised fast erosion params for all tests.
FAST_EROSION = ErosionParams(num_particles=100, max_lifetime=30)


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="include slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="use --run-slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def base_world_path(tmp_path_factory) -> Path:
    """Session-scoped world with geology + climate-hydrology ticked.

    This is the expensive part (~5s).  Individual tests copy this
    directory to get a fresh, mutable world.
    """
    path = tmp_path_factory.mktemp("base") / "world"
    w = World.create(path, seed=42)
    GeologyTier(w).tick()
    ClimateHydrologyTier(w, erosion_params=FAST_EROSION).tick()
    w.save(path / "world.json")
    w.close()
    return path


@pytest.fixture
def fresh_world(base_world_path, tmp_path) -> World:
    """Function-scoped copy of the base world, ready for ecology mutations."""
    dest = tmp_path / "world"
    shutil.copytree(base_world_path, dest)
    return World.open(dest)
