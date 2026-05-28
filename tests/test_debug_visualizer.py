"""Tests for the debug 2D visualizer module.

These tests verify that the visualizer produces valid PNG files from world
state. They test output shape (file exists, non-empty), not aesthetic
correctness — that's what looking at the images is for.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from bike_sim.extract.debug2d.visualizer import (
    render_composite,
    render_individuals,
    render_layer,
)
from bike_sim.world import World


@pytest.fixture()
def world(tmp_path):
    """Create a world with a geology heightmap already written."""
    w = World.create(tmp_path / "world", seed=42)
    yield w
    w.close()


def _write_heightmap(world, shape=(1000, 1000)):
    """Helper: write a gradient heightmap to the geology tier."""
    data = np.linspace(0, 3000, shape[0] * shape[1]).reshape(shape).astype(np.float64)
    world.rasters.write_layer("geology", "heightmap", data, tick_number=0)
    return data


def _write_moisture(world, shape=(1000, 1000)):
    """Helper: write a moisture layer to the climate_hydrology tier."""
    rng = np.random.default_rng(123)
    data = rng.random(shape).astype(np.float64)
    world.rasters.write_layer("climate_hydrology", "moisture", data, tick_number=0)
    return data


# ── test_render_layer_creates_png ───────────────────────────────────


def test_render_layer_creates_png(world, tmp_path):
    """Rendering a single raster layer produces a non-empty PNG file."""
    _write_heightmap(world)
    out = tmp_path / "heightmap.png"

    render_layer(world, tier="geology", layer_name="heightmap", output_path=out)

    assert out.exists()
    assert out.stat().st_size > 0


# ── test_render_layer_with_title ────────────────────────────────────


def test_render_layer_with_title(world, tmp_path):
    """Rendering with a custom title string doesn't crash."""
    _write_heightmap(world)
    out = tmp_path / "titled.png"

    render_layer(
        world,
        tier="geology",
        layer_name="heightmap",
        output_path=out,
        title="Heightmap — geology tick 0",
    )

    assert out.exists()
    assert out.stat().st_size > 0


# ── test_render_layer_region_zoom ───────────────────────────────────


def test_render_layer_region_zoom(world, tmp_path):
    """Rendering with a region tuple zooms to a subregion and produces a file."""
    _write_heightmap(world)
    out = tmp_path / "zoomed.png"

    render_layer(
        world,
        tier="geology",
        layer_name="heightmap",
        output_path=out,
        region=(5000, 5000, 15000, 15000),
    )

    assert out.exists()
    assert out.stat().st_size > 0


# ── test_render_composite_creates_png ───────────────────────────────


def test_render_composite_creates_png(world, tmp_path):
    """Compositing two layers produces a non-empty PNG."""
    _write_heightmap(world)
    _write_moisture(world)
    out = tmp_path / "composite.png"

    layers = [
        {"tier": "geology", "layer_name": "heightmap", "cmap": "terrain", "alpha": 1.0},
        {
            "tier": "climate_hydrology",
            "layer_name": "moisture",
            "cmap": "Blues",
            "alpha": 0.5,
        },
    ]
    render_composite(world, layers=layers, output_path=out)

    assert out.exists()
    assert out.stat().st_size > 0


# ── test_render_individuals_creates_png ─────────────────────────────


def test_render_individuals_creates_png(world, tmp_path):
    """Rendering distinguished individuals on a background produces a file."""
    _write_heightmap(world)

    world.events.add_species("oak", {"height": 20.0})
    world.events.add_individual("mother_oak", "oak", x=12500.0, y=25000.0, appeared_year=800.0)
    world.events.add_individual("old_elm", "oak", x=37000.0, y=10000.0, appeared_year=600.0)

    out = tmp_path / "individuals.png"

    render_individuals(
        world,
        output_path=out,
        background_tier="geology",
        background_layer="heightmap",
    )

    assert out.exists()
    assert out.stat().st_size > 0


# ── test_render_layer_missing_layer_raises ──────────────────────────


def test_render_layer_missing_layer_raises(world, tmp_path):
    """Rendering a layer that doesn't exist raises an error."""
    out = tmp_path / "missing.png"

    with pytest.raises((KeyError, Exception)):
        render_layer(world, tier="geology", layer_name="nonexistent", output_path=out)


# ── test_render_layer_different_cmaps ───────────────────────────────


@pytest.mark.parametrize("cmap", ["terrain", "viridis", "Blues"])
def test_render_layer_different_cmaps(world, tmp_path, cmap):
    """Rendering with different colormaps all produce valid PNGs."""
    _write_heightmap(world)
    out = tmp_path / f"heightmap_{cmap}.png"

    render_layer(
        world,
        tier="geology",
        layer_name="heightmap",
        output_path=out,
        cmap=cmap,
    )

    assert out.exists()
    assert out.stat().st_size > 0
