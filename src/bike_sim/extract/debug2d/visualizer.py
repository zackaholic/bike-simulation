"""Debug 2D visualizer for world simulation state.

Development "eyes" for the simulation: reads world state and produces
informative (not pretty) PNG images of raster layers, composites, and
individual positions. Used during development to verify that simulation
passes produce sensible output.

This is a Layer C extractor — it reads raster data directly from the
World's RasterStore rather than going through the query interface.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# World geometry constants
WORLD_EXTENT = 50_000.0  # 50km in meters
GRID_SIZE = 1000  # 1000x1000 cells
CELL_SIZE = WORLD_EXTENT / GRID_SIZE  # 50m per cell

# Small palette for coloring individuals by species
_SPECIES_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
    "#dcbeff",
]


def _region_to_slice(
    region: tuple[float, float, float, float],
) -> tuple[slice, slice, tuple[float, float, float, float]]:
    """Convert world-coordinate region to array slices and imshow extent.

    Parameters
    ----------
    region : (x_min, y_min, x_max, y_max)
        Bounding box in world coordinates (meters).

    Returns
    -------
    row_slice, col_slice, extent
        Array slices for indexing the raster, and the matplotlib extent
        tuple (left, right, bottom, top) in world coordinates.
    """
    x_min, y_min, x_max, y_max = region

    col_min = max(0, int(x_min / CELL_SIZE))
    col_max = min(GRID_SIZE, int(np.ceil(x_max / CELL_SIZE)))
    row_min = max(0, int(y_min / CELL_SIZE))
    row_max = min(GRID_SIZE, int(np.ceil(y_max / CELL_SIZE)))

    extent = (
        col_min * CELL_SIZE,
        col_max * CELL_SIZE,
        row_min * CELL_SIZE,
        row_max * CELL_SIZE,
    )

    return slice(row_min, row_max), slice(col_min, col_max), extent


def render_layer(
    world,
    tier: str,
    layer_name: str,
    output_path: Path,
    title: str | None = None,
    cmap: str = "terrain",
    region: tuple[float, float, float, float] | None = None,
) -> None:
    """Render a single raster layer as a PNG.

    Args:
        world: World instance with connected stores.
        tier: tier name (e.g. "geology").
        layer_name: layer name (e.g. "heightmap").
        output_path: where to save the PNG.
        title: optional title for the plot (defaults to "tier / layer_name").
        cmap: matplotlib colormap name.
        region: optional (x_min, y_min, x_max, y_max) in world coords to zoom.
    """
    data = world.rasters.read_layer(tier, layer_name)

    if region is not None:
        row_sl, col_sl, extent = _region_to_slice(region)
        data = data[row_sl, col_sl]
    else:
        extent = (0.0, WORLD_EXTENT, 0.0, WORLD_EXTENT)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, cmap=cmap, origin="lower", extent=extent, aspect="equal")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title or f"{tier} / {layer_name}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_composite(
    world,
    layers: list[dict],
    output_path: Path,
    title: str | None = None,
) -> None:
    """Render multiple layers overlaid as a composite PNG.

    Each dict in *layers* has keys: tier, layer_name, cmap, alpha.
    The bottom layer is drawn first (typically fully opaque); subsequent
    layers are overlaid with their specified alpha.

    Args:
        world: World instance with connected stores.
        layers: list of layer specifications.
        output_path: where to save the PNG.
        title: optional title (defaults to "Composite").
    """
    extent = (0.0, WORLD_EXTENT, 0.0, WORLD_EXTENT)

    fig, ax = plt.subplots(figsize=(10, 8))

    for layer_spec in layers:
        data = world.rasters.read_layer(layer_spec["tier"], layer_spec["layer_name"])
        ax.imshow(
            data,
            cmap=layer_spec.get("cmap", "terrain"),
            alpha=layer_spec.get("alpha", 1.0),
            origin="lower",
            extent=extent,
            aspect="equal",
        )

    ax.set_title(title or "Composite")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_individuals(
    world,
    output_path: Path,
    background_tier: str = "geology",
    background_layer: str = "heightmap",
    title: str | None = None,
) -> None:
    """Render distinguished individuals as markers on a background raster.

    Finds all individuals in the world and plots them as colored dots on
    the background layer. Each species gets a distinct color, and points
    are labeled with their individual_id.

    Args:
        world: World instance with connected stores.
        output_path: where to save the PNG.
        background_tier: tier for background raster.
        background_layer: layer name for background raster.
        title: optional title (defaults to "Individuals").
    """
    extent = (0.0, WORLD_EXTENT, 0.0, WORLD_EXTENT)

    bg_data = world.rasters.read_layer(background_tier, background_layer)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(bg_data, cmap="terrain", origin="lower", extent=extent, aspect="equal")

    # Query all individuals (center of world, radius covers everything)
    individuals = world.events.find_individuals_near(25000.0, 25000.0, 50000.0)

    # Build species -> color mapping
    species_color_map: dict[str, str] = {}
    for ind in individuals:
        sp = ind["species_id"]
        if sp not in species_color_map:
            species_color_map[sp] = _SPECIES_COLORS[len(species_color_map) % len(_SPECIES_COLORS)]

    # Plot each individual
    for ind in individuals:
        color = species_color_map[ind["species_id"]]
        ax.scatter(
            ind["x"],
            ind["y"],
            c=color,
            s=30,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )
        ax.annotate(
            ind["individual_id"],
            (ind["x"], ind["y"]),
            fontsize=4,
            textcoords="offset points",
            xytext=(3, 3),
            color="white",
            zorder=6,
        )

    # Add a legend for species colors
    for sp, color in species_color_map.items():
        ax.scatter([], [], c=color, s=30, edgecolors="black", linewidths=0.5, label=sp)
    if species_color_map:
        ax.legend(loc="upper right", fontsize=6, framealpha=0.8)

    ax.set_title(title or "Individuals")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
