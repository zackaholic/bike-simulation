"""Tile rendering for the webview world inspector.

Converts raster layers into 256x256 PNG tiles using Pillow.
Each tile is cached on the filesystem so it's only rendered once
per (version, tier, layer, zoom, x, y) combination.

Zoom levels:
    0 = 1x1 grid   (whole world in 1 tile)
    1 = 2x2 grid
    2 = 4x4 grid
    3 = 8x8 grid
    4 = 16x16 grid  (~12m per pixel)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from bike_sim.query.world_query import WorldQuery

TILE_SIZE = 256
MAX_ZOOM = 4
WORLD_EXTENT = 50_000.0  # meters


# ── Colormaps ─────────────────────────────────────────────────────

# Simple linear colormaps as (R, G, B) arrays for common layer types.
# Each maps normalized [0, 1] values to colors. We build 256-entry
# lookup tables for fast indexing.


def _lerp_color(t: float, c0: tuple[int, ...], c1: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(c0[i] + t * (c1[i] - c0[i])) for i in range(3))


def _build_lut(stops: list[tuple[float, tuple[int, ...]]]) -> np.ndarray:
    """Build a 256x3 uint8 lookup table from color stops."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        # Find bounding stops
        for s in range(len(stops) - 1):
            t0, c0 = stops[s]
            t1, c1 = stops[s + 1]
            if t0 <= t <= t1:
                frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                lut[i] = _lerp_color(frac, c0, c1)
                break
        else:
            lut[i] = stops[-1][1]
    return lut


# Terrain: blue (low) → green → brown → white (high)
_LUT_TERRAIN = _build_lut([
    (0.0, (20, 60, 120)),
    (0.15, (50, 120, 60)),
    (0.4, (80, 160, 50)),
    (0.65, (160, 140, 80)),
    (0.85, (180, 160, 140)),
    (1.0, (240, 240, 240)),
])

# Viridis-like: dark purple → teal → yellow
_LUT_VIRIDIS = _build_lut([
    (0.0, (68, 1, 84)),
    (0.25, (59, 82, 139)),
    (0.5, (33, 145, 140)),
    (0.75, (94, 201, 98)),
    (1.0, (253, 231, 37)),
])

# Blues: white → dark blue (for moisture, precipitation, flow)
_LUT_BLUES = _build_lut([
    (0.0, (240, 240, 255)),
    (0.5, (100, 140, 210)),
    (1.0, (10, 30, 100)),
])

# Greens: light → dark green (for vegetation density)
_LUT_GREENS = _build_lut([
    (0.0, (240, 250, 230)),
    (0.5, (80, 180, 60)),
    (1.0, (10, 80, 20)),
])

# Categorical: discrete colors for bedrock type, soil type, etc.
_LUT_CATEGORICAL = np.array([
    [180, 120, 80],   # 0 - sandstone
    [140, 140, 140],  # 1 - granite
    [100, 100, 120],  # 2 - shale
    [200, 200, 180],  # 3 - limestone
    [60, 60, 80],     # 4 - basalt
    [160, 130, 140],  # 5 - gneiss
    [120, 100, 100],  # 6 - default
    [150, 150, 130],  # 7
], dtype=np.uint8)

# Layer name → LUT mapping
_LAYER_LUTS: dict[str, np.ndarray] = {
    "heightmap": _LUT_TERRAIN,
    "bedrock_type": _LUT_CATEGORICAL,
    "soil_parent": _LUT_CATEGORICAL,
    "temperature": _LUT_VIRIDIS,
    "precipitation": _LUT_BLUES,
    "flow_accumulation": _LUT_BLUES,
    "solar_insolation": _LUT_VIRIDIS,
    "soil_moisture_summer": _LUT_BLUES,
    "soil_moisture_winter": _LUT_BLUES,
    "frost_pocket": _LUT_BLUES,
    "growing_degree_days": _LUT_VIRIDIS,
    "distance_to_water": _LUT_BLUES,
}


def _get_lut(layer_name: str) -> np.ndarray:
    """Return the appropriate LUT for a layer, defaulting to viridis."""
    # Species density layers use greens
    if "density" in layer_name:
        return _LUT_GREENS
    return _LAYER_LUTS.get(layer_name, _LUT_VIRIDIS)


# ── Tile coordinate math ─────────────────────────────────────────


def tile_bbox(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Return the world-coordinate bounding box for a tile.

    Returns (x_min, y_min, x_max, y_max) in meters.
    Tile (0, 0) is the bottom-left corner of the world.
    """
    n = 2 ** z
    tile_size_m = WORLD_EXTENT / n
    x_min = x * tile_size_m
    y_min = y * tile_size_m
    return (x_min, y_min, x_min + tile_size_m, y_min + tile_size_m)


def tile_valid(z: int, x: int, y: int) -> bool:
    """Check whether tile coordinates are within valid range."""
    if z < 0 or z > MAX_ZOOM:
        return False
    n = 2 ** z
    return 0 <= x < n and 0 <= y < n


# ── Rendering ─────────────────────────────────────────────────────

GRID_SIZE = 1000

# Cache full raster + (vmin, vmax) per (version, tier, layer_name) so all
# tiles in a layer use the same data and normalization range.
_layer_cache: dict[tuple[int, str, str], tuple[np.ndarray, float, float]] = {}


def _get_layer_data(
    query: WorldQuery, version: int, tier: str, layer_name: str
) -> tuple[np.ndarray, float, float]:
    """Return (full_raster, vmin, vmax) for the entire layer, cached."""
    key = (version, tier, layer_name)
    if key not in _layer_cache:
        full = query.query_raster(
            version, tier, layer_name,
            (0, 0, WORLD_EXTENT, WORLD_EXTENT),
            (GRID_SIZE, GRID_SIZE),
        )
        _layer_cache[key] = (full, float(np.nanmin(full)), float(np.nanmax(full)))
    return _layer_cache[key]


def _normalize(
    data: np.ndarray, layer_name: str, vmin: float, vmax: float
) -> np.ndarray:
    """Normalize array to 0-255 uint8 indices for LUT lookup."""
    lut = _get_lut(layer_name)

    # Categorical layers: use values directly as indices
    if lut is _LUT_CATEGORICAL:
        indices = np.clip(data.astype(int), 0, len(lut) - 1)
        return indices.astype(np.uint8)

    # Continuous layers: normalize to [0, 1] using global range
    if vmax - vmin < 1e-10:
        return np.zeros(data.shape, dtype=np.uint8)

    normalized = (data - vmin) / (vmax - vmin)
    return (np.clip(normalized, 0.0, 1.0) * 255).astype(np.uint8)


def render_tile(
    query: WorldQuery,
    version: int,
    tier: str,
    layer_name: str,
    z: int,
    x: int,
    y: int,
    cache_dir: Path,
) -> bytes:
    """Render a single tile as PNG bytes, using cache if available.

    Returns raw PNG bytes suitable for an HTTP response.
    """
    # Check cache first
    cache_path = cache_dir / str(version) / tier / layer_name / str(z) / str(x) / f"{y}.png"
    if cache_path.exists():
        return cache_path.read_bytes()

    # Get full layer (cached) for consistent normalization and clean partitioning
    full_raster, vmin, vmax = _get_layer_data(query, version, tier, layer_name)

    # Compute clean cell partition — no overlap between adjacent tiles.
    # Flip Y: Leaflet standard has y=0 at top (north), but our raster
    # has row 0 at bottom (south). Flip so server y matches Leaflet y.
    n = 2 ** z
    flipped_y = n - 1 - y
    row_start = flipped_y * GRID_SIZE // n
    row_end = (flipped_y + 1) * GRID_SIZE // n
    col_start = x * GRID_SIZE // n
    col_end = (x + 1) * GRID_SIZE // n
    crop = full_raster[row_start:row_end, col_start:col_end]

    # Resample crop to tile size using nearest-neighbor
    src_rows, src_cols = crop.shape
    row_idx = (np.arange(TILE_SIZE) * src_rows / TILE_SIZE).astype(int)
    col_idx = (np.arange(TILE_SIZE) * src_cols / TILE_SIZE).astype(int)
    row_idx = np.clip(row_idx, 0, src_rows - 1)
    col_idx = np.clip(col_idx, 0, src_cols - 1)
    data = crop[np.ix_(row_idx, col_idx)]

    # Apply colormap with global normalization
    indices = _normalize(data, layer_name, vmin, vmax)
    lut = _get_lut(layer_name)
    rgb = lut[indices]  # (256, 256, 3)

    # Flip vertically: raster row 0 is south (bottom), but image row 0 is top
    rgb = rgb[::-1]

    # Encode as PNG via Pillow
    img = Image.fromarray(rgb, mode="RGB")

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(cache_path, format="PNG")

    return cache_path.read_bytes()
