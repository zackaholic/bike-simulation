"""Build terrain mesh geometry for a single chunk.

Each chunk is a grid of vertices sampled along the path using the BakedCurve.
Vertices are stored relative to the chunk anchor (world position at chunk center)
to avoid floating-point precision loss on long paths.

Ported from bike-trainer-godot/world-builder/terrain.py with modifications:
- Removed splatmap vertex color encoding (quality_map)
- Added color_fn(absolute_elevation) for elevation-based coloring
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from bike_sim.extract.godot.baked_curve import BakedCurve

CHUNK_LENGTH: float = 150.0  # meters along path
CHUNK_WIDTH: float = 300.0  # meters across path (±150m from center)
GRID_COLS: int = 25  # vertices across width
GRID_ROWS: int = 21  # vertices along length


def elevation_color(elev: float) -> np.ndarray:
    """Default elevation-based vertex color: green → brown → grey.

    Args:
        elev: absolute elevation in meters.

    Returns:
        RGBA as float32 array.
    """
    t = np.clip(elev / 2000.0, 0.0, 1.0)
    if t < 0.3:
        # 0–600m: green
        return np.array([0.15, 0.50, 0.12, 1.0], dtype=np.float32)
    elif t < 0.6:
        # 600–1200m: blend green → brown
        s = (t - 0.3) / 0.3
        green = np.array([0.15, 0.50, 0.12, 1.0], dtype=np.float32)
        brown = np.array([0.55, 0.40, 0.25, 1.0], dtype=np.float32)
        return (green * (1 - s) + brown * s).astype(np.float32)
    else:
        # 1200–2000m: blend brown → grey
        s = (t - 0.6) / 0.4
        brown = np.array([0.55, 0.40, 0.25, 1.0], dtype=np.float32)
        grey = np.array([0.65, 0.63, 0.60, 1.0], dtype=np.float32)
        return (brown * (1 - s) + grey * s).astype(np.float32)


def _compute_normals(verts: np.ndarray, indices: np.ndarray, n_verts: int) -> np.ndarray:
    """Compute smooth per-vertex normals by averaging adjacent face normals."""
    normals = np.zeros((n_verts, 3), dtype=np.float32)
    counts = np.zeros(n_verts, dtype=np.float32)

    for i in range(0, len(indices), 3):
        i0, i1, i2 = indices[i], indices[i + 1], indices[i + 2]
        v0, v1, v2 = verts[i0], verts[i1], verts[i2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        fn_len = np.linalg.norm(face_normal)
        if fn_len > 1e-10:
            face_normal /= fn_len
        for idx in (i0, i1, i2):
            normals[idx] += face_normal
            counts[idx] += 1

    mask = counts > 0
    normals[mask] /= counts[mask, np.newaxis]
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-8, 1.0, lengths)
    normals /= lengths
    return normals.astype(np.float32)


def build_chunk_mesh(
    curve: BakedCurve,
    chunk_start: float,
    height_fn: Callable[[float, float], float],
    color_fn: Callable[[float], np.ndarray] | None = None,
) -> dict:
    """Build terrain mesh data for a chunk starting at *chunk_start* meters.

    Args:
        curve: BakedCurve describing the world path.
        chunk_start: distance along path where this chunk begins.
        height_fn: callable(distance, lateral) → float, returns terrain height
            relative to the path at that distance.
        color_fn: optional callable(absolute_elevation) → RGBA float32[4].
            Defaults to :func:`elevation_color`.

    Returns:
        dict with keys: vertices, normals, uvs, colors, indices, anchor.
    """
    if color_fn is None:
        color_fn = elevation_color

    anchor_dist = chunk_start + CHUNK_LENGTH * 0.5
    anchor = curve.sample_position(anchor_dist)

    rows = GRID_ROWS
    cols = GRID_COLS

    verts = np.zeros((rows * cols, 3), dtype=np.float32)
    uvs = np.zeros((rows * cols, 2), dtype=np.float32)
    colors = np.zeros((rows * cols, 4), dtype=np.float32)

    for r in range(rows):
        d = chunk_start + (r / (rows - 1)) * CHUNK_LENGTH
        d = min(d, curve.total_length)
        path_y = curve.sample_position(d)[1]
        for c in range(cols):
            lat = (c / (cols - 1) - 0.5) * CHUNK_WIDTH
            h = height_fn(d, lat)
            world_pos = curve.world_point(d, lat, h)
            idx = r * cols + c
            verts[idx] = (world_pos - anchor).astype(np.float32)
            uvs[idx] = [c / (cols - 1), r / (rows - 1)]
            absolute_elev = path_y + h
            colors[idx] = color_fn(absolute_elev)

    # Triangle indices — CCW winding for upward-facing normals in Godot
    n_quads = (rows - 1) * (cols - 1)
    indices = np.zeros(n_quads * 6, dtype=np.int32)
    quad_idx = 0
    for r in range(rows - 1):
        for c in range(cols - 1):
            i = r * cols + c
            i0, i1, i2, i3 = i, i + cols, i + 1, i + cols + 1
            indices[quad_idx * 6 + 0] = i0
            indices[quad_idx * 6 + 1] = i1
            indices[quad_idx * 6 + 2] = i2
            indices[quad_idx * 6 + 3] = i2
            indices[quad_idx * 6 + 4] = i1
            indices[quad_idx * 6 + 5] = i3
            quad_idx += 1

    normals = _compute_normals(verts, indices, rows * cols)

    return {
        "vertices": verts,
        "normals": normals,
        "uvs": uvs.astype(np.float32),
        "colors": colors,
        "indices": indices,
        "anchor": anchor,
    }
