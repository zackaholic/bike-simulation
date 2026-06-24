"""Extract a sim world into Godot-compatible chunk files.

Reads the world's heightmap, generates a ride path via A* pathfinding,
converts it to Bezier control points, and writes terrain-only chunks
in the binary format consumed by the Godot ChunkStreamer.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from bike_sim.extract.godot.baked_curve import BakedCurve
from bike_sim.extract.godot.chunk_writer import write_chunk, write_manifest
from bike_sim.extract.godot.path_converter import (
    path_to_bezier,
    sample_heightmap_bilinear,
)
from bike_sim.extract.godot.terrain_mesh import (
    CHUNK_LENGTH,
    build_chunk_mesh,
    elevation_color,
)
from bike_sim.extract.ride_experience import (
    _astar_segment,
    _generate_waypoints,
    _order_waypoints_tsp,
)

CELL_SIZE: float = 50.0


def _generate_linear_path(
    heightmap: np.ndarray,
    world_seed: int,
) -> list[tuple[int, int]]:
    """Generate a linear (non-looping) ride path through the world.

    Same waypoint generation and TSP ordering as the standard ride path,
    but skips the closing segment back to the start point.
    """
    from bike_sim.rng import create_rng

    rng = create_rng(world_seed, "ride", "path", 0)

    waypoints = _generate_waypoints(heightmap, rng)
    ordered = _order_waypoints_tsp(waypoints)

    full_path: list[tuple[int, int]] = []
    for i in range(len(ordered) - 1):
        segment = _astar_segment(heightmap, ordered[i], ordered[i + 1])
        if segment is None:
            continue
        if full_path and segment[0] == full_path[-1]:
            segment = segment[1:]
        full_path.extend(segment)

    return full_path


def _make_height_fn(
    heightmap: np.ndarray,
    curve: BakedCurve,
    cell_size: float = CELL_SIZE,
):
    """Create a height function that samples the sim heightmap.

    Returns a callable(distance, lateral) → float giving terrain height
    relative to the path elevation at that distance.
    """

    def height_fn(distance: float, lateral: float) -> float:
        xf = curve.sample_transform(distance)
        world_pos = xf["origin"] + xf["right"] * lateral
        absolute_h = sample_heightmap_bilinear(
            heightmap, float(world_pos[0]), float(world_pos[2]), cell_size
        )
        path_y = xf["origin"][1]
        return absolute_h - path_y

    return height_fn


def extract_godot_terrain(
    world_dir: str | Path,
    output_dir: str | Path,
    chunk_length: float = CHUNK_LENGTH,
) -> dict:
    """Extract a sim world into Godot-compatible terrain chunks.

    Args:
        world_dir: path to an existing world directory.
        output_dir: where to write .chunk files and manifest.json.
        chunk_length: length of each chunk along the path in meters.

    Returns:
        Summary dict with extraction stats.
    """
    from bike_sim.world import World

    world_dir = Path(world_dir)
    output_dir = Path(output_dir)

    world = World.open(world_dir)
    try:
        # Read heightmap — prefer eroded surface
        try:
            heightmap = world.rasters.read_layer("geology", "eroded_heightmap")
            print("Using eroded heightmap")
        except (KeyError, Exception):
            heightmap = world.rasters.read_layer("geology", "heightmap")
            print("Using raw heightmap (no eroded version)")

        print(f"Heightmap shape: {heightmap.shape}, "
              f"range: {heightmap.min():.0f}–{heightmap.max():.0f}m")

        # Generate ride path (linear, no loop)
        print("Generating ride path...")
        path = _generate_linear_path(heightmap, world.seed)
        if not path:
            raise RuntimeError("Failed to generate ride path — no passable route found")

        path_length_cells = len(path)
        path_length_m = path_length_cells * CELL_SIZE
        print(f"Path: {path_length_cells} cells, ~{path_length_m / 1000:.1f} km")

        # Convert to Bezier control points
        print("Converting to Bezier curve...")
        control_points = path_to_bezier(path, heightmap, cell_size=CELL_SIZE)
        print(f"Bezier: {len(control_points)} control points")

        # Build arc-length parameterized curve
        curve = BakedCurve(control_points)
        print(f"Curve length: {curve.total_length / 1000:.1f} km")

        # Build height function
        height_fn = _make_height_fn(heightmap, curve, CELL_SIZE)

        # Generate chunks
        n_chunks = max(1, math.ceil(curve.total_length / chunk_length))
        print(f"Generating {n_chunks} chunks...")

        entries: list[dict] = []
        for i in range(n_chunks):
            chunk_start = i * chunk_length
            chunk_file = f"chunk_{i:04d}.chunk"
            chunk_path = str(output_dir / chunk_file)

            mesh = build_chunk_mesh(
                curve, chunk_start, height_fn, color_fn=elevation_color
            )
            write_chunk(chunk_path, i, mesh, {}, chunk_start)

            entries.append({
                "id": i,
                "start_z": chunk_start,
                "path": f"user://chunks/{chunk_file}",
            })

            if (i + 1) % 50 == 0 or i == n_chunks - 1:
                print(f"  {i + 1}/{n_chunks} chunks written")

        # Serialize control points for manifest (lists, not numpy)
        manifest_points = []
        for cp in control_points:
            manifest_points.append({
                "position": [float(v) for v in cp["position"]],
                "handle_in": [float(v) for v in cp["handle_in"]],
                "handle_out": [float(v) for v in cp["handle_out"]],
            })

        write_manifest(str(output_dir), entries, manifest_points)

        summary = {
            "world_dir": str(world_dir),
            "output_dir": str(output_dir),
            "heightmap_range": (float(heightmap.min()), float(heightmap.max())),
            "path_cells": path_length_cells,
            "path_km": path_length_m / 1000,
            "curve_km": curve.total_length / 1000,
            "control_points": len(control_points),
            "chunks": n_chunks,
        }
        print(f"\nDone! {n_chunks} chunks written to {output_dir}")
        return summary

    finally:
        world.close()
