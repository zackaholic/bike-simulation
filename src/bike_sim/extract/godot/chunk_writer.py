"""Write .chunk binary files and manifest.json.

Binary format (little-endian)::

    [Header]
      4 bytes  "CHNK" magic
      1 byte   version (2)
      4 bytes  chunk_id int32
      4 bytes  vertex_count int32
      4 bytes  index_count int32
    [Terrain]
      vertex_count * 12 bytes  positions (float32 x,y,z)
      vertex_count * 12 bytes  normals   (float32 x,y,z)
      vertex_count *  8 bytes  UVs       (float32 u,v)
      vertex_count * 16 bytes  colors    (float32 r,g,b,a)
      index_count  *  4 bytes  indices   (int32)
    [MultiMesh sections]
      4 bytes  multimesh_count int32
      per section:
        4 bytes  name_length int32
        N bytes  name UTF-8
        4 bytes  instance_count int32
        instance_count * 48 bytes  transforms (12 float32 per Transform3D)
    [Metadata]
      4 bytes  json_length int32
      N bytes  JSON UTF-8

Ported from bike-trainer-godot/world-builder/chunk_writer.py.
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np


VERSION = 2


def write_chunk(
    output_path: str,
    chunk_id: int,
    mesh: dict,
    multimeshes: dict,
    start_z: float,
    asset_flags: dict | None = None,
) -> None:
    """Write a single .chunk file.

    Args:
        output_path: full file path to write.
        chunk_id: integer chunk ID.
        mesh: dict from terrain_mesh.build_chunk_mesh().
        multimeshes: dict of {name: np.ndarray(N, 12)} transforms.
        start_z: path offset (meters) where this chunk begins.
        asset_flags: optional per-asset rendering flags.
    """
    verts = mesh["vertices"]
    normals = mesh["normals"]
    uvs = mesh["uvs"]
    colors = mesh.get("colors")
    indices = mesh["indices"]
    anchor = mesh["anchor"]

    vertex_count = len(verts)
    index_count = len(indices)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "wb") as f:
        # Header
        f.write(b"CHNK")
        f.write(struct.pack("<B", VERSION))
        f.write(struct.pack("<i", chunk_id))
        f.write(struct.pack("<i", vertex_count))
        f.write(struct.pack("<i", index_count))

        # Terrain geometry
        f.write(verts.astype("<f4").tobytes())
        f.write(normals.astype("<f4").tobytes())
        f.write(uvs.astype("<f4").tobytes())
        if colors is not None:
            f.write(colors.astype("<f4").tobytes())
        else:
            f.write(np.zeros((vertex_count, 4), dtype="<f4").tobytes())
        f.write(indices.astype("<i4").tobytes())

        # MultiMesh sections
        f.write(struct.pack("<i", len(multimeshes)))
        for name, transforms in multimeshes.items():
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<i", len(name_bytes)))
            f.write(name_bytes)
            instance_count = len(transforms)
            f.write(struct.pack("<i", instance_count))
            f.write(transforms.astype("<f4").tobytes())

        # Metadata JSON
        meta = {
            "chunk_id": chunk_id,
            "start_z": float(start_z),
            "anchor_x": float(anchor[0]),
            "anchor_y": float(anchor[1]),
            "anchor_z": float(anchor[2]),
            "asset_flags": asset_flags or {},
        }
        meta_bytes = json.dumps(meta).encode("utf-8")
        f.write(struct.pack("<i", len(meta_bytes)))
        f.write(meta_bytes)


def write_manifest(
    output_dir: str,
    entries: list[dict],
    path_control_points: list[dict] | None = None,
) -> None:
    """Write manifest.json describing the world.

    Godot reads "path" in path_manager.gd and "chunks" in chunk_streamer.gd.
    """
    manifest: dict = {"chunks": entries}
    if path_control_points is not None:
        manifest["path"] = path_control_points

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "manifest.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {out_path}")
