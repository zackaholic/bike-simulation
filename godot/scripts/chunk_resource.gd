class_name ChunkResource
extends Resource

# Holds the fully parsed data for one world chunk.
# Produced by ChunkFormatLoader reading a .chunk binary file.

var chunk_id: int = 0
var start_z: float = 0.0       # meters along path where this chunk begins
var anchor: Vector3 = Vector3.ZERO  # world-space anchor; vertices are relative to this

var terrain_mesh: ArrayMesh = null
var multimeshes: Dictionary = {}    # String (asset name) -> MultiMesh
var asset_flags: Dictionary = {}    # String (asset name) -> {outline: bool, ...}
