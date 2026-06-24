class_name ChunkFormatLoader
extends ResourceFormatLoader

# Teaches Godot's ResourceLoader how to read .chunk binary files.
# Register this at startup so load_threaded_request() works natively.
#
# Register in main.gd _ready():
#   ResourceLoader.add_resource_format_loader(ChunkFormatLoader.new())

func _get_recognized_extensions() -> PackedStringArray:
	return PackedStringArray(["chunk"])

func _handles_type(type: StringName) -> bool:
	return type == &"Resource" or type == &"ChunkResource"

func _get_resource_type(path: String) -> String:
	if path.get_extension().to_lower() == "chunk":
		return "Resource"
	return ""

func _load(path: String, _original_path: String, _use_sub_threads: bool, _cache_mode: int) -> Variant:
	var f = FileAccess.open(path, FileAccess.READ)
	if f == null:
		push_error("ChunkFormatLoader: cannot open %s (error %d)" % [path, FileAccess.get_open_error()])
		return ERR_CANT_OPEN

	# --- Header ---
	var magic = f.get_buffer(4).get_string_from_ascii()
	if magic != "CHNK":
		push_error("ChunkFormatLoader: bad magic in %s" % path)
		return ERR_FILE_UNRECOGNIZED
	var version = f.get_8()
	var chunk_id = f.get_32()
	var vertex_count = f.get_32()
	var index_count = f.get_32()

	# --- Terrain geometry ---
	# Read raw byte blocks and convert to packed typed arrays in one pass
	var vert_floats  = f.get_buffer(vertex_count * 12).to_float32_array()
	var norm_floats  = f.get_buffer(vertex_count * 12).to_float32_array()
	var uv_floats    = f.get_buffer(vertex_count *  8).to_float32_array()

	var verts   = PackedVector3Array(); verts.resize(vertex_count)
	var normals = PackedVector3Array(); normals.resize(vertex_count)
	var uvs     = PackedVector2Array(); uvs.resize(vertex_count)

	for i in vertex_count:
		verts[i]   = Vector3(vert_floats[i*3],   vert_floats[i*3+1],   vert_floats[i*3+2])
		normals[i] = Vector3(norm_floats[i*3],   norm_floats[i*3+1],   norm_floats[i*3+2])
		uvs[i]     = Vector2(uv_floats[i*2],     uv_floats[i*2+1])

	# v2: per-vertex colors (RGBA float32)
	var colors = PackedColorArray()
	if version >= 2:
		var col_floats = f.get_buffer(vertex_count * 16).to_float32_array()
		colors.resize(vertex_count)
		for i in vertex_count:
			colors[i] = Color(col_floats[i*4], col_floats[i*4+1],
							  col_floats[i*4+2], col_floats[i*4+3])

	var index_bytes = f.get_buffer(index_count * 4)
	var indices = PackedInt32Array(); indices.resize(index_count)
	for i in index_count:
		indices[i] = index_bytes.decode_s32(i * 4)

	var surface_array = []
	surface_array.resize(Mesh.ARRAY_MAX)
	surface_array[Mesh.ARRAY_VERTEX] = verts
	surface_array[Mesh.ARRAY_NORMAL] = normals
	surface_array[Mesh.ARRAY_TEX_UV] = uvs
	if colors.size() > 0:
		surface_array[Mesh.ARRAY_COLOR] = colors
	surface_array[Mesh.ARRAY_INDEX]  = indices

	var arr_mesh = ArrayMesh.new()
	arr_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, surface_array)
	# Prevent frustum culling: vertices are anchor-relative so Godot's auto-AABB
	# doesn't reflect the mesh's actual world position.
	arr_mesh.custom_aabb = AABB(Vector3(-600, -60, -600), Vector3(1200, 120, 1200))

	# --- MultiMesh sections ---
	var multimesh_count = f.get_32()
	var multimeshes: Dictionary = {}

	for _m in multimesh_count:
		var name_len = f.get_32()
		var asset_name = f.get_buffer(name_len).get_string_from_utf8()
		var instance_count = f.get_32()

		var mm = MultiMesh.new()
		mm.transform_format = MultiMesh.TRANSFORM_3D
		mm.instance_count = instance_count
		mm.buffer = f.get_buffer(instance_count * 48).to_float32_array()
		multimeshes[asset_name] = mm

	# --- Metadata JSON ---
	var json_len = f.get_32()
	var json_str = f.get_buffer(json_len).get_string_from_utf8()
	var json = JSON.new()
	json.parse(json_str)
	var meta = json.get_data()

	# --- Assemble ChunkResource ---
	var chunk = ChunkResource.new()
	chunk.chunk_id    = chunk_id
	chunk.terrain_mesh = arr_mesh
	chunk.multimeshes  = multimeshes
	chunk.asset_flags  = meta.get("asset_flags", {})
	chunk.start_z      = float(meta.get("start_z", 0.0))
	chunk.anchor = Vector3(
		float(meta.get("anchor_x", 0.0)),
		float(meta.get("anchor_y", 0.0)),
		float(meta.get("anchor_z", 0.0))
	)

	return chunk
