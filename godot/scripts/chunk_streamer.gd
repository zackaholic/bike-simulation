extends Node3D

const AssetMeshBuilder = preload("res://scripts/asset_mesh_builder.gd")

const LOOKAHEAD_METERS = 600.0
const CULL_BEHIND_METERS = 450.0
const MANIFEST_PATH = "user://chunks/manifest.json"

var _manifest: Array = []          # Array of {id, start_z, path}
var _loaded_chunks: Dictionary = {}  # chunk_id -> Node3D instance
var _loading_chunks: Dictionary = {} # chunk_id -> path

# Toon shader shared across all terrain chunks (same color, so one material is fine).
var _terrain_material: ShaderMaterial
# Per-asset-type mesh and material, keyed by asset name ("shrub", "rock", …).
var _asset_meshes: Dictionary = {}
var _asset_materials: Dictionary = {}
# Single outline material shared across all asset types (same width/color for all).
var _outline_material: ShaderMaterial

func _ready() -> void:
	print("ChunkStreamer ready")
	_build_terrain_material()
	_build_asset_definitions()
	_load_manifest()

func _build_terrain_material() -> void:
	# Direct albedo from vertex colors (elevation-based coloring from extractor).
	var shader := load("res://shaders/terrain_direct.gdshader") as Shader
	_terrain_material = ShaderMaterial.new()
	_terrain_material.shader = shader

func _build_asset_definitions() -> void:
	var shader := load("res://shaders/toon_asset.gdshader") as Shader

	# Load every *.json in res://assets/ as an asset spec.
	# Adding a new spec file is all that's needed to register a new asset type.
	var dir := DirAccess.open("res://assets")
	if dir == null:
		push_error("ChunkStreamer: res://assets/ not found")
	else:
		dir.list_dir_begin()
		var fname := dir.get_next()
		while fname != "":
			if fname.ends_with(".json"):
				var asset_name := fname.get_basename()
				var spec := _load_asset_spec(asset_name)
				if not spec.is_empty():
					_asset_meshes[asset_name]    = _build_mesh_from_spec(spec)
					var mat := ShaderMaterial.new()
					mat.shader = shader
					# albedo=white — vertex colors in the spec carry all color information.
					mat.set_shader_parameter("albedo", Color(1.0, 1.0, 1.0))
					_asset_materials[asset_name] = mat
			fname = dir.get_next()
		dir.list_dir_end()

	# Outline: one shared material — same black silhouette on all asset types.
	var outline_shader := load("res://shaders/outline.gdshader") as Shader
	_outline_material = ShaderMaterial.new()
	_outline_material.shader = outline_shader
	_outline_material.set_shader_parameter("outline_width", 0.056)
	_outline_material.set_shader_parameter("outline_color", Color(0.122, 0.097, 0.068, 1.0))

func _load_asset_spec(asset_name: String) -> Dictionary:
	return AssetMeshBuilder.load_spec(asset_name)

func _build_mesh_from_spec(spec: Dictionary) -> ArrayMesh:
	return AssetMeshBuilder.build_mesh(spec)

func _load_manifest() -> void:
	if not FileAccess.file_exists(MANIFEST_PATH):
		push_warning("ChunkStreamer: no manifest at %s — world will be empty" % MANIFEST_PATH)
		return
	var f = FileAccess.open(MANIFEST_PATH, FileAccess.READ)
	var json = JSON.new()
	json.parse(f.get_as_text())
	var data = json.get_data()
	# Support both legacy format (bare Array) and current format ({chunks, path})
	if data is Array:
		_manifest = data
	else:
		_manifest = data.get("chunks", [])
	print("ChunkStreamer: loaded manifest with %d chunks" % _manifest.size())

func _process(_delta: float) -> void:
	if _manifest.is_empty():
		return
	var cam_pos = GameState.world_position
	_request_chunks_in_range(cam_pos)
	_poll_loading_chunks()
	_cull_old_chunks(cam_pos)

func _request_chunks_in_range(cam_pos: float) -> void:
	for entry in _manifest:
		var chunk_id: int = int(entry["id"])
		var chunk_start: float = float(entry["start_z"])
		var chunk_path: String = entry["path"]
		if chunk_id in _loaded_chunks or chunk_id in _loading_chunks:
			continue
		if chunk_start > cam_pos + LOOKAHEAD_METERS:
			continue
		if chunk_start < cam_pos - CULL_BEHIND_METERS:
			continue
		ResourceLoader.load_threaded_request(chunk_path)
		_loading_chunks[chunk_id] = chunk_path

func _poll_loading_chunks() -> void:
	for chunk_id in _loading_chunks.keys():
		var path = _loading_chunks[chunk_id]
		var status = ResourceLoader.load_threaded_get_status(path)
		if status == ResourceLoader.THREAD_LOAD_LOADED:
			var resource = ResourceLoader.load_threaded_get(path) as ChunkResource
			if resource:
				var node = _instantiate_chunk(resource)
				add_child(node)
				_loaded_chunks[chunk_id] = node
			_loading_chunks.erase(chunk_id)
		elif status == ResourceLoader.THREAD_LOAD_FAILED:
			push_error("ChunkStreamer: failed to load %s" % path)
			_loading_chunks.erase(chunk_id)

func _instantiate_chunk(resource: ChunkResource) -> Node3D:
	var root = Node3D.new()
	root.name = "Chunk_%d" % resource.chunk_id
	root.position = resource.anchor  # vertices are relative to anchor

	# Terrain mesh
	if resource.terrain_mesh:
		var mi = MeshInstance3D.new()
		mi.mesh = resource.terrain_mesh
		# Vertices are stored relative to anchor (world-space offset applied via node.position).
		# Godot's frustum culler uses the mesh's local AABB before the node transform is applied,
		# which incorrectly culls near chunks. Fix via three layers of defense:
		#   1. custom_aabb on the GeometryInstance3D (covers local-space culling)
		#   2. custom_aabb on the ArrayMesh itself
		#   3. RenderingServer ignore_culling (disables frustum + occlusion culling entirely)
		var big_aabb = AABB(Vector3(-600, -500, -600), Vector3(1200, 1000, 1200))
		mi.custom_aabb = big_aabb
		resource.terrain_mesh.custom_aabb = big_aabb
		mi.material_override = _terrain_material
		root.add_child(mi)

		# Collision mesh — needed for path_follower.gd's downward raycast.
		# ConcavePolygonShape3D matches the rendered terrain exactly.
		var body = StaticBody3D.new()
		var col  = CollisionShape3D.new()
		col.shape = resource.terrain_mesh.create_trimesh_shape()
		body.add_child(col)
		root.add_child(body)

	# Asset MultiMeshes — two passes per type: toon fill + inverted-hull outline.
	# Both passes share the same MultiMesh (same transforms), different material_override.
	for asset_name in resource.multimeshes:
		var mm: MultiMesh = resource.multimeshes[asset_name]
		if mm.instance_count == 0:
			continue
		if asset_name in _asset_meshes:
			mm.mesh = _asset_meshes[asset_name]

		var mmi = MultiMeshInstance3D.new()
		mmi.name = asset_name
		mmi.multimesh = mm
		if asset_name in _asset_materials:
			mmi.material_override = _asset_materials[asset_name]
		root.add_child(mmi)

		# Outline pass: driven by asset_flags["outline"] set in placement rules.
		var flags = resource.asset_flags.get(asset_name, {})
		if not flags.get("outline", true):
			continue
		var outline_mmi = MultiMeshInstance3D.new()
		outline_mmi.name = asset_name + "_outline"
		outline_mmi.multimesh = mm
		outline_mmi.material_override = _outline_material
		root.add_child(outline_mmi)

	return root

func _cull_old_chunks(cam_pos: float) -> void:
	for chunk_id in _loaded_chunks.keys():
		for entry in _manifest:
			if int(entry["id"]) == chunk_id:
				if float(entry["start_z"]) < cam_pos - CULL_BEHIND_METERS:
					_loaded_chunks[chunk_id].queue_free()
					_loaded_chunks.erase(chunk_id)
				break
