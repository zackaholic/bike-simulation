class_name AssetMeshBuilder
# Static helpers shared by ChunkStreamer (world rendering) and the preview scene.
# Agents write asset specs; this class turns them into geometry.

static func load_spec(asset_name: String) -> Dictionary:
	var path := "res://assets/%s.json" % asset_name
	if not FileAccess.file_exists(path):
		push_warning("AssetMeshBuilder: no spec at %s" % path)
		return {}
	var f := FileAccess.open(path, FileAccess.READ)
	var json := JSON.new()
	json.parse(f.get_as_text())
	return json.get_data()

static func build_mesh(spec: Dictionary) -> ArrayMesh:
	var verts: PackedVector3Array = PackedVector3Array()
	var norms: PackedVector3Array = PackedVector3Array()
	var cols:  PackedColorArray   = PackedColorArray()
	var idxs:  PackedInt32Array   = PackedInt32Array()

	for p: Dictionary in spec.get("primitives", []):
		var c: Array = p["color"]
		_append(verts, norms, cols, idxs,
			float(p["top_r"]), float(p["bot_r"]), float(p["height"]),
			int(p["segs"]),    float(p["y_base"]),
			Color(float(c[0]), float(c[1]), float(c[2])),
			float(p.get("x_offset", 0.0)), float(p.get("z_offset", 0.0)),
			float(p.get("rot_x",    0.0)), float(p.get("rot_y",    0.0)),
			float(p.get("rot_z",    0.0)))

	var arr: Array = []
	arr.resize(Mesh.ARRAY_MAX)
	arr[Mesh.ARRAY_VERTEX] = verts
	arr[Mesh.ARRAY_NORMAL]  = norms
	arr[Mesh.ARRAY_COLOR]   = cols
	arr[Mesh.ARRAY_INDEX]   = idxs

	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arr)
	return mesh

# Returns the total height of the spec (top of tallest primitive).
# Used by the preview scene to frame the camera correctly.
static func spec_height(spec: Dictionary) -> float:
	var h := 0.0
	for p: Dictionary in spec.get("primitives", []):
		h = maxf(h, float(p["y_base"]) + float(p["height"]))
	return h

# Samples a CylinderMesh (correct normals/caps/winding by construction) and
# appends its geometry into the shared packed arrays.
# top_r=0 makes a cone; top_r=bot_r makes a cylinder.
#
# Transform order (pivot = base of primitive):
#   1. Shift CylinderMesh so its base sits at y=0  (CylinderMesh is centered)
#   2. Apply rotation (rot_x/y/z in degrees, YXZ Euler order)
#   3. Translate to (x_offset, y_base, z_offset) in world space
#
# This means branches attach at (x_offset, y_base, z_offset) and lean outward
# according to their rotation — the base stays pinned, the tip swings.
static func _append(verts: PackedVector3Array, norms: PackedVector3Array,
		cols: PackedColorArray, idxs: PackedInt32Array,
		top_r: float, bot_r: float, height: float, segs: int,
		y_base: float, color: Color,
		x_offset: float = 0.0, z_offset: float = 0.0,
		rot_x: float = 0.0, rot_y: float = 0.0, rot_z: float = 0.0) -> void:
	var cyl := CylinderMesh.new()
	cyl.top_radius      = top_r
	cyl.bottom_radius   = bot_r
	cyl.height          = height
	cyl.radial_segments = segs
	cyl.rings = 1

	var src: Array                = cyl.surface_get_arrays(0)
	var src_v: PackedVector3Array = src[Mesh.ARRAY_VERTEX]
	var src_n: PackedVector3Array = src[Mesh.ARRAY_NORMAL]
	var src_i: PackedInt32Array   = src[Mesh.ARRAY_INDEX]

	var base:   int    = verts.size()
	var basis:  Basis  = Basis.from_euler(
		Vector3(deg_to_rad(rot_x), deg_to_rad(rot_y), deg_to_rad(rot_z)))
	var offset: Vector3 = Vector3(x_offset, y_base, z_offset)

	for v: Vector3 in src_v:
		var local := Vector3(v.x, v.y + height * 0.5, v.z)  # pivot at base
		verts.append(basis * local + offset)
	for n: Vector3 in src_n:
		norms.append(basis * n)
	for _k: int in src_v.size():
		cols.append(color)
	for i: int in src_i:
		idxs.append(i + base)
