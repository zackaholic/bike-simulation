extends Path3D

# Owns the Curve3D that defines the world path.
# Loads control points from the world builder manifest when available.
# Falls back to a hardcoded test curve if no manifest exists.

const MANIFEST_PATH = "user://chunks/manifest.json"

func _ready() -> void:
	curve = _load_curve()
	print("PathManager ready, path length: %.1fm" % curve.get_baked_length())

func _load_curve() -> Curve3D:
	if FileAccess.file_exists(MANIFEST_PATH):
		var f = FileAccess.open(MANIFEST_PATH, FileAccess.READ)
		var json = JSON.new()
		json.parse(f.get_as_text())
		var data = json.get_data()
		if data is Dictionary and data.has("path"):
			return _build_curve_from_points(data["path"])

	push_warning("PathManager: no manifest path found, using fallback test curve")
	return _build_fallback_curve()

func _build_curve_from_points(points: Array) -> Curve3D:
	var c = Curve3D.new()
	c.bake_interval = 1.0
	for pt in points:
		var pos = Vector3(pt["position"][0], pt["position"][1], pt["position"][2])
		var hin = Vector3(pt["handle_in"][0],  pt["handle_in"][1],  pt["handle_in"][2])
		var hout= Vector3(pt["handle_out"][0], pt["handle_out"][1], pt["handle_out"][2])
		c.add_point(pos, hin, hout)
	return c

func _build_fallback_curve() -> Curve3D:
	var c = Curve3D.new()
	c.bake_interval = 1.0
	c.add_point(Vector3(  0,  0,    0), Vector3(0,0,  0), Vector3( 0,  0, -120))
	c.add_point(Vector3(-25,  3, -300), Vector3(0,0,120), Vector3( 0,  0, -120))
	c.add_point(Vector3( 20,  1, -600), Vector3(0,0,120), Vector3( 0,  0, -120))
	c.add_point(Vector3(-10,  4, -900), Vector3(0,0,120), Vector3( 0,  0, -120))
	c.add_point(Vector3(  5,  0,-1200), Vector3(0,0,120), Vector3( 0,  0,    0))
	return c
