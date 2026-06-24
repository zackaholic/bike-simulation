extends PathFollow3D

# Advances along the path each frame at the current riding speed.
#
# Camera grounding:
#   A downward raycast finds the terrain surface each frame. Camera Y is
#   smoothed toward (terrain + EYE_HEIGHT) using exponential decay — fast
#   enough to track real hills, slow enough to absorb small bumps so the
#   camera doesn't bob on every pebble.
#
#   CAMERA_SMOOTH_SPEED controls the feel:
#     2.0 = very floaty, barely tracks surface detail
#     4.0 = natural bike-on-dirt feel  ← default
#     8.0 = snappy, tracks almost every bump
#
#   A hard floor (MIN_CLEARANCE above terrain) prevents the camera from
#   going underground when the terrain rises faster than the smooth rate.

const EYE_HEIGHT         := 1.5   # metres above terrain surface
const RAY_ABOVE          := 500.0  # cast from this far above path position
const RAY_BELOW          := 500.0  # cast this far below path position
const CAMERA_SMOOTH_SPEED := 4.0  # exponential decay rate (1/seconds)
const MIN_CLEARANCE      := 0.25  # hard floor above terrain (prevents underground)

var _smooth_y: float = 0.0
var _initialized: bool = false

func _ready() -> void:
	loop = false
	v_offset = 0.0   # Y handled by raycast + smoothing, not path offset

func _process(delta: float) -> void:
	var speed_ms := GameState.current_speed / 3.6   # km/h → m/s
	progress += speed_ms * delta
	GameState.world_position = progress

	var path_pos := global_position
	var space    := get_world_3d().direct_space_state
	var query    := PhysicsRayQueryParameters3D.create(
		path_pos + Vector3(0.0,  RAY_ABOVE, 0.0),
		path_pos + Vector3(0.0, -RAY_BELOW, 0.0)
	)
	var hit := space.intersect_ray(query)

	var camera := $Camera3D

	if hit:
		var terrain_y: float = hit.position.y
		var target_y: float  = terrain_y + EYE_HEIGHT

		# Exponential smoothing toward target — frame-rate independent.
		# 1 - exp(-speed * dt) gives correct decay regardless of frame rate.
		if not _initialized:
			_smooth_y    = target_y
			_initialized = true
		else:
			var t     := 1.0 - exp(-CAMERA_SMOOTH_SPEED * delta)
			_smooth_y  = lerp(_smooth_y, target_y, t)

		# Hard floor: never let camera drop below terrain + minimum clearance.
		# Handles the case where terrain rises faster than the smooth rate.
		_smooth_y = maxf(_smooth_y, terrain_y + MIN_CLEARANCE)

		camera.position.y = _smooth_y - path_pos.y
	else:
		# No hit — chunk boundary gap or not yet loaded. Hold last position.
		if _initialized:
			camera.position.y = _smooth_y - path_pos.y
		else:
			camera.position.y = EYE_HEIGHT
