extends Camera3D

func _process(delta: float) -> void:
	var speed_ms = GameState.current_speed / 3.6  # km/h to m/s
	var movement = -transform.basis.z * speed_ms * delta
	position += movement
	GameState.world_position += speed_ms * delta
