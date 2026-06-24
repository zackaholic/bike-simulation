extends Node3D

func _ready() -> void:
	ResourceLoader.add_resource_format_loader(ChunkFormatLoader.new())
	_make_environment()
	_make_sun()
	# Increase far plane for the sim world (50km extent, up to 2000m elevation)
	var cam := $PathManager/PathFollower/Camera3D as Camera3D
	cam.far = 10000.0

func _make_sun() -> void:
	var sun := $DirectionalLight3D as DirectionalLight3D
	sun.light_energy = 1.2
	sun.rotation_degrees = Vector3(-35.0, 45.0, 0.0)

func _make_environment() -> void:
	var sky_mat := PhysicalSkyMaterial.new()
	sky_mat.rayleigh_coefficient = 2.0
	sky_mat.mie_coefficient = 0.03
	sky_mat.mie_eccentricity = 0.85
	sky_mat.sun_disk_scale = 1.0
	sky_mat.ground_color = Color(0.35, 0.3, 0.22)
	sky_mat.energy_multiplier = 12.0

	var sky := Sky.new()
	sky.sky_material = sky_mat

	var env := Environment.new()
	env.background_mode = Environment.BG_SKY
	env.sky = sky
	env.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	env.ambient_light_energy = 1.0

	env.ssao_enabled = true
	env.ssao_radius = 1.0
	env.ssao_intensity = 1.5

	env.fog_enabled = true
	env.fog_density = 0.003
	env.fog_aerial_perspective = 0.1
	env.fog_sky_affect = 0.3

	env.glow_enabled = true
	env.glow_intensity = 0.8
	env.glow_bloom = 0.1
	env.glow_hdr_threshold = 1.0

	env.adjustment_enabled = true
	env.adjustment_brightness = 1.0
	env.adjustment_contrast = 1.1
	env.adjustment_saturation = 1.2

	var we := $WorldEnvironment as WorldEnvironment
	we.environment = env
