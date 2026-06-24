extends CanvasLayer

@onready var speed_label: Label = $SpeedLabel
@onready var hr_label: Label = $HRLabel

func _ready() -> void:
	print("HUD ready")
	speed_label.position = Vector2(20, 20)
	speed_label.size = Vector2(300, 40)
	speed_label.add_theme_font_size_override("font_size", 28)
	speed_label.add_theme_color_override("font_color", Color.WHITE)

	hr_label.position = Vector2(20, 60)
	hr_label.size = Vector2(300, 40)
	hr_label.add_theme_font_size_override("font_size", 28)
	hr_label.add_theme_color_override("font_color", Color.WHITE)

func _process(_delta: float) -> void:
	speed_label.text = "%.1f km/h" % GameState.current_speed
	hr_label.text = "%d bpm" % GameState.heart_rate
