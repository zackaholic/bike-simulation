shader_type spatial;
render_mode diffuse_toon, specular_disabled;

// Reads vertex color as direct albedo (elevation-based coloring from extractor).
void fragment() {
	ALBEDO = COLOR.rgb;
	ROUGHNESS = 1.0;
	METALLIC = 0.0;
}
