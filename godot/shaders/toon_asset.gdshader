shader_type spatial;
// Asset toon shader — vertex color is treated as actual albedo color (not splatmap weights).
// Used for all 3D placed assets (trees, shrubs, rocks, etc.).
// Terrain uses toon.gdshader which encodes vertex color as surface blend weights.
render_mode diffuse_toon, specular_disabled;

// Global tint — set to white to show vertex colors as-is.
uniform vec4 albedo : source_color = vec4(1.0, 1.0, 1.0, 1.0);

void fragment() {
    ALBEDO    = albedo.rgb * COLOR.rgb;
    ROUGHNESS = 1.0;
    METALLIC  = 0.0;
}
