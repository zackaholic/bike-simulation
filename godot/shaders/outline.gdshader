shader_type spatial;
// Inverted-hull outline technique.
//   cull_front: only back faces drawn — they appear as a silhouette rim.
//   unshaded: pure flat color, no lighting on the outline.
//   shadows_disabled: outline mesh casts/receives no shadows.
render_mode cull_front, unshaded, shadows_disabled, ambient_light_disabled;

// Width in world units. Suitable for human-scale objects (trees, rocks, structures).
// Do NOT apply to large terrain meshes — scale mismatch makes outlines invisible or huge.
uniform float outline_width : hint_range(0.001, 0.5) = 0.05;
uniform vec4 outline_color : source_color = vec4(0.0, 0.0, 0.0, 1.0);

void vertex() {
    VERTEX += NORMAL * outline_width;
}

void fragment() {
    ALBEDO = outline_color.rgb;
}
