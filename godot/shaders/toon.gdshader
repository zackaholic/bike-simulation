shader_type spatial;
render_mode diffuse_toon, specular_disabled;

// Surface type colors.
uniform vec4 color_rock  : source_color = vec4(0.52, 0.46, 0.36, 1.0);
uniform vec4 color_scrub : source_color = vec4(0.82, 0.65, 0.20, 1.0);
uniform vec4 color_grass : source_color = vec4(0.10, 0.55, 0.10, 1.0);

// Temperature tinting.
uniform vec4 temp_warm_tint : source_color = vec4(0.10, 0.02, -0.05, 0.0);
uniform vec4 temp_cool_tint : source_color = vec4(-0.04, 0.02,  0.12, 0.0);

// Per-surface detail textures — each zone gets its own noise character.
uniform sampler2D detail_rock  : filter_linear_mipmap, repeat_enable;
uniform sampler2D detail_scrub : filter_linear_mipmap, repeat_enable;
uniform sampler2D detail_grass : filter_linear_mipmap, repeat_enable;

// Tiling scale per surface type (higher = finer pattern).
uniform float scale_rock  : hint_range(1.0, 64.0) = 4.0;
uniform float scale_scrub : hint_range(1.0, 64.0) = 14.0;
uniform float scale_grass : hint_range(1.0, 64.0) = 22.0;

// Blend strength per surface type (0 = flat color, 1 = full texture multiply).
uniform float blend_rock  : hint_range(0.0, 1.0) = 0.65;
uniform float blend_scrub : hint_range(0.0, 1.0) = 0.50;
uniform float blend_grass : hint_range(0.0, 1.0) = 0.35;

void fragment() {
    float rock_w  = COLOR.r;
    float scrub_w = COLOR.g;
    float grass_w = COLOR.b;
    float temp    = COLOR.a;

    // Blend surface colors by weight.
    vec3 base = rock_w  * color_rock.rgb
              + scrub_w * color_scrub.rgb
              + grass_w * color_grass.rgb;

    // Temperature tint.
    vec3 tint = mix(temp_cool_tint.rgb, temp_warm_tint.rgb, temp);
    base = clamp(base + tint, 0.0, 1.0);

    // Sample each surface's detail texture and blend by weight.
    // mix(1.0, d, blend) = 1.0 when blend=0 (no darkening), d when blend=1 (full multiply).
    float d_rock  = mix(1.0, texture(detail_rock,  UV * scale_rock).r,  blend_rock);
    float d_scrub = mix(1.0, texture(detail_scrub, UV * scale_scrub).r, blend_scrub);
    float d_grass = mix(1.0, texture(detail_grass, UV * scale_grass).r, blend_grass);

    float detail = rock_w * d_rock + scrub_w * d_scrub + grass_w * d_grass;

    ALBEDO    = base * detail;
    ROUGHNESS = 1.0;
    METALLIC  = 0.0;
}
