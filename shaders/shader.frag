#version 330 core

layout(push_constant) uniform Configuration
{
    ivec2   grid;
    vec2    mapDims;
    int     issue;
    int     bonesPerHypothesis;
} g_configuration;

// Input from vertex shader

in block
{
    vec4        vp_pos;      // 3D position in camera space
    vec4        clip_bounds; // bounds for clipping
} inputs;

// output_ streams


layout(location = 0) out vec4   o_position;


void main()
{
    // clipping coordinates

    vec2 pos = (gl_FragCoord.xy / g_configuration.mapDims.xy) * 2.0f - 1.0f;

    if (any(lessThan(pos - inputs.clip_bounds.xz + 10e-5f, vec2(0, 0))) ||
        any(lessThan(inputs.clip_bounds.yw + 10e-5f - pos, vec2(0, 0))))
    {
        discard;
    }

    o_position = inputs.vp_pos / inputs.vp_pos.w;
}