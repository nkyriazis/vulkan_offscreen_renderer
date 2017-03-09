#version 330 core

layout(triangles) in;

in block
{
    vec4        vp_pos;      // 3D position in camera space
    vec4        clip_bounds; // bounds for clipping
} input_[];

layout(triangle_strip, max_vertices = 3) out;

out block
{
    vec4        vp_pos;      // 3D position in camera space
    vec4        clip_bounds; // bounds for clipping
} output_;

void main()
{
    for (int i = 0; i < 3; i++)
    {
        gl_Position = gl_in[i].gl_Position;
        output_.vp_pos = input_[i].vp_pos;
        output_.clip_bounds = input_[i].clip_bounds;
        EmitVertex();
    }
    EndPrimitive();
}