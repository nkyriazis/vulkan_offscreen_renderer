#version 450 core

layout(set=0, binding=0) buffer ViewMatrices
{
	mat4 g_viewMatrices[];
};

layout(set=0, binding=1) buffer ProjectionMatrices
{
	mat4 g_projectionMatrices[];
};

layout(push_constant) uniform Configuration
{
    ivec2   grid;
    vec2    mapDims;
    int     issue;
    int     bonesPerHypothesis;
} g_configuration;

// Vertex attributes


layout(location = 0)  in vec3       in_vertex;          //  x, y, z

layout(location = 1)  in ivec4      in_indices;         //  projection, view, viewport, instanceID
layout(location = 2)  in mat4       in_worldTransform;  //  mat4x4

// Vertex shader output_
//
out block
{
    vec4        vp_pos;      // 3D position in camera space
    vec4        clip_bounds; // bounds for clipping
} output_;

void main()
{
    // viewport 2D indices and floating point grid dimensions

    float vp_i = in_indices.z / g_configuration.grid.x;
    float vp_j = in_indices.z % g_configuration.grid.x;
    float v_width = g_configuration.grid.x;
    float v_height = g_configuration.grid.y;
    
    mat4 world = in_worldTransform;
        
    mat4 projection = g_projectionMatrices[in_indices.x];
    mat4 view = g_viewMatrices[in_indices.y];
        
    // view x world transformation 
    mat4 viewworld = view * world;

    // viewport description

    vec4 viewport = vec4(vp_j / v_width, vp_i / v_height, (vp_j + 1.0f) / v_width, (vp_i + 1.0f) / v_height);
    viewport = viewport * 2.0f - 1.0f;

    // camera-space position

    output_.vp_pos = viewworld * vec4(in_vertex, 1.0f);
    output_.vp_pos /= output_.vp_pos.w;

    // image-space position

    vec4 tvp_pos = projection * output_.vp_pos;

    // clipping-space 

    gl_Position = tvp_pos;
    gl_Position.xy = gl_Position.w * ((viewport.zy - viewport.xw) / 2.0f * gl_Position.xy / gl_Position.w + (viewport.xy + viewport.zw) / 2.0f);

    // clipping bounds

    output_.clip_bounds = vec4
        ( min(viewport.x, viewport.z)
        , max(viewport.x, viewport.z)
        , min(viewport.y, viewport.w)
        , max(viewport.y, viewport.w) );


}