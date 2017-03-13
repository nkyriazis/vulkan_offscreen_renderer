#version 450

out gl_PerVertex {
    vec4 gl_Position;
};

layout(location = 0) out vec4 fragColor;

layout (set=0, binding=0) buffer verticesDynStorageBuffer
{
    vec2 positions[];
};

layout(push_constant) uniform PushConstants {
	vec3 colors[16];
} pushConstants;

void main() {
	vec4 offset = vec4(2 * cos(gl_InstanceIndex/5.0f), 2 * sin(gl_InstanceIndex/5.0f), 0, gl_InstanceIndex/100.0f+1.0f);
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0) + offset;
    fragColor = vec4(pushConstants.colors[gl_InstanceIndex % 16], 1);
}