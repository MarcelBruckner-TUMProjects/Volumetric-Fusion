#version 450 core

in vec2 fragUV;

uniform sampler2D g_sampler;

out vec4 outColor;

void main()
{
    vec4 depth = texture(g_sampler, vec2(fragUV.x, 1.0 - fragUV.y)) / 5.0;
    outColor = vec4(depth.x, depth.x, depth.x, 1.0);
}