#version 450 core

in vec2 fragUV;

uniform sampler2D g_sampler;

out vec4 outColor;

void main()
{
    vec4 normal = texture(g_sampler, vec2(fragUV.x, 1.0 - fragUV.y));
    outColor = vec4((normal.xyz + 1.0)/2.0, 1.0);
}