#version 450 core

uniform mat4 g_viewMatrix;
uniform mat4 g_viewProjMatrix;

layout(location = 0) in vec4 vertexPos;

out vec3 fragPos;

void main()
{
    vec4 projPos = g_viewProjMatrix * vertexPos;
    gl_Position = projPos / projPos.w;
    fragPos = vertexPos.xyz;
}