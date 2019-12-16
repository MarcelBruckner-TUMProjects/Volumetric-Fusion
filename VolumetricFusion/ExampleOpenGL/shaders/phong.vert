#version 450 core

uniform mat4 g_viewMatrix;
uniform mat4 g_viewProjMatrix;

layout(location = 0) in vec4 vertexPos;
layout(location = 1) in vec4 vertexNormal;

out vec3 fragPos;
out vec3 fragNormal;

void main()
{
    vec4 projPos = g_viewProjMatrix * vertexPos;
    gl_Position = projPos / projPos.w;
    fragPos = vec3(g_viewMatrix * vertexPos);
    fragNormal = vec3(g_viewMatrix * vec4(vertexNormal.xyz, 0.f));
}