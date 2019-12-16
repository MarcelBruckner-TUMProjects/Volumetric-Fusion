#version 450 core

uniform mat4 g_viewMatrix;
uniform mat4 g_viewProjMatrix;

layout(location = 0) in vec4 vertexPos;
layout(location = 1) in vec4 vertexNormal;
layout(location = 2) in vec4 nodeTransform0;
layout(location = 3) in vec4 nodeTransform1;
layout(location = 4) in vec4 nodeTransform2;
layout(location = 5) in vec4 nodeColor;

out vec3 fragPos;
out vec3 fragNormal;
out vec3 fragColor;

void main()
{
    // OpenGL matrices are stored column-wise, and we provide transform rows.
    mat4 nodeTransform;
    nodeTransform[0][0] = nodeTransform0[0];
    nodeTransform[1][0] = nodeTransform0[1];
    nodeTransform[2][0] = nodeTransform0[2];
    nodeTransform[3][0] = nodeTransform0[3];
    nodeTransform[0][1] = nodeTransform1[0];
    nodeTransform[1][1] = nodeTransform1[1];
    nodeTransform[2][1] = nodeTransform1[2];
    nodeTransform[3][1] = nodeTransform1[3];
    nodeTransform[0][2] = nodeTransform2[0];
    nodeTransform[1][2] = nodeTransform2[1];
    nodeTransform[2][2] = nodeTransform2[2];
    nodeTransform[3][2] = nodeTransform2[3];
    nodeTransform[0][3] = 0;
    nodeTransform[1][3] = 0;
    nodeTransform[2][3] = 0;
    nodeTransform[3][3] = 1;
 
    vec4 projPos = g_viewProjMatrix * (nodeTransform * vertexPos);
    gl_Position = projPos / projPos.w;
    fragPos = vec3(g_viewMatrix * (nodeTransform * vertexPos));
    fragNormal = vec3(g_viewMatrix * vec4(vertexNormal.xyz, 0.f));
    fragColor = vec3(nodeColor);
}