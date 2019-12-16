#version 450 core

layout(location = 0) in vec2 vertexPos;
layout(location = 1) in vec2 vertexUV;

out vec2 fragUV;

void main()
{
    gl_Position = vec4(vertexPos, 0.0, 1.0);
    fragUV = vertexUV;
}