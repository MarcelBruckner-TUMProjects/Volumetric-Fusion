#version 330 core
layout (location = 0) in vec3 aPos;

out vec3 aColor;

uniform vec3 size;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    gl_Position.y *= -1;
    aColor = aPos / (size / 2.0f);
}
