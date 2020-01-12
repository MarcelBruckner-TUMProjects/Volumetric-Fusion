#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 relativeTransformation;
uniform mat4 model;
uniform mat4 correction;
uniform mat4 view;
uniform mat4 projection;

out VS_OUT {
    vec3 color;
} vs_out;

void main()
{
    vs_out.color = aColor;
    gl_Position = vec4(aPos, 1.0f);
    gl_Position = projection * view * model * relativeTransformation * gl_Position;
    gl_Position = correction * gl_Position;
}