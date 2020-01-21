#version 330 core

layout (location = 0) in vec4 aPos;
layout (location = 1) in vec4 aColor;

out vec4 vertexColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 coordinate_correction;

void main(){
    gl_Position = projection * view * model * aPos;
    gl_Position *= coordinate_correction;

    vertexColor = aColor;
}