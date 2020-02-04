#version 330 core

layout (location = 0) in vec4 aPos;
layout (location = 1) in vec4 aColor;
layout (location = 2) in vec4 aNormal;

out vec4 vertexColor;
out vec3 vertexNormal;
out vec3 fragPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 coordinate_correction;

void main(){
    gl_Position = projection * view * model * aPos;
    gl_Position *= coordinate_correction;

    vertexColor = aColor;
    vertexNormal = aNormal.xyz;
    fragPos = ((model * aPos)).xyz;
}