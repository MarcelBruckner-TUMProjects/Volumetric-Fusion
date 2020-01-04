#version 330 core
layout (location = 0) in vec2 aPos;

out vec2 texCoord;

uniform int width;
uniform int height;
uniform mat4 relativeTransformation;
uniform mat4 model;
uniform mat4 correction;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    vec3 pos = vec3(aPos, 1.0f);
    gl_Position = projection * view * model * relativeTransformation * vec4(pos, 1.0);
    gl_Position = correction * gl_Position;
    texCoord = aPos;
}
