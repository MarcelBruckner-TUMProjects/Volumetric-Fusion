#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 texCoord;

uniform mat4 relativeTransformation;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    vec3 pos = aPos;
    pos.y *= -1;
    gl_Position = projection * view * model * relativeTransformation * vec4(pos, 1.0);
    texCoord = aTexCoord;
}
