#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 texCoord;

uniform vec2 offset;
uniform vec2 aspect;

void main()
{
    vec2 pos = aPos;
    pos.y *= -1.0f;
    gl_Position = vec4((pos  + offset)* aspect, 1.0f, 1.0f);

    texCoord = aTexCoord;
}
