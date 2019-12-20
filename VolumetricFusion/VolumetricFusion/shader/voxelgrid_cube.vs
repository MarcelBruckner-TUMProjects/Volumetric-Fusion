#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float aSdf;
layout (location = 2) in float aWeight;
layout (location = 3) in vec3 cube;

out vec4 aColor;

uniform vec3 size;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    gl_Position.y *= -1;
    gl_PointSize = 2.0f;
    if (aSdf == 0.0f) {
        // hide
        aColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        aColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
    }
}
