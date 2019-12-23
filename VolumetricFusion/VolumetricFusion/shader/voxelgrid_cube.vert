#version 460 core
layout (location = 0) in vec3 pos;
layout (location = 1) in float sdf;
layout (location = 2) in float weight;

//out vec4 vert_color;
out VS_OUT {
    float sdf;
    float weight;
} vs_out;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(pos, 1.0);
    //gl_Position.y *= -1;
    //gl_PointSize = 2.0f;
    vs_out.sdf = sdf;
    vs_out.weight = weight;
}
