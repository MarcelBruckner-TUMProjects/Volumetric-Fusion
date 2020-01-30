#version 330 core
layout (location = 0) in vec4 aPos;
//layout (location = 1) in vec3 aColor;

//uniform vec4 aColor;
uniform mat4 relativeTransformation;
uniform mat4 model;
uniform mat4 correction;
uniform mat4 view;
uniform mat4 projection;

uniform int numberOfVertices;

out VS_OUT {
    vec4 color;
} vs_out;

vec4 colorInterpolate(float pos){
    float r, b, g;
    if(pos <= 1.0f / 3.0f){
        r = pos * 3;
        g = 1 - r;
    }else if(pos <= 2.0f / 3.0f){
        g = (pos - (1.0f / 3.0f)) * 3.0f;
        b = 1 - g;
    }else {
        b = (pos - (2.0f / 3.0f)) * 3.0f;
        r = 1 - b;
    }
    return vec4(r, g, b, 1);
}

void main()
{
    float currentPosition = 1.0f * aPos.w / (numberOfVertices + 1);

    vs_out.color = colorInterpolate(currentPosition);

    gl_Position = vec4(aPos.xyz, 1.0f);
    gl_Position = projection * view * model * relativeTransformation * gl_Position;
    gl_Position = correction * gl_Position;
}