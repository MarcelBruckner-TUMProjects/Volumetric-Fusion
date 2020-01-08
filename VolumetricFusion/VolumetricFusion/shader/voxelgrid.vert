#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float tsdf;
layout (location = 2) in float weights;
layout (location = 3) in float is_set;

out vec3 aColor;

uniform vec3 size;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    if(is_set < 1.0f){
        gl_Position = vec4(0.0f);
        aColor = vec3(0.0f, 0.0f, 0.0f);
        return;
    }

    gl_Position = projection * view * model * vec4(aPos, 1.0);
    gl_Position.xy *= -1;
//    aColor = aPos / (size / 2.0f);
    
    float t = tsdf;
//    t = 1;
    if(t < 0) {
        aColor = vec3(1.0f + t, 0.0f, 0.0f);
    } else {
        aColor = vec3(0.0, 1.0f - t, 0.0f);
    }
}
