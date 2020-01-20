#version 330 core
layout (location = 0) in vec4 aPos;
layout (location = 1) in vec4 tsdf;

uniform vec3 size;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 coordinate_correction;

uniform float truncationDistance;

out VS_OUT {
    vec4 color;
} vs_out;

void main()
{
    gl_Position = projection * view * model * aPos;
    gl_Position *= coordinate_correction;
    
    float t = tsdf.y;

    // Blue: invalid point
    if(t > 1){
        vs_out.color = vec4(0.0, 0.0, 1.0, 1.0);
        return;
    }

    if(t < 0) {
        // Red: Behind poindcloud
        vs_out.color = vec4(0.0f, (truncationDistance + t) / truncationDistance, 0.0f, 1.0f + t);
    } 
    else
    {
        // Green: Infront of pointcloud
        vs_out.color = vec4((truncationDistance - t) / truncationDistance, 0.0f, 0.0f, 1.0f - t);
    }
}
