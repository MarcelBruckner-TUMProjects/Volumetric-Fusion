#version 330 core
layout (location = 0) in vec4 aPos;
layout (location = 1) in vec4 tsdf;

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
// Blue: invalid point
    if(tsdf.z <= 0){
        vs_out.color = vec4(0.0, 0.0, 0.0, -1.0);
        gl_Position = vec4(-10.0, 0.0, 0.0, 0.0);
//        vs_out.color = vec4(0.0, 0.0, 1, .5);
        return;
    }

    gl_Position = projection * view * model * aPos;
    gl_Position *= coordinate_correction;
    
    float t = tsdf.y;
    
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
