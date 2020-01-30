#version 330 core
layout (location = 0) in vec2 aPos;

out vec2 texCoord;

uniform vec2 depth_resolution;
uniform float depth_scale;
uniform usampler2D depth_frame;
uniform mat3 cam2World;

uniform mat4 relativeTransformation;
uniform mat4 model;
uniform mat4 correction;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    vec2 uv = aPos / depth_resolution;
    float z = texture(depth_frame, uv).x * depth_scale;
    
//    vec3 pos = vec3(aPos * 2.0f, 1.0f) * cam2World; 
    vec3 pos = vec3(aPos, 1.0f) * cam2World; 
    
    pos *= z;
    
    gl_Position = vec4(pos, 1.0f);
    gl_Position = projection * view * model * relativeTransformation * gl_Position;
    gl_Position = correction * gl_Position;

    if(z <= 0.1){
        texCoord = vec2(-1.0f, -1.0f);
    }else{
        texCoord = uv;
    }
}