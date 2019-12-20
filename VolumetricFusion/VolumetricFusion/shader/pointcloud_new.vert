#version 330 core
layout (location = 0) in vec2 aPos;

out vec2 texCoord;

uniform float depth_scale;
uniform float aspect;
uniform usampler2D depth_frame;
uniform mat3 cam2World;

uniform mat4 relativeTransformation;
uniform mat4 model;
uniform mat4 correction;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    float z = texture(depth_frame, aPos).x * depth_scale;

    if(z == 0){
        gl_Position = vec4(0.0f, 0.0f, 0.0f, 0.0f);
        texCoord = vec2(-1.0f, -1.0f);
        return;
    }

    vec3 pos = vec3(aPos - 0.5f, 1.0f); 
    if(aspect > 1.0f){
        pos.y /= aspect;
    }else{
        pos.x *= aspect;
    }

    pos *= z;
    //pos = cam2World * pos;
    // pos *= z;
   
    gl_Position = projection * view * model * relativeTransformation * vec4(pos, 1.0);
    gl_Position = correction * gl_Position;

    texCoord = aPos;
}
