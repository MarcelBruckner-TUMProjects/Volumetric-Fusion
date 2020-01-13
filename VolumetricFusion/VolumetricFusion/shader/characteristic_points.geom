#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 8) out;

uniform float cube_radius;

in VS_OUT {
    vec3 color;
} gs_in[];  

out vec3 fColor;  

void assemble_cube(vec4 position)
{   
    fColor = gs_in[0].color;
//    fColor = vec3(1.0f, 0.0f, 0.0f);

    gl_Position = position + vec4(cube_radius, cube_radius, cube_radius, 0.0);    
    EmitVertex();   

    gl_Position = position + vec4(cube_radius, -cube_radius, cube_radius, 0.0);   
    EmitVertex();   

    gl_Position = position + vec4(-cube_radius, cube_radius, cube_radius, 0.0);   
    EmitVertex();   

    gl_Position = position + vec4(-cube_radius, -cube_radius, cube_radius, 0.0);  
    EmitVertex();   

    gl_Position = position + vec4(cube_radius, cube_radius, -cube_radius, 0.0);   
    EmitVertex();   

    gl_Position = position + vec4(cube_radius, -cube_radius, -cube_radius, 0.0);  
    EmitVertex();   

    gl_Position = position + vec4(-cube_radius, cube_radius, -cube_radius, 0.0);  
    EmitVertex();   

    gl_Position = position + vec4(-cube_radius, -cube_radius, -cube_radius, 0.0); 
    EmitVertex();   

    EndPrimitive();
}

void main() {    
    assemble_cube(gl_in[0].gl_Position);
}  