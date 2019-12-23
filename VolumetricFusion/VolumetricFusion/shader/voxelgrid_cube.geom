#version 460 core
layout (points) in;
layout (triangle_strip, max_vertices = 36) out;

//in vec4 vert_color[];
//out vec4 geom_color;

in VS_OUT {
    float sdf;
    float weight;
} gs_in[];

out VS_OUT {
    float sdf;
    float weight;
    vec3 normal;
    vec3 frag_pos;
} gs_out;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float resolution;

const float cube_data[] = {
    // vertex          normal
     
    -1.f,  1.f,  1.f,  0.0f,  0.0f,  -1.0f,   // Front-top-left
     1.f,  1.f,  1.f,  0.0f,  0.0f,  1.0f,   // Front-top-right
    -1.f, -1.f,  1.f,  0.0f,  0.0f,  1.0f,   // Front-bottom-left
     1.f, -1.f,  1.f,  0.0f,  -1.0f,  0.0f,   // Front-bottom-right
     1.f, -1.f, -1.f,  0.0f,  -1.0f,  0.0f,   // Back-bottom-right
     1.f,  1.f,  1.f,  1.0f,  0.0f,  0.0f,   // Front-top-right
     1.f,  1.f, -1.f,  1.0f,  0.0f,  0.0f,   // Back-top-right
    -1.f,  1.f,  1.f,  0.0f,  1.0f,  0.0f,   // Front-top-left
    -1.f,  1.f, -1.f,  0.0f,  1.0f,  0.0f,   // Back-top-left
    -1.f, -1.f,  1.f,  -1.0f,  0.0f,  0.0f,   // Front-bottom-left
    -1.f, -1.f, -1.f,  -1.0f,  0.0f,  0.0f,   // Back-bottom-left
     1.f, -1.f, -1.f,  0.0f,  -1.0f,  0.0f,   // Back-bottom-right
    -1.f,  1.f, -1.f,  0.0f,  -1.0f,  0.0f,   // Back-top-left
     1.f,  1.f, -1.f,  0.0f,  0.0f,  -1.0f,   // Back-top-right

//    -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
//     0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
//     0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
//     0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
//    -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
//    -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
//
//    -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
//     0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
//     0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
//     0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
//    -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
//    -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
//
//    -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
//    -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
//    -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
//    -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
//    -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
//    -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
//
//     0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
//     0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
//     0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
//     0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
//     0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
//     0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
//
//    -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
//     0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
//     0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
//     0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
//    -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
//    -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
//
//    -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
//     0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
//     0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
//     0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
//    -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
//    -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f
};

void main()
{
    // Must come before EndPrimitive
    gs_out.sdf = gs_in[0].sdf;
    gs_out.weight = gs_in[0].weight;
    
    //for (int i = 0; i < 6 * 6 * 6; i += 6) {
    for (int i = 0; i < cube_data.length; i += 6) {
        
        vec4 vertex = vec4(cube_data[i], cube_data[i+1], cube_data[i+2], 1.0f);
        vec3 normal = vec3(cube_data[i+3], cube_data[i+4], cube_data[i+5]);

        // scale vertex given the resolution
        vertex *= resolution;
        // normal from the raw data is already normalized, nothing to do here

        gs_out.frag_pos = vec3(model * vertex);
        gs_out.normal = mat3(transpose(inverse(model))) * normal;
        vec4 translation = projection * view * vec4(gs_out.frag_pos, 1.0f);
	    gl_Position = gl_in[0].gl_Position + translation;

        EmitVertex();
    }

    EndPrimitive();
}