#version 460 core

//in vec4 geom_color;

in VS_OUT {
    float sdf;
    float weight;
    vec3 normal;
    vec3 frag_pos;
} fs_in;

out vec4 FragColor;

uniform mat4 view;
uniform vec3 lightPos; 
uniform vec3 lightColor;
uniform vec3 objectColor;

void main()
{

    if (fs_in.sdf == 0.0f) {
        // hide
        FragColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }
    //FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f); return;
    
    // ambient
    float ambientStrength = 0.5;
    vec3 ambient = ambientStrength * lightColor;
  	
    // diffuse 
    vec3 norm = normalize(fs_in.normal);
    vec3 lightDir = normalize(lightPos - fs_in.frag_pos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // specular
    float specularStrength = 0.5;
    vec3 viewPos = vec3(view);
    vec3 viewDir = normalize(viewPos - fs_in.frag_pos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;  
        
    //vec3 result = (ambient + diffuse + specular) * objectColor;
    vec3 result = (ambient + diffuse) * objectColor;
    FragColor = vec4(result, 1.0);
}