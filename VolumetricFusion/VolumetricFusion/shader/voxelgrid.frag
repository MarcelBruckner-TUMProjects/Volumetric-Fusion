#version 330 core
out vec4 FragColor;

in vec3 aColor;


void main()
{
//    FragColor = vec4(vec3(1.0f, 1.0f, 1.0f) - aColor, 1.0f);
    FragColor = vec4(aColor, 1.0f);
}
