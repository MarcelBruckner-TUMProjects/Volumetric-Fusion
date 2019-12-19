#version 330 core
out vec4 FragColor;

in vec4 aColor;


void main()
{
    FragColor = aColor;

    vec2 pos = mod(gl_FragCoord.xy, vec2(50.0)) - vec2(25.0);
    float dist_squared = dot(pos, pos);
}