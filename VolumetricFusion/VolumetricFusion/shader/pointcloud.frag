#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D color_frame;
uniform vec4 color;

void main()
{
    if(texCoord.x < 0){
        FragColor = vec4(1.0f,0.0f,0.0f,1.0f);
        return;
    }
    
    FragColor = texture(color_frame, texCoord);
//    FragColor = vec4(texCoord, 0.0, 1.0f);
//    FragColor = vec4(0.7f, 0.8f, 0.8f, 1.0f);
}
