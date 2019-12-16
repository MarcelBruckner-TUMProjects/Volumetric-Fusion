#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D onlyColorTexture;
uniform vec4 color;

void main()
{
    FragColor = texture(onlyColorTexture, texCoord) * color;
}
