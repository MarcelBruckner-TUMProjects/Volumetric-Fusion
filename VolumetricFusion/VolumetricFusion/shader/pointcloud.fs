#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D onlyColorTexture;
uniform vec4 color;

void main()
{
    vec2 texC = texCoord;
    //texC.x *= -1;
    //texC.y *= -1;
    FragColor = texture(onlyColorTexture, texC) * color;
    // FragColor = vec4(texC, 0.0, 1.0f);
}
