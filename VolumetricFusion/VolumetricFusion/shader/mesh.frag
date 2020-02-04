#version 330 core
out vec4 FragColor;

in vec4 vertexColor;
in vec3 vertexNormal;
in vec3 fragPos;

void main()
{
	vec3 lightDir = normalize(vec3(0,0,-1000) - fragPos);
	float diff = max(dot(vertexNormal, lightDir), 0);
	vec3 diffuse = diff * vec3(1,1,1);
	FragColor = vec4(diffuse, 1) * vertexColor;
}