#version 450 core

in vec3 fragPos;
in vec3 fragNormal;

uniform vec3 g_ambient;
uniform vec3 g_diffuse;
uniform vec3 g_specular;
uniform float g_shiny;
uniform vec3 g_lightDir;
uniform vec3 g_eye;

out vec4 outColor;

void main()
{
    vec3 normal = normalize(fragNormal);
	vec3 lightDir = normalize(g_lightDir);
	vec3 viewDir = normalize(fragPos - g_eye);
	float diff = abs(dot(normal, lightDir)); // diffuse component

	// R = 2 * (N.L) * N - L
	vec3 reflect = normalize(2 * diff * normal - lightDir);
	float specularVal = 0.0;
	if (g_shiny > 0.0) {
		specularVal = pow(clamp(dot(reflect, viewDir), 0.0, 1.0), g_shiny); // R.V^n
	}
    vec3 specular = vec3(specularVal, specularVal, specularVal);

	// I = Acolor + Dcolor * N.L + (R.V)n
	vec3 color = g_ambient + 0.5 * (g_diffuse * diff + specular);
    outColor = vec4(color, 1.0);
    
    // For normal visualization.
    // outColor = vec4((normal + 1.0)/2.0, 1.0);
}