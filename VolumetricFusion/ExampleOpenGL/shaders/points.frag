#version 450 core

in vec3 fragPos;

uniform mat4 g_colorIntrinsics;
uniform mat4 g_colorExtrinsics;
uniform sampler2D g_sampler;
uniform int g_colorWidth;
uniform int g_colorHeight;

out vec4 outColor;

void main()
{
	vec4 pixelInColorImage = g_colorIntrinsics * (g_colorExtrinsics * vec4(fragPos.xyz, 1.0));
	pixelInColorImage /= pixelInColorImage.z;

	outColor = texture(g_sampler, vec2((pixelInColorImage.x + 0.5) / g_colorWidth, (pixelInColorImage.y + 0.5) / g_colorHeight));
	// outColor = texelFetch(g_sampler, ivec2(int(round(pixelInColorImage.x)), int(round(pixelInColorImage.y))), 0);
	// outColor = vec4(1, 0, 0, 1);
}