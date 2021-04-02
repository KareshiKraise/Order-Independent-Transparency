#shader VERTEX
#version 420 core

layout(location = 0) in vec3 position;
out vec3 TexCoords;

uniform mat4 M;
uniform mat4 P;
uniform mat4 V;

void main()
{
	vec4 pos = P * V * M * vec4(position , 1.0f);
	gl_Position = pos.xyww;
	TexCoords = position;
}


#shader FRAGMENT
#version 420 core

in vec3 TexCoords;
out vec4 color;

layout (binding = 0) uniform samplerCube envMap;

void main()
{
	color = texture(envMap, TexCoords);
	//color = vec4(1.0, 0.0, 0.0, 1.0f);
}