#shader VERTEX
#version 420 core

layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 tex;

uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

out vec2 Tex;

void main()
{
	vec4 Pos = P * V * M * vec4(pos, 1.0f);
	gl_Position = Pos;
	Tex = tex;
}


#shader FRAGMENT
#version 420 core

out vec4 color;

in vec2 Tex;

uniform sampler2D chess;

void main()
{
	color = vec4(texture(chess, Tex).rgb, 1.0f);
	//color = vec4(1.0, 0.0, 0.0, 1.0f);
}