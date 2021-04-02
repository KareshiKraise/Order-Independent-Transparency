#shader VERTEX
#version 420 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;

uniform mat4 MVP;

out vec4 m_color;

void main()
{
	gl_Position = (MVP)* vec4(position, 1.0);
	m_color = color;
}

#shader FRAGMENT
#version 420 core

//layout(early_fragment_tests) in;

layout(binding = 0, r32ui) uniform uimage2D counter_image;

layout(location = 0) out vec4 color;

in vec4 m_color;

void main()
{
	
	imageAtomicAdd(counter_image, ivec2(gl_FragCoord.xy), 1);
	
	color = m_color;
	//color = imageLoad(counter_image, ivec2(gl_FragCoord.xy));
	
}