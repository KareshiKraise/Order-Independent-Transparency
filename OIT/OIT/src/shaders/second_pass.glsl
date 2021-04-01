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

layout(early_fragment_tests) in;

layout(binding = 0, r32ui) uniform coherent uimage2D counter_image;
layout(binding = 1, r32ui) uniform uimage2D base_image;
layout(binding = 2, rgba32ui) uniform writeonly coherent uimageBuffer dynamic_buffer;

layout(location = 0) out vec4 color;

in vec4 m_color;

void main()
{
	uint index;
	uvec4 item;
	index = imageLoad(base_image, ivec2(gl_FragCoord.xy)).x + imageLoad(counter_image, ivec2(gl_FragCoord.xy)).x;
	imageAtomicAdd(counter_image, ivec2(gl_FragCoord.xy), 1);

	item.x = packUnorm4x8(m_color);
	item.y = 0;
	item.z = floatBitsToUint(gl_FragCoord.z);
	item.w = 0;

	imageStore(dynamic_buffer, int(index), item );

	color = m_color;
	//color = imageLoad(base_image, ivec2(gl_FragCoord.xy));
}