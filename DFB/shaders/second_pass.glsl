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
layout(binding = 1, r32ui) uniform uimage2D base_image;
layout(binding = 2, rgba32ui) uniform writeonly uimageBuffer dynamic_buffer;

layout(location = 0) out vec4 color;

in vec4 m_color;

void main()
{
	uint index1;
	uint index2;
	uvec4 item;
	index1 = imageLoad(base_image, ivec2(gl_FragCoord.xy)).x;
	//index2 = imageLoad(counter_image, ivec2(gl_FragCoord.xy)).x;

	
	index2 = imageAtomicAdd(counter_image, ivec2(gl_FragCoord.xy), 1);
	

	item.x = packUnorm4x8(m_color);
	item.y = 0;
	item.z = floatBitsToUint(gl_FragCoord.z);
	item.w = 0;	

	imageStore(dynamic_buffer, int(index1 + index2), item );	
	memoryBarrier();

	color = m_color;
	//color = imageLoad(counter_image, ivec2(gl_FragCoord.xy));
}