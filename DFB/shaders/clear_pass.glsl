#shader VERTEX
#version 420 core

layout(location = 0) in vec4 position;

void main()
{
	gl_Position = position;
}

#shader FRAGMENT
#version 420 core

layout(binding = 0, r32ui) uniform uimage2D counter_image;

void main()
{
	ivec2 coords = ivec2(gl_FragCoord.xy);
	if (coords.x >= 0 && coords.y >= 0 && coords.x < 640 && coords.y < 480)
		imageStore(counter_image, coords, ivec4(0));
	
	discard;
}