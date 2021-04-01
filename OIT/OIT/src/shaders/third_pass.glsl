#shader VERTEX
#version 420 core

layout(location = 0) in vec4 position;


void main()
{
	gl_Position = position;
	
}


#shader FRAGMENT
#version 420 core

layout(binding = 0, r32ui) uniform coherent uimage2D counter_image;
layout(binding = 1, r32ui) uniform uimage2D base_image;
layout(binding = 2, rgba32ui) uniform coherent uimageBuffer dynamic_buffer;

layout(location = 0) out vec4 color;

#define MAX_FRAGS 10

uvec4 fragment_list[MAX_FRAGS];

void main()
{
	uint frag_count = imageLoad(counter_image, ivec2(gl_FragCoord.xy)).x;
	uint base_pos = imageLoad(base_image, ivec2(gl_FragCoord.xy)).x;


	//filling frag_list
	for (int i = 0; i < int(frag_count); i++)
	{
		fragment_list[i] = imageLoad(dynamic_buffer, int(base_pos + i));
	}
	
	//sorting step
	if (frag_count > 1)
	{
		int i = 0;
		int j = 0;
		for (i = 1; i < frag_count; i++)
		{
			uvec4 key = fragment_list[i];
			float keyz = uintBitsToFloat(key.z);

			j = i - 1;
			while (j >= 0 && (uintBitsToFloat(fragment_list[j].z) < keyz))
			{
				fragment_list[j + 1] = fragment_list[j];
				j = j - 1;
			}
			fragment_list[j + 1] = key;

		}
	}


	//blend

	vec4 final_color = vec4(1.0, 1.0, 1.0, 1.0);

	for (int i = 0; i < frag_count; i++)
	{
		vec4 mod = unpackUnorm4x8(fragment_list[i].x);
		//final_color = mix(final_color, mod, mod.a);
		final_color = (1 - mod.a)*final_color + (mod * mod.a);
		//final_color = mod*(1.0 - final_color.a) + final_color;
		

	}

	color = final_color;


}