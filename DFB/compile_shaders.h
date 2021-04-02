#ifndef _COMPILE_SHADERS_
#define _COMPILE_SHADERS_

#include <iostream>
#include <GL/glew.h>
#include <string>
#include <fstream>
#include <sstream>

struct ShaderSource {
	std::string vertex_src;
	std::string frag_src;
};


static ShaderSource ParseShader(const std::string& filepath) {

	std::ifstream stream(filepath);

	enum class ShaderType {
		NONE = -1,
		VERTEX = 0,
		FRAGMENT = 1
	};

	std::string line;
	std::stringstream ss[2];
	ShaderType type = ShaderType::NONE;

	while (getline(stream, line))
	{
		if (line.find("#shader") != std::string::npos)
		{
			if (line.find("VERTEX") != std::string::npos)
			{
				//vertex
				type = ShaderType::VERTEX;
				std::cout << "found vertex" << std::endl;
			}
			else if (line.find("FRAGMENT") != std::string::npos)
			{
				//fragment
				type = ShaderType::FRAGMENT;
				std::cout << "found fragment" << std::endl;
			}
		}
		else
		{
			ss[(int)type] << line << '\n';
		}
	}

	ShaderSource src;
	src.vertex_src = ss[0].str();
	src.frag_src = ss[1].str();

	return src;


}

static unsigned int CompileShader(unsigned int type, const std::string& source) {

	unsigned int id = glCreateShader(type);
	const char* src = source.c_str();
	glShaderSource(id, 1, &src, nullptr);
	glCompileShader(id);

	int res;
	glGetShaderiv(id, GL_COMPILE_STATUS, &res);

	if (res == GL_FALSE) {

		int length;
		glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
		char *message = (char*)alloca(length * sizeof(char));
		glGetShaderInfoLog(id, length, &length, message);
		std::cout << "Failed to Compile " << (type == GL_VERTEX_SHADER ? "vertex" : "fragment") << " shader! " << std::endl;
		std::cout << message << std::endl;
		glDeleteShader(id);
		return 0;
	}

	return id;




}

static unsigned int CreateShader(const std::string& vertex, const std::string& frag) {

	unsigned int program = glCreateProgram();
	unsigned int vs = CompileShader(GL_VERTEX_SHADER, vertex);
	unsigned int fs = CompileShader(GL_FRAGMENT_SHADER, frag);

	glAttachShader(program, vs);
	glAttachShader(program, fs);

	glLinkProgram(program);
	glValidateProgram(program);

	glDeleteShader(vs);
	glDeleteShader(fs);

	return program;
}

//example usage
// ShaderSource src = ParseShader("my path");
// GLuint shader_program = CreateShader(src.vertex_src, src.frag_src);
//

#endif