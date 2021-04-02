//#include <iostream>
//#include <gl/glew.h>
//#include <gl/freeglut.h>
//#include "quads.h"
//#include "compile_shaders.h"
#include <iostream>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GLM/glm.hpp>
#include <GLM/mat4x4.hpp>
#include <GLM/gtc/matrix_transform.hpp>
#include <GLM/gtc/type_ptr.hpp>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "device_launch_parameters.h"
#include "quads.h"
#include <math.h>
#include <algorithm>
#include "compile_shaders.h"
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include "environment.h"
#include "camera.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobj_loader.h"

#ifdef _MSC_VER
#define ASSERT(x) if (!(x)) __debugbreak();
#endif

#define DEBUG

#ifdef DEBUG
#define GLCall(x) GLClearError(); \
	x;\
	ASSERT(GLLogCall(#x, __FILE__, __LINE__))
#else
#define GLCall(x) x
#endif

struct win_t
{
	float w;
	float h;
} win = { 1280.0f, 720.0f };

struct arcball
{
	int curx;
	int cury;
	int lastx;
	int lasty;
}track;

bool arcball_on = false;

struct gl_vars
{
	GLuint clearquad;

	GLuint fbo, fbo_depth, fbo_tex;

	GLuint vao, vbo, vboindex;
	GLuint first_pass, second_pass, third_pass, clear_pass;

	GLuint frag_count, frag_base;
	GLuint zero_pbo;

	GLuint dfb;
	GLuint dfb_tbo;
	GLsizei dfb_size;


	GLuint quad_vao;
	GLuint quad_vbo;

	cudaGraphicsResource *count_res;
	cudaGraphicsResource *base_res;

	GLuint *count_dev_ptr;
	GLuint *base_dev_ptr;

	cudaArray *countArray;
	cudaArray *baseArray;


} gl;

static const float quad_verts[] =
{
	-0.0f, -1.0f,
	1.0f, -1.0f,
	-0.0f,  1.0f,
	1.0f,  1.0f,
};

static const float quad_verts2[] =
{
	-1.0f, -1.0f,
	1.0f, -1.0f,
	-1.0f, 1.0f,
	1.0f, 1.0f,
};

glm::mat4 Model = glm::mat4(1.0f);
glm::vec3 eyePos = glm::vec3(0.0f, 0.0f, 1.0f);

Bound boundary;

glm::vec3 forward;
glm::mat4 View;
glm::mat4 Proj;

std::vector<glm::uvec4> tbo_data;

Camera *cam;
EnvironmentMap *envmap;
GLuint envmap_shader;
/*---------------------DRAGON TINY OBJ--------*/
tinyobj::attrib_t attrib;
std::vector<tinyobj::shape_t> shapes;
std::vector<tinyobj::material_t> materials;
std::vector<vertex2> dragonVertices;
std::vector<int> dragonIndices;
GLuint dragonVAO;
GLuint dragonVBO;
glm::vec3 objcenter;

float curX;
float curY;
float lastX;
float lastY;
bool cam_mouse_down = false;
/*Error Checking */
static void GLClearError()
{
	while (glGetError() != GL_NO_ERROR);
}

static bool GLLogCall(const char* function, const char* file, int line)
{
	while (GLenum err = glGetError())
	{
		std::cout << "[OpenGL Error] (" << std::hex << err << "):" << function << " " << file << " " << line << std::endl;
		return false;
	}
	return true;
}
/*End of error checking */

GLuint cubeVAO = 0;
GLuint cubeVBO = 0;
void RenderCube()
{
	if (cubeVAO == 0)
	{
		GLfloat vertices[] = {
			// Back face
			-0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // Bottom-left
			0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
			0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 1.0f, 0.0f, // bottom-right         
			0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f,  // top-right
			-0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f,  // bottom-left
			-0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 0.0f, 1.0f,// top-left
			 // Front face
			 -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
			 0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f, 1.0f, 0.0f,  // bottom-right
			 0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f,  // top-right
			 0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
			 -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f,  // top-left
			 -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f,  // bottom-left
			 // Left face
			 -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
			 -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-left
			 -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f,  // bottom-left
			 -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
			 -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f, 0.0f, 0.0f,  // bottom-right
			 -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
			  // Right face
			  0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
			  0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
			  0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-right         
			  0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f,  // bottom-right
			  0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f,  // top-left
			  0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-left     
			  // Bottom face
			  -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
			  0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f, 1.0f, 1.0f, // top-left
			  0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f,// bottom-left
			  0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
			  -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f, 0.0f, 0.0f, // bottom-right
			  -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
			 // Top face
			 -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f,// top-left
			 0.5f,  0.5f , 0.5f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
			 0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f, 1.0f, 1.0f, // top-right     
			 0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
			 -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f,// top-left
			-0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f, 0.0f, 0.0f // bottom-left        
		};
		glGenVertexArrays(1, &cubeVAO);
		glGenBuffers(1, &cubeVBO);
		// Fill buffer
		glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
		// Link vertex attributes
		glBindVertexArray(cubeVAO);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}
	// Render Cube
	glBindVertexArray(cubeVAO);
	glDrawArrays(GL_TRIANGLES, 0, 36);
	glBindVertexArray(0);
}

GLuint chess;
GLuint quad_base;
GLuint quadVAO = 0;
GLuint quadVBO = 0;
void RenderQuad()
{
	if (quadVAO == 0)
	{
		GLfloat quadVertices[] = {
			// Positions        // Texture Coords
			-1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
			-1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f,
			1.0f,  0.0f, 1.0f, 0.0f, 1.0f, 1.0f,
			1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f,
		};
		// Setup plane VAO
		glGenVertexArrays(1, &quadVAO);
		glGenBuffers(1, &quadVBO);
		glBindVertexArray(quadVAO);
		glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)(4 * sizeof(GLfloat)));
	}
	glBindVertexArray(quadVAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
}

void loadChess()
{
	unsigned char* image;
	int w, h, n;
	image = stbi_load("chess.png", &w, &h, &n, 0);
	if (!image)
	{
		std::cout << "failed to laod image chess" << std::endl;
	}

	GLCall(glGenTextures(1, &chess););
	GLCall(glBindTexture(GL_TEXTURE_2D, chess));
	GLCall(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, image));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));
	GLCall(glBindTexture(GL_TEXTURE_2D, 0));
}

void initShaders()
{
	ShaderSource src1, src2, src3, src4, src5, src6;
	src1 = ParseShader("shaders/first_pass.glsl");
	src2 = ParseShader("shaders/second_pass.glsl");
	src3 = ParseShader("shaders/third_pass.glsl");
	src4 = ParseShader("shaders/clear_pass.glsl");
	src5 = ParseShader("shaders/envmap.glsl");
	src6 = ParseShader("shaders/quad_base.glsl");
	gl.first_pass  = CreateShader(src1.vertex_src, src1.frag_src);
	gl.second_pass = CreateShader(src2.vertex_src, src2.frag_src);
	gl.third_pass  = CreateShader(src3.vertex_src, src3.frag_src);
	gl.clear_pass  = CreateShader(src4.vertex_src, src4.frag_src);
	envmap_shader  = CreateShader(src5.vertex_src, src5.frag_src);
	quad_base = CreateShader(src6.vertex_src, src5.frag_src);
}

void initVertexArrays()
{

	GLCall(glGenVertexArrays(1, &gl.vao));
	GLCall(glBindVertexArray(gl.vao));



	GLCall(glGenBuffers(1, &gl.vbo));
	GLCall(glBindBuffer(GL_ARRAY_BUFFER, gl.vbo));
	GLCall(glBufferData(GL_ARRAY_BUFFER, sizeof(planes), planes, GL_STATIC_DRAW));


	GLCall(glEnableVertexAttribArray(0));
	GLCall(glEnableVertexAttribArray(1));

	GLCall(glVertexAttribPointer(0, 3, GL_FLOAT, false, sizeof(vertex), 0));
	GLCall(glVertexAttribPointer(1, 4, GL_FLOAT, false, sizeof(vertex), (void*)offsetof(vertex, col)));

	GLCall(glDisableVertexAttribArray(0));
	GLCall(glDisableVertexAttribArray(1));


	GLCall(glGenBuffers(1, &gl.vboindex));
	GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gl.vboindex));
	GLCall(glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(planes_ind), planes_ind, GL_STATIC_DRAW));

	/*---------------------------------------------------------------------------*/

	GLCall(glGenTextures(1, &gl.frag_count));
	GLCall(glBindTexture(GL_TEXTURE_2D, gl.frag_count));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
	GLCall(glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, win.w, win.h, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, NULL));
	GLCall(glBindTexture(GL_TEXTURE_2D, 0));

	GLCall(glGenBuffers(1, &gl.zero_pbo));
	GLCall(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl.zero_pbo));
	GLCall(glBufferData(GL_PIXEL_UNPACK_BUFFER, win.w * win.h * sizeof(GLuint), NULL, GL_STATIC_DRAW));
	GLuint *data = (GLuint*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
	memset(data, 0x00, win.w*win.h * sizeof(GLuint));
	GLCall(glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER));


	GLCall(glGenTextures(1, &gl.frag_base));
	GLCall(glBindTexture(GL_TEXTURE_2D, gl.frag_base));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
	GLCall(glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, win.w, win.h, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, NULL));
	GLCall(glBindTexture(GL_TEXTURE_2D, 0));

	GLCall(glGenBuffers(1, &gl.dfb));
	GLCall(glBindBuffer(GL_TEXTURE_BUFFER, gl.dfb));

	gl.dfb_size = win.w * win.h * sizeof(glm::uvec4);
	GLCall(glBufferData(GL_TEXTURE_BUFFER, gl.dfb_size, NULL, GL_DYNAMIC_COPY));
	GLCall(glBindBuffer(GL_TEXTURE_BUFFER, 0));


	GLCall(glGenTextures(1, &gl.dfb_tbo));
	GLCall(glBindTexture(GL_TEXTURE_BUFFER, gl.dfb_tbo));
	GLCall(glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32UI, gl.dfb));
	GLCall(glBindTexture(GL_TEXTURE_BUFFER, 0));


	cudaError_t error;
	error = cudaGraphicsGLRegisterImage(&gl.count_res, gl.frag_count, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);
	if (error != ::cudaSuccess)
	{
		printf("cudaGraphicsGLRegisterImage error during count resource [%d]: ", error);
	}

	error = cudaGraphicsGLRegisterImage(&gl.base_res, gl.frag_base, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);
	if (error != ::cudaSuccess)
	{
		printf("cudaGraphicsGLRegisterImage error during base resource [%d]: ", error);
	}

	/*---------------------------------------------------------------------------*/

	GLCall(glBindVertexArray(0));
	GLCall(glBindBuffer(GL_ARRAY_BUFFER, 0));
	GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));

	GLCall(glGenVertexArrays(1, &gl.quad_vao));
	GLCall(glBindVertexArray(gl.quad_vao));

	GLCall(glGenBuffers(1, &gl.quad_vbo));
	GLCall(glBindBuffer(GL_ARRAY_BUFFER, gl.quad_vbo));
	GLCall(glBufferData(GL_ARRAY_BUFFER, sizeof(quad_verts), quad_verts, GL_STATIC_DRAW));

	GLCall(glEnableVertexAttribArray(0));
	GLCall(glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL));
	GLCall(glDisableVertexAttribArray(0));


	GLCall(glBindBuffer(GL_ARRAY_BUFFER, 0));
	GLCall(glBindVertexArray(0));

	/********************************************************************/


	/********************************************************************/

	//binding texture buffers
	GLCall(glBindImageTexture(0, gl.frag_count, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI));
	GLCall(glBindImageTexture(1, gl.frag_base, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI));
	GLCall(glBindImageTexture(2, gl.dfb_tbo, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32UI));

}

void initQuad()
{
	GLCall(glGenBuffers(1, &gl.clearquad));
	GLCall(glBindBuffer(GL_ARRAY_BUFFER, gl.clearquad));
	GLCall(glBufferData(GL_ARRAY_BUFFER, sizeof(quad_verts2), quad_verts2, GL_STATIC_DRAW));

	GLCall(glEnableVertexAttribArray(0));
	GLCall(glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL));
	GLCall(glDisableVertexAttribArray(0));

	GLCall(glBindBuffer(GL_ARRAY_BUFFER, 0));
}

void initDragon()
{
	GLCall(glGenVertexArrays(1, &dragonVAO));
	GLCall(glGenBuffers(1, &dragonVBO));
	
	GLCall(glBindVertexArray(dragonVAO));
	GLCall(glBindBuffer(GL_ARRAY_BUFFER, dragonVBO));
	GLCall(glBufferData(GL_ARRAY_BUFFER, dragonVertices.size() * sizeof(vertex2), &dragonVertices[0], GL_STATIC_DRAW));
	GLCall(glEnableVertexAttribArray(0));
	GLCall(glVertexAttribPointer(0, 3, GL_FLOAT ,GL_FALSE, sizeof(vertex2), (void*)0));
	GLCall(glEnableVertexAttribArray(1));
	GLCall(glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(vertex2), (void*)offsetof(vertex2, tex)));
	GLCall(glEnableVertexAttribArray(2));
	GLCall(glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(vertex2), (void*)offsetof(vertex2, col)));

	GLCall(glBindVertexArray(0));
}

void renderDragon()
{
	GLCall(glBindVertexArray(dragonVAO));
	
	GLCall(glDrawArrays(GL_TRIANGLES, 0, dragonVertices.size()));

}

glm::vec3 center_of_mass(std::vector<vertex2>& vec)
{
	float minx = dragonVertices[0].pos.x;
	float miny = dragonVertices[0].pos.y;
	float minz = dragonVertices[0].pos.z;

	float maxx = dragonVertices[0].pos.x;
	float maxy = dragonVertices[0].pos.y;
	float maxz = dragonVertices[0].pos.z;

	for (const auto& v : dragonVertices)
	{
		if (v.pos.x <= minx)
		{
			minx = v.pos.x;
		}
		if (v.pos.y <= miny)
		{
			miny = v.pos.y;
		}
		if (v.pos.z <= minz)
		{
			minz = v.pos.z;
		}
		if (v.pos.x >= maxx)
		{
			maxx = v.pos.x;
		}
		if (v.pos.y >= maxy)
		{
			maxy = v.pos.y;
		}
		if (v.pos.z >= maxz)
		{
			maxz = v.pos.z;
		}
	}
	return glm::vec3( (minx + maxx)/2.0f, 
					  (miny + maxy)/2.0f, 
		              (minz + maxz)/2.0f);
}

void resetBuffer()
{
	GLCall(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl.zero_pbo));
	GLCall(glBindTexture(GL_TEXTURE_2D, gl.frag_count));
	GLCall(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, win.w, win.h, GL_RED_INTEGER, GL_UNSIGNED_INT, NULL));
	GLCall(glBindTexture(GL_TEXTURE_2D, 0));



	GLCall(glBindImageTexture(0, gl.frag_count, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI));
	GLCall(glBindImageTexture(1, gl.frag_base, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI));
	GLCall(glBindImageTexture(2, gl.dfb_tbo, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32UI));
}

void clearBuffer()
{
	GLCall(glUseProgram(gl.clear_pass));
	GLCall(glBindBuffer(GL_ARRAY_BUFFER, gl.clearquad));

	GLCall(glEnableVertexAttribArray(0));

	GLCall(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));

	GLCall(glDisableVertexAttribArray(0));


	GLCall(glBindBuffer(GL_ARRAY_BUFFER, 0));
	GLCall(glUseProgram(0));
}

void cuda_map()
{
	cudaError_t error;
	if (cudaGraphicsMapResources(1, &gl.count_res, NULL) != ::cudaSuccess)
	{
		std::cout << "failed to map resource count" << std::endl;
	}
	error = cudaGraphicsSubResourceGetMappedArray(&gl.countArray, gl.count_res, 0, 0);
	if (error != ::cudaSuccess)
	{
		std::cout << "couldnt get mapped pointer to resource count" << std::endl;
	}


	if (cudaGraphicsMapResources(1, &gl.base_res, NULL) != ::cudaSuccess)
	{
		std::cout << "failed to map resource base" << std::endl;
	}
	error = cudaGraphicsSubResourceGetMappedArray(&gl.baseArray, gl.base_res, 0, 0);
	if (error != ::cudaSuccess)
	{
		std::cout << "couldnt get mapped pointer to resource base" << std::endl;
	}

}

void cuda_unmap()
{
	cudaGraphicsUnmapResources(1, &gl.count_res, NULL);
	cudaGraphicsUnmapResources(1, &gl.base_res, NULL);
}

/*-------------------------------------------------------------------------------*/

glm::vec3 get_arcball_vector(int _x, int _y)
{
	glm::vec3 P = glm::vec3((((float)_x / (win.w / 2.0f)) - 1.0f),
		(((float)_y / (win.h / 2.0f)) - 1.0f),
		0.0f);

	P.y = -P.y;
	float OP_squared = (P.x * P.x) + (P.y * P.y);

	if (OP_squared <= 1)
	{
		P.z = sqrt(1 - OP_squared);
		P = glm::normalize(P);
	}
	else
	{
		P = glm::normalize(P);
	}

	return P;

}

void onDisplay()
{
	Model = glm::translate(glm::mat4(1.0f), -objcenter);
	View = cam->GetViewMatrix();//glm::lookAt(eyePos, glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	Proj = glm::perspective(65.0f, (float)win.w / (float)win.h, 1.0f, 10000.0f);
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);

	//GLCall(glUseProgram(quad_base));
	glm::mat4 envM = glm::scale(glm::mat4(1.0f), glm::vec3(100.0f));
	//GLCall(glUniformMatrix4fv(glGetUniformLocation(quad_base, "M"), 1, GL_FALSE, &envM[0][0]));
	//GLCall(glUniformMatrix4fv(glGetUniformLocation(quad_base, "V"), 1, GL_FALSE, &View[0][0]));
	//GLCall(glUniformMatrix4fv(glGetUniformLocation(quad_base, "P"), 1, GL_FALSE, &Proj[0][0]));
	//GLCall(glActiveTexture(GL_TEXTURE0));
	//GLCall(glUniform1i(glGetUniformLocation(quad_base, "chess"), 0));
	//GLCall(glBindTexture(GL_TEXTURE_2D, chess));
	//
	//RenderQuad();

	glDepthFunc(GL_LEQUAL);

	GLCall(glUseProgram(envmap_shader));
	//glm::mat4 envM = glm::scale(glm::mat4(1.0f), glm::vec3(100.0f));
	
	GLCall(glUniformMatrix4fv(glGetUniformLocation(envmap_shader, "M"), 1, GL_FALSE, &envM[0][0]));
	GLCall(glUniformMatrix4fv(glGetUniformLocation(envmap_shader, "V"), 1, GL_FALSE, &View[0][0]));
	GLCall(glUniformMatrix4fv(glGetUniformLocation(envmap_shader, "P"), 1, GL_FALSE, &Proj[0][0]));

	GLCall(glActiveTexture(GL_TEXTURE0));
	GLCall(glUniform1i(glGetUniformLocation(envmap_shader, "envMap"), 0));
	GLCall(glBindTexture(GL_TEXTURE_CUBE_MAP, envmap->cubeMap));
	RenderCube();
	//GLCall(glDepthFunc(GL_LESS));
	
	




	//Disable backface culling to keep all fragments
	glDisable(GL_CULL_FACE);
	//Disable depth test
	glDisable(GL_DEPTH_TEST);
	//Disable stencil test
	glDisable(GL_STENCIL_TEST);

	/*****************************************************************/
	clearBuffer();

	GLCall(glMemoryBarrier(GL_ALL_BARRIER_BITS));
	/*****************************************************************/

	glm::mat4 MVP = Proj * View * Model;

	GLCall(glUseProgram(gl.first_pass));

	//GLCall(glBindVertexArray(gl.vao));
    //GLCall(glBindBuffer(GL_ARRAY_BUFFER, gl.vbo));
	//GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gl.vboindex));

	GLCall(glEnable(GL_BLEND));
	GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

	//GLCall(glEnableVertexAttribArray(0));
	//GLCall(glEnableVertexAttribArray(1));

	GLint matrixID = glGetUniformLocation(gl.first_pass, "MVP");
	if (matrixID == GL_INVALID_INDEX) {
		printf("Uniform %s not found in shader!", "MVP");
	}
	GLCall(glUniformMatrix4fv(matrixID, 1, GL_FALSE, &MVP[0][0]));

	renderDragon();

	//GLCall(glDrawElements(GL_TRIANGLES, (GLsizei)planes_ind_count, GL_UNSIGNED_SHORT, 0));

	//GLCall(glDisableVertexAttribArray(0));
	//GLCall(glDisableVertexAttribArray(1));

	GLCall(glUseProgram(0));
	GLCall(glBindVertexArray(0));
	//GLCall(glBindBuffer(GL_ARRAY_BUFFER, 0));
	//GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));


	GLCall(glMemoryBarrier(GL_ALL_BARRIER_BITS));
	/********************************************************************************************/
	cuda_map();

	size_t buf_size = win.w * win.h * sizeof(GLuint);
	thrust::device_vector<GLuint> count(size_t(win.w*win.h));
	thrust::device_vector<GLuint> base(size_t(win.w*win.h));

	GLuint* cnt_ptr = thrust::raw_pointer_cast(&count[0]);
	GLuint* bse_ptr = thrust::raw_pointer_cast(&base[0]);

	cudaError_t error;

	error = cudaMemcpyFromArray(cnt_ptr, gl.countArray, 0, 0, buf_size, cudaMemcpyDeviceToDevice);
	if (error != ::cudaSuccess)
	{
		std::cout << "memcpy failed when copying from count array" << std::endl;
	}

	thrust::exclusive_scan(thrust::device, count.begin(), count.end(), base.begin());

	error = cudaMemcpyToArray(gl.baseArray, 0, 0, bse_ptr, buf_size, cudaMemcpyDeviceToDevice);
	if (error != ::cudaSuccess)
	{
		std::cout << "memcpy failed when copying to base array" << std::endl;
	}


	/*zero counter*/
	thrust::fill(count.begin(), count.end(), 0);

	error = cudaMemcpyToArray(gl.countArray, 0, 0, cnt_ptr, buf_size, cudaMemcpyDeviceToDevice);
	if (error != ::cudaSuccess)
	{
		std::cout << "memcpy failed when copying to base array" << std::endl;
	}

	cuda_unmap();
	/********************************************************************************************/


	GLCall(glUseProgram(gl.second_pass));

	//GLCall(glBindVertexArray(gl.vao));
	//GLCall(glBindBuffer(GL_ARRAY_BUFFER, gl.vbo));
	//GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gl.vboindex));

	GLCall(glEnable(GL_BLEND));
	GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

	//GLCall(glEnableVertexAttribArray(0));
	//GLCall(glEnableVertexAttribArray(1));

	matrixID = glGetUniformLocation(gl.first_pass, "MVP");
	if (matrixID == GL_INVALID_INDEX) {
		printf("Uniform %s not found in shader!", "MVP");
	}
	GLCall(glUniformMatrix4fv(matrixID, 1, GL_FALSE, &MVP[0][0]));
	renderDragon();
	//GLCall(glDrawElements(GL_TRIANGLES, (GLsizei)planes_ind_count, GL_UNSIGNED_SHORT, 0));

	//GLCall(glDisableVertexAttribArray(0));
	//GLCall(glDisableVertexAttribArray(1));

	GLCall(glUseProgram(0));
	GLCall(glBindVertexArray(0));
	//GLCall(glBindBuffer(GL_ARRAY_BUFFER, 0));
	//GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));

	/************************************************************************************************/

	GLCall(glMemoryBarrier(GL_ALL_BARRIER_BITS));
	/************************************************************************************************/
	//get buffer
	//GLCall(glBindBuffer(GL_TEXTURE_BUFFER, gl.dfb));
	//GLCall(glBindTexture(GL_TEXTURE_BUFFER, gl.dfb_tbo));
	//GLCall(glGetBufferSubData(GL_TEXTURE_BUFFER, 0, gl.dfb_size, &tbo_data[0] ));
	//end of get buffer

	//from here
	GLCall(glDisable(GL_BLEND));
	
	GLCall(glUseProgram(gl.third_pass));
	
	GLCall(glBindVertexArray(gl.quad_vao));
	GLCall(glBindBuffer(GL_ARRAY_BUFFER, gl.quad_vbo));
	GLCall(glEnableVertexAttribArray(0));
	
	GLCall(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
	
	GLCall(glDisableVertexAttribArray(0));
	GLCall(glBindBuffer(GL_ARRAY_BUFFER, 0));
	GLCall(glBindVertexArray(0));
	
	
	GLCall(glMemoryBarrier(GL_ALL_BARRIER_BITS));

	glutSwapBuffers();
}

void onResize(int w, int h)
{
	glViewport(0, 0, w, h);
	win.w = w;
	win.h = h;
		
	Proj = glm::perspective(glm::radians(60.0f), (float)w / (float)h, 0.1f, 1000.0f);
	glutPostRedisplay();
}

void onIdle()
{
	//if (track.curx != track.lastx || track.cury != track.lasty)
	//{
	//	glm::vec3 va = get_arcball_vector(track.lastx, track.lasty);
	//	glm::vec3 vb = get_arcball_vector(track.curx, track.cury);
	//
	//	float angle = acos(std::min(1.0f, glm::dot(va, vb)));
	//
	//	angle = angle * 0.05f;
	//
	//	glm::vec3 axis_in_camera = glm::cross(va, vb);
	//
	//	glm::mat3 inverse = glm::inverse(glm::mat3(View) * glm::mat3(Model));
	//
	//	glm::vec3 axis_model = inverse * axis_in_camera;
	//
	//	Model = glm::translate(Model, objcenter);
	//	Model = glm::rotate(Model, glm::degrees(angle), axis_model);
	//	Model = glm::translate(Model, -objcenter);
	//
	//	//Model = glm::rotate(Model, glm::degrees(angle), axis_in_camera);
	//
	//	//View = glm::rotate(View, glm::degrees(angle), axis_in_camera);
	//	//View = glm::rotate(View, glm::degrees(angle), axis_model);
	//
	//	track.lastx = track.curx;
	//	track.lasty = track.cury;
	//
	//}
	glutPostRedisplay();

}


void onMouse(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		//arcball_on = true;
		//track.lastx = track.curx = x;
		//track.lasty = track.cury = y;
		cam_mouse_down = true;
		lastX = curX = x;
		lastY = curY = y;
		
	}

	if (button == GLUT_LEFT_BUTTON && state == GLUT_UP)
	{
		//arcball_on = false;
		cam_mouse_down = false;
	}

	glutPostRedisplay();
}

void onMotion(int x, int y)
{
	if (cam_mouse_down)
	{
		curX = x;
		curY = y;

		float xoff = curX - lastX;
		float yoff = lastY - curY;

		lastX = x;
		lastY = y;

		cam->ProcessMouseMovement(xoff, yoff);
	}
	
	if (arcball_on)
	{
		track.curx = x;
		track.cury = y;
	}
	glutPostRedisplay();
}

void onKeyboard(unsigned char key, int x, int y)
{
	switch (key) {
	case '+':
		std::cout << " + was pressed" << std::endl;
		eyePos = eyePos - glm::vec3(0, 0, 15);
		break;
	case '-':
		std::cout << " - was pressed" << std::endl;
		eyePos = eyePos + glm::vec3(0, 0, 15);
		break;
	case 'w':
		cam->ProcessKeyboard(FORWARD, 1);
		break;
	case 'a':
		cam->ProcessKeyboard(LEFT, 1);
		break;
	case 's':
		cam->ProcessKeyboard(BACKWARD, 1);
		break;
	case 'd':
		cam->ProcessKeyboard(RIGHT, 1);
		break;

	default:
		break;

	}
}

void renderScene(void) {

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glBegin(GL_TRIANGLES);
	glVertex3f(-0.5, -0.5, 0.0);
	glVertex3f(0.5, 0.0, 0.0);
	glVertex3f(0.0, 0.5, 0.0);
	glEnd();

	glutSwapBuffers();
}

void initGL(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(win.w, win.h);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);	
	glutCreateWindow("OIT- DFB");
	glClearColor(1.0, 1.0, 1.0, 1.0);
	glutDisplayFunc(onDisplay);
	glutReshapeFunc(onResize);
	glutIdleFunc(onIdle);
	glutMouseFunc(onMouse);
	glutMotionFunc(onMotion);
	glutKeyboardFunc(onKeyboard);
	
	GLenum gl_error;
	if ((gl_error = glewInit()) != GLEW_OK)
	{
		std::cout << "failed to init glew" << std::endl;
		std::cout << glewGetErrorString(gl_error) << std::endl;
	}

	glEnable(GL_DEPTH_TEST);
}

int main(int argc, char **argv)
{
	initGL(&argc, argv);

	tbo_data = std::vector<glm::uvec4>(win.w*win.h * sizeof(glm::uvec4));

	cudaDeviceProp prop;
	int device;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 3;
	prop.minor = 0;

	cudaChooseDevice(&device, &prop);
	cudaGLSetGLDevice(device);

	boundary = calc_bb();
	forward = boundary.center;
	View = glm::lookAt(eyePos, forward, glm::vec3(0.0f, 1.0f, 0.0f));
	Proj = glm::perspective(glm::radians(60.0f), (float)win.w / (float)win.h, 0.1f, 1000.0f);

	std::vector<std::string> map;
	map.push_back("yokohama/posx.jpg");
	map.push_back("yokohama/negx.jpg");
	map.push_back("yokohama/posy.jpg");
	map.push_back("yokohama/negy.jpg");
	map.push_back("yokohama/posz.jpg");
	map.push_back("yokohama/negz.jpg");
	envmap = new EnvironmentMap(map);

	loadChess();

	initQuad();
	initVertexArrays();
	initShaders();

	std::string filepath = "nanosuit/nanosuit.obj";
	std::string err;
	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filepath.c_str());
	if (!err.empty())
	{
		std::cout << err << std::endl;
	}
	if (!ret)
	{
		std::cout << "failed to load" << std::endl;
	}
	
	for (auto& s : shapes)
	{
		std::cout << "shape is named: " << s.name << std::endl;
		for (auto& index : s.mesh.indices)
		{
			vertex2 vert;
			vert.pos = glm::vec3(attrib.vertices[3 * index.vertex_index + 0], attrib.vertices[3 * index.vertex_index + 1], attrib.vertices[3 * index.vertex_index + 2]);
			vert.tex = glm::vec2(attrib.texcoords[2 * index.texcoord_index + 0] ,1.0f - attrib.texcoords[2 * index.texcoord_index + 1] );
			vert.col = glm::vec4(0.2f, 0.2f, 0.8f, 0.35f);
			
			dragonVertices.push_back(vert);			
		}
	}
	objcenter = center_of_mass(dragonVertices);
	std::cout << "center x: " << objcenter.x << std::endl;
	std::cout << "center y: " << objcenter.y << std::endl;
	std::cout << "center z: " << objcenter.z << std::endl;
	initDragon();

	Model = glm::translate(glm::mat4(1.0f), -objcenter);

	eyePos = glm::vec3(0.0f, 0.0f, 0.0f);
	cam = new Camera(eyePos, glm::vec3(0.0f, 1.0f, 0.0f));
	glutMainLoop();

	cudaGraphicsUnregisterResource(gl.count_res);
	cudaGraphicsUnregisterResource(gl.base_res);

	std::cout << "quitting" << std::endl;
	std::cin.get();

	return 0;
}