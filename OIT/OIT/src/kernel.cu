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
} win = {640.0f, 480.0f};

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
	GLuint vao, vbo, vboindex;
	GLuint first_pass, second_pass, third_pass;

	GLuint frag_count, frag_base;
	GLuint zero_pbo;

	GLuint dfb;
	GLuint dfb_tbo;
	

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

glm::mat4 Model = glm::mat4(1.0f);
glm::vec3 eyePos = glm::vec3(0.0f, 0.0f, 100.0f);

Bound boundary;

glm::vec3 forward;
glm::mat4 View;
glm::mat4 Proj;

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

void initShaders()
{
	ShaderSource src1, src2, src3;
	src1 = ParseShader("src/shaders/first_pass.glsl");
	src2 = ParseShader("src/shaders/second_pass.glsl");
	src3 = ParseShader("src/shaders/third_pass.glsl");

	gl.first_pass = CreateShader(src1.vertex_src, src1.frag_src);
	gl.second_pass = CreateShader(src2.vertex_src, src2.frag_src);
	gl.third_pass = CreateShader(src3.vertex_src, src3.frag_src);


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
	GLCall(glBufferData(GL_TEXTURE_BUFFER, 4 * win.w * win.h * sizeof(glm::uvec4), NULL, GL_DYNAMIC_COPY));
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


}

void resetBuffer()
{
	GLCall(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl.zero_pbo));
	GLCall(glBindTexture(GL_TEXTURE_2D, gl.frag_count));
	GLCall(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, win.w, win.h, GL_RED_INTEGER, GL_UNSIGNED_INT, NULL));
	GLCall(glBindTexture(GL_TEXTURE_2D, 0));

	

	GLCall(glBindImageTexture(0, gl.frag_count, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI));
	GLCall(glBindImageTexture(1, gl.frag_base, 0, GL_FALSE, 0,GL_READ_WRITE, GL_R32UI));
	GLCall(glBindImageTexture(2, gl.dfb_tbo, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32UI));
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
	View = glm::lookAt(eyePos, forward, glm::vec3(0.0f, 1.0f, 0.0f));
	Proj = glm::perspective(65.0f, (float)win.w / (float)win.h, 1.0f, 1000.0f);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	GLCall(glDisable(GL_DEPTH_TEST));	

	resetBuffer();

	glm::mat4 MVP = Proj * View * Model;	

	GLCall(glUseProgram(gl.first_pass));

	GLCall(glBindVertexArray(gl.vao));
	GLCall(glBindBuffer(GL_ARRAY_BUFFER, gl.vbo));
	GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gl.vboindex));

	GLCall(glEnable(GL_BLEND));
	GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

	GLCall(glEnableVertexAttribArray(0));
	GLCall(glEnableVertexAttribArray(1));

	GLint matrixID = glGetUniformLocation(gl.first_pass, "MVP");
	if (matrixID == GL_INVALID_INDEX) {
		printf("Uniform %s not found in shader!", "MVP");
	}
	GLCall(glUniformMatrix4fv(matrixID, 1, GL_FALSE, &MVP[0][0]));
	
	GLCall(glDrawElements(GL_TRIANGLES, (GLsizei)planes_ind_count, GL_UNSIGNED_SHORT, 0));
	glMemoryBarrierEXT(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT_EXT);

	GLCall(glDisableVertexAttribArray(0));
	GLCall(glDisableVertexAttribArray(1));

	GLCall(glUseProgram(0));
	GLCall(glBindVertexArray(0));
	GLCall(glBindBuffer(GL_ARRAY_BUFFER, 0));
	GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));


	
	/********************************************************************************************/
	cuda_map();

	size_t buf_size = win.w * win.h * sizeof(GLuint);
	thrust::device_vector<GLuint> count(size_t(win.w*win.h));
	thrust::device_vector<GLuint> base(size_t(win.w*win.h));

	GLuint* cnt_ptr = thrust::raw_pointer_cast(&count[0]);
	GLuint* bse_ptr = thrust::raw_pointer_cast(&base[0]);
	
	cudaError_t error;

	error = cudaMemcpyFromArray(cnt_ptr, gl.countArray,0,0, buf_size, cudaMemcpyDeviceToDevice );
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
	
	cuda_unmap();
	/********************************************************************************************/
	GLCall(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl.zero_pbo));
	GLCall(glBindTexture(GL_TEXTURE_2D, gl.frag_count));
	GLCall(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, win.w, win.h, GL_RED_INTEGER, GL_UNSIGNED_INT, NULL));
	GLCall(glBindTexture(GL_TEXTURE_2D, 0));
	
	GLCall(glUseProgram(gl.second_pass));

	GLCall(glBindVertexArray(gl.vao));
	GLCall(glBindBuffer(GL_ARRAY_BUFFER, gl.vbo));
	GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gl.vboindex));

	GLCall(glEnable(GL_BLEND));
	GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

	GLCall(glEnableVertexAttribArray(0));
	GLCall(glEnableVertexAttribArray(1));

	matrixID = glGetUniformLocation(gl.first_pass, "MVP");
	if (matrixID == GL_INVALID_INDEX) {
		printf("Uniform %s not found in shader!", "MVP");
	}
	GLCall(glUniformMatrix4fv(matrixID, 1, GL_FALSE, &MVP[0][0]));

	GLCall(glDrawElements(GL_TRIANGLES, (GLsizei)planes_ind_count, GL_UNSIGNED_SHORT, 0));
	glMemoryBarrierEXT(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT_EXT);

	GLCall(glDisableVertexAttribArray(0));
	GLCall(glDisableVertexAttribArray(1));

	GLCall(glUseProgram(0));
	GLCall(glBindVertexArray(0));
	GLCall(glBindBuffer(GL_ARRAY_BUFFER, 0));
	GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));

	/************************************************************************************************/
			
	
	GLCall(glDisable(GL_BLEND));
	
	GLCall(glUseProgram(gl.third_pass));
	
	GLCall(glBindVertexArray(gl.quad_vao));
	GLCall(glBindBuffer(GL_ARRAY_BUFFER, gl.quad_vbo));
	GLCall(glEnableVertexAttribArray(0));
	
	GLCall(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4 ));
	glMemoryBarrierEXT(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT_EXT);
	
	GLCall(glDisableVertexAttribArray(0));
	GLCall(glBindBuffer(GL_ARRAY_BUFFER, 0));
	GLCall(glBindVertexArray(0));
	
	glutSwapBuffers();
}

void onResize(int w, int h)
{
	glViewport(0,0,w,h);
	glutPostRedisplay();
}

void onIdle()
{
	if (track.curx != track.lastx || track.cury != track.lasty)
	{
		glm::vec3 va = get_arcball_vector(track.lastx, track.lasty);
		glm::vec3 vb = get_arcball_vector(track.curx, track.cury);

		float angle = acos(std::min(1.0f, glm::dot(va, vb)));

		angle = angle * 0.05f;

		glm::vec3 axis_in_camera = glm::cross(va, vb);

		glm::mat3 inverse = glm::inverse(glm::mat3(View) * glm::mat3(Model));

		glm::vec3 axis_model = inverse * axis_in_camera;

		Model = glm::rotate(Model, glm::degrees(angle), axis_model);

		//Model = glm::rotate(Model, glm::degrees(angle), axis_in_camera);

		//View = glm::rotate(View, glm::degrees(angle), axis_in_camera);
		//View = glm::rotate(View, glm::degrees(angle), axis_model);

		track.lastx = track.curx;
		track.lasty = track.cury;

	}
	glutPostRedisplay();
	
}

void onMouse(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		arcball_on = true;
		track.lastx = track.curx = x;
		track.lasty = track.cury = y;
	}

	if (button == GLUT_LEFT_BUTTON && state == GLUT_UP)
	{
		arcball_on = false;
	}

	glutPostRedisplay();
}

void onMotion(int x, int y)
{
	if (arcball_on)
	{
		track.curx = x;
		track.cury = y;
	}
	glutPostRedisplay();
}

void onKeyboard(unsigned char key, int x, int y)
{
	switch (key){
	case '+':
		std::cout << " + was pressed" << std::endl;
		eyePos = eyePos - glm::vec3(0, 0, 15);
		break;
	case '-':
		std::cout << " - was pressed" << std::endl;
		eyePos = eyePos + glm::vec3(0, 0, 15);
		break;
	default:
		break;
	}
}

void initGL(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(win.w, win.h);
	glutCreateWindow("OIT- DFB");

	glutDisplayFunc(onDisplay);
	glutReshapeFunc(onResize);
	glutIdleFunc(onIdle);
	glutMouseFunc(onMouse);
	glutMotionFunc(onMotion);
	glutKeyboardFunc(onKeyboard);
	
	glClearColor(1.0, 1.0, 1.0, 1.0);

	GLenum gl_error;
	if ((gl_error = glewInit()) != GLEW_OK)
	{
		std::cout << "failed to init glew" << std::endl;
		std::cout << glewGetErrorString(gl_error) << std::endl;
	}
}

int main(int argc, char **argv)
{
	initGL(&argc, argv);

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
	Proj = glm::perspective(65.0f, (float)win.w / (float)win.h, 1.0f, 1000.0f);

	initVertexArrays();
	initShaders();
	

	glutMainLoop();
	
	cudaGraphicsUnregisterResource(gl.count_res);
	cudaGraphicsUnregisterResource(gl.base_res);

	std::cout << "quitting" << std::endl;
	std::cin.get();

	return 0;
}