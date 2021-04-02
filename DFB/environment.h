#pragma once


#include <gl/glew.h>
#include <vector>
#include <string>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

class EnvironmentMap {
public:
	//cubemap texture
	GLuint cubeMap;


	//generate environment map
	//takes a vector of paths to each face in the following order:
	//right, left, top, bottom, back, front
	EnvironmentMap(std::vector<std::string> faceTextures) {
		//create cubemap
		glGenTextures(1, &cubeMap);
		glBindTexture(GL_TEXTURE_CUBE_MAP, cubeMap);

		//load environment map texture for each face
		//for (GLuint i = 0; i < 6; i++) {
		//	int width, height, n;
		//	unsigned char* image = NULL;
		//	image = stbi_load(faceTextures[i].c_str(), &width, &height, &n ,0);
		//	if (image == NULL) {
		//		std::cout << "EnvironmentMap could not load texture " << faceTextures[i] << std::endl;
		//	}
		//	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
		//	
		//}

		int width, height, n;
		unsigned char* image = NULL;
		image = stbi_load(faceTextures[0].c_str(), &width, &height, &n, 0);
		if (image == NULL) {
			std::cout << "EnvironmentMap could not load texture " << faceTextures[0] << std::endl;
		}
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
		
		image = stbi_load(faceTextures[1].c_str(), &width, &height, &n, 0);
		if (image == NULL) {
			std::cout << "EnvironmentMap could not load texture " << faceTextures[1] << std::endl;
		}
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);

		image = stbi_load(faceTextures[2].c_str(), &width, &height, &n, 0);
		if (image == NULL) {
			std::cout << "EnvironmentMap could not load texture " << faceTextures[2] << std::endl;
		}
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);

		image = stbi_load(faceTextures[3].c_str(), &width, &height, &n, 0);
		if (image == NULL) {
			std::cout << "EnvironmentMap could not load texture " << faceTextures[3] << std::endl;
		}
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
		
		image = stbi_load(faceTextures[4].c_str(), &width, &height, &n, 0);
		if (image == NULL) {
			std::cout << "EnvironmentMap could not load texture " << faceTextures[4] << std::endl;
		}
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);

		image = stbi_load(faceTextures[5].c_str(), &width, &height, &n, 0);
		if (image == NULL) {
			std::cout << "EnvironmentMap could not load texture " << faceTextures[5] << std::endl;
		}
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);

		//set cubemap settings
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
		glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
	}
	
};