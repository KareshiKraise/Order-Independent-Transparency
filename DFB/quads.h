#ifndef _SHEET_
#define _SHEET_

#include <GLM/glm.hpp>

namespace colors {
	glm::vec4 red_a(1.0, 0.0, 0.0, 0.6);
	glm::vec4 grn_a(0.0, 1.0, 0.0, 0.6);
	glm::vec4 blue_a(0.0, 0.0, 1.0, 0.6);
}

struct vertex {
	glm::vec3 vert;
    glm::vec4 col;
};

struct vertex2 {
	glm::vec3 pos;
	glm::vec2 tex;
	glm::vec4 col;
};

struct Bound {
	glm::vec3 min;
	glm::vec3 max;
	glm::vec3 center;
};

vertex planes[] = {
	//Plane 1
	//0
	{ glm::vec3(-100, 100, -15.0), colors::grn_a },
	//1		   	  
{ glm::vec3(100, 100, -15.0), colors::grn_a },
//2         
{ glm::vec3(-100, -100, -15.0), colors::grn_a },
//3        
{ glm::vec3(100, -100, -15.0), colors::grn_a },

//Plane 2
//4
{ glm::vec3(-100 + 20, 100 + 20, -45.0), colors::red_a },
//5				  
{ glm::vec3(100 + 20, 100 + 20, -45.0), colors::red_a },
//6
{ glm::vec3(-100 + 20, -100 + 20, -45.0), colors::red_a },
//7
{ glm::vec3(100 + 20, -100 + 20, -45.0), colors::red_a },

//Plane 3
//8
{ glm::vec3(-100 - 20, 100 - 20, -75.0), colors::blue_a },
//9				  
{ glm::vec3(100 - 20, 100 - 20, -75.0), colors::blue_a },
//10
{ glm::vec3(-100 - 20, -100 - 20, -75.0), colors::blue_a },
//11
{ glm::vec3(100 - 20, -100 - 20, -75.0), colors::blue_a }

};


size_t planes_size = sizeof(planes);
size_t planes_element = sizeof(planes) / sizeof(vertex);

unsigned short planes_ind[] = { 0, 1, 2, 1, 3, 2,
4, 5, 6, 5, 7, 6,
8, 9, 10, 9, 11, 10 };

size_t planes_ind_size = sizeof(planes_ind);
size_t planes_ind_count = sizeof(planes_ind) / sizeof(unsigned short);

//returns medium point , meaning the center of a bounding box that encloses the geometry
Bound calc_bb()
{
	float xmin = planes[0].vert.x;
	float xmax = planes[0].vert.x;

	float ymin = planes[0].vert.y;
	float ymax = planes[0].vert.y;

	float zmin = planes[0].vert.z;
	float zmax = planes[0].vert.z;

	for (int i = 0; i < 12; i++) {

		if (planes[i].vert.x < xmin) {
			xmin = planes[i].vert.x;
		}

		if (planes[i].vert.x > xmax) {
			xmax = planes[i].vert.x;
		}


		if (planes[i].vert.y < ymin) {
			ymin = planes[i].vert.y;
		}
		if (planes[i].vert.y > ymax) {
			ymax = planes[i].vert.y;
		}



		if (planes[i].vert.z < zmin) {
			zmin = planes[i].vert.z;
		}

		if (planes[i].vert.z < zmax) {
			zmax = planes[i].vert.z;
		}

	}

	float centerX = xmax - ((xmax + fabs(xmin)) / 2);
	float centerY = ymax - ((ymax + fabs(ymin)) / 2);
	float centerZ = zmax - ((zmax + fabs(zmin)) / 2);

	glm::vec3 min = glm::vec3(xmin, ymin, zmin);
	glm::vec3 max = glm::vec3(xmax, ymax, zmax);
	glm::vec3 center = glm::vec3(centerX, centerY, centerZ);
	return{ min, max, center };


}

#endif