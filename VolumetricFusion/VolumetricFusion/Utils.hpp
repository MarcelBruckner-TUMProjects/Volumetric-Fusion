#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <sstream>

#ifndef _UTILS_HEADER
#define _UTILS_HEADER

namespace vc::utils {
	std::string toString(glm::vec2* vec) {
		std::stringstream ss;

		ss << vec->x << ", " << vec->y;

		return ss.str();
	}

	std::string toString(glm::vec3* vec) {
		std::stringstream ss;

		ss << vec->x << ", " << vec->y << ", " << vec->z;

		return ss.str();
	}

	std::string toString(glm::vec4* vec) {
		std::stringstream ss;

		ss << vec->x << ", " << vec->y << ", " << vec->z << ", " << vec->w;

		return ss.str();
	}
}

#endif