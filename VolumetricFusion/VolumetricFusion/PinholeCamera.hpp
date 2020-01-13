#pragma once
#ifndef _PINHOLE_CAMERA_HEADER
#define _PINHOLE_CAMERA_HEADER


#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <map>
#include <functional>
#include <vector>

#include "Processing.hpp"

#include "ceres/ceres.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace vc::camera {

	class PinholeCamera {
	public:
		// Pose estimation camera stuff
		rs2_intrinsics intrinsics;
		cv::Matx33f K;
		glm::mat3 world2cam;
		Eigen::Matrix3d cam2world;
		glm::mat3 cam2world_glm;
		std::vector<float> distCoeffs;

		float depthScale;

		PinholeCamera(rs2_intrinsics intrinsics, float depthScale = 0.0f) {
			this->depthScale = depthScale;
			this->intrinsics = intrinsics;

			K = cv::Matx33f(
				intrinsics.fx, 0, intrinsics.ppx,
				0, intrinsics.fy, intrinsics.ppy,
				0, 0, 1
			);

			world2cam = glm::mat3(
				1.0f / intrinsics.fx, 0, (-intrinsics.ppx) / intrinsics.fx,
				0, 1.0f / intrinsics.fy, (-intrinsics.ppy) / intrinsics.fy,
				0, 0, 1
			);

			cam2world_glm = glm::mat3(
				1.0f / intrinsics.fx, 0, (-intrinsics.ppx) / intrinsics.fx,
				0, 1.0f / intrinsics.fy, (-intrinsics.ppy) / intrinsics.fy,
				0, 0, 1
			);

			cam2world <<
				1.0f / intrinsics.fx, 0, (-intrinsics.ppx) / intrinsics.fx,
				0, 1.0f / intrinsics.fy, (-intrinsics.ppy) / intrinsics.fy,
				0, 0, 1
				;

			for (float c : intrinsics.coeffs) {
				distCoeffs.push_back(c);
			}
		}
	};

}

#endif // !_PINHOLE_CAMERA_HEADER