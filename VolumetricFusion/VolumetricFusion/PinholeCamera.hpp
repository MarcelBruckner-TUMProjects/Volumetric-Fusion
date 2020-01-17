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
	protected:
		PinholeCamera() {}

	public:
		// Pose estimation camera stuff
		cv::Matx33f K;
		Eigen::Matrix3d world2cam;
		glm::mat3 world2cam_glm;
		Eigen::Matrix3d cam2world;
		glm::mat3 cam2world_glm;
		std::vector<float> distCoeffs;

		float depthScale;
				
		PinholeCamera(rs2_intrinsics intrinsics, float depthScale = 0.0f) {
			this->depthScale = depthScale;

			K = cv::Matx33f(
				intrinsics.fx, 0, intrinsics.ppx,
				0, intrinsics.fy, intrinsics.ppy,
				0, 0, 1
			);

			world2cam <<
				intrinsics.fx, 0, intrinsics.ppx,
				0, intrinsics.fy, intrinsics.ppy,
				0, 0, 1
				;

			world2cam_glm = glm::transpose(glm::mat3(
				intrinsics.fx, 0, intrinsics.ppx,
				0, intrinsics.fy, intrinsics.ppy,
				0, 0, 1
				));


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

	class MockPinholeCamera : public PinholeCamera {
	public:
		MockPinholeCamera() : PinholeCamera() {

			K = cv::Matx33f(0.0f);

			world2cam = Eigen::Matrix3d::Identity();

			cam2world_glm = glm::mat3(1.0f);

			cam2world = Eigen::Matrix3d::Identity();

			for (int i = 0; i < 5; i++) {
				distCoeffs.push_back(0.0f);
			}

			depthScale = -1;
		}
	};
}

#endif // !_PINHOLE_CAMERA_HEADER