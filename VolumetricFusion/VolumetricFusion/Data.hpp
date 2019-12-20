// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#pragma once

#ifndef _DATA_HEADER_
#define _DATA_HEADER_


#define NOMINMAX
#if KEVIN_MACOS
#pragma message("Included on Mac OS")
#endif

#include <librealsense2/rs.hpp>
//
//#if APPLE
//#include <glut.h>
//#else
//#include <windows.h>
//#include <GL/gl.h>
//#endif
#include <GLFW/glfw3.h>

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <map>
#include <functional>

#include "Processing.hpp"

namespace vc::processing {
	class Processing;
}

namespace vc::data {
	
	class Camera {
	public:
		// Pose estimation camera stuff
		rs2_intrinsics intrinsics;
		cv::Matx33f K;
		std::vector<float> distCoeffs;

		// TODO maybe more mvvc
		Camera(rs2_intrinsics intrinsics) {
			intrinsics = intrinsics;
			K = cv::Matx33f(
				intrinsics.fx, 0, intrinsics.ppx,
				0, intrinsics.fy, intrinsics.ppy,
				0, 0, 1
			);

			for (float c : intrinsics.coeffs) {
				distCoeffs.push_back(c);
			}
		}
	};

	class Data {
	public:

		std::string deviceName;

		//texture tex;
		rs2::colorizer colorizer;
		rs2::frame  filteredColorFrames;
		rs2::frame  filteredDepthFrames;

		rs2::pointcloud pointclouds;
		rs2::frame colorizedDepthFrames;
		rs2::points points;
		
		vc::processing::Processing* processing;

		Data() {}

	};
}
#endif // !_DATA_HEADER_
