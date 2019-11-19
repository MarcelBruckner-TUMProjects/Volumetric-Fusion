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

#ifndef PI
	const double PI = 3.14159265358979323846;
#endif
	const size_t IMU_FRAME_WIDTH = 1280;
	const size_t IMU_FRAME_HEIGHT = 720;
	
	class Camera {
	public:
		// Pose estimation camera stuff
		rs2_intrinsics intrinsics;
		cv::Matx33f cameraMatrices;
		std::vector<float> distCoeffs;
	};

    class Vertex3D {
    public:
        Eigen::Vector4d position;
        int u, v;

        Vertex3D(int x, int y, int z, int u = 0, int v = 0) : u(u), v(v){
            position << x, y, z, 1;
        }
    };

	class Data {
	public:

		std::string deviceName;

		texture tex;
		rs2::colorizer colorizer;
		rs2::frame  filteredColorFrames;
		rs2::frame  filteredDepthFrames;

		rs2::pointcloud pointclouds;
		rs2::frame colorizedDepthFrames;
		rs2::points points;
		std::vector<Vertex3D> vertices;

		Camera camera;
		vc::processing::Processing* processing;

		Data() {}

		// TODO maybe more mvvc
		void setIntrinsics(rs2_intrinsics intrinsics) {
			camera.intrinsics = intrinsics;
			camera.cameraMatrices = cv::Matx33f(
				intrinsics.fx, 0, intrinsics.ppx,
				0, intrinsics.fy, intrinsics.ppy,
				0, 0, 1
			);

			for (float c : intrinsics.coeffs) {
				camera.distCoeffs.push_back(c);
			}
		}
	};
}

#endif // !_DATA_HEADER_

