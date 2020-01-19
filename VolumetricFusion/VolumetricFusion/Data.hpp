// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#pragma once

#ifndef _DATA_HEADER_
#define _DATA_HEADER_

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

namespace vc::processing {
	class ChArUco;
}

namespace vc::data {
	class Data {
	public:

		unsigned long long frameId;
		std::string deviceName;

		//texture tex;
		rs2::frame  filteredColorFrames;
		rs2::frame  filteredDepthFrames;

		rs2::pointcloud pointclouds;
		rs2::frame colorizedDepthFrames;
		rs2::points points;
		
		vc::processing::ChArUco* processing;

	};
}
#endif // !_DATA_HEADER_
