#pragma once
#include <string>
namespace vc::enums {
	enum class RenderState {
		MULTI_POINTCLOUD,
		CALIBRATED_POINTCLOUD,
		VOXELGRID,
		ONLY_COLOR,
		ONLY_DEPTH,
		ONLY_CHARACTERISTIC_POINTS,
		COUNT
	};

	enum class CaptureState {
		STREAMING,
		RECORDING,
		PLAYING,
		COUNT
	};
}