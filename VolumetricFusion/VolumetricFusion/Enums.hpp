#pragma once
#include <string>
namespace vc::enums{
	enum class RenderState {
		MULTI_POINTCLOUD,
		ONLY_COLOR,
		ONLY_DEPTH,
		COUNT
	};

	enum class CaptureState {
		STREAMING,
		RECORDING,
		PLAYING,
		COUNT
	};
}