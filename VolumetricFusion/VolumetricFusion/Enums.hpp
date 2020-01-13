#pragma once
#ifndef _ENUMS_HEADER
#define _ENUMS_HEADER

#include <string>
#include <map>

namespace vc::enums {
	enum class RenderState {
		ONLY_COLOR,
		ONLY_DEPTH,
		MULTI_POINTCLOUD,
		CALIBRATED_POINTCLOUD,
		VOXELGRID,
		COUNT
	};

	std::map<RenderState, const char*> renderStateToName = {
		{RenderState::ONLY_COLOR, "Only Color"},
		{RenderState::ONLY_DEPTH, "Only Depth"},
		{RenderState::MULTI_POINTCLOUD, "Multi Pointclouds"},
		{RenderState::CALIBRATED_POINTCLOUD, "Aligned Pointclouds"},
		{RenderState::VOXELGRID, "Voxelgrid"}
	};

	enum class CaptureState {
		STREAMING,
		RECORDING,
		PLAYING,
		COUNT
	};
}

#endif // !_ENUMS_HEADER
