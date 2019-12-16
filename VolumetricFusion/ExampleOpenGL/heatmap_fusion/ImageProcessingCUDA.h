#ifndef IMAGEPROCESSINGCUDA_H
#define IMAGEPROCESSINGCUDA_H

#include <matrix_lib/matrix_structures/MatrixStructuresInclude.h>

#include "Texture.h"

using namespace matrix_lib;

namespace heatmap_fusion {
	namespace image_proc_gpu {
		
		void unprojectDepthImage(
			Texture2D_32F& depthImage,
			Texture2D_RGBA32F& observationImage,
			int width, int height,
			const Mat3f& depthIntrinsics,
			const Mat4f& depthExtrinsics
		);

		void computeNormalImage(
			Texture2D_RGBA32F& observationImage,
			Texture2D_RGBA32F& normalImage,
			int width, int height,
			int kernelRadius, 
			float maxDistance
		);

		void filterInvalidPoints(
			Texture2D_RGBA32F& observationImage, 
			int width, int height,
			MemoryContainer<float4>& validPoints
		);

	} // namespace image_proc_gpu
} // namespace heatmap_fusion

#endif // !IMAGEPROCESSINGCUDA_H
