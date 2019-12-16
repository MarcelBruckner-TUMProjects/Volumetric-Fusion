#ifndef IOMANAGEMENT_H
#define IOMANAGEMENT_H

#include <string>

#include <common_utils/data_structures/Array2.h>
#include <matrix_lib/matrix_structures/MatrixStructuresInclude.h>

#include "TriMesh.h"

using namespace matrix_lib;

namespace heatmap_fusion {

	namespace io_management {

		/**
		 * Mesh loading.
		 */
		TriMesh loadMeshPly(const std::string& meshPath);

		/**
		 * Image loading.
		 */
		Array2<uchar4> loadColorImage(const std::string& colorImagePath);
		Array2<float> loadDepthImage(const std::string& depthImagePath);

		/**
		 * Downsamples an intrinsics matrix.
		 * @param	intrinsicsMatrix	Original intrinsics matrix
		 * @param	downsampleFactor	The downsample factor
		 */
		inline Mat3f downsampleIntrinsicsMatrix(const Mat3f& intrinsicsMatrix, float downsampleFactor) {
			// We downsample the intrinsic matrix, if necessary.
			// x' = scalingFactor * (x + 0.5) - 0.5
			// y' = scalingFactor * (y + 0.5) - 0.5
			Mat3f scalingMatrix = Mat3f::identity();
			float inverseFactor = 1.f / downsampleFactor;
			scalingMatrix(0, 0) = inverseFactor;
			scalingMatrix(1, 1) = inverseFactor;
			scalingMatrix(0, 2) = inverseFactor * 0.5f - 0.5f;
			scalingMatrix(1, 2) = inverseFactor * 0.5f - 0.5f;
			Mat3f downsampledIntrinsicsMatrix = scalingMatrix * intrinsicsMatrix;

			return downsampledIntrinsicsMatrix;
		}

	} // namespace io_management
} // namespace heatmap_fusion

#endif //IOMANAGEMENT_H