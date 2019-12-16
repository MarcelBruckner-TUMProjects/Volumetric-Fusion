#pragma once
#include <vector>
#include <common_utils/Common.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "device_functions.h"
#include "ParameterProcessing.h"
#include "solo/data_structures/DenseMatrixInterface.h"

#define PARAMETER_UPDATE_BLOCK_SIZE 256

using std::vector;

namespace solo {

	template<typename FloatType>
	__global__ void updateEachParameter(
		DenseMatrixInterface<FloatType, MemoryTypeCUDA> increment,
		int* indexMapping,
		unsigned indexMappingSize,
		DenseMatrixInterface<FloatType, MemoryTypeCUDA> paramVector
	) {
		int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId >= increment.getSize()) return;

		if (indexMappingSize > 0) {
			int targetIdx = indexMapping[threadId];
			paramVector(targetIdx, 0) = paramVector(targetIdx, 0) + increment(threadId, 0);
		}
		else {
			paramVector(threadId, 0) = paramVector(threadId, 0) + increment(threadId, 0);
		}
	}


	template<typename FloatType>
	void updateParameterVectorGPU(
		DenseMatrix<FloatType>& increment,
		const vector<int>& indexMapping,
		DenseMatrixWrapper<FloatType>& paramVector
	) {
		const unsigned nIndices = increment.rows();
		runtime_assert(increment.getContainer().isUpdatedDevice(), "Increment should be updated in host memory.");
		runtime_assert(paramVector.getWrapper().isUpdatedDevice(), "Parameter vector should be updated in host memory.");

		DenseMatrixInterface<FloatType, MemoryTypeCUDA> iParamVector{ paramVector };
		DenseMatrixInterface<FloatType, MemoryTypeCUDA> iIncrement{ increment };

		// We copy the index mapping to device memory, if it's non-empty.
		MemoryContainer<int> indexMappingContainer;
		unsigned indexMappingSize = indexMapping.size();
		if (indexMappingSize > 0) {
			indexMappingContainer.allocate(indexMappingSize, true, true);
			for (int i = 0; i < indexMappingSize; i++) {
				indexMappingContainer.getElement(i, Type2Type<MemoryTypeCPU>()) = indexMapping[i];
			}
			indexMappingContainer.copyHostToDevice();
		}

		// Execute parameter update on the GPU.
		unsigned incrementSize = iIncrement.getSize();
		updateEachParameter<<< (incrementSize + PARAMETER_UPDATE_BLOCK_SIZE - 1) / PARAMETER_UPDATE_BLOCK_SIZE, PARAMETER_UPDATE_BLOCK_SIZE >>>(
			iIncrement, indexMappingContainer.getData(Type2Type<MemoryTypeCUDA>()), indexMappingSize, iParamVector
		);
		CUDA_CHECK_ERROR();

		paramVector.getWrapper().setUpdated(false, true);
	}


	/**
	 * Explicit instantiation.
	 */
	template void updateParameterVectorGPU<float>(
		DenseMatrix<float>& increment,
		const vector<int>& indexMapping,
		DenseMatrixWrapper<float>& paramVector
	);

	template void updateParameterVectorGPU<double>(
		DenseMatrix<double>& increment,
		const vector<int>& indexMapping,
		DenseMatrixWrapper<double>& paramVector
	);

} // namespace solo