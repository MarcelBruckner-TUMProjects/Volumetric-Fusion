#pragma once
#include <common_utils/Common.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "device_functions.h"
#include "solo/optimization_algorithms/IndexProcessing.h"
#include "solo/linear_solvers/SystemProcessingCUDA.h"

#define COLUMN_INDEX_BLOCK_SIZE 256

namespace solo {

	template<typename IndexStorageType>
	__global__ void computeColumnIndexVectorPerConstraintAndElement(
		Type2Type<IndexStorageType>,
		DenseMatrixInterface<int, MemoryTypeCUDA> iIndexMatrix,
		const int* mapSparseToDenseIndexPtr,
		unsigned mapSparseToDenseIndexSize,
		unsigned residualDim,
		unsigned nResiduals,
		unsigned totalParamDim,
		unsigned indexVectorOffset,
		DenseMatrixInterface<int, MemoryTypeCUDA> iIndexVector
	) {
		int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId >= totalParamDim * nResiduals * residualDim) return;

		// threadIdx = paramId * nResiduals * residualDim + componentIdx * residualDim + residualId
		int residualId = threadId % residualDim;
		int componentIdx = (threadId / residualDim) % nResiduals;
		int paramId = threadId / (nResiduals * residualDim);

		computeColumnIndexVectorElement(
			Type2Type<IndexStorageType>(),
			iIndexMatrix, mapSparseToDenseIndexPtr, mapSparseToDenseIndexSize,
			residualId, componentIdx, paramId, residualDim, nResiduals, totalParamDim, indexVectorOffset,
			iIndexVector
		);
	}

	template<typename IndexStorageType>
	void computeColumnIndexVectorPerConstraintGPU(
		Type2Type<IndexStorageType>,
		MemoryContainer<int>& mapSparseToDenseIndex,
		DenseMatrix<int>& indexVector,
		unsigned indexVectorOffset,
		DenseMatrixWrapper<int>& indexMatrix,
		unsigned residualDim,
		unsigned nResiduals,
		unsigned totalParamDim
	) {
		if (!indexMatrix.getWrapper().isUpdatedDevice()) {
			indexMatrix.getWrapper().copyHostToDevice();
			indexMatrix.getWrapper().setUpdated(true, true);
		}
		if (!mapSparseToDenseIndex.isUpdatedDevice() && mapSparseToDenseIndex.getSize() > 0) {
			mapSparseToDenseIndex.copyHostToDevice();
			mapSparseToDenseIndex.setUpdated(true, true);
		}

		DenseMatrixInterface<int, MemoryTypeCUDA> iIndexMatrix{ indexMatrix };
		DenseMatrixInterface<int, MemoryTypeCUDA> iIndexVector{ indexVector };
		int* mapSparseToDenseIndexPtr = mapSparseToDenseIndex.getData(Type2Type<MemoryTypeCUDA>());
		unsigned mapSparseToDenseIndexSize = mapSparseToDenseIndex.getSize();

		unsigned totalColumnVectorSizePerConstraint = residualDim * nResiduals * totalParamDim;
		computeColumnIndexVectorPerConstraintAndElement<<< (totalColumnVectorSizePerConstraint + COLUMN_INDEX_BLOCK_SIZE - 1) / COLUMN_INDEX_BLOCK_SIZE, COLUMN_INDEX_BLOCK_SIZE >>>(
			Type2Type<IndexStorageType>(),
			iIndexMatrix, mapSparseToDenseIndexPtr, mapSparseToDenseIndexSize,
			residualDim, nResiduals, totalParamDim, indexVectorOffset,
			iIndexVector
		);
		CUDA_CHECK_ERROR();

		indexVector.getContainer().setUpdated(false, true);
	}

	/**
	 * Explicit instantiation.
	 */
	template void computeColumnIndexVectorPerConstraintGPU<RowWiseStorage>(
		Type2Type<RowWiseStorage>,
		MemoryContainer<int>& mapSparseToDenseIndex,
		DenseMatrix<int>& indexVector,
		unsigned indexVectorOffset,
		DenseMatrixWrapper<int>& indexMatrix,
		unsigned residualDim,
		unsigned nResiduals,
		unsigned totalParamDim
	);

	template void computeColumnIndexVectorPerConstraintGPU<ColumnWiseStorage>(
		Type2Type<ColumnWiseStorage>,
		MemoryContainer<int>& mapSparseToDenseIndex,
		DenseMatrix<int>& indexVector,
		unsigned indexVectorOffset,
		DenseMatrixWrapper<int>& indexMatrix,
		unsigned residualDim,
		unsigned nResiduals,
		unsigned totalParamDim
	);


	__global__ void computeMaxParamIndex(
		int* indexColumnVector,
		int size,
		int* maxElement
	) {
		int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId >= size) return;

		atomicMax(maxElement, indexColumnVector[threadId]);
	}

	int computeMaxIndexGPU(DenseMatrix<int>& indexVector) {
		runtime_assert(indexVector.getContainer().isUpdatedDevice(), "Device memory should be updated.");
		
		unsigned totalColumnVectorSize = indexVector.getSize();
		MemoryContainer<int> maxIdx;
		maxIdx.allocate(1, true, true);

		// Initialize max index to -1.
		maxIdx.getElement(0, Type2Type<MemoryTypeCPU>()) = -1;
		maxIdx.copyHostToDevice();

		computeMaxParamIndex<<< (totalColumnVectorSize + COLUMN_INDEX_BLOCK_SIZE - 1) / COLUMN_INDEX_BLOCK_SIZE, COLUMN_INDEX_BLOCK_SIZE >>>(
			indexVector.getData(Type2Type<MemoryTypeCUDA>()), totalColumnVectorSize, maxIdx.getData(Type2Type<MemoryTypeCUDA>())
		);
		CUDA_CHECK_ERROR();

		maxIdx.copyDeviceToHost();
		return maxIdx.getElement(0, Type2Type<MemoryTypeCPU>());
	}

} // namespace solo