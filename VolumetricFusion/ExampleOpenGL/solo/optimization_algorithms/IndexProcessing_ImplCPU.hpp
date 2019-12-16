#pragma once

#include "solo/optimization_algorithms/IndexProcessing.h"

namespace solo {

	template<typename IndexStorageType>
	void computeColumnIndexVectorPerConstraintCPU(
		Type2Type<IndexStorageType>,
		MemoryContainer<int>& mapSparseToDenseIndex,
		DenseMatrix<int>& indexVector,
		unsigned indexVectorOffset,
		DenseMatrixWrapper<int>& indexMatrix,
		unsigned residualDim,
		unsigned nResiduals,
		unsigned totalParamDim
	) {
		if (!indexMatrix.getWrapper().isUpdatedHost()) {
#			ifdef COMPILE_CUDA
			indexMatrix.getWrapper().copyDeviceToHost();
			indexMatrix.getWrapper().setUpdated(true, true);
#			endif
		}
		if (!mapSparseToDenseIndex.isUpdatedHost() && mapSparseToDenseIndex.getSize() > 0) {
#			ifdef COMPILE_CUDA
			mapSparseToDenseIndex.copyDeviceToHost();
			mapSparseToDenseIndex.setUpdated(true, true);
#			endif
		}

		DenseMatrixInterface<int, MemoryTypeCPU> iIndexMatrix{ indexMatrix };
		DenseMatrixInterface<int, MemoryTypeCPU> iIndexVector{ indexVector };
		int* mapSparseToDenseIndexPtr = mapSparseToDenseIndex.getData(Type2Type<MemoryTypeCPU>());
		unsigned mapSparseToDenseIndexSize = mapSparseToDenseIndex.getSize();

#		pragma omp for
		for (int paramId = 0; paramId < totalParamDim; paramId++) {
			for (int componentIdx = 0; componentIdx < nResiduals; ++componentIdx) {
				for (int residualId = 0; residualId < residualDim; residualId++) {
					computeColumnIndexVectorElement(
						Type2Type<IndexStorageType>(),
						iIndexMatrix, mapSparseToDenseIndexPtr, mapSparseToDenseIndexSize,
						residualId, componentIdx, paramId, residualDim, nResiduals, totalParamDim, indexVectorOffset,
						iIndexVector
					);
				}
			}
		}

		indexVector.getContainer().setUpdated(true, false);
	}

	/**
	 * Explicit instantiation.
	 */
	template void computeColumnIndexVectorPerConstraintCPU<RowWiseStorage>(
		Type2Type<RowWiseStorage>,
		MemoryContainer<int>& mapSparseToDenseIndex,
		DenseMatrix<int>& indexVector,
		unsigned indexVectorOffset,
		DenseMatrixWrapper<int>& indexMatrix,
		unsigned residualDim,
		unsigned nResiduals,
		unsigned totalParamDim
	);

	template void computeColumnIndexVectorPerConstraintCPU<ColumnWiseStorage>(
		Type2Type<ColumnWiseStorage>,
		MemoryContainer<int>& mapSparseToDenseIndex,
		DenseMatrix<int>& indexVector,
		unsigned indexVectorOffset,
		DenseMatrixWrapper<int>& indexMatrix,
		unsigned residualDim,
		unsigned nResiduals,
		unsigned totalParamDim
	);


	int computeMaxIndexCPU(DenseMatrix<int>& indexVector)	{
		runtime_assert(indexVector.getContainer().isUpdatedHost(), "Host memory should be updated.");
		int maxIdx = -1;
		unsigned nIndices = indexVector.getSize();

		DenseMatrixInterface<int, MemoryTypeCPU> iIndexVector{ indexVector };

		for (int i = 0; i < nIndices; i++) {
			int currentIdx = iIndexVector(i, 0);
			if (currentIdx > maxIdx) maxIdx = currentIdx;
		}

		return maxIdx;
	}

} // namespace solo