#pragma once
#include <common_utils/Common.h>

namespace solo {

	/**
	 * Flag for the type of the index vector computation.
	 */
	struct RowWiseStorage {};
	struct ColumnWiseStorage {};


	/**
	 * Returns the local column vector index, depending on the storage type.
	 */
	CPU_AND_GPU inline unsigned computeLocalColumnIdx(
		Type2Type<RowWiseStorage>,
		unsigned residualId,
		unsigned componentIdx,
		unsigned paramId,
		unsigned residualDim,
		unsigned nResiduals,
		unsigned totalParamDim
	) {
		return residualId * nResiduals * totalParamDim + componentIdx * totalParamDim + paramId;
	}

	CPU_AND_GPU inline unsigned computeLocalColumnIdx(
		Type2Type<ColumnWiseStorage>,
		unsigned residualId,
		unsigned componentIdx,
		unsigned paramId,
		unsigned residualDim,
		unsigned nResiduals,
		unsigned totalParamDim
	) {
		return paramId * nResiduals * residualDim + residualId * nResiduals + componentIdx;
	}

	/**
	 * Computes an element of column index vector.
	 */
	template<typename IndexStorageType, typename MemoryStorageType>
	CPU_AND_GPU inline void computeColumnIndexVectorElement(
		Type2Type<IndexStorageType>,
		const DenseMatrixInterface<int, MemoryStorageType>& iIndexMatrix,
		const int* mapSparseToDenseIndexPtr,
		unsigned mapSparseToDenseIndexSize,
		unsigned residualId,
		unsigned componentIdx,
		unsigned paramId,
		unsigned residualDim,
		unsigned nResiduals,
		unsigned totalParamDim,
		unsigned indexVectorOffset,
		DenseMatrixInterface<int, MemoryStorageType>& iIndexVector
	) {
		const unsigned localIdx = computeLocalColumnIdx(Type2Type<IndexStorageType>(), residualId, componentIdx, paramId, residualDim, nResiduals, totalParamDim);
		if (mapSparseToDenseIndexSize > 0) {
			// We need to put the index in the correct range.
			int convertedIdx = mapSparseToDenseIndexPtr[iIndexMatrix(componentIdx, paramId)];
			iIndexVector(indexVectorOffset + localIdx, 0) = convertedIdx;
		}
		else {
			// We keep the original index.
			iIndexVector(indexVectorOffset + localIdx, 0) = iIndexMatrix(componentIdx, paramId);
		}
	}


} // namespace solo