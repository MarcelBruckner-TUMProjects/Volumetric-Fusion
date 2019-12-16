#pragma once

#include "solo/data_structures/DenseMatrix.h"
#include "solo/constraint_evaluation/Constraint.h"
#include "solo/optimization_algorithms/SolverProcessing.h"
#include "solo/optimization_algorithms/IndexProcessingShared.h"

namespace solo {

	/**
	 * Computes the column index vector part, belonging to particular constraints.
	 * We support both CPU and GPU execution.
	 */
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
	);

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
	);

	
	/**
	 * Computes the maximum index in the column index vector.
	 * We support both CPU and GPU execution.
	 */
	int computeMaxIndexCPU(DenseMatrix<int>& indexVector);
	int computeMaxIndexGPU(DenseMatrix<int>& indexVector);

} // namespace solo