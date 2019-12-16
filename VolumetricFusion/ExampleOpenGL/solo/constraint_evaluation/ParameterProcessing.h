#pragma once
#include "solo/data_structures/DenseMatrixWrapper.h"

namespace solo {

	/**
	 * Methods for updating the parameter vector from the increment vector.
	 * Both CPU and GPU implementations are present.
	 */
	template<typename FloatType>
	void updateParameterVectorCPU(
		DenseMatrix<FloatType>& increment,
		const vector<int>& indexMapping,
		DenseMatrixWrapper<FloatType>& paramVector
	);

	template<typename FloatType>
	void updateParameterVectorGPU(
		DenseMatrix<FloatType>& increment,
		const vector<int>& indexMapping,
		DenseMatrixWrapper<FloatType>& paramVector
	);

} // namespace solo