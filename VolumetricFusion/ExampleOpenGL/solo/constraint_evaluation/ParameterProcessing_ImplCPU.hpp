#pragma once
#include <vector>

#include "ParameterProcessing.h"
#include "solo/data_structures/DenseMatrixInterface.h"

using std::vector;

namespace solo {

	template<typename FloatType>
	void updateParameterVectorCPU(
		DenseMatrix<FloatType>& increment,
		const vector<int>& indexMapping,
		DenseMatrixWrapper<FloatType>& paramVector
	) {
		const unsigned nIndices = increment.rows();
		runtime_assert(increment.getContainer().isUpdatedHost(), "Increment should be updated in host memory.");
		runtime_assert(paramVector.getWrapper().isUpdatedHost(), "Parameter vector should be updated in host memory.");
	
		DenseMatrixInterface<FloatType, MemoryTypeCPU> iParamVector{ paramVector };
		DenseMatrixInterface<FloatType, MemoryTypeCPU> iIncrement{ increment };

		unsigned indexMappingSize = indexMapping.size();

#		pragma omp parallel for
		for (int i = 0; i < nIndices; ++i) {
			if (indexMappingSize > 0) {
				int targetIdx = indexMapping[i];
				iParamVector(targetIdx, 0) = iParamVector(targetIdx, 0) + iIncrement(i, 0);
			}
			else {
				iParamVector(i, 0) = iParamVector(i, 0) + iIncrement(i, 0);
			}
			
		}

		paramVector.getWrapper().setUpdated(true, false);
	}

	/**
	 * Explicit instantiation.
	 */
	template void updateParameterVectorCPU<float>(
		DenseMatrix<float>& increment,
		const vector<int>& indexMapping,
		DenseMatrixWrapper<float>& paramVector
	);

	template void updateParameterVectorCPU<double>(
		DenseMatrix<double>& increment,
		const vector<int>& indexMapping,
		DenseMatrixWrapper<double>& paramVector
	);

} // namespace solo