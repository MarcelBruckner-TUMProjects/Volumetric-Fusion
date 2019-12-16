#pragma once
#include "LossComputation.h"
#include "SystemProcessingCPU.h"

namespace solo {

	template<typename FloatType>
	FloatType computeLossCPU(vector<ResidualVector<FloatType>>& residuals) {
		const unsigned nConstraints = residuals.size();

		// Copy residuals to host memory, if required.
		system_proc::updateResidualMemory(residuals);


		// We compute the total required memory size for residual vector.
		unsigned totalResidualSize = 0;
		for (int i = 0; i < nConstraints; ++i) {
			auto& residualVector = residuals[i];
			totalResidualSize += residualVector.getSize();
		}

		// Copy all residual vectors to a one dense vector.
		DenseMatrix<FloatType> residualValues;
		system_proc::prepareResidualVector(residuals, residualValues, totalResidualSize);

		// We map a dense vector around residual values, to compute the squared loss.
		Eigen::Map<Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic>> residualMap(residualValues.getData(Type2Type<MemoryTypeCPU>()), residualValues.rows(), residualValues.cols());
		FloatType squaredLoss = residualMap.squaredNorm();

		return squaredLoss;
	}


	/**
	 * Explicit instantiation.
	 */
	template float computeLossCPU<float>(vector<ResidualVector<float>>& residuals);
	template double computeLossCPU<double>(vector<ResidualVector<double>>& residuals);

} // namespace solo