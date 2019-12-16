#pragma once
#include "LossComputation.h"
#include "SystemProcessingCUDA.h"

namespace solo {

	template<typename FloatType>
	FloatType computeLossGPU(vector<ResidualVector<FloatType>>& residuals) {
		const unsigned nConstraints = residuals.size();

		// Copy residuals to device memory, if required.
		for (int i = 0; i < nConstraints; ++i) {
			auto& residualVector = residuals[i];
			auto& residualContainer = residualVector.getContainer();
			if (residualContainer.isUpdatedHost() && !residualContainer.isAllocatedDevice()) {
				residualContainer.copyHostToDevice();
				residualContainer.setUpdated(true, true);
			}
		}

		// Create cuBLAS handle and bind it to a common stream.
		cublasHandle_t cublasHandle = NULL;
		CUBLAS_SAFE_CALL(cublasCreate(&cublasHandle));

		// Set cuBLAS pointer modes to host (we never use constants, stored on the device).
		CUBLAS_SAFE_CALL(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST));

		// We compute the squared loss of residuals.
		FloatType squaredLoss = FloatType(0);
		for (int i = 0; i < nConstraints; ++i) {
			auto& residualVector = residuals[i];
			squaredLoss += system_proc::computeSquaredLoss(cublasHandle, residualVector.getContainer());
		}
		runtime_assert(std::isfinite(squaredLoss), "Squared loss (r^T r) is not finite.");

		return squaredLoss;
	}


	/**
	 * Explicit instantiation.
	 */
	template float computeLossGPU<float>(vector<ResidualVector<float>>& residuals);
	template double computeLossGPU<double>(vector<ResidualVector<double>>& residuals);

} // namespace solo