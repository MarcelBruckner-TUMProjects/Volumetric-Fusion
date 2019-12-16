#pragma once

namespace solo {

	/**
	 * Helper methods to compute loss either using CPU or GPU.
	 */
	template<typename FloatType>
	FloatType computeLossCPU(vector<ResidualVector<FloatType>>& residuals);

	template<typename FloatType>
	FloatType computeLossGPU(vector<ResidualVector<FloatType>>& residuals);


	/**
	 * Loss computation interface.
	 */
	class LossComputation {
	public:
		/**
		 * Computes the squared loss (sum of squares) of the given residual vectors.
		 */
		template<typename FloatType>
		static FloatType compute(vector<ResidualVector<FloatType>>& residuals) {
			if (residuals.empty()) {
				return FloatType(0);
			}

			if (residuals[0].getContainer().isUpdatedDevice()) {
#				ifdef COMPILE_CUDA
				return computeLossGPU(residuals);
#				else
				throw std::runtime_error("LossComputationGPU is not supported without a COMPILE_CUDA preprocessor flag.");
#				endif
			}
			else {
				return computeLossCPU(residuals);
			}
		}
	};

} // namespace solo