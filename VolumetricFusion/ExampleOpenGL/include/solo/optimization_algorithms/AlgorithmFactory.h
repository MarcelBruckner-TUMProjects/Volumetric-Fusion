#pragma once
#include "SolverInterface.h"
#include "Algorithm.h"
#include "GaussNewtonAlgorithm.h"

namespace solo {

	/**
	 * Factory for different optimization algorithm.
	 */
	template<typename FloatType, typename ParamStorageType, typename ...Constraints>
	class AlgorithmFactory {
	public:
		static std::unique_ptr<Algorithm<FloatType, ParamStorageType, Constraints...>> get(AlgorithmType algorithmType, const Settings& settings, LinearSolver<FloatType>& linearSolver) {
			switch (algorithmType) {
			case AlgorithmType::GAUSS_NEWTON:
				return std::make_unique<GaussNewtonAlgorithm<FloatType, ParamStorageType, Constraints...>>(settings, linearSolver);
			default:
				runtime_assert(false, "The algorithm type is not supported by the algorithm factory!");
				break;
			}
			return nullptr;
		}
	};

} // namespace solo