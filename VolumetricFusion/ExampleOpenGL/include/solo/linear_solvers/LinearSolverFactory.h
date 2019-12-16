#pragma once
#include "LinearSolver.h"
#include "DenseQRSolverCPU.h"
#include "SparsePCGSolverCPU.h"
#include "SparsePCGSolverCompleteGPU.h"
#include "SparsePCGSolverSequentialGPU.h"
#include "SparsePCGSolverAtomicGPU.h"

namespace solo {

	/**
	 * Factory for different linear solvers.
	 */
	template<typename FloatType>
	class LinearSolverFactory {
	public:
		static std::unique_ptr<LinearSolver<FloatType>> get(LinearSolverType linearSolverType, const Settings& settings) {
			switch (linearSolverType) {
			case LinearSolverType::DENSE_QR_CPU:
				return std::make_unique<DenseQRSolverCPU<FloatType>>(settings);
			case LinearSolverType::SPARSE_PCG_CPU:
				return std::make_unique<SparsePCGSolverCPU<FloatType>>(settings);
			case LinearSolverType::SPARSE_PCG_COMPLETE_GPU:
				return std::make_unique<SparsePCGSolverCompleteGPU<FloatType>>(settings);
			case LinearSolverType::SPARSE_PCG_SEQUENTIAL_GPU:
				return std::make_unique<SparsePCGSolverSequentialGPU<FloatType>>(settings);
			case LinearSolverType::SPARSE_PCG_ATOMIC_GPU:
				return std::make_unique<SparsePCGSolverAtomicGPU<FloatType>>(settings);
			default:
				runtime_assert(false, "The linear solver type is not supported by the linear solver factory!");
				break;
			}
			return nullptr;
		}
	};

} // namespace solo