#pragma once
#include "Solo/utils/Common.h"
#include "LinearSolver.h"
#include "Solo/optimization_algorithms/Settings.h"

namespace solo {

	/**
	 * Solve method definitions.
	 */
	template<typename FloatType>
	FloatType solveImpl_SparsePCGSolverCompleteGPU(
		vector<std::pair<ResidualVector<FloatType>, JacobianMatrix<FloatType>>>& systemComponents,
		DenseMatrix<int>& columnIndexVector,
		DenseMatrix<int>& rowIndexVector,
		unsigned nParameters,
		FloatType lambda,
		unsigned nMaxIterations,
		bool bOrderedParameters,
		const TerminationParameters& terminationParameters,
		TerminationCriteria& terminationCriteria,
		int& nIterations,
		DenseMatrix<FloatType>& solution
	);

	template<typename FloatType>
	FloatType solve_SparsePCGSolverCompleteGPU(
		vector<std::pair<ResidualVector<FloatType>, JacobianMatrix<FloatType>>>& systemComponents,
		DenseMatrix<int>& columnIndexVector,
		DenseMatrix<int>& rowIndexVector,
		unsigned nParameters,
		FloatType lambda,
		unsigned nMaxIterations,
		bool bOrderedParameters,
		const TerminationParameters& terminationParameters,
		TerminationCriteria& terminationCriteria,
		int& nIterations,
		DenseMatrix<FloatType>& solution
	) {
		runtime_assert(false, "Only float and double precision types are supported by SparsePCGSolverCPU.");
		return FloatType(-1);
	}

	template<>
	inline float solve_SparsePCGSolverCompleteGPU<float>(
		vector<std::pair<ResidualVector<float>, JacobianMatrix<float>>>& systemComponents,
		DenseMatrix<int>& columnIndexVector,
		DenseMatrix<int>& rowIndexVector,
		unsigned nParameters,
		float lambda,
		unsigned nMaxIterations,
		bool bOrderedParameters,
		const TerminationParameters& terminationParameters,
		TerminationCriteria& terminationCriteria,
		int& nIterations,
		DenseMatrix<float>& solution
	) {
		return solveImpl_SparsePCGSolverCompleteGPU(
			systemComponents, columnIndexVector, rowIndexVector, nParameters, lambda, nMaxIterations, 
			bOrderedParameters, terminationParameters, terminationCriteria, nIterations, solution
		);
	}

	template<>
	inline double solve_SparsePCGSolverCompleteGPU<double>(
		vector<std::pair<ResidualVector<double>, JacobianMatrix<double>>>& systemComponents,
		DenseMatrix<int>& columnIndexVector,
		DenseMatrix<int>& rowIndexVector,
		unsigned nParameters,
		double lambda,
		unsigned nMaxIterations,
		bool bOrderedParameters,
		const TerminationParameters& terminationParameters,
		TerminationCriteria& terminationCriteria,
		int& nIterations,
		DenseMatrix<double>& solution
	) {
		return solveImpl_SparsePCGSolverCompleteGPU(
			systemComponents, columnIndexVector, rowIndexVector, nParameters, lambda, nMaxIterations, 
			bOrderedParameters, terminationParameters, terminationCriteria, nIterations, solution
		);
	}


	/**
	 * Solver implementation.
	 */
	template<typename FloatType>
	class SparsePCGSolverCompleteGPU : public LinearSolver<FloatType> {
	public:
		SparsePCGSolverCompleteGPU(const Settings& settings) :
			m_nMaxIterations{ settings.linearSolverMaxNumIterations },
			m_bOrderedParameters{ settings.bOrderedParameters },
			m_terminationParameters{ settings.gradientTolerance, settings.bEnableAlgorithmEarlyStop, settings.qTolerance, settings.rTolerance, settings.bUseQStoppingCriteria }
		{ }

		/**
		 * Interface implementation.
		 */
		FloatType solve(
			vector<std::pair<ResidualVector<FloatType>, JacobianMatrix<FloatType>>>& systemComponents,
			DenseMatrix<int>& columnIndexVector,
			DenseMatrix<int>& rowIndexVector,
			unsigned nParameters, 
			FloatType lambda,
			TerminationCriteria& terminationCriteria,
			int& nIterations,
			DenseMatrix<FloatType>& solution
		) override {
			#ifdef COMPILE_CUDA
			return solve_SparsePCGSolverCompleteGPU(
				systemComponents, columnIndexVector, rowIndexVector, nParameters, lambda, m_nMaxIterations, 
				m_bOrderedParameters, m_terminationParameters, terminationCriteria, nIterations, solution
			);
			#else
			throw std::runtime_error("SparsePCGSolverCompleteGPU is not supported without a COMPILE_CUDA preprocessor flag.");
			#endif
		}
		
		LinearSolverType getType() const override {
			return LinearSolverType::SPARSE_PCG_COMPLETE_GPU;
		}

	private:
		unsigned m_nMaxIterations{ 0 };
		bool m_bOrderedParameters{ false };
		TerminationParameters m_terminationParameters;
	};

} // namespace solo