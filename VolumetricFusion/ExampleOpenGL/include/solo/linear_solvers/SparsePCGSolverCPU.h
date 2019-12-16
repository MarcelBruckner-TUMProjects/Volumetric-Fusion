#pragma once
#include "LinearSolver.h"
#include "Solo/optimization_algorithms/Settings.h"

namespace solo {

	/**
	 * Solve method definitions.
	 */
	template<typename FloatType>
	FloatType solveImpl_SparsePCGSolverCPU(
		vector<std::pair<ResidualVector<FloatType>, JacobianMatrix<FloatType>>>& systemComponents,
		DenseMatrix<int>& jacobianInnerColumnIndices,
		DenseMatrix<int>& jacobianOuterRowStarts,
		unsigned nParameters,
		FloatType lambda,
		unsigned nMaxIterations,
		bool bOrderedParameters,
		bool bCheckTerminationCriteria,
		TerminationCriteria& terminationCriteria,
		int& nIterations,
		DenseMatrix<FloatType>& solution
	);

	template<typename FloatType> 
	inline FloatType solve_SparsePCGSolverCPU(
		vector<std::pair<ResidualVector<FloatType>, JacobianMatrix<FloatType>>>& systemComponents,
		DenseMatrix<int>& jacobianInnerColumnIndices,
		DenseMatrix<int>& jacobianOuterRowStarts,
		unsigned nParameters,
		FloatType lambda,
		unsigned nMaxIterations,
		bool bOrderedParameters,
		bool bCheckTerminationCriteria,
		TerminationCriteria& terminationCriteria,
		int& nIterations,
		DenseMatrix<FloatType>& solution
	) {
		runtime_assert(false, "Only float and double precision types are supported by SparsePCGSolverCPU.");
		return FloatType(-1);
	}

	template<>
	inline float solve_SparsePCGSolverCPU<float>(
		vector<std::pair<ResidualVector<float>, JacobianMatrix<float>>>& systemComponents,
		DenseMatrix<int>& jacobianInnerColumnIndices,
		DenseMatrix<int>& jacobianOuterRowStarts,
		unsigned nParameters,
		float lambda,
		unsigned nMaxIterations,
		bool bOrderedParameters,
		bool bCheckTerminationCriteria,
		TerminationCriteria& terminationCriteria,
		int& nIterations,
		DenseMatrix<float>& solution
	) {
		return solveImpl_SparsePCGSolverCPU(
			systemComponents, jacobianInnerColumnIndices, jacobianOuterRowStarts, nParameters, lambda, 
			nMaxIterations, bOrderedParameters, bCheckTerminationCriteria, terminationCriteria, nIterations, solution
		);
	}

	template<>
	inline double solve_SparsePCGSolverCPU<double>(
		vector<std::pair<ResidualVector<double>, JacobianMatrix<double>>>& systemComponents,
		DenseMatrix<int>& jacobianInnerColumnIndices,
		DenseMatrix<int>& jacobianOuterRowStarts,
		unsigned nParameters,
		double lambda,
		unsigned nMaxIterations,
		bool bOrderedParameters,
		bool bCheckTerminationCriteria,
		TerminationCriteria& terminationCriteria,
		int& nIterations,
		DenseMatrix<double>& solution
	) {
		return solveImpl_SparsePCGSolverCPU(
			systemComponents, jacobianInnerColumnIndices, jacobianOuterRowStarts, nParameters, lambda,
			nMaxIterations, bOrderedParameters, bCheckTerminationCriteria, terminationCriteria, nIterations, solution
		);
	}
	

	/**
	 * Solver implementation.
	 */
	template<typename FloatType>
	class SparsePCGSolverCPU : public LinearSolver<FloatType> {
	public:
		SparsePCGSolverCPU(const Settings& settings) : 
			m_nMaxIterations{ settings.linearSolverMaxNumIterations },
			m_bOrderedParameters{ settings.bOrderedParameters },
			m_bCheckTerminationCriteria{ settings.bEnableAlgorithmEarlyStop }
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

			return solve_SparsePCGSolverCPU(
				systemComponents, columnIndexVector, rowIndexVector, nParameters, lambda, m_nMaxIterations, 
				m_bOrderedParameters, m_bCheckTerminationCriteria, terminationCriteria, nIterations, solution
			);
		}

		LinearSolverType getType() const override {
			return LinearSolverType::SPARSE_PCG_CPU;
		}

	private:
		unsigned m_nMaxIterations{ 0 };
		bool m_bOrderedParameters{ false };
		bool m_bCheckTerminationCriteria{ false };
	};

} // namespace solo