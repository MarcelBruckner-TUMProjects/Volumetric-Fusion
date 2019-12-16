#pragma once
#include "LinearSolver.h"
#include "Solo/optimization_algorithms/Settings.h"

namespace solo {
	
	/**
	 * Solve method definitions.
	 */
	template<typename FloatType>
	FloatType solveImpl_DenseQRSolverCPU(
		vector<std::pair<ResidualVector<FloatType>, JacobianMatrix<FloatType>>>& systemComponents,
		DenseMatrix<int>& jacobianInnerColumnIndices,
		DenseMatrix<int>& jacobianOuterRowStarts,
		unsigned nParameters,
		FloatType lambda,
		bool bOrderedParameters,
		bool bCheckTerminationCriteria,
		TerminationCriteria& terminationCriteria,
		int& nIterations,
		DenseMatrix<FloatType>& solution
	);

	template<typename FloatType>
	inline FloatType solve_DenseQRSolverCPU(
		vector<std::pair<ResidualVector<FloatType>, JacobianMatrix<FloatType>>>& systemComponents,
		DenseMatrix<int>& jacobianInnerColumnIndices,
		DenseMatrix<int>& jacobianOuterRowStarts,
		unsigned nParameters,
		FloatType lambda,
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
	inline float solve_DenseQRSolverCPU<float>(
		vector<std::pair<ResidualVector<float>, JacobianMatrix<float>>>& systemComponents,
		DenseMatrix<int>& jacobianInnerColumnIndices,
		DenseMatrix<int>& jacobianOuterRowStarts,
		unsigned nParameters,
		float lambda,
		bool bOrderedParameters,
		bool bCheckTerminationCriteria,
		TerminationCriteria& terminationCriteria,
		int& nIterations,
		DenseMatrix<float>& solution
	) {
		return solveImpl_DenseQRSolverCPU(
			systemComponents, jacobianInnerColumnIndices, jacobianOuterRowStarts, nParameters, lambda, 
			bOrderedParameters, bCheckTerminationCriteria, terminationCriteria, nIterations, solution
		);
	}

	template<>
	inline double solve_DenseQRSolverCPU<double>(
		vector<std::pair<ResidualVector<double>, JacobianMatrix<double>>>& systemComponents,
		DenseMatrix<int>& jacobianInnerColumnIndices,
		DenseMatrix<int>& jacobianOuterRowStarts,
		unsigned nParameters,
		double lambda,
		bool bOrderedParameters,
		bool bCheckTerminationCriteria,
		TerminationCriteria& terminationCriteria,
		int& nIterations,
		DenseMatrix<double>& solution
	) {
		return solveImpl_DenseQRSolverCPU(
			systemComponents, jacobianInnerColumnIndices, jacobianOuterRowStarts, nParameters, lambda, 
			bOrderedParameters, bCheckTerminationCriteria, terminationCriteria, nIterations, solution
		);
	}


	/**
	 * Solver implementation.
	 */
	template<typename FloatType>
	class DenseQRSolverCPU : public LinearSolver<FloatType> {
	public:
		DenseQRSolverCPU(const Settings& settings) : 
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
			return solve_DenseQRSolverCPU(
				systemComponents, columnIndexVector, rowIndexVector, nParameters, lambda, 
				m_bOrderedParameters, m_bCheckTerminationCriteria, terminationCriteria, nIterations, solution
			);
		}

		LinearSolverType getType() const override {
			return LinearSolverType::DENSE_QR_CPU;
		}

		// TODO: Implement dense version.
		//void solve(
		//	const vector<std::pair<ResidualVector<FloatType>, JacobianMatrix<FloatType>>>& systemComponents,
		//	DenseMatrix<FloatType>& solution
		//) override {
		//	// TODO: Check if the memory is on GPU.

		//	// We have same (fixed) parameter indices, so all Jacobian matrices have the same dimension 
		//	// and correspond to the same parameter components, therefore it's faster to compute J^T J 
		//	// using dense matrices.
		//}

	private:
		bool m_bOrderedParameters{ false };
		bool m_bCheckTerminationCriteria{ false };
	};

} // namespace solo