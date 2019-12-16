#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "DenseQRSolverCPU.h"
#include "SystemProcessingCPU.h"

// Important: This file can only be included in exactly one .cpp file, otherwise the methods would get compiled 
// multiple times.

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
	) {
		// We don't use iterative solver.
		nIterations = 0;

		// Check whether any constraints are even used in the optimization problem.
		if (systemComponents.size() == 0) return -1.f;

		// Copy index vectors to CPU memory, if necessary.
#		ifdef COMPILE_CUDA
		if (!jacobianInnerColumnIndices.getContainer().isUpdatedHost()) {
			jacobianInnerColumnIndices.getContainer().copyDeviceToHost();
			jacobianInnerColumnIndices.getContainer().setUpdated(true, true);
		}

		if (!jacobianOuterRowStarts.getContainer().isUpdatedHost()) {
			jacobianOuterRowStarts.getContainer().copyDeviceToHost();
			jacobianOuterRowStarts.getContainer().setUpdated(true, true);
		}
#		endif

		// Prepare memory for sparse matrix operations. 
		DenseMatrix<FloatType> residualValues;
		DenseMatrix<FloatType> jacobianValues;
		Eigen::SparseMatrix<FloatType, Eigen::RowMajor, int> sparseJacobianMatrix(jacobianOuterRowStarts.getSize() - 1, nParameters);
		if (bOrderedParameters) {	
			system_proc::prepareSparseSystemEigenOrdered(systemComponents, residualValues, jacobianValues);

			// We construct a sparse Jacobian matrix using raw memory. The Jacobian is of dimensions
			// nTotalResiduals x nParameters.
			sparseJacobianMatrix = Eigen::Map<Eigen::SparseMatrix<FloatType, Eigen::RowMajor, int>>(
				residualValues.getSize(), nParameters, jacobianValues.getSize(),
				jacobianOuterRowStarts.getData(Type2Type<MemoryTypeCPU>()), jacobianInnerColumnIndices.getData(Type2Type<MemoryTypeCPU>()), jacobianValues.getData(Type2Type<MemoryTypeCPU>())
			);
		}
		else {
			system_proc::prepareSparseSystemEigenNonOrdered(systemComponents, residualValues, jacobianInnerColumnIndices, sparseJacobianMatrix);
		}

		// We map a dense vector around residual values.
		Eigen::Map<Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic>> residualMap(residualValues.getData(Type2Type<MemoryTypeCPU>()), residualValues.rows(), residualValues.cols());
		FloatType squaredLoss = residualMap.squaredNorm();

		// Compute transpose matrix J^T.
		auto sparseJacobianMatrixTranspose = sparseJacobianMatrix.transpose();

		// Compute the sparse matrix A = J^T J + lambda I and dense vector b = -J^T r.
		Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic> A = sparseJacobianMatrixTranspose * sparseJacobianMatrix;;
		Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic> b = -sparseJacobianMatrixTranspose * residualMap;
		
		// We add Levenberg-Marquardt factor if necessary.
		if (std::abs(lambda) > 0) {
			A.diagonal().array() += lambda;
		}

		// Solve the system Ax = b using Householder QR decomposition.
		Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic> x = A.householderQr().solve(b);

		// Copy the solution vector.
		runtime_assert(x.rows() == int(nParameters), "Solution of wrong row dimension was returned.");
		runtime_assert(x.cols() == 1, "Solution of wrong column dimension was returned.");

		solution.allocate(nParameters, 1);
		memcpy(solution.getData(Type2Type<MemoryTypeCPU>()), x.data(), nParameters * sizeof(FloatType));

		// Set the correct update flags.
		solution.getContainer().setUpdated(true, false);

		// Check for termination criteria, if necessary.
		// We opt for early stop if ||Jx|| < eps (1 + ||r||)
		if (bCheckTerminationCriteria) {
			FloatType eps = std::numeric_limits<FloatType>::epsilon();
			FloatType JxNorm = (sparseJacobianMatrix * x).norm();
			if (JxNorm < eps * (1 + std::sqrt(squaredLoss)))
				terminationCriteria.bStopEarly = true;
			else
				terminationCriteria.bStopEarly = false;
		}

		return squaredLoss;
	}


	/**
	 * Explicit declarations of methods that we support, in order to compile them.
	 */
	template float solveImpl_DenseQRSolverCPU<float>(
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
	);

	template double solveImpl_DenseQRSolverCPU<double>(
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
	);

} // namespace solo