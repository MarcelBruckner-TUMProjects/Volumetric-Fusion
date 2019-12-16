#pragma once
#include <common_utils/timing/TimerGPU.h>

#include "SparsePCGSolverSequentialGPU.h"
#include "SystemProcessingCUDA.h"

// Important: This file can only be included in exactly one .cpp file, otherwise the methods would get compiled 
// multiple times.

namespace solo {

	/**
	 * General implemention of SparsePCGSolverAtomicGPU.
	 */
	template<typename FloatType>
	FloatType solveImpl_SparsePCGSolverAtomicGPU(
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
		// Check whether any constraints are even used in the optimization problem.
		if (systemComponents.size() == 0) return -1.f;

		TIME_GPU_START(PCGSolver_Initialize);

		const unsigned nConstraints = systemComponents.size();

		// Copy index vectors, residual vectors and Jacobian matrices to GPU, if necessary.
		// Check that indices are stored on the GPU (otherwise update).
		MemoryContainer<int>& columnIndices = columnIndexVector.getContainer();
		MemoryContainer<int>& constraintStarts = rowIndexVector.getContainer();

		if (columnIndices.isUpdatedHost() && !columnIndices.isAllocatedDevice()) {
			columnIndices.copyHostToDevice();
			columnIndices.setUpdated(true, true);
		}

		if (constraintStarts.isUpdatedHost() && !constraintStarts.isAllocatedDevice()) {
			constraintStarts.copyHostToDevice();
			constraintStarts.setUpdated(true, true);
		}

		for (int i = 0; i < nConstraints; ++i) {
			auto& residualVector = systemComponents[i].first;
			auto& residualContainer = residualVector.getContainer();
			if (residualContainer.isUpdatedHost() && !residualContainer.isAllocatedDevice()) {
				residualContainer.copyHostToDevice();
				residualContainer.setUpdated(true, true);
			}

			auto& jacobianMatrix = systemComponents[i].second;
			auto& jacobianContainer = jacobianMatrix.getContainer();
			if (jacobianContainer.isUpdatedHost() && !jacobianContainer.isAllocatedDevice()) {
				jacobianContainer.copyHostToDevice();
				jacobianContainer.setUpdated(true, true);
			}
		}

		// Create cuBLAS/cuSPARSE handles and bind them to a common stream.
		cublasHandle_t cublasHandle = NULL;
		cusparseHandle_t cusparseHandle = NULL;

		CUBLAS_SAFE_CALL(cublasCreate(&cublasHandle));
		CUSPARSE_SAFE_CALL(cusparseCreate(&cusparseHandle));

		// Set both cuBLAS and cuSPARSE pointer modes to host (we never use constants, stored on the device).
		CUBLAS_SAFE_CALL(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST));
		CUSPARSE_SAFE_CALL(cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST));

		// Create dummy descriptors for J and J^T.
		cusparseMatDescr_t jacobianDesc;
		CUSPARSE_SAFE_CALL(cusparseCreateMatDescr(&jacobianDesc));
		CUSPARSE_SAFE_CALL(cusparseSetMatType(jacobianDesc, CUSPARSE_MATRIX_TYPE_GENERAL));
		CUSPARSE_SAFE_CALL(cusparseSetMatIndexBase(jacobianDesc, CUSPARSE_INDEX_BASE_ZERO));

		MemoryContainer<FloatType> jacobianValues;
		MemoryContainer<int> jacobianInnerColumnIndices;
		MemoryContainer<int> jacobianOuterRowStarts;

		cusparseMatDescr_t jacobianTDesc;
		CUSPARSE_SAFE_CALL(cusparseCreateMatDescr(&jacobianTDesc));
		CUSPARSE_SAFE_CALL(cusparseSetMatType(jacobianTDesc, CUSPARSE_MATRIX_TYPE_GENERAL));
		CUSPARSE_SAFE_CALL(cusparseSetMatIndexBase(jacobianTDesc, CUSPARSE_INDEX_BASE_ZERO));

		MemoryContainer<FloatType> jacobianTValues;
		MemoryContainer<int> jacobianTInnerColumnIndices;
		MemoryContainer<int> jacobianTOuterRowStarts;

		// Create a dummy common residual vector.
		MemoryContainer<FloatType> residualValues;

		// Create a dummy descriptor for A.
		// Dimensions of A: nParameters x nParameters.
		cusparseMatDescr_t A_desc;
		CUSPARSE_SAFE_CALL(cusparseCreateMatDescr(&A_desc));
		CUSPARSE_SAFE_CALL(cusparseSetMatType(A_desc, CUSPARSE_MATRIX_TYPE_GENERAL));
		CUSPARSE_SAFE_CALL(cusparseSetMatIndexBase(A_desc, CUSPARSE_INDEX_BASE_ZERO));

		MemoryContainer<FloatType> A_values;
		MemoryContainer<int> A_innerColumnIndices;
		MemoryContainer<int> A_outerRowStarts;

		// Allocate the solution vector and initialize it to 0.
		solution.allocate(nParameters, 1, Type2Type<MemoryTypeCUDA>());
		CUDA_SAFE_CALL(cudaMemset(solution.getData(Type2Type<MemoryTypeCUDA>()), 0, nParameters * sizeof(FloatType)));
		auto& x = solution.getContainer();

		// Mark the changes to the solution vector.
		x.setUpdated(false, true);

		TIME_GPU_STOP(PCGSolver_Initialize);
		TIME_GPU_START(PCGSolver_ComputeSquaredLoss);

		// We compute the squared loss that will be returned from the linear solver.
		FloatType squaredLoss = FloatType(0);
		for (int i = 0; i < nConstraints; ++i) {
			auto& residualVector = systemComponents[i].first;
			squaredLoss += system_proc::computeSquaredLoss(cublasHandle, residualVector.getContainer());
		}
		runtime_assert(std::isfinite(squaredLoss), "Squared loss (r^T r) is not finite.");

		TIME_GPU_STOP(PCGSolver_ComputeSquaredLoss);
		TIME_GPU_START(PCGSolver_ComputeDiagonalPreconditioner);

		// Compute diagonal preconditioner.
		MemoryContainer<FloatType> diagonalPreconditioner;
		diagonalPreconditioner.allocate(nParameters, Type2Type<MemoryTypeCUDA>());
		cudaMemset(diagonalPreconditioner.getData(Type2Type<MemoryTypeCUDA>()), 0, diagonalPreconditioner.getByteSize());

		// Compute the sum of squares for each parameter.
		for (int i = 0; i < nConstraints; ++i) {
			auto& jacobianMatrix = systemComponents[i].second;
			const unsigned totalResidualDim = jacobianMatrix.getNumResiduals() * jacobianMatrix.getResidualDim();
			const unsigned totalParamDim = jacobianMatrix.getParamDim();
		
			system_proc::initializeDiagonalPreconditioner<<< (totalResidualDim + DIAG_PRECOND_BLOCK_SIZE - 1) / DIAG_PRECOND_BLOCK_SIZE, DIAG_PRECOND_BLOCK_SIZE >>>(
				jacobianMatrix.getData(Type2Type<MemoryTypeCUDA>()), columnIndices.getData(Type2Type<MemoryTypeCUDA>()), constraintStarts.getData(Type2Type<MemoryTypeCUDA>()), 
				i, totalParamDim, totalResidualDim, diagonalPreconditioner.getData(Type2Type<MemoryTypeCUDA>())
			);
			CUDA_CHECK_ERROR();
		}

		// Invert the sum of squares.
		system_proc::invertDiagonalPreconditioner<<< (nParameters + DIAG_PRECOND_BLOCK_SIZE - 1) / DIAG_PRECOND_BLOCK_SIZE, DIAG_PRECOND_BLOCK_SIZE >>>(
			lambda, nParameters, diagonalPreconditioner.getData(Type2Type<MemoryTypeCUDA>())
		);
		CUDA_CHECK_ERROR();

		TIME_GPU_STOP(PCGSolver_ComputeDiagonalPreconditioner);
		TIME_GPU_START(PCGSolver_CheckGradientNorm);

		// Check the gradient norm, if early stopping is enabled.
		if (terminationParameters.bUseGNEarlyStopping) {
			MemoryContainer<FloatType> g;
			g.allocate(nParameters, Type2Type<MemoryTypeCUDA>());
			system_proc::computeGradientWithAtomics(systemComponents, columnIndexVector, rowIndexVector, g);

			float gMaxNorm = system_proc::computeMaxElement(cublasHandle, g);
			terminationCriteria.maxGradientNorm = gMaxNorm;

			if (gMaxNorm < terminationParameters.gradientTolerance) {
				terminationCriteria.bStopEarly = true;
				return squaredLoss;
			}
		}

		TIME_GPU_STOP(PCGSolver_CheckGradientNorm);
		TIME_GPU_START(PCGSolver_ExecuteAlgorithm);

		// Execute PCG algorithm.
		nIterations = system_proc::executePCGAlgorithm(
			cusparseHandle, cublasHandle,
			systemComponents, 
			columnIndexVector, 
			rowIndexVector,
			residualValues,
			jacobianDesc, jacobianValues, jacobianInnerColumnIndices, jacobianOuterRowStarts,
			jacobianTDesc, jacobianTValues, jacobianTInnerColumnIndices, jacobianTOuterRowStarts,
			A_desc, A_values, A_innerColumnIndices, A_outerRowStarts,
			diagonalPreconditioner,
			x,
			lambda, nParameters, jacobianValues.getSize(), residualValues.getSize(), nMaxIterations,
			terminationParameters.bUsePCGEarlyStopping, terminationParameters.qTolerance, terminationParameters.rTolerance,
			system_proc::JTJApplicationType::ATOMIC
		);

		// Mark the changes to the solution vector.
		x.setUpdated(false, true);

		TIME_GPU_STOP(PCGSolver_ExecuteAlgorithm);
		//TIME_GPU_START(PCGSolver_CheckEarlyStop);

		//// Check for termination criteria, if necessary
		//// We opt for early stop if ||Jx|| < eps (1 + ||r||)
		//if (bCheckTerminationCriteria) {
		//	FloatType eps = std::numeric_limits<FloatType>::epsilon();

		//	const unsigned nConstraints = systemComponents.size();
		//	FloatType alpha = FloatType(1);

		//	FloatType totalJxNorm2 = FloatType(0);

		//	// Compute the squared norms of Jx parts for each constraint.
		//	for (int i = 0; i < nConstraints; ++i) {
		//		auto& residualVector = systemComponents[i].first;
		//		auto& jacobianMatrix = systemComponents[i].second;
		//		const unsigned totalResidualDim = jacobianMatrix.getNumResiduals() * jacobianMatrix.getResidualDim();
		//		const unsigned totalParamDim = jacobianMatrix.getParamDim();

		//		// Initialize current Jx.
		//		MemoryContainer<FloatType> Jx;
		//		Jx.allocate(totalResidualDim, Type2Type<MemoryTypeCUDA>());

		//		// Reset the initial vector Jx to 0.
		//		cudaMemset(Jx.getData(Type2Type<MemoryTypeCUDA>()), 0, Jx.getByteSize());

		//		// Compute Jx.
		//		system_proc::applySparseJ<<< (totalResidualDim + J_APPLICATION_BLOCK_SIZE - 1) / J_APPLICATION_BLOCK_SIZE, J_APPLICATION_BLOCK_SIZE >>> (
		//			jacobianMatrix.getData(Type2Type<MemoryTypeCUDA>()), columnIndexVector.getData(Type2Type<MemoryTypeCUDA>()), rowIndexVector.getData(Type2Type<MemoryTypeCUDA>()),
		//			i, totalParamDim, totalResidualDim, alpha,
		//			residualVector.getData(Type2Type<MemoryTypeCUDA>()), Jx.getData(Type2Type<MemoryTypeCUDA>())
		//		);
		//		CUDA_CHECK_ERROR();

		//		// Compute squared norm of Jx.
		//		totalJxNorm2 += system_proc::computeSquaredLoss(cublasHandle, Jx);
		//	}

		//	FloatType totalJxNorm = std::sqrt(totalJxNorm2);
		//	cout << "totalJxNorm = " << totalJxNorm << endl;
		//	cout << "std::sqrt(squaredLoss) = " << std::sqrt(squaredLoss) << endl;
		//	cout << "eps * (1 + std::sqrt(squaredLoss)) = " << eps * (1 + std::sqrt(squaredLoss)) << endl;
		//	if (totalJxNorm < eps * (1 + std::sqrt(squaredLoss)))
		//		terminationCriteria.bStopEarly = true;
		//	else
		//		terminationCriteria.bStopEarly = false;
		//}

		//TIME_GPU_STOP(PCGSolver_CheckEarlyStop);
		TIME_GPU_START(PCGSolver_CleanUp);

		CUSPARSE_SAFE_CALL(cusparseDestroy(cusparseHandle));
		CUBLAS_SAFE_CALL(cublasDestroy(cublasHandle));

		TIME_GPU_STOP(PCGSolver_CleanUp);

		return squaredLoss;
	}


	/**
	 * Explicit declarations of methods that we support, in order to compile them.
	 */
	template float solveImpl_SparsePCGSolverAtomicGPU<float>(
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
	);

	template double solveImpl_SparsePCGSolverAtomicGPU<double>(
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
	);

} // namespace solo