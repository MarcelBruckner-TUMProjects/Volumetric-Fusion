#pragma once
#include <common_utils/timing/TimerGPU.h>

#include "device_functions.h"
#include "SparsePCGSolverSequentialGPU.h"
#include "SystemProcessingCUDA.h"

// Important: This file can only be included in exactly one .cpp file, otherwise the methods would get compiled 
// multiple times.

namespace solo {

	/**
	 * General implemention of SparsePCGSolverSequentialGPU.
	 */
	template<typename FloatType>
	FloatType solveImpl_SparsePCGSolverSequentialGPU(
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

		// Create cuBLAS/cuSPARSE handles and bind them to a common stream.
		cublasHandle_t cublasHandle = NULL;
		cusparseHandle_t cusparseHandle = NULL;

		CUBLAS_SAFE_CALL(cublasCreate(&cublasHandle));
		CUSPARSE_SAFE_CALL(cusparseCreate(&cusparseHandle));

		// Set both cuBLAS and cuSPARSE pointer modes to host (we never use constants, stored on the device).
		CUBLAS_SAFE_CALL(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST));
		CUSPARSE_SAFE_CALL(cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST));

		// Create descriptors for J and J^T.
		cusparseMatDescr_t jacobianDesc;
		CUSPARSE_SAFE_CALL(cusparseCreateMatDescr(&jacobianDesc));
		CUSPARSE_SAFE_CALL(cusparseSetMatType(jacobianDesc, CUSPARSE_MATRIX_TYPE_GENERAL));
		CUSPARSE_SAFE_CALL(cusparseSetMatIndexBase(jacobianDesc, CUSPARSE_INDEX_BASE_ZERO));

		cusparseMatDescr_t jacobianTDesc;
		CUSPARSE_SAFE_CALL(cusparseCreateMatDescr(&jacobianTDesc));
		CUSPARSE_SAFE_CALL(cusparseSetMatType(jacobianTDesc, CUSPARSE_MATRIX_TYPE_GENERAL));
		CUSPARSE_SAFE_CALL(cusparseSetMatIndexBase(jacobianTDesc, CUSPARSE_INDEX_BASE_ZERO));

		// Create a dummy for A.
		// Dimensions of A: nParameters x nParameters.
		cusparseMatDescr_t A_desc;
		CUSPARSE_SAFE_CALL(cusparseCreateMatDescr(&A_desc));
		CUSPARSE_SAFE_CALL(cusparseSetMatType(A_desc, CUSPARSE_MATRIX_TYPE_GENERAL));
		CUSPARSE_SAFE_CALL(cusparseSetMatIndexBase(A_desc, CUSPARSE_INDEX_BASE_ZERO));

		MemoryContainer<FloatType> A_values;
		MemoryContainer<int> A_innerColumnIndices;
		MemoryContainer<int> A_outerRowStarts;

		TIME_GPU_STOP(PCGSolver_Initialize);
		TIME_GPU_START(PCGSolver_PrepareResidualsAndJacobian);

		// Prepare complete dense residual vector and sparse Jacobian matrix (in CSR representation).
		MemoryContainer<FloatType> residualValues;
		MemoryContainer<FloatType> jacobianValuesUnordered;
		auto& jacobianInnerColumnIndices = columnIndexVector.getContainer();
		auto& jacobianOuterRowStarts = rowIndexVector.getContainer();
		system_proc::prepareResidualAndJacobian(
			cublasHandle,
			systemComponents,
			residualValues,
			jacobianValuesUnordered,
			jacobianInnerColumnIndices,
			jacobianOuterRowStarts
		);

		TIME_GPU_STOP(PCGSolver_PrepareResidualsAndJacobian);
		TIME_GPU_START(PCGSolver_ComputeSquaredLoss);

		// We compute the squared loss that will be returned from the linear solver.
		FloatType squaredLoss = system_proc::computeSquaredLoss(cublasHandle, residualValues);
		runtime_assert(std::isfinite(squaredLoss), "Squared loss (r^T r) is not finite.");
		
		TIME_GPU_STOP(PCGSolver_ComputeSquaredLoss)
		TIME_GPU_START(PCGSolver_SortJacobianColumns);

		// Sort the columns of the Jacobian to get a proper (sorted) CSR format.
		MemoryContainer<FloatType> jacobianValues;
		if (!bOrderedParameters) {
			jacobianValues.allocate(jacobianValuesUnordered.getSize(), Type2Type<MemoryTypeCUDA>());

			system_proc::orderJacobian(
				cusparseHandle, jacobianDesc, jacobianValuesUnordered, jacobianValues, jacobianInnerColumnIndices,
				jacobianOuterRowStarts, nParameters, jacobianValuesUnordered.getSize(), residualValues.getSize()
			);
		}
		else {
			jacobianValues = std::move(jacobianValuesUnordered);
		}
		
		TIME_GPU_STOP(PCGSolver_SortJacobianColumns);
		TIME_GPU_START(PCGSolver_ConstructJacobianTranspose);

		// Construct cusparse CSR matrix J and compute J^T with converting to CSC transformation.
		// Jacobian^T dimension: nParameters x totalResidualSize
		MemoryContainer<FloatType> jacobianTValues;
		MemoryContainer<int> jacobianTInnerColumnIndices;
		MemoryContainer<int> jacobianTOuterRowStarts;

		system_proc::constructJacobianTranspose(
			cusparseHandle,
			jacobianValues,
			jacobianInnerColumnIndices,
			jacobianOuterRowStarts,
			jacobianTValues,
			jacobianTInnerColumnIndices,
			jacobianTOuterRowStarts,
			nParameters,
			jacobianValues.getSize(),
			residualValues.getSize()
		);

		TIME_GPU_STOP(PCGSolver_ConstructJacobianTranspose);
		TIME_GPU_START(PCGSolver_ComputeDiagonalPreconditioner);

		// Compute diagonal preconditioner.
		MemoryContainer<FloatType> diagonalPreconditioner;
		diagonalPreconditioner.allocate(nParameters, Type2Type<MemoryTypeCUDA>());

		system_proc::computeDiagonalPreconditioner<<< nParameters, DIAG_PRECOND_BLOCK_SIZE >>>(
			lambda, jacobianTValues.getData(Type2Type<MemoryTypeCUDA>()), jacobianTOuterRowStarts.getData(Type2Type<MemoryTypeCUDA>()), diagonalPreconditioner.getData(Type2Type<MemoryTypeCUDA>())
		);
		CUDA_CHECK_ERROR();

		TIME_GPU_STOP(PCGSolver_ComputeDiagonalPreconditioner);
		TIME_GPU_START(PCGSolver_ExecuteAlgorithm);

		// Allocate the solution vector and initialize it to 0.
		solution.allocate(nParameters, 1, Type2Type<MemoryTypeCUDA>());
		CUDA_SAFE_CALL(cudaMemset(solution.getData(Type2Type<MemoryTypeCUDA>()), 0, nParameters * sizeof(FloatType)));
		auto& x = solution.getContainer();

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
			system_proc::JTJApplicationType::SEQUENTIAL
		);

		// Mark the changes to the solution vector.
		x.setUpdated(false, true);

		TIME_GPU_STOP(PCGSolver_ExecuteAlgorithm);
		// TODO: Refactor early stop test.
		//TIME_GPU_START(PCGSolver_CheckEarlyStop);

		//// Check for termination criteria, if necessary.
		//// We opt for early stop if ||Jx|| < eps (1 + ||r||)
		//if (bCheckTerminationCriteria) {
		//	FloatType eps = std::numeric_limits<FloatType>::epsilon();
		//	const FloatType one{ 1.0 };
		//	const FloatType zero{ 0.0 };

		//	MemoryContainer<FloatType> Jx;
		//	Jx.allocate(residualValues.getSize(), Type2Type<MemoryTypeCUDA>());

		//	CUSPARSE_SAFE_CALL(system_proc::cusparseXcsrmv(
		//		cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		//		residualValues.getSize(), nParameters, jacobianValues.getSize(), &one, jacobianDesc,
		//		jacobianValues.getData(Type2Type<MemoryTypeCUDA>()), jacobianOuterRowStarts.getData(Type2Type<MemoryTypeCUDA>()), jacobianInnerColumnIndices.getData(Type2Type<MemoryTypeCUDA>()),
		//		x.getData(Type2Type<MemoryTypeCUDA>()), &zero, Jx.getData(Type2Type<MemoryTypeCUDA>())
		//	));

		//	FloatType JxNorm = std::sqrt(system_proc::computeSquaredLoss(cublasHandle, Jx));
		//	if (JxNorm < eps * (1 + std::sqrt(squaredLoss)))
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
	template float solveImpl_SparsePCGSolverSequentialGPU<float>(
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

	template double solveImpl_SparsePCGSolverSequentialGPU<double>(
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