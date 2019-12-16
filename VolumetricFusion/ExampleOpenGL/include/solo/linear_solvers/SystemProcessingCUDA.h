#pragma once
#include "Solo/utils/IncludeCUDA.h"

#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_runtime.h>

#include "device_functions.h"

namespace solo {
	namespace  system_proc {

		#define DIAG_PRECOND_BLOCK_SIZE 64
		#define JTJ_APPLICATION_BLOCK_SIZE 256
		#define JT_APPLICATION_BLOCK_SIZE 256
		#define J_APPLICATION_BLOCK_SIZE 256

		/**
		 * Decides about the J^T J application type on a vector x.
		 */
		enum class JTJApplicationType {
			SEQUENTIAL,
			COMPLETE,
			ATOMIC
		};


		/**
		 * For a warp we have a guarantee that every line of code is executed at the same time for all thread
		 * in a warp.
		 */
		template<class T>
		inline __device__ void warpReduce(volatile T* sharedData, int threadId) {
			sharedData[threadId] += sharedData[threadId + 32];
			sharedData[threadId] += sharedData[threadId + 16];
			sharedData[threadId] += sharedData[threadId + 8];
			sharedData[threadId] += sharedData[threadId + 4];
			sharedData[threadId] += sharedData[threadId + 2];
			sharedData[threadId] += sharedData[threadId + 1];
		}


		/**
		 * For a warp we have a guarantee that every line of code is executed at the same time for all thread
		 * in a warp.
		 */
		template<class T>
		inline __device__ void warpMax(volatile T* sharedData, int threadId) {
			sharedData[threadId] = max(sharedData[threadId], sharedData[threadId + 32]);
			sharedData[threadId] = max(sharedData[threadId], sharedData[threadId + 16]);
			sharedData[threadId] = max(sharedData[threadId], sharedData[threadId + 8]);
			sharedData[threadId] = max(sharedData[threadId], sharedData[threadId + 4]);
			sharedData[threadId] = max(sharedData[threadId], sharedData[threadId + 2]);
			sharedData[threadId] = max(sharedData[threadId], sharedData[threadId + 1]);
		};


		/**
		 * AtomicAdd for float/double types, if architecture doesn't support them yet.
		 */
#		if __CUDA_ARCH__ < 600
		inline __device__ double atomicAdd(double* address, double val) {
			unsigned long long int* address_as_ull =
				(unsigned long long int*)address;
			unsigned long long int old = *address_as_ull, assumed;

			do {
				assumed = old;
				old = atomicCAS(address_as_ull, assumed,
					__double_as_longlong(val +
						__longlong_as_double(assumed)));

				// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
			} while (assumed != old);

			return __longlong_as_double(old);
		}

		inline __device__ float atomicAdd(float* address, float val) {
			unsigned int* address_as_ull =
				(unsigned int*)address;
			unsigned int old = *address_as_ull, assumed;

			do {
				assumed = old;
				old = atomicCAS(address_as_ull, assumed,
					__float_as_uint(val +
						__uint_as_float(assumed)));

				// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
			} while (assumed != old);

			return __uint_as_float(old);
		}
#		endif


		/**
		 * Kernel that computes a diagonal element of J^T J, given J^T in CSR format.
		 */
		template<typename FloatType>
		__global__ void computeDiagonalPreconditioner(FloatType lambda, const FloatType* values, const int* rowOuterStarts, FloatType* inverseDiagonalOfJTJ) {
			int row = blockIdx.x;
			int rowBegin = rowOuterStarts[row];
			int rowEnd = rowOuterStarts[row + 1];
			int threadId = threadIdx.x;
			int valueIdx = rowBegin + threadId;

			// Compute the sum of squares for a given row => one warp per row.
			FloatType sumOfSquares{ FloatType(0) };
			while (valueIdx < rowEnd) {
				FloatType value = values[valueIdx];
				sumOfSquares += value * value;
				valueIdx += DIAG_PRECOND_BLOCK_SIZE;
			}

			// Write the sum of squares (per thread) in shared memory.
			__shared__ FloatType sumOfSquaresComponents[DIAG_PRECOND_BLOCK_SIZE];
			sumOfSquaresComponents[threadId] = sumOfSquares;
			__syncthreads();

			// Execute reduction.
			for (unsigned s = DIAG_PRECOND_BLOCK_SIZE >> 1; s > 32; s >>= 1) {
				if (threadId < s) {
					sumOfSquaresComponents[threadId] = sumOfSquaresComponents[threadId + s];
					__syncthreads();
				}
			}

			if (threadId < 32) warpReduce(sumOfSquaresComponents, threadId);

			if (threadId == 0) {
				FloatType diagInv = sumOfSquaresComponents[0] + lambda;
				if (diagInv > FloatType(0))
					inverseDiagonalOfJTJ[row] = FloatType(1) / diagInv;
				else
					inverseDiagonalOfJTJ[row] = FloatType(1);
			}
		}


		/**
		 * Kernel that computes diagonal elements of J^T J, given J, stored column-wise, element index vector I, stored
		 * column-wise, and element index vector offset vector O (together with the index of offset).
		 * Each thread handles the contribution of one residual to the diagonal elements. Atomic operations are used to
		 * handle thread conflicts.
		 */
		template<typename FloatType>
		__global__ void initializeDiagonalPreconditioner(
			const FloatType* values, 
			const int* indices, 
			const int* offsets, 
			int offsetIdx, 
			int totalParamDim,
			int totalResidualDim,
			FloatType* diagonalOfJTJ
		) {
			int threadId = blockIdx.x * blockDim.x + threadIdx.x;
			int memoryOffset = offsets[offsetIdx];

			// Execute the sum of squares for all jacobian values, corresponding to each parameter index.
			if (threadId < totalResidualDim) {
				for (int paramId = 0; paramId < totalParamDim; ++paramId) {
					int localMemoryIdx = threadId + paramId * totalResidualDim;
					FloatType value = values[localMemoryIdx];
					int index = indices[memoryOffset + localMemoryIdx];
					atomicAdd(diagonalOfJTJ + index, value * value);
				}
			}
		}

		template<typename FloatType>
		__global__ void invertDiagonalPreconditioner(
			FloatType lambda,
			int nParameters,
			FloatType* inverseDiagonalOfJTJ
		) {
			int threadId = blockIdx.x * blockDim.x + threadIdx.x;

			// Invert the sum of squares (numerically stable).
			if (threadId < nParameters) {
				FloatType diagInv = inverseDiagonalOfJTJ[threadId] + lambda;
				if (diagInv > FloatType(0))
					inverseDiagonalOfJTJ[threadId] = FloatType(1) / diagInv;
				else
					inverseDiagonalOfJTJ[threadId] = FloatType(1);
			}
		}


		/**
		 * Kernel that applies a sparse column-major matrix J to a dense vector x as y = (J^T J) x. 
		 * The sparse vector indices are provided in the columnIndices vector (also column-major).
		 * Each thread takes care for one residual, i.e. one row in matrix J and one column in 
		 * matrix J^T, and atomically sums the contributions of this residual to the final result
		 * vector y.
		 * Indices are given with a memory offset, and offset for the current constraint is provided
		 * in the offsets array.
		 */
		template<typename FloatType>
		__global__ void applySparseJTJ(
			const FloatType* values,
			const int* indices,
			const int* offsets,
			int offsetIdx,
			int totalParamDim,
			int totalResidualDim,
			const FloatType* x,
			FloatType* y
		) {
			int threadId = blockIdx.x * blockDim.x + threadIdx.x;
			int memoryOffset = offsets[offsetIdx];

			if (threadId < totalResidualDim) {
				// Compute the Jx component for the current residual id.
				FloatType Jx = FloatType(0);
				for (int paramId = 0; paramId < totalParamDim; ++paramId) {
					int localMemoryIdx = threadId + paramId * totalResidualDim;
					FloatType value = values[localMemoryIdx];
					int index = indices[memoryOffset + localMemoryIdx];
					Jx += value * x[index];
				}

				// Store the (J^T J)x contributions for the current residual id.
				for (int paramId = 0; paramId < totalParamDim; ++paramId) {
					int localMemoryIdx = threadId + paramId * totalResidualDim;
					FloatType value = values[localMemoryIdx];
					int index = indices[memoryOffset + localMemoryIdx];
					atomicAdd(y + index, value * Jx);
				}
			}
		}


		/**
		 * Kernel that applies a sparse column-major matrix J to a dense vector x as y = y + alpha (J x),
		 * where y is the resulting vector.
		 * The sparse vector indices are provided in the columnIndices vector (also column-major).
		 * Each thread takes care for one residual, i.e. one row in matrix J, and computes one component
		 * of the final result vector y.
		 * Indices are given with a memory offset, and offset for the current constraint is provided
		 * in the offsets array.
		 */
		template<typename FloatType>
		__global__ void applySparseJ(
			const FloatType* values,
			const int* indices,
			const int* offsets,
			int offsetIdx,
			int totalParamDim,
			int totalResidualDim,
			FloatType alpha,
			const FloatType* x,
			FloatType* y
		) {
			int threadId = blockIdx.x * blockDim.x + threadIdx.x;

			if (threadId < totalResidualDim) {
				// Compute the current residual row of alpha (J x).
				int memoryOffset = offsets[offsetIdx];
				FloatType yValue = y[threadId];

				for (int paramId = 0; paramId < totalParamDim; ++paramId) {
					int localMemoryIdx = threadId + paramId * totalResidualDim;
					FloatType JValue = values[localMemoryIdx];
					int index = indices[memoryOffset + localMemoryIdx];
					yValue = yValue + alpha * JValue * x[index];
				}

				y[threadId] = yValue;
			}
		}


		/**
		 * Kernel that applies a sparse column-major matrix J to a dense vector x as y = y + alpha (J^T x),
		 * where y is the resulting vector.
		 * The sparse vector indices are provided in the columnIndices vector (also column-major).
		 * Each thread takes care for one residual, i.e. one column in matrix J^T, and atomically 
		 * sums the contributions of this residual to the final result vector y.
		 * Indices are given with a memory offset, and offset for the current constraint is provided
		 * in the offsets array.
		 */
		template<typename FloatType>
		__global__ void applySparseJT(
			const FloatType* values,
			const int* indices,
			const int* offsets,
			int offsetIdx,
			int totalParamDim,
			int totalResidualDim,
			FloatType alpha,
			const FloatType* x,
			FloatType* y
		) {
			int threadId = blockIdx.x * blockDim.x + threadIdx.x;

			if (threadId < totalResidualDim) {
				// Compute the alpha (J^T x) contributions for the current residual id.
				FloatType xValue = x[threadId];
				int memoryOffset = offsets[offsetIdx];

				for (int paramId = 0; paramId < totalParamDim; ++paramId) {
					int localMemoryIdx = threadId + paramId * totalResidualDim;
					FloatType JTValue = values[localMemoryIdx];
					int index = indices[memoryOffset + localMemoryIdx];
					atomicAdd(y + index, alpha * JTValue * xValue);
				}
			}
		}


		/**
		 * Kernel that executes element-wise vector multiplication.
		 */
		template<typename FloatType>
		__global__ void elementWiseMultiplication(const FloatType* a, const FloatType* b, FloatType* out, int N) {
			const int i = blockIdx.x * blockDim.x + threadIdx.x;
			if (i < N) {
				out[i] = a[i] * b[i];
			}
		}


		/**
		 * Kernel that sets the FloatType vector to 1-vector (all components are equal to 1).
		 */
		template<typename FloatType>
		__global__ void setOneVector(FloatType* v, int N) {
			const int i = blockIdx.x * blockDim.x + threadIdx.x;
			if (i < N) {
				v[i] = FloatType(1.0);
			}
		}


		/**
		 * Floating-point type invariant cuBLAS and cuSPARSE functions.
		 */
		inline cublasStatus_t cublasXgeam(
			cublasHandle_t handle,
			cublasOperation_t transA,
			cublasOperation_t transB,
			int m, int n,
			const double *alpha,
			const double *A, int lda,
			const double *beta,
			const double *B, int ldb,
			double *C, int ldc
		) {
			return cublasDgeam(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
		}

		inline cublasStatus_t cublasXgeam(
			cublasHandle_t handle,
			cublasOperation_t transA,
			cublasOperation_t transB,
			int m, int n,
			const float *alpha,
			const float *A, int lda,
			const float *beta,
			const float *B, int ldb,
			float *C, int ldc
		) {
			return cublasSgeam(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
		}

		inline cublasStatus_t cublasIxamax(
			cublasHandle_t handle, 
			int n, const float* x, int incx, 
			int* result
		) { 
			return cublasIsamax(handle, n, x, incx, result);
		}

		inline cublasStatus_t cublasIxamax(
			cublasHandle_t handle,
			int n, const double* x, int incx,
			int* result
		) {
			return cublasIdamax(handle, n, x, incx, result);
		}

		inline cublasStatus_t cublasXdot(
			cublasHandle_t handle, int n,
			const double *x, int incx,
			const double *y, int incy,
			double *result
		) {
			return cublasDdot(handle, n, x, incx, y, incy, result);
		}

		inline cublasStatus_t cublasXdot(
			cublasHandle_t handle, int n,
			const float *x, int incx,
			const float *y, int incy,
			float *result
		) {
			return cublasSdot(handle, n, x, incx, y, incy, result);
		}

		inline cublasStatus_t cublasXscal(
			cublasHandle_t handle, int n,
			const double *alpha, double *x, int incx
		) {
			return cublasDscal(handle, n, alpha, x, incx);
		}

		inline cublasStatus_t cublasXscal(
			cublasHandle_t handle, int n,
			const float *alpha, float *x, int incx
		) {
			return cublasSscal(handle, n, alpha, x, incx);
		}

		inline cublasStatus_t cublasXaxpy(
			cublasHandle_t handle, int n, const double *alpha,
			const double *x, int incx,
			double *y, int incy
		) {
			return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
		}

		inline cublasStatus_t cublasXaxpy(
			cublasHandle_t handle, int n, const float *alpha,
			const float *x, int incx,
			float *y, int incy
		) {
			return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
		}

		inline cusparseStatus_t cusparseXgthr(
			cusparseHandle_t handle, int nnz,
			const double *y, double *xVal, const int *xInd,
			cusparseIndexBase_t idxBase
		) {
			return cusparseDgthr(handle, nnz, y, xVal, xInd, idxBase);
		}

		inline cusparseStatus_t cusparseXgthr(
			cusparseHandle_t handle, int nnz,
			const float *y, float *xVal, const int *xInd,
			cusparseIndexBase_t idxBase
		) {
			return cusparseSgthr(handle, nnz, y, xVal, xInd, idxBase);
		}

		inline cusparseStatus_t cusparseXcsr2csc(
			cusparseHandle_t handle,
			int m, int n, int nnz,
			const double *csrVal, const int *csrRowPtr, const int *csrColInd,
			double *cscVal, int *cscRowInd, int *cscColPtr,
			cusparseAction_t copyValues, cusparseIndexBase_t idxBase
		) {
			return cusparseDcsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd, cscColPtr, copyValues, idxBase);
		}

		inline cusparseStatus_t cusparseXcsr2csc(
			cusparseHandle_t handle,
			int m, int n, int nnz,
			const float *csrVal, const int *csrRowPtr, const int *csrColInd,
			float *cscVal, int *cscRowInd, int *cscColPtr,
			cusparseAction_t copyValues, cusparseIndexBase_t idxBase
		) {
			return cusparseScsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd, cscColPtr, copyValues, idxBase);
		}

		inline cusparseStatus_t cusparseXcsrgemm(
			cusparseHandle_t handle,
			cusparseOperation_t transA, cusparseOperation_t transB,
			int m, int n, int k,
			const cusparseMatDescr_t descrA, const int nnzA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA,
			const cusparseMatDescr_t descrB, const int nnzB, const double *csrValB, const int *csrRowPtrB, const int *csrColIndB,
			const cusparseMatDescr_t descrC, double *csrValC, const int *csrRowPtrC, int *csrColIndC
		) {
			return cusparseDcsrgemm(
				handle, transA, transB, m, n, k,
				descrA, nnzA, csrValA, csrRowPtrA, csrColIndA,
				descrB, nnzB, csrValB, csrRowPtrB, csrColIndB,
				descrC, csrValC, csrRowPtrC, csrColIndC
			);
		}

		inline cusparseStatus_t cusparseXcsrgemm(
			cusparseHandle_t handle,
			cusparseOperation_t transA, cusparseOperation_t transB,
			int m, int n, int k,
			const cusparseMatDescr_t descrA, const int nnzA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
			const cusparseMatDescr_t descrB, const int nnzB, const float *csrValB, const int *csrRowPtrB, const int *csrColIndB,
			const cusparseMatDescr_t descrC, float *csrValC, const int *csrRowPtrC, int *csrColIndC
		) {
			return cusparseScsrgemm(
				handle, transA, transB, m, n, k,
				descrA, nnzA, csrValA, csrRowPtrA, csrColIndA,
				descrB, nnzB, csrValB, csrRowPtrB, csrColIndB,
				descrC, csrValC, csrRowPtrC, csrColIndC
			);
		}

		inline cusparseStatus_t cusparseXcsrmv(
			cusparseHandle_t handle, cusparseOperation_t transA,
			int m, int n, int nnz, const double *alpha,
			const cusparseMatDescr_t descrA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA,
			const double *x, const double *beta, double *y
		) {
			return cusparseDcsrmv(handle, transA, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y);
		}

		inline cusparseStatus_t cusparseXcsrmv(
			cusparseHandle_t handle, cusparseOperation_t transA,
			int m, int n, int nnz, const float *alpha,
			const cusparseMatDescr_t descrA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
			const float *x, const float *beta, float *y
		) {
			return cusparseScsrmv(handle, transA, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y);
		}


		/**
		 * Helper fuction for computation of squared loss (i.e. L2 norm) of given vector.
		 */
		template<typename FloatType>
		FloatType computeSquaredLoss(
			cublasHandle_t& cublasHandle,
			const MemoryContainer<FloatType>& vector
		) {
			FloatType squaredLoss;
			CUBLAS_SAFE_CALL(cublasXdot(cublasHandle, vector.getSize(), vector.getData(Type2Type<MemoryTypeCUDA>()), 1, vector.getData(Type2Type<MemoryTypeCUDA>()), 1, &squaredLoss));
			return squaredLoss;
		}

		/**
		 * Helper fuction for computation of maximum (i.e. inf-norm) of given vector.
		 */
		template<typename FloatType>
		FloatType computeMaxElement(
			cublasHandle_t& cublasHandle,
			const MemoryContainer<FloatType>& vector
		) {
			// Query the element index with maximum magnitude (absolute values are used).
			// Important: The function returns a 1-based indexing.
			int maxElementIdx{ -1 };
			CUBLAS_SAFE_CALL(cublasIxamax(cublasHandle, vector.getSize(), vector.getData(Type2Type<MemoryTypeCUDA>()), 1, &maxElementIdx));
			runtime_assert(maxElementIdx > 0 && maxElementIdx <= vector.getSize(), "The max-element index must be in the right range.");
			
			// Copy the max element value to host memory.
			FloatType maxElementVal;
			CUDA_SAFE_CALL(cudaMemcpy(&maxElementVal, vector.getData(Type2Type<MemoryTypeCUDA>()) + maxElementIdx - 1, sizeof(FloatType), cudaMemcpyDeviceToHost));

			return std::abs(maxElementVal);
		}


		/**
		 * Helper function for computing the complete dense residual vector and sparse Jacobian matrix in
		 * CSR representation.
		 */
		template<typename FloatType>
		inline void prepareResidualAndJacobian(
			cublasHandle_t& cublasHandle,
			vector<std::pair<ResidualVector<FloatType>, JacobianMatrix<FloatType>>>& systemComponents,
			MemoryContainer<FloatType>& residualValues,
			MemoryContainer<FloatType>& jacobianValues,
			MemoryContainer<int>& jacobianInnerColumnIndices,
			MemoryContainer<int>& jacobianOuterRowStarts
		) {
			// Constant coefficients.
			const FloatType one{ 1.0 };
			const FloatType minusOne{ -1.0 };
			const FloatType zero{ 0.0 };

			const unsigned nConstraints = systemComponents.size();

			// Check that indices are stored on the GPU (otherwise update).
			if (jacobianInnerColumnIndices.isUpdatedHost() && !jacobianInnerColumnIndices.isAllocatedDevice()) {
				jacobianInnerColumnIndices.copyHostToDevice();
				jacobianInnerColumnIndices.setUpdated(true, true);
			}

			if (jacobianOuterRowStarts.isUpdatedHost() && !jacobianOuterRowStarts.isAllocatedDevice()) {
				jacobianOuterRowStarts.copyHostToDevice();
				jacobianOuterRowStarts.setUpdated(true, true);
			}

			// Compute the total required memory sizes.
			unsigned totalResidualSize = 0;
			unsigned totalJacobianSize = 0;
			for (int i = 0; i < nConstraints; ++i) {
				auto& residualVector = systemComponents[i].first;
				totalResidualSize += residualVector.getSize();

				auto& jacobianMatrix = systemComponents[i].second;
				totalJacobianSize += jacobianMatrix.getSize();
			}

			// Reserve the complete GPU memory for full residual and Jacobian matrix.
			residualValues.allocate(totalResidualSize, Type2Type<MemoryTypeCUDA>());
			jacobianValues.allocate(totalJacobianSize, Type2Type<MemoryTypeCUDA>());

			// Copy the residual vector into the complete memory.
			FloatType* d_residualData = residualValues.getData(Type2Type<MemoryTypeCUDA>());
			int memoryOffset = 0;
			for (int i = 0; i < nConstraints; ++i) {
				auto& residualVector = systemComponents[i].first;
				auto& container = residualVector.getContainer();
				if (container.isUpdatedHost() && !container.isAllocatedDevice()) {
					container.copyHostToDevice();
					container.setUpdated(true, true);
				}

				CUDA_SAFE_CALL(cudaMemcpy(d_residualData + memoryOffset, residualVector.getData(Type2Type<MemoryTypeCUDA>()), residualVector.getSize() * sizeof(FloatType), cudaMemcpyDeviceToDevice));
				memoryOffset += residualVector.getSize();
			}

			// Compute the CSR (compresssed-sparse-row) representation of Jacobian matrix.
			// Jacobian dimension: totalResidualSize x nParameters
			FloatType* d_jacobianData = jacobianValues.getData(Type2Type<MemoryTypeCUDA>());
			memoryOffset = 0;
			for (int i = 0; i < nConstraints; ++i) {
				auto& jacobianMatrix = systemComponents[i].second;
				auto& container = jacobianMatrix.getContainer();
				if (container.isUpdatedHost() && !container.isAllocatedDevice()) {
					container.copyHostToDevice();
					container.setUpdated(true, true);
				}

				const int m{ int(jacobianMatrix.mat().cols()) };
				const int n{ int(jacobianMatrix.mat().rows()) };

				FloatType* d_jacobianIn = container.getData(Type2Type<MemoryTypeCUDA>());
				CUBLAS_SAFE_CALL(cublasXgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &one, d_jacobianIn, n, &zero, d_jacobianIn, n, d_jacobianData + memoryOffset, m));
				memoryOffset += jacobianMatrix.getSize();
			}
		}


		/**
		 * Helper function for ordering the unordered sparse matrix in CSR format.
		 * Before using any cuSPARSE function, sparse matrices need to be in ordered CSR
		 * format. The column indices are ordered in-place, while the values are written
		 * to a different output array.
		 */
		template<typename FloatType>
		inline void orderJacobian(
			cusparseHandle_t& cusparseHandle,
			cusparseMatDescr_t& jacobianDesc,
			const MemoryContainer<FloatType>& jacobianValuesUnordered,
			MemoryContainer<FloatType>& jacobianValuesOrdered,
			MemoryContainer<int>& jacobianInnerColumnIndices,
			const MemoryContainer<int>& jacobianOuterRowStarts,
			unsigned nParameters,
			unsigned totalJacobianSize,
			unsigned totalResidualSize
		) {
			size_t pBufferSizeInBytes = 0;
			void *pBuffer = NULL;
			int *P = NULL;

			// Step 1: allocate buffer. 
			const unsigned nnz = totalJacobianSize;
			const unsigned m = totalResidualSize;
			const unsigned n = nParameters;
			CUSPARSE_SAFE_CALL(cusparseXcsrsort_bufferSizeExt(cusparseHandle, m, n, nnz, jacobianOuterRowStarts.getData(Type2Type<MemoryTypeCUDA>()), jacobianInnerColumnIndices.getData(Type2Type<MemoryTypeCUDA>()), &pBufferSizeInBytes));
			CUDA_SAFE_CALL(cudaMalloc(&pBuffer, sizeof(char)*pBufferSizeInBytes));

			// Step 2: setup permutation vector P to identity. 
			CUDA_SAFE_CALL(cudaMalloc((void**)&P, sizeof(int)*nnz));
			CUSPARSE_SAFE_CALL(cusparseCreateIdentityPermutation(cusparseHandle, nnz, P));

			// Step 3: sort CSR format. 
			CUSPARSE_SAFE_CALL(cusparseXcsrsort(cusparseHandle, m, n, nnz, jacobianDesc, jacobianOuterRowStarts.getData(Type2Type<MemoryTypeCUDA>()), jacobianInnerColumnIndices.getData(Type2Type<MemoryTypeCUDA>()), P, pBuffer));

			// Step 4: gather sorted csrVal. 
			CUSPARSE_SAFE_CALL(cusparseXgthr(cusparseHandle, nnz, jacobianValuesUnordered.getData(Type2Type<MemoryTypeCUDA>()), jacobianValuesOrdered.getData(Type2Type<MemoryTypeCUDA>()), P, CUSPARSE_INDEX_BASE_ZERO));

			// Step 5: clean up.
			CUDA_SAFE_CALL(cudaFree(pBuffer));
			CUDA_SAFE_CALL(cudaFree(P));
		}


		/**
		 * Helper function for construction of a sparse tranposed Jacobian matrix, where the Jacobian matrix
		 * is in a CSR representation.
		 */
		template<typename FloatType>
		inline void constructJacobianTranspose(
			cusparseHandle_t& cusparseHandle,
			const MemoryContainer<FloatType>& jacobianValues,
			const MemoryContainer<int>& jacobianInnerColumnIndices,
			const MemoryContainer<int>& jacobianOuterRowStarts,
			MemoryContainer<FloatType>& jacobianTValues,
			MemoryContainer<int>& jacobianTInnerColumnIndices,
			MemoryContainer<int>& jacobianTOuterRowStarts,
			unsigned nParameters,
			unsigned totalJacobianSize,
			unsigned totalResidualSize
		) {
			// Construct cusparse CSR matrix J and compute J^T with converting to CSC transformation.
			// Jacobian^T dimension: nParameters x totalResidualSize
			jacobianTValues.allocate(totalJacobianSize, Type2Type<MemoryTypeCUDA>());
			jacobianTInnerColumnIndices.allocate(totalJacobianSize, Type2Type<MemoryTypeCUDA>());
			jacobianTOuterRowStarts.allocate(nParameters + 1, Type2Type<MemoryTypeCUDA>());

			CUSPARSE_SAFE_CALL(cusparseXcsr2csc(
				cusparseHandle, totalResidualSize, nParameters, totalJacobianSize,
				jacobianValues.getData(Type2Type<MemoryTypeCUDA>()), jacobianOuterRowStarts.getData(Type2Type<MemoryTypeCUDA>()), jacobianInnerColumnIndices.getData(Type2Type<MemoryTypeCUDA>()),
				jacobianTValues.getData(Type2Type<MemoryTypeCUDA>()), jacobianTInnerColumnIndices.getData(Type2Type<MemoryTypeCUDA>()), jacobianTOuterRowStarts.getData(Type2Type<MemoryTypeCUDA>()),
				CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO
			));
		}


		/**
		 * Helper function for construction of a sparse J^T J matrix from given J^T and J matrices, both in
		 * CSR format.
		 */
		template<typename FloatType>
		inline void constructJTJ(
			cusparseHandle_t& cusparseHandle,
			cusparseMatDescr_t& jacobianDesc,
			const MemoryContainer<FloatType>& jacobianValues,
			const MemoryContainer<int>& jacobianInnerColumnIndices,
			const MemoryContainer<int>& jacobianOuterRowStarts,
			cusparseMatDescr_t& jacobianTDesc,
			const MemoryContainer<FloatType>& jacobianTValues,
			const MemoryContainer<int>& jacobianTInnerColumnIndices,
			const MemoryContainer<int>& jacobianTOuterRowStarts,
			cusparseMatDescr_t& A_desc,
			MemoryContainer<FloatType>& A_values,
			MemoryContainer<int>& A_innerColumnIndices,
			MemoryContainer<int>& A_outerRowStarts,
			unsigned nParameters,
			unsigned totalJacobianSize,
			unsigned totalResidualSize
		) {
			// Compute A = J^T J.
			// Dimensions of A: nParameters x nParameters.
			// We first compute the final number of nonzero elements in A.
			A_outerRowStarts.allocate(nParameters + 1, Type2Type<MemoryTypeCUDA>());

			int baseA, nnzA; // nnzTotalDevHostPtr points to host memory 
			int *nnzTotalDevHostPtr = &nnzA;
			CUSPARSE_SAFE_CALL(cusparseXcsrgemmNnz(
				cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
				nParameters, nParameters, totalResidualSize,
				jacobianTDesc, totalJacobianSize, jacobianTOuterRowStarts.getData(Type2Type<MemoryTypeCUDA>()), jacobianTInnerColumnIndices.getData(Type2Type<MemoryTypeCUDA>()),
				jacobianDesc, totalJacobianSize, jacobianOuterRowStarts.getData(Type2Type<MemoryTypeCUDA>()), jacobianInnerColumnIndices.getData(Type2Type<MemoryTypeCUDA>()),
				A_desc, A_outerRowStarts.getData(Type2Type<MemoryTypeCUDA>()), nnzTotalDevHostPtr
			));

			// We synchronize the device, since the nnzTotalDevHostPtr is a host pointer, returned from the device.
			// By default, these kind of calls are not blocking (although the example in the cuSPARSE documentation
			// doesn't include explicit synchronization).
			CUDA_SAFE_CALL(cudaDeviceSynchronize());

			if (NULL != nnzTotalDevHostPtr) {
				nnzA = *nnzTotalDevHostPtr;
			}
			else {
				CUDA_SAFE_CALL(cudaMemcpy(&nnzA, A_outerRowStarts.getData(Type2Type<MemoryTypeCUDA>()) + nParameters, sizeof(int), cudaMemcpyDeviceToHost));
				CUDA_SAFE_CALL(cudaMemcpy(&baseA, A_outerRowStarts.getData(Type2Type<MemoryTypeCUDA>()), sizeof(int), cudaMemcpyDeviceToHost));
				nnzA -= baseA;
			}

			// Now we can compute the sparse matrix-matrix product.
			A_values.allocate(nnzA, Type2Type<MemoryTypeCUDA>());
			A_innerColumnIndices.allocate(nnzA, Type2Type<MemoryTypeCUDA>());

			CUSPARSE_SAFE_CALL(cusparseXcsrgemm(
				cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
				nParameters, nParameters, totalResidualSize,
				jacobianTDesc, totalJacobianSize, jacobianTValues.getData(Type2Type<MemoryTypeCUDA>()),
				jacobianTOuterRowStarts.getData(Type2Type<MemoryTypeCUDA>()), jacobianTInnerColumnIndices.getData(Type2Type<MemoryTypeCUDA>()),
				jacobianDesc, totalJacobianSize, jacobianValues.getData(Type2Type<MemoryTypeCUDA>()),
				jacobianOuterRowStarts.getData(Type2Type<MemoryTypeCUDA>()), jacobianInnerColumnIndices.getData(Type2Type<MemoryTypeCUDA>()),
				A_desc, A_values.getData(Type2Type<MemoryTypeCUDA>()), A_outerRowStarts.getData(Type2Type<MemoryTypeCUDA>()), A_innerColumnIndices.getData(Type2Type<MemoryTypeCUDA>())
			));
		}


		/**
		 * Computes right side of the linear system r as r = -J^T F for sparse input matrix J^T and dense vector F,
		 * using cuSPARSE. The vector r should already be properly allocated.
		 */
		template<typename FloatType>
		inline void constructRightSide(
			cusparseHandle_t& cusparseHandle,
			cusparseMatDescr_t& jacobianTDesc,
			const MemoryContainer<FloatType>& jacobianTValues,
			const MemoryContainer<int>& jacobianTInnerColumnIndices,
			const MemoryContainer<int>& jacobianTOuterRowStarts,
			const MemoryContainer<FloatType>& F,
			MemoryContainer<FloatType>& r,
			unsigned nParameters,
			unsigned totalResidualSize,
			unsigned totalJacobianSize
		) {
			// Constant coefficients.
			const FloatType minusOne{ -1.0 };
			const FloatType zero{ 0.0 };

			CUSPARSE_SAFE_CALL(cusparseXcsrmv(
				cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				nParameters, totalResidualSize, totalJacobianSize, &minusOne, jacobianTDesc,
				jacobianTValues.getData(Type2Type<MemoryTypeCUDA>()), jacobianTOuterRowStarts.getData(Type2Type<MemoryTypeCUDA>()), jacobianTInnerColumnIndices.getData(Type2Type<MemoryTypeCUDA>()),
				F.getData(Type2Type<MemoryTypeCUDA>()), &zero, r.getData(Type2Type<MemoryTypeCUDA>())
			));
		}


		/**
		 * Computes right side of the linear system r as r = -J^T F for sparse input matrix J and dense vector F,
		 * using atomic operations. The vector r should already be properly allocated.
		 */
		template<typename FloatType>
		inline void constructRightSideWithAtomics(
			vector<std::pair<ResidualVector<FloatType>, JacobianMatrix<FloatType>>>& systemComponents,
			DenseMatrix<int>& columnIndices,
			DenseMatrix<int>& constraintStarts,
			MemoryContainer<FloatType>& r
		) {
			const unsigned nConstraints = systemComponents.size();
			FloatType alpha = FloatType(-1);

			// Reset the initial vector y to 0.
			cudaMemset(r.getData(Type2Type<MemoryTypeCUDA>()), 0, r.getByteSize());

			// Compute the r parts for each constraint.
			for (int i = 0; i < nConstraints; ++i) {
				auto& residualVector = systemComponents[i].first;
				auto& jacobianMatrix = systemComponents[i].second;
				const unsigned totalResidualDim = jacobianMatrix.getNumResiduals() * jacobianMatrix.getResidualDim();
				const unsigned totalParamDim = jacobianMatrix.getParamDim();

				applySparseJT<<< (totalResidualDim + JT_APPLICATION_BLOCK_SIZE - 1) / JT_APPLICATION_BLOCK_SIZE, JT_APPLICATION_BLOCK_SIZE >>>(
					jacobianMatrix.getData(Type2Type<MemoryTypeCUDA>()), columnIndices.getData(Type2Type<MemoryTypeCUDA>()), constraintStarts.getData(Type2Type<MemoryTypeCUDA>()),
					i, totalParamDim, totalResidualDim, alpha,
					residualVector.getData(Type2Type<MemoryTypeCUDA>()), r.getData(Type2Type<MemoryTypeCUDA>())
				);
				CUDA_CHECK_ERROR();
			}
		}


		/**
		 * Applies sparse matrix A to the vector x and adds a lambda factor to the result, i.e. y = Ax + lambda I.
		 */
		template<typename FloatType>
		inline void applyA(
			cusparseHandle_t& cusparseHandle,
			cublasHandle_t& cublasHandle,
			cusparseMatDescr_t& A_desc,
			const MemoryContainer<FloatType>& A_values,
			const MemoryContainer<int>& A_innerColumnIndices,
			const MemoryContainer<int>& A_outerRowStarts,
			const MemoryContainer<FloatType>& x,
			MemoryContainer<FloatType>& y,
			MemoryContainer<FloatType>& oneVec,
			unsigned nParameters,
			FloatType lambda
		) {
			// Constant coefficients.
			const FloatType one{ 1.0 };
			const FloatType zero{ 0.0 };

			CUSPARSE_SAFE_CALL(cusparseXcsrmv(
				cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, nParameters, nParameters, A_values.getSize(), &one, A_desc,
				A_values.getData(Type2Type<MemoryTypeCUDA>()), A_outerRowStarts.getData(Type2Type<MemoryTypeCUDA>()), A_innerColumnIndices.getData(Type2Type<MemoryTypeCUDA>()),
				x.getData(Type2Type<MemoryTypeCUDA>()), &zero, y.getData(Type2Type<MemoryTypeCUDA>())
			));

			CUBLAS_SAFE_CALL(cublasXaxpy(cublasHandle, nParameters, &lambda, oneVec.getData(Type2Type<MemoryTypeCUDA>()), 1, y.getData(Type2Type<MemoryTypeCUDA>()), 1));
		}


		/**
		 * Applies sparse matrices J^T and J to the vector x and adds a lambda factor to the result, i.e. y = J^T (J x) + lambda I.
		 * Additional tmpVec and oneVec (should be already allocated) are needed for efficient computation.
		 */
		template<typename FloatType>
		inline void applyJTJ(
			cusparseHandle_t& cusparseHandle,
			cublasHandle_t& cublasHandle,
			cusparseMatDescr_t& jacobianDesc,
			const MemoryContainer<FloatType>& jacobianValues,
			const MemoryContainer<int>& jacobianInnerColumnIndices,
			const MemoryContainer<int>& jacobianOuterRowStarts,
			cusparseMatDescr_t& jacobianTDesc,
			const MemoryContainer<FloatType>& jacobianTValues,
			const MemoryContainer<int>& jacobianTInnerColumnIndices,
			const MemoryContainer<int>& jacobianTOuterRowStarts,
			const MemoryContainer<FloatType>& x,
			MemoryContainer<FloatType>& y,
			MemoryContainer<FloatType>& tmpVec,
			MemoryContainer<FloatType>& oneVec,
			unsigned nParameters,
			unsigned totalResidualSize,
			FloatType lambda
		) {
			// Constant coefficients.
			const FloatType one{ 1.0 };
			const FloatType zero{ 0.0 };
			const unsigned totalJacobianSize = jacobianValues.getSize();

			CUSPARSE_SAFE_CALL(cusparseXcsrmv(
				cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				totalResidualSize, nParameters, totalJacobianSize, &one, jacobianDesc,
				jacobianValues.getData(Type2Type<MemoryTypeCUDA>()), jacobianOuterRowStarts.getData(Type2Type<MemoryTypeCUDA>()), jacobianInnerColumnIndices.getData(Type2Type<MemoryTypeCUDA>()),
				x.getData(Type2Type<MemoryTypeCUDA>()), &zero, tmpVec.getData(Type2Type<MemoryTypeCUDA>())
			));

			CUSPARSE_SAFE_CALL(cusparseXcsrmv(
				cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				nParameters, totalResidualSize, totalJacobianSize, &one, jacobianTDesc,
				jacobianTValues.getData(Type2Type<MemoryTypeCUDA>()), jacobianTOuterRowStarts.getData(Type2Type<MemoryTypeCUDA>()), jacobianTInnerColumnIndices.getData(Type2Type<MemoryTypeCUDA>()),
				tmpVec.getData(Type2Type<MemoryTypeCUDA>()), &zero, y.getData(Type2Type<MemoryTypeCUDA>())
			));

			CUBLAS_SAFE_CALL(cublasXaxpy(cublasHandle, nParameters, &lambda, oneVec.getData(Type2Type<MemoryTypeCUDA>()), 1, y.getData(Type2Type<MemoryTypeCUDA>()), 1));
		}


		/**
		 * Applies sparse matrix J (as J^T J) to the vector x and adds a lambda factor to the result, i.e. y = J^T (J x) + lambda I.
		 * One matrix Ji should be provided for each constraint (as system componenets argument). Atomics are used for thread-safe 
		 * additions. Additional oneVec (should be already allocated and set to 1.0) is needed for efficient computation.
		 */
		template<typename FloatType>
		inline void applyJTJWithAtomics(
			cublasHandle_t& cublasHandle,
			vector<std::pair<ResidualVector<FloatType>, JacobianMatrix<FloatType>>>& systemComponents,
			DenseMatrix<int>& columnIndices,
			DenseMatrix<int>& constraintStarts,
			const MemoryContainer<FloatType>& x,
			MemoryContainer<FloatType>& y,
			MemoryContainer<FloatType>& oneVec,
			unsigned nParameters,
			FloatType lambda
		) {
			const unsigned nConstraints = systemComponents.size();

			// Reset the initial vector y to 0.
			cudaMemset(y.getData(Type2Type<MemoryTypeCUDA>()), 0, y.getByteSize());

			// Compute the y parts for each constraint.
			for (int i = 0; i < nConstraints; ++i) {
				auto& jacobianMatrix = systemComponents[i].second;
				const unsigned totalResidualDim = jacobianMatrix.getNumResiduals() * jacobianMatrix.getResidualDim();
				const unsigned totalParamDim = jacobianMatrix.getParamDim();

				applySparseJTJ<<< (totalResidualDim + JTJ_APPLICATION_BLOCK_SIZE - 1) / JTJ_APPLICATION_BLOCK_SIZE, JTJ_APPLICATION_BLOCK_SIZE >>>(
					jacobianMatrix.getData(Type2Type<MemoryTypeCUDA>()), columnIndices.getData(Type2Type<MemoryTypeCUDA>()), constraintStarts.getData(Type2Type<MemoryTypeCUDA>()),
					i, totalParamDim, totalResidualDim, x.getData(Type2Type<MemoryTypeCUDA>()), y.getData(Type2Type<MemoryTypeCUDA>())
				);
				CUDA_CHECK_ERROR();
			}

			// Add application of lambda * I to vector x.
			CUBLAS_SAFE_CALL(cublasXaxpy(cublasHandle, nParameters, &lambda, oneVec.getData(Type2Type<MemoryTypeCUDA>()), 1, y.getData(Type2Type<MemoryTypeCUDA>()), 1));
		}


		/**
		 * Computes gradient vector as g = J^T r, using atomic operations.
		 */
		template<typename FloatType>
		inline void computeGradientWithAtomics(
			vector<std::pair<ResidualVector<FloatType>, JacobianMatrix<FloatType>>>& systemComponents,
			DenseMatrix<int>& columnIndices,
			DenseMatrix<int>& constraintStarts,
			MemoryContainer<FloatType>& g
		) {
			const unsigned nConstraints = systemComponents.size();
			FloatType alpha = FloatType(1);

			// Reset the initial vector y to 0.
			cudaMemset(g.getData(Type2Type<MemoryTypeCUDA>()), 0, g.getByteSize());

			// Compute the r parts for each constraint.
			for (int i = 0; i < nConstraints; ++i) {
				auto& residualVector = systemComponents[i].first;
				auto& jacobianMatrix = systemComponents[i].second;
				const unsigned totalResidualDim = jacobianMatrix.getNumResiduals() * jacobianMatrix.getResidualDim();
				const unsigned totalParamDim = jacobianMatrix.getParamDim();

				applySparseJT<<< (totalResidualDim + JT_APPLICATION_BLOCK_SIZE - 1) / JT_APPLICATION_BLOCK_SIZE, JT_APPLICATION_BLOCK_SIZE >>>(
					jacobianMatrix.getData(Type2Type<MemoryTypeCUDA>()), 
					columnIndices.getData(Type2Type<MemoryTypeCUDA>()), 
					constraintStarts.getData(Type2Type<MemoryTypeCUDA>()),
					i, totalParamDim, totalResidualDim, alpha,
					residualVector.getData(Type2Type<MemoryTypeCUDA>()), 
					g.getData(Type2Type<MemoryTypeCUDA>())
				);
				CUDA_CHECK_ERROR();
			}
		}


		/**
		 * Helper function that executes PCG algorithm for a given number of iterations.
		 * The linear system solution is stored in the memory container x.
		 * @returns	The number of executed iterations.
		 */
		template<typename FloatType>
		inline int executePCGAlgorithm(
			cusparseHandle_t& cusparseHandle,
			cublasHandle_t& cublasHandle,
			vector<std::pair<ResidualVector<FloatType>, JacobianMatrix<FloatType>>>& systemComponents,
			DenseMatrix<int>& columnIndexVector,
			DenseMatrix<int>& rowIndexVector,
			const MemoryContainer<FloatType>& residualValues,
			cusparseMatDescr_t& jacobianDesc,
			const MemoryContainer<FloatType>& jacobianValues,
			const MemoryContainer<int>& jacobianInnerColumnIndices,
			const MemoryContainer<int>& jacobianOuterRowStarts,
			cusparseMatDescr_t& jacobianTDesc,
			const MemoryContainer<FloatType>& jacobianTValues,
			const MemoryContainer<int>& jacobianTInnerColumnIndices,
			const MemoryContainer<int>& jacobianTOuterRowStarts,
			cusparseMatDescr_t& A_desc,
			const MemoryContainer<FloatType>& A_values,
			const MemoryContainer<int>& A_innerColumnIndices,
			const MemoryContainer<int>& A_outerRowStarts,
			const MemoryContainer<FloatType>& diagonalPreconditioner,
			MemoryContainer<FloatType>& x,
			FloatType lambda,
			unsigned nParameters,
			unsigned totalJacobianSize,
			unsigned totalResidualSize,
			unsigned nMaxIterations,
			bool bUseQStoppingCriteria,
			float qTolerance,
			float rTolerance,
			JTJApplicationType applicationType
		) {
			// Execute PCG algorithm, as described at:
			// https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method

			TIME_GPU_START(PCGAlgorithm_TotalTime);

			// Constant coefficients.
			const FloatType one{ 1.0 };
			const FloatType minusOne{ -1.0 };
			const FloatType zero{ 0.0 };

			runtime_assert(rTolerance >= std::numeric_limits<FloatType>::epsilon(), "r-tolerance needs to be high enough for numerical stability.");

			TIME_GPU_START(PCGAlgorithm_ComputeRightSide);

			// First initialize the PCG algorithm.
			// Compute r0 = -J^T F.
			MemoryContainer<FloatType> r;
			r.allocate(nParameters, Type2Type<MemoryTypeCUDA>());

			if (applicationType == JTJApplicationType::COMPLETE || applicationType == JTJApplicationType::SEQUENTIAL)
				constructRightSide(
					cusparseHandle,
					jacobianTDesc, jacobianTValues, jacobianTInnerColumnIndices, jacobianTOuterRowStarts,
					residualValues, r,
					nParameters, totalResidualSize, totalJacobianSize
				);
			else if (applicationType == JTJApplicationType::ATOMIC)
				constructRightSideWithAtomics(
					systemComponents, columnIndexVector, rowIndexVector, r
				);
			else
				throw std::runtime_error("Unsupported JTJ application type.");

			runtime_assert(std::isfinite(computeSquaredLoss(cublasHandle, r)), "Right side of the system (-J^T r) is not finite: " + std::to_string(computeSquaredLoss(cublasHandle, r)));

			FloatType rInitialNormSquared = FloatType(0);
			CUBLAS_SAFE_CALL(cublasXdot(cublasHandle, nParameters, r.getData(Type2Type<MemoryTypeCUDA>()), 1, r.getData(Type2Type<MemoryTypeCUDA>()), 1, &rInitialNormSquared));

			if (rInitialNormSquared == 0) {
				return 0;
			}

			FloatType threshold = rInitialNormSquared * rTolerance * rTolerance;
		
			TIME_GPU_STOP(PCGAlgorithm_ComputeRightSide);
			TIME_GPU_START(PCGAlgorithm_Initialize);

			// Constant vector one.
			MemoryContainer<FloatType> oneVec;
			oneVec.allocate(nParameters, Type2Type<MemoryTypeCUDA>());
			setOneVector<<< (nParameters + 255) / 256, 256 >>>(
				oneVec.getData(Type2Type<MemoryTypeCUDA>()),
				nParameters
			);

			// Preparation for q stopping criteria.
			MemoryContainer<FloatType> b;
			b.allocate(nParameters, Type2Type<MemoryTypeCUDA>());
			CUDA_SAFE_CALL(cudaMemcpy(b.getData(Type2Type<MemoryTypeCUDA>()), r.getData(Type2Type<MemoryTypeCUDA>()), r.getByteSize(), cudaMemcpyDeviceToDevice));

			MemoryContainer<FloatType> rCurrentPlusB;
			rCurrentPlusB.allocate(nParameters, Type2Type<MemoryTypeCUDA>());

			// Helper vectors.
			MemoryContainer<FloatType> g;
			g.allocate(nParameters, Type2Type<MemoryTypeCUDA>());

			MemoryContainer<FloatType> z;
			z.allocate(nParameters, Type2Type<MemoryTypeCUDA>());

			MemoryContainer<FloatType> tmpVec;
			tmpVec.allocate(totalResidualSize, Type2Type<MemoryTypeCUDA>());

			// Descent step in k-th iteration.
			// p0 = M^-1 r0
			MemoryContainer<FloatType> p;
			p.allocate(nParameters, Type2Type<MemoryTypeCUDA>());
			elementWiseMultiplication<<< (nParameters + 255) / 256, 256 >>>(
				diagonalPreconditioner.getData(Type2Type<MemoryTypeCUDA>()),
				r.getData(Type2Type<MemoryTypeCUDA>()),
				p.getData(Type2Type<MemoryTypeCUDA>()),
				nParameters
			);
			CUDA_CHECK_ERROR();
			runtime_assert(std::isfinite(computeSquaredLoss(cublasHandle, p)), "Vector p is not finite: " + std::to_string(computeSquaredLoss(cublasHandle, p)));

			// Step size in k-th iteration.
			FloatType alpha{ 0.0 };

			// Helper coefficients.
			FloatType beta{ 0.0 };
			FloatType gamma{ 0.0 };
			FloatType delta{ 0.0 };
			FloatType omega{ 0.0 };

			// Stopping criteria coefficient q.
			FloatType qPrev{ 0.f };

			// gamma_0 = r0^T p0
			CUBLAS_SAFE_CALL(cublasXdot(cublasHandle, nParameters, r.getData(Type2Type<MemoryTypeCUDA>()), 1, p.getData(Type2Type<MemoryTypeCUDA>()), 1, &gamma));
			runtime_assert(std::isfinite(gamma), "Gamma value not finite: " + std::to_string(gamma));

			TIME_GPU_STOP(PCGAlgorithm_Initialize);

			// Execute PCG algorithm.
			int nIterations = 0;
			for (int i = 0; i < nMaxIterations; ++i) {
				TIME_GPU_START(PCGAlgorithm_OneIteration);

				// g_k = J^T J p_k
				if (applicationType == JTJApplicationType::COMPLETE)
					applyA(cusparseHandle, cublasHandle, A_desc, A_values, A_innerColumnIndices, A_outerRowStarts, p, g, oneVec, nParameters, lambda);
				else if (applicationType == JTJApplicationType::SEQUENTIAL)
					applyJTJ(
						cusparseHandle, cublasHandle,
						jacobianDesc, jacobianValues, jacobianInnerColumnIndices, jacobianOuterRowStarts,
						jacobianTDesc, jacobianTValues, jacobianTInnerColumnIndices, jacobianTOuterRowStarts,
						p, g, tmpVec, oneVec, nParameters, totalResidualSize, lambda
					);
				else if (applicationType == JTJApplicationType::ATOMIC)
					applyJTJWithAtomics(
						cublasHandle,
						systemComponents, columnIndexVector, rowIndexVector,
						p, g, oneVec,
						nParameters, lambda
					);
				else
					throw std::runtime_error("Unsupported JTJ application type.");

				// delta_k = p_k^T g_k
				CUBLAS_SAFE_CALL(cublasXdot(cublasHandle, nParameters, p.getData(Type2Type<MemoryTypeCUDA>()), 1, g.getData(Type2Type<MemoryTypeCUDA>()), 1, &delta));
				runtime_assert(std::isfinite(delta), "Delta value not finite: " + std::to_string(delta));

				alpha = gamma / delta;
				runtime_assert(std::isfinite(alpha), "Alpha value not finite: " + std::to_string(alpha));

				// x_(k+1) = x_k + alpha_k * p_k
				CUBLAS_SAFE_CALL(cublasXaxpy(cublasHandle, nParameters, &alpha, p.getData(Type2Type<MemoryTypeCUDA>()), 1, x.getData(Type2Type<MemoryTypeCUDA>()), 1));

				// r_(k+1) = r_k - alpha_k * g_k
				alpha = -alpha;
				CUBLAS_SAFE_CALL(cublasXaxpy(cublasHandle, nParameters, &alpha, g.getData(Type2Type<MemoryTypeCUDA>()), 1, r.getData(Type2Type<MemoryTypeCUDA>()), 1));
				alpha = -alpha;

				// Update the iteration counter.
				nIterations++;

				// Check if squared residual norm is low enough, to return early.
				// We check: |Ax - b| / |b| < rTolerance => Convergence!
				FloatType rNormSquared = FloatType(0);
				CUBLAS_SAFE_CALL(cublasXdot(cublasHandle, nParameters, r.getData(Type2Type<MemoryTypeCUDA>()), 1, r.getData(Type2Type<MemoryTypeCUDA>()), 1, &rNormSquared));

				if (rNormSquared < threshold) {
					break;
				}

				// Another stopping criteria: q number.
				// q = 0.5 x^T A x = 0.5 x^T (r + b)
				// We check: |(q - qPrev) / q| < qTolerance => Convergence!
				CUDA_SAFE_CALL(cudaMemcpy(rCurrentPlusB.getData(Type2Type<MemoryTypeCUDA>()), r.getData(Type2Type<MemoryTypeCUDA>()), r.getByteSize(), cudaMemcpyDeviceToDevice));
				CUBLAS_SAFE_CALL(cublasXaxpy(cublasHandle, nParameters, &one, b.getData(Type2Type<MemoryTypeCUDA>()), 1, rCurrentPlusB.getData(Type2Type<MemoryTypeCUDA>()), 1));
				FloatType q{ 0.f };
				CUBLAS_SAFE_CALL(cublasXdot(cublasHandle, nParameters, x.getData(Type2Type<MemoryTypeCUDA>()), 1, rCurrentPlusB.getData(Type2Type<MemoryTypeCUDA>()), 1, &q));
				q = 0.5 * q;

				if (bUseQStoppingCriteria && (q - qPrev) < FloatType(qTolerance) * q) {
					break;
				}

				qPrev = q;
				
				// z_(k+1) = M^-1 r_(k+1)
				elementWiseMultiplication<<< (nParameters + 255) / 256, 256 >>>(
					diagonalPreconditioner.getData(Type2Type<MemoryTypeCUDA>()),
					r.getData(Type2Type<MemoryTypeCUDA>()),
					z.getData(Type2Type<MemoryTypeCUDA>()),
					nParameters
				);
				CUDA_CHECK_ERROR();

				// omega_k = z_(k+1)^T r_(k+1)
				CUBLAS_SAFE_CALL(cublasXdot(cublasHandle, nParameters, z.getData(Type2Type<MemoryTypeCUDA>()), 1, r.getData(Type2Type<MemoryTypeCUDA>()), 1, &omega));
				runtime_assert(std::isfinite(omega), "Omega value not finite: " + std::to_string(omega));

				beta = omega / gamma;
				runtime_assert(std::isfinite(beta), "Beta value not finite: " + std::to_string(beta));

				// p_(k+1) = z_(k+1) + beta_k * p_k
				CUBLAS_SAFE_CALL(cublasXscal(cublasHandle, nParameters, &beta, p.getData(Type2Type<MemoryTypeCUDA>()), 1));												// p_k' = beta_k * p_k
				CUBLAS_SAFE_CALL(cublasXaxpy(cublasHandle, nParameters, &one, z.getData(Type2Type<MemoryTypeCUDA>()), 1, p.getData(Type2Type<MemoryTypeCUDA>()), 1));	// p_(k+1) = z_(k+1) + p_k'
				
				// gamma_(k+1) = omega_k
				gamma = omega;

				TIME_GPU_STOP(PCGAlgorithm_OneIteration);
			}

			TIME_GPU_STOP(PCGAlgorithm_TotalTime);

			return nIterations;
		}

	} // namespace  system_proc
} // namespace solo