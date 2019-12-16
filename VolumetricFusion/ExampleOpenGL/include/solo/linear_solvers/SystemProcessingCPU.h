#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "Solo/data_structures/ResidualVectorInterface.h"
#include "Solo/data_structures/JacobianMatrixInterface.h"

namespace solo {
	namespace  system_proc {
		
		/**
		 * Takes care for updating the host memory of residual vectors.
		 */
		template<typename FloatType>
		inline void updateResidualMemory(vector<ResidualVector<FloatType>>& residuals) {
			const unsigned nConstraints = residuals.size();

			// Check if the copy to host (CPU) memory is needed.
#			ifdef COMPILE_CUDA
			for (int i = 0; i < nConstraints; ++i) {
				auto& residualVectorContainer = residuals[i].getContainer();
				if (residualVectorContainer.isUpdatedDevice() && !residualVectorContainer.isUpdatedHost()) {
					residualVectorContainer.copyDeviceToHost();
					residualVectorContainer.setUpdated(true, true);
				}
			}
#			endif
		}


		/**
		 * Takes care for updating the host memory of residual and Jacobian matrices.
		 */
		template<typename FloatType>
		inline void updateResidualAndJacobianMemory(vector<std::pair<ResidualVector<FloatType>, JacobianMatrix<FloatType>>>& systemComponents) {
			const unsigned nConstraints = systemComponents.size();

			// Check if the copy to host (CPU) memory is needed.
#			ifdef COMPILE_CUDA
			for (int i = 0; i < nConstraints; ++i) {
				auto& residualVectorContainer = systemComponents[i].first.getContainer();
				if (residualVectorContainer.isUpdatedDevice() && !residualVectorContainer.isUpdatedHost()) {
					residualVectorContainer.copyDeviceToHost();
					residualVectorContainer.setUpdated(true, true);
				}

				auto& jacobianMatrixContainer = systemComponents[i].second.getContainer();
				if (jacobianMatrixContainer.isUpdatedDevice() && !jacobianMatrixContainer.isUpdatedHost()) {
					jacobianMatrixContainer.copyDeviceToHost();
					jacobianMatrixContainer.setUpdated(true, true);
				}
			}
#			endif
		}


		/**
		 * Helper method to prepare a complete (consisting of all constraints) dense residual vector.
		 */
		template<typename FloatType>
		void prepareResidualVector(
			vector<std::pair<ResidualVector<FloatType>, JacobianMatrix<FloatType>>>& systemComponents,
			DenseMatrix<FloatType>& residualValues,
			unsigned totalResidualSize
		) {
			const unsigned nConstraints = systemComponents.size();
			residualValues.allocate(totalResidualSize, 1);

			unsigned residualMemoryOffset = 0;
			for (int i = 0; i < nConstraints; ++i) {
				auto& residualVector = systemComponents[i].first;
				const unsigned residualVectorSize = residualVector.getSize();
				memcpy(residualValues.getData(Type2Type<MemoryTypeCPU>()) + residualMemoryOffset, residualVector.getData(Type2Type<MemoryTypeCPU>()), residualVectorSize * sizeof(FloatType));
				residualMemoryOffset += residualVectorSize;
			}
		}

		template<typename FloatType>
		void prepareResidualVector(
			vector<ResidualVector<FloatType>>& residuals,
			DenseMatrix<FloatType>& residualValues,
			unsigned totalResidualSize
		) {
			const unsigned nConstraints = residuals.size();
			residualValues.allocate(totalResidualSize, 1);

			unsigned residualMemoryOffset = 0;
			for (int i = 0; i < nConstraints; ++i) {
				auto& residualVector = residuals[i];
				const unsigned residualVectorSize = residualVector.getSize();
				memcpy(residualValues.getData(Type2Type<MemoryTypeCPU>()) + residualMemoryOffset, residualVector.getData(Type2Type<MemoryTypeCPU>()), residualVectorSize * sizeof(FloatType));
				residualMemoryOffset += residualVectorSize;
			}
		}


		/**
		 * Prepares Eigen matrices, necessary for sparse linear system solving. The parameters are
		 * assumed to be ordered increasing, according to their indices.
		 * Returns the sparse matrix A = J^T J and dense vector b = J^T r, computed using sparse
		 * matrix operations.
		 */
		template<typename FloatType>
		void prepareSparseSystemEigenOrdered(
			vector<std::pair<ResidualVector<FloatType>, JacobianMatrix<FloatType>>>& systemComponents,
			DenseMatrix<FloatType>& residualValues,
			DenseMatrix<FloatType>& jacobianValues
		) {
			const unsigned nConstraints = systemComponents.size();
			updateResidualAndJacobianMemory(systemComponents);			

			// We have different (varying) parameter indices, therefore we compute J^T J using sparse 
			// matrices. We use compressed sparse row (CSR) format. Since input Jacobian matrices are 
			// stored in memory column-wise, we have to tranpose it first. It's is important that we
			// use default column-major Eigen matrices.

			// We compute the total required memory sizes.
			unsigned totalResidualSize = 0;
			unsigned totalJacobianSize = 0;
			for (int i = 0; i < nConstraints; ++i) {
				auto& residualVector = systemComponents[i].first;
				totalResidualSize += residualVector.getSize();

				auto& jacobianMatrix = systemComponents[i].second;
				totalJacobianSize += jacobianMatrix.getSize();
			}

			// We copy all residuals and Jacobian matrices into a common contiguous memory.
			prepareResidualVector(systemComponents, residualValues, totalResidualSize);
			
			jacobianValues.allocate(totalJacobianSize, 1);
			unsigned jacobianMemoryOffset = 0;

			for (int i = 0; i < nConstraints; ++i) {
				// We transpose the Jacobian matrices (to get the correct alignment in the memory).
				auto& jacobianMatrix = systemComponents[i].second;
				Eigen::Map<Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic>> jacobianMap(
					jacobianMatrix.getData(Type2Type<MemoryTypeCPU>()), jacobianMatrix.mat().rows(), jacobianMatrix.mat().cols()
				);
				Eigen::Map<Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic>> jacobianTranposedMap(
					jacobianValues.getData(Type2Type<MemoryTypeCPU>()) + jacobianMemoryOffset, jacobianMatrix.mat().cols(), jacobianMatrix.mat().rows()
				);
				jacobianTranposedMap = jacobianMap.transpose();

				const unsigned jacobianMatrixSize = jacobianMatrix.getSize();
				jacobianMemoryOffset += jacobianMatrixSize;
			}
		}


		/**
		 * Prepares Eigen matrices, necessary for sparse linear system solving. The parameters don't
		 * need to be in a particular order, they will get sorted.
		 * Returns the sparse matrix A = J^T J and dense vector b = J^T r, computed using sparse
		 * matrix operations.
		 */
		template<typename FloatType>
		void prepareSparseSystemEigenNonOrdered(
			vector<std::pair<ResidualVector<FloatType>, JacobianMatrix<FloatType>>>& systemComponents,
			DenseMatrix<FloatType>& residualValues,
			const DenseMatrix<int>& jacobianInnerColumnIndices,
			Eigen::SparseMatrix<FloatType, Eigen::RowMajor, int>& jacobianSparseMatrix
		) {
			const unsigned nConstraints = systemComponents.size();
			updateResidualAndJacobianMemory(systemComponents);

			// We have different (varying) parameter indices, therefore we compute J^T J using sparse 
			// matrices. We use compressed sparse row (CSR) format. Since input Jacobian matrices are 
			// stored in memory column-wise, we have to tranpose it first. It's is important that we
			// use default column-major Eigen matrices.

			// We compute the total required memory size for residual vector.
			unsigned totalResidualSize = 0;
			for (int i = 0; i < nConstraints; ++i) {
				auto& residualVector = systemComponents[i].first;
				totalResidualSize += residualVector.getSize();
			}

			// We compute the column width for each row of the Jacobian matrix, to enable efficient 
			// element insertion.
			vector<int> columnWidths(totalResidualSize);
			unsigned nProcessedRows{ 0 };
			for (int i = 0; i < nConstraints; ++i) {
				auto& jacobianMatrix = systemComponents[i].second;
				const unsigned nRows = jacobianMatrix.getNumResiduals() * jacobianMatrix.getResidualDim();
				const unsigned nCols = jacobianMatrix.getParamDim();

				#pragma omp parallel for
				for (int j = 0; j < nRows; ++j) {
					columnWidths[nProcessedRows + j] = nCols;
				}

				nProcessedRows += nRows;			
			}

			// We copy all residuals and Jacobian matrices into a common contiguous memory.
			prepareResidualVector(systemComponents, residualValues, totalResidualSize);

			jacobianSparseMatrix.reserve(columnWidths);
			unsigned jacobianInnerColumnIndex{ 0 };
			unsigned jacobianRowIndex{ 0 };

			for (int i = 0; i < nConstraints; ++i) {
				DenseMatrix<FloatType>& jacobianMatrix = systemComponents[i].second.mat();
				DenseMatrixInterface<FloatType, MemoryTypeCPU> iJacobianMatrix{ jacobianMatrix };

				const unsigned nRows = jacobianMatrix.rows();
				const unsigned nCols = jacobianMatrix.cols();

				for (int y = 0; y < nRows; ++y) {
					for (int x = 0; x < nCols; ++x) {
						int columnIndex = jacobianInnerColumnIndices(jacobianInnerColumnIndex, 0);
						jacobianSparseMatrix.insert(jacobianRowIndex, columnIndex) = iJacobianMatrix(y, x);
						jacobianInnerColumnIndex++;
					}
					jacobianRowIndex++;
				}
			}
		}

	} // namespace system_proc
} // namespace solo