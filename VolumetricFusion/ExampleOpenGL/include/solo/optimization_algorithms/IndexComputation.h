#pragma once
#include <algorithm>
#include <common_utils/timing/TimerCPU.h>

#include "solo/data_structures/DenseMatrix.h"
#include "solo/constraint_evaluation/Constraint.h"
#include "solo/optimization_algorithms/SolverProcessing.h"
#include "solo/optimization_algorithms/IndexProcessing.h"

namespace solo {

	/**
	 * Index computation interface.
	 */
	class IndexComputation {
	public:
		/**
		 * Computes the column index vector that stores the inner column indices of all parameters and 
		 * outer row starting indices for every residual.
		 * 
		 * Our final Jacobian matrix will have one row for each residual and the number of non-zero column 
		 * elements corresponds of each row to the number of paramaters of the corresponding residual. We
		 * use CSR (compressed sparse rows) matrix representation, which means that we need to compute unique
		 * indices for all parameters, and store for each residual only the indices of parameters that are
		 * present in the residual. Additionally, we need the vector of starting indices for each residual,
		 * componentIdx.e. at which memory location a new row (residual) starts. That also carries information about 
		 * the number of parameters in each residual.
		 * 
		 * @returns The number of parameter components in the optimization system.
		 */
		template<typename IndexStorageType, typename ...Constraints>
		static unsigned computeIndexVectors(
			Type2Type<IndexStorageType>, 
			bool bFilterUnusedParams,
			bool bComputeUsingGPU,
			DenseMatrix<int>& columnIndexVector,
			DenseMatrix<int>& rowIndexVector,
			vector<int>& indexMapping, 
			Constraints&&... constraints
		) {
			unsigned nParameters = computeColumnIndexVector(
				Type2Type<IndexStorageType>(), bFilterUnusedParams, bComputeUsingGPU, columnIndexVector, indexMapping, std::forward<Constraints>(constraints)...
			);
			unsigned nResiduals = computerRowIndexVector(
				Type2Type<IndexStorageType>(), rowIndexVector, std::forward<Constraints>(constraints)...
			);

			return nParameters;
		}

		/**
		 * Computes the column index vector that stores an index (id) for every parameter component. The index
		 * vector has dimension (size, 1), where size is the number of all parameter components in all residuals
		 * in all constraints.
		 * The parameters are indexed considering their order in memory, with setting the lowest index to the 
		 * parameter with earliest position in memory. 
		 * There are two types of storage in memory:
		 * - row-wise: First indices of first residual are listed, then indices of second residual, etc.
		 * - col-wise: A little more complicated, first indices of first constraint are listed, in column-
		 *			   major storage, then the indices of second constraint, also in column-major storage,
		 *			   etc.
		 * @returns The number of parameter components in the optimization system.
		 */
		template<typename IndexStorageType, typename ...Constraints>
		static unsigned computeColumnIndexVector(
			Type2Type<IndexStorageType>, 
			bool bFilterUnusedParams, 
			bool bComputeUsingGPU,
			DenseMatrix<int>& columnIndexVector, 
			vector<int>& indexMapping, 
			Constraints&&... constraints
		) {
			TIME_CPU_START(ComputeColumnIndexVector_Initialization);

			// Compute the size of the column index vector and total number of parameter pointers.
			const unsigned jacobianSize = computeNumJacobianEntries(std::forward<Constraints>(constraints)...);
			const unsigned nIndices = computeTotalNumIndices(std::forward<Constraints>(constraints)...);

			TIME_CPU_STOP(ComputeColumnIndexVector_Initialization);

			if (nIndices == 0) {
				runtime_assert(jacobianSize == 0, "We should have zero indices only if no constraints are present.");
				return 0;
			}

			MemoryContainer<int> mapSparseToDenseIndex;
			if (bFilterUnusedParams) {
				TIME_CPU_START(ComputeColumnIndexVector_PrepareIndices);

				// Copy the indices into one contiguous vector.
				vector<int> paramIndices(nIndices);
				joinIndexMatrices(0, paramIndices, std::forward<Constraints>(constraints)...);

				TIME_CPU_STOP(ComputeColumnIndexVector_PrepareIndices);
				TIME_CPU_START(ComputeColumnIndexVector_SortIndices);

				std::sort(paramIndices.begin(), paramIndices.end());

				TIME_CPU_STOP(ComputeColumnIndexVector_SortIndices);
				TIME_CPU_START(ComputeColumnIndexVector_ComputeMaxIndex);

				auto maxIndexIter = std::max_element(paramIndices.begin(), paramIndices.end());
				int maxIndex = *maxIndexIter;

				TIME_CPU_STOP(ComputeColumnIndexVector_ComputeMaxIndex);
				TIME_CPU_START(ComputeColumnIndexVector_ComputeIndices);

				// Generate starting index for each parameter index (we need to take care for repeating indices).
				mapSparseToDenseIndex.allocate(maxIndex + 1);
				indexMapping.clear();

				int denseIndex{ 0 };
				int previousIndex{ -1 };
				for (int i = 0; i < nIndices; ++i) {
					int currentIndex = paramIndices[i];
					if (currentIndex != previousIndex) {
						// We got a new unique index.
						mapSparseToDenseIndex.getElement(currentIndex, Type2Type<MemoryTypeCPU>()) = denseIndex;
						indexMapping.push_back(currentIndex);
						denseIndex++;
						previousIndex = currentIndex;
					}
				}

				mapSparseToDenseIndex.setUpdated(true, false);

				TIME_CPU_STOP(ComputeColumnIndexVector_ComputeIndices);
			}

			TIME_CPU_START(ComputeColumnIndexVector_FillIndices);

			// Compute the indices of all parameter components.
			if (bComputeUsingGPU) {
#				ifdef COMPILE_CUDA
				columnIndexVector.allocate(jacobianSize, 1, Type2Type<MemoryTypeCUDA>());
#				else
				runtime_assert(false, "COMPILE_CUDA flag needs to be enabled to use GPU index vector computation.");
#				endif
			}
			else {
				columnIndexVector.allocate(jacobianSize, 1, Type2Type<MemoryTypeCPU>());
			}

			fillColumnIndices(Type2Type<IndexStorageType>(), mapSparseToDenseIndex, columnIndexVector, 0, bComputeUsingGPU, std::forward<Constraints>(constraints)...);

			TIME_CPU_STOP(ComputeColumnIndexVector_FillIndices);
			TIME_CPU_START(ComputeColumnIndexVector_ComputeNumParameters);

			unsigned nParameters{ 0 };
			if (bComputeUsingGPU) {
#				ifdef COMPILE_CUDA
				nParameters = computeMaxIndexGPU(columnIndexVector) + 1;
#				else
				runtime_assert(false, "COMPILE_CUDA flag needs to be enabled to use GPU index vector computation.");
#				endif
			} 
			else {
				nParameters = computeMaxIndexCPU(columnIndexVector) + 1;
			}

			TIME_CPU_STOP(ComputeColumnIndexVector_ComputeNumParameters);

			return nParameters;
		}

		/**
		 * Computes the row index vector. 
		 * There are two versions:
		 * - row-wise: It stores an index for every residual that corresponds to the memory location, 
		 *			   where a new row (residual) starts. The index vector has dimension (size + 1, 1), 
		 *			   where size is the number of all residuals in all constraints. The last component 
		 *			   tells the number of the nonzero elements.
		 * - col-wise: It stores the offsets of column index vector for every constraint. The index 
		 *			   vector has dimension (nConstraints + 1, 1). The first element is 0, the next
		 *			   element is the number of total number of column indices for the first constraint,
		 *			   etc. The last element tells the size of the column index vector.
		 * @returns The number of all residuals (if row-wise) or the number of constraint (if col-wise)
		 *			in the optimization system.
		 */
		template<typename ...Constraints>
		static unsigned computerRowIndexVector(Type2Type<RowWiseStorage>, DenseMatrix<int>& rowIndexVector, Constraints&&... constraints) {
			const unsigned size = computeNumResiduals(std::forward<Constraints>(constraints)...);
			rowIndexVector.allocate(size + 1, 1);

			DenseMatrixInterface<int, MemoryTypeCPU> iRowIndexVector{ rowIndexVector };

			// Initialize the outer start index and the number of processed residuals.
			int outerStart = 0;
			unsigned nProcessed = 0;

			// Compute the starting row indices.
			computeOuterRowStarts(outerStart, nProcessed, iRowIndexVector, std::forward<Constraints>(constraints)...);

			// Set the updated flag.
			rowIndexVector.getContainer().setUpdated(true, false);

			return size;
		}

		template<typename ...Constraints>
		static unsigned computerRowIndexVector(Type2Type<ColumnWiseStorage>, DenseMatrix<int>& rowIndexVector, Constraints&&... constraints) {
			const unsigned size = solver_proc::computeNumUsedConstraints(std::forward<Constraints>(constraints)...);
			rowIndexVector.allocate(size + 1, 1, Type2Type<MemoryTypeCPU>());

			DenseMatrixInterface<int, MemoryTypeCPU> iRowIndexVector{ rowIndexVector };

			// Initialize the index offset and the number of processed constraints.
			int indexOffset = 0;
			unsigned nProcessed = 0;

			// Compute the column index offsets.
			computeColumnIndexOffset(indexOffset, nProcessed, iRowIndexVector, std::forward<Constraints>(constraints)...);

			// Set the updated flag.
			rowIndexVector.getContainer().setUpdated(true, false);

			return size;
		}

	private:
		/**
		 * Computes the total number of parameters (by looping through all constraints).
		 */
		template<typename FloatType, typename CostFunction, typename LocalData, typename GlobalData, unsigned NumBlocks, typename ...Constraints>
		static unsigned computeNumJacobianEntries(Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>& constraint, Constraints&&... constraints) {
			using CostFunctionSignature = typename Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>::CostFunctionInterfaceSignature;

			if (constraint.isUsedInOptimization())
				return constraint.getNumResiduals() * GetResidualDim<CostFunctionSignature>::value * GetTotalParamDim<CostFunctionSignature>::value + 
				computeNumJacobianEntries(std::forward<Constraints>(constraints)...);
			else 
				return computeNumJacobianEntries(std::forward<Constraints>(constraints)...);
		}

		static unsigned computeNumJacobianEntries() {
			return 0;
		}


		template<typename FloatType, typename CostFunction, typename LocalData, typename GlobalData, unsigned NumBlocks, typename ...OtherConstraints>
		static unsigned computeTotalNumIndices(Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>& constraint, OtherConstraints&&... otherConstraints) {
			using CostFunctionSignature = typename Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>::CostFunctionInterfaceSignature;

			if (constraint.isUsedInOptimization())
				return constraint.getIndexMatrix().getSize() + computeTotalNumIndices(std::forward<OtherConstraints>(otherConstraints)...);
			else
				return computeTotalNumIndices(std::forward<OtherConstraints>(otherConstraints)...);
		}

		static unsigned computeTotalNumIndices() {
			return 0;
		}

		/**
		 * Computes the total number of residuals (by looping through all constraints).
		 */
		template<typename FloatType, typename CostFunction, typename LocalData, typename GlobalData, unsigned NumBlocks, typename ...Constraints>
		static unsigned computeNumResiduals(Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>& constraint, Constraints&&... constraints) {
			using CostFunctionSignature = typename Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>::CostFunctionInterfaceSignature;

			if (constraint.isUsedInOptimization())
				return constraint.getNumResiduals() * GetResidualDim<CostFunctionSignature>::value + 
				computeNumResiduals(std::forward<Constraints>(constraints)...);
			else
				return computeNumResiduals(std::forward<Constraints>(constraints)...);
		}

		static unsigned computeNumResiduals() {
			return 0;
		}

		/**
		 * Copies index matrices into one global index array (in host/CPU memory).
		 */
		template<typename FloatType, typename CostFunction, typename LocalData, typename GlobalData, unsigned NumBlocks, typename ...OtherConstraints>
		static void joinIndexMatrices(
			unsigned offset,
			vector<int>& joinedParamIndices,
			Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>& constraint,
			OtherConstraints&&... otherConstraints
		) {
			using CostFunctionSignature = typename Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>::CostFunctionInterfaceSignature;

			if (constraint.isUsedInOptimization()) {
				DenseMatrixWrapper<int>& indexMatrixForConstraint = constraint.getIndexMatrix();
				if (!indexMatrixForConstraint.getWrapper().isUpdatedHost()) {
#					ifdef COMPILE_CUDA
					indexMatrixForConstraint.getWrapper().copyDeviceToHost();
					indexMatrixForConstraint.getWrapper().setUpdated(true, true);
#					endif
				}

				memcpy(joinedParamIndices.data() + offset, indexMatrixForConstraint.getData(Type2Type<MemoryTypeCPU>()), indexMatrixForConstraint.getSize() * sizeof(int));
	
				joinIndexMatrices(offset + indexMatrixForConstraint.getSize(), joinedParamIndices, std::forward<OtherConstraints>(otherConstraints)...);
			}
			else {
				joinIndexMatrices(offset, joinedParamIndices, std::forward<OtherConstraints>(otherConstraints)...);
			}
		}

		static void joinIndexMatrices(
			unsigned offset,
			vector<int>& joinedParamIndices
		) { }

		/**
		 * Helper methods for column index computation.
		 */
		template<typename FloatType, typename IndexStorageType, typename CostFunction, typename LocalData, typename GlobalData, unsigned NumBlocks, typename ...OtherConstraints>
		static void fillColumnIndices(
			Type2Type<IndexStorageType>,
			MemoryContainer<int>& mapSparseToDenseIndex,
			DenseMatrix<int>& indexVector,
			unsigned indexVectorOffset,
			bool bComputeUsingGPU,
			Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>& constraint,
			OtherConstraints&&... otherConstraints
		) {
			using CostFunctionSignature = typename Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>::CostFunctionInterfaceSignature;

			if (constraint.isUsedInOptimization()) {
				const unsigned nResiduals = constraint.getNumResiduals();
				const unsigned residualDim = constraint.getResidualDim();
				const unsigned totalParamDim = constraint.getTotalParamDim();

				DenseMatrixWrapper<int>& indexMatrix = constraint.getIndexMatrix();
				if (bComputeUsingGPU) {
#					ifdef COMPILE_CUDA
					computeColumnIndexVectorPerConstraintGPU(
						Type2Type<IndexStorageType>(),
						mapSparseToDenseIndex,
						indexVector,
						indexVectorOffset,
						indexMatrix,
						residualDim,
						nResiduals,
						totalParamDim
					);
#					else
					runtime_assert(false, "Add COMPILE_CUDA flag to use GPU methods.");
#					endif
				}
				else {
					computeColumnIndexVectorPerConstraintCPU(
						Type2Type<IndexStorageType>(),
						mapSparseToDenseIndex,
						indexVector,
						indexVectorOffset,
						indexMatrix,
						residualDim,
						nResiduals,
						totalParamDim
					);
				}

				indexVectorOffset += nResiduals * residualDim * totalParamDim;
			}

			fillColumnIndices(
				Type2Type<IndexStorageType>(), mapSparseToDenseIndex, indexVector, indexVectorOffset, bComputeUsingGPU, std::forward<OtherConstraints>(otherConstraints)...
			);
		}

		template<typename IndexStorageType>
		static void fillColumnIndices(
			Type2Type<IndexStorageType>,
			MemoryContainer<int>& mapSparseToDenseIndex,
			DenseMatrix<int>& indexVector,
			unsigned indexVectorOffset,
			bool bComputeUsingGPU
		) { }

		/**
		 * Computes the row index vector for the current constraint.
		 */
		template<typename FloatType, typename CostFunction, typename LocalData, typename GlobalData, unsigned NumBlocks, typename ...Constraints>
		static void computeOuterRowStarts(
			int& outerStart,
			unsigned& nProcessed,
			DenseMatrixInterface<int, MemoryTypeCPU>& rowIndexVector,
			Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>& constraint,
			Constraints&&... otherConstraints
		) {
			using CostFunctionSignature = typename Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>::CostFunctionInterfaceSignature;

			if (constraint.isUsedInOptimization()) {
				const unsigned nResiduals = constraint.getNumResiduals();

				// We generate a correct outer start for each residual in the constraint.
				#pragma omp parallel for
				for (int i = 0; i < nResiduals; ++i) {
					// For multi-dimensional residuals, we need to add one index per each residual dimension.
					for (int j = 0; j < GetResidualDim<CostFunctionSignature>::value; ++j) {
						rowIndexVector(nProcessed + i * GetResidualDim<CostFunctionSignature>::value + j, 0) = outerStart +
							i * GetResidualDim<CostFunctionSignature>::value * GetTotalParamDim<CostFunctionSignature>::value +
							j * GetTotalParamDim<CostFunctionSignature>::value;
					}
				}
				outerStart += nResiduals * GetResidualDim<CostFunctionSignature>::value * GetTotalParamDim<CostFunctionSignature>::value;
				nProcessed += nResiduals * GetResidualDim<CostFunctionSignature>::value;
			}

			// Recursively we process all the rest of the constraints.
			computeOuterRowStarts(outerStart, nProcessed, rowIndexVector, std::forward<Constraints>(otherConstraints)...);
		}

		static void computeOuterRowStarts(
			int& outerStart,
			unsigned& nProcessed,
			DenseMatrixInterface<int, MemoryTypeCPU>& rowIndexVector
		) {
			// We note down the number of all parameter components, which is the same as the most
			// recent outer row start, which points to the index of the last value + 1.
			rowIndexVector(nProcessed, 0) = outerStart;
		}

		/**
		 * Computes the column index offsest for the current constraint.
		 */
		template<typename FloatType, typename CostFunction, typename LocalData, typename GlobalData, unsigned NumBlocks, typename ...Constraints>
		static void computeColumnIndexOffset(
			int& indexOffset,
			unsigned& nProcessed,
			DenseMatrixInterface<int, MemoryTypeCPU>& rowIndexVector,
			Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>& constraint,
			Constraints&&... otherConstraints
		) {
			using CostFunctionSignature = typename Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>::CostFunctionInterfaceSignature;

			if (constraint.isUsedInOptimization()) {
				const unsigned nResiduals = constraint.getNumResiduals();

				rowIndexVector(nProcessed, 0) = indexOffset;
				indexOffset += nResiduals * GetResidualDim<CostFunctionSignature>::value * GetTotalParamDim<CostFunctionSignature>::value;
				nProcessed++;
			}

			computeColumnIndexOffset(indexOffset, nProcessed, rowIndexVector, std::forward<Constraints>(otherConstraints)...);
		}

		static void computeColumnIndexOffset(
			int& indexOffset,
			unsigned& nProcessed,
			DenseMatrixInterface<int, MemoryTypeCPU>& rowIndexVector
		) {
			// We note down the number of all residual components, which is the same as the most
			// recent index offset start, which points to the index of the last value + 1.
			rowIndexVector(nProcessed, 0) = indexOffset;
		}		
	};

} // namespace solo