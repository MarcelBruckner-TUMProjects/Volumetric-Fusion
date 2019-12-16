#pragma once
#include <common_utils/timing/TimerCPU.h>
#include <common_utils/meta_structures/BasicTypes.h>
#include "ConstraintProcessingCPU.h"

namespace solo {

	/**
	 * Implementation of the evaluate() method.
	 */
	template<typename FloatType, typename ConstraintType, typename JacobianEvaluationFlag, typename GradientCheckOrEvaluationMode>
	void ConstraintProcessingCPU::evaluate(
		ConstraintType& constraint,
		DenseMatrixWrapper<FloatType>& paramVector,
		DenseMatrixWrapper<int>& indexMatrix,
		ResidualVector<FloatType>& residuals,
		JacobianMatrix<FloatType>& jacobian,
		Type2Type<JacobianEvaluationFlag>,
		Type2Type<GradientCheckOrEvaluationMode>
	) {
		evaluateImpl(constraint, paramVector, indexMatrix, residuals, jacobian, Type2Type<JacobianEvaluationFlag>(), Type2Type<GradientCheckOrEvaluationMode>());
	}

	template<typename FloatType, typename CostFunction, typename LocalData, typename GlobalData, unsigned NumBlocks, typename JacobianEvaluationFlag, typename GradientCheckOrEvaluationMode>
	void ConstraintProcessingCPU::evaluateImpl(
		Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>& constraint,
		DenseMatrixWrapper<FloatType>& paramVector,
		DenseMatrixWrapper<int>& indexMatrix,
		ResidualVector<FloatType>& residuals,
		JacobianMatrix<FloatType>& jacobian,
		Type2Type<JacobianEvaluationFlag>,
		Type2Type<GradientCheckOrEvaluationMode>
	) {
		using CostFunctionSignature = typename Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>::CostFunctionInterfaceSignature;
		
		TIME_CPU_START(ConstraintProcessingCPU_Initialize);

		// Allocate host memory for data structures, if necessary.
		const unsigned nResiduals = constraint.getNumResiduals();
		residuals.allocate(nResiduals, GetResidualDim<CostFunctionSignature>::value, Type2Type<MemoryTypeCPU>());
		if (typeid(JacobianEvaluationFlag) == typeid(ResidualAndJacobianEvaluation)) {
			jacobian.allocate(nResiduals, GetResidualDim<CostFunctionSignature>::value, GetTotalParamDim<CostFunctionSignature>::value, Type2Type<MemoryTypeCPU>());
		}

		// Check the given global and local data.
		auto& globalData = constraint.getGlobalData();
		auto& localData = constraint.getLocalData();

		using GlobalDataType = typename std::remove_reference<decltype(globalData)>::type;
		using LocalDataType = typename std::remove_reference<decltype(localData)>::type;

		using GlobalPointers = typename constraint_proc::TupleOfDataPointers<GlobalDataType>::type;
		using LocalPointers = typename constraint_proc::TupleOfDataPointers<LocalDataType>::type;

		// Update the wrappers' memory, if necessary.
		static_for<TupleSize<GlobalPointers>::value, constraint_proc::UpdateWrapperMemory>(globalData, Type2Type<MemoryTypeCPU>());
		static_for<TupleSize<LocalDataType>::value, constraint_proc::UpdateWrapperMemory>(localData, Type2Type<MemoryTypeCPU>());

		if (!paramVector.getWrapper().isUpdatedHost()) {
#			ifdef COMPILE_CUDA
			paramVector.getWrapper().copyDeviceToHost();
			paramVector.getWrapper().setUpdated(true, true);
#			endif
		}

		if (!indexMatrix.getWrapper().isUpdatedHost()) {
#			ifdef COMPILE_CUDA
			indexMatrix.getWrapper().copyDeviceToHost();
			indexMatrix.getWrapper().setUpdated(true, true);
#			endif
		}

		// Prepare host data pointers.
		GlobalPointers globalDataPtrs;
		static_for<TupleSize<GlobalPointers>::value, constraint_proc::CopyPointers>(globalData, globalDataPtrs, Type2Type<MemoryTypeCPU>());
		LocalPointers localDataPtrs;
		static_for<TupleSize<LocalPointers>::value, constraint_proc::CopyPointers>(localData, localDataPtrs, Type2Type<MemoryTypeCPU>());

		// Create interfaces around data structures, which store only some meta-data (such as number of 
		// residuals, etc.) and pointers to the real data (in our case, on host memory).
		DenseMatrixInterface<FloatType, MemoryTypeCPU> paramVectorInterface{ paramVector };
		DenseMatrixInterface<int, MemoryTypeCPU> indexMatrixInterface{ indexMatrix };
		ResidualVectorInterface<FloatType, MemoryTypeCPU> residualsInterface{ residuals };
		JacobianMatrixInterface<FloatType, MemoryTypeCPU> jacobianInterface{ jacobian };

		TIME_CPU_STOP(ConstraintProcessingCPU_Initialize);
		TIME_CPU_START(ConstraintProcessingCPU_Evaluate);

		// Execute evaluation for each residual.
#		pragma omp parallel for
		for (int i = 0; i < nResiduals; ++i) {
			auto localDataPtrsForResidual = localDataPtrs;
			static_for<TupleSize<LocalPointers>::value, constraint_proc::IncrementLocalPointers>(i, localDataPtrsForResidual);

			CostFunctionSignature::evaluateInternal(
				i, paramVectorInterface, indexMatrixInterface, residualsInterface, jacobianInterface, localDataPtrsForResidual, globalDataPtrs,
				Unsigned2Type<NumBlocks>(), Type2Type<JacobianEvaluationFlag>(), Type2Type<GradientCheckOrEvaluationMode>()
			);
		}

		// We set the updated flags.
		residuals.getContainer().setUpdated(true, false);
		jacobian.getContainer().setUpdated(true, false);

		TIME_CPU_STOP(ConstraintProcessingCPU_Evaluate);
	}

} // namespace solo