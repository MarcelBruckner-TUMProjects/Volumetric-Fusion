#pragma once
#include <common_utils/timing/TimerGPU.h>

#include "solo/utils/Common.h"
#include "ConstraintProcessingGPU.h"
#include "solo/data_structures/JacobianMatrix.h"
#include "solo/data_structures/ResidualVector.h"
#include "solo/data_structures/ResidualVectorInterface.h"
#include "solo/data_structures/jacobianMatrixInterface.h"

namespace solo {

	/**
	 * Evaluation kernel that is run on the GPU. 
	 */
	template<typename FloatType, typename ConstraintType, typename LocalData, typename GlobalData, unsigned NumBlocks, typename JacobianEvaluationFlag, typename GradientCheckOrEvaluationMode, typename MemoryStorageType>
	__global__ void evaluationKernel(
		Type2Type<ConstraintType>,
		DenseMatrixInterface<FloatType, MemoryStorageType> paramVector,
		DenseMatrixInterface<int, MemoryStorageType> indexMatrix,
		ResidualVectorInterface<FloatType, MemoryStorageType> residuals,
		JacobianMatrixInterface<FloatType, MemoryStorageType> jacobian,
		LocalData localData,
		GlobalData globalData,
		Unsigned2Type<NumBlocks>,
		Type2Type<JacobianEvaluationFlag>,
		Type2Type<GradientCheckOrEvaluationMode>
	) {
		using CostFunctionSignature = typename ConstraintType::CostFunctionInterfaceSignature;

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < residuals.getNumResiduals()) {
			static_for<TupleSize<LocalData>::value, constraint_proc::IncrementLocalPointers>(i, localData);
		
			CostFunctionSignature::evaluateInternal(
				i, paramVector, indexMatrix, residuals, jacobian, localData, globalData, Unsigned2Type<NumBlocks>(),
				Type2Type<JacobianEvaluationFlag>(), Type2Type<GradientCheckOrEvaluationMode>()
			);
		}
	}


	/**
	 * Implementation of the evaluate() method.
	 */
	template <typename FloatType, typename ConstraintType, typename JacobianEvaluationFlag, typename GradientCheckOrEvaluationMode>
	void ConstraintProcessingGPU::evaluate(
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

	template <typename FloatType, typename CostFunction, typename LocalData, typename GlobalData, unsigned NumBlocks, typename JacobianEvaluationFlag, typename GradientCheckOrEvaluationMode>
	void ConstraintProcessingGPU::evaluateImpl(
		Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>& constraint,
		DenseMatrixWrapper<FloatType>& paramVector,
		DenseMatrixWrapper<int>& indexMatrix,
		ResidualVector<FloatType>& residuals,
		JacobianMatrix<FloatType>& jacobian, 
		Type2Type<JacobianEvaluationFlag>,
		Type2Type<GradientCheckOrEvaluationMode>
	) {
		using CostFunctionSignature = typename Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>::CostFunctionInterfaceSignature;

		TIME_GPU_START(ConstraintProcessingGPU_TotalTime);
		TIME_GPU_START(ConstraintProcessingGPU_AllocateMemory);

		// Allocate host memory for data structures, if necessary.
		const unsigned nResiduals = constraint.getNumResiduals();
		residuals.allocate(nResiduals, GetResidualDim<CostFunctionSignature>::value, Type2Type<MemoryTypeCUDA>());
		if (typeid(JacobianEvaluationFlag) == typeid(ResidualAndJacobianEvaluation)) {
			jacobian.allocate(nResiduals, GetResidualDim<CostFunctionSignature>::value, GetTotalParamDim<CostFunctionSignature>::value, Type2Type<MemoryTypeCUDA>());
		}

		TIME_GPU_STOP(ConstraintProcessingGPU_AllocateMemory);
		TIME_GPU_START(ConstraintProcessingGPU_CopyData);

		// Check the given global and local data.
		auto& globalData = constraint.getGlobalData();
		auto& localData = constraint.getLocalData();

		using GlobalDataType = typename std::remove_reference<decltype(globalData)>::type;
		using LocalDataType = typename std::remove_reference<decltype(localData)>::type;

		using GlobalPointers = typename constraint_proc::TupleOfDataPointers<GlobalDataType>::type;
		using LocalPointers = typename constraint_proc::TupleOfDataPointers<LocalDataType>::type;

		// Update the wrappers' memory, if necessary.
		static_for<TupleSize<GlobalPointers>::value, constraint_proc::UpdateWrapperMemory>(globalData, Type2Type<MemoryTypeCUDA>());
		static_for<TupleSize<LocalDataType>::value, constraint_proc::UpdateWrapperMemory>(localData, Type2Type<MemoryTypeCUDA>());

		TIME_GPU_STOP(ConstraintProcessingGPU_CopyData);
		TIME_GPU_START(ConstraintProcessingGPU_CopyParams);
		
		if (!paramVector.getWrapper().isUpdatedDevice()) {
#			ifdef COMPILE_CUDA
			paramVector.getWrapper().copyHostToDevice();
			paramVector.getWrapper().setUpdated(true, true);
#			endif
		}

		if (!indexMatrix.getWrapper().isUpdatedDevice()) {
#			ifdef COMPILE_CUDA
			indexMatrix.getWrapper().copyHostToDevice();
			indexMatrix.getWrapper().setUpdated(true, true);
#			endif
		}

		TIME_GPU_STOP(ConstraintProcessingGPU_CopyParams);
		TIME_GPU_START(ConstraintProcessingGPU_InitializePointers);

		// Get the device data pointers.
		GlobalPointers globalDataPtrs;
		static_for<TupleSize<GlobalPointers>::value, constraint_proc::CopyPointers>(globalData, globalDataPtrs, Type2Type<MemoryTypeCUDA>());
		LocalPointers localDataPtrs;
		static_for<TupleSize<LocalPointers>::value, constraint_proc::CopyPointers>(localData, localDataPtrs, Type2Type<MemoryTypeCUDA>());

		// Create interfaces around data structures, which store only some meta-data (such as number of 
		// residuals, etc.) and pointers to the real data (in our case, on device memory).
		DenseMatrixInterface<FloatType, MemoryTypeCUDA> paramVectorInterface{ paramVector };
		DenseMatrixInterface<int, MemoryTypeCUDA> indexMatrixInterface{ indexMatrix };
		ResidualVectorInterface<FloatType, MemoryTypeCUDA> residualsInterface{ residuals };
		JacobianMatrixInterface<FloatType, MemoryTypeCUDA> jacobianInterface{ jacobian };

		TIME_GPU_STOP(ConstraintProcessingGPU_InitializePointers);
		TIME_GPU_START(ConstraintProcessingGPU_Evaluate);

		// Perform the evaluation with starting the evaluation kernel.
		const unsigned nThreadsPerBlock = 256;
		evaluationKernel<<< (nResiduals + nThreadsPerBlock - 1) / nThreadsPerBlock, nThreadsPerBlock >>>(
			Type2Type<Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>>(),
			paramVectorInterface,
			indexMatrixInterface,
			residualsInterface,
			jacobianInterface,
			localDataPtrs,
			globalDataPtrs,
			Unsigned2Type<NumBlocks>(),
			Type2Type<JacobianEvaluationFlag>(), 
			Type2Type<GradientCheckOrEvaluationMode>()
		);
		CUDA_CHECK_ERROR();

		// We set the updated flags.
		residuals.getContainer().setUpdated(false, true);
		jacobian.getContainer().setUpdated(false, true);

		TIME_GPU_STOP(ConstraintProcessingGPU_Evaluate);
		TIME_GPU_STOP(ConstraintProcessingGPU_TotalTime);
	}

} // namespace solo