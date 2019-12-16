#pragma once
#include "ParameterParser.h"
#include "CostFunctionInterface.h"
#include "CostFunctionProcessing.h"

namespace solo {

	template<typename CostFunction, unsigned ResidualDim, class ...Params>
	class AnalyticCostFunction :
		public CostFunctionInterface<
			AnalyticCostFunction<CostFunction, ResidualDim, Params...>,
			ResidualDim,
			typename param_parser::CreateTypeListOfParams<Params...>::type
		>
	{
	public:
		using ParamTypeList = typename param_parser::CreateTypeListOfParams<Params...>::type;

		/**
		 * The CostFunction needs to implement a static method evaluate(), since that is called when
		 * we evaluate the residuals/Jacobian.
		 */
		template<typename FloatType, typename LocalData, typename GlobalData, typename MemoryStorageType, unsigned NumBlocks>
		CPU_AND_GPU static void evaluateInternalImpl(
			unsigned componentIdx,
			const DenseMatrixInterface<FloatType, MemoryStorageType>& paramVector,
			const DenseMatrixInterface<int, MemoryStorageType>& indexMatrix,
			ResidualVectorInterface<FloatType, MemoryStorageType>& residualVector,
			JacobianMatrixInterface<FloatType, MemoryStorageType>& jacobianMatrix,
			LocalData& localData,
			GlobalData& globalData,
			Unsigned2Type<NumBlocks>,
			Type2Type<OnlyResidualEvaluation>,
			Type2Type<ConstraintEvaluationMode>
		) {
			// We evaluate only residuals, so the Jacobian should be a (double) empty tuple.
			param_parser::FloatParamSet<FloatType, ParamTypeList> paramSet;
			int idx = 0;
			static_for<TypeListLength<ParamTypeList>::value, cost_function_proc::LoadParametersOuter>(
				idx, componentIdx, paramVector, indexMatrix, paramSet, Type2Type<ParamTypeList>()
			);

			param_parser::FloatResidualSet<FloatType, ResidualDim> residualSet;
			static_for<ResidualDim, cost_function_proc::InitializeResiduals>(residualSet);
			DoubleEmptyTuple<FloatType> jacobianPlaceholder;

			CostFunction::evaluate(paramSet, residualSet, jacobianPlaceholder, localData, globalData);

			static_for<ResidualDim, cost_function_proc::StoreResidualsFloat>(componentIdx, residualVector, residualSet);
		}

		template<typename FloatType, typename LocalData, typename GlobalData, typename MemoryStorageType, unsigned NumBlocks>
		CPU_AND_GPU static void evaluateInternalImpl(
			unsigned componentIdx,
			const DenseMatrixInterface<FloatType, MemoryStorageType>& paramVector,
			const DenseMatrixInterface<int, MemoryStorageType>& indexMatrix,
			ResidualVectorInterface<FloatType, MemoryStorageType>& residualVector,
			JacobianMatrixInterface<FloatType, MemoryStorageType>& jacobianMatrix,
			LocalData& localData,
			GlobalData& globalData,
			Unsigned2Type<NumBlocks>,
			Type2Type<ResidualAndJacobianEvaluation>,
			Type2Type<ConstraintEvaluationMode>
		) {
			// We evaluate both residuals and the Jacobian.
			param_parser::FloatParamSet<FloatType, ParamTypeList> paramSet;
			int idx = 0;
			static_for<TypeListLength<ParamTypeList>::value, cost_function_proc::LoadParametersOuter>(
				idx, componentIdx, paramVector, indexMatrix, paramSet, Type2Type<ParamTypeList>()
			);

			param_parser::FloatResidualSet<FloatType, ResidualDim> residualSet;
			static_for<ResidualDim, cost_function_proc::InitializeResiduals>(residualSet);
			param_parser::FloatJacobianSet<FloatType, ResidualDim, param_parser::GetTotalParamDimension<ParamTypeList>::value> jacobianSet;
			static_for<ResidualDim, cost_function_proc::InitializeJacobianOuter>(jacobianSet, Type2Type<ParamTypeList>());

			CostFunction::evaluate(paramSet, residualSet, jacobianSet, localData, globalData);

			static_for<ResidualDim, cost_function_proc::StoreResidualsFloat>(componentIdx, residualVector, residualSet);
			static_for<ResidualDim, cost_function_proc::StoreJacobianFloatOuter>(componentIdx, jacobianMatrix, jacobianSet, Type2Type<ParamTypeList>());
		}

		template<typename FloatType, typename LocalData, typename GlobalData, typename MemoryStorageType, unsigned NumBlocks>
		CPU_AND_GPU static void evaluateInternalImpl(
			unsigned componentIdx,
			const DenseMatrixInterface<FloatType, MemoryStorageType>& paramVector,
			const DenseMatrixInterface<int, MemoryStorageType>& indexMatrix,
			ResidualVectorInterface<FloatType, MemoryStorageType>& residualVector,
			JacobianMatrixInterface<FloatType, MemoryStorageType>& jacobianMatrix,
			LocalData& localData,
			GlobalData& globalData,
			Unsigned2Type<NumBlocks>,
			Type2Type<ResidualAndJacobianEvaluation>,
			Type2Type<GradientCheckMode>
		) {
			const FloatType eps = sqrtf(FLT_EPSILON);

			// We evaluate residuals and Jacobian numerically, only used for gradient checking.
			param_parser::FloatParamSet<FloatType, ParamTypeList> paramSet;
			int idx = 0;
			static_for<TypeListLength<ParamTypeList>::value, cost_function_proc::LoadParametersOuter>(
				idx, componentIdx, paramVector, indexMatrix, paramSet, Type2Type<ParamTypeList>()
			);

			param_parser::FloatResidualSet<FloatType, ResidualDim> residualSet;
			static_for<ResidualDim, cost_function_proc::InitializeResiduals>(residualSet);
			DoubleEmptyTuple<FloatType> jacobianPlaceholder;
			CostFunction::evaluate(paramSet, residualSet, jacobianPlaceholder, localData, globalData);
			static_for<ResidualDim, cost_function_proc::StoreResidualsFloat>(componentIdx, residualVector, residualSet);

			idx = 0;
			static_for<TypeListLength<ParamTypeList>::value, cost_function_proc::ComputeAndStorePartialNumericDerivativeOuter>(
				idx, componentIdx, paramSet, residualSet, localData, globalData, jacobianMatrix, eps, Type2Type<CostFunction>(), Unsigned2Type<ResidualDim>(), Bool2Type<true>(), Type2Type<ParamTypeList>()
			);
		}
	};

} // namespace solo