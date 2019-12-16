#pragma once
#include "ParameterParser.h"
#include "CostFunctionInterface.h"
#include "CostFunctionProcessing.h"

namespace solo {
	
	template<typename CostFunction, unsigned ResidualDim, class ...Params>
	class AutoDiffCostFunction : 
		public CostFunctionInterface<
			AutoDiffCostFunction<CostFunction, ResidualDim, Params...>, 
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
			// We evaluate only residuals, therefore we pass only FloatType values through the cost function.	
			param_parser::FloatParamSet<FloatType, ParamTypeList> paramSet;
			int idx = 0;
			static_for<TypeListLength<ParamTypeList>::value, cost_function_proc::LoadParametersOuter>(
				idx, componentIdx, paramVector, indexMatrix, paramSet, Type2Type<ParamTypeList>()
			);
				
			param_parser::FloatResidualSet<FloatType, ResidualDim> residualSet;
			static_for<ResidualDim, cost_function_proc::InitializeResiduals>(residualSet);
			CostFunction::evaluate(paramSet, residualSet, localData, globalData);

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
			using BlockSize = Unsigned2Type<(param_parser::GetTotalParamDimension<ParamTypeList>::value + NumBlocks - 1) / NumBlocks>;

			// Check that the provided number of blocks is valid.
			checkNumBlocks(Unsigned2Type<NumBlocks>(), Unsigned2Type<param_parser::GetTotalParamDimension<ParamTypeList>::value>());

			// Evaluate residuals and Jacobian for all parameter blocks.
			static_for<NumBlocks, ParamBlockEvaluator>(
				componentIdx, paramVector, indexMatrix, residualVector, jacobianMatrix, localData, globalData,
				BlockSize(), Unsigned2Type<param_parser::GetTotalParamDimension<ParamTypeList>::value>()
			);
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
			CostFunction::evaluate(paramSet, residualSet, localData, globalData);
			static_for<ResidualDim, cost_function_proc::StoreResidualsFloat>(componentIdx, residualVector, residualSet);

			idx = 0;
			static_for<TypeListLength<ParamTypeList>::value, cost_function_proc::ComputeAndStorePartialNumericDerivativeOuter>(
				idx, componentIdx, paramSet, residualSet, localData, globalData, jacobianMatrix, eps, Type2Type<CostFunction>(), Unsigned2Type<ResidualDim>(), Bool2Type<false>(), Type2Type<ParamTypeList>()
			);
		}

	private:
		/**
		 * Helper methods for block-wise auto-differentiation.
		 */
		template<unsigned NumBlocks, unsigned TotalParamDim>
		CPU_AND_GPU static void checkNumBlocks(Unsigned2Type<NumBlocks>, Unsigned2Type<TotalParamDim>) {
			if (NumBlocks < 1) {
				printf("The number of parameter blocks should be greater or equal to 1.");
				exit(-1);
			}

			if (NumBlocks > TotalParamDim) {
				printf("The number of parameter blocks shouldn't exceed the total dimension of parameters.");
				exit(-1);
			}
		}

		struct ParamBlockEvaluator {
			template<int BlockIdx, typename FloatType, typename LocalData, typename GlobalData, typename MemoryStorageType, unsigned BlockSize, unsigned TotalParamDim>
			CPU_AND_GPU static void f(
				unsigned componentIdx,
				const DenseMatrixInterface<FloatType, MemoryStorageType>& paramVector,
				const DenseMatrixInterface<int, MemoryStorageType>& indexMatrix,
				ResidualVectorInterface<FloatType, MemoryStorageType>& residualVector,
				JacobianMatrixInterface<FloatType, MemoryStorageType>& jacobianMatrix,
				LocalData& localData,
				GlobalData& globalData,
				Unsigned2Type<BlockSize>,
				Unsigned2Type<TotalParamDim>
			) {
				// We evaluate residuals and Jacobian, therefore we pass Dual values through the cost function.
				// We only compute the partial derivatives for parameters in block [BlockStart, BlockEnd].
				using BlockStart = Unsigned2Type<BlockIdx * BlockSize>;
				using BlockEnd = Unsigned2Type<MinValue<(BlockIdx + 1) * BlockSize - 1, TotalParamDim - 1>::value>;

				param_parser::DualParamSet<FloatType, ParamTypeList, ExtractValue<BlockStart>::value, ExtractValue<BlockEnd>::value> paramSet;
				int idx = 0;
				static_for<TypeListLength<ParamTypeList>::value, cost_function_proc::LoadParametersOuter>(
					idx, componentIdx, paramVector, indexMatrix, paramSet, Type2Type<ParamTypeList>()
				);

				param_parser::DualResidualSet<FloatType, ResidualDim, ExtractValue<BlockStart>::value, ExtractValue<BlockEnd>::value> residualSet;
				static_for<ResidualDim, cost_function_proc::InitializeResiduals>(residualSet);

				CostFunction::evaluate(paramSet, residualSet, localData, globalData);

				static_for<ResidualDim, cost_function_proc::StoreResidualsDual>(componentIdx, residualVector, residualSet);
				static_for<ResidualDim, cost_function_proc::StoreJacobianDualOuter>(componentIdx, jacobianMatrix, residualSet, BlockStart(), BlockEnd());
			}
		};
	};

} // namespace solo