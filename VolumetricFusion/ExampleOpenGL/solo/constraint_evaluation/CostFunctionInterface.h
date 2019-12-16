#pragma once
#include <common_utils/meta_structures/BasicOperations.h>

#include "solo/data_structures/ResidualVector.h"
#include "solo/data_structures/JacobianMatrix.h"
#include "ParameterParser.h"
#include "solo/data_structures/ResidualVectorInterface.h"
#include "solo/data_structures/jacobianMatrixInterface.h"

namespace solo {

	/**
	 * Interface for the CostFunction classes. 
	 * Derived classes need to implement the following functions:
	 *		
	 *		template<typename FloatType, unsigned ResidualDim, unsigned ParamDim, typename LocalData, typename GlobalData, unsigned NumBlocks>
	 *		static void evaluateInternalImpl(
	 *			unsigned componentIdx, 
	 *			const ParameterMatrix<FloatType, ParamDim>& parameterMatrix, 
	 *			ResidualVector<FloatType, ResidualDim>& residualVector,
	 *			JacobianMatrix<FloatType, ResidualDim, ParamDim>& jacobianMatrix,
	 *			const LocalData& localData,
	 *			const GlobalData& globalData,
	 *			Unsigned2Type<NumBlocks>,
	 *			Type2Type<JacobianEvaluationFlag>,
	 *			Type2Type<GradientCheckOrEvaluationMode>
	 *		);
	 */
	template<typename CostFunctionType, unsigned ResidualDim, typename ParamTypeList>
	class CostFunctionInterface {
	public:
		/**
		 * The CostFunction needs to implement a static method Evaluate(), since that is called when
		 * we evaluate the residuals/Jacobian.
		 */
		template<typename FloatType, typename LocalData, typename GlobalData, unsigned NumBlocks, typename JacobianEvaluationFlag, typename GradientCheckOrEvaluationMode, typename MemoryStorageType>
		CPU_AND_GPU static void evaluateInternal(
			unsigned componentIdx,
			const DenseMatrixInterface<FloatType, MemoryStorageType>& paramVector,
			const DenseMatrixInterface<int, MemoryStorageType>& indexMatrix,
			ResidualVectorInterface<FloatType, MemoryStorageType>& residualVector,
			JacobianMatrixInterface<FloatType, MemoryStorageType>& jacobianMatrix,
			LocalData& localData,
			GlobalData& globalData,
			Unsigned2Type<NumBlocks>,
			Type2Type<JacobianEvaluationFlag>, 
			Type2Type<GradientCheckOrEvaluationMode>
		) {
			CostFunctionType::evaluateInternalImpl(
				componentIdx, paramVector, indexMatrix, residualVector, jacobianMatrix, localData, globalData, Unsigned2Type<NumBlocks>(),
				Type2Type<JacobianEvaluationFlag>(), Type2Type<GradientCheckOrEvaluationMode>()
			);
		}
	};

	/**
	 * Converts a class that derived from some CostFunctionInterface<CostFunctionType, ResidualDim, ParamTypeList>
	 * to CostFunctionInterface<CostFunctionType, ResidualDim, ParamTypeList>.
	 */
	template <typename CostFunctionType, unsigned ResidualDim, typename ParamTypeList>
	Type2Type<CostFunctionInterface<CostFunctionType, ResidualDim, ParamTypeList>> getCostFunctionInterfaceType(CostFunctionInterface<CostFunctionType, ResidualDim, ParamTypeList>*);

	template<typename CostFunction>
	struct CostFunctionInterfaceType {
		using type = typename ExtractType<decltype(getCostFunctionInterfaceType(std::declval<CostFunction*>()))>::type;
	};


	/**
	 * Returns the dimension of residuals of the given cost function.
	 */
	template<typename CostFunction>
	struct GetResidualDim;

	template<typename CostFunctionType, unsigned ResidualDim, typename ParamTypeList>
	struct GetResidualDim<CostFunctionInterface<CostFunctionType, ResidualDim, ParamTypeList>> {
		enum {
			value = ResidualDim
		};
	};


	/**
	 * Returns the total (summed) dimension of parameters of the given cost function.
	 */
	template<typename CostFunction>
	struct GetTotalParamDim;

	template<typename CostFunctionType, unsigned ResidualDim, typename ParamTypeList>
	struct GetTotalParamDim<CostFunctionInterface<CostFunctionType, ResidualDim, ParamTypeList>> {
		enum {
			value = param_parser::GetTotalParamDimension<ParamTypeList>::value
		};
	};


	/**
	 * Returns the specific dimension of a parameter at a given index, of the given cost function.
	 */
	template<unsigned Idx, typename CostFunction>
	struct GetSpecificParamDim;

	template<unsigned Idx, typename CostFunctionType, unsigned ResidualDim, typename ParamTypeList>
	struct GetSpecificParamDim<Idx, CostFunctionInterface<CostFunctionType, ResidualDim, ParamTypeList>> {
		enum {
			value = param_parser::GetParamDimension<Idx, ParamTypeList>::value
		};
	};


	/**
	 * Returns the number of all parameters of the given cost function.
	 */
	template<typename CostFunction>
	struct GetNumParams;

	template<typename CostFunctionType, unsigned ResidualDim, typename ParamTypeList>
	struct GetNumParams<CostFunctionInterface<CostFunctionType, ResidualDim, ParamTypeList>> {
		enum {
			value = TypeListLength<ParamTypeList>::value
		};
	};


	/**
	 * Flags for residual/Jacobian evaluation mode.
	 */
	struct OnlyResidualEvaluation {};
	struct ResidualAndJacobianEvaluation {};


	/**
	 * Flags for gradient check/constraint evaluation mode.
	 */
	struct GradientCheckMode;
	struct ConstraintEvaluationMode;

} // namespace solo