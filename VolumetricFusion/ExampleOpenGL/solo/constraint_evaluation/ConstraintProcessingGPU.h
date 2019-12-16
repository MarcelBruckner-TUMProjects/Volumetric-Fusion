#pragma once
#include "Constraint.h"
#include "AnalyticCostFunction.h"
#include "AutoDiffCostFunction.h"
#include "ConstraintProcessingShared.h"

namespace solo {

	class ConstraintProcessingGPU {
	public:
		template<typename FloatType, typename ConstraintType, typename JacobianEvaluationFlag, typename GradientCheckOrEvaluationMode>
		static void evaluate(
			ConstraintType& constraint,
			DenseMatrixWrapper<FloatType>& paramVector,
			DenseMatrixWrapper<int>& indexMatrix,
			ResidualVector<FloatType>& residuals,
			JacobianMatrix<FloatType>& jacobian,
			Type2Type<JacobianEvaluationFlag>,
			Type2Type<GradientCheckOrEvaluationMode>
		);

	private:
		template<typename FloatType, typename CostFunction, typename LocalData, typename GlobalData, unsigned NumBlocks, typename JacobianEvaluationFlag, typename GradientCheckOrEvaluationMode>
		static void evaluateImpl(
			Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>& constraint,
			DenseMatrixWrapper<FloatType>& paramVector,
			DenseMatrixWrapper<int>& indexMatrix,
			ResidualVector<FloatType>& residuals,
			JacobianMatrix<FloatType>& jacobian,
			Type2Type<JacobianEvaluationFlag>,
			Type2Type<GradientCheckOrEvaluationMode>
		);
	};

} // namespace solo