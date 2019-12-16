#pragma once
#include "Constraint.h"
#include "Solo/data_structures/ResidualVector.h"
#include "Solo/data_structures/JacobianMatrix.h"
#include "CostFunctionInterface.h"

/**
 * Include the right header for constraint evaluation.
 */
#ifdef USE_GPU_EVALUATION
#include "ConstraintProcessingGPU.h"
#else
#include "ConstraintProcessingCPU.h"
#endif

namespace solo {
	
	class ConstraintEvaluator {
	public:
		/**
		 * Evalutes the residuals of the constraint.
		 */
		template<typename FloatType, typename CostFunction, typename LocalData, typename GlobalData, unsigned NumBlocks>
		static void computeResiduals(
			Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>& constraint,
			DenseMatrixWrapper<FloatType>& paramVector,
			ResidualVector<FloatType>& residuals
		) {
			using CostFunctionSignature = typename Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>::CostFunctionInterfaceSignature;

			// Check the parameter matrix dimensions.
			runtime_assert(paramVector.cols() == 1, "Parameter vector column dimension is wrong.");

			DenseMatrixWrapper<int>& indexMatrix = constraint.getIndexMatrix();

			runtime_assert(indexMatrix.rows() == constraint.getNumResiduals(), "Index matrix row dimension is wrong.");
			runtime_assert(indexMatrix.cols() == GetTotalParamDim<CostFunctionSignature>::value, "Index matrix column dimension is wrong.");

			// Evaluate the residuals, passing an empty jacobian (it won't be used at all).
			JacobianMatrix<FloatType> jacobian;
			#ifdef USE_GPU_EVALUATION
			ConstraintProcessingGPU::evaluate(constraint, paramVector, indexMatrix, residuals, jacobian, Type2Type<OnlyResidualEvaluation>(), Type2Type<ConstraintEvaluationMode>());
			#else
			ConstraintProcessingCPU::evaluate(constraint, paramVector, indexMatrix, residuals, jacobian, Type2Type<OnlyResidualEvaluation>(), Type2Type<ConstraintEvaluationMode>());
			#endif
		}

		/**
		 * Evalutes the residuals and Jacobian matrix of the constraint.
		 * If gradient check mode is set, the numerical derivative is used to compute the 
		 * partial derivatives (by default it's NOT set).
		 */
		template<typename FloatType, typename CostFunction, typename LocalData, typename GlobalData, unsigned NumBlocks>
		static void computeResidualsAndJacobian(
			Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>& constraint,
			DenseMatrixWrapper<FloatType>& paramVector,
			ResidualVector<FloatType>& residuals,
			JacobianMatrix<FloatType>& jacobian
		) {
			computeResidualsAndJacobian(constraint, paramVector, residuals, jacobian, Type2Type<ConstraintEvaluationMode>());
		}

		template<typename FloatType, typename CostFunction, typename LocalData, typename GlobalData, unsigned NumBlocks, typename GradientCheckOrEvaluationMode>
		static void computeResidualsAndJacobian(
			Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>& constraint,
			DenseMatrixWrapper<FloatType>& paramVector,
			ResidualVector<FloatType>& residuals, 
			JacobianMatrix<FloatType>& jacobian,
			Type2Type<GradientCheckOrEvaluationMode>
		) {
			using CostFunctionSignature = typename Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>::CostFunctionInterfaceSignature;

			// Check the parameter matrix dimensions.
			runtime_assert(paramVector.cols() == 1, "Parameter vector column dimension is wrong.");

			DenseMatrixWrapper<int>& indexMatrix = constraint.getIndexMatrix();

			runtime_assert(indexMatrix.rows() == constraint.getNumResiduals(), "Index matrix row dimension is wrong.");
			runtime_assert(indexMatrix.cols() == GetTotalParamDim<CostFunctionSignature>::value, "Index matrix column dimension is wrong.");

			// Evaluate the residuals and jacobian matrix.
			#ifdef USE_GPU_EVALUATION
			ConstraintProcessingGPU::evaluate(constraint, paramVector, indexMatrix, residuals, jacobian, Type2Type<ResidualAndJacobianEvaluation>(), Type2Type<GradientCheckOrEvaluationMode>());
			#else
			ConstraintProcessingCPU::evaluate(constraint, paramVector, indexMatrix, residuals, jacobian, Type2Type<ResidualAndJacobianEvaluation>(), Type2Type<GradientCheckOrEvaluationMode>());
			#endif
		}
	};

} // namespace solo