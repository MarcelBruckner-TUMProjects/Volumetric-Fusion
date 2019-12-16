#pragma once
#include "solo/constraint_evaluation/ConstraintEvaluator.h"
#include "solo/constraint_evaluation/ParameterManager.h"
#include "solo/data_structures/ParamVector.h"

namespace solo {

	template<typename FloatType, typename ParamStorageType>
	class GradientChecker {
	public:
		/**
		 * Checks the gradients of all present constraints.
		 * @param	eps		Maximum allowed absolute difference between numerical and 
		 *					actual derivatives
		 * @returns True if all partial derivative match, otherwise false
		 */
		template<typename ...Constraints>
		static bool check(double eps, ParamVector<FloatType, ParamStorageType> paramVector, Constraints&&... constraints) {
			// Initialize a wrapper around parameter vector.
			DenseMatrixWrapper<FloatType> paramVectorWrapper{ paramVector.getData(), paramVector.getSize(), 1, Type2Type<ParamStorageType>() };

			// Run gradient check.
			return checkGradients(eps, 0, paramVectorWrapper, std::forward<Constraints>(constraints)...);
		}

	private:
		/**
		 * Recursively checks all present constraints.
		 */
		template<typename CostFunction, typename LocalData, typename GlobalData, unsigned NumBlocks, typename ...OtherConstraints>
		static bool checkGradients(
			double eps, 
			unsigned constraintNb, 
			DenseMatrixWrapper<FloatType>& paramVector, 
			Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>& constraint, 
			OtherConstraints&&... otherConstraints
		) {
			using CostFunctionSignature = typename Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>::CostFunctionInterfaceSignature;

			// Compute the residual vector and Jacobian matrix using original cost functions.
			ResidualVector<FloatType> residualsComputed;
			JacobianMatrix<FloatType> jacobianComputed;
			ConstraintEvaluator::computeResidualsAndJacobian(constraint, paramVector, residualsComputed, jacobianComputed, Type2Type<ConstraintEvaluationMode>());

			// Compute the residual vector and Jacobian matrix using numerical estimation of derivatives.
			ResidualVector<FloatType> residualsNumerical;
			JacobianMatrix<FloatType> jacobianNumerical;
			ConstraintEvaluator::computeResidualsAndJacobian(constraint, paramVector, residualsNumerical, jacobianNumerical, Type2Type<GradientCheckMode>());

			// Check if the copy to host (CPU) memory is needed.
			#ifdef COMPILE_CUDA
			auto& jacobianComputedContainer = jacobianComputed.getContainer();
			if (jacobianComputedContainer.isUpdatedDevice() && !jacobianComputedContainer.isUpdatedHost()) {
				jacobianComputedContainer.copyDeviceToHost();
				jacobianComputedContainer.setUpdated(true, true);
			}

			auto& jacobianNumericalContainer = jacobianNumerical.getContainer();
			if (jacobianNumericalContainer.isUpdatedDevice() && !jacobianNumericalContainer.isUpdatedHost()) {
				jacobianNumericalContainer.copyDeviceToHost();
				jacobianNumericalContainer.setUpdated(true, true);
			}
			#endif

			// Compare all derivatives and report if differences occur.
			const unsigned nResiduals = jacobianComputed.getNumResiduals();
			const unsigned residualDim = jacobianComputed.getResidualDim();
			const unsigned paramDim = jacobianComputed.getParamDim();
			bool bAllDerivativesMatch{ true };

			JacobianMatrixInterface<FloatType, MemoryTypeCPU> iJacobianComputed{ jacobianComputed };
			JacobianMatrixInterface<FloatType, MemoryTypeCPU> iJacobianNumerical{ jacobianNumerical };

			for (int componentIdx = 0; componentIdx < nResiduals; ++componentIdx) {
				for (int residualId = 0; residualId < residualDim; ++residualId) {
					for (int paramId = 0; paramId < paramDim; ++paramId) {
						const FloatType derivativeComputed = iJacobianComputed(componentIdx, residualId, paramId);
						const FloatType derivativeNumerical = iJacobianNumerical(componentIdx, residualId, paramId);

						FloatType diff = abs(derivativeComputed - derivativeNumerical);
						if (diff > eps) {
							cout << "Derivative mismatch: constraint = " << constraintNb << ", residual component = " << componentIdx << ", residual id = " << residualId << ", paramId = " << paramId << endl;
							cout << "Computed value:  " << derivativeComputed << endl;
							cout << "Numerical value: " << derivativeNumerical << endl;
							bAllDerivativesMatch = false;
						}
					}
				}
			}

			// Recursively evaluate all the other constraints.
			return checkGradients(eps, constraintNb + 1, paramVector, std::forward<OtherConstraints>(otherConstraints)...) && bAllDerivativesMatch;
		}

		static bool checkGradients(double eps, unsigned constraintNb, DenseMatrixWrapper<FloatType>& paramVector) {
			return true;
		}
	};

} // namespace solo