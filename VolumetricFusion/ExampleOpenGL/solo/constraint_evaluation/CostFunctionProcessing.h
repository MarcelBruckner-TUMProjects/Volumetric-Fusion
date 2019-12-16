#pragma once
#include <common_utils/meta_structures/BasicTypes.h>

#include "ParameterParser.h"

namespace solo {
	namespace cost_function_proc {
		
		/**
		 * Helper methods for loading initial parameter values.
		 */
		template<typename FloatType, typename IdxList>
		CPU_AND_GPU void initializeParameter(Dual<FloatType, IdxList>& dual, FloatType paramVal) {
			#ifdef ENABLE_ASSERTIONS
			if (!isfinite(paramVal)) printf("Parameter value not finite: %f\n", paramVal);
			#endif

			dual.init(paramVal);
		}

		template<typename FloatType>
		CPU_AND_GPU void initializeParameter(FloatType& floatParam, FloatType paramVal) {
			#ifdef ENABLE_ASSERTIONS
			if (!isfinite(paramVal)) printf("Parameter value not finite: %f\n", paramVal);
			#endif

			floatParam = paramVal;
		}

		struct LoadParametersInner {
			template<int k, int j, typename FloatType, typename MemoryStorageType, typename ParamSet>
			CPU_AND_GPU static void f(
				int& idx, 
				unsigned componentIdx, 
				const DenseMatrixInterface<FloatType, MemoryStorageType>& paramVector,
				const DenseMatrixInterface<int, MemoryStorageType>& indexMatrix, 
				ParamSet& paramSet, 
				Int2Type<j>
			) {
				initializeParameter(paramSet[I<j>()][I<k>()], paramVector(indexMatrix(componentIdx, idx), 0));
				idx++;
			}
		};

		struct LoadParametersOuter {
			template<int j, typename FloatType, typename MemoryStorageType, typename ParamSet, typename ParamTypeList>
			CPU_AND_GPU static void f(
				int& idx, 
				unsigned componentIdx, 
				const DenseMatrixInterface<FloatType, MemoryStorageType>& paramVector,
				const DenseMatrixInterface<int, MemoryStorageType>& indexMatrix, 
				ParamSet& paramSet, 
				Type2Type<ParamTypeList>
			) {
				// We iterate through all every component of the current parameter, and load it.
				static_for<param_parser::GetParamDimension<j, ParamTypeList>::value, LoadParametersInner>(
					idx, componentIdx, paramVector, indexMatrix, paramSet, Int2Type<j>()
				);
			}
		};


		/**
		 * Helper methods for initializing a residual set (tuple of residual objects) with zero residual (and derivative) values.
		 */
		template<typename FloatType, typename IdxList>
		CPU_AND_GPU void initializeResidual(Dual<FloatType, IdxList>& dual) {
			dual.set(FloatType(0), FloatType(0));
		}

		template<typename FloatType>
		CPU_AND_GPU void initializeResidual(FloatType& floatParam) {
			floatParam = FloatType(0);
		}

		struct InitializeResiduals {
			template<int idx, typename ResSet>
			CPU_AND_GPU static void f(ResSet& residualSet) {
				initializeResidual(residualSet[I<idx>()]);
			}
		};


		/**
		 * Helper methods for initializing a jacobian set (tuple of floating-point tuples) with zero derivative values.
		 */
		template<typename FloatType>
		CPU_AND_GPU void initializeJacobian(FloatType& floatParam) {
			floatParam = FloatType(0);
		}

		struct InitializeJacobianInner {
			template<int paramId, int residualId, typename JacSet>
			CPU_AND_GPU static void f(JacSet& jacobianSet, Int2Type<residualId>) {
				initializeJacobian(jacobianSet[I<residualId>()][I<paramId>()]);
			}
		};

		struct InitializeJacobianOuter {
			template<int residualId, typename JacSet, typename ParamTypeList>
			CPU_AND_GPU static void f(JacSet& jacobianSet, Type2Type<ParamTypeList>) {
				static_for<param_parser::GetTotalParamDimension<ParamTypeList>::value, InitializeJacobianInner>(jacobianSet, Int2Type<residualId>());
			}
		};


		/**
		 * Helper methods for storing final residual/jacobian values for FloatType data structures.
		 */
		struct StoreResidualsFloat {
			template<int idx, typename ResVec, typename ResSet>
			CPU_AND_GPU static void f(unsigned componentIdx, ResVec& residualVector, const ResSet& residualSet) {
				#ifdef ENABLE_ASSERTIONS
				if (!isfinite(residualSet[I<idx>()])) printf("Residual value not finite: %f\n", residualSet[I<idx>()]);
				#endif

				residualVector(componentIdx, I<idx>()) = residualSet[I<idx>()];
			}
		};

		struct StoreJacobianFloatInner {
			template<int paramId, int residualId, typename JacobianMat, typename JacSet>
			CPU_AND_GPU static void f(unsigned componentIdx, JacobianMat& jacobianMatrix, const JacSet& jacobianSet, Int2Type<residualId>) {
				#ifdef ENABLE_ASSERTIONS
				if (!isfinite(jacobianSet[I<residualId>()][I<paramId>()])) printf("Jacobian value not finite: %f\n", jacobianSet[I<residualId>()][I<paramId>()]);
				#endif

				jacobianMatrix(componentIdx, I<residualId>(), I<paramId>()) = jacobianSet[I<residualId>()][I<paramId>()];
			}
		};

		struct StoreJacobianFloatOuter {
			template<int residualId, typename JacobianMat, typename JacSet, typename ParamTypeList>
			CPU_AND_GPU static void f(unsigned componentIdx, JacobianMat& jacobianMatrix, const JacSet& jacobianSet, Type2Type<ParamTypeList>) {
				static_for<param_parser::GetTotalParamDimension<ParamTypeList>::value, StoreJacobianFloatInner>(componentIdx, jacobianMatrix, jacobianSet, Int2Type<residualId>());
			}
		};


		/**
		 * Helper methods for storing final residual/jacobian values for Dual data structures.
		 */
		struct StoreResidualsDual {
			template<int idx, typename ResVec, typename ResSet> 
			CPU_AND_GPU static void f(unsigned componentIdx, ResVec& residualVector, const ResSet& residualSet) {
				#ifdef ENABLE_ASSERTIONS
				if (!isfinite(residualSet[I<idx>()].r())) printf("Residual value not finite: %f\n", residualSet[I<idx>()].r());
				#endif

				residualVector(componentIdx, I<idx>()) = residualSet[I<idx>()].r();
			}
		};

		struct StoreJacobianDualInner {
			template<int paramId, int residualId, typename JacobianMat, typename ResSet> 
			CPU_AND_GPU static void f(unsigned componentIdx, JacobianMat& jacobianMatrix, const ResSet& residualSet, Int2Type<residualId>) {
				#ifdef ENABLE_ASSERTIONS
				if (!isfinite(residualSet[I<residualId>()].i()[I<paramId>()])) printf("Jacobian value not finite: %f\n", residualSet[I<residualId>()].i()[I<paramId>()]);
				#endif

				jacobianMatrix(componentIdx, I<residualId>(), I<paramId>()) = residualSet[I<residualId>()].i()[I<paramId>()];
			}
		};

		struct StoreJacobianDualOuter {
			template<int residualId, typename JacobianMat, typename ResSet, unsigned BlockStart, unsigned BlockEnd>
			CPU_AND_GPU static void f(unsigned componentIdx, JacobianMat& jacobianMatrix, const ResSet& residualSet, Unsigned2Type<BlockStart>, Unsigned2Type<BlockEnd>) {
				static_for<BlockStart, BlockEnd, StoreJacobianDualInner>(componentIdx, jacobianMatrix, residualSet, Int2Type<residualId>());
			}
		};	
		

		/**
		 * Evaluates the cost function, either in the form of analytic cost functions (if the Jacobian
		 * argument is present), or in the form of auto-diff or numerical cost functions (otherwise).
		 */
		template<typename ParamSet, typename ResSet, typename LocalData, typename GlobalData, typename CostFunction, typename FloatType, bool JacobianArgPresent>
		CPU_AND_GPU void evaluateCostFunction(
			ParamSet& paramSet, ResSet& residualSet,
			LocalData& localData, GlobalData& globalData, 
			Type2Type<CostFunction>, Type2Type<FloatType>, Bool2Type<JacobianArgPresent>
		);


		template<typename ParamSet, typename ResSet, typename LocalData, typename GlobalData, typename CostFunction, typename FloatType>
		CPU_AND_GPU void evaluateCostFunction(
			ParamSet& paramSet, ResSet& residualSet,
			LocalData& localData, GlobalData& globalData,
			Type2Type<CostFunction>, Type2Type<FloatType>, Bool2Type<false>
		) {
			CostFunction::evaluate(paramSet, residualSet, localData, globalData);
		}

		template<typename ParamSet, typename ResSet, typename LocalData, typename GlobalData, typename CostFunction, typename FloatType>
		CPU_AND_GPU void evaluateCostFunction(
			ParamSet& paramSet, ResSet& residualSet,
			LocalData& localData, GlobalData& globalData,
			Type2Type<CostFunction>, Type2Type<FloatType>, Bool2Type<true>
		) {
			DoubleEmptyTuple<FloatType> jacobianPlaceholder;
			CostFunction::evaluate(paramSet, residualSet, jacobianPlaceholder, localData, globalData);
		}


		/**
		 * Computes and stores the numerical partial derivative for the given parameter set and already
		 * computed residual values.
		 */
		struct ComputeAndStoreJacobianComponent {
			template<int residualId, typename JacobianMat, typename ResSet, typename FloatType>
			CPU_AND_GPU static void f(unsigned componentIdx, JacobianMat& jacobianMatrix, const ResSet& residualSet, const ResSet& residualAtPlusEpsSet, FloatType hInv, int paramId) {
				#ifdef ENABLE_ASSERTIONS
				if (!isfinite((residualAtPlusEpsSet[I<residualId>()] - residualSet[I<residualId>()]) * hInv)) printf("Jacobian value not finite: %f\n", (residualAtPlusEpsSet[I<residualId>()] - residualSet[I<residualId>()]) * hInv);
				#endif

				jacobianMatrix(componentIdx, residualId, paramId) = (residualAtPlusEpsSet[I<residualId>()] - residualSet[I<residualId>()]) * hInv;
			}
		};

		struct ComputeAndStorePartialNumericDerivativeInner {
			template<int k, int j, typename ParamSet, typename ResSet, typename JacobianMat, typename LocalData, typename GlobalData, unsigned ResidualDim, typename CostFunction, typename FloatType, bool JacobianArgPresent>
			CPU_AND_GPU static void f(
				int& paramId, unsigned componentIdx,
				ParamSet& paramSet, ResSet& residualSet,
				LocalData& localData, GlobalData& globalData,
				JacobianMat& jacobianMatrix, FloatType eps,
				Type2Type<CostFunction>, Unsigned2Type<ResidualDim>, Bool2Type<JacobianArgPresent>, Int2Type<j>
			) {
				// Set the value of the current param component to its value + eps.
				FloatType paramValue = paramSet[I<j>()][I<k>()];
				FloatType h = eps * (abs(paramValue) + eps);
				paramSet[I<j>()][I<k>()] = paramValue + h;

				// Evaluate the cost function at the incremented param value.
				param_parser::FloatResidualSet<FloatType, ResidualDim> residualAtPlusEpsSet;
				evaluateCostFunction(paramSet, residualAtPlusEpsSet, localData, globalData, Type2Type<CostFunction>(), Type2Type<FloatType>(), Bool2Type<JacobianArgPresent>());

				// Store the resulting partial derivative to the jacobian matrix (in global memory).
				FloatType hInv = FloatType(1.0) / h;
				static_for<ResidualDim, ComputeAndStoreJacobianComponent>(componentIdx, jacobianMatrix, residualSet, residualAtPlusEpsSet, hInv, paramId);

				// Reset the value of the current param component.
				paramSet[I<j>()][I<k>()] = paramValue;
				paramId++;
			}
		};

		struct ComputeAndStorePartialNumericDerivativeOuter {
			template<int j, typename ParamSet, typename ResSet, typename JacobianMat, typename LocalData, typename GlobalData, unsigned ResidualDim, typename CostFunction, typename FloatType, bool JacobianArgPresent, typename ParamTypeList>
			CPU_AND_GPU static void f(
				int& paramId, unsigned componentIdx,
				ParamSet& paramSet, ResSet& residualSet,
				LocalData& localData, GlobalData& globalData,
				JacobianMat& jacobianMatrix, FloatType eps, 
				Type2Type<CostFunction>, Unsigned2Type<ResidualDim>, Bool2Type<JacobianArgPresent>, Type2Type<ParamTypeList>
			) {
				// We iterate through all every component of the current parameter, and compute and store partial derivate of it.
				static_for<param_parser::GetParamDimension<j, ParamTypeList>::value, ComputeAndStorePartialNumericDerivativeInner>(
					paramId, componentIdx, paramSet, residualSet, localData, globalData, jacobianMatrix, eps, Type2Type<CostFunction>(), Unsigned2Type<ResidualDim>(), Bool2Type<JacobianArgPresent>(), Int2Type<j>()
				);
			}
		};

	} // namespace cost_function_proc
} // namespace solo