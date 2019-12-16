#pragma once
#include "ParameterParser.h"
#include "Solo/meta_structures/TypeFinder.h"

namespace solo {

	/**
	 * Bridge class is a bridge between automatic and analytic differentiation. 
	 * You can use Bridge in AutoDiffCostFunction evaluation methods in order to write
	 * analytic derivatives for a small part of the cost function, i.e. for image querying.
	 * Bridge objects accept Dual vectors and return the resulting Dual elements, using 
	 * the chain rule and analytically computed residual and Jacobian values.
	 */
	template<typename FloatType, unsigned ResidualDim, unsigned ParamDim>
	class Bridge {
	public:
		param_parser::FloatResidualSet<FloatType, ResidualDim> r;
		param_parser::FloatJacobianSet<FloatType, ResidualDim, ParamDim> J;		

		/**
		 * Implementation for the real valued input.
		 * We don't need to propagate any gradient, we just return the residual.
		 */
		CPU_AND_GPU auto apply(const Tuple<typename AddElements<ParamDim, FloatType, NullType>::type>& paramSet) -> param_parser::FloatResidualSet<FloatType, ResidualDim> {
			return r;
		}

		/**
		 * Implementation for the dual valued input.
		 * We need to propagate the gradient using the chain rule, and return appropriate
		 * dual objects.
		 */
		template<typename TList>
		CPU_AND_GPU auto apply(const Tuple<TList>& paramSet) -> Tuple<typename AddElements<ResidualDim, typename ResultTypeHelper<TList>::type, NullType>::type> {
			using ResultTupleType = Tuple<typename AddElements<ResidualDim, typename ResultTypeHelper<TList>::type, NullType>::type>;
			static_assert(TypeListLength<TList>::value == ParamDim, "The input tuple dimension mast match the param dimension.");

			ResultTupleType resultTuple;
			static_for<ResidualDim, SetResultTuple>(resultTuple, paramSet, r, J);

			return resultTuple;
		}

	private:
		/**
		 * Helper methods for applying the chain rule.
		 */
		template<typename ResultDual>
		CPU_AND_GPU static void computePartialDerivativeContribution(ResultDual& resultDual, FloatType param, FloatType alpha) {
			// If parameter is not dual, but a floating point value, the derivative contribution is 0.0.
			// That equals to no addition operation at all.
		}

		template<typename ResultDual, typename ParamIdxList>
		CPU_AND_GPU static void computePartialDerivativeContribution(ResultDual& resultDual, const Dual<FloatType, ParamIdxList>& param, FloatType alpha) {
			// If parameter is dual, we scale it by the alpha value and add it to the partial derivative.
			computeSparseVecScaledAddition(resultDual.i(), param.i(), alpha);
		}

		struct ComputeSpecificDual {
			template<int ParamId, int ResidualId, typename ResultDual, typename ParamSet, typename JacobianSet>
			CPU_AND_GPU static void f(ResultDual& resultDual, const ParamSet& paramSet, const JacobianSet& jacobianSet, Int2Type<ResidualId>) {
				// We add a scaled input parameter dual, scaling it by the suitable partial derivative.
				FloatType alpha = jacobianSet[I<ResidualId>()][I<ParamId>()];
				Bridge<FloatType, ResidualDim, ParamDim>::computePartialDerivativeContribution(resultDual, paramSet[I<ParamId>()], alpha);
			}
		};

		struct SetResultTuple {
			template<int ResidualId, typename ResultTupleType, typename ParamSet, typename ResidualSet, typename JacobianSet> 
			CPU_AND_GPU static void f(ResultTupleType& resultTuple, const ParamSet& paramSet, const ResidualSet& residualSet, const JacobianSet& jacobianSet) {
				auto& dual = resultTuple[I<ResidualId>()];
				
				// Set the residual value and initialize Jacobian values to 0.0.
				dual.set(residualSet[I<ResidualId>()], FloatType(0.0));

				// Compute Jacobian values by adding a linear combination of input param dual vectors.
				// The linear coefficients are exactly the partial derivatives of the current derivative,
				// which follows directly from the chain rule.
				static_for<ParamDim, ComputeSpecificDual>(dual, paramSet, jacobianSet, Int2Type<ResidualId>());
			}
		};
	};

} // namespace solo