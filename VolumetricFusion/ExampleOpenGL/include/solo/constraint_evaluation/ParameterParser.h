#pragma once
#include <common_utils/meta_structures/BasicTypes.h>
#include <common_utils/meta_structures/Tuple.h>

#include "solo/meta_structures/Dual.h"

namespace solo {

	/**
	 * Params struct includes the dimension of a single parameter.
	 */
	template<unsigned Dimension>
	struct Param { };


	/**
	 * Params struct includes the information about particular parameters.
	 * The struct includes Count parameters with given Dimension.
	 */
	template<unsigned Count, unsigned Dimension>
	struct Params { };

	
	namespace param_parser {
		
		/**
		 * ParamUnit struct is used internally for joint representation of both Param and Params structs.
		 */
		template<unsigned Dimension>
		struct ParamUnit { };

		/**
		 * Creates a typelist, filled with parameters, given with ... notation.
		 * It creates a ParamUnit<Dim> type for each parameter. The multi-dimensional parameters (set
		 * as Params<Count, Dimension>) are expanded into single Param<Dimension> classes.
		 */
		template<typename ...Params>
		struct CreateTypeListOfParams;

		template<unsigned Count, unsigned Dimension, typename ...OtherParams>
		struct CreateTypeListOfParams<Params<Count, Dimension>, OtherParams...> {
			using type = typename AddElements<Count, ParamUnit<Dimension>, typename CreateTypeListOfParams<OtherParams...>::type>::type;
		};

		template<unsigned Dimension, typename ...OtherParams>
		struct CreateTypeListOfParams<Param<Dimension>, OtherParams...> {
			using type = TypeList<ParamUnit<Dimension>, typename CreateTypeListOfParams<OtherParams...>::type>;
		};

		template<>
		struct CreateTypeListOfParams<> {
			using type = NullType;
		};


		/**
		 * Extracts a dimension value from the ParamUnit class.
		 */
		template<typename ParamUnitClass>
		struct ExtractDimension;

		template<unsigned i>
		struct ExtractDimension<ParamUnit<i>> {
			enum {
				value = i
			};
		};


		/**
		 * Returns a dimension of a param at a given index.
		 * The input typelist TList should include only ParamUnit<Dim> classes.
		 */
		template<unsigned Idx, typename TList>
		struct GetParamDimension {
			enum {
				value = ExtractDimension<typename TypeAt<Idx, TList>::type>::value
			};
		};


		/**
		 * Returns a total (summed) dimension of all params in the param list.
		 * The input typelist TList should include only ParamUnit<Dim> classes.
		 */
		template<typename TList>
		struct GetTotalParamDimension;

		template<unsigned Dim, typename T>
		struct GetTotalParamDimension<TypeList<ParamUnit<Dim>, T>> {
			enum {
				value = Dim + GetTotalParamDimension<T>::value
			};
		};

		template<>
		struct GetTotalParamDimension<NullType> {
			enum {
				value = 0
			};
		};

		/**
		 * Helper methods for FloatParamSet class.
		 */
		template<typename FloatType, typename TList>
		struct PrepareFloatParamList;

		template<typename FloatType>
		struct PrepareFloatParamList<FloatType, NullType> {
			using type = NullType;
		};

		template<typename FloatType, unsigned Dim, typename T>
		struct PrepareFloatParamList<FloatType, TypeList<ParamUnit<Dim>, T>> {
			using type = TypeList<Tuple<typename AddElements<Dim, FloatType, NullType>::type>, typename PrepareFloatParamList<FloatType, T>::type>;
		};


		/**
		 * FloatParamSet constructs a tuple of tuples, with one tuple for each parameter (with the
		 * specified dimension). All raw elements are of type FloatType.
		 * The input typelist TList should include only ParamUnit<Dim> classes.
		 *
		 * For example, if the TList = { ParamUnit<3>, ParamUnit<2> }, then the resulting class will
		 * have the following form:
		 *
		 *		Tuple(
		 *			Tuple( FloatType, FloatType, FloatType ),
		 *			Tuple( FloatType, FloatType)
		 *		)
		 */
		template<typename FloatType, typename TList>
		using FloatParamSet = Tuple<typename PrepareFloatParamList<FloatType, TList>::type>;


		/**
		 * Helper methods for DualParamSet class.
		 */
		template<typename FloatType, unsigned Idx, typename LowerLimitSatisfied, typename UpperLimitSatisfied>
		struct DualOrFloatType {
			using type = FloatType;
		};

		template<typename FloatType, unsigned Idx>
		struct DualOrFloatType<FloatType, Idx, Bool2Type<true>, Bool2Type<true>> {
			using type = Dual<FloatType, INDEXLIST_1(Idx)>;
		};

		template<typename FloatType, unsigned Idx, unsigned nRemaining, unsigned BlockStart, unsigned BlockEnd>
		struct PrepareDualSubParamList {
			using type = TypeList<
				typename DualOrFloatType<FloatType, Idx, Bool2Type<Idx >= BlockStart>, Bool2Type<Idx <= BlockEnd>>::type,
				typename PrepareDualSubParamList<FloatType, Idx + 1, nRemaining - 1, BlockStart, BlockEnd>::type
			>;
		};

		template<typename FloatType, unsigned Idx, unsigned BlockStart, unsigned BlockEnd>
		struct PrepareDualSubParamList<FloatType, Idx, 0, BlockStart, BlockEnd> {
			using type = NullType;
		};

		template<typename FloatType, typename TList, unsigned Idx, unsigned BlockStart, unsigned BlockEnd>
		struct PrepareDualParamList;

		template<typename FloatType, unsigned Idx, unsigned BlockStart, unsigned BlockEnd>
		struct PrepareDualParamList<FloatType, NullType, Idx, BlockStart, BlockEnd> {
			using type = NullType;
		};

		template<typename FloatType, unsigned Dim, typename T, unsigned Idx, unsigned BlockStart, unsigned BlockEnd>
		struct PrepareDualParamList<FloatType, TypeList<ParamUnit<Dim>, T>, Idx, BlockStart, BlockEnd> {
			using type = TypeList<Tuple<typename PrepareDualSubParamList<FloatType, Idx, Dim, BlockStart, BlockEnd>::type>, typename PrepareDualParamList<FloatType, T, Idx + Dim, BlockStart, BlockEnd>::type>;
		};


		/**
		 * DualParamSet constructs a tuple of tuples, with one tuple for each parameter (with the
		 * specified dimension). All raw elements are of type Dual<FloatType, INDEXLIST_1(Idx)>,
		 * where the Idx is the index of the current parameter component.
		 * The input typelist TList should include only ParamUnit<Dim> classes.
		 *
		 * For example, if the TList = { ParamUnit<3>, ParamUnit<2> }, then the resulting class will
		 * have the following form:
		 *
		 *		Tuple(
		 *			Tuple( Dual<FloatType, INDEXLIST_1(0)>, Dual<FloatType, INDEXLIST_1(1)>,
		 *				   Dual<FloatType, INDEXLIST_1(2)> ),
		 *			Tuple( Dual<FloatType, INDEXLIST_1(3)>, Dual<FloatType, INDEXLIST_1(4)>)
		 *		)
		 *
		 * The additional parameters control for how many parameters we actually compute partial 
		 * derivatives. We compute partial derivatives only for parameters with indices [BlockStart,
		 * BlockEnd]. The example before with BlockStart = 1 and BlockEnd = 3 would generate:
		 * 
		 *		Tuple(
		 *			Tuple( FloatType, Dual<FloatType, INDEXLIST_1(1)>,
		 *				   Dual<FloatType, INDEXLIST_1(2)> ),
		 *			Tuple( Dual<FloatType, INDEXLIST_1(3)>, FloatType)
		 *		) 
		 */
		template<typename FloatType, typename TList, unsigned BlockStart, unsigned BlockEnd>
		using DualParamSet = Tuple<typename PrepareDualParamList<FloatType, TList, 0, BlockStart, BlockEnd>::type>;

		
		/**
		 * FloatResidualSet constructs a tuple for holding all residuals of type FloatType.
		 */
		template<typename FloatType, unsigned ResidualDim>
		using FloatResidualSet = Tuple<typename AddElements<ResidualDim, FloatType, NullType>::type>;

		/**
		 * FloatJacobianSet constructs a double tuple for holding all jacobian elements of type FloatType.
		 * To query a partial derivative value of residual Ri with respect to parameter Pj, you need to
		 * query:
		 * deriv = floatJacobianSet[i][j]
		 */
		template<typename FloatType, unsigned ResidualDim, unsigned ParamDim>
		using FloatJacobianSet = Tuple<typename AddElements<ResidualDim, Tuple<typename AddElements<ParamDim, FloatType, NullType>::type>, NullType>::type>;


		/**
		 * DualResidualSet constructs a tuple for holding all residuals of type Dual, all with the same 
		 * index list covering all parameters inside the given parameter block [BlockStart, BlockEnd].
		 */
		template<typename FloatType, unsigned ResidualDim, unsigned BlockStart, unsigned BlockEnd>
		using DualResidualSet = Tuple<typename AddElements<ResidualDim, Dual<FloatType, typename IndexListFromRange<BlockStart, BlockEnd + 1>::type>, NullType>::type>;

	} // namespace param_parser

} // namespace solo