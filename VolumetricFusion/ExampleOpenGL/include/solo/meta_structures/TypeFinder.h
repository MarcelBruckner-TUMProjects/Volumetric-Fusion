#pragma once
#include <common_utils/meta_structures/BasicTypes.h>
#include <common_utils/meta_structures/Tuple.h>
#include <common_utils/meta_structures/TypeList.h>
#include <common_utils/meta_structures/SoA.h>

#include "Dual.h"

namespace solo {

	/**
	 * Returns the base floating-point type of the given type.
	 * Type can be valid basic aritmetic type, tuple, type list, dual or SoA.
	 */
	template <typename T>
	struct BaseTypeHelper;

	// Dual.
	template<typename FloatType, typename IdxList>
	struct BaseTypeHelper<Dual<FloatType, IdxList>> {
		using type = FloatType;
	};

	// Type list.
	template<typename T1, typename T2>
	struct BaseTypeHelper<TypeList<T1, T2>> {
		using type = typename std::enable_if<
			std::is_same<typename BaseTypeHelper<T1>::type, typename BaseTypeHelper<T2>::type>::value,
			typename BaseTypeHelper<T1>::type
		>::type;
	};

	template<typename T1>
	struct BaseTypeHelper<TypeList<T1, NullType>> {
		using type = typename BaseTypeHelper<T1>::type;
	};

	// Tuple.
	template<typename TList>
	struct BaseTypeHelper<Tuple<TList>> {
		using type = typename BaseTypeHelper<TList>::type;
	};

	// Arithmetic & SoA type.
	template <typename T, typename BoolDerivedFromSoA>
	struct BaseTypeArithmeticOrSoA;

	template <typename Derived>
	struct BaseTypeArithmeticOrSoA<Derived, Bool2Type<true>> {
		using type = typename BaseTypeHelper<
			typename ExtractSoATypeList<typename SoAType<Derived>::type>::type
		>::type;
	};

	template <typename T>
	struct BaseTypeArithmeticOrSoA<T, Bool2Type<false>> {
		using type = typename std::enable_if<std::is_arithmetic<T>::value, T>::type;
	};

	template <typename T>
	struct BaseTypeHelper {
		using type = typename BaseTypeArithmeticOrSoA<T, typename IsDerivedFromSoA<T>::type>::type;
	};

	// Dot notation.
	template <typename ...Types>
	struct BaseType {
		using type = typename BaseTypeHelper<typename TL<Types...>::type>::type;
	};


	/**
	 * Returns the appropriate result type, which can be assigned all/any of the given base
	 * input types (which can involve Dual objects). 
	 * That means that if the given types don't include any Dual objects, the result is the
	 * same as calling the BaseType<> method. But if some Dual objects are involved in the 
	 * given Types (maybe only as a sub-expression), then the resulting type will be a Dual
	 * object with index list that includes all indices of involved Dual objects.
	 */
	template<typename T>
	struct ResultTypeHelper;

	// Dual.
	template<typename FloatType, typename IdxList>
	struct ResultTypeHelper<Dual<FloatType, IdxList>> {
		using type = Dual<FloatType, IdxList>;
	};

	// Type list.
	template<typename D1, typename D2>
	struct ResultTypeMergeDuals;

	template<typename D1, typename D2>
	struct ResultTypeMergeDuals {
		using type = D1;
	};

	template<typename FloatType, typename IdxList1, typename D2>
	struct ResultTypeMergeDuals<Dual<FloatType, IdxList1>, D2> {
		using type = Dual<FloatType, IdxList1>;
	};

	template<typename D1, typename FloatType, typename IdxList2>
	struct ResultTypeMergeDuals<D1, Dual<FloatType, IdxList2>> {
		using type = Dual<FloatType, IdxList2>;
	};

	template<typename FloatType, typename IdxList1, typename IdxList2>
	struct ResultTypeMergeDuals<Dual<FloatType, IdxList1>, Dual<FloatType, IdxList2>> {
		using type = Dual<FloatType, typename JointIndexList<IdxList1, IdxList2>::type>;
	};

	template<typename T1, typename T2>
	struct ResultTypeHelper<TypeList<T1, T2>> {
		using Dual1 = typename ResultTypeHelper<T1>::type;
		using Dual2 = typename ResultTypeHelper<T2>::type;

		using type = typename ResultTypeMergeDuals<Dual1, Dual2>::type;
	};

	template<typename T1>
	struct ResultTypeHelper<TypeList<T1, NullType>> {
		using type = typename ResultTypeHelper<T1>::type;
	};

	// Tuple.
	template<typename TList>
	struct ResultTypeHelper<Tuple<TList>> {
		using type = typename ResultTypeHelper<TList>::type;
	};

	// For other types we use base type.
	template<typename T>
	struct ResultTypeHelper {
		using type = typename BaseTypeHelper<T>::type;
	};

	 // Dot notation.
	template <typename ...Types>
	struct ResultType {
		using type = typename ResultTypeHelper<typename TL<Types...>::type>::type;
	};

} // namespace solo