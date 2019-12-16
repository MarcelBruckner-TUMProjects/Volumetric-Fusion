#pragma once
#include "BasicTypes.h"

namespace common_utils {

	/**
	 * Base struct for representing a compile-time list of types.
	 * The idea for typelists is taken from the book Modern C++ Design by Andrei Alexandrescu (and the 
	 * correspondending Loki library).
	 */
	template<class H, class T>
	struct TypeList {
		using Head = H;
		using Tail = T;
	};


	/**
	 * Compile-time shortcut to generate type lists.
	 */
	#define TYPELIST_1(T1) TypeList<T1, NullType>
	#define TYPELIST_2(T1, T2) TypeList<T1, TYPELIST_1(T2)>
	#define TYPELIST_3(T1, T2, T3) TypeList<T1, TYPELIST_2(T2, T3)>
	#define TYPELIST_4(T1, T2, T3, T4) TypeList<T1, TYPELIST_3(T2, T3, T4)>
	#define TYPELIST_5(T1, T2, T3, T4, T5) TypeList<T1, TYPELIST_4(T2, T3, T4, T5)>
	#define TYPELIST_6(T1, T2, T3, T4, T5, T6) TypeList<T1, TYPELIST_5(T2, T3, T4, T5, T6)>
	#define TYPELIST_7(T1, T2, T3, T4, T5, T6, T7) TypeList<T1, TYPELIST_6(T2, T3, T4, T5, T6, T7)>
	#define TYPELIST_8(T1, T2, T3, T4, T5, T6, T7, T8) TypeList<T1, TYPELIST_7(T2, T3, T4, T5, T6, T7, T8)>
	#define TYPELIST_9(T1, T2, T3, T4, T5, T6, T7, T8, T9) TypeList<T1, TYPELIST_8(T2, T3, T4, T5, T6, T7, T8, T9)>
	#define TYPELIST_10(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10) TypeList<T1, TYPELIST_9(T2, T3, T4, T5, T6, T7, T8, T9, T10)>


	/**
	 * Helper method for creating a typelist of any length.
	 */
	template<typename ...Types>
	struct TL;

	template<>
	struct TL<> {
		using type = NullType;
	};

	template<typename T, typename ...OtherTypes>
	struct TL<T, OtherTypes...> {
		using type = TypeList<T, typename TL<OtherTypes...>::type>;
	};


	/**
	 * Computes the length of the type list, i.e. the number of types.
	 */
	template<class TList>
	struct TypeListLength;

	template<>
	struct TypeListLength<NullType> {
		enum{ value = 0 };
	};

	template<class H, class T>
	struct TypeListLength<TypeList<H, T>> {
		enum{ value = 1 + TypeListLength<T>::value };
	};


	/**
	 * Returns the type at the given position.
	 * If the type list's length is <= pos, NullType is returned.
	 */
	template<unsigned pos, class TList>
	struct TypeAt;

	template<class H, class T>
	struct TypeAt<0, TypeList<H, T>>  {
		using type = H;
	};

	template<unsigned pos>
	struct TypeAt<pos, NullType>  {
		using type = NullType;
	};

	template<unsigned pos, class H, class T>
	struct TypeAt<pos, TypeList<H, T>>  {
		using type = typename TypeAt<pos - 1, T>::type;
	};


	/**
	 * Adds N copies of class E to the typelist by adding the elements at the front.
	 */
	template<unsigned N, class E, class TList>
	struct AddElements {
		using type = typename AddElements<N - 1, E, TypeList<E, TList>>::type;
	};

	template<class E, class TList>
	struct AddElements<0, E, TList> {
		using type = TList;
	};


	/**
	 * Returns a type list, where each element is defined as WrappedClass<Type> for each
	 * Type element of the input type list TList.
	 */
	template<typename TList, template <class> class WrapperClass>
	struct WrappedTypeList;

	template<template <class> class WrapperClass>
	struct WrappedTypeList<NullType, WrapperClass> {
		using type = NullType;
	};

	template<typename T1, typename T2, template <class> class WrapperClass>
	struct WrappedTypeList<TypeList<T1, T2>, WrapperClass> {
		using type = TypeList<WrapperClass<T1>, typename WrappedTypeList<T2, WrapperClass>::type>;
	};


	/**
	 * Merges two type lists together into one common type list.
	 */
	template<typename TList1, typename TList2>
	struct MergedTypeList;

	template<typename TList>
	struct MergedTypeList<NullType, TList> {
		using type = TList;
	};

	template<typename TList1Head, typename TList1Tail, typename TList2>
	struct MergedTypeList<TypeList<TList1Head, TList1Tail>, TList2> {
		using type = TypeList<TList1Head, typename MergedTypeList<TList1Tail, TList2>::type>;
	};

} // namespace common_utils